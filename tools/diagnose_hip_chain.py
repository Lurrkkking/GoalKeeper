"""High-ball hip_pitch execution chain diagnostic — A0-A4 ablation."""
import sys, os, yaml, argparse, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.q1_goalkeeper_mujoco_sim2sim import (
    build_mujoco_model, build_joint_index_map,
    reset_robot_and_ball, get_robot_state,
    compute_ball_feature, build_single_obs, action_to_target_dof_pos,
    pd_control, load_onnx_policy, compute_shot_from_mode,
    get_ball_state, quat_rotate_inverse, get_contact_debug,
    advance_ball_obs_state, create_ball_obs_state,
)
import mujoco
import onnxruntime as ort

def euler_from_quat(q_wxyz):
    w, x, y, z = q_wxyz
    sinr = 2.0*(w*x + y*z); cosr = 1.0 - 2.0*(x*x + y*y)
    roll = np.arctan2(sinr, cosr)
    sinp = 2.0*(w*y - z*x); pitch = np.arcsin(np.clip(sinp, -1, 1))
    siny = 2.0*(w*z + x*y); cosy = 1.0 - 2.0*(y*y + z*z)
    yaw = np.arctan2(siny, cosy)
    return roll, pitch, yaw

def run_hip_diagnostic(cfg, shot_mode, dump_steps=60):
    model, data = build_mujoco_model(cfg)
    index_map = build_joint_index_map(model, cfg)

    # Extract joint ranges from MuJoCo model
    dof_lower, dof_upper = [], []
    for i in range(cfg["num_dofs"]):
        jid = model.dof_jntid[i]
        dof_lower.append(float(model.jnt_range[jid][0]))
        dof_upper.append(float(model.jnt_range[jid][1]))
    cfg["dof_pos_lower"] = dof_lower
    cfg["dof_pos_upper"] = dof_upper

    session = ort.InferenceSession(cfg["policy_path"])

    hp_L_idx = cfg["joint_names"].index("left_hip_pitch_joint")
    hp_R_idx = cfg["joint_names"].index("right_hip_pitch_joint")
    dof_lower = np.array(cfg.get("dof_pos_lower", [-np.inf]*22))
    dof_upper = np.array(cfg.get("dof_pos_upper", [np.inf]*22))
    tau_limit = np.array(cfg["tau_limit"])
    margin = cfg.get("target_clip_margin", 0.0)
    delta_limit = cfg.get("target_delta_limit", 0.0)

    obs_buf = np.zeros(cfg["num_obs"], dtype=np.float32)
    last_action = np.zeros(cfg["num_actions"], dtype=np.float32)
    ball_obs_state = create_ball_obs_state(cfg)
    n_policy = cfg["control_decimation"]

    shot_pos, shot_vel = compute_shot_from_mode(shot_mode)
    cfg["shot_init_pos"], cfg["shot_init_vel"] = shot_pos.tolist(), shot_vel.tolist()
    reset_robot_and_ball(model, data, index_map, cfg, shot_mode=shot_mode)

    target_dof_pos = np.array(cfg["default_dof_pos"], dtype=np.float64)

    rows = []

    for step in range(dump_steps):
        rs = get_robot_state(model, data, index_map, cfg)
        ball_pos_w, _ = get_ball_state(model, data, index_map, cfg)
        ball_feature = compute_ball_feature(ball_pos_w, rs["torso_pos"], rs["base_quat"], cfg, ball_obs_state)
        advance_ball_obs_state(ball_obs_state)

        single_obs = build_single_obs(rs, ball_feature, last_action, cfg)
        obs_buf = np.concatenate([obs_buf[cfg["num_single_obs"]:], single_obs])

        raw_action = session.run(None, {"obs": obs_buf.reshape(1,-1).astype(np.float32)})[0][0]
        raw_action = np.clip(raw_action, -cfg["clip_actions"], cfg["clip_actions"])

        # Target computation (mirrors runner)
        target_raw = np.array(cfg["default_dof_pos"]) + raw_action * np.array(cfg["action_scale_vec"])
        target_clip = np.clip(target_raw, dof_lower + margin, dof_upper - margin)

        # Delta limit
        if delta_limit > 0:
            delta = target_clip - target_dof_pos
            delta = np.clip(delta, -delta_limit, delta_limit)
            target_clip = target_dof_pos + delta

        target_dof_pos = target_clip

        # PD control
        tau_raw = np.array(cfg["kps"]) * (target_dof_pos - rs["dof_pos"]) - np.array(cfg["kds"]) * rs["dof_vel"]
        tau_clip = np.clip(tau_raw, -tau_limit, tau_limit)

        # Simulate
        for _ in range(n_policy):
            for i, act_id in enumerate(index_map["actuator_ids"]):
                data.ctrl[act_id] = tau_clip[i]
            mujoco.mj_step(model, data)

        roll, pitch, yaw = euler_from_quat(rs["base_quat"])
        foot_force, _ = get_contact_debug(model, data)

        rows.append({
            "step": step,
            "raw_a_hpL": float(raw_action[hp_L_idx]),
            "raw_a_hpR": float(raw_action[hp_R_idx]),
            "target_raw_hpL": float(target_raw[hp_L_idx]),
            "target_clip_hpL": float(target_clip[hp_L_idx]),
            "target_clip_hpR": float(target_clip[hp_R_idx]),
            "dof_pos_hpL": float(rs["dof_pos"][hp_L_idx]),
            "dof_pos_hpR": float(rs["dof_pos"][hp_R_idx]),
            "dof_vel_hpL": float(rs["dof_vel"][hp_L_idx]),
            "dof_vel_hpR": float(rs["dof_vel"][hp_R_idx]),
            "tau_raw_hpL": float(tau_raw[hp_L_idx]),
            "tau_clip_hpL": float(tau_clip[hp_L_idx]),
            "tau_raw_hpR": float(tau_raw[hp_R_idx]),
            "tau_clip_hpR": float(tau_clip[hp_R_idx]),
            "root_pitch": float(pitch),
            "proj_grav_x": float(rs["projected_gravity"][0]),
            "proj_grav_z": float(rs["projected_gravity"][2]),
            "ncon": int(data.ncon),
            "foot_force": float(foot_force),
            "hpL_over_upper": bool(rs["dof_pos"][hp_L_idx] > dof_upper[hp_L_idx]),
            "hpR_over_upper": bool(rs["dof_pos"][hp_R_idx] > dof_upper[hp_R_idx]),
        })
        last_action = raw_action.copy()

    return rows, dof_lower[hp_L_idx], dof_upper[hp_L_idx]

def print_table(rows, hpL_lo, hpL_hi):
    header = f"{'st':>3s} {'raw_hpL':>8s} {'tgt_raw':>8s} {'tgt_clip':>8s} {'dof_hpL':>8s} {'vel_hpL':>8s} {'tau_raw':>8s} {'tau_clip':>8s} {'pitch':>7s} {'ncon':>4s} {'over':>4s}"
    print(header)
    print(f"  hpL range: [{hpL_lo:.4f}, {hpL_hi:.4f}]")
    print("-" * len(header))
    for r in rows[::2]:
        over = "YES" if r["hpL_over_upper"] else ""
        print(f"{r['step']:3d} {r['raw_a_hpL']:8.3f} {r['target_raw_hpL']:8.4f} {r['target_clip_hpL']:8.4f} "
              f"{r['dof_pos_hpL']:8.4f} {r['dof_vel_hpL']:8.3f} {r['tau_raw_hpL']:8.1f} {r['tau_clip_hpL']:8.1f} "
              f"{r['root_pitch']:7.3f} {r['ncon']:4d} {over:>4s}")
    n = len(rows)
    pitches = [r["root_pitch"] for r in rows]
    overs = sum(1 for r in rows if r["hpL_over_upper"])
    print(f"\nSurvived {n} steps. max_pitch={max(pitches):.3f} min_pitch={min(pitches):.3f} "
          f"hpL_range=[{min(r['dof_pos_hpL'] for r in rows):.3f},{max(r['dof_pos_hpL'] for r in rows):.3f}] "
          f"n_over_upper={overs}/{n}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--shot-mode", type=int, default=2)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--label", default="")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rows, hpL_lo, hpL_hi = run_hip_diagnostic(cfg, args.shot_mode, args.steps)
    label = args.label or os.path.basename(args.config)
    print(f"\n===== {label} =====")
    print_table(rows, hpL_lo, hpL_hi)

    # Summary for grep
    pitches = [r["root_pitch"] for r in rows]
    overs = sum(1 for r in rows if r["hpL_over_upper"])
    print(f"SUMMARY|{label}|steps={len(rows)}|pitch_min={min(pitches):.3f}|pitch_max={max(pitches):.3f}|pitch_last={pitches[-1]:.3f}|over_upper={overs}|ncon_last={rows[-1]['ncon']}")

if __name__ == "__main__":
    main()
