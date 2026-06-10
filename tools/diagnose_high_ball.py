"""High-ball sim2sim diagnostic: per-step dump for hip_pitch / contact analysis.

Usage:
  MUJOCO_GL=egl python tools/diagnose_high_ball.py \
      --config /tmp/test_v2_14000.yaml --shot-mode 2 --dump-steps 100
"""
import argparse, os, sys, json, time
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.q1_goalkeeper_mujoco_sim2sim import (
    build_mujoco_model, build_joint_index_map,
    reset_robot_and_ball, get_robot_state,
    compute_ball_feature, build_single_obs, action_to_target_dof_pos,
    pd_control, load_onnx_policy, compute_shot_from_mode,
    get_ball_state, quat_rotate_inverse, get_body_angular_velocity_world,
    get_contact_debug,
)
import mujoco
import onnxruntime as ort

def euler_from_quat(q_wxyz):
    """Return roll, pitch, yaw from wxyz quaternion."""
    w, x, y, z = q_wxyz
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def run_highball_diagnostic(cfg, shot_mode, dump_steps=100, open_loop=False, action_seq=None):
    """Run a single high-ball shot and dump per-step diagnostics."""
    model, data = build_mujoco_model(cfg)
    index_map = build_joint_index_map(model, cfg)
    onnx_path = cfg.get("policy_path", "")
    session = ort.InferenceSession(onnx_path) if onnx_path and not open_loop else None

    dt = cfg["control_decimation"] * cfg["simulation_dt"]
    policy_dt = dt  # 0.02s

    obs_buf = np.zeros(cfg["num_obs"], dtype=np.float32)
    last_action = np.zeros(cfg["num_actions"], dtype=np.float32)
    ball_obs_state = {"catchstep": 50, "startstep": 46, "control_startstep": 47,
                      "vanish_step": 0, "ball_last": np.zeros(3, dtype=np.float32)}

    # Ball launch
    shot_pos, shot_vel = compute_shot_from_mode(shot_mode)
    cfg["shot_init_pos"] = shot_pos.tolist()
    cfg["shot_init_vel"] = shot_vel.tolist()

    reset_robot_and_ball(model, data, index_map, cfg, shot_mode=shot_mode)

    records = []
    n_physics = cfg["control_decimation"]

    for step in range(dump_steps):
        # Get robot state
        rs = get_robot_state(model, data, index_map, cfg)

        # Ball state
        ball_pos_w, ball_vel_w = get_ball_state(model, data, index_map, cfg)
        ball_feature = compute_ball_feature(ball_pos_w, rs["torso_pos"], rs["base_quat"], cfg, ball_obs_state)
        # Advance ball obs state (catchstep decrement, matching main runner)
        if ball_obs_state is not None:
            ball_obs_state["catchstep"] -= 1

        # Build obs
        single_obs = build_single_obs(rs, ball_feature, last_action, cfg)
        obs_buf = np.concatenate([obs_buf[cfg["num_single_obs"]:], single_obs])

        # Policy inference
        if session is not None:
            obs_tensor = obs_buf.astype(np.float32).reshape(1, -1)
            raw_action = session.run(None, {"obs": obs_tensor})[0][0]
        elif action_seq is not None and step < len(action_seq):
            raw_action = action_seq[step]
        else:
            raw_action = np.zeros(cfg["num_actions"], dtype=np.float32)

        action = np.clip(raw_action, -cfg["clip_actions"], cfg["clip_actions"])
        target_dof_pos = action_to_target_dof_pos(action, cfg)

        # --- Pre-step recording ---
        roll, pitch, yaw = euler_from_quat(rs["base_quat"])
        foot_force, _ = get_contact_debug(model, data)
        ncon = data.ncon

        record = {
            "step": step,
            "time": step * policy_dt,
            "ball_z_world": float(ball_pos_w[2]),
            "ball_feature_z": float(ball_feature[2]),
            "ball_visible": bool(ball_feature[2] != 0 or ball_feature[0] != 0),
            "root_pitch": float(pitch),
            "root_roll": float(roll),
            "root_z": float(rs["base_pos"][2]),
            "torso_z": float(rs["torso_pos"][2]),
            "dof_pos_hip_pitch_L": float(rs["dof_pos"][0]),
            "dof_pos_hip_pitch_R": float(rs["dof_pos"][6]),
            "dof_pos_knee_L": float(rs["dof_pos"][3]),
            "dof_pos_knee_R": float(rs["dof_pos"][9]),
            "dof_pos_ankle_pitch_L": float(rs["dof_pos"][4]),
            "dof_pos_ankle_pitch_R": float(rs["dof_pos"][10]),
            "target_hip_pitch_L": float(target_dof_pos[0]),
            "target_hip_pitch_R": float(target_dof_pos[6]),
            "target_knee_L": float(target_dof_pos[3]),
            "target_knee_R": float(target_dof_pos[9]),
            "target_ankle_pitch_L": float(target_dof_pos[4]),
            "target_ankle_pitch_R": float(target_dof_pos[10]),
            "action_hip_pitch_L": float(action[0]),
            "action_hip_pitch_R": float(action[6]),
            "action_mean_abs": float(np.mean(np.abs(action))),
            "action_max_abs": float(np.max(np.abs(action))),
            "ncon": int(ncon),
            "foot_contact_total": float(foot_force),
        }
        records.append(record)

        # PD control + simulate physics for decimation steps
        for _ in range(n_physics):
            tau = pd_control(target_dof_pos, rs["dof_pos"], rs["dof_vel"], cfg)
            for i, act_id in enumerate(index_map["actuator_ids"]):
                data.ctrl[act_id] = tau[i]
            mujoco.mj_step(model, data)

        last_action = action.copy()

    return records


def print_summary(records):
    """Print key metrics from diagnostic records."""
    print(f"\n{'step':>5s} {'t(s)':>6s} {'ball_z':>7s} {'bf_z':>7s} {'pitch':>7s} "
          f"{'hpL':>8s} {'hpR':>8s} {'knL':>8s} {'apL':>8s} {'ncon':>4s} {'fc_tot':>7s} {'a_max':>7s}")
    for r in records[::2]:  # every other step
        print(f"{r['step']:5d} {r['time']:6.2f} {r['ball_z_world']:7.3f} {r['ball_feature_z']:7.3f} "
              f"{r['root_pitch']:7.3f} {r['dof_pos_hip_pitch_L']:8.4f} {r['dof_pos_hip_pitch_R']:8.4f} "
              f"{r['dof_pos_knee_L']:8.4f} {r['dof_pos_ankle_pitch_L']:8.4f} {r['ncon']:4d} "
              f"{r['foot_contact_total']:7.1f} {r['action_max_abs']:7.3f}")

    # Key metrics
    pitches = [r["root_pitch"] for r in records]
    hps_L = [r["dof_pos_hip_pitch_L"] for r in records]
    hps_R = [r["dof_pos_hip_pitch_R"] for r in records]
    ncons = [r["ncon"] for r in records]
    ball_z = [r["ball_z_world"] for r in records]
    n = len(records)

    print(f"\n--- Summary ({n} steps) ---")
    print(f"  pitch@last: {pitches[-1]:.3f}  max_pitch: {max(pitches):.3f}  min_pitch: {min(pitches):.3f}")
    print(f"  hp_L range: [{min(hps_L):.3f}, {max(hps_L):.3f}]  hp_R range: [{min(hps_R):.3f}, {max(hps_R):.3f}]")
    print(f"  ncon range: [{min(ncons)}, {max(ncons)}]")
    print(f"  ball_z range: [{min(ball_z):.3f}, {max(ball_z):.3f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--shot-mode", type=int, default=2)
    parser.add_argument("--dump-steps", type=int, default=100)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    records = run_highball_diagnostic(cfg, args.shot_mode, args.dump_steps)
    print_summary(records)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(records, f, indent=2)
        print(f"\nFull records saved to {args.output}")


if __name__ == "__main__":
    main()
