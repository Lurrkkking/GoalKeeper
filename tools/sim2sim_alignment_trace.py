#!/usr/bin/env python3
"""Runtime traces for Q1 goalkeeper IsaacGym vs MuJoCo alignment.

This tool does not train and does not modify policy/reward/physics files.
It writes compact npz traces plus a text comparison summary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import yaml


REPO = Path(__file__).resolve().parents[1]
LEGGED_GYM_DIR = REPO / "legged_gym"
SCRIPTS_DIR = REPO / "scripts"


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    for key in ("xml_path", "policy_path"):
        if key in cfg and not os.path.isabs(cfg[key]):
            cfg[key] = str((path.parent / cfg[key]).resolve())
    return cfg


def load_onnx(path: str):
    sess = ort.InferenceSession(path)
    return sess, sess.get_inputs()[0].name, sess.get_outputs()[0].name


def infer_onnx(policy, obs: np.ndarray) -> np.ndarray:
    sess, input_name, output_name = policy
    return sess.run([output_name], {input_name: obs.astype(np.float32).reshape(1, -1)})[0][0]


def hash_obs(obs: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(obs.astype(np.float32)).tobytes()).hexdigest()[:12]


def q_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)


def quat_inverse_wxyz(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_rotate_wxyz(q, v):
    qv = np.array([q[1], q[2], q[3]], dtype=np.float64)
    t = 2.0 * np.cross(qv, v)
    return v + q[0] * t + np.cross(qv, t)


def quat_rotate_inverse_wxyz(q, v):
    return quat_rotate_wxyz(quat_inverse_wxyz(q), v)


def split_history(obs: np.ndarray, single_dim: int = 75) -> np.ndarray:
    return obs.reshape(-1, single_dim)


def make_trace_dict(n_steps: int, n_dof: int = 22, obs_dim: int = 750):
    return {
        "t": np.zeros(n_steps, dtype=np.float32),
        "root_pos": np.zeros((n_steps, 3), dtype=np.float32),
        "root_quat_xyzw": np.zeros((n_steps, 4), dtype=np.float32),
        "root_lin_vel": np.zeros((n_steps, 3), dtype=np.float32),
        "root_ang_vel_world": np.zeros((n_steps, 3), dtype=np.float32),
        "base_ang_vel_local": np.zeros((n_steps, 3), dtype=np.float32),
        "projected_gravity": np.zeros((n_steps, 3), dtype=np.float32),
        "dof_pos": np.zeros((n_steps, n_dof), dtype=np.float32),
        "dof_vel": np.zeros((n_steps, n_dof), dtype=np.float32),
        "ball_world_pos": np.zeros((n_steps, 3), dtype=np.float32),
        "ball_world_vel": np.zeros((n_steps, 3), dtype=np.float32),
        "obs": np.zeros((n_steps, obs_dim), dtype=np.float32),
        "obs_ball": np.zeros((n_steps, 3), dtype=np.float32),
        "obs_base_ang_vel": np.zeros((n_steps, 3), dtype=np.float32),
        "obs_projected_gravity": np.zeros((n_steps, 3), dtype=np.float32),
        "obs_dof_pos": np.zeros((n_steps, n_dof), dtype=np.float32),
        "obs_dof_vel": np.zeros((n_steps, n_dof), dtype=np.float32),
        "obs_last_action": np.zeros((n_steps, n_dof), dtype=np.float32),
        "action": np.zeros((n_steps, n_dof), dtype=np.float32),
        "target_dof_pos": np.zeros((n_steps, n_dof), dtype=np.float32),
        "torque": np.zeros((n_steps, n_dof), dtype=np.float32),
        "obs_hash": np.empty(n_steps, dtype="U12"),
    }


def record_obs_segments(trace, k, obs):
    newest = split_history(obs)[-1]
    trace["obs"][k] = obs
    trace["obs_ball"][k] = newest[0:3]
    trace["obs_base_ang_vel"][k] = newest[3:6]
    trace["obs_projected_gravity"][k] = newest[6:9]
    trace["obs_dof_pos"][k] = newest[9:31]
    trace["obs_dof_vel"][k] = newest[31:53]
    trace["obs_last_action"][k] = newest[53:75]
    trace["obs_hash"][k] = hash_obs(obs)


def run_mujoco_trace(cfg: dict, out_path: Path, steps: int, use_f1: bool):
    import mujoco

    sys.path.insert(0, str(SCRIPTS_DIR))
    import q1_goalkeeper_mujoco_sim2sim as mjs2s

    if use_f1:
        cfg = dict(cfg)
        cfg["xml_path"] = str(SCRIPTS_DIR / "q1_22dof_goalkeeper_ball_F1_body_fric10.xml")

    model, data = mjs2s.build_mujoco_model(cfg)
    index_map = mjs2s.build_joint_index_map(model, cfg)
    policy = load_onnx(cfg["policy_path"])
    mjs2s.reset_robot_and_ball(model, data, index_map, cfg)

    history = mjs2s.create_history_buffer(cfg)
    last_action = np.zeros(cfg["num_actions"], dtype=np.float32)
    action_scale = np.asarray(cfg["action_scale_vec"], dtype=np.float32)
    default_pos = np.asarray(cfg["default_dof_pos"], dtype=np.float32)
    trace = make_trace_dict(steps, cfg["num_actions"], cfg["num_obs"])
    dt = float(cfg["simulation_dt"]) * int(cfg["control_decimation"])

    ball_obs_state = mjs2s.create_ball_obs_state(cfg)

    for k in range(steps):
        rs = mjs2s.get_robot_state(model, data, index_map, cfg)
        bp, bv = mjs2s.get_ball_state(model, data, index_map, cfg)
        bf = mjs2s.compute_ball_feature(bp, rs["torso_pos"], rs["base_quat"], cfg, ball_obs_state)
        single = mjs2s.build_single_obs(rs, bf, last_action, cfg)
        obs = mjs2s.update_history_and_get_obs(history, single, cfg)
        obs = np.clip(obs, -cfg["clip_observations"], cfg["clip_observations"])
        action = np.clip(infer_onnx(policy, obs), -cfg["clip_actions"], cfg["clip_actions"])
        target = default_pos + action * action_scale

        trace["t"][k] = k * dt
        trace["root_pos"][k] = data.qpos[0:3]
        trace["root_quat_xyzw"][k] = q_wxyz_to_xyzw(data.qpos[3:7])
        trace["root_lin_vel"][k] = data.qvel[0:3]
        trace["root_ang_vel_world"][k] = mjs2s.get_body_angular_velocity_world(model, data, index_map["imu_body_id"])
        trace["base_ang_vel_local"][k] = rs["ang_vel_base"]
        trace["projected_gravity"][k] = rs["projected_gravity"]
        trace["dof_pos"][k] = rs["dof_pos"]
        trace["dof_vel"][k] = rs["dof_vel"]
        trace["ball_world_pos"][k] = bp
        trace["ball_world_vel"][k] = bv
        trace["action"][k] = action
        trace["target_dof_pos"][k] = target
        record_obs_segments(trace, k, obs)

        torque = None
        for _ in range(int(cfg["control_decimation"])):
            rs2 = mjs2s.get_robot_state(model, data, index_map, cfg)
            torque = mjs2s.pd_control(target, rs2["dof_pos"], rs2["dof_vel"], cfg)
            for i, act_id in enumerate(index_map["actuator_ids"]):
                data.ctrl[act_id] = torque[i]
            mujoco.mj_step(model, data)
        trace["torque"][k] = torque
        last_action = action.copy()
        mjs2s.advance_ball_obs_state(ball_obs_state)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace.update({
        "sim_dt": np.array(float(cfg["simulation_dt"]), dtype=np.float32),
        "model_timestep": np.array(float(model.opt.timestep), dtype=np.float32),
        "decimation": np.array(int(cfg["control_decimation"]), dtype=np.int32),
        "policy_dt": np.array(float(model.opt.timestep) * int(cfg["control_decimation"]), dtype=np.float32),
        "xml_path": np.array(str(cfg["xml_path"])),
    })
    np.savez(out_path, **trace)
    return trace


def run_isaac_trace(cfg: dict, out_path: Path, steps: int):
    # Import order is mandatory for IsaacGym.
    import isaacgym  # noqa: F401
    from isaacgym import gymapi, gymtorch
    from isaacgym.torch_utils import quat_rotate_inverse
    import torch

    sys.path.insert(0, str(LEGGED_GYM_DIR))
    from legged_gym.envs import LeggedRobot  # noqa: F401
    from legged_gym.utils import task_registry

    args = SimpleNamespace(
        physics_engine=gymapi.SIM_PHYSX,
        device="cuda",
        use_gpu=True,
        use_gpu_pipeline=True,
        subscenes=0,
        num_threads=10,
        sim_device="cuda:0",
        rl_device="cuda:0",
        headless=True,
        num_envs=1,
        seed=1,
        max_iterations=None,
        resume=False,
        experiment_name=None,
        run_name=None,
        load_run=None,
        checkpoint=None,
        exptid="sim2sim_alignment",
        resumeid=None,
    )

    env_cfg, _ = task_registry.get_cfgs("q1")
    env_cfg.env.num_envs = 1
    env_cfg.env.play = False
    env_cfg.noise.add_noise = False
    dr = env_cfg.domain_rand
    dr.randomize_initial_joint_pos = False
    dr.randomize_friction = False
    dr.randomize_restitution = False
    dr.randomize_kp = False
    dr.randomize_kd = False
    dr.randomize_payload_mass = False
    dr.randomize_com_displacement = False
    dr.randomize_link_mass = False
    dr.randomize_motor_strength = False
    dr.randomize_joint_injection = False
    dr.randomize_actuation_offset = False
    dr.push_robots = False
    dr.max_ball_vel = 0.0
    dr.ball_interval_s = 999.0
    dr.randomize_reset_velocity = False

    env, _ = task_registry.make_env(name="q1", args=args, env_cfg=env_cfg)
    policy = load_onnx(cfg["policy_path"])
    device = env.device
    env.env_origins[:] = 0.0

    def set_fixed_state():
        env.root_states[0] = env.base_init_state
        env.root_states[0, :3] = torch.tensor(cfg.get("init_root_pos", [0, 0, 0.415]), device=device)
        env.root_states[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        env.root_states[0, 7:13] = 0.0
        env.dof_pos[0] = env.default_dof_pos[0]
        env.dof_vel[0] = 0.0
        env.ball_states[0] = env.base_init_state
        env.ball_states[0, :3] = torch.tensor(cfg.get("shot_init_pos", [4.0, 0, 0.5]), device=device)
        env.ball_states[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        env.ball_states[0, 7:10] = torch.tensor(cfg.get("shot_init_vel", [-6.14, 0, 3.43]), device=device)
        env.ball_states[0, 10:13] = 0.0
        env.ball_vel[0] = env.ball_states[0, 7]
        env.catchstep[0] = 50
        env.startstep = 47
        env.vanish_step[0] = 0
        env.ball_last[0] = 0.0
        env.obs_buf[0] = 0.0
        env.actions[0] = 0.0
        env.last_actions[0] = 0.0
        env.last_last_actions[0] = 0.0
        env.last_dof_vel[0] = 0.0
        env.last_torques[0] = 0.0
        env.Kp_factors[0] = 1.0
        env.Kd_factors[0] = 1.0
        env.motor_strength[0] = 1.0
        env.joint_injection[0] = 0.0
        env.actuation_offset[0] = 0.0

        actor_ids = torch.tensor([0, 1], dtype=torch.int32, device=device)
        all_states = torch.cat((env.root_states.unsqueeze(1), env.ball_states.unsqueeze(1)), dim=1).view(-1, 13)
        env.gym.set_actor_root_state_tensor_indexed(
            env.sim, gymtorch.unwrap_tensor(all_states), gymtorch.unwrap_tensor(actor_ids), len(actor_ids)
        )
        dof_actor_ids = torch.tensor([0], dtype=torch.int32, device=device)
        env.gym.set_dof_state_tensor_indexed(
            env.sim, gymtorch.unwrap_tensor(env.dof_state), gymtorch.unwrap_tensor(dof_actor_ids), 1
        )
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)

    def update_derived():
        env.base_quat[:] = env.root_states[:, 3:7]
        upper_q = env.rigid_body_states[:, env.upper_body_index, 3:7]
        env.base_lin_vel = quat_rotate_inverse(upper_q, env.rigid_body_states[:, env.upper_body_index, 7:10])
        env.base_ang_vel = quat_rotate_inverse(upper_q, env.rigid_body_states[:, env.upper_body_index, 10:13])
        env.torso_pos = env.rigid_body_states[:, env.torso_index, 0:3]
        env.projected_gravity[:] = quat_rotate_inverse(upper_q, env.gravity_vec)

    set_fixed_state()
    update_derived()
    env.compute_observations()
    obs = torch.clip(env.obs_buf, -env.cfg.normalization.clip_observations, env.cfg.normalization.clip_observations)

    n = env.num_actions
    trace = make_trace_dict(steps, n, env.num_obs)
    action_scale = env.action_scale_vec.detach().cpu().numpy()
    default_pos = env.default_dof_pos[0].detach().cpu().numpy()
    policy_dt = float(env.dt)

    for k in range(steps):
        update_derived()
        obs_np = obs[0].detach().cpu().numpy().astype(np.float32)
        action_np = np.clip(
            infer_onnx(policy, obs_np),
            -env.cfg.normalization.clip_actions,
            env.cfg.normalization.clip_actions,
        )
        target_np = default_pos + action_np * action_scale

        trace["t"][k] = k * policy_dt
        trace["root_pos"][k] = env.root_states[0, :3].detach().cpu().numpy()
        trace["root_quat_xyzw"][k] = env.root_states[0, 3:7].detach().cpu().numpy()
        trace["root_lin_vel"][k] = env.root_states[0, 7:10].detach().cpu().numpy()
        trace["root_ang_vel_world"][k] = env.root_states[0, 10:13].detach().cpu().numpy()
        trace["base_ang_vel_local"][k] = env.base_ang_vel[0].detach().cpu().numpy()
        trace["projected_gravity"][k] = env.projected_gravity[0].detach().cpu().numpy()
        trace["dof_pos"][k] = env.dof_pos[0].detach().cpu().numpy()
        trace["dof_vel"][k] = env.dof_vel[0].detach().cpu().numpy()
        trace["ball_world_pos"][k] = env.ball_states[0, :3].detach().cpu().numpy()
        trace["ball_world_vel"][k] = env.ball_states[0, 7:10].detach().cpu().numpy()
        trace["action"][k] = action_np
        trace["target_dof_pos"][k] = target_np
        record_obs_segments(trace, k, obs_np)

        action_t = torch.tensor(action_np, device=device, dtype=torch.float32).view(1, -1)
        obs, _, _, _, _, _, _ = env.step(action_t)
        trace["torque"][k] = env.torques[0].detach().cpu().numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace.update({
        "sim_dt": np.array(float(env.sim_params.dt), dtype=np.float32),
        "model_timestep": np.array(float(env.sim_params.dt), dtype=np.float32),
        "decimation": np.array(int(env.cfg.control.decimation), dtype=np.int32),
        "policy_dt": np.array(float(env.dt), dtype=np.float32),
        "xml_path": np.array("IsaacGym"),
    })
    np.savez(out_path, **trace)
    return trace


def compare_traces(isaac_path: Path, mujoco_path: Path, out_path: Path):
    a = np.load(isaac_path)
    b = np.load(mujoco_path)
    n = min(len(a["t"]), len(b["t"]))
    fields = [
        ("obs_max_diff", "obs"),
        ("ball_feature_diff", "obs_ball"),
        ("base_ang_vel_diff", "obs_base_ang_vel"),
        ("projected_gravity_diff", "obs_projected_gravity"),
        ("dof_pos_diff", "obs_dof_pos"),
        ("dof_vel_diff", "obs_dof_vel"),
        ("action_diff", "action"),
        ("target_diff", "target_dof_pos"),
        ("root_pose_diff", "root_pos"),
        ("ball_pos_diff", "ball_world_pos"),
        ("ball_vel_diff", "ball_world_vel"),
    ]

    def scalar(src, name, default=np.nan):
        return float(np.asarray(src[name])) if name in src.files else default

    def first_nonzero_ball(src):
        norms = np.linalg.norm(src["obs_ball"][:n], axis=1)
        idx = np.flatnonzero(norms > 1e-6)
        return int(idx[0]) if len(idx) else -1

    def fmt_vec(v):
        return "[" + ", ".join("{:.6g}".format(float(x)) for x in v) + "]"

    mujoco_xml = str(np.asarray(b["xml_path"])) if "xml_path" in b.files else "unknown"
    lines = []
    lines.append("timing")
    lines.append("  isaac_sim_dt={:.6g} mujoco_model_timestep={:.6g}".format(scalar(a, "sim_dt"), scalar(b, "model_timestep")))
    lines.append("  isaac_decimation={} mujoco_decimation={}".format(int(scalar(a, "decimation")), int(scalar(b, "decimation"))))
    lines.append("  isaac_policy_dt={:.6g} mujoco_policy_dt={:.6g}".format(scalar(a, "policy_dt"), scalar(b, "policy_dt")))
    lines.append("  mujoco_xml={}".format(mujoco_xml))
    lines.append("")

    lines.append("reset_newest_obs")
    lines.append("  isaac_ball={} mujoco_ball={}".format(fmt_vec(a["obs_ball"][0]), fmt_vec(b["obs_ball"][0])))
    lines.append("  isaac_projected_gravity={} mujoco_projected_gravity={}".format(fmt_vec(a["obs_projected_gravity"][0]), fmt_vec(b["obs_projected_gravity"][0])))
    lines.append("  isaac_base_ang_vel={} mujoco_base_ang_vel={}".format(fmt_vec(a["obs_base_ang_vel"][0]), fmt_vec(b["obs_base_ang_vel"][0])))
    lines.append("")

    lines.append("ball_visibility")
    lines.append("  first_nonzero_ball_step_isaac={}".format(first_nonzero_ball(a)))
    lines.append("  first_nonzero_ball_step_mujoco={}".format(first_nonzero_ball(b)))
    lines.append("")

    h_a = split_history(a["obs"][0])
    h_b = split_history(b["obs"][0])
    lines.append("reset_history_frames_oldest_to_newest")
    lines.append("  env frame ball3 base_ang_vel3 gravity3 dof_pos_norm dof_vel_norm last_action_norm")
    for env_name, hist in (("isaac", h_a), ("mujoco", h_b)):
        for i, frame in enumerate(hist):
            lines.append(
                "  {} {:02d} {} {} {} {:.6g} {:.6g} {:.6g}".format(
                    env_name,
                    i,
                    fmt_vec(frame[0:3]),
                    fmt_vec(frame[3:6]),
                    fmt_vec(frame[6:9]),
                    np.linalg.norm(frame[9:31]),
                    np.linalg.norm(frame[31:53]),
                    np.linalg.norm(frame[53:75]),
                )
            )
    lines.append("")

    lines.append("step_diffs")
    lines.append("step " + " ".join(name for name, _ in fields))
    for k in range(min(n, 20)):
        vals = []
        for _, key in fields:
            vals.append(float(np.max(np.abs(a[key][k] - b[key][k]))))
        lines.append("{:04d} ".format(k) + " ".join("{:.6g}".format(v) for v in vals))

    lines.append("")
    lines.append("ball_velocity_samples")
    for k in range(min(n, 10)):
        lines.append("  {:04d} isaac={} mujoco={}".format(k, fmt_vec(a["ball_world_vel"][k]), fmt_vec(b["ball_world_vel"][k])))

    out_path.write_text(chr(10).join(lines) + chr(10))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["isaac", "mujoco", "compare", "all"], default="all")
    ap.add_argument("--config", default=str(SCRIPTS_DIR / "q1_goalkeeper_mujoco_config.yaml"))
    ap.add_argument("--out-dir", default=str(REPO / "outputs" / "sim2sim_alignment"))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--use-f1", action="store_true", help="Use F1_fric10 XML for MuJoCo trace.")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    out_dir = Path(args.out_dir)

    if args.mode == "all":
        this_script = str(Path(__file__).resolve())
        config_arg = str(Path(args.config).resolve())
        common = ["--config", config_arg, "--out-dir", str(out_dir), "--steps", str(args.steps)]
        isaac_py = sys.executable if importlib.util.find_spec("isaacgym") else "/root/miniconda3/envs/rl/bin/python"
        mujoco_py = sys.executable if importlib.util.find_spec("mujoco") else "/root/miniconda3/bin/python"
        subprocess.check_call([isaac_py, this_script, "--mode", "isaac", *common])
        mujoco_cmd = [mujoco_py, this_script, "--mode", "mujoco", *common]
        if args.use_f1:
            mujoco_cmd.append("--use-f1")
        subprocess.check_call(mujoco_cmd)
        compare_traces(out_dir / "isaac_trace.npz", out_dir / "mujoco_trace.npz", out_dir / "comparison.txt")
        return

    if args.mode == "isaac":
        run_isaac_trace(cfg, out_dir / "isaac_trace.npz", args.steps)
    elif args.mode == "mujoco":
        run_mujoco_trace(cfg, out_dir / "mujoco_trace.npz", args.steps, args.use_f1)
    else:
        compare_traces(out_dir / "isaac_trace.npz", out_dir / "mujoco_trace.npz", out_dir / "comparison.txt")


if __name__ == "__main__":
    main()
