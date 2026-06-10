"""Gym-side high-ball diagnostic: dump per-step data for comparison with MuJoCo.

Usage:
  conda run -n rl python tools/diagnose_high_ball_gym.py \
      --checkpoint legged_gym/logs/q1/stand_urdf_5_014_contact_dr_v2/model_14000.pt \
      --shot-mode 2 --dump-steps 80
"""
import os, sys, argparse, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'legged_gym'))

import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.scripts.play import get_load_path
import onnxruntime as ort


def make_gym_env(headless=True):
    """Create a single-env IsaacGym simulation with DR disabled."""
    args = type('Args', (), {})()
    args.task = 'q1_contact'
    args.exptid = 'diagnose'
    args.headless = headless
    args.rl_device = 'cuda:0'
    args.sim_device = 'cuda:0'
    args.num_envs = 1
    args.resume = False
    args.load_run = -1
    args.checkpoint = -1
    args.resumeid = None
    args.run_name = None
    args.experiment_name = None
    args.seed = None
    args.max_iterations = None
    args.horovod = False
    args.compute_device_id = 0
    args.sim_device_id = 0
    args.pipeline = 'gpu'
    args.graphics_device_id = 0
    args.physics_engine = isaacgym.gymapi.SIM_PHYSX
    args.use_gpu = True
    args.use_gpu_pipeline = True
    args.subscenes = 0
    args.num_threads = 2
    args.device = 'cuda'

    # Get config FIRST, disable DR, THEN create env
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    dr = env_cfg.domain_rand
    dr.randomize_friction = False
    dr.randomize_kp = False; dr.randomize_kd = False
    dr.randomize_motor_strength = False
    dr.randomize_payload_mass = False
    dr.randomize_torque_noise = False
    dr.randomize_action_delay = False; dr.randomize_action_filter = False
    dr.randomize_ball_obs_noise = False; dr.randomize_ball_obs_dropout = False
    dr.randomize_ball_obs_delay = False
    dr.randomize_joint_damping = False; dr.randomize_armature_scale = False
    dr.randomize_ground_slope = False; dr.randomize_ground_height = False
    dr.push_robots = False; dr.randomize_restitution = False
    dr.randomize_initial_joint_pos = False; dr.randomize_reset_velocity = False
    dr.randomize_contact_friction = False
    if hasattr(dr, 'push_at_contact_loss'): dr.push_at_contact_loss = False
    if hasattr(dr, 'push_during_ball'): dr.push_during_ball = False
    if hasattr(dr, 'ball_vanish_jitter'): dr.ball_vanish_jitter = False
    dr.push_interval = 9999999
    dr.ball_interval = 9999999

    # Create env with modified config
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    env.Kp_factors[:] = 1.0
    env.Kd_factors[:] = 1.0
    env.motor_strength[:] = 1.0

    return env


def load_policy_onnx(onnx_path):
    """Load ONNX policy for inference."""
    session = ort.InferenceSession(onnx_path)
    return session


def euler_from_quat_torch(q_wxyz):
    """Return roll, pitch, yaw from wxyz quaternion (torch)."""
    w, x, y, z = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def run_gym_diagnostic(env, session, shot_mode, dump_steps=80):
    """Run a single gym episode and dump per-step data."""
    device = env.device

    # Reset all envs
    obs, privileged_obs = env.reset()

    # Force specific shot_mode by overriding command ranges
    # shot_mode maps to end_region in gym
    env.end_regions[:] = shot_mode

    # Override ball_interval and push_interval to avoid random pushes
    env.cfg.domain_rand.push_interval = 999999
    env.cfg.domain_rand.ball_interval = 999999

    records = []
    n_policy = env.cfg.control.decimation

    for step in range(dump_steps):
        # Get current state before step
        base_quat = env.root_states[0, 3:7]
        roll, pitch, yaw = euler_from_quat_torch(base_quat.unsqueeze(0))

        # Ball state
        ball_pos_w = env.ball_states[0, :3].cpu().numpy()
        ball_vel_w = env.ball_states[0, 7:10].cpu().numpy()

        # Robot state
        torso_pos = env.rigid_body_states[0, env.torso_index, 0:3].cpu().numpy()
        dof_pos = env.dof_pos[0].cpu().numpy()
        dof_vel = env.dof_vel[0].cpu().numpy()
        base_pos = env.root_states[0, :3].cpu().numpy()
        base_ang_vel = env.base_ang_vel[0].cpu().numpy()
        ncon = int((torch.norm(env.contact_forces[0, env.contact_feet_indices, :], dim=-1) > 1.0).sum().item())

        # Policy inference via ONNX
        obs_np = obs[0].cpu().numpy().astype(np.float32).reshape(1, -1)
        action = session.run(None, {"obs": obs_np})[0][0]

        # Target DOF positions
        action_scaled = action * env.action_scale_vec.cpu().numpy()
        target = env.default_dof_pos[0].cpu().numpy() + action_scaled

        # Ball feature from obs (first 3 dims)
        ball_feature = obs_np[0, :3]

        record = {
            "step": step,
            "time": step * env.dt,
            "ball_z_world": float(ball_pos_w[2]),
            "ball_feature_z": float(ball_feature[2]),
            "ball_visible": bool(np.linalg.norm(ball_feature) > 1e-6),
            "root_pitch": float(pitch.item()),
            "root_roll": float(roll.item()),
            "root_z": float(base_pos[2]),
            "torso_z": float(torso_pos[2]),
            "dof_pos_hip_pitch_L": float(dof_pos[0]),
            "dof_pos_hip_pitch_R": float(dof_pos[6]),
            "dof_pos_knee_L": float(dof_pos[3]),
            "dof_pos_knee_R": float(dof_pos[9]),
            "dof_pos_ankle_pitch_L": float(dof_pos[4]),
            "dof_pos_ankle_pitch_R": float(dof_pos[10]),
            "target_hip_pitch_L": float(target[0]),
            "target_hip_pitch_R": float(target[6]),
            "target_knee_L": float(target[3]),
            "target_ankle_pitch_L": float(target[4]),
            "action_hip_pitch_L": float(action[0]),
            "action_hip_pitch_R": float(action[6]),
            "action_mean_abs": float(np.mean(np.abs(action))),
            "action_max_abs": float(np.max(np.abs(action))),
            "ncon": ncon,
            "obs_first3": ball_feature.tolist(),
        }
        records.append(record)

        # Step gym
        obs, privileged_obs, rew, done, extras, term_ids, term_priv = env.step(
            torch.tensor(action, dtype=torch.float, device=device).unsqueeze(0))

        if done[0]:
            print(f"  Episode ended at step {step}")
            break

    return records


def print_summary(records):
    print(f"\n{'step':>5s} {'t(s)':>6s} {'ball_z':>7s} {'bf_z':>7s} {'pitch':>7s} "
          f"{'hpL':>8s} {'hpR':>8s} {'knL':>8s} {'apL':>8s} {'ncon':>4s} {'a_max':>7s}")
    for r in records[::2]:
        print(f"{r['step']:5d} {r['time']:6.2f} {r['ball_z_world']:7.3f} {r['ball_feature_z']:7.3f} "
              f"{r['root_pitch']:7.3f} {r['dof_pos_hip_pitch_L']:8.4f} {r['dof_pos_hip_pitch_R']:8.4f} "
              f"{r['dof_pos_knee_L']:8.4f} {r['dof_pos_ankle_pitch_L']:8.4f} {r['ncon']:4d} "
              f"{r['action_max_abs']:7.3f}")

    pitches = [r["root_pitch"] for r in records]
    hps_L = [r["dof_pos_hip_pitch_L"] for r in records]
    ncons = [r["ncon"] for r in records]
    a_maxs = [r["action_max_abs"] for r in records]
    n = len(records)
    print(f"\n--- Summary ({n} steps) ---")
    print(f"  pitch@last: {pitches[-1]:.3f}  max_pitch: {max(pitches):.3f} @step {np.argmax(pitches)}")
    print(f"  hp_L range: [{min(hps_L):.3f}, {max(hps_L):.3f}]")
    print(f"  ncon range: [{min(ncons)}, {max(ncons)}]")
    print(f"  a_max range: [{min(a_maxs):.3f}, {max(a_maxs):.3f}]  mean: {np.mean(a_maxs):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--shot-mode", type=int, default=2)
    parser.add_argument("--dump-steps", type=int, default=80)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("Creating gym env...")
    env = make_gym_env(headless=True)
    print(f"  dof_names[0]={env.dof_names[0]}, dof_names[6]={env.dof_names[6]}")
    print(f"  default_dof_pos hp_L={env.default_dof_pos[0,0]:.3f} hp_R={env.default_dof_pos[0,6]:.3f}")
    print(f"  Kp_factors[0]={env.Kp_factors[0,0]:.3f} motor_strength[0]={env.motor_strength[0,0]:.3f}")
    print(f"  p_gains[0]={env.p_gains[0]:.1f} d_gains[0]={env.d_gains[0]:.1f}")
    print(f"  action_scale_vec[0]={env.action_scale_vec[0]:.3f}")
    print(f"  torque_limits[0]={env.torque_limits[0]:.1f}")

    print("Loading ONNX policy...")
    session = load_policy_onnx(args.onnx)

    print(f"Running shot_mode={args.shot_mode}...")
    records = run_gym_diagnostic(env, session, args.shot_mode, args.dump_steps)
    print_summary(records)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(records, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
