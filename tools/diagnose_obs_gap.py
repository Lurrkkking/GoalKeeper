"""Diagnose Gym-vs-MuJoCo obs gap at single-frame level.
Takes a Gym state, builds Gym-style and MuJoCo-style obs, compares.
"""
import os, sys, argparse, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'legged_gym'))
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import task_registry

# MuJoCo obs builder imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.q1_goalkeeper_mujoco_sim2sim import (
    compute_ball_feature, build_single_obs, quat_rotate_inverse,
)

def make_gym_env():
    args = type('Args', (), {})()
    args.task = 'q1_contact'; args.headless = True
    args.rl_device = 'cuda:0'; args.sim_device = 'cuda:0'; args.num_envs = 1
    args.resume = False; args.load_run = -1; args.checkpoint = -1
    args.resumeid = None; args.run_name = None; args.experiment_name = None
    args.seed = None; args.max_iterations = None; args.horovod = False
    args.compute_device_id = 0; args.sim_device_id = 0
    args.pipeline = 'gpu'; args.graphics_device_id = 0
    args.physics_engine = isaacgym.gymapi.SIM_PHYSX
    args.use_gpu = True; args.use_gpu_pipeline = True
    args.subscenes = 0; args.num_threads = 2; args.device = 'cuda'

    env_cfg, _ = task_registry.get_cfgs(name='q1_contact')
    dr = env_cfg.domain_rand
    for attr in ['randomize_friction','randomize_kp','randomize_kd','randomize_motor_strength',
                 'randomize_payload_mass','randomize_torque_noise','randomize_action_delay',
                 'randomize_action_filter','randomize_ball_obs_noise','randomize_ball_obs_dropout',
                 'randomize_ball_obs_delay','randomize_joint_damping','randomize_armature_scale',
                 'randomize_ground_slope','randomize_ground_height','push_robots',
                 'randomize_restitution','randomize_initial_joint_pos','randomize_reset_velocity',
                 'randomize_contact_friction']:
        if hasattr(dr, attr): setattr(dr, attr, False)
    dr.push_interval = 9999999; dr.ball_interval = 9999999
    if hasattr(dr, 'push_at_contact_loss'): dr.push_at_contact_loss = False
    if hasattr(dr, 'push_during_ball'): dr.push_during_ball = False
    env, _ = task_registry.make_env(name='q1_contact', args=args, env_cfg=env_cfg)
    env.Kp_factors[:] = 1.0; env.Kd_factors[:] = 1.0; env.motor_strength[:] = 1.0
    return env

def build_mujoco_style_obs(env, env_id, last_action, ball_obs_state):
    """Build single-frame MuJoCo-style obs from Gym state for env_id."""
    # Extract state from Gym (all as numpy)
    base_quat = env.root_states[env_id, 3:7].cpu().numpy()  # wxyz
    torso_pos = env.rigid_body_states[env_id, env.torso_index, 0:3].cpu().numpy()
    ball_pos_w = env.ball_states[env_id, :3].cpu().numpy()
    dof_pos = env.dof_pos[env_id].cpu().numpy()
    dof_vel = env.dof_vel[env_id].cpu().numpy()

    # MuJoCo-style ball feature
    cfg = {
        "num_single_obs": 75, "num_actions": 22, "num_ballobs": 3,
        "default_dof_pos": env.default_dof_pos[0].cpu().numpy().tolist(),
        "obs_scale_dof_pos": env.cfg.normalization.obs_scales.dof_pos,
        "obs_scale_dof_vel": env.cfg.normalization.obs_scales.dof_vel,
        "obs_scale_ang_vel": env.cfg.normalization.obs_scales.ang_vel,
    }
    ball_feature = compute_ball_feature(ball_pos_w, torso_pos, base_quat, cfg, ball_obs_state)

    # MuJoCo-style ang_vel_base
    ang_vel_world = env.rigid_body_states[env_id, env.upper_body_index, 10:13].cpu().numpy()
    ang_vel_base = quat_rotate_inverse(base_quat, ang_vel_world)

    # MuJoCo-style projected_gravity
    gravity_world = np.array([0.0, 0.0, -1.0])
    proj_grav = quat_rotate_inverse(base_quat, gravity_world)

    # MuJoCo-style single obs
    robot_state = {
        "ang_vel_base": ang_vel_base.astype(np.float32),
        "projected_gravity": proj_grav.astype(np.float32),
        "dof_pos": dof_pos.astype(np.float32),
        "dof_vel": dof_vel.astype(np.float32),
    }
    mj_single = build_single_obs(robot_state, ball_feature, last_action, cfg)

    # Gym single obs (first 75 dims from compute_observations)
    gym_single = env.obs_buf[env_id, -75:].cpu().numpy()  # latest frame is at the end

    return mj_single, gym_single, ball_feature, ang_vel_base, proj_grav, dof_pos

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    print("Creating Gym env...")
    env = make_gym_env()
    env.reset()
    device = env.device

    # Run a few steps with zero action to collect state
    zero_action = torch.zeros(1, 22, device=device)
    ball_obs_state = {"catchstep": 50, "startstep": 46, "ball_last": np.zeros(3, dtype=np.float32)}
    last_action = np.zeros(22, dtype=np.float32)
    done = torch.zeros(1, dtype=torch.bool, device=device)

    for step in range(args.steps):
        obs, _, _, _, _, _, _ = env.step(zero_action)
        if done[0]:
            env.reset()

        # Build MuJoCo-style obs from same Gym state
        mj_single, gym_single, bf, ang_vel, grav, dof = build_mujoco_style_obs(env, 0, last_action, ball_obs_state)

        # Sections of 75-dim obs
        def chunks(name, start, end):
            g = gym_single[start:end]
            m = mj_single[start:end]
            d = np.abs(g - m)
            return f"  {name:12s} [{start:2d}:{end:2d}] gym={_fmt(g)} mj={_fmt(m)} diff_max={d.max():.6f}"

        # Advance ball obs state
        ball_obs_state["catchstep"] -= 1
        bf_np = np.array(bf)
        if np.linalg.norm(bf_np) > 1e-6:
            ball_obs_state["ball_last"] = bf_np.astype(np.float32)

        # Print first few steps and last step
        if step < 3 or step == args.steps - 1:
            print(f"\n--- Step {step} ---")
            print(f"  ball_pos_w={env.ball_states[0,:3].cpu().numpy()} torso={env.rigid_body_states[0,env.torso_index,0:3].cpu().numpy()}")
            print(f"  bf_mj={bf} bf_gym={gym_single[:3]}")
            print(chunks("ball_feat", 0, 3))
            print(chunks("ang_vel", 3, 6))
            print(chunks("proj_grav", 6, 9))
            print(chunks("dof_pos", 9, 31))
            print(chunks("dof_vel", 31, 53))
            print(chunks("last_action", 53, 75))
            print(f"  FULL_75 max_diff={np.abs(gym_single - mj_single).max():.6f}")

        # Compare DOF names
        for d in range(22):
            if abs(dof[d] - env.dof_pos[0, d].item()) > 1e-6:
                print(f"  DOF mismatch idx={d}: gym={env.dof_pos[0,d]:.6f} mj={dof[d]:.6f}")

        if not done[0]:
            ball_obs_state["catchstep"] -= 1

def _fmt(a, precision=4):
    arr = np.asarray(a).reshape(-1)
    if len(arr) > 5:
        return f"[{arr[0]:.{precision}f}..{arr[-1]:.{precision}f}] (range={arr.min():.{precision}f}~{arr.max():.{precision}f})"
    return "[" + ",".join(f"{v:.{precision}f}" for v in arr) + "]"

if __name__ == "__main__":
    main()
