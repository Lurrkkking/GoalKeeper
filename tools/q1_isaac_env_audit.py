#!/usr/bin/env python
"""Runtime audit for Q1 goalkeeper Isaac Gym env.

This script does not train. It creates short-lived envs, runs zero-action
rollouts, and prints reset/DR/friction/PD/obs/checkpoint diagnostics.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaacgym import gymapi, gymtorch  # noqa: E402
from isaacgym.torch_utils import quat_rotate_inverse  # noqa: E402
import torch  # noqa: E402

from legged_gym import LEGGED_GYM_ROOT_DIR  # noqa: E402
from legged_gym.envs.base.legged_robot import LeggedRobot, euler_from_quaternion  # noqa: E402
from legged_gym.envs.q1.q1_goalkeeper_config import Q1GoalkeeperCfg, Q1GoalkeeperCfgPPO  # noqa: E402
from rsl_rl.runners import HIMOnPolicyRunner  # noqa: E402
from legged_gym.utils.helpers import class_to_dict  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "outputs" / "q1_isaac_env_audit"


def sim_params():
    params = gymapi.SimParams()
    params.dt = 1.0 / 200.0
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    params.physx.solver_type = 1
    params.physx.num_position_iterations = 4
    params.physx.num_velocity_iterations = 0
    params.physx.num_threads = 10
    params.physx.use_gpu = True
    params.use_gpu_pipeline = True
    return params


def base_cfg(num_envs):
    cfg = copy.deepcopy(Q1GoalkeeperCfg())
    cfg.env.num_envs = num_envs
    cfg.init_state.default_joint_angles = dict(cfg.init_state.default_joint_angles)
    cfg.init_state.init_pos = list(cfg.init_state.init_pos)
    cfg.control.stiffness = dict(cfg.control.stiffness)
    cfg.control.damping = dict(cfg.control.damping)
    cfg.control.per_joint_action_scale = dict(cfg.control.per_joint_action_scale)
    cfg.terrain.mesh_type = "plane"
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_initial_joint_pos = False
    cfg.domain_rand.randomize_joint_injection = False
    cfg.domain_rand.randomize_actuation_offset = False
    cfg.domain_rand.randomize_com_displacement = False
    cfg.domain_rand.randomize_link_mass = False
    cfg.domain_rand.randomize_restitution = False
    cfg.domain_rand.randomize_reset_velocity = False
    return cfg


def apply_case(cfg, case_name):
    dr = cfg.domain_rand
    dr.randomize_friction = False
    dr.randomize_payload_mass = False
    dr.randomize_motor_strength = False
    dr.randomize_kp = False
    dr.randomize_kd = False
    dr.push_robots = False

    if case_name == "A0_original_no_dr":
        # Key runtime physics fields from q1_goalkeeper_config.py.bak.
        cfg.asset.file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/q1/q1_collision_viz.urdf"
        cfg.asset.armature = 0.001
        cfg.init_state.pos = [0.0, 0.0, 0.413]
        cfg.init_state.default_joint_angles["left_ankle_roll_joint"] = -0.01
        cfg.init_state.default_joint_angles["right_ankle_roll_joint"] = -0.03
        cfg.init_state.init_pos = [
            -0.087, 0, 0, 0.175, -0.087, -0.01,
            -0.087, 0, 0, 0.175, -0.087, -0.03,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]
        cfg.control.stiffness = {
            "hip_yaw": 60, "hip_roll": 60, "hip_pitch": 60, "knee": 80,
            "ankle": 25, "waist": 50, "shoulder": 20, "elbow": 20,
        }
        cfg.control.damping = {
            "hip_yaw": 2.0, "hip_roll": 2.0, "hip_pitch": 2.0, "knee": 3.0,
            "ankle": 0.9, "waist": 2.0, "shoulder": 0.8, "elbow": 0.8,
        }
        cfg.termination.gravity_threshold = 0.95
    elif case_name == "A1_current_dr_forced_off":
        pass
    elif case_name == "A2_friction_only":
        dr.randomize_friction = True
    elif case_name == "A3_payload_only":
        dr.randomize_payload_mass = True
    elif case_name == "A4_motor_strength_only":
        dr.randomize_motor_strength = True
    elif case_name == "A5_kp_kd_only":
        dr.randomize_kp = True
        dr.randomize_kd = True
    elif case_name == "A6_all_weak_dr_push_off":
        dr.randomize_friction = True
        dr.randomize_payload_mass = True
        dr.randomize_motor_strength = True
        dr.randomize_kp = True
        dr.randomize_kd = True
    elif case_name == "A7_all_weak_dr_push_on":
        dr.randomize_friction = True
        dr.randomize_payload_mass = True
        dr.randomize_motor_strength = True
        dr.randomize_kp = True
        dr.randomize_kd = True
        dr.push_robots = True
    else:
        raise ValueError(case_name)


def make_env(case_name, num_envs):
    cfg = base_cfg(num_envs)
    apply_case(cfg, case_name)
    env = LeggedRobot(cfg, sim_params(), gymapi.SIM_PHYSX, "cuda:0", True)
    return env


def tensor_stats(x):
    x = x.detach().float()
    return {
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
    }


def vec3(x):
    return [float(v) for v in x.detach().float().cpu().reshape(-1)[:3]]


def disable_ball(env):
    def no_randomize_balls():
        return None

    env._randomize_balls = no_randomize_balls
    env.ball_states[:, :3] = env.env_origins + torch.tensor([100.0, 0.0, 1.5], device=env.device)
    env.ball_states[:, 7:13] = 0.0
    env.ball_vel[:] = 0.0
    env.catchstep[:] = 0
    env.ball_last[:] = 0.0
    all_states = torch.cat((env.root_states.unsqueeze(1), env.ball_states.unsqueeze(1)), dim=1).view(-1, 13)
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(all_states))


def refresh_kinematics(env):
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.base_quat[:] = env.root_states[:, 3:7]
    env.roll, env.pitch, env.yaw = euler_from_quaternion(env.base_quat)
    env.base_lin_vel = quat_rotate_inverse(
        env.rigid_body_states[:, env.upper_body_index, 3:7],
        env.rigid_body_states[:, env.upper_body_index, 7:10],
    )
    env.base_ang_vel = quat_rotate_inverse(
        env.rigid_body_states[:, env.upper_body_index, 3:7],
        env.rigid_body_states[:, env.upper_body_index, 10:13],
    )
    env.torso_pos = env.rigid_body_states[:, env.torso_index, 0:3]
    env.projected_gravity[:] = quat_rotate_inverse(
        env.rigid_body_states[:, env.upper_body_index, 3:7], env.gravity_vec
    )


def friction_dump(env, max_shapes=64):
    robot = []
    for i, props in enumerate(env.gym.get_actor_rigid_shape_properties(env.envs[0], env.actor_handles[0])):
        robot.append({
            "shape": i,
            "friction": float(props.friction),
            "restitution": float(props.restitution),
        })
        if len(robot) >= max_shapes:
            break
    ball = []
    for i, props in enumerate(env.gym.get_actor_rigid_shape_properties(env.envs[0], env.ball_handles[0])):
        ball.append({
            "shape": i,
            "friction": float(props.friction),
            "restitution": float(props.restitution),
        })
    return {
        "ground_cfg": {
            "static_friction": float(env.cfg.terrain.static_friction),
            "dynamic_friction": float(env.cfg.terrain.dynamic_friction),
            "restitution": float(env.cfg.terrain.restitution),
        },
        "robot_shapes_env0": robot,
        "ball_shapes_env0": ball,
    }


def body_mass_dump(env, env_id=0):
    props = env.gym.get_actor_rigid_body_properties(env.envs[env_id], env.actor_handles[env_id])
    names = env.gym.get_actor_rigid_body_names(env.envs[env_id], env.actor_handles[env_id])
    masses = []
    for i, p in enumerate(props):
        masses.append({
            "index": i,
            "name": names[i] if i < len(names) else f"body_{i}",
            "original_mass": float(env.default_rigid_body_mass[i].item()),
            "actual_mass": float(p.mass),
        })
    return masses


def reset_dump(env):
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    disable_ball(env)
    refresh_kinematics(env)
    roll, pitch, yaw = euler_from_quaternion(env.root_states[:, 3:7])
    foot_z = env.rigid_body_states[0, env.contact_feet_indices, 2]
    foot_cf = torch.norm(env.contact_forces[0, env.contact_feet_indices, :], dim=-1)
    return {
        "root_pos": vec3(env.root_states[0, :3]),
        "root_quat": [float(v) for v in env.root_states[0, 3:7].detach().cpu()],
        "root_euler_rpy": [float(roll[0]), float(pitch[0]), float(yaw[0])],
        "root_lin_vel": vec3(env.root_states[0, 7:10]),
        "root_ang_vel": vec3(env.root_states[0, 10:13]),
        "base_ang_vel_local": vec3(env.base_ang_vel[0]),
        "projected_gravity": vec3(env.projected_gravity[0]),
        "dof_pos_minus_default_max_abs": float(torch.max(torch.abs(env.dof_pos[0] - env.default_dof_pos[0])).item()),
        "dof_vel_max_abs": float(torch.max(torch.abs(env.dof_vel[0])).item()),
        "foot_world_z": [float(v) for v in foot_z.detach().cpu()],
        "foot_contact_force_norm": [float(v) for v in foot_cf.detach().cpu()],
    }


def zero_action_case(case_name, num_envs, steps):
    env = make_env(case_name, num_envs)
    try:
        initial = reset_dump(env)
        zero = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        first_torque = env._compute_torques(zero)
        first_pd = {
            "target_equals_default_max_abs": float(torch.max(torch.abs(env.joint_pos_target - env.default_dof_poses)).item()),
            "pos_error_max_abs": float(torch.max(torch.abs(env.joint_pos_target - env.dof_pos)).item()),
            "torque_max_abs": float(torch.max(torch.abs(first_torque)).item()),
            "torque_saturation_ratio": float((torch.abs(first_torque) >= env.torque_limits.unsqueeze(0) * 0.999).float().mean().item()),
        }
        reset_step = None
        samples = []
        for step in range(steps):
            _, _, _, dones, _, _, _ = env.step(zero)
            if step in (0, steps // 2, steps - 1):
                foot_cf = torch.norm(env.contact_forces[:, env.contact_feet_indices, :], dim=-1)
                foot_vel_xy = torch.norm(env.rigid_body_states[:, env.contact_feet_indices, 7:9], dim=-1)
                contact_mask = foot_cf > 20.0
                slip = foot_vel_xy[contact_mask]
                samples.append({
                    "step": step,
                    "root_z_mean": float(env.root_states[:, 2].mean().item()),
                    "root_z_min": float(env.root_states[:, 2].min().item()),
                    "roll_abs_max": float(torch.max(torch.abs(env.roll)).item()),
                    "pitch_abs_max": float(torch.max(torch.abs(env.pitch)).item()),
                    "base_lin_vel_xy_mean": float(torch.norm(env.root_states[:, 7:9], dim=-1).mean().item()),
                    "base_ang_vel_mean": float(torch.norm(env.base_ang_vel, dim=-1).mean().item()),
                    "dof_vel_max": float(torch.max(torch.abs(env.dof_vel)).item()),
                    "foot_contact_count_mean": float((foot_cf > 20.0).float().sum(dim=1).mean().item()),
                    "foot_slip_xy_mean_when_contact": float(slip.mean().item()) if slip.numel() else 0.0,
                    "projected_gravity_xy_max": float(torch.norm(env.projected_gravity[:, :2], dim=-1).max().item()),
                })
            if reset_step is None and bool(dones.any().item()):
                reset_step = step
                break

        summary = {
            "case": case_name,
            "stable_5s": reset_step is None,
            "survival_steps": steps if reset_step is None else reset_step,
            "dt": float(env.dt),
            "push_interval_steps": int(env.cfg.domain_rand.push_interval),
            "push_would_happen_at_reset": bool(env.cfg.domain_rand.push_robots and env.common_step_counter == 0),
            "dr": {
                "friction": bool(env.cfg.domain_rand.randomize_friction),
                "payload": bool(env.cfg.domain_rand.randomize_payload_mass),
                "motor_strength": bool(env.cfg.domain_rand.randomize_motor_strength),
                "kp": bool(env.cfg.domain_rand.randomize_kp),
                "kd": bool(env.cfg.domain_rand.randomize_kd),
                "push": bool(env.cfg.domain_rand.push_robots),
            },
            "initial": initial,
            "first_zero_action_pd": first_pd,
            "friction_coeffs": tensor_stats(env.friction_coeffs),
            "payload": tensor_stats(env.payload),
            "motor_strength": tensor_stats(env.motor_strength),
            "kp_scale": tensor_stats(env.Kp_factors),
            "kd_scale": tensor_stats(env.Kd_factors),
            "friction": friction_dump(env),
            "base_mass_env0": body_mass_dump(env, 0)[0],
            "samples": samples,
        }
    finally:
        env.gym.destroy_sim(env.sim)
    return summary


def obs_dump(num_envs, checkpoint=None):
    env = make_env("A1_current_dr_forced_off", num_envs)
    try:
        env.reset_idx(torch.arange(env.num_envs, device=env.device))
        refresh_kinematics(env)
        env.compute_observations()
        one = env.obs_buf[0, -env.num_one_step_obs:].detach().clone()
        base_pos = env.torso_pos[0]
        base_quat = env.base_quat[0:1]
        ball_delta_world = env.ball_states[0, :3] - base_pos
        ball_local = quat_rotate_inverse(base_quat, ball_delta_world.view(1, 3))[0]
        ball_vel_local = quat_rotate_inverse(base_quat, env.ball_states[0:1, 7:10])[0]
        out = {
            "num_one_step_obs": int(env.num_one_step_obs),
            "layout": "ball3, base_ang_vel3, projected_gravity3, dof_pos_minus_default22, dof_vel22, last_action22",
            "raw_ball_world_pos": vec3(env.ball_states[0, :3]),
            "torso_world_pos": vec3(base_pos),
            "ball_delta_world": vec3(ball_delta_world),
            "ball_pos_local_expected": vec3(ball_local),
            "obs_ball_feature": vec3(one[:3]),
            "ball_vel_world": vec3(env.ball_states[0, 7:10]),
            "ball_vel_local_privileged": vec3(ball_vel_local),
            "obs_base_ang_vel": vec3(one[3:6]),
            "base_ang_vel_local_expected_scaled": vec3(env.base_ang_vel[0] * env.obs_scales.ang_vel),
            "obs_projected_gravity": vec3(one[6:9]),
            "projected_gravity_expected": vec3(env.projected_gravity[0]),
            "last_action_max_abs": float(torch.max(torch.abs(one[-env.num_actions:])).item()),
            "history_nonzero_counts_per_frame": [
                int(torch.count_nonzero(env.obs_buf[0, i * env.num_one_step_obs:(i + 1) * env.num_one_step_obs]).item())
                for i in range(env.actor_history_length)
            ],
        }
        if checkpoint:
            cfg_train = class_to_dict(Q1GoalkeeperCfgPPO())
            runner = HIMOnPolicyRunner(env, cfg_train, None, device=env.device)
            runner.load(str(checkpoint), load_optimizer=False)
            policy = runner.get_inference_policy(device=env.device)
            with torch.inference_mode():
                action = policy(env.obs_buf)
            abs_action = torch.abs(action[0])
            top_vals, top_idx = torch.topk(abs_action, k=min(5, abs_action.numel()))
            out["checkpoint"] = str(checkpoint)
            out["policy_current_learning_iteration"] = int(runner.current_learning_iteration)
            out["action_mean_abs"] = float(abs_action.mean().item())
            out["action_max_abs"] = float(abs_action.max().item())
            out["top5_action_joints"] = [
                {
                    "index": int(i.item()),
                    "joint": env.dof_names[int(i.item())],
                    "abs_action": float(v.item()),
                    "signed_action": float(action[0, int(i.item())].item()),
                }
                for v, i in zip(top_vals, top_idx)
            ]
        return out
    finally:
        env.gym.destroy_sim(env.sim)


def checkpoint_audit(paths):
    rows = []
    for path in paths:
        data = torch.load(path, map_location="cpu")
        rows.append({
            "path": str(path),
            "exists": True,
            "iter": int(data.get("iter", -1)),
            "has_optimizer": "optimizer_state_dict" in data,
            "has_model": "model_state_dict" in data,
            "num_model_tensors": len(data.get("model_state_dict", {})),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--skip-matrix", action="store_true")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    steps = int(round(args.seconds / (4 * (1.0 / 200.0))))
    cases = [
        "A0_original_no_dr",
        "A1_current_dr_forced_off",
        "A2_friction_only",
        "A3_payload_only",
        "A4_motor_strength_only",
        "A5_kp_kd_only",
        "A6_all_weak_dr_push_off",
        "A7_all_weak_dr_push_on",
    ]

    report = {
        "training_resume_command_observation": {
            "train_q1_rand_weak_sh": str(PROJECT_ROOT / "legged_gym/legged_gym/scripts/train_q1_rand_weak.sh"),
            "resume_required_flags": ["--resume", "--resumeid", "--checkpoint"],
        },
        "zero_action_matrix": [] if args.skip_matrix else [
            zero_action_case(case, args.num_envs, steps) for case in cases
        ],
        "obs_dump": obs_dump(args.num_envs, args.checkpoint or None),
    }

    ckpt_paths = []
    if args.checkpoint:
        ckpt_paths.append(Path(args.checkpoint))
    for candidate in [
        PROJECT_ROOT / "legged_gym/logs/q1/stand_urdf_5_014_rand_weak/model_0.pt",
        PROJECT_ROOT / "legged_gym/logs/q1/stand_urdf_5_014/model_9500.pt",
    ]:
        if candidate.exists():
            ckpt_paths.append(candidate)
    if ckpt_paths:
        report["checkpoint_audit"] = checkpoint_audit(ckpt_paths)

    out_path = REPORT_DIR / "report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nreport_json={out_path}")


if __name__ == "__main__":
    main()
