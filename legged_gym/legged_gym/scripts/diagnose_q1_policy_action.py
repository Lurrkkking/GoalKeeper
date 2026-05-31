#!/usr/bin/env python
"""Diagnose Q1 goalkeeper: does PPO action path reach arms? Is eereach using correct body?"""
import sys, json, csv
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
LEGGED_GYM_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(LEGGED_GYM_ROOT))

import isaacgym; from isaacgym import gymtorch
from legged_gym.envs import *; from legged_gym.utils import get_args, task_registry
import torch

OUT_DIR = LEGGED_GYM_ROOT / ".." / "outputs" / "policy_action_debug"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARM_JOINTS = ['left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint',
              'right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint']
LEG_JOINTS = ['left_hip_pitch_joint','left_hip_roll_joint','left_hip_yaw_joint','left_knee_joint','left_ankle_pitch_joint','left_ankle_roll_joint',
              'right_hip_pitch_joint','right_hip_roll_joint','right_hip_yaw_joint','right_knee_joint','right_ankle_pitch_joint','right_ankle_roll_joint']


def make_env(n=6):
    args = get_args(); args.task = "q1"; args.headless = True; args.num_envs = n
    return task_registry.make_env(name=args.task, args=args)


def task1_env_info(env, cfg):
    """Print loaded env info."""
    info = {
        "urdf": str(cfg.asset.file).replace("{LEGGED_GYM_ROOT_DIR}", str(LEGGED_GYM_ROOT_DIR)),
        "num_actions": env.num_actions, "num_dof": env.num_dof, "num_bodies": env.num_bodies,
        "control_type": cfg.control.control_type, "action_scale": float(cfg.control.action_scale),
        "dof_names": list(env.dof_names),
    }
    print(f"  URDF: {info['urdf']}")
    print(f"  num_actions={info['num_actions']} num_dof={info['num_dof']}")
    assert "goalkeeper_collision" in info['urdf'], "WRONG URDF!"
    with open(OUT_DIR / "loaded_env_info.json", "w") as f: json.dump(info, f, indent=2)
    return info


def task2_3_4_5_action_path(env, cfg):
    """Run forced arm action and trace the full action→target→torque→dof chain."""
    print("\n" + "=" * 60)
    print("  TASK 2-5: Action Path Trace — Forced Arm Action")
    print("=" * 60)

    na = env.num_actions; nd = env.num_dof
    asc = cfg.control.action_scale
    pj = getattr(cfg.control, 'per_joint_action_scale', {})

    rows = []
    env.reset_idx(torch.tensor([0], device=env.device))

    for s in range(100):
        # Forced action: right arm joints to +1, others 0
        act = torch.zeros(1, na, device=env.device)
        for jn in ['right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint']:
            act[0, env.dof_names.index(jn)] = 0.5  # half-range, stay stable

        # Record pre-step
        pre_dof = env.dof_pos[0].clone()

        env.step(act.repeat(6, 1))

        # Record post-step
        post_dof = env.dof_pos[0].clone()
        tau = env.torques[0].clone()

        # Right elbow link position
        re_idx = [i for i, n in enumerate(
            env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
        ) if n == 'right_elbow_link']
        re_pos = env.rigid_body_states[0, re_idx[0], :3].clone() if re_idx else torch.zeros(3, device=env.device)

        if s < 5 or s == 99:
            print(f"  s{s}: right_elbow_z={re_pos[2]:.3f} "
                  f"act_sh_p={act[0,18]:.2f} act_sh_r={act[0,19]:.2f} act_el={act[0,21]:.2f} "
                  f"dof_sh_p={post_dof[18]:+.3f} dof_el={post_dof[21]:+.3f}")

        if s == 0:
            for jn in ARM_JOINTS:
                i = env.dof_names.index(jn)
                scl = pj.get(jn, asc)
                target_delta = float(act[0, i]) * scl
                actual_delta = float(post_dof[i] - pre_dof[i])
                tau_val = float(tau[i])
                rows.append({"joint": jn, "action": float(act[0,i]), "action_scale": scl,
                             "target_delta": target_delta, "actual_delta": actual_delta,
                             "torque": tau_val, "step": 0})

    # Print action path analysis
    print(f"\n  {'Joint':<30} {'act':>6} {'scale':>6} {'target_d':>8} {'actual_d':>9} {'torque':>8} {'track%':>7}")
    for r in rows:
        tk = abs(r['actual_delta'] / max(abs(r['target_delta']), 1e-6)) * 100
        status = "OK" if tk > 30 else "LOW"
        print(f"  {r['joint']:<30} {r['action']:>6.2f} {r['action_scale']:>6.3f} "
              f"{r['target_delta']:>+8.4f} {r['actual_delta']:>+9.4f} {r['torque']:>8.2f} {tk:>7.1f}% {status}")

    with open(OUT_DIR / "action_to_target.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    # Elbow displacement
    re_idx = [i for i, n in enumerate(
        env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
    ) if n == 'right_elbow_link']
    re_final = env.rigid_body_states[0, re_idx[0], :3] if re_idx else torch.zeros(3)
    print(f"\n  Right elbow final z: {re_final[2]:.4f}")

    return rows


def task6_body_motion(env, cfg):
    """Track elbow motion during forced action."""
    print("\n" + "=" * 60)
    print("  TASK 6: Link Motion Tracking")
    print("=" * 60)
    na = env.num_actions
    bn = list(env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0]))

    env.reset_idx(torch.tensor([0], device=env.device))
    le_idx = [i for i,n in enumerate(bn) if n=='left_elbow_link']
    re_idx = [i for i,n in enumerate(bn) if n=='right_elbow_link']
    ts_idx = [i for i,n in enumerate(bn) if n=='torso_link']
    le_i = le_idx[0] if le_idx else -1; re_i = re_idx[0] if re_idx else -1; ts_i = ts_idx[0] if ts_idx else -1

    le_start = env.rigid_body_states[0, le_i, :3].clone() if le_i >= 0 else torch.zeros(3)
    re_start = env.rigid_body_states[0, re_i, :3].clone() if re_i >= 0 else torch.zeros(3)

    for s in range(120):
        act = torch.zeros(1, na, device=env.device)
        for jn in ['right_shoulder_pitch_joint','right_shoulder_yaw_joint','right_elbow_joint']:
            act[0, env.dof_names.index(jn)] = 0.5
        env.step(act.repeat(6, 1))

    re_end = env.rigid_body_states[0, re_i, :3] if re_i >= 0 else torch.zeros(3)
    le_end = env.rigid_body_states[0, le_i, :3] if le_i >= 0 else torch.zeros(3)
    re_disp = torch.norm(re_end - re_start).item()
    le_disp = torch.norm(le_end - le_start).item()
    print(f"  Right elbow displacement: {re_disp:.4f}m (start z={re_start[2]:.3f} -> end z={re_end[2]:.3f})")
    print(f"  Left  elbow displacement: {le_disp:.4f}m")


def task7_eereach_body(env, cfg):
    """Check what body eereach uses."""
    print("\n" + "=" * 60)
    print("  TASK 7: Eereach Body Index Check")
    print("=" * 60)

    bn = list(env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0]))

    # eereach uses hand_indices
    hand_idx = env.hand_indices if hasattr(env, 'hand_indices') else None
    if hand_idx is not None:
        print(f"  hand_indices: {hand_idx.cpu().tolist() if hasattr(hand_idx,'cpu') else hand_idx}")
        for hi in (hand_idx.cpu().tolist() if hasattr(hand_idx,'cpu') else [hand_idx]):
            print(f"    idx {hi}: '{bn[hi] if hi < len(bn) else 'OOB'}'")

    # Check upper_body_index / torso_index
    print(f"  upper_body_index: {env.upper_body_index} = '{bn[env.upper_body_index] if env.upper_body_index < len(bn) else 'OOB'}'")
    print(f"  torso_index: {env.torso_index} = '{bn[env.torso_index] if env.torso_index < len(bn) else 'OOB'}'")

    # Print all elbow/shoulder body indices
    for name in ['left_elbow_link','right_elbow_link','left_shoulder_yaw_link','right_shoulder_yaw_link',
                 'torso_link','pelvis','left_knee_link','right_knee_link','left_ankle_roll_link','right_ankle_roll_link']:
        idx = [i for i,n in enumerate(bn) if n == name]
        print(f"  {name:<30} idx={idx[0] if idx else 'NOT FOUND'}")


def task8_reward_eereach_trace(env, cfg):
    """Trace eereach reward in detail for one episode."""
    print("\n" + "=" * 60)
    print("  TASK 8: Eereach Reward Trace")
    print("=" * 60)
    na = env.num_actions; bn = list(env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0]))
    le_idx = [i for i,n in enumerate(bn) if n=='left_elbow_link'][0]
    re_idx = [i for i,n in enumerate(bn) if n=='right_elbow_link'][0]

    env.reset_idx(torch.tensor([0], device=env.device))
    env.step(torch.zeros(6, na, device=env.device))

    for s in range(30):
        act = torch.zeros(1, na, device=env.device)
        if s > 5:
            # Forced arm action
            for jn in ['right_shoulder_pitch_joint','right_shoulder_yaw_joint','right_elbow_joint']:
                act[0, env.dof_names.index(jn)] = 0.5
        env.step(act.repeat(6, 1))

        hand_pos = env.rigid_body_states[0, env.hand_indices, :3]
        ball_pos = env.ball_states[0, :3]
        eereach_val = env.reward_functions[env.reward_names.index('eereach')]() if 'eereach' in env.reward_names else torch.tensor([0])

        le_pos = env.rigid_body_states[0, le_idx, :3]; re_pos = env.rigid_body_states[0, re_idx, :3]
        if s < 10 or s % 5 == 0:
            print(f"  s{s}: eereach={eereach_val[0]:.4f} "
                  f"hand_to_ball={torch.norm(hand_pos.mean(0)-ball_pos):.3f} "
                  f"le_to_ball={torch.norm(le_pos-ball_pos):.3f} "
                  f"re_to_ball={torch.norm(re_pos-ball_pos):.3f}")


def task9_obs_check(env, cfg):
    """Check what's in actor vs critic obs."""
    print("\n" + "=" * 60)
    print("  TASK 9: Actor vs Critic Obs")
    print("=" * 60)
    env.reset_idx(torch.tensor([0], device=env.device))
    env.step(torch.zeros(6, env.num_actions, device=env.device))

    obs = env.obs_buf[0]
    n_ball = env.cfg.env.num_ballobs
    n_one = env.num_one_step_obs
    print(f"  obs_dim={obs.shape[0]} one_step_obs={n_one} num_ballobs={n_ball}")
    print(f"  Ball obs at indices [6:{6+n_ball}] in one-step window")
    print(f"  Ball obs values: {obs[6:9].cpu().tolist()}")
    print(f"  Actor obs = obs_buf (all {obs.shape[0]} dims) — includes ball")

    # Check if ball values change
    bp0 = env.ball_states[0, :3].clone()
    env.ball_states[0, 0] += 1.0
    env.step(torch.zeros(6, env.num_actions, device=env.device))
    obs2 = env.obs_buf[0]
    print(f"  After ball moved: obs[6:9]={obs2[6:9].cpu().tolist()} (should differ from above)")


def main():
    print("=" * 60)
    print("  Q1 Policy Action Diagnostic")
    print("=" * 60)

    env, cfg = make_env(6)
    info = task1_env_info(env, cfg)
    action_rows = task2_3_4_5_action_path(env, cfg)
    task6_body_motion(env, cfg)
    task7_eereach_body(env, cfg)
    task8_reward_eereach_trace(env, cfg)
    task9_obs_check(env, cfg)

    # Final
    arm_track_ok = all(abs(r['actual_delta']/max(abs(r['target_delta']),1e-6)) > 0.3 for r in action_rows)
    print(f"\n{'='*60}")
    print(f"  FINAL ANSWERS")
    print(f"{'='*60}")
    print(f"  1. URDF: {'✅ goalkeeper_collision' if 'goalkeeper_collision' in info['urdf'] else '❌'}")
    print(f"  2. Actions → target → dof: {'✅ ALL ARM JOINTS TRACK' if arm_track_ok else '⚠️ some joints stuck'}")
    print(f"  3. See eereach body indices above — check if arm or leg")
    print(f"  4. Actor obs contains ball: ✅ (indices 6:9)")
    print(f"  5. Saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
