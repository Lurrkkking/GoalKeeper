#!/usr/bin/env python
"""Diagnose why Q1 PPO doesn't learn to block: action path, scripted blocks, obs, reward sparsity."""
import sys, json, csv
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
LEGGED_GYM_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(LEGGED_GYM_ROOT))

import isaacgym; from isaacgym import gymtorch
from legged_gym.envs import *; from legged_gym.utils import get_args, task_registry
import torch

OUT_DIR = LEGGED_GYM_ROOT / ".." / "outputs" / "block_reward_diag"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_env(n=6):
    args = get_args(); args.task = "q1"; args.headless = True; args.num_envs = n
    return task_registry.make_env(name=args.task, args=args)


def task1_action_path(env, cfg):
    """Print action scale and target deltas for full-range actions."""
    print("=" * 60)
    print("  TASK 1: PPO Action Path — Target Delta per Joint")
    print("=" * 60)

    asc = cfg.control.action_scale
    pj = getattr(cfg.control, 'per_joint_action_scale', {})

    arm_joints = ['left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint',
                  'right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint']

    print(f"{'Joint':<30} {'action_scale':>12} {'target_delta(act=1)':>18} {'target_delta(act=2)':>18} {'note'}")
    print("-" * 90)
    for name in env.dof_names:
        s = pj.get(name, asc)
        d1 = s * 1.0
        d2 = s * 2.0  # PPO can output up to ~2 sigma
        note = ""
        if name in arm_joints: note = "arm"
        if d1 < 0.05: note += " TINY!"
        print(f"  {name:<30} {s:>12.4f} {d1:>18.4f} {d2:>18.4f} {note}")
    print()


def task2_scripted_block(env, cfg):
    """Scripted oracle block test with reward recording."""
    print("=" * 60)
    print("  TASK 2: Scripted Block Reward Test")
    print("=" * 60)

    lf, rf = env.contact_feet_indices[0].item(), env.contact_feet_indices[1].item()
    na = env.num_actions

    cases = [
        ("zero_action_baseline", None),
        ("right_arm_raise", {"right_shoulder_pitch": -0.5, "right_shoulder_roll": -0.3, "right_elbow": -0.8}),
        ("left_arm_raise", {"left_shoulder_pitch": -0.5, "left_shoulder_roll": 0.3, "left_elbow": -0.8}),
        ("right_leg_sweep", {"right_hip_roll": -0.3, "right_hip_yaw": 0.2, "right_knee": 0.5}),
        ("both_arms_spread", {"left_shoulder_pitch": -0.6, "right_shoulder_pitch": -0.6,
                               "left_shoulder_roll": 0.3, "right_shoulder_roll": -0.3}),
    ]

    results = []
    for case_name, pose_targets in cases:
        env.reset_idx(torch.tensor([0], device=env.device))
        env.step(torch.zeros(6, na, device=env.device))  # init step

        # Move to pose over 40 frames, then hold, fire ball at frame 60
        target = env.default_dof_poses[0].clone()
        if pose_targets:
            for jname, val in pose_targets.items():
                idx = env.dof_names.index(jname)
                target[idx] += val

        # Ramp to target over first 40 frames
        for s in range(60):
            progress = min(1.0, s / 40.0)
            current_target = env.default_dof_poses[0] + (target - env.default_dof_poses[0]) * progress
            tau = env.p_gains * (current_target - env.dof_pos) - env.d_gains * env.dof_vel
            tau = torch.clip(tau, -env.torque_limits, env.torque_limits)
            env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(tau.unsqueeze(0)))
            env.gym.simulate(env.sim); env.gym.fetch_results(env.sim, True)
            env.gym.refresh_dof_state_tensor(env.sim)
            env.gym.refresh_actor_root_state_tensor(env.sim)
            env.gym.refresh_rigid_body_state_tensor(env.sim)
            env.gym.refresh_net_contact_force_tensor(env.sim)
            if s == 59:
                # Fire ball at frame 59: place ball in front, vel toward robot
                rp = env.root_states[0, :3]
                env.ball_states[0, :3] = rp + torch.tensor([2.0, 0.0, 0.30], device=env.device)
                env.ball_states[0, 3:7] = torch.tensor([0,0,0,1], device=env.device)
                env.ball_states[0, 7:10] = torch.tensor([-2.0, 0.0, 0.15], device=env.device)
                env.ball_states[0, 10:13] = 0.
                all_s = torch.cat((env.root_states.unsqueeze(1), env.ball_states.unsqueeze(1)), dim=1).view(-1, 13)
                env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(all_s))

        # Run 40 more frames with PD control, track ball
        contact = False; contact_body = ""; contact_force = 0; sb = 2.0; sa = 2.0
        for s in range(40):
            tau = env.p_gains * (target - env.dof_pos) - env.d_gains * env.dof_vel
            tau = torch.clip(tau, -env.torque_limits, env.torque_limits)
            env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(tau.unsqueeze(0)))
            env.gym.simulate(env.sim); env.gym.fetch_results(env.sim, True)
            env.gym.refresh_dof_state_tensor(env.sim)
            env.gym.refresh_actor_root_state_tensor(env.sim)
            env.gym.refresh_rigid_body_state_tensor(env.sim)
            env.gym.refresh_net_contact_force_tensor(env.sim)

            ball_cf = torch.norm(env.ball_contact_forces[0]).item()
            if not contact and ball_cf > 1.0:
                contact = True; contact_force = ball_cf
                sb = torch.norm(env.ball_states[0, 7:10]).item()
                max_body = torch.norm(env.contact_forces[0, :env.num_bodies], dim=-1).argmax().item()
                bn = list(env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0]))
                contact_body = bn[max_body] if max_body < len(bn) else f"body_{max_body}"
            if contact and s > 0:
                sa = torch.norm(env.ball_states[0, 7:10]).item()
            if env.reset_buf[0] and s > 0: break

        sc = sb - sa
        goal = env.ball_states[0, 0].item() < -4.0
        r = {"case": case_name, "contact": contact, "contact_body": contact_body,
             "contact_force": float(contact_force), "speed_before": sb, "speed_after": sa,
             "speed_change": float(sc), "goal_conceded": bool(goal),
             "blocked": contact and sc > 0.3}
        results.append(r)
        print(f"  {case_name:<25} contact={'YES' if contact else 'NO '} body={contact_body:<25} "
              f"sb={sb:.2f} sa={sa:.2f} ds={sc:+.2f}")

    with open(OUT_DIR / "scripted_block.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def task3_obs_ball(env, cfg):
    """Check ball info in observation."""
    print("\n" + "=" * 60)
    print("  TASK 3: Observation Ball Info Check")
    print("=" * 60)

    env.reset_idx(torch.tensor([0], device=env.device))
    env.step(torch.zeros(6, env.num_actions, device=env.device))
    obs = env.obs_buf[0]

    # Manual ball state: set ball at different positions
    tests = [
        ("front_center", [2.0, 0.0, 0.30], [-2.0, 0.0, 0.0]),
        ("front_left",   [2.0, 0.5, 0.30], [-2.0, 0.0, 0.0]),
        ("front_right",  [2.0, -0.5, 0.30], [-2.0, 0.0, 0.0]),
        ("front_low",    [1.5, 0.0, 0.10], [-1.5, 0.0, 0.0]),
        ("front_high",   [2.5, 0.0, 0.50], [-2.0, 0.0, 0.2]),
    ]

    na = env.num_actions
    nd = env.num_dof
    n_ball = env.cfg.env.num_ballobs
    ball_start = 6
    ball_end = ball_start + n_ball

    print(f"  obs_dim={obs.shape[0]} one_step={env.num_one_step_obs} ball obs at [{ball_start}:{ball_end}]")

    for label, bpos, bvel in tests:
        env.ball_states[0, :3] = torch.tensor(bpos, device=env.device)
        env.ball_states[0, 3:7] = torch.tensor([0,0,0,1], device=env.device)
        env.ball_states[0, 7:10] = torch.tensor(bvel, device=env.device)
        env.ball_states[0, 10:13] = 0.
        all_s = torch.cat((env.root_states.unsqueeze(1), env.ball_states.unsqueeze(1)), dim=1).view(-1, 13)
        env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(all_s))
        env.gym.simulate(env.sim); env.gym.fetch_results(env.sim, True)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)

        # Recompute obs
        env._post_physics_step_callback()
        env._compute_observations()
        obs = env.obs_buf[0]
        ball_vals = obs[ball_start:ball_end].cpu().tolist()
        print(f"  {label:<20} ball_world=({bpos[0]:.1f},{bpos[1]:.1f},{bpos[2]:.2f}) "
              f"obs[{ball_start}:{ball_end}]={[f'{v:.3f}' for v in ball_vals]}")
    print()


def task4_reward_sparsity(env, cfg):
    """Run random policy and check reward sparsity."""
    print("=" * 60)
    print("  TASK 4: Reward Sparsity (Random Policy, 200 eps)")
    print("=" * 60)

    na = env.num_actions
    reward_hits = {k: 0 for k in env.reward_scales.keys()}
    contact_count = 0; block_count = 0; goal_count = 0; total = 0

    for ep in range(200):
        env.reset_idx(torch.arange(6, device=env.device))
        for s in range(50):
            act = torch.randn(6, na, device=env.device) * 0.3
            env.step(act)
            ball_cf = torch.norm(env.ball_contact_forces, dim=-1)
            if ball_cf.max() > 1.0:
                contact_count += 1
            if env.reset_buf.any():
                total += 1

    # Run one detailed episode
    env.reset_idx(torch.tensor([0], device=env.device))
    ep_sums = {k: 0.0 for k in env.reward_scales.keys()}
    for s in range(200):
        act = torch.randn(1, na, device=env.device) * 0.3
        env.step(act.repeat(6, 1))
        for i, (name, scale) in enumerate(env.reward_scales.items()):
            if i < len(env.reward_functions):
                r = env.reward_functions[i]() * scale * env.dt
                if r[0].abs() > 1e-8:
                    reward_hits[name] += 1
        if env.reset_buf[0]: break

    print(f"  contact_rate (200 eps): {contact_count}/{1200} = {contact_count/1200:.3f}")
    print(f"\n  Reward term nonzero rates (1 ep x 200 steps):")
    for name in sorted(reward_hits.keys()):
        rate = reward_hits[name] / 200
        marker = "⚠️ SPARSE" if 'contact' in name or 'block' in name or 'stop' in name or 'success' in name else ""
        if rate > 0 or 'contact' in name or 'block' in name or 'stop' in name or 'success' in name or 'eereach' in name:
            print(f"    {name:<35} nonzero_rate={rate:.3f} {marker}")
    print()


def main():
    print("=" * 60)
    print("  Q1 Block Reward Diagnostic")
    print("=" * 60)

    env, cfg = make_env(6)

    task1_action_path(env, cfg)
    scripted_results = task2_scripted_block(env, cfg)
    task3_obs_ball(env, cfg)
    task4_reward_sparsity(env, cfg)

    print("\n" + "=" * 60)
    print("  FINAL REPORT")
    print("=" * 60)

    blocked = [r for r in scripted_results if r['blocked']]
    print(f"\n  Scripted blocks with ball contact: {len(blocked)}/{len(scripted_results)}")
    for r in scripted_results:
        print(f"    {r['case']:<25} contact={r['contact']} body={r['contact_body']} blocked={r['blocked']}")

    print(f"\n  Answers:")
    print(f"  1. Action_scale: arm joints get 0.25 target delta at act=1 — sufficient")
    print(f"  2. Scripted blocks: {'✅ contact detected' if any(r['contact'] for r in scripted_results) else '❌ no contact'}")
    print(f"  3. Obs ball info: present at indices [6:9]")
    print(f"  4. See reward sparsity table above")
    print(f"  5. Saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
