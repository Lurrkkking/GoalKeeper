#!/usr/bin/env python
"""Eereach counterfactual: arm reach vs ball motion — uses env.step() for proper state."""
import sys, json, csv
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
LEGGED_GYM_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(LEGGED_GYM_ROOT))

import isaacgym; from isaacgym import gymtorch
from legged_gym.envs import *; from legged_gym.utils import get_args, task_registry
import torch

OUT_DIR = LEGGED_GYM_ROOT / ".." / "outputs" / "eereach_counterfactual"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_env(n=6):
    args = get_args(); args.task = "q1"; args.headless = True; args.num_envs = n
    return task_registry.make_env(name=args.task, args=args)


def set_ball(env, pos, vel):
    """Set ball world position & vel (env 0 only)."""
    env.ball_states[0, :3] = torch.tensor(pos, device=env.device)
    env.ball_states[0, 3:7] = torch.tensor([0., 0., 0., 1.], device=env.device)
    env.ball_states[0, 7:10] = torch.tensor(vel, device=env.device)
    env.ball_states[0, 10:13] = 0.
    all_s = torch.cat((env.root_states.unsqueeze(1), env.ball_states.unsqueeze(1)), dim=1).view(-1, 13)
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(all_s))


def get_ee_raw(env):
    if 'eereach' not in env.reward_names: return torch.tensor([0.], device=env.device)
    return env.reward_functions[env.reward_names.index('eereach')]()


def record(env, s, ee_raw):
    bn = list(env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0]))
    le = [i for i,n in enumerate(bn) if n=='left_elbow_link'][0]
    re = [i for i,n in enumerate(bn) if n=='right_elbow_link'][0]
    bp = env.ball_states[0, :3]; lp = env.rigid_body_states[0, le, :3]; rp = env.rigid_body_states[0, re, :3]
    return {"frame": s,
        "ball_x": float(bp[0]), "ball_y": float(bp[1]), "ball_z": float(bp[2]),
        "re_x": float(rp[0]), "re_y": float(rp[1]), "re_z": float(rp[2]),
        "le_x": float(lp[0]), "le_y": float(lp[1]), "le_z": float(lp[2]),
        "re_to_ball": float(torch.norm(rp - bp).item()),
        "le_to_ball": float(torch.norm(lp - bp).item()),
        "ee_raw": float(ee_raw[0].item()) if ee_raw.numel() > 0 else 0,
    }


def run_case(env, label, ball_pos, ball_vel, arm_fn, steps=100):
    env.reset_idx(torch.tensor([0], device=env.device))
    na = env.num_actions; asc = env.action_scale_vec
    default = env.default_dof_poses[0].clone()
    frames = []

    for s in range(steps):
        # compute action from desired target
        if arm_fn:
            desired = arm_fn(default, s, steps)
        else:
            desired = default.clone()
        action = (desired - default) / (asc + 1e-8)
        act_tensor = action.unsqueeze(0).repeat(6, 1)

        # Override ball position if needed
        if ball_pos and s == 0:
            set_ball(env, ball_pos, ball_vel)
        if ball_vel[0] != 0 and s > 0:
            # Let ball fly naturally with env.step
            pass

        env.step(act_tensor)

        ee_raw = get_ee_raw(env)
        frames.append(record(env, s, ee_raw))

        if env.reset_buf[0] and s > 0: break

    dists = [f['re_to_ball'] for f in frames]
    ees = [f['ee_raw'] for f in frames]
    d_s, d_e = np.mean(dists[:5]), np.mean(dists[-5:])
    e_s, e_e = np.mean(ees[:5]), np.mean(ees[-5:])
    print(f"  {label:<40} dist:{d_s:.2f}→{d_e:.2f} ee:{e_s:.4f}→{e_e:.4f}")

    p = OUT_DIR / f"{label}.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=frames[0].keys()); w.writeheader(); w.writerows(frames)
    return {"label": label, "d_s": d_s, "d_e": d_e, "e_s": e_s, "e_e": e_e,
            "dist_delta": d_e - d_s, "ee_delta": e_e - e_s}


def main():
    print("=" * 60)
    print("  Eereach Counterfactual (via env.step)")
    print("=" * 60)
    env, cfg = make_env(6)
    na = env.num_actions

    print(f"  hand_indices={env.hand_indices.cpu().tolist()}  scale={env.reward_scales.get('eereach',0):.4f}\n")

    # Ball positions relative to robot root
    rp = env.root_states[0, :3].clone()
    BALL_NEAR = [float(rp[0])+1.0, float(rp[1])-0.05, 0.35]
    BALL_FAR  = [float(rp[0])+3.0, float(rp[1])-0.05, 0.35]

    def reach_right(tgt, s, steps):
        t = tgt.clone(); p = min(1.0, s/max(1,steps/3))
        for jn,v in {'right_shoulder_pitch_joint':-0.5,'right_shoulder_yaw_joint':0.3,'right_elbow_joint':-0.7}.items():
            t[env.dof_names.index(jn)] += v * p
        return t

    def away_right(tgt, s, steps):
        t = tgt.clone(); p = min(1.0, s/max(1,steps/3))
        for jn,v in {'right_shoulder_pitch_joint':0.3,'right_shoulder_roll_joint':0.4,'right_elbow_joint':0.3}.items():
            t[env.dof_names.index(jn)] += v * p
        return t

    def reach_left(tgt, s, steps):
        t = tgt.clone(); p = min(1.0, s/max(1,steps/3))
        for jn,v in {'left_shoulder_pitch_joint':-0.5,'left_shoulder_yaw_joint':-0.3,'left_elbow_joint':-0.7}.items():
            t[env.dof_names.index(jn)] += v * p
        return t

    results = []
    for label, bpos, bvel, fn in [
        ("A_ball_near_hand_static", BALL_NEAR, [0,0,0], None),
        ("B_right_arm_REACH_to_ball", BALL_NEAR, [0,0,0], reach_right),
        ("C_right_arm_AWAY_from_ball", BALL_NEAR, [0,0,0], away_right),
        ("D_hand_static_ball_INCOMING", BALL_FAR, [-2.0,0,0], None),
        ("E_left_arm_REACH_to_ball", BALL_NEAR, [0,0,0], reach_left),
    ]:
        results.append(run_case(env, label, bpos, bvel, fn))

    print(f"\n{'='*60}")
    print(f"  REPORT")
    print(f"{'='*60}")
    print(f"\n{'Case':<40} {'dist Δ':>8} {'ee Δ':>10} {'Verdict'}")
    print("-" * 75)
    for r in results:
        dd, ed = r['dist_delta'], r['ee_delta']
        if abs(dd) < 0.02: v = "static baseline"
        elif dd < -0.03 and ed > 0.002: v = "✅ REACH REWARDED"
        elif dd < -0.03: v = "❌ EE BLIND TO REACH"
        elif dd > 0.03 and ed < -0.002: v = "✅ AWAY PENALIZED"
        else: v = "⚠️ unclear"
        print(f"  {r['label']:<40} {dd:>+8.3f} {ed:>+10.5f}   {v}")

    # Key answers
    b = results[1]; d = results[3]
    print(f"\n  Key:")
    print(f"  Reach to ball: dist {b['d_s']:.2f}→{b['d_e']:.2f}, ee {b['e_s']:.4f}→{b['e_e']:.4f}")
    print(f"  Ball incoming:  dist {d['d_s']:.2f}→{d['d_e']:.2f}, ee {d['e_s']:.4f}→{d['e_e']:.4f}")
    print(f"  Saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
