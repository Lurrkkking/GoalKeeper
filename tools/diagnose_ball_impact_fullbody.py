#!/usr/bin/env python
"""Full-body ball impact test for q1_22dof_goalkeeper_collision.urdf."""
import sys, json, csv, time
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "legged_gym" / "legged_gym" / "scripts"))

import isaacgym; from isaacgym import gymtorch
from legged_gym.envs import *; from legged_gym.utils import get_args, task_registry
import torch

OUT_DIR = SCRIPT_DIR / "outputs" / "ball_impact"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Key target bodies and their approximate local offsets from pelvis
TARGETS = [
    ("pelvis", [0.0, 0.0, 0.02]),
    ("torso_link", [0.0, 0.0, 0.15]),
    ("left_knee_link", [0.0, 0.08, -0.18]),
    ("left_ankle_roll_link", [0.05, 0.05, -0.37]),
    ("right_knee_link", [0.0, -0.08, -0.18]),
    ("right_ankle_roll_link", [0.05, -0.05, -0.37]),
    ("left_shoulder_yaw_link", [0.0, 0.15, 0.28]),
    ("left_elbow_link", [0.0, 0.15, 0.18]),
    ("right_shoulder_yaw_link", [0.0, -0.15, 0.28]),
    ("right_elbow_link", [0.0, -0.15, 0.18]),
    ("head_link", [0.0, 0.0, 0.40]),
]

BALL_SPEED = 1.5  # m/s
STEPS = 100

def make_env(n=6):
    args = get_args(); args.task = "q1"; args.headless = True; args.num_envs = n
    return task_registry.make_env(name=args.task, args=args)


def run_case(env, target_body, local_offset):
    """Spawn ball near target body with velocity toward it, record contact."""
    # Find target body index
    body_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
    try:
        body_idx = list(body_names).index(target_body)
    except ValueError:
        return {"target": target_body, "status": "BODY_NOT_FOUND", "error": f"body '{target_body}' not in actor"}

    # Reset to prepared pose
    env.reset_idx(torch.tensor([0], device=env.device))

    # Get target body world position
    target_pos = env.rigid_body_states[0, body_idx, :3].clone()

    # Place ball 0.3m in front of target (along x), with velocity toward target
    ball_pos = target_pos.clone()
    ball_pos[0] += 0.3  # 30cm in front
    ball_pos[2] = max(ball_pos[2], 0.12)  # above ground

    ball_vel = torch.zeros(3, device=env.device)
    ball_vel[0] = -BALL_SPEED  # toward robot (-x)

    # Manually set ball state
    env.ball_states[0, :3] = ball_pos
    env.ball_states[0, 3:7] = torch.tensor([0., 0., 0., 1.], device=env.device)
    env.ball_states[0, 7:10] = ball_vel
    env.ball_states[0, 10:13] = 0.
    all_states = torch.cat((env.root_states.unsqueeze(1), env.ball_states.unsqueeze(1)), dim=1).view(-1, 13)
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(all_states))

    # Run steps with zero action
    contact_detected = False
    contact_body = ""
    contact_force = 0.0
    ball_speed_before = BALL_SPEED
    ball_speed_after = BALL_SPEED
    contact_step = -1
    first_contact_body_idx = -1

    for s in range(STEPS):
        env.step(torch.zeros(6, 22, device=env.device))

        # Check contacts
        cf = torch.norm(env.contact_forces[0], dim=-1)  # (num_bodies,)
        robot_cf = cf[:env.num_bodies]
        max_force, max_body = torch.max(robot_cf, dim=0)

        # Ball contact force (stored separately in ball_contact_forces)
        ball_cf = torch.norm(env.ball_contact_forces[0], dim=-1).item()

        if not contact_detected and ball_cf > 1.0:
            contact_detected = True
            contact_step = s
            contact_force = ball_cf
            first_contact_body_idx = max_body.item()
            contact_body = body_names[first_contact_body_idx] if first_contact_body_idx < len(body_names) else f"body_{first_contact_body_idx}"
            ball_speed_before = torch.norm(env.ball_states[0, 7:10]).item()

        # Check for new contact body at next step
        if contact_detected and s == contact_step + 2:
            ball_speed_after = torch.norm(env.ball_states[0, 7:10]).item()

        if env.reset_buf[0] and s > 0:
            break

    if not contact_detected:
        # Record final ball speed anyway
        ball_speed_after = torch.norm(env.ball_states[0, 7:10]).item()

    speed_change = ball_speed_before - ball_speed_after
    # Deflection: compare initial vel direction (-x) with final vel direction
    deflection = 0.0
    if contact_detected:
        final_vel = env.ball_states[0, 7:10]
        init_dir = torch.tensor([-1., 0., 0.], device=env.device)
        final_dir = final_vel / (final_vel.norm() + 1e-8)
        deflection = torch.acos(torch.clamp(torch.dot(init_dir, final_dir), -1, 1)).item() * 180 / 3.14159

    return {
        "target": target_body,
        "target_body_idx": body_idx,
        "contact": contact_detected,
        "contact_body": contact_body,
        "contact_body_idx": first_contact_body_idx,
        "contact_force": contact_force,
        "contact_step": contact_step,
        "speed_before": ball_speed_before,
        "speed_after": ball_speed_after,
        "speed_change": speed_change,
        "deflection_deg": deflection,
        "ball_pos_init": ball_pos.cpu().tolist(),
        "target_pos": target_pos.cpu().tolist(),
    }


def main():
    print("=" * 70)
    print("  Q1 Goalkeeper Full-Body Ball Impact Test")
    print(f"  URDF: q1_22dof_goalkeeper_collision.urdf")
    print("=" * 70)

    env, cfg = make_env(6)

    # Print shape info
    actor_handle = env.actor_handles[0]
    props = env.gym.get_actor_rigid_shape_properties(env.envs[0], actor_handle)
    print(f"\nTotal rigid shapes: {len(props)} (was 7 with original URDF)")
    body_names = env.gym.get_actor_rigid_body_names(env.envs[0], actor_handle)
    print(f"Rigid bodies: {len(body_names)}")

    results = []
    for target_body, local_offset in TARGETS:
        r = run_case(env, target_body, local_offset)
        results.append(r)
        status = "✅ CONTACT" if r["contact"] else "❌ NO CONTACT"
        cb = r["contact_body"] if r["contact"] else "(none)"
        sc = r["speed_change"]
        dg = r["deflection_deg"]
        print(f"  {status} | {target_body:<28} → {cb:<28} | speed Δ={sc:+.2f} m/s | deflection={dg:.0f}° | force={r['contact_force']:.1f}N")

    # Summary
    passed = sum(1 for r in results if r["contact"])
    total = len(results)
    print(f"\n  Result: {passed}/{total} targets made contact with ball")

    failures = [r for r in results if not r["contact"]]
    if failures:
        print(f"  FAILURES:")
        for r in failures:
            print(f"    - {r['target']}")

    # Save
    with open(OUT_DIR / "impact_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {OUT_DIR / 'impact_results.json'}")

    return results


if __name__ == "__main__":
    main()
