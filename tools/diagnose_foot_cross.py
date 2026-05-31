#!/usr/bin/env python
"""Diagnose Q1 goalkeeper right-foot-crossing-left-foot issue.

Runs A/B/C/D tests, records foot positions, detects foot crossing.
"""
import sys, json, csv, time
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "legged_gym" / "legged_gym" / "scripts"))

import isaacgym; from isaacgym import gymtorch, gymapi
from legged_gym.envs import *; from legged_gym.utils import get_args, task_registry
import torch

OUT_DIR = SCRIPT_DIR / "outputs" / "diagnose_foot_cross"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAMES = 200
CROSS_Y_THRESHOLD = 0.0    # right foot y passes left foot y → crossing
TOO_CLOSE_THRESHOLD = 0.06  # metres, foot xy separation too small


def make_env(n=6):
    args = get_args(); args.task = "q1"; args.headless = True; args.num_envs = n
    return task_registry.make_env(name=args.task, args=args)


def get_foot_positions(env):
    """Return left/right foot world positions."""
    lf = env.contact_feet_indices[0].item()
    rf = env.contact_feet_indices[1].item()
    lwp = env.rigid_body_states[0, lf, :3].clone()
    rwp = env.rigid_body_states[0, rf, :3].clone()
    # Simple world-frame: assume robot faces +x, y is lateral
    # right foot should have more NEGATIVE y than left (or vice versa)
    lwy = lwp[1].item()
    rwy = rwp[1].item()
    sep_y = rwy - lwy  # positive = feet apart normal, negative = crossed
    xy_dist = torch.norm(lwp[:2] - rwp[:2]).item()
    return lwp, rwp, lwy, rwy, sep_y, xy_dist


def record_frame(env, step, lwp, rwp, lwy, rwy, sep_y, xy_dist):
    lf = env.contact_feet_indices[0].item()
    rf = env.contact_feet_indices[1].item()
    cf = torch.norm(env.contact_forces[0], dim=-1)
    tau = env.torques[0]
    d = {"step": step}
    d["lf_wx"], d["lf_wy"], d["lf_wz"] = lwp[0].item(), lwp[1].item(), lwp[2].item()
    d["rf_wx"], d["rf_wy"], d["rf_wz"] = rwp[0].item(), rwp[1].item(), rwp[2].item()
    d["foot_sep_y"] = sep_y
    d["foot_xy_dist"] = xy_dist
    d["foot_cross"] = 1 if sep_y < CROSS_Y_THRESHOLD else 0  # negative sep = crossed
    d["foot_too_close"] = 1 if xy_dist < TOO_CLOSE_THRESHOLD else 0
    d["cf_lf"] = cf[lf].item(); d["cf_rf"] = cf[rf].item()
    d["cf_max_body"] = cf[:env.num_bodies].argmax().item()
    d["cf_max_val"] = cf[:env.num_bodies].max().item()
    d["tau_max"] = tau.abs().max().item()
    d["pg_z"] = env.projected_gravity[0, 2].item()
    for ji, jn in [(0,"lhp"),(2,"lhy"),(3,"lk"),(4,"lap"),(5,"lar"),
                    (6,"rhp"),(8,"rhy"),(9,"rk"),(10,"rap"),(11,"rar")]:
        d[f"t_{jn}"] = tau[ji].item()
        d[f"dp_{jn}"] = env.dof_pos[0, ji].item()
    return d


def run_case(env, label, mode, n_frames=FRAMES):
    """Run one diagnostic case. mode: 'zero_action', 'zero_torque', 'random', 'checkpoint'"""
    frames = []
    crossing_frames = []
    too_close_frames = []

    env.reset_idx(torch.tensor([0], device=env.device))

    # If checkpoint mode, load policy
    policy = None
    if mode == "random":
        pass  # use random actions below

    for s in range(n_frames):
        # Generate action
        if mode == "zero_action":
            action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        elif mode == "zero_torque":
            action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        elif mode == "random":
            action = torch.randn(env.num_envs, env.num_actions, device=env.device) * 0.3
        else:
            action = torch.zeros(env.num_envs, env.num_actions, device=env.device)

        # Step
        if mode == "zero_torque":
            env.torques[:env.num_envs] = 0.0
            env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.torques))
            env.gym.simulate(env.sim); env.gym.fetch_results(env.sim, True)
            env.gym.refresh_dof_state_tensor(env.sim)
            env.gym.refresh_actor_root_state_tensor(env.sim)
            env.gym.refresh_net_contact_force_tensor(env.sim)
            env.gym.refresh_rigid_body_state_tensor(env.sim)
            env.gym.refresh_jacobian_tensors(env.sim)
        else:
            env.step(action)

        lwp, rwp, lwy, rwy, sep_y, xy_dist = get_foot_positions(env)
        d = record_frame(env, s, lwp, rwp, lwy, rwy, sep_y, xy_dist)
        frames.append(d)

        if d["foot_cross"]:
            crossing_frames.append(s)
        if d["foot_too_close"]:
            too_close_frames.append(s)

        if env.reset_buf[0] and s > 0:
            break

    first_cross = crossing_frames[0] if crossing_frames else -1
    rate_cross = len(crossing_frames) / len(frames)
    rate_close = len(too_close_frames) / len(frames)
    min_sep = min(f["foot_sep_y"] for f in frames) if frames else 0

    print(f"  {label:<22}: frames={len(frames):>3} cross={len(crossing_frames):>3}({rate_cross:.2f}) "
          f"close={len(too_close_frames):>3}({rate_close:.2f}) first_cross@s={first_cross} min_sepY={min_sep:.3f}")

    return {
        "label": label, "mode": mode,
        "frames": frames, "n_frames": len(frames),
        "cross_count": len(crossing_frames), "cross_rate": rate_cross,
        "close_count": len(too_close_frames), "close_rate": rate_close,
        "first_cross_step": first_cross, "min_foot_sep_y": min_sep,
    }


def main():
    print("=" * 60)
    print("  Q1 Foot Crossing Diagnostic")
    print("=" * 60)

    env, cfg = make_env(6)
    lf, rf = env.contact_feet_indices[0].item(), env.contact_feet_indices[1].item()
    print(f"\nFoot body indices: L={lf} R={rf}")
    print(f"root_z={cfg.init_state.pos[2]} knee_thr={cfg.termination.knee_height_threshold}")

    # Print foot collision sizes from URDF
    import xml.etree.ElementTree as ET
    urdf_p = '/root/autodl-tmp/Humanoid-Goalkeeper/legged_gym/resources/robots/q1/q1_22dof_goalkeeper_collision.urdf'
    tree = ET.parse(urdf_p)
    for link_name in ['left_ankle_roll_link','right_ankle_roll_link','left_ankle_pitch_link','right_ankle_pitch_link',
                       'left_knee_link','right_knee_link','left_hip_yaw_link','right_hip_yaw_link']:
        link = tree.find(f".//link[@name='{link_name}']")
        if link is not None:
            for col in link.findall('collision'):
                geo = col.find('geometry'); orig = col.find('origin')
                xyz = orig.get('xyz','0 0 0') if orig is not None else '0 0 0'
                gtype = list(geo)[0].tag if geo is not None else '?'
                sz = ''
                if geo is not None:
                    c = list(geo)[0]
                    sz = c.get('size','') or c.get('radius','')+' '+c.get('length','')
                print(f"  {link_name:<30} {gtype:<10} origin=({xyz}) size=({sz})")

    # Reset once, check initial foot separation
    env.reset_idx(torch.tensor([0], device=env.device))
    lwp, rwp, lwy, rwy, sep_y, xy_dist = get_foot_positions(env)
    init_sep_y = sep_y
    init_dist = xy_dist
    print(f"\nReset foot separation: y_sep={init_sep_y:.4f} xy_dist={init_dist:.4f} (L_y={lwy:.4f} R_y={rwy:.4f})")

    # Run test cases
    print("\n--- Running Tests ---")
    results = []
    for label, mode in [
        ("A_zero_action", "zero_action"),
        ("B_zero_torque", "zero_torque"),
        ("C_random_policy", "random"),
    ]:
        r = run_case(env, label, mode)
        results.append(r)

    # Also try D with trained checkpoint if available
    ckpt_dir = Path('/root/autodl-tmp/Humanoid-Goalkeeper/legged_gym/logs/q1')
    latest = sorted(ckpt_dir.glob('*/model_*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
    if latest:
        print(f"\n--- D: Trained checkpoint: {latest[0]} ---")
        ckpt = torch.load(latest[0], map_location='cpu')
        # Just note what's available
        print(f"  Found checkpoint: iter info not extracted (not loading full policy)")
    else:
        print(f"\n--- D: No trained checkpoint found ---")

    # Report
    print("\n" + "=" * 60)
    print("  REPORT")
    print("=" * 60)
    print(f"\n{'Case':<22} {'Cross%':>8} {'Close%':>8} {'1stCross':>8} {'MinSepY':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['label']:<22} {r['cross_rate']:>8.3f} {r['close_rate']:>8.3f} "
              f"{r['first_cross_step']:>8} {r['min_foot_sep_y']:>10.4f}")

    # Save per-case CSVs
    for r in results:
        if r["frames"]:
            csv_p = OUT_DIR / f"{r['label']}_frames.csv"
            with open(csv_p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(r["frames"][0].keys())); w.writeheader(); w.writerows(r["frames"])
            print(f"  CSV: {csv_p}")

    # Save summary
    summary = {"init_foot_sep_y": init_sep_y, "init_foot_dist": init_dist, "cases": [
        {k: v for k, v in r.items() if k != "frames"} for r in results
    ]}
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary: {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
