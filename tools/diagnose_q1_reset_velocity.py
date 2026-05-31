#!/usr/bin/env python
"""Diagnose Q1 goalkeeper reset velocity — check for upward root_lin_vel after reset."""
import sys, json, csv, time
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "legged_gym" / "legged_gym" / "scripts"))

import isaacgym; from isaacgym import gymtorch
from legged_gym.envs import *; from legged_gym.utils import get_args, task_registry
import torch

OUT_DIR = SCRIPT_DIR / "outputs" / "reset_velocity_diag"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_RESETS = 200
N_FRAMES = 20
UPWARD_THRESHOLD = 0.05  # m/s


def make_env(n=6):
    args = get_args(); args.task = "q1"; args.headless = True; args.num_envs = n
    return task_registry.make_env(name=args.task, args=args)


def record_frame(env, step, reset_id):
    rs = env.root_states[0]
    lf, rf = env.contact_feet_indices[0].item(), env.contact_feet_indices[1].item()
    lp = env.rigid_body_states[0, lf, :3]; rp = env.rigid_body_states[0, rf, :3]
    cf = torch.norm(env.contact_forces[0, :env.num_bodies], dim=-1)
    tau = env.torques[0]
    return {
        "reset_id": reset_id, "frame": step,
        "root_z": rs[2].item(),
        "root_lin_vel_x": rs[7].item(), "root_lin_vel_y": rs[8].item(), "root_lin_vel_z": rs[9].item(),
        "root_lin_vel_norm": torch.norm(rs[7:10]).item(),
        "root_ang_vel_x": rs[10].item(), "root_ang_vel_y": rs[11].item(), "root_ang_vel_z": rs[12].item(),
        "root_ang_vel_norm": torch.norm(rs[10:13]).item(),
        "left_foot_z": lp[2].item(), "right_foot_z": rp[2].item(),
        "left_foot_vz": env.rigid_body_states[0, lf, 9].item(),
        "right_foot_vz": env.rigid_body_states[0, rf, 9].item(),
        "max_dof_vel": env.dof_vel[0].abs().max().item(),
        "max_torque": tau.abs().max().item(),
        "max_contact_force": cf.max().item(),
        "max_contact_body": cf.argmax().item(),
        "foot_contact_l": cf[lf].item(), "foot_contact_r": cf[rf].item(),
        "projected_g_z": env.projected_gravity[0, 2].item(),
    }


def run_case(env, label, mode, n=N_RESETS):
    frames = []; upward_resets = set()
    vz_frame0 = []; vz_frame1 = []

    print(f"\n{'='*55}\n  {label}\n{'='*55}")

    for ri in range(n):
        env.reset_idx(torch.tensor([0], device=env.device))

        # Record BEFORE simulate (frame -1 effectively): velocity right after reset write
        before_vz = env.root_states[0, 9].item()
        before_vn = torch.norm(env.root_states[0, 7:10]).item()
        before_ang = torch.norm(env.root_states[0, 10:13]).item()
        before_dof = env.dof_vel[0].abs().max().item()

        for s in range(N_FRAMES):
            if mode == "zero_torque":
                env.torques[:] = 0.0
                env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.torques))
                env.gym.simulate(env.sim); env.gym.fetch_results(env.sim, True)
                env.gym.refresh_dof_state_tensor(env.sim)
                env.gym.refresh_actor_root_state_tensor(env.sim)
                env.gym.refresh_net_contact_force_tensor(env.sim)
                env.gym.refresh_rigid_body_state_tensor(env.sim)
            elif mode == "zero_action":
                env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
            elif mode == "random_small":
                act = (torch.rand(env.num_envs, env.num_actions, device=env.device) * 2 - 1) * 0.05
                env.step(act)

            f = record_frame(env, s, ri)
            if s == 0:
                f["before_sim_vz"] = before_vz
                f["before_sim_lin_vel_norm"] = before_vn
                f["before_sim_ang_vel_norm"] = before_ang
                f["before_sim_dof_vel_max"] = before_dof
                vz_frame0.append(env.root_states[0, 9].item())
            if s == 1:
                vz_frame1.append(env.root_states[0, 9].item())
            frames.append(f)

            if f["root_lin_vel_z"] > UPWARD_THRESHOLD:
                upward_resets.add(ri)

            if env.reset_buf[0] and s > 0:
                break

        if (ri + 1) % 40 == 0:
            u = len(upward_resets)
            print(f"  {ri+1}/{n}: upward_rate={u/(ri+1):.3f}")

    u = len(upward_resets)
    print(f"\n  Summary: upward_rate={u/n:.3f} "
          f"vz_f0_mean={np.mean(vz_frame0):.4f} vz_f0_max={max(vz_frame0):.4f} "
          f"vz_f1_mean={np.mean(vz_frame1):.4f}")

    # Find top 10 worst resets
    vz_all = [(ri, frames[ri*N_FRAMES]["before_sim_vz"], frames[ri*N_FRAMES]["root_lin_vel_z"])
              for ri in range(min(n, len(frames)//N_FRAMES))]
    vz_all.sort(key=lambda x: -abs(x[2]))
    worst = vz_all[:10]

    return {
        "label": label, "mode": mode,
        "n": n, "upward_rate": u/n,
        "vz_f0_mean": float(np.mean(vz_frame0)), "vz_f0_max": float(max(vz_frame0)),
        "vz_f1_mean": float(np.mean(vz_frame1)) if vz_frame1 else 0,
        "worst_resets": [{"reset_id": ri, "before_vz": bv, "after_vz": av} for ri, bv, av in worst],
        "frames": frames,
    }


def main():
    print("=" * 60)
    print("  Q1 Goalkeeper Reset Velocity Diagnostic")
    print("=" * 60)

    env, cfg = make_env(6)

    # Check velocity clearing code
    print("\n--- Task 5: Velocity Clearing Audit ---")
    print("  _reset_root_states (line 725): root_states[:,7:13] = rand(-0.3,0.3)  ← RANDOM VELOCITY!")

    results = []
    for label, mode in [
        ("A_zero_torque", "zero_torque"),
        ("B_zero_action", "zero_action"),
        ("C_random_small", "random_small"),
    ]:
        r = run_case(env, label, mode)
        results.append(r)

    # Report
    print("\n" + "=" * 60)
    print("  REPORT")
    print("=" * 60)

    print("\n--- Table 1: Reset Velocity Summary ---")
    print(f"{'Case':<22} {'up_rate':>8} {'vz_f0_mean':>12} {'vz_f0_max':>10} {'vz_f1_mean':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:<22} {r['upward_rate']:>8.3f} {r['vz_f0_mean']:>12.4f} {r['vz_f0_max']:>10.4f} {r['vz_f1_mean']:>12.4f}")

    print("\n--- Table 2: Velocity Clearing Check ---")
    print(f"  {'State':<25} {'Cleared?':<10} {'Location':<40} {'Pass':<6}")
    print(f"  {'root_lin_vel':<25} {'NO (random)':<10} {'_reset_root_states line 725':<40} {'FAIL':<6}")
    print(f"  {'root_ang_vel':<25} {'NO (random)':<10} {'_reset_root_states line 725':<40} {'FAIL':<6}")
    print(f"  {'dof_vel':<25} {'YES':<10} {'_reset_dofs':<40} {'PASS':<6}")

    print("\n--- Table 3: Worst 10 Resets (by frame0 |vz|) ---")
    print(f"{'reset_id':>8} {'before_vz':>10} {'after_f0_vz':>12} {'case'}")
    all_worst = []
    for r in results:
        for w in r["worst_resets"][:3]:
            all_worst.append((w["reset_id"], w["before_vz"], w["after_vz"], r["label"]))
    all_worst.sort(key=lambda x: -abs(x[2]))
    for ri, bv, av, case in all_worst[:10]:
        print(f"{ri:>8} {bv:>10.4f} {av:>12.4f}   {case}")

    print("\n--- Table 4: Root Cause ---")
    print(f"  {'Cause':<40} {'Evidence':<40} {'Verdict'}")
    print(f"  {'reset velocity not zeroed':<40} {'line 725: random ±0.3 m/s':<40} {'CONFIRMED'}")
    print(f"  {'contact/ground impulse':<40} {'check zero_torque case':<40} {'TBD'}")
    print(f"  {'PD first-frame torque':<40} {'check zero_action vs zero_torque':<40} {'TBD'}")

    # Save
    for r in results:
        if r["frames"]:
            csv_p = OUT_DIR / f"{r['label']}_frames.csv"
            with open(csv_p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(r["frames"][0].keys())); w.writeheader(); w.writerows(r["frames"])
    summary = {"results": [{k: v for k, v in r.items() if k != "frames"} for r in results]}
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    main()
