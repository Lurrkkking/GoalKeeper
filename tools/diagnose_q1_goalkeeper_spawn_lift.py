#!/usr/bin/env python
"""Diagnose Q1 goalkeeper feet-lift on reset — legged_gym version.

Runs N resets across 4 cases, records per-frame state, detects bad resets.

Usage:
    cd /root/autodl-tmp/Humanoid-Goalkeeper/legged_gym/legged_gym/scripts
    python ../../../tools/diagnose_q1_goalkeeper_spawn_lift.py
"""
import csv, json, os, sys
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
LEGGED_GYM_ROOT = SCRIPT_DIR.parent / "legged_gym"
sys.path.insert(0, str(LEGGED_GYM_ROOT / "legged_gym" / "scripts"))

import isaacgym
from isaacgym import gymtorch, gymapi
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

OUT_DIR = SCRIPT_DIR / "outputs" / "diagnose_spawn_lift"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_RESETS = 200
FRAMES = 20
BAD_Z_LIFT = 0.03
BAD_VEL = 0.2


def make_env(n=1):
    args = get_args()
    args.task = "q1"; args.headless = True; args.num_envs = n
    return task_registry.make_env(name=args.task, args=args)


def record(env, step, r):
    """Record frame — r is result dict for this reset."""
    rs = env.root_states[0]
    pg = env.projected_gravity[0]
    tau = env.torques[0]
    dof = env.dof_pos[0]; dvel = env.dof_vel[0]
    # Feet from rigid body states using contact_feet_indices
    lf = env.contact_feet_indices[0].item()
    rf = env.contact_feet_indices[1].item()
    lp = env.rigid_body_states[0, lf, :3]; lv = env.rigid_body_states[0, lf, 7:10]
    rp = env.rigid_body_states[0, rf, :3]; rv = env.rigid_body_states[0, rf, 7:10]
    cf = torch.norm(env.contact_forces[0], dim=-1)

    d = {"step": step}
    d["root_z"] = rs[2].item(); d["root_vz"] = rs[9].item()
    d["root_ang_vx"] = rs[10].item(); d["root_ang_vy"] = rs[11].item()
    d["proj_gz"] = pg[2].item()
    d["lf_z"] = lp[2].item(); d["rf_z"] = rp[2].item()
    d["lf_vz"] = lv[2].item(); d["rf_vz"] = rv[2].item()
    # Foot bottom clearance (approx: foot z minus half link height ~0.03m)
    d["lf_clear"] = lp[2].item() - 0.03; d["rf_clear"] = rp[2].item() - 0.03
    d["cf_lf"] = cf[lf].item(); d["cf_rf"] = cf[rf].item()
    d["cf_max"] = cf[:env.num_bodies].max().item()
    d["cf_max_body"] = cf[:env.num_bodies].argmax().item()
    d["dof_vel_max"] = dvel.abs().max().item()
    d["tau_max"] = tau.abs().max().item()
    d["tau_sat"] = (tau.abs() / env.torque_limits.clamp(1e-6)).max().item()
    # Ball
    bp = env.ball_states[0, :3]; bv = env.ball_states[0, 7:10]
    d["ball_z"] = bp[2].item()
    d["ball_dist_lf"] = torch.norm(bp - lp).item()
    d["ball_dist_rf"] = torch.norm(bp - rp).item()
    d["ball_overlap"] = 1 if min(d["ball_dist_lf"], d["ball_dist_rf"]) < 0.12 else 0

    # DOF key joints
    for ji, jn in [(0,"lhp"), (3,"lk"), (6,"rhp"), (9,"rk"), (4,"lap"), (10,"rap")]:
        d[f"d_{jn}"] = dof[ji].item()
        d[f"dv_{jn}"] = dvel[ji].item()
        d[f"t_{jn}"] = tau[ji].item()

    r.append(d)


def detect_bad(frames):
    if len(frames) < 10: return False, "short"
    lz0, rz0 = frames[0]["lf_z"], frames[0]["rf_z"]
    for i in range(1, min(10, len(frames))):
        dl = frames[i]["lf_z"] - lz0; dr = frames[i]["rf_z"] - rz0
        if dl > BAD_Z_LIFT and dr > BAD_Z_LIFT:
            return True, f"lift@{i} dl={dl:.4f} dr={dr:.4f}"
        if abs(frames[i]["lf_vz"]) > BAD_VEL and abs(frames[i]["rf_vz"]) > BAD_VEL:
            return True, f"vel@{i} lvz={frames[i]['lf_vz']:.3f} rvz={frames[i]['rf_vz']:.3f}"
    return False, "ok"


def run_case(env, label, ball, zero_torque, n=N_RESETS):
    bad = 0; bad_list = []; all_f0 = []
    print(f"\n{'='*55}\n  {label}\n  Ball={'ON' if ball else 'OFF'} Torque={'ZERO' if zero_torque else 'PD'}\n{'='*55}")

    for ri in range(n):
        env.reset_idx(torch.tensor([0], device=env.device))
        frames = []
        for s in range(FRAMES):
            if zero_torque:
                env.torques[:] = 0.0
                env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.torques))
                env.gym.simulate(env.sim)
                env.gym.fetch_results(env.sim, True)
                env.gym.refresh_dof_state_tensor(env.sim)
                env.gym.refresh_actor_root_state_tensor(env.sim)
                env.gym.refresh_net_contact_force_tensor(env.sim)
                env.gym.refresh_rigid_body_state_tensor(env.sim)
                env.gym.refresh_jacobian_tensors(env.sim)
            else:
                env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
            record(env, s, frames)
            if env.reset_buf[0] and s > 0: break

        is_bad, reason = detect_bad(frames)
        if is_bad: bad += 1; bad_list.append({"ri": ri, "reason": reason, "f0": frames[0]})
        all_f0.append(frames[0])
        if (ri+1) % 50 == 0: print(f"  {ri+1}/{n}, bad={bad}")

    rate = bad / n
    g0 = [f for i, f in enumerate(all_f0) if not any(b["ri"]==i for b in bad_list)]
    b0 = [b["f0"] for b in bad_list]
    print(f"  Bad: {bad}/{n} = {rate:.3f}")
    return {"label": label, "ball": ball, "zero_torque": zero_torque,
            "bad": bad, "rate": rate, "n": n,
            "good_f0": {k: np.mean([g[k] for g in g0]) for k in g0[0]} if g0 else {},
            "bad_f0": {k: np.mean([b[k] for b in b0]) for k in b0[0]} if b0 else {},
            "bad_list": bad_list}


def main():
    env, cfg = make_env(6)  # must be multiple of 6 for end_regions
    lf_idx = env.contact_feet_indices[0].item()
    rf_idx = env.contact_feet_indices[1].item()
    print(f"Q1 GK: root_z={cfg.init_state.pos[2]}, num_bodies={env.num_bodies}")
    print(f"Foot indices: L={lf_idx} R={rf_idx}")
    print(f"DOF names[0:6]: {env.dof_names[:6]}")
    print(f"Stand pos[0:6]: {env.standpos[0,:6].cpu().tolist()}")
    print(f"Torque lims[0:6]: {env.torque_limits[:6].cpu().tolist()}")

    # Print foot clearance right after reset
    env.reset_idx(torch.tensor([0], device=env.device))
    lp0 = env.rigid_body_states[0, lf_idx, 2].item()
    rp0 = env.rigid_body_states[0, rf_idx, 2].item()
    print(f"After 1 reset: L_foot_z={lp0:.5f} R_foot_z={rp0:.5f} clearance_est={lp0-0.03:.5f}/{rp0-0.03:.5f}")

    results = []
    for label, ball, zt in [
        ("A_zeroT_noBall", False, True),
        ("B_zeroAct_noBall", False, False),
        ("C_zeroT_ball", True, True),
        ("D_zeroAct_ball", True, False),
    ]:
        results.append(run_case(env, label, ball, zt))

    # --- Report ---
    print("\n" + "="*60)
    print("  REPORT")
    print("="*60)

    print("\n--- Table 1: Bad Reset Rate ---")
    print(f"{'Case':<25} {'Ball':>5} {'T':>6} {'Bad/N':>10} {'Rate':>8}")
    for r in results:
        print(f"{r['label']:<25} {'ON' if r['ball'] else 'OFF':>5} {'ZERO' if r['zero_torque'] else 'PD':>6} {r['bad']}/{r['n']:>5} {r['rate']:>8.3f}")

    print("\n--- Table 2: Good vs Bad Frame 0 ---")
    keys = ["lf_clear", "rf_clear", "root_z", "root_vz", "root_ang_vx", "dof_vel_max",
            "tau_max", "tau_sat", "cf_lf", "cf_rf", "cf_max", "ball_dist_lf", "ball_dist_rf", "ball_overlap"]
    for r in results:
        if r["bad"] == 0: print(f"\n  {r['label']}: all good"); continue
        if not r["good_f0"] or not r["bad_f0"]: continue
        print(f"\n  {r['label']}: good vs bad frame0")
        for k in keys:
            g = r["good_f0"].get(k, float("nan"))
            b = r["bad_f0"].get(k, float("nan"))
            flag = "⚠️" if (abs(g) > 1e-6 and abs(b-g) > 0.5*abs(g) + 0.001) else ""
            print(f"    {k:<20} good={g:>10.5f} bad={b:>10.5f} {flag}")

    # Save
    out = {"results": [{k: v for k, v in r.items() if k != "bad_list"} for r in results]}
    with open(OUT_DIR / "diagnose_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {OUT_DIR / 'diagnose_results.json'}")

    # Save bad sample CSVs
    for r in results:
        if r["bad_list"]:
            p = OUT_DIR / f"{r['label']}_bad_sample.csv"
            # re-run one bad case to get full frames
            env.reset_idx(torch.tensor([0], device=env.device))
            frames = []
            for s in range(FRAMES):
                if r["zero_torque"]:
                    env.torques[:] = 0.0
                    env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.torques))
                    env.gym.simulate(env.sim); env.gym.fetch_results(env.sim, True)
                    env.gym.refresh_dof_state_tensor(env.sim)
                    env.gym.refresh_actor_root_state_tensor(env.sim)
                    env.gym.refresh_net_contact_force_tensor(env.sim)
                    env.gym.refresh_rigid_body_state_tensor(env.sim)
                else:
                    env.step(torch.zeros(1, env.num_actions, device=env.device))
                record(env, s, frames)
                if env.reset_buf[0] and s > 0: break
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(frames[0].keys())); w.writeheader(); w.writerows(frames)
            print(f"  Bad sample: {p}")

    return results


if __name__ == "__main__":
    main()
