#!/usr/bin/env python
"""Q1 left arm asymmetry diagnosis: sign search, torque saturation, joint symmetry."""
import sys, json, csv
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
LEGGED_GYM_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(LEGGED_GYM_ROOT))

import isaacgym; from isaacgym import gymtorch, gymapi
from legged_gym.envs import *; from legged_gym.utils import get_args, task_registry
import torch

OUT_DIR = LEGGED_GYM_ROOT / ".." / "outputs" / "q1_arm_debug"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEFT_ARM  = ['left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint']
RIGHT_ARM = ['right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint']


def make_env(n=6):
    args = get_args(); args.task = "q1"; args.headless = True; args.num_envs = n
    return task_registry.make_env(name=args.task, args=args)


def step_pd_fixed(env, target, saved_root, saved_quat):
    """PD step with root locked."""
    tau = env.p_gains * (target - env.dof_pos) - env.d_gains * env.dof_vel
    tau = torch.clip(tau, -env.torque_limits, env.torque_limits)
    env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(tau))
    env.gym.simulate(env.sim); env.gym.fetch_results(env.sim, True)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)
    env.root_states[:, :3] = saved_root.unsqueeze(0).repeat(env.num_envs, 1)
    env.root_states[:, 3:7] = saved_quat.unsqueeze(0).repeat(env.num_envs, 1)
    env.root_states[:, 7:13] = 0.
    all_s = torch.cat((env.root_states.unsqueeze(1), env.ball_states.unsqueeze(1)), dim=1).view(-1, 13)
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(all_s))


def get_setup(env):
    env.reset_idx(torch.tensor([0], device=env.device))
    return env.root_states[0, :3].clone(), env.root_states[0, 3:7].clone(), env.default_dof_poses[0].clone()


def task1_action_scale_check(env):
    """Verify all 8 arm joints have scale=0.5."""
    print("=" * 60)
    print("  TASK 1: Arm action_scale check")
    print("=" * 60)
    rows = []
    for jn in LEFT_ARM + RIGHT_ARM:
        i = env.dof_names.index(jn)
        s = env.action_scale_vec[i].item()
        rows.append({"dof_name": jn, "dof_idx": i, "action_scale": s})
        status = "✅" if abs(s - 0.5) < 0.01 else "⚠️ WRONG!"
        print(f"  {jn:<30} idx={i:>2} scale={s:.3f} {status}")
    with open(OUT_DIR / "action_scale_vec_check.json", "w") as f: json.dump(rows, f, indent=2)
    return rows


def task2_right_mag_scan(env):
    """Right arm: scan action magnitudes 0.1 to 1.0."""
    print("\n" + "=" * 60)
    print("  TASK 2: Right arm magnitude scan")
    print("=" * 60)
    bn = list(env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0]))
    re_i = [i for i,n in enumerate(bn) if n=='right_elbow_link'][0]
    results = []

    for mag in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        saved_root, saved_quat, default = get_setup(env)
        e0 = env.rigid_body_states[0, re_i, :3].clone()
        max_disp = 0; sat_rate = 0; limit_margin_min = 99

        for s in range(80):
            act = torch.zeros(1, env.num_actions, device=env.device)
            for jn in RIGHT_ARM: act[0, env.dof_names.index(jn)] = mag
            desired = default.clone()
            for jn in RIGHT_ARM: desired[env.dof_names.index(jn)] += mag * env.action_scale_vec[env.dof_names.index(jn)]
            step_pd_fixed(env, desired.unsqueeze(0).repeat(6, 1), saved_root, saved_quat)
            ep = env.rigid_body_states[0, re_i, :3]; disp = torch.norm(ep - e0).item()
            if disp > max_disp: max_disp = disp
            sat = (env.torques[0].abs() / env.torque_limits.clamp(1e-6)).max().item()
            if sat > 0.95: sat_rate += 1
            dof = env.dof_pos[0]; lim = env.dof_pos_limits
            for ji in [env.dof_names.index(jn) for jn in RIGHT_ARM]:
                m = min(float(dof[ji] - lim[ji,0]), float(lim[ji,1] - dof[ji])); limit_margin_min = min(limit_margin_min, m)

        e_final = env.rigid_body_states[0, re_i, :3]; final_disp = torch.norm(e_final - e0).item()
        sat_pct = sat_rate / 80
        results.append({"mag": mag, "max_disp": max_disp, "final_disp": final_disp,
                        "sat_rate": sat_pct, "limit_margin": limit_margin_min})
        print(f"  mag={mag:.1f}: max_disp={max_disp:.4f} final={final_disp:.4f} sat={sat_pct:.2f} lim_margin={limit_margin_min:.3f}")

    p = OUT_DIR / "right_arm_mag_scan.csv"
    with open(p, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
    return results


def task3_left_sign_search(env):
    """Left arm: search sign combinations."""
    print("\n" + "=" * 60)
    print("  TASK 3: Left arm sign search")
    print("=" * 60)
    bn = list(env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0]))
    le_i = [i for i,n in enumerate(bn) if n=='left_elbow_link'][0]
    results = []

    # Full 3^4 = 81 combinations
    for sp in [-1, 0, +1]:
        for sr in [-1, 0, +1]:
            for sy in [-1, 0, +1]:
                for se in [-1, 0, +1]:
                    saved_root, saved_quat, default = get_setup(env)
                    e0 = env.rigid_body_states[0, le_i, :3].clone()
                    max_disp = 0; sat_rate = 0; limiting_margin = 99
                    mag = 0.5

                    for s in range(80):
                        desired = default.clone()
                        for jn, sign in zip(LEFT_ARM, [sp, sr, sy, se]):
                            ji = env.dof_names.index(jn)
                            desired[ji] += sign * mag * env.action_scale_vec[ji]
                        step_pd_fixed(env, desired.unsqueeze(0).repeat(6, 1), saved_root, saved_quat)
                        ep = env.rigid_body_states[0, le_i, :3]; disp = torch.norm(ep - e0).item()
                        if disp > max_disp: max_disp = disp
                        sat = (env.torques[0].abs() / env.torque_limits.clamp(1e-6)).max().item()
                        if sat > 0.95: sat_rate += 1

                    e_final = env.rigid_body_states[0, le_i, :3]; final_disp = torch.norm(e_final - e0).item()
                    fwd = float(e_final[0] - e0[0]); out = float(e_final[1] - e0[1]); up = float(e_final[2] - e0[2])
                    results.append({"sp": sp, "sr": sr, "sy": sy, "se": se,
                        "max_disp": max_disp, "final_disp": final_disp,
                        "fwd": fwd, "outward": out, "upward": up,
                        "sat_rate": sat_rate/80})

    results.sort(key=lambda x: -x['max_disp'])
    print(f"  Top 10 (by max_disp):")
    for i, r in enumerate(results[:10]):
        p_str = f"sp={r['sp']:+d} sr={r['sr']:+d} sy={r['sy']:+d} se={r['se']:+d}"
        print(f"  [{i}] {p_str}: max={r['max_disp']:.4f} fwd={r['fwd']:+.3f} out={r['outward']:+.3f} up={r['upward']:+.3f} sat={r['sat_rate']:.2f}")

    p = OUT_DIR / "left_arm_sign_search.csv"
    with open(p, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
    return results


def task4_joint_symmetry(env, cfg):
    """Compare left vs right arm joint limits, efforts, axes."""
    print("\n" + "=" * 60)
    print("  TASK 4: Left/Right Joint Symmetry")
    print("=" * 60)
    pairs = [('left_shoulder_pitch_joint','right_shoulder_pitch_joint'),
             ('left_shoulder_roll_joint','right_shoulder_roll_joint'),
             ('left_shoulder_yaw_joint','right_shoulder_yaw_joint'),
             ('left_elbow_joint','right_elbow_joint')]

    rows = []
    for ln, rn in pairs:
        li = env.dof_names.index(ln); ri = env.dof_names.index(rn)
        l_lo = env.dof_pos_limits[li, 0].item(); l_hi = env.dof_pos_limits[li, 1].item()
        r_lo = env.dof_pos_limits[ri, 0].item(); r_hi = env.dof_pos_limits[ri, 1].item()
        l_eff = env.torque_limits[li].item(); r_eff = env.torque_limits[ri].item()
        l_kp = env.p_gains[li].item(); r_kp = env.p_gains[ri].item()
        l_kd = env.d_gains[li].item(); r_kd = env.d_gains[ri].item()
        sym_limit = "✅" if abs(l_lo + r_hi) < 0.05 and abs(l_hi + r_lo) < 0.05 else "⚠️"
        sym_eff = "✅" if abs(l_eff - r_eff) < 0.1 else "⚠️"
        sym_kp = "✅" if abs(l_kp - r_kp) < 0.1 else "⚠️"
        print(f"  {ln[:25]}: lim=[{l_lo:.2f},{l_hi:.2f}] eff={l_eff:.1f} kp={l_kp:.1f} kd={l_kd:.2f}")
        print(f"  {rn[:25]}: lim=[{r_lo:.2f},{r_hi:.2f}] eff={r_eff:.1f} kp={r_kp:.1f} kd={r_kd:.2f}")
        print(f"    limit_sym={sym_limit} effort_sym={sym_eff} kp_sym={sym_kp}")
        rows.append({"pair": f"{ln}/{rn}", "l_lo": l_lo, "l_hi": l_hi, "r_lo": r_lo, "r_hi": r_hi,
                     "l_eff": l_eff, "r_eff": r_eff, "limit_sym": sym_limit, "effort_sym": sym_eff})
    p = OUT_DIR / "left_right_arm_dof_symmetry.json"
    with open(p, "w") as f: json.dump(rows, f, indent=2)
    return rows


def task7_fixed_base_workspace_compare(env):
    """Compare left vs right arm workspace under direct PD target."""
    print("\n" + "=" * 60)
    print("  TASK 7: Fixed-base workspace L/R compare")
    print("=" * 60)
    bn = list(env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0]))

    for side, arm_joints, elbow_name in [
        ("right", RIGHT_ARM, "right_elbow_link"),
        ("left", LEFT_ARM, "left_elbow_link")]:
        saved_root, saved_quat, default = get_setup(env)
        ei = [i for i,n in enumerate(bn) if n==elbow_name][0]
        e_default = env.rigid_body_states[0, ei, :3].clone()
        points = []
        max_disp = 0

        # Single-joint sweeps
        for jn in arm_joints:
            ji = env.dof_names.index(jn)
            for delta in [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]:
                target = default.clone(); target[ji] += delta
                for _ in range(60): step_pd_fixed(env, target.unsqueeze(0).repeat(6, 1), saved_root, saved_quat)
                ep = env.rigid_body_states[0, ei, :3]; disp = torch.norm(ep - e_default).item()
                points.append(ep.cpu().tolist())
                if disp > max_disp: max_disp = disp

        # Random combos
        for _ in range(200):
            target = default.clone()
            for jn in arm_joints:
                ji = env.dof_names.index(jn); lo, hi = env.dof_pos_limits[ji,0].item(), env.dof_pos_limits[ji,1].item()
                target[ji] = np.random.uniform(max(lo, default[ji]-0.8), min(hi, default[ji]+0.8))
            for _ in range(60): step_pd_fixed(env, target.unsqueeze(0).repeat(6, 1), saved_root, saved_quat)
            ep = env.rigid_body_states[0, ei, :3]; disp = torch.norm(ep - e_default).item()
            points.append(ep.cpu().tolist())
            if disp > max_disp: max_disp = disp

        xs = [p[0] for p in points]; ys = [p[1] for p in points]; zs = [p[2] for p in points]
        print(f"  {side}: max_disp={max_disp:.3f}m x=[{min(xs):.3f},{max(xs):.3f}] y=[{min(ys):.3f},{max(ys):.3f}] z=[{min(zs):.3f},{max(zs):.3f}]")

    return


def main():
    print("=" * 60)
    print("  Q1 Left Arm Asymmetry Diagnostic")
    print("=" * 60)
    env, cfg = make_env(6)
    task1_action_scale_check(env)
    mag_results = task2_right_mag_scan(env)
    sign_results = task3_left_sign_search(env)
    sym_results = task4_joint_symmetry(env, cfg)
    task7_fixed_base_workspace_compare(env)

    # Report
    print(f"\n{'='*60}")
    print(f"  ANSWERS")
    print(f"{'='*60}")
    best_r = max(mag_results, key=lambda x: x['max_disp'])
    print(f"  1. Right best mag: {best_r['mag']} → max_disp={best_r['max_disp']:.3f}m")
    print(f"  2. Right mag=1.0: {'torque sat' if best_r['sat_rate']>0.5 else 'unknown'}")

    best_l = sign_results[0] if sign_results else None
    if best_l:
        print(f"  3. Left best sign: sp={best_l['sp']:+d} sr={best_l['sr']:+d} sy={best_l['sy']:+d} se={best_l['se']:+d} → max_disp={best_l['max_disp']:.3f}m")
    else:
        print(f"  3. No left sign results")

    print(f"  4. Saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
