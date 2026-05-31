#!/usr/bin/env python
"""Q1 arm workspace + eereach target comparison (fixed-base)."""
import sys, json, csv
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
LEGGED_GYM_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(LEGGED_GYM_ROOT))

import isaacgym; from isaacgym import gymtorch
from legged_gym.envs import *; from legged_gym.utils import get_args, task_registry
import torch

OUT_DIR = LEGGED_GYM_ROOT / ".." / "outputs" / "arm_workspace_eereach_diag"
OUT_DIR.mkdir(parents=True, exist_ok=True)

try: import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; HAS_MPL = True
except: HAS_MPL = False

RIGHT_ARM = ['right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint']
LEFT_ARM  = ['left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint']


def make_env(n=6):
    args = get_args(); args.task = "q1"; args.headless = True; args.num_envs = n
    return task_registry.make_env(name=args.task, args=args)


def step_pd_fixed(env, target, saved_root, saved_root_quat):
    """PD step with root/base locked to saved pose."""
    tau = env.p_gains * (target - env.dof_pos) - env.d_gains * env.dof_vel
    tau = torch.clip(tau, -env.torque_limits, env.torque_limits)
    env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(tau))
    env.gym.simulate(env.sim); env.gym.fetch_results(env.sim, True)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)
    # Lock root
    env.root_states[:, :3] = saved_root.unsqueeze(0).repeat(env.num_envs, 1)
    env.root_states[:, 3:7] = saved_root_quat.unsqueeze(0).repeat(env.num_envs, 1)
    env.root_states[:, 7:13] = 0.
    all_s = torch.cat((env.root_states.unsqueeze(1), env.ball_states.unsqueeze(1)), dim=1).view(-1, 13)
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(all_s))


def task3_fixed_base_setup(env):
    """Save root pose for fixed-base simulation."""
    env.reset_idx(torch.tensor([0], device=env.device))
    saved_root = env.root_states[0, :3].clone()
    saved_root_quat = env.root_states[0, 3:7].clone()
    return saved_root, saved_root_quat


def task4_5_arm_workspace(env, saved_root, saved_root_quat, side, arm_joints):
    """Scan arm workspace — single joint sweeps + random combos."""
    print(f"\n  --- {side} arm workspace scan ---")
    default = env.default_dof_poses[0].clone()
    bn = list(env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0]))
    elbow_name = f'{side}_elbow_link'
    elbow_idx = [i for i,n in enumerate(bn) if n == elbow_name][0]
    elbow_default = env.rigid_body_states[0, elbow_idx, :3].clone()

    all_points = []

    # A: Single joint sweep
    for jn in arm_joints:
        ji = env.dof_names.index(jn)
        for delta in [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]:
            target = default.clone(); target[ji] += delta
            for _ in range(60):
                step_pd_fixed(env, target.unsqueeze(0).repeat(6, 1), saved_root, saved_root_quat)
            ep = env.rigid_body_states[0, elbow_idx, :3].clone()
            disp = torch.norm(ep - elbow_default).item()
            all_points.append({"side": side, "joint": jn, "delta": delta, "type": "single",
                               "elbow_x": float(ep[0]), "elbow_y": float(ep[1]), "elbow_z": float(ep[2]),
                               "displacement": disp})

    # B: Random combos
    for _ in range(200):
        target = default.clone()
        for jn in arm_joints:
            ji = env.dof_names.index(jn)
            lo, hi = env.dof_pos_limits[ji, 0].item(), env.dof_pos_limits[ji, 1].item()
            target[ji] = np.random.uniform(max(lo, default[ji].item()-0.8), min(hi, default[ji].item()+0.8))
        for _ in range(60):
            step_pd_fixed(env, target.unsqueeze(0).repeat(6, 1), saved_root, saved_root_quat)
        ep = env.rigid_body_states[0, elbow_idx, :3].clone()
        disp = torch.norm(ep - elbow_default).item()
        all_points.append({"side": side, "joint": "combo", "delta": 0, "type": "combo",
                           "elbow_x": float(ep[0]), "elbow_y": float(ep[1]), "elbow_z": float(ep[2]),
                           "displacement": disp})

    xs = [p['elbow_x'] for p in all_points]; ys = [p['elbow_y'] for p in all_points]; zs = [p['elbow_z'] for p in all_points]
    max_disp = max(p['displacement'] for p in all_points)
    print(f"    max_disp={max_disp:.3f}m  range: x=[{min(xs):.3f},{max(xs):.3f}] y=[{min(ys):.3f},{max(ys):.3f}] z=[{min(zs):.3f},{max(zs):.3f}]")

    # Save
    p = OUT_DIR / f"{side}_arm_workspace.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_points[0].keys()); w.writeheader(); w.writerows(all_points)

    # Plot
    if HAS_MPL:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        combo_pts = [pt for pt in all_points if pt['type'] == 'combo']
        ax1.scatter([pt['elbow_x'] for pt in combo_pts], [pt['elbow_y'] for pt in combo_pts], s=2, alpha=0.5)
        ax1.scatter(elbow_default[0].item(), elbow_default[1].item(), c='red', marker='*', s=100, label='default')
        ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_title(f'{side} elbow workspace (top view)'); ax1.legend(); ax1.grid(True)
        ax2.scatter([pt['elbow_x'] for pt in combo_pts], [pt['elbow_z'] for pt in combo_pts], s=2, alpha=0.5)
        ax2.scatter(elbow_default[0].item(), elbow_default[2].item(), c='red', marker='*', s=100, label='default')
        ax2.set_xlabel('x'); ax2.set_ylabel('z'); ax2.set_title(f'{side} elbow workspace (front view)'); ax2.legend(); ax2.grid(True)
        fig.tight_layout(); fig.savefig(OUT_DIR / f"{side}_arm_workspace.png", dpi=120); plt.close(fig)

    return all_points, elbow_default


def task6_action_path_displacement(env, saved_root, saved_root_quat):
    """Measure elbow displacement from PPO action path."""
    print("\n  --- Action path → elbow displacement ---")
    default = env.default_dof_poses[0].clone()
    bn = list(env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0]))
    asc = env.action_scale_vec
    results = []

    for side, arm_joints in [("right", RIGHT_ARM), ("left", LEFT_ARM)]:
        elbow_name = f'{side}_elbow_link'
        elbow_idx = [i for i,n in enumerate(bn) if n == elbow_name][0]

        for mag in [0.25, 0.5, 1.0, 1.5, 2.0]:
            # Single joint tests
            for jn in arm_joints:
                ji = env.dof_names.index(jn)
                elbow_start = env.rigid_body_states[0, elbow_idx, :3].clone()
                target = default.clone()
                for _ in range(60):
                    target_delta = mag * asc[ji]
                    desired = default.clone(); desired[ji] += target_delta
                    step_pd_fixed(env, desired.unsqueeze(0).repeat(6, 1), saved_root, saved_root_quat)
                ep = env.rigid_body_states[0, elbow_idx, :3]
                disp = torch.norm(ep - elbow_start).item()
                actual_delta = float(env.dof_pos[0, ji] - default[ji])
                tau_max = env.torques[0].abs().max().item()
                sat = float((env.torques[0].abs() / env.torque_limits.clamp(1e-6)).max())
                results.append({"side": side, "joint": jn, "action_mag": mag,
                    "target_delta": float(mag * asc[ji]), "actual_delta": actual_delta,
                    "elbow_displacement": disp, "tau_max": tau_max, "saturation": sat})
                # Reset
                env.reset_idx(torch.tensor([0], device=env.device))
                saved_root2, saved_root_quat2 = task3_fixed_base_setup(env)
                saved_root, saved_root_quat = saved_root2, saved_root_quat2
                default = env.default_dof_poses[0].clone()

            # Combination reach
            elbow_start = env.rigid_body_states[0, elbow_idx, :3].clone()
            for _ in range(60):
                desired = default.clone()
                for jn in arm_joints:
                    ji = env.dof_names.index(jn); desired[ji] += mag * asc[ji] * 0.5
                step_pd_fixed(env, desired.unsqueeze(0).repeat(6, 1), saved_root, saved_root_quat)
            ep = env.rigid_body_states[0, elbow_idx, :3]; disp = torch.norm(ep - elbow_start).item()
            results.append({"side": side, "joint": "combo_reach", "action_mag": mag,
                "target_delta": 0, "actual_delta": 0, "elbow_displacement": disp,
                "tau_max": 0, "saturation": 0})
            env.reset_idx(torch.tensor([0], device=env.device))
            saved_root3, saved_root_quat3 = task3_fixed_base_setup(env)
            saved_root, saved_root_quat = saved_root3, saved_root_quat3
            default = env.default_dof_poses[0].clone()

    # Print key results
    for r in results:
        if r['action_mag'] >= 1.0 and r['joint'] == 'combo_reach':
            print(f"    {r['side']} combo mag={r['action_mag']}: elbow_disp={r['elbow_displacement']:.3f}m")

    p = OUT_DIR / "action_to_elbow_displacement.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
    return results


def task7_eereach_target_compare(env):
    """Compare end_target vs current_ball for eereach computation."""
    print("\n  --- Eereach: end_target vs current_ball ---")
    default = env.default_dof_poses[0].clone()
    bn = list(env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0]))
    re_idx = [i for i,n in enumerate(bn) if n=='right_elbow_link'][0]

    frames = []
    env.reset_idx(torch.tensor([0], device=env.device))

    for s in range(150):
        env.step(torch.zeros(6, env.num_actions, device=env.device))
        bp = env.ball_states[0, :3]; bv = env.ball_states[0, 7:10]
        et = env.end_target[0].clone() if hasattr(env, 'end_target') else bp.clone()
        hp = env.rigid_body_states[0, re_idx, :3]
        ee_raw = env.reward_functions[env.reward_names.index('eereach')]() if 'eereach' in env.reward_names else torch.tensor([0.])

        dist_et = torch.norm(hp - et).item()
        dist_cur = torch.norm(hp - bp).item()
        dist_int = torch.norm(hp - (bp + bv * 0.2)).item()  # 0.2s intercept

        frames.append({"frame": s, "ball_x": float(bp[0]), "ball_y": float(bp[1]), "ball_z": float(bp[2]),
            "end_target_x": float(et[0]), "end_target_y": float(et[1]), "end_target_z": float(et[2]),
            "dist_to_end_target": dist_et, "dist_to_current_ball": dist_cur,
            "dist_to_intercept_0p2": dist_int, "ee_raw": float(ee_raw[0]) if ee_raw.numel()>0 else 0,
            "et_ball_err": torch.norm(et - bp).item()})

        if env.reset_buf[0] and s > 0: break

    p = OUT_DIR / "eereach_target_compare.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=frames[0].keys()); w.writeheader(); w.writerows(frames)

    # Lag analysis
    errs = [f['et_ball_err'] for f in frames]
    print(f"    end_target vs ball error: mean={np.mean(errs):.3f}m max={max(errs):.3f}m")

    if HAS_MPL:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        xs = [f['frame'] for f in frames]
        ax1.plot(xs, [f['dist_to_end_target'] for f in frames], 'b-', label='dist to end_target')
        ax1.plot(xs, [f['dist_to_current_ball'] for f in frames], 'r--', label='dist to current_ball')
        ax1.set_ylabel('distance (m)'); ax1.legend(); ax1.grid(True)
        ax2.plot(xs, [f['ee_raw'] for f in frames], 'g-', label='ee_raw')
        ax2.plot(xs, errs, 'orange', label='|end_target - ball|')
        ax2.set_ylabel('value'); ax2.set_xlabel('frame'); ax2.legend(); ax2.grid(True)
        fig.suptitle('Eereach: end_target vs current_ball'); fig.tight_layout()
        fig.savefig(OUT_DIR / "eereach_original_vs_current_ball.png", dpi=120); plt.close(fig)

    return frames


def main():
    print("=" * 60)
    print("  Q1 Arm Workspace + Eereach Target Diagnostic")
    print("=" * 60)

    env, cfg = make_env(6)
    hi = env.hand_indices.cpu().tolist() if hasattr(env, 'hand_indices') else []
    bn = list(env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0]))
    print(f"  URDF: {cfg.asset.file}")
    print(f"  hand_indices: {hi} = {[bn[i] for i in hi if i<len(bn)]}")
    print(f"  eereach scale: {env.reward_scales.get('eereach', 0):.4f}")

    # Fixed-base setup
    saved_root, saved_root_quat = task3_fixed_base_setup(env)

    # Arm workspace scan
    right_pts, re_default = task4_5_arm_workspace(env, saved_root, saved_root_quat, "right", RIGHT_ARM)
    # Re-setup for left (root may have drifted)
    env.reset_idx(torch.tensor([0], device=env.device))
    saved_root2, saved_root_quat2 = task3_fixed_base_setup(env)
    left_pts, le_default = task4_5_arm_workspace(env, saved_root2, saved_root_quat2, "left", LEFT_ARM)

    # Re-setup for action path
    env.reset_idx(torch.tensor([0], device=env.device))
    saved_root3, saved_root_quat3 = task3_fixed_base_setup(env)
    action_results = task6_action_path_displacement(env, saved_root3, saved_root_quat3)

    # Eereach target compare (free standing)
    ee_frames = task7_eereach_target_compare(env)

    # Report
    max_disp_r = max(p['displacement'] for p in right_pts)
    max_disp_l = max(p['displacement'] for p in left_pts)
    combo_disps = [r for r in action_results if r['joint']=='combo_reach' and r['action_mag']>=1.0]
    errs = [f['et_ball_err'] for f in ee_frames]

    print(f"\n{'='*60}")
    print(f"  REPORT")
    print(f"{'='*60}")
    print(f"  Right elbow max displacement: {max_disp_r:.3f}m")
    print(f"  Left  elbow max displacement: {max_disp_l:.3f}m")
    print(f"  Action mag=1 combo reach: R={combo_disps[0]['elbow_displacement']:.3f}m L={combo_disps[1]['elbow_displacement']:.3f}m")
    print(f"  end_target vs ball error: mean={np.mean(errs):.3f}m max={max(errs):.3f}m")
    print(f"  Saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
