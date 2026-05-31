#!/usr/bin/env python
"""Q1 Goalkeeper actuation diagnostic — tests if joints can actually move.

Usage:
    cd /root/autodl-tmp/Humanoid-Goalkeeper
    python legged_gym/legged_gym/scripts/diagnose_q1_actuation.py
"""
import sys, os, json, csv, time
from pathlib import Path
import numpy as np

# Setup path
SCRIPT_DIR = Path(__file__).resolve().parent
LEGGED_GYM_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(LEGGED_GYM_ROOT))

import isaacgym; from isaacgym import gymtorch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

OUT_DIR = LEGGED_GYM_ROOT / ".." / "outputs" / "q1_actuation_diag"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAMES_ZERO = 20
FRAMES_ACTION = 80
ONE_HOT_VALUE = 1.0
DIRECT_TARGET_VALUE = 0.2  # rad


def make_env(n=1):
    args = get_args(); args.task = "q1"; args.headless = True; args.num_envs = 6
    return task_registry.make_env(name=args.task, args=args)


def record_asset_info(env, cfg):
    """Task 1: Print and save actual loaded asset info."""
    info = {
        "urdf_path": str(cfg.asset.file).replace("{LEGGED_GYM_ROOT_DIR}", str(LEGGED_GYM_ROOT_DIR)),
        "num_dofs": env.num_dof,
        "num_actions": env.num_actions,
        "num_bodies": env.num_bodies,
        "dof_names": list(env.dof_names),
        "action_scale": float(cfg.control.action_scale),
        "control_type": cfg.control.control_type,
        "decimation": int(cfg.control.decimation),
        "dt": float(env.dt) if hasattr(env, 'dt') else 0.02,
        "stiffness": dict(cfg.control.stiffness),
        "damping": dict(cfg.control.damping),
        "default_dof_pos": env.default_dof_pos[0].cpu().tolist(),
        "torque_limits": env.torque_limits.cpu().tolist(),
        "dof_pos_lower": env.dof_pos_limits[:, 0].cpu().tolist(),
        "dof_pos_upper": env.dof_pos_limits[:, 1].cpu().tolist(),
        "root_z": float(cfg.init_state.pos[2]),
        "action_scale_vec": env.action_scale_vec.cpu().tolist() if hasattr(env, 'action_scale_vec') else [],
    }
    with open(OUT_DIR / "asset_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)

    print("=" * 70)
    print("  TASK 1: Asset Info")
    print("=" * 70)
    print(f"  URDF: {info['urdf_path']}")
    assert "goalkeeper_collision" in info['urdf_path'], "WRONG URDF! Expected goalkeeper_collision"
    print(f"  num_dofs={info['num_dofs']} num_actions={info['num_actions']} bodies={info['num_bodies']}")
    print(f"  control={info['control_type']} action_scale={info['action_scale']} root_z={info['root_z']}")
    print(f"  per_joint_action_scale: {len(info['action_scale_vec'])} values")
    print()

    # Task 2: DOF/action mapping table
    print("=" * 70)
    print("  TASK 2: DOF/Action Order")
    print("=" * 70)
    print(f"  {'idx':>3} {'dof_name':<30} {'default':>8} {'lower':>8} {'upper':>8} {'effort':>8} {'act_scale':>8}")
    for i in range(env.num_dof):
        asc = info['action_scale_vec'][i] if i < len(info['action_scale_vec']) else info['action_scale']
        print(f"  {i:>3} {info['dof_names'][i]:<30} {info['default_dof_pos'][i]:>8.3f} "
              f"{info['dof_pos_lower'][i]:>8.3f} {info['dof_pos_upper'][i]:>8.3f} "
              f"{info['torque_limits'][i]:>8.1f} {asc:>8.3f}")
    print()

    return info


def run_one_hot_scan(env, info):
    """Task 3: One-hot action scan for each action index."""
    print("=" * 70)
    print("  TASK 3: One-Hot Action Scan")
    print("=" * 70)

    results = []
    na = env.num_actions
    nd = env.num_dof

    for ai in range(na):
        dof_name = info['dof_names'][ai]

        # Test +1.0
        env.reset_idx(torch.tensor([0], device=env.device))
        for _ in range(FRAMES_ZERO):
            env.step(torch.zeros(6, na, device=env.device))
        for _ in range(FRAMES_ACTION):
            act = torch.zeros(6, na, device=env.device)
            act[:, ai] = ONE_HOT_VALUE
            env.step(act)
        # Record final state
        dof_delta_pos = (env.dof_pos[0] - env.default_dof_pos[0]).abs()
        max_idx = dof_delta_pos.argmax().item()
        max_delta_pos = dof_delta_pos[max_idx].item()
        actual_delta_pos = dof_delta_pos[ai].item()
        tau_max = env.torques[0].abs().max().item()
        tau_sat = (env.torques[0].abs() / env.torque_limits.clamp(1e-6)).max().item()

        # Test -1.0
        env.reset_idx(torch.tensor([0], device=env.device))
        for _ in range(FRAMES_ZERO):
            env.step(torch.zeros(6, na, device=env.device))
        for _ in range(FRAMES_ACTION):
            act = torch.zeros(6, na, device=env.device)
            act[:, ai] = -ONE_HOT_VALUE
            env.step(act)
        dof_delta_neg = (env.dof_pos[0] - env.default_dof_pos[0]).abs()
        actual_delta_neg = dof_delta_neg[ai].item()
        tau_max2 = env.torques[0].abs().max().item()

        mapping_ok = (max_idx == ai)
        action_scale = info['action_scale_vec'][ai] if ai < len(info['action_scale_vec']) else info['action_scale']
        target_delta = ONE_HOT_VALUE * action_scale
        tracking = actual_delta_pos / max(target_delta, 1e-6)

        status = "OK" if mapping_ok and tracking > 0.1 else ("MAP_ERR" if not mapping_ok else "LOW_TRACK")
        print(f"  [{ai:>2}] {dof_name:<30} +: max_idx={max_idx}({info['dof_names'][max_idx]}) "
              f"d={actual_delta_pos:.4f} target={target_delta:.4f} track={tracking:.2f} "
              f"tau={tau_max:.1f} sat={tau_sat:.2f} -: d={actual_delta_neg:.4f} {status}")

        results.append({
            "action_idx": ai, "dof_name": dof_name,
            "max_moved_idx": max_idx, "max_moved_name": info['dof_names'][max_idx],
            "mapping_correct": mapping_ok,
            "actual_delta_pos": float(actual_delta_pos),
            "actual_delta_neg": float(actual_delta_neg),
            "target_delta": float(target_delta),
            "tracking_ratio_pos": float(tracking),
            "tau_max_pos": float(tau_max), "tau_max_neg": float(tau_max2),
            "saturation_rate": float(tau_sat),
            "status": status,
        })

    with open(OUT_DIR / "one_hot_scan.json", "w") as f:
        json.dump(results, f, indent=2)

    ok = sum(1 for r in results if r['mapping_correct'])
    moved = sum(1 for r in results if r['actual_delta_pos'] > 0.01)
    print(f"\n  Mapping: {ok}/{na} correct. Joints moved >0.01rad: {moved}/{na}")
    return results


def run_direct_target_scan(env, info):
    """Task 4: Direct DOF target test — bypass action scale."""
    print("\n" + "=" * 70)
    print("  TASK 4: Direct DOF Target Scan (±0.2 rad)")
    print("=" * 70)

    results = []
    nd = env.num_dof

    for i in range(nd):
        dof_name = info['dof_names'][i]

        for sign, label in [(+1, '+'), (-1, '-')]:
            env.reset_idx(torch.tensor([0], device=env.device))
            # Override PD target directly
            target = env.default_dof_poses[0].clone()
            target[i] += DIRECT_TARGET_VALUE * sign
            for s in range(FRAMES_ACTION):
                env.joint_pos_target[0] = target
                env.torques = env.p_gains * (target - env.dof_pos) - env.d_gains * env.dof_vel
                env.torques = torch.clip(env.torques, -env.torque_limits, env.torque_limits)
                env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.torques.unsqueeze(0)))
                env.gym.simulate(env.sim); env.gym.fetch_results(env.sim, True)
                env.gym.refresh_dof_state_tensor(env.sim)
                env.gym.refresh_actor_root_state_tensor(env.sim)
                env.gym.refresh_rigid_body_state_tensor(env.sim)

            actual_delta = (env.dof_pos[0, i] - env.default_dof_pos[0, i]).item()
            tracking = abs(actual_delta / DIRECT_TARGET_VALUE) if abs(DIRECT_TARGET_VALUE) > 1e-6 else 0
            tau = env.torques[0, i].abs().item()
            status = "OK" if tracking > 0.3 else "STUCK"

        results.append({
            "joint_idx": i, "dof_name": dof_name,
            "actual_delta": float(actual_delta),
            "tracking_ratio": float(tracking),
            "torque": float(tau),
            "status": status,
        })

    moved = sum(1 for r in results if r['status'] == 'OK')
    print(f"  Joints moved >30% target: {moved}/{nd}")
    for r in results:
        if r['status'] != 'OK':
            print(f"  ⚠️ {r['dof_name']:<30} delta={r['actual_delta']:+.4f} tracking={r['tracking_ratio']:.2f}")
    print()

    with open(OUT_DIR / "direct_target_scan.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_observation_check(env):
    """Task 10: Check what's in the observation."""
    print("=" * 70)
    print("  TASK 10: Observation Check")
    print("=" * 70)
    env.reset_idx(torch.tensor([0], device=env.device))
    env.step(torch.zeros(6, env.num_actions, device=env.device))
    obs = env.obs_buf[0]
    print(f"  obs_dim: {obs.shape[0]}")
    print(f"  num_one_step_obs: {env.num_one_step_obs}")
    print(f"  actor_history: {env.actor_history_length}")
    # The one-step obs breakdown (from env config)
    # 6(base) + num_ballobs(3) + num_dof(22)*2 + num_actions(22) = 75
    ball_start = 6
    ball_end = 6 + env.cfg.env.num_ballobs
    dof_start = ball_end
    dof_end = dof_start + env.num_dof
    dof_vel_start = dof_end
    dof_vel_end = dof_vel_start + env.num_dof
    act_start = dof_vel_end
    act_end = act_start + env.num_actions
    print(f"  Obs structure: base[0:6] ball[{ball_start}:{ball_end}] dof_pos[{dof_start}:{dof_end}] dof_vel[{dof_vel_start}:{dof_vel_end}] actions[{act_start}:{act_end}]")
    print(f"  ball obs (first 3): {obs[ball_start:ball_end].cpu().tolist()}")
    print(f"  dof_pos (first 6): {obs[dof_start:dof_start+6].cpu().tolist()}")
    print()

    obs_info = {
        "obs_dim": obs.shape[0],
        "num_one_step_obs": env.num_one_step_obs,
        "actor_history": env.actor_history_length,
        "ball_in_obs": ball_start < ball_end,
        "ball_obs_range": [ball_start, ball_end],
    }
    with open(OUT_DIR / "obs_debug.json", "w") as f:
        json.dump(obs_info, f, indent=2)
    return obs_info


def main():
    print("=" * 70)
    print("  Q1 Goalkeeper Actuation Diagnostic")
    print("=" * 70)

    env, cfg = make_env()

    # Task 1+2
    info = record_asset_info(env, cfg)

    # Task 3
    one_hot_results = run_one_hot_scan(env, info)

    # Task 4
    direct_results = run_direct_target_scan(env, info)

    # Task 10
    obs_info = run_observation_check(env)

    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL REPORT")
    print("=" * 70)

    # Table 2: one-hot mapping
    mapping_ok = sum(1 for r in one_hot_results if r['mapping_correct'])
    print(f"\n  One-hot mapping: {mapping_ok}/{len(one_hot_results)} correct")
    for r in one_hot_results:
        if not r['mapping_correct']:
            print(f"    ⚠️ action[{r['action_idx']}]={r['dof_name']} -> max moved: {r['max_moved_name']}")

    # Table 3: direct target
    direct_moved = sum(1 for r in direct_results if r['status'] == 'OK')
    print(f"\n  Direct target: {direct_moved}/{len(direct_results)} joints moved >30% target")
    stuck = [r for r in direct_results if r['status'] != 'OK']
    if stuck:
        print(f"  Stuck joints:")
        for r in stuck:
            print(f"    ⚠️ {r['dof_name']}: delta={r['actual_delta']:.4f} tracking={r['tracking_ratio']:.2f}")

    # Key answers
    print(f"\n  Answers:")
    print(f"  1. URDF: {'✅ goalkeeper_collision' if 'goalkeeper_collision' in info['urdf_path'] else '❌ WRONG'}")
    print(f"  2. Mapping: {'✅' if mapping_ok == len(one_hot_results) else '❌ {}/{}}'.format(mapping_ok, len(one_hot_results))}")
    print(f"  3. One-hot moves joints: {'✅' if sum(1 for r in one_hot_results if r['actual_delta_pos']>0.01) > 15 else '⚠️ few joints move'}")
    print(f"  4. Direct target moves joints: {'✅' if direct_moved > 15 else '⚠️ few joints move'}")
    print(f"  5. Ball in obs: {'✅' if obs_info['ball_in_obs'] else '❌'}")

    print(f"\n  Saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
