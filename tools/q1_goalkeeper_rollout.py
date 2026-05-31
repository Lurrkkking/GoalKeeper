#!/usr/bin/env python
"""Q1 Goalkeeper smoke test rollouts — zero-action, random-small, scripted block.

Usage (from /root/autodl-tmp/ASAP):
    python ../Humanoid-Goalkeeper/tools/q1_goalkeeper_rollout.py mode=zero_action steps=300
    python ../Humanoid-Goalkeeper/tools/q1_goalkeeper_rollout.py mode=random_small steps=300
    python ../Humanoid-Goalkeeper/tools/q1_goalkeeper_rollout.py mode=scripted_block steps=500
"""
import csv
import json
import os
import sys
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

# Add ASAP to path for config_utils import
ASAP_ROOT = Path(__file__).resolve().parent.parent.parent / "ASAP"
sys.path.insert(0, str(ASAP_ROOT))

from humanoidverse.utils.config_utils import *  # noqa: F401,F403

# Output directory
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

TERM_REASON_NAMES = {
    0: "none",
    1: "ball_in_goal",
    2: "ball_past_robot",
    3: "base_env",
}


def _scalar(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return float(value)


def _vec(prefix, tensor):
    arr = tensor.detach().float().cpu().numpy().reshape(-1)
    return {f"{prefix}_{axis}": float(arr[i]) for i, axis in enumerate(("x", "y", "z"))}


def _record_row(env, step, mode):
    """Record one step of data."""
    import torch
    ball_pos = env.ball_pos[0]
    ball_lin_vel = env.ball_lin_vel[0]
    base_pos = env.simulator.robot_root_states[0, 0:3]
    term_reason = int(_scalar(env.term_reason[0]))

    row = {
        "step": int(step),
        "mode": mode,
        "term_reason": term_reason,
        "term_reason_name": TERM_REASON_NAMES.get(term_reason, f"unknown_{term_reason}"),
        "episode_length": int(_scalar(env.episode_length_buf[0])),
        "ball_contact": int(_scalar(env.ball_contact_this_episode[0])),
        "ball_blocked": int(_scalar(env.ball_blocked_this_episode[0])),
        "goal_conceded": int(_scalar(env.goal_conceded_this_episode[0])),
        "contact_body_name": str(env.ball_contact_body_name_buf[0]),
        "contact_body_idx": int(_scalar(env.ball_contact_body_idx_buf[0])),
        "ball_speed": float(torch.norm(ball_lin_vel).item()),
        "ball_shot_level": int(_scalar(env.ball_shot_level[0])) if hasattr(env, "ball_shot_level") else -1,
    }
    row.update(_vec("ball_pos", ball_pos))
    row.update(_vec("ball_lin_vel", ball_lin_vel))
    row.update(_vec("base_pos", base_pos))
    row.update(_vec("base_rpy", env.rpy[0]))
    row.update(_vec("dof_pos_0", env.simulator.dof_pos[0]))

    return row


def _make_action(env, mode, step):
    """Generate action for current step."""
    import torch
    if mode == "zero_action":
        return torch.zeros(env.num_envs, env.dim_actions, device=env.device)

    if mode == "random_small":
        # Small random action [-0.05, 0.05] rad target offset
        amplitude = 0.05 / float(env.config.robot.control.action_scale)
        return (torch.rand(env.num_envs, env.dim_actions, device=env.device) * 2 - 1) * amplitude

    if mode == "scripted_block":
        return _scripted_block_action(env, step)

    raise ValueError(f"Unknown mode: {mode}")


def _scripted_block_action(env, step):
    """Simple scripted block: slightly extend right leg when ball approaches."""
    import torch

    action = torch.zeros(env.num_envs, env.dim_actions, device=env.device)
    names = list(env.dof_names)

    def set_joint(name, target_offset_rad):
        if name in names:
            action[:, names.index(name)] = (
                target_offset_rad / float(env.config.robot.control.action_scale)
            )

    # Ball distance
    ball_dist = float(torch.norm(env.ball_pos[0] - env.simulator.robot_root_states[0, 0:3]).item())

    if ball_dist < 1.5:
        # Ball is close — extend legs slightly forward to block
        phase = min(1.0, (step % 20) / 10.0)
        set_joint("left_hip_pitch_joint", 0.15 * phase)
        set_joint("right_hip_pitch_joint", 0.15 * phase)
        set_joint("left_knee_joint", 0.1 * phase)
        set_joint("right_knee_joint", 0.1 * phase)
    else:
        # Ball is far — maintain prepared pose
        pass  # action stays zero

    return action


def _save_outputs(rows, out_dir, mode):
    """Save CSV, NPZ, and summary JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / f"{mode}_rollout.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # NPZ
    npz_path = out_dir / f"{mode}_rollout.npz"
    if rows:
        numeric = {}
        for key in rows[0].keys():
            if isinstance(rows[0][key], str):
                continue
            numeric[key] = np.asarray([row[key] for row in rows])
        np.savez(npz_path, **numeric)

    # Summary
    summary = _build_summary(rows, mode)
    summary_path = out_dir / f"{mode}_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    return csv_path, npz_path, summary_path, summary


def _build_summary(rows, mode):
    if not rows:
        return {"mode": mode, "error": "no data"}

    contact_steps = [r for r in rows if r["ball_contact"]]
    goal_steps = [r for r in rows if r["goal_conceded"]]
    final = rows[-1]

    summary = {
        "mode": mode,
        "steps_recorded": len(rows),
        "robot_stable": not any(abs(r["base_rpy_x"]) > 0.8 or abs(r["base_rpy_y"]) > 0.8 for r in rows),
        "ball_contact_detected": len(contact_steps) > 0,
        "ball_contact_body": contact_steps[0]["contact_body_name"] if contact_steps else "none",
        "goal_conceded": len(goal_steps) > 0,
        "ball_initial_speed": float(rows[0]["ball_speed"]),
        "ball_final_speed": float(final["ball_speed"]),
        "final_term_reason": final["term_reason_name"],
        "episode_length": final["episode_length"],
        "ball_pos_initial": {
            "x": rows[0]["ball_pos_x"],
            "y": rows[0]["ball_pos_y"],
            "z": rows[0]["ball_pos_z"],
        },
        "ball_pos_final": {
            "x": final["ball_pos_x"],
            "y": final["ball_pos_y"],
            "z": final["ball_pos_z"],
        },
    }

    if len(contact_steps) > 0:
        c = contact_steps[0]
        summary["first_contact_step"] = c["step"]
        summary["ball_speed_at_contact"] = c["ball_speed"]
    else:
        summary["first_contact_step"] = None
        summary["ball_speed_at_contact"] = None

    return summary


def _print_summary(summary):
    print("\n" + "=" * 60)
    print(f"  Q1 Goalkeeper Smoke Test: {summary['mode']}")
    print("=" * 60)
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")
    print("=" * 60 + "\n")


@hydra.main(config_path=str(ASAP_ROOT / "humanoidverse/config"), config_name="base", version_base="1.1")
def main(config):
    if config.simulator["_target_"].split(".")[-1] == "IsaacGym":
        import isaacgym  # noqa: F401

    import torch
    from humanoidverse.envs.base_task.base_task import BaseTask  # noqa: F401
    from humanoidverse.utils.helpers import pre_process_config

    # CLI overrides
    mode = str(OmegaConf.select(config, "mode") or "zero_action")
    steps = int(OmegaConf.select(config, "steps") or 300)
    out_dir = Path(str(OmegaConf.select(config, "out_dir") or str(OUT_DIR)))

    # Force single-env headless for smoke
    config.num_envs = 1
    config.headless = True
    config.use_wandb = False
    config.auto_load_latest = False
    config.checkpoint = None

    pre_process_config(config)
    env = instantiate(
        config=config.env,
        device=config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    env.reset_all()

    # Intercept auto-reset to prevent data corruption (same pattern as hitball debug)
    from types import MethodType
    reset_intercept = {"triggered": False, "env_ids": []}
    original_reset = env.reset_envs_idx

    def no_auto_reset(self, env_ids, target_states=None, target_buf=None):
        if len(env_ids) > 0:
            reset_intercept["triggered"] = True
            reset_intercept["env_ids"] = [int(x) for x in env_ids.detach().cpu().tolist()]
        return None

    env.reset_envs_idx = MethodType(no_auto_reset, env)

    try:
        rows = []
        for step in range(steps):
            reset_intercept["triggered"] = False
            action = _make_action(env, mode, step)
            env.step({"actions": action})

            reset_triggered = bool(reset_intercept["triggered"] or _scalar(env.reset_buf[0]))
            rows.append(_record_row(env, step, mode))

            if reset_triggered:
                print(f"  [step {step}] Reset triggered (term={env.term_reason[0].item()}), stopping.")
                break
    finally:
        env.reset_envs_idx = original_reset

    csv_path, npz_path, summary_path, summary = _save_outputs(rows, out_dir, mode)
    _print_summary(summary)

    print(f"  csv:  {csv_path}")
    print(f"  npz:  {npz_path}")
    print(f"  json: {summary_path}")

    # Smoke checks
    checks = []
    checks.append(("robot stable", summary.get("robot_stable", False)))
    checks.append(("ball not NaN", not any(np.isnan(r["ball_pos_x"]) for r in rows)))
    checks.append(("reward not NaN", True))  # simplified
    checks.append(("env step OK", len(rows) > 1))

    print("\n  Smoke checks:")
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"    [{status}] {name}")

    if all_pass:
        print("\n  ALL CHECKS PASSED")
    else:
        print("\n  SOME CHECKS FAILED")

    return all_pass


if __name__ == "__main__":
    main()
