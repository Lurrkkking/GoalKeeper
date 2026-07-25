"""Shared ball-free observation helpers for the ready-stand ONNX policies."""

import os
import numpy as np


def resolve_standby_policy_path(cfg, cli_path):
    path = cli_path or cfg.get("standby_policy_path")
    if not path:
        return None
    if not os.path.isabs(path):
        path = os.path.normpath(os.path.join(cfg["_config_dir"], path))
    return path


def standby_dimensions(cfg):
    n = int(cfg["num_actions"])
    single = int(cfg.get("standby_num_single_obs", 6 + 3 * n))
    stack = int(cfg.get("standby_frame_stack", cfg["frame_stack"]))
    total = int(cfg.get("standby_num_obs", single * stack))
    if single != 6 + 3 * n or total != single * stack:
        raise ValueError("Invalid standby observation dimensions in config")
    return single, stack, total


def create_standby_history(cfg):
    single, stack, _ = standby_dimensions(cfg)
    return np.zeros((stack, single), dtype=np.float32)


def build_standby_observation(robot_state, last_action, cfg):
    default = np.asarray(cfg["standby_default_dof_pos"], dtype=np.float32)
    n = int(cfg["num_actions"])
    if default.shape != (n,) or last_action.shape != (n,):
        raise ValueError("Standby pose/action dimension does not match robot action count")
    single, _, _ = standby_dimensions(cfg)
    obs = np.concatenate((
        robot_state["ang_vel_base"] * cfg["obs_scale_ang_vel"],
        robot_state["projected_gravity"],
        (robot_state["dof_pos"] - default) * cfg["obs_scale_dof_pos"],
        robot_state["dof_vel"] * cfg["obs_scale_dof_vel"],
        last_action,
    )).astype(np.float32)
    if obs.shape != (single,):
        raise AssertionError(f"Standby single obs shape mismatch: {obs.shape}")
    return obs


def update_standby_history(history, single_obs, cfg):
    _, _, total = standby_dimensions(cfg)
    history[:-1] = history[1:]
    history[-1] = single_obs
    obs = history.reshape(-1).astype(np.float32)
    if obs.shape != (total,):
        raise AssertionError(f"Standby obs shape mismatch: {obs.shape}")
    return obs


def standby_action_to_target(action, cfg):
    return np.asarray(cfg["standby_default_dof_pos"], dtype=np.float64) + action * np.asarray(cfg["action_scale_vec"], dtype=np.float64)
