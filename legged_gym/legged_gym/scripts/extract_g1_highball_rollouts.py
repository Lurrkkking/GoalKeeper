#!/usr/bin/env python3
"""
extract_g1_highball_rollouts.py — G1 goalkeeper high-ball rollout extraction & analysis.

Extracts raw trajectory data from G1 high-ball episodes, classifies whether G1 actually
jumps to block or just stands/reaches/leans back. Generates low-dimensional skill priors
for Q1 migration.

Does NOT modify training, reward, policy, or env physics.
"""

import os
import sys
import json
import argparse
import csv
import time as time_module
from pathlib import Path
from collections import defaultdict

import numpy as np

# NOTE: isaacgym MUST be imported before torch
import isaacgym
from isaacgym import gymtorch, gymapi, gymutil
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import get_load_path, class_to_dict
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.legged_robot import LeggedRobot


# ─────────────────────────────────────────────────────────────
# Monkey-patch BaseTask for headless GPU rendering (same as play_video.py)
# ─────────────────────────────────────────────────────────────
_original_base_init = BaseTask.__init__


def _patched_base_init(self, cfg, sim_params, physics_engine, sim_device, headless):
    self.gym = gymapi.acquire_gym()
    self.sim_params = sim_params
    self.physics_engine = physics_engine
    self.sim_device = sim_device
    sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)

    if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:
        self.device = self.sim_device
    else:
        self.device = 'cpu'

    if headless:
        self.graphics_device_id = self.sim_device_id
        self.headless = True
    else:
        self.graphics_device_id = self.sim_device_id
        self.headless = False

    self.num_envs = cfg.env.num_envs
    self.num_obs = cfg.env.num_observations
    self.num_privileged_obs = cfg.env.num_privileged_obs
    self.num_actions = cfg.env.num_actions
    self.num_one_step_obs = cfg.env.num_one_step_observations

    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)

    self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
    self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
    self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
    self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    if self.num_privileged_obs is not None:
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
    else:
        self.privileged_obs_buf = None

    self.extras = {}
    self.create_sim()
    self.gym.prepare_sim(self.sim)
    self.enable_viewer_sync = True
    self.viewer = None


BaseTask.__init__ = _patched_base_init


# ─────────────────────────────────────────────────────────────
# Body name lookup helper
# ─────────────────────────────────────────────────────────────
CANDIDATE_BODY_KEYWORDS = [
    # hands / arms
    "hand", "wrist", "forearm", "elbow",
    # legs
    "foot", "ankle", "toe", "shin", "knee",
    # torso / pelvis
    "pelvis", "torso", "waist", "chest", "hip",
]


def find_candidate_body_indices(body_names, env, actor_handle, env_handle):
    """Find body indices for a broad set of candidate blocking bodies."""
    found = {}
    missing = []
    for keyword in CANDIDATE_BODY_KEYWORDS:
        matches = [n for n in body_names if keyword in n.lower()]
        if matches:
            for name in matches:
                idx = env.gym.find_actor_rigid_body_handle(env_handle, actor_handle, name)
                found[name] = idx
        else:
            missing.append(keyword)
    return found, missing


# ─────────────────────────────────────────────────────────────
# Main extraction function
# ─────────────────────────────────────────────────────────────
def extract(args):
    print("=" * 60)
    print("[extract_g1_highball_rollouts] Starting...")
    print("=" * 60)

    # ── 1. Setup env config ──
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = args.num_envs
    env_cfg.env.episode_length_s = 3
    env_cfg.env.play = True

    # Disable all noise / randomization / push
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_initial_joint_pos = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 999999
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.continue_keep = False
    env_cfg.domain_rand.randomize_kp = False
    env_cfg.domain_rand.randomize_kd = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.domain_rand.randomize_com_displacement = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_joint_injection = False
    env_cfg.domain_rand.randomize_actuation_offset = False
    env_cfg.domain_rand.delay = False
    env_cfg.domain_rand.ball_interval = 999999
    env_cfg.domain_rand.ball_interval_s = 999999
    # Disable additional DR flags if they exist
    for attr in [
        "randomize_motor_strength", "randomize_torque_noise",
        "randomize_action_delay", "randomize_action_filter",
        "randomize_ball_obs_noise", "randomize_ball_obs_dropout",
        "randomize_reset_velocity", "randomize_backward_lean_reset",
        "randomize_hip_pitch_actuator", "randomize_ball_visible_pitch_disturb",
    ]:
        if hasattr(env_cfg.domain_rand, attr):
            setattr(env_cfg.domain_rand, attr, False)

    # ── 2. Create env ──
    print(f"[ENV] Creating {args.num_envs} environments for task={args.task}...")
    # Build args-like object for make_env
    sim_args = type('Args', (), {})()
    sim_args.task = args.task
    sim_args.exptid = args.exptid or "extract"
    sim_args.headless = args.headless
    sim_args.rl_device = "cuda:0"
    sim_args.sim_device = "cuda:0"
    sim_args.num_envs = args.num_envs
    sim_args.resume = False
    sim_args.load_run = -1
    sim_args.checkpoint = -1
    sim_args.resumeid = None
    sim_args.run_name = None
    sim_args.experiment_name = None
    sim_args.seed = None
    sim_args.max_iterations = None
    sim_args.horovod = False
    sim_args.compute_device_id = 0
    sim_args.sim_device_id = 0
    sim_args.pipeline = 'gpu'
    sim_args.graphics_device_id = 0
    sim_args.physics_engine = gymapi.SIM_PHYSX
    sim_args.use_gpu = True
    sim_args.use_gpu_pipeline = True
    sim_args.subscenes = 0
    sim_args.num_threads = 2
    sim_args.device = 'cuda'

    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=sim_args, env_cfg=env_cfg)
    print(f"[ENV] Created. num_envs={env.num_envs}, num_dof={env.num_dof}, num_bodies={env.num_bodies}")

    # Reset Kp, Kd, motor_strength to 1.0 (no randomization)
    env.Kp_factors[:] = 1.0
    env.Kd_factors[:] = 1.0
    if hasattr(env, 'motor_strength'):
        env.motor_strength[:] = 1.0

    # ── 3. Load explicit checkpoint ──
    checkpoint_path = args.checkpoint_path
    print(f"\n[CHECKPOINT] Loading from: {checkpoint_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"[CHECKPOINT] File exists: {checkpoint_path}")
    print(f"[CHECKPOINT] File size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.1f} MB")

    # Create runner without auto-resume, then load explicit checkpoint
    train_cfg.runner.resume = False
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=sim_args, train_cfg=train_cfg
    )
    # Explicitly load the specified checkpoint
    ppo_runner.load(checkpoint_path, load_optimizer=False)
    print(f"[CHECKPOINT] Successfully loaded: {checkpoint_path}")

    policy = ppo_runner.get_inference_policy(device=env.device)
    print("[POLICY] Inference policy ready.")

    # ── 4. Save body/dof names ──
    os.makedirs(args.out_dir, exist_ok=True)
    body_names = list(env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0]))

    with open(os.path.join(args.out_dir, "body_names.txt"), "w") as f:
        for i, name in enumerate(body_names):
            f.write(f"{i}: {name}\n")
    print(f"[SAVE] body_names.txt ({len(body_names)} bodies)")

    with open(os.path.join(args.out_dir, "dof_names.txt"), "w") as f:
        for i, name in enumerate(env.dof_names):
            f.write(f"{i}: {name}\n")
    print(f"[SAVE] dof_names.txt ({len(env.dof_names)} dofs)")

    # Print body names for reference
    print("\n[BODY_NAMES]")
    for i, name in enumerate(body_names):
        print(f"  {i:3d}: {name}")

    # Find candidate blocking body indices
    candidate_body_map, missing_keywords = find_candidate_body_indices(
        body_names, env, env.actor_handles[0], env.envs[0]
    )
    print(f"\n[CANDIDATE_BODIES] Found {len(candidate_body_map)} bodies:")
    for name, idx in sorted(candidate_body_map.items(), key=lambda x: x[1]):
        print(f"  idx={idx:3d}: {name}")
    if missing_keywords:
        print(f"[CANDIDATE_BODIES] Missing keywords: {missing_keywords}")

    # ── 5. Print env mode info and verify high-ball envs ──
    print("\n" + "=" * 60)
    print("[MODE_VERIFY] Checking environment mode assignments...")
    print("=" * 60)

    env_ids_to_extract = list(args.highball_env_ids)
    mode_ok = True
    highball_env_ids_actual = []

    for env_id in range(args.num_envs):
        mode_id = env.end_regions[env_id].item()
        cmd_ranges = env.command_ranges[env_id].cpu().numpy()
        height_range = (cmd_ranges[2], cmd_ranges[3])

        ball_start = env.ball_start[env_id].cpu().numpy() if hasattr(env, 'ball_start') else np.zeros(3)
        ball_init_z = ball_start[2] if len(ball_start) > 2 else 0.0

        is_highball = mode_id in (2, 3)
        marker = " <<< HIGH-BALL" if is_highball else ""
        print(f"  env_id={env_id:3d}  mode={mode_id}  height_z=[{height_range[0]:.2f}, {height_range[1]:.2f}]  "
              f"ball_init_z={ball_init_z:.3f}{marker}")

        if is_highball:
            highball_env_ids_actual.append(env_id)

    print(f"\n[MODE_VERIFY] Actual high-ball envs (mode 2/3): {highball_env_ids_actual}")
    print(f"[MODE_VERIFY] Requested envs to extract: {env_ids_to_extract}")

    for env_id in env_ids_to_extract:
        mode_id = env.end_regions[env_id].item()
        if mode_id not in (2, 3):
            print(f"[WARNING] env_id={env_id} is mode={mode_id}, NOT a high-ball mode!")
            mode_ok = False

    if not mode_ok:
        print("[ERROR] Some requested env_ids are NOT high-ball. Aborting.")
        print(f"[ERROR] Actual high-ball env_ids: {highball_env_ids_actual}")
        sys.exit(1)

    if not highball_env_ids_actual:
        print("[ERROR] No high-ball envs found at all. Check mode distribution.")
        sys.exit(1)

    # Use actual high-ball envs
    target_env_ids = env_ids_to_extract

    # ── 6. Zero root velocity on initial reset (like deterministic_demo_reset) ──
    print("\n[INIT] Zeroing root velocity for all envs...")
    env.root_states[:, 7:13] = 0.0
    all_states = torch.cat(
        (env.root_states.unsqueeze(1), env.ball_states.unsqueeze(1)), dim=1
    ).view(-1, 13)
    env_ids_int32 = torch.arange(2 * args.num_envs, dtype=torch.int32, device=env.device)
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(all_states),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )

    obs = env.get_observations()

    # ── 7. Get contact feet indices for foot contact detection ──
    foot_contact_names_in_cfg = []
    for name in body_names:
        if env.cfg.asset.contact_foot_names in name:
            foot_contact_names_in_cfg.append(name)

    contact_feet_idx_list = []
    for fname in foot_contact_names_in_cfg:
        idx = env.gym.find_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], fname)
        contact_feet_idx_list.append(idx)

    print(f"[FEET] Contact feet: {foot_contact_names_in_cfg} -> indices {contact_feet_idx_list}")

    # ── 8. Main rollout loop ──
    print("\n" + "=" * 60)
    print(f"[ROLLOUT] Starting {args.max_steps} steps for env_ids={target_env_ids}")
    print("=" * 60)

    # Per-env episode buffers
    episode_buffers = {eid: defaultdict(list) for eid in target_env_ids}
    episode_counter = {eid: 0 for eid in target_env_ids}
    step_counter = 0

    # Track previous reset buf to detect episode boundaries
    prev_reset = {eid: True for eid in target_env_ids}  # Start as "just reset"

    raw_dir = os.path.join(args.out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Run for max_steps
    for step in range(args.max_steps):
        # Pre-step: compute actions
        with torch.inference_mode():
            raw_actions = policy(obs.detach())

        # Pre-step: capture pre-step state
        pre_obs = obs.detach().clone()
        pre_raw_actions = raw_actions.detach().clone()
        pre_dof_pos = env.dof_pos.detach().clone()
        pre_dof_vel = env.dof_vel.detach().clone()
        pre_root_states = env.root_states.detach().clone()
        pre_ball_states = env.ball_states.detach().clone()

        # Step environment
        obs, privileged_obs, rews, dones, infos, termination_ids, termination_priv_obs = env.step(raw_actions.detach())

        # Post-step: capture all states
        post_actions = env.actions.detach().clone()  # clipped/executed actions
        post_root_states = env.root_states.detach().clone()
        post_ball_states = env.ball_states.detach().clone()
        post_dof_pos = env.dof_pos.detach().clone()
        post_dof_vel = env.dof_vel.detach().clone()
        post_projected_gravity = env.projected_gravity.detach().clone()
        post_base_ang_vel = env.base_ang_vel.detach().clone() if hasattr(env, 'base_ang_vel') else torch.zeros_like(env.root_states[:, 10:13])
        post_rigid_body_pos = env.rigid_body_states[:, :, 0:3].detach().clone()
        post_rigid_body_quat = env.rigid_body_states[:, :, 3:7].detach().clone()
        post_contact_forces = env.contact_forces.detach().clone()
        post_reset_buf = env.reset_buf.detach().clone()
        post_episode_length = env.episode_length_buf.detach().clone()

        # Reward terms if accessible
        rew_terms = {}
        if hasattr(env, 'episode_sums'):
            for key in env.episode_sums:
                rew_terms[key] = env.episode_sums[key].detach().clone()

        # For each target env, buffer the step data
        for eid in target_env_ids:
            buf = episode_buffers[eid]

            # Always record this step for the current episode
            buf['step'].append(step)
            buf['time'].append(step * env.dt)
            buf['obs'].append(pre_obs[eid].cpu().numpy())
            buf['raw_actions'].append(pre_raw_actions[eid].cpu().numpy())
            buf['executed_actions'].append(post_actions[eid].cpu().numpy())
            buf['root_state'].append(pre_root_states[eid].cpu().numpy())
            buf['root_pos'].append(pre_root_states[eid, :3].cpu().numpy())
            buf['root_quat'].append(pre_root_states[eid, 3:7].cpu().numpy())
            buf['root_lin_vel'].append(pre_root_states[eid, 7:10].cpu().numpy())
            buf['root_ang_vel'].append(pre_root_states[eid, 10:13].cpu().numpy())
            buf['ball_state'].append(pre_ball_states[eid].cpu().numpy())
            buf['ball_pos'].append(pre_ball_states[eid, :3].cpu().numpy())
            buf['ball_vel'].append(pre_ball_states[eid, 7:10].cpu().numpy())
            buf['dof_pos'].append(pre_dof_pos[eid].cpu().numpy())
            buf['dof_vel'].append(pre_dof_vel[eid].cpu().numpy())
            buf['projected_gravity'].append(post_projected_gravity[eid].cpu().numpy())
            buf['base_ang_vel'].append(post_base_ang_vel[eid].cpu().numpy())
            buf['rigid_body_pos'].append(post_rigid_body_pos[eid].cpu().numpy())
            buf['rigid_body_quat'].append(post_rigid_body_quat[eid].cpu().numpy())
            buf['contact_forces'].append(post_contact_forces[eid].cpu().numpy())
            buf['reset_flag'].append(int(post_reset_buf[eid].item()))
            buf['episode_length'].append(int(post_episode_length[eid].item()))

            # Check if episode just ended (reset was triggered)
            just_reset = bool(post_reset_buf[eid].item() and not prev_reset.get(eid, True))
            prev_reset[eid] = bool(post_reset_buf[eid].item())

            if just_reset:
                # Save completed episode and reset buffer
                ep_id = episode_counter[eid]
                _save_episode_npz(buf, raw_dir, eid, ep_id, body_names, env.dof_names)
                episode_counter[eid] += 1
                # Clear buffer for next episode
                episode_buffers[eid] = defaultdict(list)

        step_counter += 1

    # Save any remaining buffered data as partial episodes
    for eid in target_env_ids:
        buf = episode_buffers[eid]
        if len(buf.get('step', [])) > 0:
            ep_id = episode_counter[eid]
            _save_episode_npz(buf, raw_dir, eid, ep_id, body_names, env.dof_names)
            episode_counter[eid] += 1

    print(f"\n[ROLLOUT] Complete. Saved {sum(episode_counter.values())} episodes across {len(target_env_ids)} envs.")

    # ── 9. Analyze episodes ──
    print("\n" + "=" * 60)
    print("[ANALYZE] Analyzing episodes for jump classification...")
    print("=" * 60)

    analysis_results = []
    for eid in target_env_ids:
        raw_dir_e = raw_dir
        for ep_id in range(episode_counter[eid]):
            npz_path = os.path.join(raw_dir_e, f"g1_highball_env{eid}_ep{ep_id:03d}.npz")
            if os.path.exists(npz_path):
                result = analyze_episode(npz_path, candidate_body_map, foot_contact_names_in_cfg, contact_feet_idx_list)
                result['env_id'] = eid
                result['episode_id'] = ep_id
                analysis_results.append(result)

    if not analysis_results:
        print("[ERROR] No episodes to analyze!")
        # Still print summary of what we know
        _print_final_report([], args)
        return

    # ── 10. Generate summary CSV ──
    summary_path = os.path.join(args.out_dir, "summary.csv")
    fieldnames = [
        "env_id", "episode_id", "category",
        "root_z_init", "root_z_max", "root_z_delta", "root_vz_max",
        "upright_min", "both_feet_airborne_duration", "both_feet_airborne_steps",
        "contact_recovered_after_airborne",
        "nearest_body_name", "nearest_body_min_distance",
        "ball_speed_before", "ball_speed_after", "ball_speed_reduction",
        "root_pitch_max", "root_pitch_min",
        "fall_or_reset", "num_steps",
        "block_time", "ball_z_at_block", "root_z_at_block",
        "foot_contact_left_at_block", "foot_contact_right_at_block",
        "recommendation",
    ]

    with open(summary_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in analysis_results:
            row = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row)
    print(f"[SAVE] summary.csv ({len(analysis_results)} episodes)")

    # ── 11. Generate processed priors for good episodes ──
    processed_dir = os.path.join(args.out_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    for r in analysis_results:
        if r['recommendation'] in ('use_as_jump_prior', 'use_as_high_reach_prior'):
            _save_processed_prior(r, processed_dir)

    # ── 12. Final report ──
    _print_final_report(analysis_results, args)


# ─────────────────────────────────────────────────────────────
# Episode saving
# ─────────────────────────────────────────────────────────────
def _save_episode_npz(buf, raw_dir, env_id, ep_id, body_names, dof_names):
    """Convert buffered per-step lists to numpy arrays and save as npz."""
    if len(buf['step']) == 0:
        return

    data = {}
    for key in buf:
        if key in ('step', 'time', 'episode_length', 'reset_flag'):
            data[key] = np.array(buf[key])
        elif key in ('body_names', 'dof_names'):
            continue
        else:
            data[key] = np.stack(buf[key], axis=0)

    # Add metadata
    data['body_names'] = np.array(body_names)
    data['dof_names'] = np.array(dof_names)

    fname = f"g1_highball_env{env_id}_ep{ep_id:03d}.npz"
    fpath = os.path.join(raw_dir, fname)
    np.savez_compressed(fpath, **data)
    file_size_kb = os.path.getsize(fpath) / 1024
    print(f"  [SAVE] {fname} ({len(buf['step'])} steps, {file_size_kb:.1f} KB)")


# ─────────────────────────────────────────────────────────────
# Episode analysis
# ─────────────────────────────────────────────────────────────
def _to_native(val):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def analyze_episode(npz_path, candidate_body_map, foot_contact_names, contact_feet_indices):
    """Analyze a single episode npz and classify jump type."""
    data = np.load(npz_path, allow_pickle=True)

    root_pos = data['root_pos']  # (T, 3)
    root_quat = data['root_quat']  # (T, 4)
    projected_gravity = data['projected_gravity']  # (T, 3)
    contact_forces = data['contact_forces']  # (T, num_bodies, 3)
    ball_pos = data['ball_pos']  # (T, 3)
    ball_vel = data['ball_vel']  # (T, 3)
    rigid_body_pos = data['rigid_body_pos']  # (T, num_bodies, 3)
    reset_flags = data.get('reset_flag', np.zeros(len(root_pos)))
    body_names = data.get('body_names', np.array([]))

    T = len(root_pos)
    result = {'num_steps': T}

    # Recover dt from saved time array
    saved_time = data.get('time', None)
    if saved_time is not None and len(saved_time) >= 2:
        dt = float(saved_time[1] - saved_time[0])
    else:
        dt = 0.02  # fallback: 50Hz
    result['dt'] = dt

    # --- Root Z analysis ---
    root_z = root_pos[:, 2]
    result['root_z_init'] = float(root_z[0])
    result['root_z_max'] = float(np.max(root_z))
    result['root_z_delta'] = float(np.max(root_z) - root_z[0])
    result['root_vz_max'] = float(np.max(np.abs(data['root_lin_vel'][:, 2])))

    # --- Upright ---
    # upright = -projected_gravity_z  (1.0 = fully upright)
    upright = -projected_gravity[:, 2]
    result['upright_min'] = float(np.min(upright))

    # --- Root pitch ---
    # Compute pitch from quaternion
    from isaacgym.torch_utils import quat_rotate_inverse
    pitches = []
    for t in range(T):
        q = root_quat[t]
        w, x, y, z = q[0], q[1], q[2], q[3]
        # pitch = asin(2*(w*y - z*x))
        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)
        pitches.append(pitch)
    pitches = np.array(pitches)
    result['root_pitch_max'] = float(np.max(pitches))
    result['root_pitch_min'] = float(np.min(pitches))

    # --- Foot contact analysis ---
    foot_contact_left = np.zeros(T, dtype=bool)
    foot_contact_right = np.zeros(T, dtype=bool)
    if len(contact_feet_indices) >= 2:
        # Contact force Z > 1N threshold
        foot_contact_left = contact_forces[:, contact_feet_indices[0], 2] > 1.0
        foot_contact_right = contact_forces[:, contact_feet_indices[1], 2] > 1.0
    elif len(contact_feet_indices) == 1:
        foot_contact_left = contact_forces[:, contact_feet_indices[0], 2] > 1.0

    both_feet_contact = foot_contact_left & foot_contact_right
    both_feet_airborne = ~foot_contact_left & ~foot_contact_right

    # Find airborne segments
    airborne_steps = int(np.sum(both_feet_airborne))
    result['both_feet_airborne_steps'] = airborne_steps

    # Estimate airborne duration: count consecutive frames, use max segment
    airborne_segments = []
    if airborne_steps > 0:
        in_segment = False
        seg_start = 0
        for t in range(T):
            if both_feet_airborne[t] and not in_segment:
                in_segment = True
                seg_start = t
            elif not both_feet_airborne[t] and in_segment:
                in_segment = False
                airborne_segments.append((seg_start, t - 1))
        if in_segment:
            airborne_segments.append((seg_start, T - 1))

        # Max segment duration in seconds (dt = 0.02 typical for decimation=4)
        max_frames = max((e - s + 1) for s, e in airborne_segments)
        max_duration = max_frames * dt
        result['both_feet_airborne_duration'] = max_duration
        result['both_feet_airborne_max_segment_frames'] = max_frames
    else:
        result['both_feet_airborne_duration'] = 0.0
        result['both_feet_airborne_max_segment_frames'] = 0

    # Contact recovery
    if airborne_steps > 0:
        # Check if both feet regain contact after the last airborne segment
        last_airborne_end = max((e for _, e in airborne_segments)) if airborne_segments else T - 1
        if last_airborne_end < T - 1:
            has_contact_after = np.any(both_feet_contact[last_airborne_end:])
            result['contact_recovered_after_airborne'] = has_contact_after
        else:
            result['contact_recovered_after_airborne'] = False
    else:
        result['contact_recovered_after_airborne'] = True  # never airborne

    # --- Fall / reset detection ---
    # A "fall" is when reset_buf triggers NOT due to timeout but due to gravity or knee
    # We consider fall if reset was triggered and upright_min < 0.7 or both feet lost for long
    has_early_reset = np.any(reset_flags[:-1]) if len(reset_flags) > 1 else False
    is_fall = has_early_reset or (result['upright_min'] < 0.5) or (result['both_feet_airborne_duration'] > 0.5 and not result['contact_recovered_after_airborne'])
    result['fall_or_reset'] = is_fall

    # --- Ball speed analysis ---
    ball_speed = np.linalg.norm(ball_vel, axis=1)
    # Ball speed in first 10% vs last 10% of trajectory
    n_early = max(1, T // 10)
    n_late = max(1, T // 10)
    ball_speed_early = float(np.mean(ball_speed[:n_early]))
    ball_speed_late = float(np.mean(ball_speed[-n_late:]))
    result['ball_speed_before'] = ball_speed_early
    result['ball_speed_after'] = ball_speed_late
    result['ball_speed_reduction'] = ball_speed_early - ball_speed_late

    # --- Nearest body to ball ---
    # Compute min distance from each candidate body to the ball over the trajectory
    nearest_body_name = "unknown"
    nearest_body_min_dist = float('inf')
    nearest_body_idx = -1
    block_time = -1

    # For each candidate body, compute min distance to ball
    for body_name, body_idx in sorted(candidate_body_map.items()):
        if body_idx >= rigid_body_pos.shape[1]:
            continue
        body_positions = rigid_body_pos[:, body_idx, :]  # (T, 3)
        dists = np.linalg.norm(body_positions - ball_pos, axis=1)  # (T,)
        min_dist = float(np.min(dists))
        if min_dist < nearest_body_min_dist:
            nearest_body_min_dist = min_dist
            nearest_body_name = body_name
            nearest_body_idx = body_idx
            block_time = float(np.argmin(dists) * dt)

    result['nearest_body_name'] = nearest_body_name
    result['nearest_body_min_distance'] = nearest_body_min_dist
    result['block_idx_candidate'] = nearest_body_idx
    result['block_time'] = block_time

    # At block time, record relevant states
    block_step = int(np.argmin(np.linalg.norm(
        rigid_body_pos[:, nearest_body_idx, :] - ball_pos, axis=1
    ))) if nearest_body_idx >= 0 else 0
    result['ball_z_at_block'] = float(ball_pos[block_step, 2])
    result['root_z_at_block'] = float(root_pos[block_step, 2])
    if len(contact_feet_indices) >= 2:
        result['foot_contact_left_at_block'] = bool(foot_contact_left[block_step])
        result['foot_contact_right_at_block'] = bool(foot_contact_right[block_step])
    else:
        result['foot_contact_left_at_block'] = False
        result['foot_contact_right_at_block'] = False

    # --- Classification ---
    category, recommendation = classify_episode(result)
    result['category'] = category
    result['recommendation'] = recommendation

    # Convert all numpy types to native Python for downstream JSON/CSV
    native_result = {}
    for k, v in result.items():
        native_result[k] = _to_native(v)
    return native_result


def classify_episode(r):
    """Classify episode as jump_like, stand_reach, or bad_backlean."""
    root_z_delta = r['root_z_delta']
    root_vz_max = r['root_vz_max']
    airborne_dur = r['both_feet_airborne_duration']
    recovered = r['contact_recovered_after_airborne']
    upright_min = r['upright_min']
    is_fall = r['fall_or_reset']

    # Jump-like criteria
    is_jump_like = (
        root_z_delta > 0.05
        and root_vz_max > 0.2
        and airborne_dur > 0.04
        and recovered
        and not is_fall
    )

    if is_jump_like:
        return "jump_like", "use_as_jump_prior"

    # Bad backlean / fall
    if is_fall or upright_min < 0.5:
        return "bad_backlean", "discard"

    # Stand reach: not jumping, not falling, but got close to ball
    if r['nearest_body_min_distance'] < 0.5:
        return "stand_reach", "use_as_high_reach_prior"

    # Other
    return "other", "discard"


# ─────────────────────────────────────────────────────────────
# Processed prior saving
# ─────────────────────────────────────────────────────────────
def _save_processed_prior(result, processed_dir):
    """Save low-dimensional skill structure for usable episodes."""
    # Create a simple JSON summary of the key skill parameters
    prior = {}
    for k, v in {
        'category': result['category'],
        'recommendation': result['recommendation'],
        'env_id': result.get('env_id', -1),
        'episode_id': result.get('episode_id', -1),
        'root_z_delta': result['root_z_delta'],
        'root_vz_max': result['root_vz_max'],
        'upright_min': result['upright_min'],
        'both_feet_airborne_duration': result['both_feet_airborne_duration'],
        'contact_recovered_after_airborne': result['contact_recovered_after_airborne'],
        'nearest_body_name': result['nearest_body_name'],
        'nearest_body_min_distance': result['nearest_body_min_distance'],
        'ball_z_at_block': result.get('ball_z_at_block', 0),
        'root_z_at_block': result.get('root_z_at_block', 0),
        'foot_contact_left_at_block': result.get('foot_contact_left_at_block', False),
        'foot_contact_right_at_block': result.get('foot_contact_right_at_block', False),
        'block_time': result.get('block_time', -1),
        'root_pitch_max': result['root_pitch_max'],
    }.items():
        prior[k] = _to_native(v)

    fname = f"prior_env{result.get('env_id', -1)}_ep{result.get('episode_id', -1):03d}.json"
    fpath = os.path.join(processed_dir, fname)
    with open(fpath, 'w') as f:
        json.dump(prior, f, indent=2)
    print(f"  [PRIOR] {fname}")


# ─────────────────────────────────────────────────────────────
# Final report
# ─────────────────────────────────────────────────────────────
def _print_final_report(analysis_results, args):
    print("\n" + "=" * 60)
    print("FINAL REPORT: G1 High-Ball Rollout Extraction")
    print("=" * 60)

    # 1. Checkpoint actually loaded
    print(f"\n1. ACTUAL CHECKPOINT LOADED:")
    print(f"   {args.checkpoint_path}")
    print(f"   Verified: {os.path.isfile(args.checkpoint_path)}")
    print(f"   Size: {os.path.getsize(args.checkpoint_path) / 1024 / 1024:.1f} MB")

    # 2. High-ball env verification
    print(f"\n2. HIGH-BALL ENV VERIFICATION:")
    print(f"   Requested env_ids: {list(args.highball_env_ids)}")
    print(f"   (Verification printed above during env setup)")

    # 3. Episode count
    print(f"\n3. EPISODES COLLECTED:")
    print(f"   Total: {len(analysis_results)} episodes")

    # 4. Category breakdown
    categories = defaultdict(int)
    for r in analysis_results:
        categories[r['category']] += 1
    print(f"\n4. CLASSIFICATION BREAKDOWN:")
    for cat in ['jump_like', 'stand_reach', 'bad_backlean', 'other']:
        count = categories.get(cat, 0)
        print(f"   {cat}: {count}")

    # 5. Jump metrics
    jump_eps = [r for r in analysis_results if r['category'] == 'jump_like']
    stand_eps = [r for r in analysis_results if r['category'] == 'stand_reach']
    all_usable = jump_eps + stand_eps

    print(f"\n5. G1 HIGH-BALL BEHAVIOR ANALYSIS:")
    if all_usable:
        avg_z_delta = np.mean([r['root_z_delta'] for r in all_usable])
        avg_vz_max = np.mean([r['root_vz_max'] for r in all_usable])
        avg_airborne = np.mean([r['both_feet_airborne_duration'] for r in all_usable])
        print(f"   avg root_z_delta (all usable): {avg_z_delta:.4f} m")
        print(f"   avg root_vz_max (all usable): {avg_vz_max:.4f} m/s")
        print(f"   avg both_feet_airborne_duration: {avg_airborne:.4f} s")

    if jump_eps:
        print(f"\n   JUMP-LIKE episodes ({len(jump_eps)}):")
        print(f"   avg root_z_delta: {np.mean([r['root_z_delta'] for r in jump_eps]):.4f} m")
        print(f"   avg root_vz_max: {np.mean([r['root_vz_max'] for r in jump_eps]):.4f} m/s")
        print(f"   avg airborne_dur: {np.mean([r['both_feet_airborne_duration'] for r in jump_eps]):.4f} s")
        print(f"   contact recovery rate: {np.mean([1.0 if r['contact_recovered_after_airborne'] else 0.0 for r in jump_eps]):.2%}")

    if stand_eps:
        print(f"\n   STAND-REACH episodes ({len(stand_eps)}):")
        print(f"   avg root_z_delta: {np.mean([r['root_z_delta'] for r in stand_eps]):.4f} m")
        print(f"   avg root_vz_max: {np.mean([r['root_vz_max'] for r in stand_eps]):.4f} m/s")

    # 6. Blocking body analysis
    print(f"\n6. PRIMARY BLOCKING BODY:")
    body_counts = defaultdict(int)
    for r in analysis_results:
        body_counts[r['nearest_body_name']] += 1
    for body_name, count in sorted(body_counts.items(), key=lambda x: -x[1]):
        print(f"   {body_name}: {count} episodes")

    # 7. Conclusion
    print(f"\n7. CONCLUSION:")
    if len(jump_eps) >= len(stand_eps) and len(jump_eps) > 0:
        print(f"   G1 high-ball IS jump-like ({len(jump_eps)}/{len(all_usable)} episodes).")
        print(f"   → Extract jump-block prior for Q1: root_z_rel + root_vz + blocking body trajectory + foot contact phase.")
    elif len(stand_eps) > len(jump_eps):
        print(f"   G1 high-ball is NOT primarily jump-like. It is stand_reach ({len(stand_eps)}/{len(all_usable)} episodes).")
        print(f"   → Q1 should NOT learn a jump prior. Extract high-reach/block prior instead.")
    elif len(jump_eps) == 0 and len(stand_eps) == 0:
        print(f"   No usable episodes found. Check data quality.")
    else:
        print(f"   Mixed behavior. Review individual episodes.")

    # 8. Specific blocking body
    if body_counts:
        top_body = max(body_counts, key=body_counts.get)
        print(f"\n8. G1 MAIN BLOCKING BODY: {top_body}")
        print(f"   This suggests Q1 prior should focus on {top_body} trajectory, not just hand.")

    # 9. Recommended prior
    print(f"\n9. RECOMMENDED Q1 LOW-DIM PRIOR:")
    print(f"   - root_z_rel (relative to init)")
    print(f"   - root_vz")
    print(f"   - blocking_body_pos_rel_to_pelvis")
    print(f"   - foot_contact_phase (left/right)")
    print(f"   - upright trajectory (-projected_gravity_z)")
    print(f"   - takeoff_idx / block_idx / landing_idx if jump-like")
    print(f"   NOT: G1 dof_pos → Q1 retargeting")

    # 10. Output locations
    print(f"\n10. OUTPUT FILES:")
    print(f"    Raw rollouts: {os.path.join(args.out_dir, 'raw/')}")
    print(f"    Processed priors: {os.path.join(args.out_dir, 'processed/')}")
    print(f"    Summary CSV: {os.path.join(args.out_dir, 'summary.csv')}")
    print(f"    Body names: {os.path.join(args.out_dir, 'body_names.txt')}")
    print(f"    DOF names: {os.path.join(args.out_dir, 'dof_names.txt')}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract G1 high-ball rollouts for Q1 prior analysis")
    parser.add_argument("--task", type=str, default="29", help="Task name (default: 29 for G1)")
    parser.add_argument("--exptid", type=str, default="gk_resume_8000ep", help="Experiment ID")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Explicit path to checkpoint .pt file")
    parser.add_argument("--num_envs", type=int, default=6, help="Number of environments")
    parser.add_argument("--highball_env_ids", type=int, nargs="+", default=[2, 3],
                        help="Env IDs to extract high-ball data from")
    parser.add_argument("--max_steps", type=int, default=3000, help="Maximum rollout steps")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless")
    parser.add_argument("--no-headless", action="store_true", help="Run with viewer")

    args = parser.parse_args()

    if args.no_headless:
        args.headless = False

    extract(args)
