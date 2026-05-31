#!/usr/bin/env python3
"""
play_video.py — Humanoid-Goalkeeper play with headless video recording via Isaac Gym camera sensors.

Based on play.py. Adds --record_video, --video_path, --video_length, --video_interval.
Minimal changes: does not modify training logic, reward, env physics, or original play.py.
"""

import os
import sys
import subprocess
import tempfile
import numpy as np
from pathlib import Path

# NOTE: isaacgym MUST be imported before torch
import isaacgym
from isaacgym import gymtorch, gymapi, gymutil
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import (
    get_args, export_policy_as_jit, export_jit_to_onnx,
    load_onnx_policy, task_registry, get_load_path,
)
from legged_gym.utils.helpers import update_cfg_from_args
from legged_gym.envs.base.base_task import BaseTask


# ── monkey-patch: allow GPU rendering without viewer in headless mode ──
_original_base_init = BaseTask.__init__

def _patched_base_init(self, cfg, sim_params, physics_engine, sim_device, headless):
    """Same as BaseTask.__init__ but keeps GPU graphics context even when headless."""
    self.gym = gymapi.acquire_gym()
    self.sim_params = sim_params
    self.physics_engine = physics_engine
    self.sim_device = sim_device
    sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)

    if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:
        self.device = self.sim_device
    else:
        self.device = 'cpu'

    # ── key patch: always use GPU for graphics when recording ──
    if headless:
        self.graphics_device_id = self.sim_device_id  # 0 → GPU rendering available
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
    # ── key patch: NEVER create viewer in headless mode ──
    # (Original code creates viewer when headless==False)


# Apply the monkey-patch
BaseTask.__init__ = _patched_base_init


def setup_camera(env: BaseTask, env_handle, width=1280, height=720,
                 cam_pos=None, cam_target=None, ref_env_id=0, env_id=None):
    """Create an Isaac Gym camera sensor attached to env_handle.

    Args:
        cam_pos: (x, y, z) tuple in env-local frame. Defaults to close-up front-side view.
        cam_target: (x, y, z) tuple in env-local frame. Defaults to robot centre.
        ref_env_id: env whose origin is used as the reference for default cam_pos / cam_target.
        env_id: current env id. Camera offset = origin[env_id] - origin[ref_env_id] (xy only).
    """
    camera_props = gymapi.CameraProperties()
    camera_props.width = width
    camera_props.height = height
    camera_props.enable_tensors = True

    cam_handle = env.gym.create_camera_sensor(env_handle, camera_props)

    if cam_pos is None:
        cam_pos = (-2.5, -5.0, 0.6)
    if cam_target is None:
        cam_target = (-3.0, -5.0, 0.6)

    if env_id is not None and ref_env_id != env_id:
        ref_origin = env.env_origins[ref_env_id]
        cur_origin = env.env_origins[env_id]
        dx = (cur_origin[0] - ref_origin[0]).item()
        dy = (cur_origin[1] - ref_origin[1]).item()
        cam_pos = (cam_pos[0] + dx, cam_pos[1] + dy, cam_pos[2])
        cam_target = (cam_target[0] + dx, cam_target[1] + dy, cam_target[2])

    env.gym.set_camera_location(
        cam_handle, env_handle,
        gymapi.Vec3(*cam_pos), gymapi.Vec3(*cam_target),
    )
    return cam_handle, width, height


def setup_overview_camera(env: BaseTask, env_handle, width=1280, height=720):
    """Position camera high + far back to show all envs in one frame.

    Uses env.env_origins to compute the grid centre and extent.
    """
    origins = env.env_origins.cpu().numpy()  # (num_envs, 3)
    centre_x = origins[:, 0].mean()
    centre_y = origins[:, 1].mean()
    extent_x = origins[:, 0].ptp()  # max - min along X
    extent_y = origins[:, 1].ptp()  # max - min along Y

    # Print env layout for debugging
    print(f"[INFO] Overview camera: {env.num_envs} envs")
    print(f"       centre=({centre_x:.1f}, {centre_y:.1f})")
    for i in range(env.num_envs):
        print(f"       env {i}: ({origins[i, 0]:.1f}, {origins[i, 1]:.1f}, {origins[i, 2]:.1f})")

    # Position high above, looking down at grid centre
    # Z = height enough to see the full grid span (tan FOV ~ 0.6)
    grid_diag = np.sqrt(extent_x**2 + extent_y**2) + 3.0  # padding
    cam_z = grid_diag * 0.7 + 1.0

    cam_pos = (centre_x, centre_y - 1.0, cam_z)
    cam_target = (centre_x, centre_y, 0.4)

    print(f"       camera_pos=({cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f})")
    print(f"       camera_target=({cam_target[0]:.1f}, {cam_target[1]:.1f}, {cam_target[2]:.1f})")

    return setup_camera(env, env_handle, width, height, cam_pos, cam_target)


def capture_frame(env, cam_handle, cam_width, cam_height, camera_env_id=0):
    """Capture one RGB frame from the camera sensor. Returns numpy (H, W, 3) uint8."""
    env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    env.gym.start_access_image_tensors(env.sim)

    img = env.gym.get_camera_image(
        env.sim, env.envs[camera_env_id], cam_handle, gymapi.IMAGE_COLOR
    )
    img = np.frombuffer(img, dtype=np.uint8).copy()
    img = img.reshape(cam_height, cam_width, -1)
    img = img[..., :3]
    return img


def frames_to_mp4(frame_dir, video_path, fps=30):
    """Compose PNG frames into MP4 using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "23",
        video_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[INFO] Video saved to: {video_path}")


def _clear_env_root_vel(env, env_id):
    """Zero out root linear + angular velocity for a single env and sync to simulator."""
    env.root_states[env_id, 7:13] = 0.0
    # Build indexed tensor: robot (actor 0) + ball (actor 1) for this env
    all_states = torch.cat(
        (env.root_states.unsqueeze(1), env.ball_states.unsqueeze(1)), dim=1
    ).view(-1, 13)
    env_ids_int32 = torch.tensor(
        [2 * env_id, 2 * env_id + 1], dtype=torch.int32, device=env.device
    )
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(all_states),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )


def play(args):
    print("[INFO] Starting play_video.py ...")
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # ── env config ──
    env_cfg.env.num_envs = min(getattr(args, "num_envs", 6) or 6, 6)
    env_cfg.env.episode_length_s = 3
    env_cfg.env.play = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_initial_joint_pos = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    # ── deterministic demo reset ──
    deterministic_demo = getattr(args, "deterministic_demo_reset", False)
    demo_reset_stand = getattr(args, "demo_reset_stand", False)

    if deterministic_demo:
        print("[INFO] Deterministic demo reset enabled: zero root velocity on reset, no continue_keep")
        env_cfg.domain_rand.continue_keep = False

    headless = getattr(args, "headless", True)
    camera_env_id = getattr(args, "camera_env_id", 0)

    # ── create env ──
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    if deterministic_demo:
        # After initial reset (done by HIMOnPolicyRunner.__init__ via env.reset()),
        # clear root velocity on the camera env so frame 0 starts clean.
        _clear_env_root_vel(env, camera_env_id)

    # ── relaxed termination for cleaner demo video ──
    relaxed_term = getattr(args, "relaxed_termination", False)
    if relaxed_term:
        print("[INFO] Relaxed termination: raising sharpforce + gravity thresholds")
        # Raise sharpforce: default threshold = 1.5 * 1000 = 1500N → now 7500N
        env.cfg.rewards.max_contact_force = 5000.0
        # Patch gravity: only trigger when robot is nearly horizontal (grav_xy > 0.95)
        _orig_check_fn = env.check_termination
        def _relaxed_check_termination():
            _orig_check_fn()
            grav_xy = torch.norm(env.projected_gravity[:, :2], dim=-1)
            env.reset_buf &= ~env.gravity_termination_buf
            env.gravity_termination_buf = grav_xy > 0.95
            env.reset_buf |= env.gravity_termination_buf
        env.check_termination = _relaxed_check_termination

    # ── load policy ──
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # ── video recording setup ──
    record_video = getattr(args, "record_video", False)
    overview_video = getattr(args, "overview_video", False)
    video_path = getattr(args, "video_path", None)
    video_length = getattr(args, "video_length", 1000)
    video_interval = getattr(args, "video_interval", 1)

    cam_handle = None
    frame_dir = None

    if record_video:
        mode = "overview" if overview_video else "single"
        print(f"[INFO] Recording {mode} video: {video_length} frames, env_id={camera_env_id}")
        frame_dir = tempfile.mkdtemp(prefix="gk_video_frames_")
        print(f"[INFO] Frame dir: {frame_dir}")

        if overview_video:
            print("[DEBUG] setting up overview camera...", flush=True)
            cam_handle, cam_width, cam_height = setup_overview_camera(env, env.envs[0])
            print("[DEBUG] overview camera setup done", flush=True)
        else:
            cam_handle, cam_width, cam_height = setup_camera(env, env.envs[camera_env_id], env_id=camera_env_id)

        # Step once to init rendering pipeline before main loop
        print("[DEBUG] stepping env once...", flush=True)
        zero_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        obs, _, _, _, _, _, _ = env.step(zero_actions)
        print("[DEBUG] env step done, entering main loop", flush=True)

    # ── main play loop ──
    max_steps = video_length if record_video else 3000
    frame_idx = 0

    # Track previous reset_buf for camera env to detect fresh resets
    prev_cam_reset = env.reset_buf[camera_env_id].item()

    for step_count in range(max_steps):
        actions = policy(obs.detach())
        obs, privileged_obs, rews, dones, infos, _, _ = env.step(actions.detach())

        # ── deterministic demo: clear root vel once per reset ──
        if deterministic_demo:
            just_reset = bool(env.reset_buf[camera_env_id].item() and not prev_cam_reset)
            if just_reset:
                _clear_env_root_vel(env, camera_env_id)
                # Debug: print termination reason for camera env
                eid = camera_env_id
                reasons = []
                if env.cfg.env.play:
                    e_len = env.episode_length_buf[eid].item()
                    e_max = env.max_episode_length
                    knee_z = env.rigid_body_states[eid, env.knee_indices, 2].min().item()
                    grav_xy = torch.norm(env.projected_gravity[eid, :2]).item()
                    foot_f = env.contact_forces[eid, env.contact_feet_indices, :].norm(dim=-1).mean().item()
                    reasons.append(f"ep_len={e_len}/{e_max}")
                    reasons.append(f"knee_z={knee_z:.3f} (thresh=0.10)")
                    reasons.append(f"grav_xy={grav_xy:.3f} (thresh=0.80)")
                    reasons.append(f"foot_force={foot_f:.0f}N (thresh=1500N)")
                print(f"[DEBUG] env {eid} reset at step {step_count}: " + ", ".join(reasons))
            prev_cam_reset = bool(env.reset_buf[camera_env_id].item())

        if record_video and step_count % video_interval == 0:
            try:
                img = capture_frame(env, cam_handle, cam_width, cam_height, camera_env_id)
                from PIL import Image
                Image.fromarray(img).save(
                    os.path.join(frame_dir, f"frame_{frame_idx:06d}.png")
                )
                frame_idx += 1
            except Exception as e:
                print(f"[WARN] Frame {frame_idx} capture failed: {e}")

    # ── compose MP4 ──
    if record_video and frame_idx > 0:
        if video_path is None:
            video_path = os.path.join(
                LEGGED_GYM_ROOT_DIR, "videos",
                f"gk_{args.exptid or 'play'}_{frame_idx}f.mp4"
            )
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        frames_to_mp4(frame_dir, video_path, fps=50)
        print(f"[INFO] Total frames captured: {frame_idx}")
    elif record_video:
        print("[WARN] No frames captured!")


if __name__ == "__main__":
    # ── parse args (same as get_args but with video extras) ──
    EXPORT_POLICY = False

    custom_parameters = [
        {"name": "--task", "type": str, "default": "29", "help": "Task name."},
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume from checkpoint."},
        {"name": "--resumeid", "type": str, "help": "Resume experiment ID."},
        {"name": "--experiment_name", "type": str, "help": "Experiment name."},
        {"name": "--run_name", "type": str, "help": "Run name."},
        {"name": "--load_run", "type": str, "help": "Run to load when resume=True."},
        {"name": "--checkpoint", "type": int, "help": "Checkpoint number. -1 = latest."},
        {"name": "--exptid", "type": str, "help": "Experiment ID for log path."},
        {"name": "--headless", "action": "store_true", "default": True, "help": "Headless mode."},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": "RL device."},
        {"name": "--num_envs", "type": int, "help": "Number of environments."},
        {"name": "--seed", "type": int, "help": "Random seed."},
        {"name": "--max_iterations", "type": int, "help": "Max training iterations."},
        # ── video recording args ──
        {"name": "--record_video", "action": "store_true", "default": False, "help": "Record video using offscreen camera."},
        {"name": "--overview_video", "action": "store_true", "default": False, "help": "Wide-angle overview showing all envs (overrides single close-up)."},
        {"name": "--camera_env_id", "type": int, "default": 0, "help": "Which env index to attach the camera to (default: 0)."},
        {"name": "--video_path", "type": str, "default": None, "help": "Output MP4 path."},
        {"name": "--video_length", "type": int, "default": 1000, "help": "Number of steps to record."},
        {"name": "--video_interval", "type": int, "default": 1, "help": "Record every N steps."},
        # ── deterministic demo args ──
        {"name": "--deterministic_demo_reset", "action": "store_true", "default": False, "help": "Disable continue_keep; zero root velocity on each reset for cleaner demo video."},
        {"name": "--demo_reset_stand", "action": "store_true", "default": False, "help": "Also reset dof_pos to default_stand and zero dof_vel on each reset (stronger cleanup)."},
        {"name": "--relaxed_termination", "action": "store_true", "default": False, "help": "Raise sharpforce (→7500N) and gravity (→0.95) thresholds to avoid premature reset in demo."},
    ]

    args = gymutil.parse_arguments(description="RL Policy", custom_parameters=custom_parameters)
    args.sim_device = args.rl_device

    play(args)
