#!/usr/bin/env python3
"""
play_save_video.py — g1_dive_save专用可视化脚本。

复制自play_video.py，改动：
  - 不强制覆盖 episode_length_s（用cfg默认值）
  - 默认开启 record_video
  - 相机默认从侧面看球门区域
"""

import os, sys, subprocess, tempfile
import numpy as np
from pathlib import Path

import isaacgym
from isaacgym import gymtorch, gymapi, gymutil
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import (
    get_args, export_policy_as_jit, export_jit_to_onnx,
    load_onnx_policy, task_registry, get_load_path,
)
from legged_gym.envs.base.base_task import BaseTask

# ── monkey-patch: GPU rendering in headless ──
_original_base_init = BaseTask.__init__

def _patched_base_init(self, cfg, sim_params, physics_engine, sim_device, headless):
    self.gym = gymapi.acquire_gym()
    self.sim_params = sim_params
    self.physics_engine = physics_engine
    self.sim_device = sim_device
    sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
    self.device = self.sim_device if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline else 'cpu'
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
    self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float) if self.num_privileged_obs is not None else None
    self.extras = {}
    self.create_sim()
    self.gym.prepare_sim(self.sim)
    self.enable_viewer_sync = True
    self.viewer = None

BaseTask.__init__ = _patched_base_init


def setup_camera(env, env_handle, width=1280, height=720, cam_pos=None, cam_target=None, ref_env_id=0, env_id=None):
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
    env.gym.set_camera_location(cam_handle, env_handle, gymapi.Vec3(*cam_pos), gymapi.Vec3(*cam_target))
    return cam_handle, width, height


def capture_frame(env, cam_handle, cam_width, cam_height, camera_env_id=0):
    env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    env.gym.start_access_image_tensors(env.sim)
    img = env.gym.get_camera_image(env.sim, env.envs[camera_env_id], cam_handle, gymapi.IMAGE_COLOR)
    img = np.frombuffer(img, dtype=np.uint8).copy().reshape(cam_height, cam_width, -1)[..., :3]
    return img


def frames_to_mp4(frame_dir, video_path, fps=30):
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i",
           os.path.join(frame_dir, "frame_%06d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "23", video_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[INFO] Video saved to: {video_path}")


def play(args):
    print("[INFO] Starting play_save_video.py ...")

    assert args.task == "g1_dive_save", "This script is only for g1_dive_save"

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(getattr(args, "num_envs", 6) or 6, 6)
    # Do NOT force episode_length_s — use task config default
    env_cfg.env.play = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_initial_joint_pos = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.continue_keep = False
    env_cfg.domain_rand.randomize_reset_velocity = False
    print(f"[INFO] episode_length_s = {env_cfg.env.episode_length_s} (from cfg)")

    # Optional: override target_z range
    if getattr(args, "target_z_low", None) is not None:
        env_cfg.rewards.target_z_low = float(args.target_z_low)
        env_cfg.rewards.target_z_high = float(getattr(args, "target_z_high", 1.65))
        print(f"[INFO] target_z override: [{env_cfg.rewards.target_z_low}, {env_cfg.rewards.target_z_high}]")

    camera_env_id = getattr(args, "camera_env_id", 0)
    headless = getattr(args, "headless", True)

    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # Load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # Video recording (always on for this script)
    video_path = getattr(args, "video_path", None)
    video_length = getattr(args, "video_length", 300)
    if video_path is None:
        video_path = os.path.join(LEGGED_GYM_ROOT_DIR, "videos", f"dive_save_{args.exptid or 'play'}.mp4")

    print(f"[INFO] Recording: {video_length} frames → {video_path}")

    frame_dir = tempfile.mkdtemp(prefix="save_video_frames_")
    ref_env_id = camera_env_id
    cam_handle, cam_width, cam_height = setup_camera(env, env.envs[camera_env_id], env_id=camera_env_id)

    # Step once to init rendering
    zero_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    env.step(zero_actions)

    # Check target/bounds before loop
    for i in range(min(4, env.num_envs)):
        tp = env.target_pos[i] - env.env_origins[i]
        print(f"[INFO] env{i}: target(local)=({tp[0]:.2f},{tp[1]:.2f},{tp[2]:.2f}) would_score={env.would_score[i].item()}")

    frame_idx = 0
    for step_count in range(video_length):
        actions = policy(obs.detach())
        obs, _, _, dones, infos, _, _ = env.step(actions.detach())

        img = capture_frame(env, cam_handle, cam_width, cam_height, camera_env_id)
        from PIL import Image
        Image.fromarray(img).save(os.path.join(frame_dir, f"frame_{frame_idx:06d}.png"))
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"[PROGRESS] {frame_idx}/{video_length} frames")

        if dones[camera_env_id]:
            print(f"[INFO] camera env reset at step {step_count}")

    if frame_idx > 0:
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        frames_to_mp4(frame_dir, video_path, fps=50)
        print(f"[INFO] Total frames: {frame_idx}")
    else:
        print("[WARN] No frames captured!")


if __name__ == "__main__":
    # Grab video args before original get_args parser sees them
    video_path = None; video_length = 300; camera_env_id = 0
    target_z_low = None; target_z_high = None
    clean_argv = []
    i = 1
    while i < len(sys.argv):
        a = sys.argv[i]
        if a.startswith("--video_path="):
            video_path = a.split("=", 1)[1]
        elif a.startswith("--video_length="):
            video_length = int(a.split("=", 1)[1])
        elif a.startswith("--camera_env_id="):
            camera_env_id = int(a.split("=", 1)[1])
        elif a.startswith("--target_z_low="):
            target_z_low = float(a.split("=", 1)[1])
        elif a.startswith("--target_z_high="):
            target_z_high = float(a.split("=", 1)[1])
        elif a in ("--video_path", "--video_length", "--camera_env_id", "--target_z_low", "--target_z_high"):
            if i + 1 < len(sys.argv):
                val = sys.argv[i + 1]
                if a == "--video_path": video_path = val
                elif a == "--video_length": video_length = int(val)
                elif a == "--camera_env_id": camera_env_id = int(val)
                elif a == "--target_z_low": target_z_low = float(val)
                elif a == "--target_z_high": target_z_high = float(val)
                i += 1
        else:
            clean_argv.append(a)
        i += 1
    sys.argv = [sys.argv[0]] + clean_argv

    from legged_gym.utils.helpers import get_args as _orig_get_args
    args = _orig_get_args()
    args.task = "g1_dive_save"
    args.video_path = video_path
    args.video_length = video_length
    args.camera_env_id = camera_env_id
    args.target_z_low = target_z_low
    args.target_z_high = target_z_high
    play(args)
