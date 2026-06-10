"""Gym high-ball evaluation: per-height-bin statistics.
Usage:
  conda run -n rl python tools/eval_high_ball.py \
      --checkpoint legged_gym/logs/q1/stand_urdf_5_014_contact_dr_v2/model_14000.pt \
      --episodes 200
"""
import os, sys, argparse, json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'legged_gym'))
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import task_registry
import onnxruntime as ort


def make_args():
    args = type('Args', (), {})()
    args.task = 'q1_contact'
    args.headless = True
    args.rl_device = 'cuda:0'; args.sim_device = 'cuda:0'
    args.num_envs = 128  # fewer envs for eval
    args.resume = False; args.load_run = -1; args.checkpoint = -1
    args.resumeid = None; args.run_name = None; args.experiment_name = None
    args.seed = None; args.max_iterations = None; args.horovod = False
    args.compute_device_id = 0; args.sim_device_id = 0
    args.pipeline = 'gpu'; args.graphics_device_id = 0
    args.physics_engine = isaacgym.gymapi.SIM_PHYSX
    args.use_gpu = True; args.use_gpu_pipeline = True
    args.subscenes = 0; args.num_threads = 2; args.device = 'cuda'
    return args


def make_env(onnx_path):
    args = make_args()
    env_cfg, _ = task_registry.get_cfgs(name='q1_contact')
    env_cfg.env.play = True  # evaluation mode: narrower ball ranges
    # Disable all DR
    dr = env_cfg.domain_rand
    for attr in ['randomize_friction','randomize_kp','randomize_kd','randomize_motor_strength',
                 'randomize_payload_mass','randomize_torque_noise','randomize_action_delay',
                 'randomize_action_filter','randomize_ball_obs_noise','randomize_ball_obs_dropout',
                 'randomize_ball_obs_delay','randomize_joint_damping','randomize_armature_scale',
                 'randomize_ground_slope','randomize_ground_height','push_robots',
                 'randomize_restitution','randomize_initial_joint_pos','randomize_reset_velocity',
                 'randomize_contact_friction']:
        if hasattr(dr, attr): setattr(dr, attr, False)
    dr.push_interval = 9999999; dr.ball_interval = 9999999
    if hasattr(dr, 'push_at_contact_loss'): dr.push_at_contact_loss = False
    if hasattr(dr, 'push_during_ball'): dr.push_during_ball = False
    env, _ = task_registry.make_env(name='q1_contact', args=args, env_cfg=env_cfg)
    env.Kp_factors[:] = 1.0; env.Kd_factors[:] = 1.0; env.motor_strength[:] = 1.0
    session = ort.InferenceSession(onnx_path)
    return env, session


def run_episode(env, session):
    """Run one episode, return per-step records."""
    device = env.device
    obs, _ = env.reset()
    records = []
    hpL_idx = env.dof_names.index("left_hip_pitch_joint")
    hpR_idx = env.dof_names.index("right_hip_pitch_joint")
    apL_idx = env.dof_names.index("left_ankle_pitch_joint")
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for step in range(150):  # max 3s at 50Hz
        obs_np = obs.cpu().numpy().astype(np.float32)
        action = session.run(None, {"obs": obs_np})[0]

        # Record before step
        ball_z = env.ball_states[:, 2].cpu().numpy()
        pitch = env.pitch.cpu().numpy()
        dof_pos = env.dof_pos.cpu().numpy()
        ncon = (torch.norm(env.contact_forces[:, env.contact_feet_indices, :], dim=-1) > 1.0).sum(dim=1).cpu().numpy()
        root_z = env.root_states[:, 2].cpu().numpy()
        foot_force = torch.norm(env.contact_forces[:, env.contact_feet_indices, :], dim=-1).sum(dim=1).cpu().numpy()

        for e in range(env.num_envs):
            if not done[e]:
                records.append({
                    "env": e, "step": step, "ball_z": float(ball_z[e]),
                    "pitch": float(pitch[e]), "hpL": float(dof_pos[e, hpL_idx]),
                    "hpR": float(dof_pos[e, hpR_idx]), "apL": float(dof_pos[e, apL_idx]),
                    "ncon": int(ncon[e]), "root_z": float(root_z[e]),
                    "foot_force": float(foot_force[e]),
                    "a_hpL": float(action[e, hpL_idx]),
                    "a_hpR": float(action[e, hpR_idx]),
                    "a_max": float(np.max(np.abs(action[e]))),
                })

        obs, _, rew, done, _, _, _ = env.step(torch.tensor(action, dtype=torch.float, device=device))

        if done.all():
            break

    # Per-env episode summary
    summaries = []
    for e in range(env.num_envs):
        env_recs = [r for r in records if r["env"] == e]
        if not env_recs:
            continue
        ball_zs = [r["ball_z"] for r in env_recs]
        pitches = [r["pitch"] for r in env_recs]
        ncons = [r["ncon"] for r in env_recs]
        hpLs = [r["hpL"] for r in env_recs]
        summaries.append({
            "env": e,
            "ball_z_max": max(ball_zs),
            "ball_z_mean": np.mean(ball_zs),
            "pitch_max": max(pitches),
            "pitch_min": min(pitches),
            "pitch_final": pitches[-1],
            "ncon_min": min(ncons),
            "ncon_final": ncons[-1],
            "hpL_max": max(hpLs),
            "a_hpL_max": max(r["a_hpL"] for r in env_recs),
            "a_max_max": max(r["a_max"] for r in env_recs),
            "survived": len(env_recs),
            "fell": len(env_recs) < 150,
            "root_z_final": env_recs[-1]["root_z"],
            "fall_reason": "root_height" if env_recs[-1]["root_z"] < 0.25 else ("timeout" if len(env_recs)>=150 else "other"),
        })
    return summaries


def bin_by_height(summaries):
    """bin by ball_z_max: low <0.5, mid 0.5-0.8, high 0.8-1.1, very_high >1.1"""
    bins = {"low": [], "mid": [], "high": [], "very_high": []}
    for s in summaries:
        z = s["ball_z_max"]
        if z < 0.5:
            bins["low"].append(s)
        elif z < 0.8:
            bins["mid"].append(s)
        elif z < 1.1:
            bins["high"].append(s)
        else:
            bins["very_high"].append(s)
    return bins


def print_bin_stats(bin_name, items):
    if not items:
        print(f"  {bin_name}: 0 episodes")
        return
    n = len(items)
    survived = sum(1 for s in items if not s["fell"])
    pitch_maxs = [s["pitch_max"] for s in items]
    pitch_mins = [s["pitch_min"] for s in items]
    ncon_mins = [s["ncon_min"] for s in items]
    hpL_maxs = [s["hpL_max"] for s in items]
    a_hpL = [s["a_hpL_max"] for s in items]
    ball_z = [s["ball_z_max"] for s in items]
    print(f"  {bin_name}: n={n} survived={survived}({survived/n*100:.0f}%) "
          f"ball_z=[{min(ball_z):.2f},{max(ball_z):.2f}] "
          f"pitch=[{np.mean(pitch_mins):.2f},{np.mean(pitch_maxs):.2f}] "
          f"ncon_min={np.mean(ncon_mins):.1f} hpL_max={np.mean(hpL_maxs):.2f} "
          f"a_hpL={np.mean(a_hpL):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args()

    print("Creating env...", flush=True)
    env, session = make_env(args.onnx)
    print(f"  num_envs={env.num_envs}", flush=True)

    all_summaries = []
    episodes_needed = args.episodes
    rounds = 0
    while len(all_summaries) < episodes_needed and rounds < 10:
        rounds += 1
        summaries = run_episode(env, session)
        all_summaries.extend(summaries)
        print(f"  round {rounds}: collected {len(summaries)} episodes, total={len(all_summaries)}", flush=True)

    all_summaries = all_summaries[:episodes_needed]

    bins = bin_by_height(all_summaries)
    print("\n===== Per Height Bin =====")
    for bin_name in ["low", "mid", "high", "very_high"]:
        print_bin_stats(bin_name, bins[bin_name])

    print(f"\n===== Overall ({len(all_summaries)} eps) =====")
    fell = sum(1 for s in all_summaries if s["fell"])
    print(f"  fall rate: {fell}/{len(all_summaries)} ({fell/len(all_summaries)*100:.1f}%)")
    for reason in ["root_height", "timeout", "other"]:
        cnt = sum(1 for s in all_summaries if s["fall_reason"] == reason)
        print(f"    {reason}: {cnt}")


if __name__ == "__main__":
    main()
