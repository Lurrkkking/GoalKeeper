"""Task 1: ONNX vs PyTorch action consistency test.

Loads the PyTorch checkpoint and ONNX model, runs both on the same random
obs batches, and compares outputs. Also tests edge cases (zeros, extreme values).
"""

import os, sys
import numpy as np
import torch
import onnxruntime as ort

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "rsl_rl"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "legged_gym"))

from rsl_rl.modules.actor_critic import ActorCritic


class PolicyOnnx(torch.nn.Module):
    """Mirrors helpers.py PolicyOnnx exactly."""
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = actor_critic.actor
        self.history_encoder = actor_critic.history_encoder
        self.ball_estimator = actor_critic.ball_estimator
        self.region_estimator = actor_critic.region_estimator
        self.actor_history_length = actor_critic.actor_history_length
        self.num_one_step_obs = actor_critic.num_one_step_obs

    def forward(self, x):
        history_latent = self.history_encoder(x)
        estimate_ball = self.ball_estimator(x)
        estimate_region = self.region_estimator(x)
        estimate_region = torch.argmax(estimate_region, dim=-1, keepdim=True)
        actor_input = torch.cat(
            (x[:, -self.num_one_step_obs:], history_latent, estimate_ball, estimate_region),
            dim=-1,
        )
        return self.actor(actor_input)


def main():
    ckpt_path = os.path.join(_PROJECT_ROOT, "legged_gym/logs/q1/stand_urdf_5_014/model_9500.pt")
    onnx_path = os.path.join(_PROJECT_ROOT, "legged_gym/logs/q1/exported/policies/goalkeeper.onnx")

    # --- Load PyTorch model ---
    print("=== Loading PyTorch model ===")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]

    actor_critic = ActorCritic(
        num_actor_obs=750, num_critic_obs=92, num_one_step_obs=75,
        actor_history_length=10, num_actions=22,
        actor_hidden_dims=[512, 256, 256], critic_hidden_dims=[512, 256, 256],
        activation="elu", init_noise_std=1.0,
    )
    actor_critic.load_state_dict(state_dict, strict=False)
    actor_critic.eval()

    torch_policy = PolicyOnnx(actor_critic).to("cpu")
    torch_policy.eval()

    # --- Load ONNX model ---
    print("=== Loading ONNX model ===")
    session = ort.InferenceSession(onnx_path)
    inp_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    print(f"ONNX input: {inp_name} shape={session.get_inputs()[0].shape}")
    print(f"ONNX output: {out_name} shape={session.get_outputs()[0].shape}")

    def onnx_infer(x_np):
        return session.run([out_name], {inp_name: x_np.astype(np.float32)})[0]

    # --- Test 1: all-zeros ---
    print("\n=== Test 1: all-zeros obs ===")
    obs_np = np.zeros((1, 750), dtype=np.float32)
    obs_pt = torch.zeros(1, 750, dtype=torch.float32)

    with torch.no_grad():
        act_pt = torch_policy(obs_pt).numpy()
    act_onnx = onnx_infer(obs_np)

    diff = np.abs(act_pt - act_onnx)
    print(f"  PT shape={act_pt.shape}, ONNX shape={act_onnx.shape}")
    print(f"  mean_abs_diff={diff.mean():.8f}, max_abs_diff={diff.max():.8f}")
    if diff.max() > 1e-4:
        print(f"  *** FAIL: max_abs_diff {diff.max():.2e} > 1e-4 ***")
        sys.exit(1)
    print("  PASS")

    # --- Test 2: random normal ---
    print("\n=== Test 2: random normal obs (10 batches) ===")
    all_diffs = []
    for seed in range(10):
        rng = np.random.RandomState(seed)
        obs_np = rng.randn(8, 750).astype(np.float32) * 0.5  # ~training scale
        obs_pt = torch.from_numpy(obs_np)
        with torch.no_grad():
            act_pt = torch_policy(obs_pt).numpy()
        act_onnx = onnx_infer(obs_np)
        diff = np.abs(act_pt - act_onnx)
        all_diffs.append(diff.max())
        print(f"  batch {seed}: mean={diff.mean():.8f}, max={diff.max():.8f}")

    worst = max(all_diffs)
    print(f"  Overall worst max_diff={worst:.8f}")
    if worst > 1e-4:
        print(f"  *** FAIL: max_diff {worst:.2e} > 1e-4 ***")
        sys.exit(1)
    print("  PASS")

    # --- Test 3: clipped extreme values ---
    print("\n=== Test 3: clipped extreme obs (±100) ===")
    obs_np = np.full((1, 750), 100.0, dtype=np.float32)  # clip_observations
    obs_pt = torch.full((1, 750), 100.0, dtype=torch.float32)
    with torch.no_grad():
        act_pt = torch_policy(obs_pt).numpy()
    act_onnx = onnx_infer(obs_np)
    diff = np.abs(act_pt - act_onnx)
    print(f"  mean={diff.mean():.8f}, max={diff.max():.8f}")
    if diff.max() > 1e-4:
        print(f"  *** FAIL ***")
        sys.exit(1)
    print("  PASS")

    # --- Test 4: single-frame pattern ---
    print("\n=== Test 4: realistic single-frame pattern ===")
    # Simulate a typical observation: ball visible, slight ang_vel, etc.
    obs_np = np.zeros((1, 750), dtype=np.float32)
    for frame in range(10):
        start = frame * 75
        obs_np[0, start:start+3] = [2.0, 0.0, 0.5]      # ball base-frame pos
        obs_np[0, start+3:start+6] = [0.0, 0.0, 0.01]    # ang_vel scaled
        obs_np[0, start+6:start+9] = [0.0, 0.0, -1.0]    # gravity
        obs_np[0, start+9:start+31] = 0.01                # dof_pos offset
        obs_np[0, start+53:start+75] = 0.0                # last_action=0
    obs_pt = torch.from_numpy(obs_np)
    with torch.no_grad():
        act_pt = torch_policy(obs_pt).numpy()
    act_onnx = onnx_infer(obs_np)
    diff = np.abs(act_pt - act_onnx)
    print(f"  mean={diff.mean():.8f}, max={diff.max():.8f}")
    print(f"  PT   action mean={act_pt.mean():.4f}, max={act_pt.max():.4f}")
    print(f"  ONNX action mean={act_onnx.mean():.4f}, max={act_onnx.max():.4f}")
    if diff.max() > 1e-4:
        print(f"  *** FAIL ***")
        sys.exit(1)
    print("  PASS")

    print("\n=== ALL TESTS PASSED ===")
    print("ONNX and PyTorch produce identical actions (diff < 1e-4)")


if __name__ == "__main__":
    main()
