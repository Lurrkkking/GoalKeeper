"""Export Q1 goalkeeper checkpoint to ONNX via PolicyOnnx wrapper.

Usage:
    python tools/export_q1_goalkeeper_onnx.py \
        --checkpoint legged_gym/logs/q1/stand_urdf_5_014/model_9500.pt \
        --output legged_gym/logs/q1/exported/policies/goalkeeper.onnx
"""

import os
import sys
import argparse
import torch
import copy

# Add project paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "rsl_rl"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "legged_gym"))

from rsl_rl.modules.actor_critic import ActorCritic


class PolicyOnnx(torch.nn.Module):
    """Wraps the ActorCritic for ONNX export, mirroring helpers.py"""

    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.history_encoder = copy.deepcopy(actor_critic.history_encoder)
        self.ball_estimator = copy.deepcopy(actor_critic.ball_estimator)
        self.region_estimator = copy.deepcopy(actor_critic.region_estimator)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="legged_gym/logs/q1/stand_urdf_5_014/model_9500.pt")
    parser.add_argument("--output", type=str,
                        default="legged_gym/logs/q1/exported/policies/goalkeeper.onnx")
    parser.add_argument("--opset", type=int, default=11)
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt_path = os.path.join(_PROJECT_ROOT, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    onnx_path = os.path.join(_PROJECT_ROOT, args.output) if not os.path.isabs(args.output) else args.output

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    iteration = ckpt.get("iter", "unknown")
    print(f"  Iteration: {iteration}")

    # Build ActorCritic with Q1 goalkeeper dimensions
    # num_actor_obs=750, num_critic_obs=92, num_one_step_obs=75, actor_history_length=10, num_actions=22
    actor_critic = ActorCritic(
        num_actor_obs=750,
        num_critic_obs=92,
        num_one_step_obs=75,
        actor_history_length=10,
        num_actions=22,
        actor_hidden_dims=[512, 256, 256],
        critic_hidden_dims=[512, 256, 256],
        activation="elu",
        init_noise_std=1.0,
    )

    # Load weights
    missing, unexpected = actor_critic.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    actor_critic.eval()

    # Wrap in PolicyOnnx
    policy_onnx = PolicyOnnx(actor_critic).to("cpu")
    policy_onnx.eval()

    # Verify forward pass
    dummy_input = torch.zeros(1, 750, dtype=torch.float32)
    with torch.no_grad():
        output = policy_onnx(dummy_input)
    print(f"Dummy inference: input {dummy_input.shape} -> output {output.shape}")
    assert output.shape == (1, 22), f"Expected (1, 22), got {output.shape}"
    print(f"  Action mean={output.mean().item():.4f}, max={output.max().item():.4f}")

    # Export to ONNX
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(
        policy_onnx,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch_size"}, "actions": {0: "batch_size"}},
    )
    print(f"Exported ONNX to: {onnx_path}")

    # Quick validation: load ONNX and run inference
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    print(f"ONNX input:  name={inp.name}, shape={inp.shape}")
    print(f"ONNX output: name={out.name}, shape={out.shape}")
    ort_out = session.run([out.name], {inp.name: dummy_input.numpy()})[0]
    print(f"ONNX inference: shape={ort_out.shape}, mean={ort_out.mean():.4f}")
    assert ort_out.shape == (1, 22)


if __name__ == "__main__":
    main()
