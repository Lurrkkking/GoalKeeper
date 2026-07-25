"""Export a Q1/G1 ready-stand checkpoint to ONNX.

Example:
    python tools/export_ready_stand_onnx.py --task q1_ready_stand \
        --checkpoint legged_gym/logs/q1_ready_stand/<run>/model_*.pt
"""

import argparse
import copy
import os
import sys

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "rsl_rl"))

from rsl_rl.modules.actor_critic import ActorCritic


class PolicyOnnx(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.history_encoder = copy.deepcopy(actor_critic.history_encoder)
        self.ball_estimator = copy.deepcopy(actor_critic.ball_estimator)
        self.region_estimator = copy.deepcopy(actor_critic.region_estimator)
        self.num_one_step_obs = actor_critic.num_one_step_obs

    def forward(self, obs):
        history_latent = self.history_encoder(obs)
        estimate_ball = self.ball_estimator(obs)
        estimate_region = torch.argmax(self.region_estimator(obs), dim=-1, keepdim=True)
        actor_input = torch.cat((obs[:, -self.num_one_step_obs:], history_latent, estimate_ball, estimate_region), dim=-1)
        return self.actor(actor_input)


TASK_SPECS = {
    "q1_ready_stand": {"actor_obs": 720, "critic_obs": 72, "single_obs": 72, "history": 10, "actions": 22},
    "g1_ready_stand": {"actor_obs": 930, "critic_obs": 93, "single_obs": 93, "history": 10, "actions": 29},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASK_SPECS, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--opset", type=int, default=11)
    args = parser.parse_args()

    spec = TASK_SPECS[args.task]
    checkpoint = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(_PROJECT_ROOT, args.checkpoint)
    if args.output is None:
        output = os.path.join(_PROJECT_ROOT, "legged_gym", "logs", args.task, "exported", "policies", "ready_stand.onnx")
    else:
        output = args.output if os.path.isabs(args.output) else os.path.join(_PROJECT_ROOT, args.output)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)

    ckpt = torch.load(checkpoint, map_location="cpu")
    actor_critic = ActorCritic(
        num_actor_obs=spec["actor_obs"], num_critic_obs=spec["critic_obs"],
        num_one_step_obs=spec["single_obs"], actor_history_length=spec["history"],
        num_actions=spec["actions"], actor_hidden_dims=[512, 256, 256],
        critic_hidden_dims=[512, 256, 256], activation="elu", init_noise_std=1.0,
    )
    missing, unexpected = actor_critic.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint architecture mismatch; missing={missing}, unexpected={unexpected}")
    policy = PolicyOnnx(actor_critic).eval()
    dummy = torch.zeros(1, spec["actor_obs"], dtype=torch.float32)
    with torch.no_grad():
        assert policy(dummy).shape == (1, spec["actions"])

    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.onnx.export(policy, dummy, output, export_params=True, opset_version=args.opset,
                      do_constant_folding=True, input_names=["obs"], output_names=["actions"],
                      dynamic_axes={"obs": {0: "batch_size"}, "actions": {0: "batch_size"}})
    print(f"Exported {args.task} ready-stand policy: {output}")


if __name__ == "__main__":
    main()
