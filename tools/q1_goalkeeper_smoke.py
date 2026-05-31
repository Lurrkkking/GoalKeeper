"""
Q1 Goalkeeper Smoke — zero-torque, zero-action, small-random, env check.
"""
import sys, os, numpy as np, csv
_saved = sys.path.copy()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from isaacgym import gymapi, gymtorch
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.q1.q1_goalkeeper_config import Q1GoalkeeperCfg


def run_case(label, action_mode, n_frames=300):
    cfg = Q1GoalkeeperCfg()
    cfg.env.num_envs = 1
    cfg.terrain.mesh_type = 'plane'
    cfg.domain_rand.randomize_initial_joint_pos = False
    cfg.domain_rand.push_robots = False
    cfg.asset.self_collisions = 0

    # Create sim params matching train.py
    sim_params = gymapi.SimParams()
    sim_params.dt = 1./200.
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0., 0., -9.81)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = 10
    sim_params.physx.use_gpu = True
    sim_params.use_gpu_pipeline = True

    env = LeggedRobot(cfg, sim_params, gymapi.SIM_PHYSX, "cuda:0", True)
    env.reset()
    print("\n%s:" % label)
    print("  init rz=%.3f knee=[%.3f,%.3f] dof_vel_max=%.3f" %
          (env.root_states[0,2].item(), env.dof_pos[0,3].item(), env.dof_pos[0,9].item(),
           env.dof_vel[0].abs().max().item()))

    yaw_cum = 0; last_yaw = None; tau_max = 0; tau_sat = 0; fell_frame = None

    for step in range(n_frames):
        if action_mode == "zero_torque":
            actions = torch.zeros(1, env.num_actions, device=env.device)
            # Bypass PD: directly apply zero torque
            env.torques[:] = 0.0
            env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.torques))
            for _ in range(env.cfg.control.decimation):
                env.gym.simulate(env.sim)
                if env.device == 'cpu': env.gym.fetch_results(env.sim, True)
            env.gym.refresh_dof_state_tensor(env.sim)
            env.gym.refresh_actor_root_state_tensor(env.sim)
            tau_np = np.zeros(env.num_dof)
        elif action_mode == "zero_action":
            actions = torch.zeros(1, env.num_actions, device=env.device)
            env.step(actions)
            tau_np = env.torques[0].cpu().numpy()
        elif action_mode == "random_small":
            actions = torch.randn(1, env.num_actions, device=env.device) * 0.05
            env.step(actions)
            tau_np = env.torques[0].cpu().numpy()

        tau_max = max(tau_max, abs(tau_np).max())
        tau_sat = max(tau_sat, (abs(tau_np) > 0.95 * env.torque_limits.cpu().numpy()).mean())

        rs = env.root_states[0].cpu().numpy()
        qx, qy, qz, qw = rs[3:7]
        yaw = np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
        if last_yaw is not None:
            dy = yaw - last_yaw
            if dy > np.pi: dy -= 2*np.pi
            elif dy < -np.pi: dy += 2*np.pi
            yaw_cum += dy
        last_yaw = yaw

        pgz = 1 - 2*(qx*qx + qy*qy)
        if fell_frame is None and pgz < 0.5: fell_frame = step

        if step < 5 or step % 50 == 0:
            print("  f%d: rz=%.2f pgz=%.3f tau=%.1f sat=%.0f%% yaw=%.1f" %
                  (step, rs[2], pgz, abs(tau_np).max(), tau_sat*100, np.degrees(yaw)))

    result = {
        "label": label, "yaw_deg": np.degrees(yaw_cum), "tau_max": tau_max,
        "tau_sat_pct": tau_sat*100, "fell_frame": fell_frame,
        "knee_final": (env.dof_pos[0,3].item(), env.dof_pos[0,9].item()),
    }
    print("  => yaw=%.1f deg tau_max=%.1f sat=%.0f%% fell=%s" %
          (result["yaw_deg"], result["tau_max"], result["tau_sat_pct"], fell_frame))
    return result


print("=" * 60)
print("Q1 GOALKEEPER SMOKE TESTS")
print("=" * 60)

results = []
results.append(run_case("A_zero_torque", "zero_torque"))
results.append(run_case("B_zero_action", "zero_action"))
results.append(run_case("C_random_small", "random_small"))

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for r in results:
    print("  %-20s yaw=%+7.1f deg tau=%.1f sat=%.0f%% fell=%s" %
          (r["label"], r["yaw_deg"], r["tau_max"], r["tau_sat_pct"], r["fell_frame"]))
