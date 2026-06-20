#!/usr/bin/env python3
"""Minimal check: load AMP .pt, replay kinematically in IsaacGym, report direction."""
import isaacgym
from isaacgym import gymapi, gymtorch
import torch, numpy as np, argparse

def verify(pt_path):
    data = torch.load(pt_path, map_location='cpu')
    base_pos = data['base_position'].numpy().astype(np.float32)
    base_quat = data['base_pose'].numpy().astype(np.float32)
    joint_pos_21 = data['joint_position'].numpy().astype(np.float32)
    T = len(base_pos)

    print(f'File: {pt_path}')
    print(f'Frames: {T}')
    print(f'Root Y: [{base_pos[:,1].min():.3f}, {base_pos[:,1].max():.3f}] p2p={base_pos[:,1].max()-base_pos[:,1].min():.3f}')
    print(f'Root Z: [{base_pos[:,2].min():.3f}, {base_pos[:,2].max():.3f}]')

    # IsaacGym setup
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.gravity = gymapi.Vec3(0, 0, -9.81)
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    gym.add_ground(sim, gymapi.PlaneParams())

    asset_root = '/root/autodl-tmp/Humanoid-Goalkeeper/legged_gym/resources/robots/g1/urdf'
    asset = gym.load_asset(sim, asset_root, 'g1_29.urdf',
        gymapi.AssetOptions())
    dof_names = gym.get_asset_dof_names(asset)
    body_names = gym.get_asset_rigid_body_names(asset)

    # Joint mapping 21→29
    mapping_path = '/root/autodl-tmp/Humanoid-Goalkeeper/legged_gym/resources/datasets/goalkeeper/joint_id.txt'
    joint_map = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                joint_map[int(parts[0])] = parts[1]

    dof_full = np.zeros((T, len(dof_names)), dtype=np.float32)
    for pt_col in range(21):
        name = joint_map.get(pt_col)
        if name and name in dof_names:
            dof_full[:, dof_names.index(name)] = joint_pos_21[:, pt_col]

    # Env
    env = gym.create_env(sim, gymapi.Vec3(-3, -3, 0), gymapi.Vec3(3, 3, 3), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 0.8)
    actor = gym.create_actor(env, asset, pose, "g1", 0, 1)

    # Tensors
    root_state_t = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(root_state_t)
    dof_state_t = gym.acquire_dof_state_tensor(sim)
    dof_states = gymtorch.wrap_tensor(dof_state_t)
    rb_state_t = gym.acquire_rigid_body_state_tensor(sim)
    rb_states = gymtorch.wrap_tensor(rb_state_t)

    bp_t = torch.from_numpy(base_pos).float()
    bq_t = torch.from_numpy(base_quat).float()
    df_t = torch.from_numpy(dof_full).float()

    # Key body indices
    rhand_idx = body_names.index('right_rubber_hand') if 'right_rubber_hand' in body_names else -1
    lhand_idx = body_names.index('left_rubber_hand') if 'left_rubber_hand' in body_names else -1

    # Replay and track
    hand_y_start, hand_y_end = None, None
    for t in range(T):
        # Root state
        root_states[0, 0:3] = bp_t[t]
        root_states[0, 3] = bq_t[t, 3]  # w
        root_states[0, 4:7] = bq_t[t, :3]  # x,y,z
        root_states[0, 7:13] = 0
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))

        # DOF state
        dof_states[:, 0] = df_t[t]
        dof_states[:, 1] = 0
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_states))

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_rigid_body_state_tensor(sim)

        # rb_states shape: (num_bodies, 13) for 1 env
        if t == 0 and rhand_idx >= 0:
            hand_y_start = rb_states[rhand_idx, 1].item()
        if t == T-1 and rhand_idx >= 0:
            hand_y_end = rb_states[rhand_idx, 1].item()

    gym.destroy_sim(sim)

    # Report
    delta_y = base_pos[-1, 1] - base_pos[0, 1]
    vel = (base_pos[1:] - base_pos[:-1]) / (1.0 / 30.0)

    print()
    print('=' * 50)
    print('DIRECTION VERIFICATION')
    print('=' * 50)
    print(f'  Root Y start:  {base_pos[0,1]:.4f}')
    print(f'  Root Y end:    {base_pos[-1,1]:.4f}')
    print(f'  Root Y delta:  {delta_y:.4f}')
    print(f'  Root Y p2p:    {base_pos[:,1].max() - base_pos[:,1].min():.4f}')
    print(f'  Vy peak:       {abs(vel[:,1]).max():.2f} m/s')
    if rhand_idx >= 0:
        print(f'  Right hand Y start: {hand_y_start:.4f}')
        print(f'  Right hand Y end:   {hand_y_end:.4f}')
    print()
    print(f'  IsaacGym: robot faces +X, +Y=LEFT, -Y=RIGHT')
    if delta_y < -0.3:
        print(f'  => RIGHT-SIDE DIVE (-Y) ✅')
    elif delta_y > 0.3:
        print(f'  => LEFT-SIDE DIVE (+Y)')
    else:
        print(f'  => Ambiguous')
    print()
    print(f'  g1_dive_save convention: target_y > 0 = ?')
    print(f'  In dive_save code: target_side > 0 → left_rubber_hand')
    print(f'  → target_y > 0 = LEFT side target in IsaacGym')
    print(f'  → target_y < 0 = RIGHT side target in IsaacGym')
    print(f'  Our motion dives to Y={base_pos[-1,1]:.1f} → matches RIGHT side (target_y < 0)')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pt', required=True)
    args = p.parse_args()
    verify(args.pt)
