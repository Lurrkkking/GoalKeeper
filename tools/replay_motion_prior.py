#!/usr/bin/env python3
"""Kinematic replay of AMP motion prior .pt using IsaacGym tensor API.

Usage:
  conda activate rl
  python tools/replay_motion_prior.py \
    --pt resources/datasets/goalkeeper/right_dive_gmr.pt \
    --output debug_motion_replay/right_dive_gmr_replay.mp4
"""
import isaacgym
from isaacgym import gymapi, gymtorch
import torch, os, sys, argparse, numpy as np

def replay_motion(pt_path, output_video=None, loops=2):
    # ── Load motion ──
    data = torch.load(pt_path, map_location='cpu')
    base_pos = data['base_position'].numpy().astype(np.float32)
    base_quat_xyzw = data['base_pose'].numpy().astype(np.float32)
    joint_pos_21 = data['joint_position'].numpy().astype(np.float32)
    T = len(base_pos)
    fps = 30

    print(f'Motion: {os.path.basename(pt_path)}')
    print(f'  Frames: {T} @ {fps}fps')
    print(f'  Root Y: [{base_pos[:,1].min():.3f}, {base_pos[:,1].max():.3f}] p2p={base_pos[:,1].max()-base_pos[:,1].min():.3f}')
    print(f'  Root Z: [{base_pos[:,2].min():.3f}, {base_pos[:,2].max():.3f}]')
    delta_y = base_pos[-1,1] - base_pos[0,1]
    print(f'  Y delta: {delta_y:.3f}  → {"+Y (LEFT in IsaacGym)" if delta_y > 0 else "-Y (RIGHT in IsaacGym)"}')

    # ── IsaacGym setup ──
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.gravity = gymapi.Vec3(0, 0, -9.81)
    sim_params.up_axis = gymapi.UP_AXIS_Z

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # Load G1
    asset_root = '/root/autodl-tmp/Humanoid-Goalkeeper/legged_gym/resources/robots/g1/urdf'
    asset_file = 'g1_29.urdf'
    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = 3  # POSITION
    asset_options.collapse_fixed_joints = False
    asset_options.fix_base_link = False

    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    dof_names = gym.get_asset_dof_names(asset)
    num_dof = len(dof_names)
    body_names = gym.get_asset_rigid_body_names(asset)
    print(f'  G1: {num_dof} DOF, {len(body_names)} bodies')

    # Joint mapping: 21 → 29
    mapping_path = '/root/autodl-tmp/Humanoid-Goalkeeper/legged_gym/resources/datasets/goalkeeper/joint_id.txt'
    joint_map = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                joint_map[int(parts[0])] = parts[1]

    dof_full = np.zeros((T, num_dof), dtype=np.float32)
    for pt_col in range(21):
        name = joint_map.get(pt_col)
        if name and name in dof_names:
            full_idx = dof_names.index(name)
            dof_full[:, full_idx] = joint_pos_21[:, pt_col]

    # Create env
    env_spacing = 3.0
    env = gym.create_env(sim, gymapi.Vec3(-env_spacing, -env_spacing, 0.0),
                         gymapi.Vec3(env_spacing, env_spacing, env_spacing), 1)

    start_pose = gymapi.Transform()
    start_pose.p = gymapi.Vec3(base_pos[0, 0], base_pos[0, 1], base_pos[0, 2])
    q = base_quat_xyzw[0]
    start_pose.r = gymapi.Quat(q[0], q[1], q[2], q[3])  # x,y,z,w
    actor_handle = gym.create_actor(env, asset, start_pose, "g1", 0, 1)

    # Set PD gains
    dof_props = gym.get_actor_dof_properties(env, actor_handle)
    for i in range(num_dof):
        name = dof_names[i]
        if 'knee' in name:
            dof_props['stiffness'][i] = 300.0
        elif 'ankle' in name:
            dof_props['stiffness'][i] = 40.0
        elif 'wrist' in name:
            dof_props['stiffness'][i] = 20.0
        else:
            dof_props['stiffness'][i] = 150.0
        dof_props['damping'][i] = dof_props['stiffness'][i] * 0.01
    gym.set_actor_dof_properties(env, actor_handle, dof_props)

    # Camera
    cam_props = gymapi.CameraProperties()
    cam_props.width = 1920
    cam_props.height = 1080
    cam_handle = gym.create_camera_sensor(env, cam_props)
    # Side view to show lateral motion
    mid_y = (base_pos[:,1].min() + base_pos[:,1].max()) / 2
    gym.set_camera_location(cam_handle, env,
        gymapi.Vec3(base_pos[:,0].mean() + 2.0, mid_y, 1.2),   # eye
        gymapi.Vec3(base_pos[:,0].mean(), mid_y, 0.7))           # target

    # ── Prepare tensors ──
    # Root state tensor: (num_actors, 13) = [x,y,z, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz]
    root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(root_state_tensor)
    # DOF state tensor
    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    dof_states = gymtorch.wrap_tensor(dof_state_tensor)
    # DOF position targets
    dof_pos_targets = torch.zeros(1, num_dof, dtype=torch.float32)

    # Convert to torch tensors for assignment
    base_pos_t = torch.from_numpy(base_pos).float()
    base_quat_t = torch.from_numpy(base_quat_xyzw).float()
    dof_full_t = torch.from_numpy(dof_full).float()
    print('Starting replay...')

    # Video writer
    writer = None
    if output_video:
        import imageio
        writer = imageio.get_writer(output_video, fps=fps)

    for loop in range(loops):
        for t in range(T):
            # Set root state: position + quat (wxyz for Isaac)
            q_xyzw = base_quat_t[t]
            root_states[0, 0] = base_pos_t[t, 0]
            root_states[0, 1] = base_pos_t[t, 1]
            root_states[0, 2] = base_pos_t[t, 2]
            root_states[0, 3] = q_xyzw[3]  # w
            root_states[0, 4] = q_xyzw[0]  # x
            root_states[0, 5] = q_xyzw[1]  # y
            root_states[0, 6] = q_xyzw[2]  # z
            root_states[0, 7:13] = 0  # velocity
            gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))

            # Set DOF state: shape (num_dof, 2) — [pos, vel] for 1 env
            dof_states[:, 0] = dof_full_t[t]  # pos
            dof_states[:, 1] = 0  # vel
            gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_states))

            # Set PD targets (for rendering only, we use kinematic DOF)
            gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(
                dof_full_t[t].unsqueeze(0)))

            # Step graphics + simulate
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_dof_state_tensor(sim)
            gym.refresh_actor_root_state_tensor(sim)

            # Render to viewer
            gym.step_graphics(sim)
            gym.draw_viewer(gym.create_viewer(sim, gymapi.CameraProperties()), sim, True)

    if writer:
        writer.close()
        print(f'Video saved: {output_video}')

    # Summary
    print()
    print('=' * 50)
    print('DIRECTION CHECK')
    print('=' * 50)
    print(f'  Root Y start:  {base_pos[0,1]:.4f}')
    print(f'  Root Y end:    {base_pos[-1,1]:.4f}')
    print(f'  Root Y delta:  {base_pos[-1,1] - base_pos[0,1]:.4f}')
    print(f'  Root Y p2p:    {base_pos[:,1].max() - base_pos[:,1].min():.4f}')
    vel = (base_pos[1:] - base_pos[:-1]) / (1.0 / fps)
    print(f'  Vy peak:       {abs(vel[:,1]).max():.2f} m/s')
    print(f'  IsaacGym: +Y=LEFT, -Y=RIGHT (robot faces +X)')
    if delta_y < -0.3:
        print(f'  → Robot dives RIGHT (-Y) ✅ matches "right_dive" name')
    elif delta_y > 0.3:
        print(f'  → Robot dives LEFT (+Y) — opposite of expected right dive!')
    else:
        print(f'  → Ambiguous lateral motion')

    # FK for hand positions
    gym.refresh_rigid_body_state_tensor(sim)
    rb_tensor = gym.acquire_rigid_body_state_tensor(sim)
    rb_states = gymtorch.wrap_tensor(rb_tensor)
    # Find rubber hand indices
    for name in ['right_rubber_hand', 'left_rubber_hand']:
        if name in body_names:
            idx = body_names.index(name)
            y = rb_states[0, idx, 1].item()
            print(f'  {name} world Y: {y:.4f} (current frame)')

    gym.destroy_sim(sim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--loops', type=int, default=2)
    args = parser.parse_args()
    replay_motion(args.pt, args.output, args.loops)
