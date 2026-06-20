#!/usr/bin/env python3
"""Convert GMR retargeted G1 motion pkl → Humanoid-Goalkeeper AMP .pt format.

Usage:
  python tools/convert_gmr_pkl_to_amp_pt.py \
    --pkl /root/autodl-tmp/GMR/unitree_g1_gmr/goalkeep_level1_right_dive.pkl \
    --output /root/autodl-tmp/Humanoid-Goalkeeper/legged_gym/resources/datasets/goalkeeper/right_dive_gmr.pt \
    --joint-mapping /root/autodl-tmp/Humanoid-Goalkeeper/legged_gym/resources/datasets/goalkeeper/joint_id.txt \
    --target-fps 30
"""
import argparse, sys, os
import numpy as np
import torch
import joblib

# MuJoCo FK for link data (optional, NOT needed if G1 URDF has no keyframe bodies)
_HAS_MUJOCO = False
try:
    sys.path.insert(0, '/root/autodl-tmp/GMR')
    from general_motion_retargeting.kinematics_model import KinematicsModel
    from general_motion_retargeting import ROBOT_XML_DICT
    _HAS_MUJOCO = True
except Exception:
    pass


def load_joint_mapping(mapping_path):
    """Parse joint_id.txt: 'index joint_name' per line -> dict {joint_name: index}."""
    mapping = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            idx = int(parts[0])
            name = parts[1]
            mapping[name] = idx
    print(f'Loaded joint mapping: {len(mapping)} joints')
    return mapping


def resample_linear(data, src_fps, tgt_fps):
    """Linear resample (T, *) along axis 0 from src_fps to tgt_fps."""
    T = data.shape[0]
    if src_fps == tgt_fps:
        return data.copy()
    ratio = tgt_fps / src_fps
    T_new = max(1, int(T * ratio))
    src_t = np.arange(T, dtype=np.float64)
    tgt_t = np.linspace(0, T - 1, T_new, dtype=np.float64)
    if data.ndim == 1:
        return np.interp(tgt_t, src_t, data).astype(data.dtype)
    # For multi-dim, do per-column
    result = np.zeros((T_new,) + data.shape[1:], dtype=data.dtype)
    for i in range(data.shape[1]):
        result[:, i] = np.interp(tgt_t, src_t, data[:, i])
    return result


def resample_quat(data, src_fps, tgt_fps):
    """SLERP resample quaternion array (T, 4) xyzw from src_fps to tgt_fps."""
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp
    T = data.shape[0]
    if src_fps == tgt_fps:
        return data.copy()
    ratio = tgt_fps / src_fps
    T_new = max(1, int(T * ratio))
    src_t = np.arange(T, dtype=np.float64)
    tgt_t = np.linspace(0, T - 1, T_new, dtype=np.float64)
    rots = R.from_quat(data)  # expects xyzw
    slerp = Slerp(src_t, rots)
    interp = slerp(tgt_t)
    return interp.as_quat()  # returns xyzw


def compute_velocities(positions, fps):
    """Central difference velocities from positions (T, D)."""
    T = positions.shape[0]
    vel = np.zeros_like(positions)
    if T < 2:
        return vel
    dt = 1.0 / fps
    # Central difference for interior
    if T >= 3:
        vel[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
    # Forward/backward for boundaries
    vel[0] = (positions[1] - positions[0]) / dt
    vel[-1] = (positions[-1] - positions[-2]) / dt
    return vel


def compute_angular_velocities(quats_xyzw, fps):
    """Compute angular velocity from quaternion sequence (T, 4) xyzw."""
    from scipy.spatial.transform import Rotation as R
    T = quats_xyzw.shape[0]
    ang_vel = np.zeros((T, 3), dtype=np.float32)
    if T < 2:
        return ang_vel
    dt = 1.0 / fps
    rots = R.from_quat(quats_xyzw)
    for i in range(1, T):
        # delta rotation: r_i = delta * r_{i-1}  => delta = r_i * inv(r_{i-1})
        delta = rots[i] * rots[i-1].inv()
        ang_vel[i] = delta.as_rotvec() / dt
    ang_vel[0] = ang_vel[1]  # forward fill for first frame
    return ang_vel


def convert(pkl_path, output_path, joint_mapping_path, target_fps, generate_links=False):
    # ── Load GMR pkl ──
    with open(pkl_path, 'rb') as f:
        gmr = joblib.load(f)

    src_fps = float(gmr['fps'])
    root_pos_src = np.asarray(gmr['root_pos'], dtype=np.float64)     # (T, 3)
    root_rot_src = np.asarray(gmr['root_rot'], dtype=np.float64)     # (T, 4) xyzw
    joint_names = list(gmr['joint_names'])
    dof_pos_src = np.asarray(gmr['dof_pos'], dtype=np.float64)       # (T, 29)
    joint_vel_src = np.asarray(gmr['joint_vel'], dtype=np.float32)   # (T, 29)

    T_src = root_pos_src.shape[0]
    print(f'Source: {T_src} frames @ {src_fps}fps, {dof_pos_src.shape[1]} DOF')

    # ── Validate ──
    assert dof_pos_src.shape[1] == 29, f'Expected 29 DOF, got {dof_pos_src.shape[1]}'
    assert root_rot_src.shape[1] == 4
    # Check quat norm
    norms = np.linalg.norm(root_rot_src, axis=1)
    assert np.allclose(norms, 1.0, atol=0.01), f'Quat norms not ≈1: {norms.min():.4f} ~ {norms.max():.4f}'
    assert not np.isnan(root_pos_src).any() and not np.isnan(dof_pos_src).any(), 'NaN in source!'
    assert root_rot_src.shape[1] == 4, f'root_rot is not quaternion shape: {root_rot_src.shape}'

    # ── Quat convention check ──
    # GMR stores xyzw (scipy standard). Verify: w should be the largest component for near-upright poses.
    avg_w_last = abs(root_rot_src[:, 3]).mean()  # xyzw → w at index 3
    avg_w_first = abs(root_rot_src[:, 0]).mean() # wxyz → w at index 0
    if avg_w_first > avg_w_last * 1.5:
        print('WARNING: root_rot appears to be wxyz, converting to xyzw')
        root_rot_src = root_rot_src[:, [1, 2, 3, 0]]
    print(f'Quat format confirmed: xyzw (avg |w| at idx3={avg_w_last:.3f}, idx0={avg_w_first:.3f})')

    # ── FPS resampling ──
    if abs(src_fps - target_fps) > 0.1:
        print(f'Resampling: {src_fps} → {target_fps} fps')
        root_pos = resample_linear(root_pos_src, src_fps, target_fps)
        root_rot = resample_quat(root_rot_src, src_fps, target_fps)
        dof_pos_29 = resample_linear(dof_pos_src, src_fps, target_fps)
        joint_vel_29 = resample_linear(joint_vel_src, src_fps, target_fps)
    else:
        root_pos = root_pos_src.copy()
        root_rot = root_rot_src.copy()
        dof_pos_29 = dof_pos_src.copy()
        joint_vel_29 = joint_vel_src.copy()

    T = root_pos.shape[0]
    print(f'After resample: {T} frames @ {target_fps}fps')

    # ── Joint mapping: 29 → 21 ──
    joint_map = load_joint_mapping(joint_mapping_path)
    # joint_map is {name: source_index} — we need: for each GMR joint name, what's its index in the 29-DOF array?
    # Actually joblib stored joint_names in order. The mapping.txt says "source_index joint_name".
    # In MotionLib: dof_pos[:, mapping[name]] — mapping[name] is the index in the SOURCE data (our 29-DOF).
    # But wait: joint_id.txt maps from the OLD 21-DOF source to robot DOF names.
    # For our case, we have 29 DOFs matching robot DOF names directly.
    # MotionLib code: for each robot DOF name, if name in mapping → use mapping[name] as INDEX into source data.
    # So our 29-DOF data needs to be reordered so that mapping[name] works correctly.
    #
    # Actually, let's just produce a 21-DOF .pt that matches the existing joint_id.txt.
    # We'll extract the 21 joints in the order specified by joint_id.txt values.
    # joint_id.txt: index→name. The INDEX is the column in the .pt's joint_position array.
    # So we need joint_position[:, k] = dof_pos_29[:, index_of(joint_id.txt[k])]

    # Build inverse: name -> index in 29-DOF GMR array
    gmr_dof_idx = {name: i for i, name in enumerate(joint_names)}
    n_pt_joints = len(joint_map)
    dof_pos = np.zeros((T, n_pt_joints), dtype=np.float32)
    dof_vel = np.zeros((T, n_pt_joints), dtype=np.float32)

    # joint_id.txt maps: old_source_index → name
    # In the .pt we produce, column k = dof for joint name at mapping[k]
    # Rebuild: pt_column -> gmr_index
    for name, pt_col in sorted(joint_map.items(), key=lambda x: x[1]):
        if name in gmr_dof_idx:
            gmr_col = gmr_dof_idx[name]
            dof_pos[:, pt_col] = dof_pos_29[:, gmr_col].astype(np.float32)
            dof_vel[:, pt_col] = joint_vel_29[:, gmr_col].astype(np.float32)
        else:
            print(f'WARNING: {name} not found in GMR DOF names, column {pt_col} will be zero')

    print(f'DOF mapping: 29 → {n_pt_joints}')
    total_mapped = sum(1 for name in joint_map if name in gmr_dof_idx)
    print(f'  Mapped: {total_mapped}/{n_pt_joints}')

    # ── Compute base velocities ──
    base_vel = compute_velocities(root_pos, target_fps).astype(np.float32)
    base_ang_vel = compute_angular_velocities(root_rot, target_fps).astype(np.float32)

    # ── Build output dict ──
    amp_data = {
        'base_position': torch.from_numpy(root_pos.astype(np.float32)),
        'base_pose': torch.from_numpy(root_rot.astype(np.float32)),
        'base_velocity': torch.from_numpy(base_vel),
        'base_angular_velocity': torch.from_numpy(base_ang_vel),
        'joint_position': torch.from_numpy(dof_pos),
        'joint_velocity': torch.from_numpy(dof_vel),
    }

    # ── Optional: link data (NOT needed — G1 URDF has no 'keyframe' bodies) ──
    if generate_links:
        if not _HAS_MUJOCO:
            print('WARNING: MuJoCo FK not available, skipping link data')
        else:
            print('Generating link positions via FK...')
            km = KinematicsModel(str(ROBOT_XML_DICT['unitree_g1']), device='cpu')
            body_names = km.body_names
            # root_rot is xyzw; FK expects wxyz
            rr_wxyz = torch.from_numpy(root_rot[:, [3, 0, 1, 2]].copy()).float()
            rp_t = torch.from_numpy(root_pos.copy()).float()
            # Need 29 DOF for FK, but we have 21 — pad with default dof
            dof_full = torch.zeros(T, 29)
            for pt_col, name in sorted(joint_map.items(), key=lambda x: x[1]):
                if name in gmr_dof_idx:
                    dof_full[:, gmr_dof_idx[name]] = torch.from_numpy(dof_pos_29[:, gmr_dof_idx[name]]).float()
            bp, br = km.forward_kinematics(rp_t, rr_wxyz, dof_full)
            link_pos = bp.numpy().astype(np.float32)  # (T, N_bodies, 3)
            link_orient = br.numpy().astype(np.float32)  # (T, N_bodies, 4) wxyz
            # Convert wxyz→xyzw for consistency
            link_orient_xyzw = link_orient[:, :, [1, 2, 3, 0]]
            link_vel = compute_velocities(link_pos.reshape(T, -1), target_fps).reshape(T, -1, 3).astype(np.float32)
            amp_data['link_position'] = torch.from_numpy(link_pos)
            amp_data['link_orientation'] = torch.from_numpy(link_orient_xyzw)
            amp_data['link_velocity'] = torch.from_numpy(link_vel)
            amp_data['link_angular_velocity'] = torch.zeros(T, len(body_names), 3)  # placeholder
            print(f'Link data: {len(body_names)} bodies')

    # ── Validate before saving ──
    print()
    print('=== Pre-save validation ===')
    for k, v in amp_data.items():
        is_finite = torch.isfinite(v).all().item()
        vmin = float(v.min()) if v.numel() > 0 else None
        vmax = float(v.max()) if v.numel() > 0 else None
        print(f'  {k:<25s} shape={str(tuple(v.shape)):<20s} dtype={str(v.dtype):<10s} finite={is_finite} range=[{vmin:.4f}, {vmax:.4f}]')
    print(f'  Root Y p2p: {root_pos[:,1].max() - root_pos[:,1].min():.4f}m')
    print(f'  Root Z range: [{root_pos[:,2].min():.4f}, {root_pos[:,2].max():.4f}]')
    print(f'  DOF pos range: [{dof_pos.min():.4f}, {dof_pos.max():.4f}]')
    print(f'  DOF vel range: [{dof_vel.min():.4f}, {dof_vel.max():.4f}]')

    # ── Save ──
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(amp_data, output_path)
    print(f'\nSaved: {output_path}')
    return amp_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--joint-mapping', required=True)
    parser.add_argument('--target-fps', type=float, default=30.0)
    parser.add_argument('--generate-links', action='store_true', default=False)
    args = parser.parse_args()
    convert(args.pkl, args.output, args.joint_mapping, args.target_fps, args.generate_links)
