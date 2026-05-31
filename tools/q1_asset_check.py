"""Q1 Asset Check — verify URDF, config, body names, joint limits."""
import sys, os, numpy as np
_saved = sys.path.copy()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.q1.q1_goalkeeper_config import Q1GoalkeeperCfg
from isaacgym import gymapi, gymtorch
import torch

print("=" * 60)
print("Q1 ASSET CHECK")
print("=" * 60)

cfg = Q1GoalkeeperCfg()

# Load URDF
urdf_path = cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
print(f"URDF: {urdf_path}")
print(f"URDF exists: {os.path.exists(urdf_path)}")

gym = gymapi.acquire_gym()
sp = gymapi.SimParams()
sp.dt = 0.005; sp.up_axis = gymapi.UP_AXIS_Z
sp.gravity = gymapi.Vec3(0, 0, -9.81)
sp.physx.use_gpu = True; sp.use_gpu_pipeline = True
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)

opts = gymapi.AssetOptions()
opts.collapse_fixed_joints = cfg.asset.collapse_fixed_joints
opts.replace_cylinder_with_capsule = cfg.asset.replace_cylinder_with_capsule
opts.default_dof_drive_mode = cfg.asset.default_dof_drive_mode
opts.density = cfg.asset.density
opts.angular_damping = cfg.asset.angular_damping
opts.linear_damping = cfg.asset.linear_damping
opts.max_angular_velocity = cfg.asset.max_angular_velocity
opts.max_linear_velocity = cfg.asset.max_linear_velocity
opts.armature = cfg.asset.armature
opts.thickness = cfg.asset.thickness
opts.fix_base_link = cfg.asset.fix_base_link

asset = gym.load_asset(sim, os.path.dirname(urdf_path), os.path.basename(urdf_path), opts)
dof_names = gym.get_asset_dof_names(asset)
body_names = gym.get_asset_rigid_body_names(asset)
num_dof = gym.get_asset_dof_count(asset)
num_bodies = gym.get_asset_rigid_body_count(asset)
num_shapes = gym.get_asset_rigid_shape_count(asset)

print(f"\nDOF: {num_dof} (expected 22) {'PASS' if num_dof == 22 else 'FAIL'}")
print(f"Bodies: {num_bodies}")
print(f"Shapes: {num_shapes}")
print(f"\nDOF names:")
for i, n in enumerate(dof_names):
    print(f"  {i:2d}: {n}")

# Check config vs URDF
cfg_dof_names = list(cfg.init_state.default_joint_angles.keys())
dof_match = len(dof_names) == len(cfg_dof_names) and dof_names == cfg_dof_names
print(f"\nDOF names match config: {'PASS' if dof_match else 'FAIL'}")
if not dof_match:
    only_urdf = set(dof_names) - set(cfg_dof_names)
    only_cfg = set(cfg_dof_names) - set(dof_names)
    if only_urdf: print(f"  Only in URDF: {only_urdf}")
    if only_cfg: print(f"  Only in config: {only_cfg}")

# Body check
print(f"\nBody names:")
for i, n in enumerate(body_names):
    print(f"  {i:2d}: {n}")

# Check key bodies
env = gym.create_env(sim, gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 0), 1)
ah = gym.create_actor(env, asset, gymapi.Transform(p=gymapi.Vec3(0, 0, 0.39)), "q1", 0, -1, 0)

def check_body(name):
    try:
        idx = gym.find_actor_rigid_body_index(env, ah, name, gymapi.IndexDomain.DOMAIN_ENV)
        return idx, "PASS"
    except:
        return -1, "FAIL"

bodies_to_check = [
    ("pelvis", "pelvis"),
    ("torso_link", "torso_link"),
    ("left_ankle_roll_link", cfg.asset.foot_name),
    ("right_ankle_roll_link", cfg.asset.foot_name),
    ("left_knee_link", cfg.asset.knee_names[0]),
    ("right_knee_link", cfg.asset.knee_names[1]),
    ("left_elbow_link", cfg.asset.hand_name),
    ("right_elbow_link", cfg.asset.hand_name),
]

print(f"\nKey body indices:")
for name, desc in bodies_to_check:
    idx, status = check_body(name)
    print(f"  {name:30s} idx={idx:3d} {status}")

# Joint limits check
print(f"\nJoint limits:")
dof_props = gym.get_actor_dof_properties(env, ah)
for i in range(num_dof):
    lo = dof_props['lower'][i]; hi = dof_props['upper'][i]
    vel = dof_props['velocity'][i]; eff = dof_props['effort'][i]
    print(f"  {dof_names[i]:35s} lower={lo:+8.4f} upper={hi:+8.4f} vel={vel:6.1f} effort={eff:6.1f}")

# Config dimensions
print(f"\nConfig check:")
print(f"  num_actions={cfg.env.num_actions} (expected 22) {'PASS' if cfg.env.num_actions == 22 else 'FAIL'}")
print(f"  num_dofs={cfg.env.num_dofs} (expected 22) {'PASS' if cfg.env.num_dofs == 22 else 'FAIL'}")
print(f"  default_joint_angles count={len(cfg.init_state.default_joint_angles)} (expected 22) {'PASS' if len(cfg.init_state.default_joint_angles) == 22 else 'FAIL'}")
print(f"  init_pos count={len(cfg.init_state.init_pos)} (expected 22) {'PASS' if len(cfg.init_state.init_pos) == 22 else 'FAIL'}")
print(f"  knee default: {cfg.init_state.default_joint_angles.get('left_knee_joint','?'):.3f} (expected ~0.42)")

# Stiffness/damping check
print(f"\nStiffness keys: {list(cfg.control.stiffness.keys())}")
print(f"Damping keys: {list(cfg.control.damping.keys())}")
print(f"Action scale: {cfg.control.action_scale}")

gym.destroy_sim(sim)
print("\nDone.")
