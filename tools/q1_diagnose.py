"""
Q1 Goalkeeper Sanity Check — URDF / Joint / PD / Mapping / Zero-Action / Friction audit.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from isaacgym import gymapi, gymtorch
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.q1.q1_goalkeeper_config import Q1GoalkeeperCfg

cfg = Q1GoalkeeperCfg()
cfg.env.num_envs = 6
cfg.terrain.mesh_type = 'plane'
cfg.domain_rand.randomize_initial_joint_pos = False
cfg.domain_rand.push_robots = False
cfg.domain_rand.randomize_friction = False
cfg.asset.self_collisions = 0

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

print("=" * 70)
print("Q1 SANITY CHECK")
print("=" * 70)

# ===========================================================================
# 1. Config dump
# ===========================================================================
print("\n--- 1. CONFIG ---")
print(f"  URDF: {cfg.asset.file}")
print(f"  num_dof: {cfg.env.num_dofs}")
print(f"  num_actions: {cfg.env.num_actions}")
print(f"  pos (init_state): {cfg.init_state.pos}")
print(f"  capture_default_dof_pos_from_sim: {cfg.init_state.capture_default_dof_pos_from_sim}")
print(f"  action_scale: {cfg.control.action_scale}")
print(f"  init_noise_std: PPO runner default (check standalone cfg)")
print(f"  friction: static={cfg.terrain.static_friction}, dynamic={cfg.terrain.dynamic_friction}, restitution={cfg.terrain.restitution}")
print(f"  randomize_friction: {cfg.domain_rand.randomize_friction}")
if cfg.domain_rand.randomize_friction:
    print(f"  friction_range: {cfg.domain_rand.friction_range}")
print(f"  stiffness (per group): {cfg.control.stiffness}")
print(f"  damping (per group): {cfg.control.damping}")

print(f"\n  default_joint_angles ({len(cfg.init_state.default_joint_angles)} joints):")
for i, (name, angle) in enumerate(cfg.init_state.default_joint_angles.items()):
    print(f"    [{i:2d}] {name:35s} = {angle:+.4f}")

print(f"\n  init_pos vector ({len(cfg.init_state.init_pos)} dims):")
print(f"    {cfg.init_state.init_pos}")

print(f"\n  per_joint_action_scale overrides:")
for k, v in cfg.control.per_joint_action_scale.items():
    print(f"    {k}: {v}")

# ===========================================================================
# 2. URDF parse
# ===========================================================================
print("\n--- 2. URDF PARSE ---")
import xml.etree.ElementTree as ET
urdf_path = cfg.asset.file.replace('{LEGGED_GYM_ROOT_DIR}', LEGGED_GYM_ROOT_DIR)
tree = ET.parse(urdf_path)
root = tree.getroot()

print(f"  URDF path: {urdf_path}")
print(f"  Robot name: {root.attrib.get('name', 'N/A')}")

joints = []
for j in root.findall("joint"):
    name = j.attrib["name"]
    typ = j.attrib["type"]
    parent = j.find("parent").attrib["link"]
    child = j.find("child").attrib["link"]
    ax = j.find("axis")
    axis_str = ax.attrib["xyz"] if ax is not None else "N/A"
    lim = j.find("limit")
    if lim is not None and typ != "fixed":
        lo, hi = float(lim.attrib["lower"]), float(lim.attrib["upper"])
        effort = float(lim.attrib.get("effort", 0))
        velocity = float(lim.attrib.get("velocity", 0))
    else:
        lo, hi, effort, velocity = 0, 0, 0, 0
    joints.append({
        "name": name, "type": typ, "parent": parent, "child": child,
        "axis": axis_str, "lower": lo, "upper": hi,
        "effort": effort, "velocity": velocity,
    })

print(f"\n  Movable joints ({sum(1 for j in joints if j['type'] != 'fixed')}):")
for i, j in enumerate(joints):
    if j["type"] == "fixed":
        continue
    print(f"  [{i:2d}] {j['name']:35s} type={j['type']:10s} "
          f"axis=({j['axis']}) "
          f"limit=({j['lower']:+7.3f}, {j['upper']:+7.3f}) "
          f"effort={j['effort']} vel={j['velocity']}")

print(f"\n  Links ({len(root.findall('link'))}):")
total_mass = 0
foot_links = []
for link in root.findall("link"):
    name = link.attrib["name"]
    mass = 0
    inert = link.find("inertial")
    if inert is not None:
        m = inert.find("mass")
        if m is not None:
            mass = float(m.attrib.get("value", 0))
    total_mass += mass
    has_collision = len(link.findall("collision")) > 0
    coll_types = []
    for col in link.findall("collision"):
        geo = col.find("geometry")
        if geo is not None:
            for child in geo:
                coll_types.append(child.tag)
    mesh_ok = True
    for col in link.findall("collision"):
        geo = col.find("geometry")
        if geo is not None:
            mesh = geo.find("mesh")
            if mesh is not None:
                fp = mesh.attrib.get("filename", "")
                full = os.path.join(os.path.dirname(urdf_path), fp)
                if not os.path.exists(full):
                    mesh_ok = False
    if "ankle" in name.lower() or "foot" in name.lower():
        foot_links.append((name, mass, has_collision, coll_types, mesh_ok))
    print(f"  {name:35s} mass={mass:7.3f}  collision={has_collision}  "
          f"types={coll_types}  mesh_ok={mesh_ok}")

print(f"\n  Total mass: {total_mass:.3f} kg")

print(f"\n  Foot links:")
for n, m, hc, ct, mo in foot_links:
    print(f"    {n}: mass={m:.3f}  collision_types={ct}  mesh_ok={mo}")

# ===========================================================================
# 3. Sim test
# ===========================================================================
print("\n--- 3. SIM ENV CREATE ---")
env = LeggedRobot(cfg, sim_params, gymapi.SIM_PHYSX, "cuda:0", True)

# Print actual DOF names from sim
print(f"\n  Sim DOF names ({len(env.dof_names)}):")
for i, name in enumerate(env.dof_names):
    def_angle = cfg.init_state.default_joint_angles.get(name, "N/A")
    print(f"    [{i:2d}] {name:35s}  default={def_angle}")

# Check mapping: default_dof_pos to joint names
print(f"\n  default_dof_pos ({env.default_dof_pos.shape}):")
for i in range(env.num_dof):
    print(f"    [{i:2d}] {env.dof_names[i]:35s} = {env.default_dof_pos[0, i].item():+.4f}")

# Check PD params
print(f"\n  PD gains (per joint):")
print(f"  p_gains shape={env.p_gains.shape}  d_gains shape={env.d_gains.shape}  "
      f"torque_limits shape={env.torque_limits.shape}  action_scale_vec shape={env.action_scale_vec.shape}")
for i in range(env.num_dof):
    kp = env.p_gains.flatten()[i].item()
    kd = env.d_gains.flatten()[i].item()
    eff = env.torque_limits.flatten()[i].item()
    sc = env.action_scale_vec.flatten()[i].item()
    print(f"    [{i:2d}] {env.dof_names[i]:35s} Kp={kp:.0f}  Kd={kd:.1f}  effort={eff:.0f}  scale={sc:.4f}")

# Check foot body indices from env
print(f"\n  Foot body indices (contact_feet_indices): {env.contact_feet_indices}")
left_foot_idx = env.contact_feet_indices[0].item()
right_foot_idx = env.contact_feet_indices[1].item()
print(f"  Left foot body idx: {left_foot_idx}, Right: {right_foot_idx}")

# ===========================================================================
# 4. Zero-action rollout
# ===========================================================================
print("\n--- 4. ZERO-ACTION ROLLOUT (5s) ---")
env.reset()
print(f"  Initial root_state: pos={env.root_states[0, :3].cpu().numpy()} "
      f"ori={env.root_states[0, 3:7].cpu().numpy()}")

n_frames = 1000  # 5s at 200Hz
root_positions = []
root_yaws = []
root_lin_vels = []
root_ang_vels = []
base_rpy = []
foot_contact_forces = []
dof_positions = []
dof_velocities = []
print(f"  Using foot idx: L={left_foot_idx} R={right_foot_idx}")

for step in range(n_frames):
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    env.step(actions)

    rs = env.root_states[0].cpu().numpy()
    qx, qy, qz, qw = rs[3], rs[4], rs[5], rs[6]
    yaw = np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
    roll = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy))
    pitch = np.arcsin(np.clip(2*(qw*qy-qz*qx), -1, 1))

    root_positions.append(rs[:3].copy())
    root_yaws.append(yaw)
    root_lin_vels.append(rs[7:10].copy())
    root_ang_vels.append(rs[10:13].copy())
    base_rpy.append([roll, pitch, yaw])
    foot_contact_forces.append(env.contact_forces[0, [left_foot_idx, right_foot_idx]].cpu().numpy().copy())
    dof_positions.append(env.dof_pos[0].cpu().numpy().copy())
    dof_velocities.append(env.dof_vel[0].cpu().numpy().copy())

root_positions = np.array(root_positions)
root_lin_vels = np.array(root_lin_vels)
root_ang_vels = np.array(root_ang_vels)
base_rpy = np.array(base_rpy)
foot_contact_forces = np.array(foot_contact_forces)
dof_positions = np.array(dof_positions)
dof_velocities = np.array(dof_velocities)

print(f"\n  Root position drift (start -> end):")
print(f"    X: {root_positions[0,0]:+.4f} -> {root_positions[-1,0]:+.4f}  "
      f"delta={root_positions[-1,0]-root_positions[0,0]:+.4f} m")
print(f"    Y: {root_positions[0,1]:+.4f} -> {root_positions[-1,1]:+.4f}  "
      f"delta={root_positions[-1,1]-root_positions[0,1]:+.4f} m")
print(f"    Z: {root_positions[0,2]:+.4f} -> {root_positions[-1,2]:+.4f}  "
      f"delta={root_positions[-1,2]-root_positions[0,2]:+.4f} m")
print(f"  Root yaw change: {np.rad2deg(root_yaws[0]):+.1f}° -> {np.rad2deg(root_yaws[-1]):+.1f}° "
      f"(delta={np.rad2deg(root_yaws[-1]-root_yaws[0]):+.2f}°)")

print(f"\n  Mean |vel|:  lin={np.mean(np.linalg.norm(root_lin_vels[:,:2], axis=1)):.4f} m/s  "
      f"ang={np.mean(np.linalg.norm(root_ang_vels[:,:3], axis=1)):.4f} rad/s")
print(f"  Final base RPY: roll={np.rad2deg(base_rpy[-1,0]):+.2f}° "
      f"pitch={np.rad2deg(base_rpy[-1,1]):+.2f}° yaw={np.rad2deg(base_rpy[-1,2]):+.2f}°")

# Check foot contact forces
print(f"\n  Foot contact forces (last 10 steps):")
lf = foot_contact_forces[-10:, 0]  # left foot
rf = foot_contact_forces[-10:, 1]  # right foot
lf_mag = np.linalg.norm(lf, axis=1)
rf_mag = np.linalg.norm(rf, axis=1)
print(f"    Left  foot: mean={lf_mag.mean():.1f} N")
print(f"    Right foot: mean={rf_mag.mean():.1f} N")

# Check DOF drift
print(f"\n  DOF position change (start -> end):")
for i in range(env.num_dof):
    delta = dof_positions[-1,i] - dof_positions[0,i]
    if abs(delta) > 0.001:
        print(f"    [{i:2d}] {env.dof_names[i]:35s} {dof_positions[0,i]:+.4f} -> {dof_positions[-1,i]:+.4f}  (delta={delta:+.4f})")

# ===========================================================================
# 5. Single joint probe
# ===========================================================================
print("\n--- 5. SINGLE JOINT PROBE ---")
# Restart env — just reuse existing env with reset
env.reset()

target_joints = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

for jname in target_joints:
    jidx = env.dof_names.index(jname)
    env.reset()
    # Wait for settle (0.5s)
    for _ in range(100):
        env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))

    rs = env.root_states[0].cpu()
    before_pos = rs[:3].numpy().copy()
    before_dof = env.dof_pos[0, jidx].item()

    # Apply +0.1 to this joint
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    actions[0, jidx] = 0.1
    for _ in range(50):
        env.step(actions)

    after_pos = env.root_states[0, :3].cpu().numpy().copy()
    after_dof = env.dof_pos[0, jidx].item()
    delta_dof = after_dof - before_dof

    print(f"  {jname:35s}  dof: {before_dof:+.4f} -> {after_dof:+.4f} (delta={delta_dof:+.4f})  "
          f"base_dx={after_pos[1]-before_pos[1]:+.4f} m  "
          f"base_dz={after_pos[2]-before_pos[2]:+.4f} m")


# ===========================================================================
# 6. Reduced action scale test
# ===========================================================================
print("\n--- 6. REDUCED ACTION SCALE TEST ---")
cfg3 = Q1GoalkeeperCfg()
cfg3.env.num_envs = 6
cfg3.terrain.mesh_type = 'plane'
cfg3.domain_rand.randomize_initial_joint_pos = False
cfg3.domain_rand.push_robots = False
cfg3.domain_rand.randomize_friction = False
cfg3.asset.self_collisions = 0
cfg3.control.action_scale = 0.05
cfg3.init_state.capture_default_dof_pos_from_sim = False
cfg3.terrain.static_friction = 1.0
cfg3.terrain.dynamic_friction = 1.0
cfg3.terrain.restitution = 0.0

env3 = LeggedRobot(cfg3, sim_params, gymapi.SIM_PHYSX, "cuda:0", True)
env3.reset()
print(f"  Action scale: {cfg3.control.action_scale}")
print(f"  Friction: static=1.0, dynamic=1.0, restitution=0.0")

n_frames = 200
for step in range(n_frames):
    env3.step(torch.zeros(env3.num_envs, env3.num_actions, device=env3.device))

rs = env3.root_states[0].cpu().numpy()
print(f"  After 1s zero-action:")
print(f"    Root pos: ({rs[0]:+.4f}, {rs[1]:+.4f}, {rs[2]:+.4f})")
print(f"    Root yaw: {np.rad2deg(np.arctan2(2*(rs[6]*rs[5]+rs[3]*rs[4]), 1-2*(rs[4]**2+rs[5]**2))):+.2f}°")

env3.shutdown()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
