#!/usr/bin/env python3
"""
Q1 Goalkeeper MuJoCo sim2sim — minimal single-policy closed-loop verification.

Loads a Q1 goalkeeper ONNX policy (750-dim obs, 22-dim action),
runs it in a MuJoCo simulation with ball physics, PD control,
and records a video.

Usage:
  # Sanity check 0: zero-action, no policy, no ball launch
  python scripts/q1_goalkeeper_mujoco_sim2sim.py \
      --config scripts/q1_goalkeeper_mujoco_config.yaml \
      --sanity-zero-action --duration 5.0 --headless

  # Sanity check 1: dummy policy inference (no MuJoCo)
  python scripts/q1_goalkeeper_mujoco_sim2sim.py \
      --config scripts/q1_goalkeeper_mujoco_config.yaml \
      --sanity-policy-dummy

  # Default closed-loop with ball launch
  python scripts/q1_goalkeeper_mujoco_sim2sim.py \
      --config scripts/q1_goalkeeper_mujoco_config.yaml \
      --duration 5.0 --headless \
      --video-out scripts/outputs/q1_goalkeeper_mujoco.mp4
"""

import os
import sys
import argparse
import time
import yaml
import numpy as np

# Must set MUJOCO_GL before importing mujoco for headless rendering
if "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco
import mujoco.viewer
import onnxruntime as ort

from standby_policy import (
    build_standby_observation,
    create_standby_history,
    resolve_standby_policy_path,
    standby_action_to_target,
    standby_dimensions,
    update_standby_history,
)


# ==============================================================================
# Quaternion helpers (scipy.spatial.transform-free, for minimal deps)
# ==============================================================================

def quat_inverse(q):
    """Inverse of quaternion [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_rotate(q, v):
    """Rotate vector v by quaternion q = [w, x, y, z]. Returns v' = q * v * q^-1."""
    q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]
    # Compute q * v (as quaternion with w=0)
    v_x, v_y, v_z = v[0], v[1], v[2]
    # Formula: v' = v + 2*w*(qv × v) + 2*(qv × (qv × v))
    # Using scipy-free cross products
    qv = np.array([q_x, q_y, q_z])
    t = 2.0 * np.cross(qv, v)
    return v + q_w * t + np.cross(qv, t)


def quat_rotate_inverse(q, v):
    """Rotate vector v by inverse of quaternion q."""
    return quat_rotate(quat_inverse(q), v)


# ==============================================================================
# Config reader
# ==============================================================================

def read_conf(config_path):
    """Read YAML config. Raises on missing file or required keys."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    required = [
        "xml_path", "policy_path",
        "num_actions", "num_single_obs", "frame_stack", "num_obs",
        "joint_names", "default_dof_pos",
        "kps", "kds", "tau_limit", "action_scale_vec",
        "simulation_dt", "control_decimation",
    ]
    for k in required:
        if k not in cfg:
            raise KeyError(f"Config missing required key: {k}")

    # Resolve paths relative to config file directory
    cfg_dir = os.path.dirname(os.path.abspath(config_path))
    cfg["_config_dir"] = cfg_dir
    for path_key in ["xml_path", "policy_path"]:
        p = cfg[path_key]
        if not os.path.isabs(p):
            cfg[path_key] = os.path.normpath(os.path.join(cfg_dir, p))

    # Validate vector lengths
    n = cfg["num_actions"]
    for vec_key in ["joint_names", "default_dof_pos", "kps", "kds", "tau_limit", "action_scale_vec"]:
        vec = cfg[vec_key]
        assert len(vec) == n, f"{vec_key}: expected {n}, got {len(vec)}"

    return cfg


# ==============================================================================
# ONNX policy
# ==============================================================================

def load_onnx_policy(policy_path):
    """Load ONNX policy, return session + input/output metadata."""
    if not os.path.isfile(policy_path):
        raise FileNotFoundError(f"ONNX policy not found: {policy_path}")
    session = ort.InferenceSession(policy_path)
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    return {"session": session, "input_name": inp.name, "output_name": out.name}


def inspect_policy_io(policy, expected_obs, expected_act):
    """Validate ONNX input/output shapes. Raises on mismatch."""
    inp_name = policy["input_name"]
    out_name = policy["output_name"]
    inp_shape = policy["session"].get_inputs()[0].shape
    out_shape = policy["session"].get_outputs()[0].shape

    print(f"ONNX input:  name={inp_name}, shape={inp_shape}")
    print(f"ONNX output: name={out_name}, shape={out_shape}")

    # Check last dim
    if inp_shape[-1] != expected_obs and inp_shape[-1] not in ("obs", None):
        raise ValueError(f"Expected ONNX input dim={expected_obs}, got {inp_shape[-1]}")
    if out_shape[-1] != expected_act and out_shape[-1] not in ("actions", None):
        raise ValueError(f"Expected ONNX output dim={expected_act}, got {out_shape[-1]}")

    # Verify with dummy input
    dummy = np.zeros((1, expected_obs), dtype=np.float32)
    result = policy["session"].run([out_name], {inp_name: dummy})[0]
    assert result.shape == (1, expected_act), f"Dummy inference shape: expected (1,{expected_act}), got {result.shape}"
    assert np.isfinite(result).all(), "Dummy inference returned non-finite values"
    print(f"  Dummy inference OK: shape={result.shape}, mean={result.mean():.4f}, max={result.max():.4f}")


def policy_infer(policy, obs):
    """Run ONNX inference."""
    out_name = policy["output_name"]
    inp_name = policy["input_name"]
    result = policy["session"].run([out_name], {inp_name: obs.astype(np.float32)})[0]
    return result


# ==============================================================================
# MuJoCo model
# ==============================================================================

def build_mujoco_model(cfg):
    """Load MuJoCo model, data, and set solver params."""
    xml_path = cfg["xml_path"]
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"XML not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = float(cfg["simulation_dt"])
    data = mujoco.MjData(model)

    # Solver settings
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = cfg.get("solver_iterations", 100)
    model.opt.ls_iterations = cfg.get("ls_iterations", 50)

    log_mujoco_model_summary(model, cfg)

    return model, data


def log_mujoco_model_summary(model, cfg):
    """Print the loaded model timing and key physical parameters."""
    decimation = int(cfg["control_decimation"])
    policy_dt = float(model.opt.timestep) * decimation
    print("[MUJOCO_XML] path={}".format(cfg["xml_path"]))
    print(f"[MUJOCO_TIMING] model.opt.timestep={model.opt.timestep:.6f}")
    print(f"[MUJOCO_TIMING] decimation={decimation}")
    print(f"[MUJOCO_TIMING] policy_dt={policy_dt:.6f}")

    ball_id = model.body("ball").id
    ball_geoms = np.where(model.geom_bodyid == ball_id)[0]
    if len(ball_geoms) > 0:
        geom_id = int(ball_geoms[0])
        print(
            "[MUJOCO_BALL] "
            f"mass={model.body_mass[ball_id]:.6g}, "
            f"radius={model.geom_size[geom_id, 0]:.6g}, "
            f"friction={model.geom_friction[geom_id].tolist()}"
        )

    robot_geom_ids = [
        gid for gid in range(model.ngeom)
        if model.geom_bodyid[gid] != ball_id and model.geom_bodyid[gid] != 0
    ]
    if robot_geom_ids:
        gid = int(robot_geom_ids[0])
        print(
            "[MUJOCO_BODY_CONTACT] "
            f"body_friction={model.geom_friction[gid].tolist()}, "
            f"body_condim={int(model.geom_condim[gid])}"
        )


def build_joint_index_map(model, cfg):
    """Map joint names to actuator and qpos/qvel indices.

    Returns dict:
      - actuator_ids: list[int]  — actuator indices in policy joint order
      - qpos_ids: list[int]      — qpos indices in policy joint order
      - qvel_ids: list[int]      — qvel indices in policy joint order
      - imu_body_id: int         — body index for policy base/IMU frame (pelvis)
      - torso_body_id: int       — body index for ball position reference
      - ball_body_id: int        — body index for ball
    """
    joint_names = cfg["joint_names"]
    n = len(joint_names)

    actuator_ids = []
    qpos_ids = []
    qvel_ids = []

    # Get all actuator names (MuJoCo 3.x API)
    actuator_names = {}
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]  # transmission joint id
        if jid >= 0:
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jname:
                actuator_names[jname] = i

    # Map each joint
    for jname in joint_names:
        # Check joint exists
        jid = model.joint(jname).id
        if jid < 0:
            raise ValueError(f"Joint '{jname}' not found in MuJoCo model")

        # Check actuator exists
        if jname not in actuator_names:
            raise ValueError(f"No actuator found for joint '{jname}'")
        actuator_ids.append(actuator_names[jname])

        # qpos/qvel addresses (MuJoCo 3.x: jnt_qposadr, jnt_dofadr)
        qpos_ids.append(model.jnt_qposadr[jid])
        qvel_ids.append(model.jnt_dofadr[jid])

    # Policy base frame. IsaacGym q1 uses control.upper_body_link="pelvis".
    imu_name = cfg.get("imu_body_name", "pelvis")
    imu_body_id = model.body(imu_name).id
    if imu_body_id < 0:
        raise ValueError(f"IMU body {imu_name} not found in MuJoCo model")

    # Ball feature position reference. IsaacGym uses torso_pos here.
    torso_name = cfg.get("torso_body_name", "torso_link")
    torso_body_id = model.body(torso_name).id
    if torso_body_id < 0:
        raise ValueError(f"Torso body {torso_name} not found in MuJoCo model")

    # Ball body
    ball_body_id = model.body("ball").id
    if ball_body_id < 0:
        raise ValueError("Ball body not found in MuJoCo model")

    # Verify no gaps in actuator mapping
    assert len(actuator_ids) == n, f"Expected {n} actuators, found {len(actuator_ids)}"
    assert len(qpos_ids) == n
    assert len(qvel_ids) == n

    return {
        "actuator_ids": actuator_ids,
        "qpos_ids": qpos_ids,
        "qvel_ids": qvel_ids,
        "imu_body_id": imu_body_id,
        "torso_body_id": torso_body_id,
        "ball_body_id": ball_body_id,
    }


# 6 shot modes matching training Q1GoalkeeperCfg.commands
SHOT_MODES = {
    0: {"height": [0.26, 0.78], "width": [0.13, 0.78]},    # right, mid-low
    1: {"height": [0.26, 0.78], "width": [-0.78, -0.13]},   # left, mid-low
    2: {"height": [0.78, 1.04], "width": [0.0, 0.65]},      # right, high
    3: {"height": [0.78, 1.04], "width": [-0.65, 0.0]},     # left, high
    4: {"height": [0.07, 0.20], "width": [0.13, 0.78]},     # right, low
    5: {"height": [0.07, 0.20], "width": [-0.78, -0.13]},   # left, low
}

def compute_shot_from_mode(mode, g=9.81, t_flight=None):
    """Compute shot_init_pos and shot_init_vel matching assign_ball_states random sampling."""
    if mode not in SHOT_MODES:
        raise ValueError(f"Invalid shot mode {mode}, must be 0-5")
    r = SHOT_MODES[mode]
    w0, w1 = r["width"]
    h0, h1 = r["height"]
    # Random sampling matching training assign_ball_states
    start_x = np.random.rand() * 2.0 + 3.0          # [3, 5]
    start_y = np.random.rand() * (w1 - w0) + w0     # [w0, w1]
    start_z = np.random.rand() * (h1 - h0) + h0     # [h0, h1]
    end_x = -np.random.rand() * 0.5 - 0.1            # [-0.6, -0.1]
    end_y = np.random.rand() * (w1 - w0) + w0       # [w0, w1]
    end_z = np.random.rand() * (h1 - h0) + h0       # [h0, h1]
    if t_flight is None:
        t_flight = 0.4 + np.random.rand() * 0.6      # [0.4, 1.0]
    vx = (end_x - start_x) / t_flight
    vy = (end_y - start_y) / t_flight
    vz = (end_z - start_z + 0.5 * g * t_flight**2) / t_flight
    return np.array([start_x, start_y, start_z]), np.array([vx, vy, vz])

def reset_robot_and_ball(model, data, index_map, cfg, shot_mode=-1):
    """Reset robot to default pose and ball to initial position/velocity."""
    # Robot root
    root_pos = np.array(cfg.get("init_root_pos", [0.0, 0.0, 0.79]))
    root_quat = np.array(cfg.get("init_root_quat", [1.0, 0.0, 0.0, 0.0]))
    data.qpos[0:3] = root_pos
    data.qpos[3:7] = root_quat  # wxyz

    # Robot joints
    default_pos = np.array(cfg["default_dof_pos"])
    for i, qpos_id in enumerate(index_map["qpos_ids"]):
        data.qpos[qpos_id] = default_pos[i]

    # Zero velocities
    data.qvel[:] = 0.0

    # Ball
    ball_qpos_start = model.jnt_qposadr[model.joint("ball_free").id]
    ball_qvel_start = model.jnt_dofadr[model.joint("ball_free").id]

    if shot_mode >= 0:
        shot_pos, shot_vel = compute_shot_from_mode(shot_mode)
    else:
        shot_pos = np.array(cfg.get("shot_init_pos", [2.5, 0.0, 0.35]))
        shot_vel = np.array(cfg.get("shot_init_vel", [-5.0, 0.0, 0.0]))

    data.qpos[ball_qpos_start:ball_qpos_start + 3] = shot_pos
    data.qpos[ball_qpos_start + 3:ball_qpos_start + 7] = [1.0, 0.0, 0.0, 0.0]  # wxyz
    data.qvel[ball_qvel_start:ball_qvel_start + 3] = shot_vel
    data.qvel[ball_qvel_start + 3:ball_qvel_start + 6] = 0.0  # zero angular vel

    mujoco.mj_forward(model, data)


def get_robot_state(model, data, index_map, cfg):
    """Extract robot state from MuJoCo data.

    Returns dict with:
      - base_quat: [w,x,y,z] pelvis/root orientation
      - base_pos: [x,y,z] world position of pelvis/root body
      - torso_pos: [x,y,z] world position of torso_link
      - ang_vel_world: [wx,wy,wz] world-frame angular velocity of pelvis/root body
      - dof_pos: [22] joint positions
      - dof_vel: [22] joint velocities
      - projected_gravity: [gx,gy,gz] unit gravity direction in base frame
    """
    imu_id = index_map["imu_body_id"]
    torso_id = index_map["torso_body_id"]

    base_quat = data.xquat[imu_id].copy()  # wxyz
    base_pos = data.xpos[imu_id].copy()
    torso_pos = data.xpos[torso_id].copy()

    ang_vel_world = get_body_angular_velocity_world(model, data, imu_id)

    dof_pos = np.array([data.qpos[qid] for qid in index_map["qpos_ids"]])
    dof_vel = np.array([data.qvel[vid] for vid in index_map["qvel_ids"]])

    gravity_world = np.array([0.0, 0.0, -1.0])
    projected_gravity = quat_rotate_inverse(base_quat, gravity_world)
    ang_vel_base = quat_rotate_inverse(base_quat, ang_vel_world)

    return {
        "base_quat": base_quat,
        "base_pos": base_pos,
        "torso_pos": torso_pos,
        "ang_vel_base": ang_vel_base,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "projected_gravity": projected_gravity,
    }


def get_ball_state(model, data, index_map, cfg):
    """Get ball world position and velocity."""
    ball_body_id = index_map["ball_body_id"]
    ball_pos = data.xpos[ball_body_id].copy()
    ball_vel_world = get_body_linear_velocity_world(model, data, ball_body_id)
    return ball_pos, ball_vel_world


def _object_velocity_world(model, data, body_id):
    """Return MuJoCo object velocity as [angular, linear] in world frame."""
    vel = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel, 0)
    return vel


def get_body_angular_velocity_world(model, data, body_id):
    """World-frame body angular velocity, avoiding ambiguous cvel slicing."""
    if hasattr(data, "xvelr"):
        return data.xvelr[body_id].copy()
    return _object_velocity_world(model, data, body_id)[0:3].copy()


def get_body_linear_velocity_world(model, data, body_id):
    """World-frame body linear velocity, avoiding ambiguous cvel slicing."""
    if hasattr(data, "xvelp"):
        return data.xvelp[body_id].copy()
    return _object_velocity_world(model, data, body_id)[3:6].copy()


# ==============================================================================
# Observation construction
# ==============================================================================

def create_history_buffer(cfg):
    """Create zero-initialized history buffer. Shape: (frame_stack, single_obs)."""
    return np.zeros((cfg["frame_stack"], cfg["num_single_obs"]), dtype=np.float32)


def create_ball_obs_state(cfg):
    """Create state for IsaacGyms stateful ball visibility mask."""
    return {
        "catchstep": int(cfg.get("ball_catchstep", 50)),
        # MuJoCo observes before its next decimated physics block, while the
        # Isaac trace consumes the post-step obs. Delay only the visibility mask
        # by one policy step; keep control_startstep aligned to Isaac.
        "startstep": int(cfg.get("ball_obs_startstep", 46)),
        "control_startstep": int(cfg.get("control_startstep", 47)),
        "vanish_step": int(cfg.get("ball_vanish_step", 0)),
        "ball_last": np.zeros(3, dtype=np.float32),
    }


def advance_ball_obs_state(ball_obs_state):
    """Match IsaacGym post_physics_step catchstep update at the policy rate."""
    if ball_obs_state is not None:
        ball_obs_state["catchstep"] -= 1


def compute_ball_feature(ball_pos_world, torso_pos, base_quat, cfg, ball_obs_state=None):
    """Compute IsaacGym actor ball feature.

    base_quat is the pelvis/root quaternion. torso_pos is the torso_link
    world position used as the relative-position reference in training.
    """
    ball_rel_world = ball_pos_world - torso_pos
    raw_local = quat_rotate_inverse(base_quat, ball_rel_world).astype(np.float32)

    if ball_obs_state is None:
        return raw_local

    initial_visible = ball_obs_state["catchstep"] < ball_obs_state["startstep"]
    end_target_local = raw_local if initial_visible else np.zeros(3, dtype=np.float32)
    ball_last = ball_obs_state["ball_last"]
    flying = (
        (end_target_local[0] > 0.05)
        and (end_target_local[0] < 3.4)
        and (end_target_local[1] > -2.0)
        and (end_target_local[1] < 2.0)
        and (end_target_local[2] < 1.8)
        and (ball_obs_state["catchstep"] > 0)
        and ((end_target_local[0] < ball_last[0]) or (ball_last[0] == 0.0))
    )
    ball_obs_state["ball_last"] = end_target_local.astype(np.float32)
    if not flying:
        return np.zeros(3, dtype=np.float32)
    return end_target_local.astype(np.float32)


def build_single_obs(robot_state, ball_feature, last_action, cfg):
    """Build single-frame 75-dim observation.

    Order (from training compute_observations, first 75 dims):
      [0:3]   ball_feature          — base-frame relative ball position (unscaled)
      [3:6]   ang_vel_base * 0.25   — scaled
      [6:9]   projected_gravity     — unscaled
      [9:31]  (dof_pos - default) * 1.0
      [31:53] dof_vel * 0.05
      [53:75] last_action           — previous step's policy output
    """
    default_pos = np.array(cfg["default_dof_pos"])

    obs = np.concatenate([
        ball_feature,                                                   # 3
        robot_state["ang_vel_base"] * cfg["obs_scale_ang_vel"],         # 3
        robot_state["projected_gravity"],                               # 3
        (robot_state["dof_pos"] - default_pos) * cfg["obs_scale_dof_pos"],  # 22
        robot_state["dof_vel"] * cfg["obs_scale_dof_vel"],              # 22
        last_action,                                                    # 22
    ]).astype(np.float32)

    assert obs.shape == (75,), f"Single obs shape: expected (75,), got {obs.shape}"
    return obs


def update_history_and_get_obs(history, single_obs, cfg):
    """Push single_obs to history (old→new), return flattened 750-dim obs.

    History layout: row 0 = oldest, row frame_stack-1 = newest.
    On update: shift rows up (drop oldest), new frame at bottom.
    """
    # Shift: row 0..frame_stack-2 → row 1..frame_stack-1
    history[:-1] = history[1:]
    history[-1] = single_obs

    # Flatten to (750,)
    obs = history.flatten().astype(np.float32)
    assert obs.shape == (cfg["num_obs"],), f"Obs shape: expected ({cfg['num_obs']},), got {obs.shape}"
    return obs


# ==============================================================================
# Control
# ==============================================================================

def pd_control(target_pos, dof_pos, dof_vel, cfg):
    """Compute PD torque: tau = kp*(target - pos) - kd*vel. Clip to tau_limit."""
    kps = np.array(cfg["kps"])
    kds = np.array(cfg["kds"])
    tau_limit = np.array(cfg["tau_limit"])

    tau = kps * (target_pos - dof_pos) - kds * dof_vel
    tau = np.clip(tau, -tau_limit, tau_limit)
    return tau


def action_to_target_dof_pos(action, cfg):
    """Convert policy action to target DOF positions: target = default + action * action_scale_vec."""
    default_pos = np.array(cfg["default_dof_pos"])
    action_scale_vec = np.array(cfg["action_scale_vec"])
    return default_pos + action * action_scale_vec


def _fmt_arr(x, precision=3):
    arr = np.asarray(x).reshape(-1)
    return "[" + ",".join(("{:." + str(precision) + "f}").format(float(v)) for v in arr) + "]"


def get_contact_debug(model, data):
    contact_forces = np.zeros((data.ncon, 6), dtype=np.float64)
    contacts = []
    foot_force = 0.0
    for ci in range(data.ncon):
        mujoco.mj_contactForce(model, data, ci, contact_forces[ci])
        g1 = int(data.contact[ci].geom1)
        g2 = int(data.contact[ci].geom2)
        b1 = model.body(int(model.geom_bodyid[g1])).name if g1 < model.ngeom else "world"
        b2 = model.body(int(model.geom_bodyid[g2])).name if g2 < model.ngeom else "world"
        fn = float(np.linalg.norm(contact_forces[ci, :3]))
        if "ankle" in b1 or "ankle" in b2 or "foot" in b1 or "foot" in b2:
            foot_force += fn
        contacts.append((fn, b1, b2, float(data.contact[ci].dist)))
    contacts.sort(key=lambda x: -x[0])
    return foot_force, contacts[:5]


def make_qacc_debug_record(model, data, index_map, cfg, policy_step, physics_step, raw_action,
                           applied_target, current_rs, tau, ball_feature, ball_visible,
                           catchstep_override_active):
    qacc_abs = np.abs(data.qacc)
    tau_abs = np.abs(tau)
    target_dev = applied_target - current_rs["dof_pos"]
    target_dev_abs = np.abs(target_dev)
    grav = current_rs["projected_gravity"]
    roll = np.arctan2(grav[1], -grav[2]) if abs(grav[2]) > 0.1 else 0.0
    pitch = np.arctan2(-grav[0], -grav[2]) if abs(grav[2]) > 0.1 else 0.0
    foot_force, top_contacts = get_contact_debug(model, data)
    n_dof = len(cfg["joint_names"])
    qacc_dof = qacc_abs[:n_dof]
    top_qacc = np.argsort(-qacc_dof)[:5]
    top_tau = np.argsort(-tau_abs)[:5]
    top_target = np.argsort(-target_dev_abs)[:5]
    top_action = np.argsort(-np.abs(raw_action))[:5]
    return {
        "time": float(data.time),
        "policy_step": int(policy_step),
        "physics_step": int(physics_step),
        "action_mean_abs": float(np.mean(np.abs(raw_action))),
        "action_max_abs": float(np.max(np.abs(raw_action))),
        "top_action": [(cfg["joint_names"][int(i)], float(raw_action[int(i)])) for i in top_action],
        "top_target_dev": [(cfg["joint_names"][int(i)], float(target_dev[int(i)])) for i in top_target],
        "dof_pos": current_rs["dof_pos"].copy(),
        "dof_vel": current_rs["dof_vel"].copy(),
        "qacc_max": float(np.max(qacc_abs)),
        "top_qacc": [(cfg["joint_names"][int(i)], float(qacc_dof[int(i)])) for i in top_qacc],
        "torque_max": float(np.max(tau_abs)),
        "top_tau": [(cfg["joint_names"][int(i)], float(tau[int(i)])) for i in top_tau],
        "torque_sat_ratio": float((tau_abs >= np.asarray(cfg["tau_limit"]) * 0.95).mean()),
        "root_roll_deg": float(np.rad2deg(roll)),
        "root_pitch_deg": float(np.rad2deg(pitch)),
        "pelvis_ang_vel": current_rs["ang_vel_base"].copy(),
        "foot_contact_force": float(foot_force),
        "top_contacts": top_contacts,
        "ball_visible": bool(ball_visible),
        "ball_feature": np.asarray(ball_feature).copy(),
        "catchstep_override_active": bool(catchstep_override_active),
    }


def print_qacc_debug_window(records, trigger_index):
    if trigger_index is None:
        print("[QACC_DEBUG] no qacc trigger recorded")
        return
    start = max(0, trigger_index - 10)
    end = min(len(records), trigger_index + 11)
    print("[QACC_DEBUG] trigger_index={}, window=[{},{})".format(trigger_index, start, end))
    for rec in records[start:end]:
        print(
            "[QACC_DEBUG] t={:.5f} policy={} phys={} qacc_max={:.3g} torque_max={:.3g} "
            "sat={:.2f} action_mean|max={:.3f}|{:.3f} roll|pitch={:.2f}|{:.2f} "
            "footF={:.3g} ball_visible={} catchstep_override={} ball={}".format(
                rec["time"],
                rec["policy_step"],
                rec["physics_step"],
                rec["qacc_max"],
                rec["torque_max"],
                rec["torque_sat_ratio"],
                rec["action_mean_abs"],
                rec["action_max_abs"],
                rec["root_roll_deg"],
                rec["root_pitch_deg"],
                rec["foot_contact_force"],
                rec["ball_visible"],
                rec["catchstep_override_active"],
                _fmt_arr(rec["ball_feature"]),
            )
        )
        print("  top_action={}".format(rec["top_action"]))
        print("  top_target_dev={}".format(rec["top_target_dev"]))
        print("  top_qacc={}".format(rec["top_qacc"]))
        print("  top_tau={}".format(rec["top_tau"]))
        print("  pelvis_ang_vel={}".format(_fmt_arr(rec["pelvis_ang_vel"])))
        print("  dof_pos={}".format(_fmt_arr(rec["dof_pos"])))
        print("  dof_vel={}".format(_fmt_arr(rec["dof_vel"])))
        print("  top_contacts={}".format(rec["top_contacts"]))


# ==============================================================================
# Ball-state activation wrapper
# ==============================================================================

def check_ball_launch_trigger(ball_pos_history, ball_init_pos, policy_dt, args):
    """Detect ball launch from position-history velocity estimation.

    Uses ball_pos_history for 差分 velocity estimation (no MuJoCo ground-truth).
    Returns (triggered, ball_vel_est, speed, v_toward_goal, displacement).
    """
    speed_thresh = float(getattr(args, "trigger_speed_threshold", 0.3))
    toward_thresh = float(getattr(args, "trigger_toward_speed_threshold", 0.2))
    disp_thresh = float(getattr(args, "trigger_displacement_threshold", 0.02))

    if len(ball_pos_history) < 5:
        return False, np.zeros(3), 0.0, 0.0, 0.0

    ball_vel_est = (ball_pos_history[-1] - ball_pos_history[-5]) / (5.0 * policy_dt)
    speed = float(np.linalg.norm(ball_vel_est))

    # Goal direction: from ball toward goal center (goalkeeper frame origin)
    goal_center = np.array([0.0, 0.0, 0.5])
    to_goal = goal_center - ball_pos_history[-1]
    to_goal_norm = float(np.linalg.norm(to_goal))
    goal_dir = to_goal / to_goal_norm if to_goal_norm > 1e-6 else np.zeros(3)
    v_toward_goal = float(np.dot(ball_vel_est, goal_dir))

    displacement = float(np.linalg.norm(ball_pos_history[-1] - ball_init_pos))

    triggered = (
        speed > speed_thresh
        and v_toward_goal > toward_thresh
        and displacement > disp_thresh
    )
    return triggered, ball_vel_est, speed, v_toward_goal, displacement


# ==============================================================================
# Main simulation loop
# ==============================================================================

def run_mujoco(cfg, args):
    """Run MuJoCo sim2sim with goalkeeper policy."""
    model, data = build_mujoco_model(cfg)
    index_map = build_joint_index_map(model, cfg)

    # Load policy
    policy = load_onnx_policy(cfg["policy_path"])
    inspect_policy_io(policy, cfg["num_obs"], cfg["num_actions"])

    standby_policy_path = resolve_standby_policy_path(cfg, getattr(args, "standby_policy", None))
    standby_policy = None
    if standby_policy_path is not None:
        if "standby_default_dof_pos" not in cfg:
            raise KeyError("standby_default_dof_pos is required when using --standby-policy")
        _, _, standby_obs_dim = standby_dimensions(cfg)
        standby_policy = load_onnx_policy(standby_policy_path)
        inspect_policy_io(standby_policy, standby_obs_dim, cfg["num_actions"])
        print(f"  [WRAPPER] standby policy: {standby_policy_path}")

    # Control parameters
    sim_dt = cfg["simulation_dt"]
    decimation = cfg["control_decimation"]
    policy_dt = sim_dt * decimation  # 0.02s = 50Hz

    # Reset
    shot_mode = getattr(args, "shot_mode", -1)
    if shot_mode >= 0:
        sp, sv = compute_shot_from_mode(shot_mode)
        print(f"  shot_mode={shot_mode}: pos={sp} vel={sv}")
    reset_robot_and_ball(model, data, index_map, cfg, shot_mode=shot_mode)

    # Observation buffers. IsaacGym starts with zero history and appends current obs.
    history = create_history_buffer(cfg)
    ball_obs_state = create_ball_obs_state(cfg)
    last_action = np.zeros(cfg["num_actions"], dtype=np.float32)
    target_dof_pos = np.array(cfg["default_dof_pos"], dtype=np.float64)
    init_dof_pos = np.array(cfg["default_dof_pos"], dtype=np.float64)
    standby_history = create_standby_history(cfg) if standby_policy is not None else None
    standby_last_action = np.zeros(cfg["num_actions"], dtype=np.float32)
    standby_target_dof_pos = init_dof_pos.copy()
    debug_zero_action = bool(getattr(args, "debug_zero_action", False))
    debug_policy_no_apply = bool(getattr(args, "debug_policy_no_apply", False))
    debug_apply_isaac_catchstep = bool(getattr(args, "debug_apply_isaac_catchstep", False))
    debug_qacc_log = bool(getattr(args, "debug_qacc_log", False))
    debug_qacc_threshold = getattr(args, "debug_qacc_threshold", None)
    if debug_qacc_threshold is None:
        debug_qacc_threshold = cfg.get("max_abs_qacc", 50000.0)
    debug_qacc_threshold = float(debug_qacc_threshold)

    # --- Ball-state activation wrapper ---
    activation_mode = str(getattr(args, "activation_mode", "ball_state_trigger"))
    gk_visible_mode = str(getattr(args, "gk_visible_after_trigger", "train_timing"))
    debounce_steps = int(getattr(args, "trigger_debounce_steps", 2))

    STATE_WAIT, STATE_ACTIVE, STATE_RECOVER = "WAIT", "ACTIVE", "RECOVER"
    state = STATE_ACTIVE if activation_mode == "always_on" else STATE_WAIT
    ball_pos_history = []
    ball_init_pos = None
    debounce_counter = 0
    gk_local_step = 0
    trigger_info = None
    ball_qvel_start = model.jnt_dofadr[model.joint("ball_free").id]

    if activation_mode == "always_on":
        print("  [WRAPPER] mode=always_on — policy active from t=0")
    else:
        print(f"  [WRAPPER] mode={activation_mode}, visible={gk_visible_mode}, "
              f"speed_thresh={getattr(args, 'trigger_speed_threshold', 0.3):.1f}, "
              f"debounce={debounce_steps}")

    if getattr(args, "no_ball_launch", False):
        data.qvel[ball_qvel_start:ball_qvel_start + 6] = 0.0
        state = STATE_ACTIVE
        mujoco.mj_forward(model, data)

    # Ball launch delay (physics-only): freeze ball for N seconds, independent of wrapper
    ball_launch_delay = float(getattr(args, "ball_launch_delay", 0.0))
    ball_launch_delay_steps = int(ball_launch_delay / policy_dt) if ball_launch_delay > 0 else 0
    ball_qpos_start = model.jnt_qposadr[model.joint("ball_free").id]
    saved_ball_qpos = None
    saved_ball_qvel = None
    if ball_launch_delay > 0:
        saved_ball_qpos = data.qpos[ball_qpos_start:ball_qpos_start + 7].copy()
        saved_ball_qvel = data.qvel[ball_qvel_start:ball_qvel_start + 6].copy()
        data.qvel[ball_qvel_start:ball_qvel_start + 6] = 0.0
        mujoco.mj_forward(model, data)
        print(f"  Ball launch delay: {ball_launch_delay}s ({ball_launch_delay_steps} steps), ball frozen")

    # Video recording setup
    renderer = None
    if args.video_out:
        from mujoco.renderer import Renderer
        width = getattr(args, "width", 1280)
        height = getattr(args, "height", 720)
        # Headless: use EGL/OSMesa for offscreen rendering
        gl_context = mujoco.GLContext(max_width=width, max_height=height)
        gl_context.make_current()
        renderer = Renderer(model, height=height, width=width)
        # Camera: side view showing robot (x≈0) and ball start (x≈4)
        # MuJoCo 3.x MjvGLCamera: set pos (camera position) and forward (look direction)
        cam_pos = np.array([2.0, -4.0, 1.5])  # camera at side, elevated
        cam_lookat = np.array([2.0, 0.0, 0.5])  # look at midpoint
        cam_forward = cam_lookat - cam_pos
        cam_forward = cam_forward / np.linalg.norm(cam_forward)
        renderer.scene.camera[0].pos[:] = cam_pos
        renderer.scene.camera[0].forward[:] = cam_forward
        os.makedirs(os.path.dirname(os.path.abspath(args.video_out)) or ".", exist_ok=True)
        print(f"Recording video to: {args.video_out}")

    # Viewer
    viewer = None
    if not args.headless and not args.video_out:
        import mujoco.viewer as mujoco_viewer
        viewer = mujoco_viewer.launch_passive(model, data)

    # Simulation loop: control-step based
    # Each iteration = 1 control step (policy_dt = decimation * sim_dt)
    frames = []
    root_positions = []
    control_step = 0
    stop_reason = "timeout"

    # Ablation setup
    ablation = getattr(args, "ablation", None)
    fixed_ball_feat = None
    if args.fixed_ball_feat:
        fixed_ball_feat = np.array([float(x) for x in args.fixed_ball_feat.split(",")], dtype=np.float32)
    if ablation == "C_zero_action":
        print("  Ablation C: zero-action (PD lock default pose, no policy)")
    elif ablation == "D_fixed_ball":
        print(f"  Ablation D: fixed ball_feature = {fixed_ball_feat}")
    if debug_zero_action:
        print("  Debug: zero-action target, policy inference skipped")
    if debug_policy_no_apply:
        print("  Debug: policy runs but applied target stays at init/default pose")
    if debug_apply_isaac_catchstep:
        print("  Debug: Isaac catchstep target override enabled")
    if debug_qacc_log:
        print(f"  Debug: qacc window logging enabled, threshold={debug_qacc_threshold}")

    qacc_debug_records = []
    qacc_trigger_index = None
    qacc_post_trigger_remaining = 0
    physics_step_global = 0
    num_control_steps = int(cfg.get("simulation_duration", 5.0) / policy_dt)
    t0 = time.time()

    # Wrapper diagnostics
    first_nonzero_ball_feature_local_step = None
    policy_action_max_after_trigger = 0.0
    trigger_control_step = -1

    try:
        for control_step in range(num_control_steps):
            # ============================================
            # STATE: WAIT — PD hold, monitor ball for launch
            # ============================================
            if state == STATE_WAIT:
                # Ball launch delay unfreeze
                if ball_launch_delay > 0 and control_step == ball_launch_delay_steps:
                    data.qpos[ball_qpos_start:ball_qpos_start + 7] = saved_ball_qpos
                    data.qvel[ball_qvel_start:ball_qvel_start + 6] = saved_ball_qvel
                    ball_init_pos = None  # reset so displacement is relative to launch pos
                    ball_pos_history = []  # reset velocity estimation
                    debounce_counter = 0
                    print(f"  [BALL_UNFREEZE] t={control_step * policy_dt:.2f}s — ball launched")

                ball_body_id = index_map["ball_body_id"]
                ball_pos = data.xpos[ball_body_id].copy()
                if ball_init_pos is None:
                    ball_init_pos = ball_pos.copy()
                ball_pos_history.append(ball_pos)
                if len(ball_pos_history) > 12:  # keep ~0.24s
                    ball_pos_history.pop(0)

                # Check launch trigger
                triggered, vel_est, speed, v_tg, disp = check_ball_launch_trigger(
                    ball_pos_history, ball_init_pos, policy_dt, args)

                if triggered:
                    debounce_counter += 1
                else:
                    debounce_counter = 0

                if debounce_counter >= debounce_steps:
                    # ====================================
                    # [GK_TRIGGER] — reset GK policy state
                    # ====================================
                    history = create_history_buffer(cfg)
                    ball_obs_state = create_ball_obs_state(cfg)
                    last_action = np.zeros(cfg["num_actions"], dtype=np.float32)
                    gk_local_step = 0
                    trigger_control_step = control_step

                    if gk_visible_mode == "immediate":
                        ball_obs_state["catchstep"] = ball_obs_state["startstep"] - 1

                    t_trigger = control_step * policy_dt
                    print(f"\n[GK_TRIGGER] at t={t_trigger:.2f}s (step {control_step})")
                    print(f"  ball_pos=[{ball_pos[0]:.3f},{ball_pos[1]:.3f},{ball_pos[2]:.3f}]")
                    print(f"  ball_vel_est=[{vel_est[0]:.3f},{vel_est[1]:.3f},{vel_est[2]:.3f}]")
                    print(f"  speed={speed:.3f} m/s, v_toward_goal={v_tg:.3f} m/s, displacement={disp:.4f} m")
                    print(f"  debounce={debounce_counter}/{debounce_steps}")
                    print(f"  visible_mode={gk_visible_mode}, catchstep={ball_obs_state['catchstep']}")
                    if standby_policy is not None:
                        print(f"  standby-to-GK target max delta={np.max(np.abs(standby_target_dof_pos - init_dof_pos)):.4f} rad")
                    state = STATE_ACTIVE

                # WAIT physics: ready-stand policy when provided, otherwise legacy PD hold.
                if state == STATE_WAIT:
                    wait_target_dof_pos = init_dof_pos
                    if standby_policy is not None:
                        wait_robot_state = get_robot_state(model, data, index_map, cfg)
                        wait_single_obs = build_standby_observation(wait_robot_state, standby_last_action, cfg)
                        wait_obs = update_standby_history(standby_history, wait_single_obs, cfg)
                        wait_action = np.clip(policy_infer(standby_policy, wait_obs.reshape(1, -1))[0], -cfg["clip_actions"], cfg["clip_actions"])
                        if not np.isfinite(wait_action).all():
                            stop_reason = "standby_action_non_finite"
                            break
                        standby_last_action = wait_action.copy()
                        standby_target_dof_pos = standby_action_to_target(wait_action, cfg)
                        wait_target_dof_pos = standby_target_dof_pos
                    for d in range(decimation):
                        rs = get_robot_state(model, data, index_map, cfg)
                        tau = pd_control(wait_target_dof_pos, rs["dof_pos"], rs["dof_vel"], cfg)
                        for i, act_id in enumerate(index_map["actuator_ids"]):
                            data.ctrl[act_id] = tau[i]
                        mujoco.mj_step(model, data)
                        physics_step_global += 1
                        if viewer is not None:
                            viewer.sync()
                        root_positions.append(data.xpos[index_map["imu_body_id"]].copy())
                    if renderer is not None:
                        renderer.update_scene(data)
                        frames.append(renderer.render())
                    # Log every 20 steps
                    if control_step % 20 == 0:
                        t = control_step * policy_dt
                        dbg = f"debounce={debounce_counter}/{debounce_steps}" if triggered else "idle"
                        print(f"  t={t:.2f}s [WAIT {dbg}] | ball_pos=[{ball_pos[0]:.2f},{ball_pos[1]:.2f},{ball_pos[2]:.2f}] "
                              f"speed={speed:.2f} v_tg={v_tg:.2f} disp={disp:.3f}")
                    root_z = data.xpos[index_map["imu_body_id"]][2]
                    if root_z < 0.2 or root_z > 3.0:
                        stop_reason = f"root_height_violation(z={root_z:.3f})"
                        break
                    continue

            # ============================================
            # STATE: ACTIVE — GK policy loop
            # ============================================
            if state == STATE_ACTIVE:
                gk_local_step += 1

                # (A) Build observation
                robot_state = get_robot_state(model, data, index_map, cfg)
                ball_pos_w, ball_vel_w = get_ball_state(model, data, index_map, cfg)
                ball_feature = compute_ball_feature(
                    ball_pos_w,
                    robot_state["torso_pos"],
                    robot_state["base_quat"],
                    cfg,
                    ball_obs_state,
                )

                if ablation == "D_fixed_ball" and fixed_ball_feat is not None:
                    ball_feature = fixed_ball_feat.copy()
                if getattr(args, "zero_ball_obs", False):
                    ball_feature = np.zeros(3, dtype=np.float32)
                # vanish_steps relative to trigger (local step)
                vanish_steps = cfg.get("ball_vanish_steps", 10) if gk_visible_mode == "train_timing" else 0
                if gk_local_step <= vanish_steps:
                    ball_feature = np.zeros(3, dtype=np.float32)

                if first_nonzero_ball_feature_local_step is None and np.linalg.norm(ball_feature) > 1e-6:
                    first_nonzero_ball_feature_local_step = gk_local_step

                single_obs = build_single_obs(robot_state, ball_feature, last_action, cfg)
                obs = update_history_and_get_obs(history, single_obs, cfg)
                obs = np.clip(obs, -cfg["clip_observations"], cfg["clip_observations"])

                if not np.isfinite(obs).all():
                    stop_reason = "obs_non_finite"
                    break

                # (B) Policy inference
                if ablation == "C_zero_action" or debug_zero_action:
                    raw_action = np.zeros(cfg["num_actions"], dtype=np.float32)
                else:
                    raw_action = policy_infer(policy, obs.reshape(1, -1))[0]
                    raw_action = np.clip(raw_action, -cfg["clip_actions"], cfg["clip_actions"])

                if not np.isfinite(raw_action).all():
                    stop_reason = "action_non_finite"
                    break

                policy_target_dof_pos = action_to_target_dof_pos(raw_action, cfg)
                catchstep_override_active = debug_apply_isaac_catchstep and (ball_obs_state["catchstep"] > ball_obs_state["control_startstep"])
                if debug_zero_action or debug_policy_no_apply or catchstep_override_active:
                    target_dof_pos = init_dof_pos.copy()
                else:
                    target_dof_pos = policy_target_dof_pos
                last_action = raw_action.copy()
                ball_visible = bool(np.linalg.norm(ball_feature) > 1e-6)
                if ball_visible:
                    policy_action_max_after_trigger = max(policy_action_max_after_trigger, float(np.max(np.abs(raw_action))))

                # (C) Physics steps
                for d in range(decimation):
                    rs = get_robot_state(model, data, index_map, cfg)
                    tau = pd_control(target_dof_pos, rs["dof_pos"], rs["dof_vel"], cfg)
                    for i, act_id in enumerate(index_map["actuator_ids"]):
                        data.ctrl[act_id] = tau[i]
                    mujoco.mj_step(model, data)
                    physics_step_global += 1
                    if viewer is not None:
                        viewer.sync()
                    root_positions.append(data.xpos[index_map["imu_body_id"]].copy())
                    qacc = np.abs(data.qacc)
                    if debug_qacc_log:
                        qacc_max = float(np.max(qacc))
                        rec = make_qacc_debug_record(
                            model, data, index_map, cfg, control_step, physics_step_global,
                            raw_action, target_dof_pos, rs, tau, ball_feature, ball_visible,
                            catchstep_override_active,
                        )
                        qacc_debug_records.append(rec)
                        if qacc_trigger_index is None and qacc_max > debug_qacc_threshold:
                            qacc_trigger_index = len(qacc_debug_records) - 1
                            qacc_post_trigger_remaining = 10
                        elif qacc_post_trigger_remaining > 0:
                            qacc_post_trigger_remaining -= 1
                    if np.any(qacc > cfg.get("max_abs_qacc", 50000.0)):
                        stop_reason = f"qacc_violation(max={qacc.max():.0f})"
                        if not debug_qacc_log or qacc_post_trigger_remaining <= 0:
                            break

                advance_ball_obs_state(ball_obs_state)

                if renderer is not None:
                    renderer.update_scene(data)
                    frames.append(renderer.render())

                if stop_reason != "timeout":
                    break

                # Active end: local time > 2.0s
                if gk_local_step * policy_dt > 2.0:
                    stop_reason = "gk_active_timeout"
                    break

                # Logging
                if control_step % 20 == 0 or gk_local_step <= 2:
                    t = control_step * policy_dt
                    rs = get_robot_state(model, data, index_map, cfg)
                    bf_obs = single_obs[0:3]
                    grav = rs["projected_gravity"]
                    tau_abs = np.abs(tau)
                    roll = np.arctan2(grav[1], -grav[2]) if abs(grav[2]) > 0.1 else 0.0
                    pitch = np.arctan2(-grav[0], -grav[2]) if abs(grav[2]) > 0.1 else 0.0
                    print(f"  t={t:.2f}s [ACTIVE L+{gk_local_step}] | z={rs['base_pos'][2]:.3f} | "
                          f"roll={np.rad2deg(roll):.1f}° pitch={np.rad2deg(pitch):.1f}° | "
                          f"bf=[{bf_obs[0]:.3f},{bf_obs[1]:.3f},{bf_obs[2]:.3f}] | "
                          f"act_max={np.max(np.abs(raw_action)):.2f} tau_max={tau_abs.max():.1f}")

                # Safety
                root_z = data.xpos[index_map["imu_body_id"]][2]
                if root_z < 0.2 or root_z > 3.0:
                    stop_reason = f"root_height_violation(z={root_z:.3f})"
                    break

                continue

            # ============================================
            # STATE: RECOVER — PD hold, wait for reset
            # ============================================
            if state == STATE_RECOVER:
                for d in range(decimation):
                    rs = get_robot_state(model, data, index_map, cfg)
                    tau = pd_control(init_dof_pos, rs["dof_pos"], rs["dof_vel"], cfg)
                    for i, act_id in enumerate(index_map["actuator_ids"]):
                        data.ctrl[act_id] = tau[i]
                    mujoco.mj_step(model, data)
                    physics_step_global += 1
                    if viewer is not None:
                        viewer.sync()
                    root_positions.append(data.xpos[index_map["imu_body_id"]].copy())
                if renderer is not None:
                    renderer.update_scene(data)
                    frames.append(renderer.render())
                root_z = data.xpos[index_map["imu_body_id"]][2]
                if root_z < 0.2 or root_z > 3.0:
                    stop_reason = f"root_height_violation(z={root_z:.3f})"
                    break

    except KeyboardInterrupt:
        stop_reason = "user_interrupt"
    finally:
        elapsed = time.time() - t0
        if viewer is not None:
            viewer.close()

    if debug_qacc_log:
        print_qacc_debug_window(qacc_debug_records, qacc_trigger_index)

    # --- Summary ---
    root_positions = np.array(root_positions)
    print(f"\n--- Simulation Summary ---")
    print(f"  Control steps: {control_step + 1}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Stop reason: {stop_reason}")
    print(f"  Activation: {activation_mode}, visible: {gk_visible_mode}")
    if trigger_control_step >= 0:
        print(f"  Trigger at: step={trigger_control_step}, t={trigger_control_step * policy_dt:.2f}s")
    print(f"  First nonzero ball_feature (local step): {first_nonzero_ball_feature_local_step}")
    print(f"  Policy action max after trigger: {policy_action_max_after_trigger:.4f}")
    if len(root_positions) > 0:
        print(f"  Root z: min={root_positions[:, 2].min():.3f}, max={root_positions[:, 2].max():.3f}")
        print(f"  Root y: min={root_positions[:, 1].min():.3f}, max={root_positions[:, 1].max():.3f}")

    # Save video
    if renderer is not None and frames:
        _save_video(frames, args.video_out, fps=cfg.get("video_fps", 50))
    if renderer is not None:
        renderer.close()


# ==============================================================================
# Sanity modes
# ==============================================================================

def sanity_zero_action(cfg, args):
    """Run MuJoCo with zero action (default pose) for N seconds, no policy, no ball."""
    model, data = build_mujoco_model(cfg)
    index_map = build_joint_index_map(model, cfg)

    # Reset robot only (no ball launch)
    reset_robot_and_ball(model, data, index_map, cfg)
    # Remove ball velocity
    ball_qvel_start = model.jnt_dofadr[model.joint("ball_free").id]
    data.qvel[ball_qvel_start:ball_qvel_start + 6] = 0.0

    # Move ball far away so it doesn't affect robot
    ball_qpos_start = model.jnt_qposadr[model.joint("ball_free").id]
    data.qpos[ball_qpos_start:ball_qpos_start + 3] = [10.0, 10.0, 10.0]

    mujoco.mj_forward(model, data)

    sim_dt = cfg["simulation_dt"]
    num_steps = int(cfg.get("simulation_duration", 5.0) / sim_dt)
    target_dof_pos = np.array(cfg["default_dof_pos"], dtype=np.float64)

    root_positions = []
    max_abs_qvel = 0.0
    max_abs_qacc = 0.0
    step = 0

    print(f"Zero-action simulation: {num_steps} steps at {sim_dt}s dt")

    for step in range(num_steps):
        robot_state = get_robot_state(model, data, index_map, cfg)
        tau = pd_control(target_dof_pos, robot_state["dof_pos"], robot_state["dof_vel"], cfg)
        for i, act_id in enumerate(index_map["actuator_ids"]):
            data.ctrl[act_id] = tau[i]
        mujoco.mj_step(model, data)
        root_positions.append(robot_state["base_pos"].copy())
        max_abs_qvel = max(max_abs_qvel, np.abs(data.qvel).max())
        max_abs_qacc = max(max_abs_qacc, np.abs(data.qacc).max())

    root_positions = np.array(root_positions)
    print(f"\n--- Zero-Action Summary ---")
    print(f"  Steps: {step + 1}")
    print(f"  Root z: min={root_positions[:, 2].min():.3f}, max={root_positions[:, 2].max():.3f}")
    print(f"  Root height range: {root_positions[:, 2].max() - root_positions[:, 2].min():.3f}")
    print(f"  Max abs qvel: {max_abs_qvel:.3f}")
    print(f"  Max abs qacc: {max_abs_qacc:.1f}")
    print(f"  Termination: {'STABLE' if max_abs_qacc < cfg.get('max_abs_qacc', 50000) else 'UNSTABLE'}")


def sanity_policy_dummy(cfg, args):
    """Load ONNX, run one inference with dummy zero obs, check output."""
    policy = load_onnx_policy(cfg["policy_path"])
    inspect_policy_io(policy, cfg["num_obs"], cfg["num_actions"])

    # Dummy inference
    obs = np.zeros((1, cfg["num_obs"]), dtype=np.float32)
    action = policy_infer(policy, obs)[0]
    print(f"\n--- Policy Dummy Inference ---")
    print(f"  Input:  zeros(1, {cfg['num_obs']})")
    print(f"  Output: shape={action.shape}, mean={action.mean():.4f}, max={action.max():.4f}, min={action.min():.4f}")
    print(f"  Finite: {np.isfinite(action).all()}")


# ==============================================================================
# Video saving
# ==============================================================================

def _save_video(frames, output_path, fps=50):
    """Save list of RGB frames to H.264 MP4 video via ffmpeg pipe."""
    import subprocess
    if not frames:
        print("Warning: No frames to save")
        return
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"Video saved: {output_path} ({len(frames)} frames)")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Q1 Goalkeeper MuJoCo sim2sim")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    parser.add_argument("--video-out", type=str, default=None, help="Path for output MP4 video")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument("--duration", type=float, default=None, help="Override simulation duration (seconds)")
    parser.add_argument("--sanity-zero-action", action="store_true", help="Run zero-action stability test")
    parser.add_argument("--sanity-policy-dummy", action="store_true", help="Run dummy ONNX inference only")
    parser.add_argument("--no-ball-launch", action="store_true", help="Keep ball stationary at init position")
    parser.add_argument("--ball-launch-delay", type=float, default=5.0,
                        help="Freeze ball at spawn for N seconds (physics-only, independent of GK wrapper).")
    parser.add_argument("--standby-policy", type=str, default=None,
                        help="Optional ready-stand ONNX used while waiting for a ball-state trigger.")
    parser.add_argument("--activation-mode", type=str, default="ball_state_trigger",
                        choices=["ball_state_trigger", "always_on"],
                        help="GK activation mode: ball_state_trigger (default) detects launch from ball state; "
                             "always_on runs policy from t=0 (OOD baseline).")
    parser.add_argument("--gk-visible-after-trigger", type=str, default="train_timing",
                        choices=["train_timing", "immediate"],
                        help="Ball visibility after trigger: train_timing keeps vanish_steps; immediate shows ball right away.")
    parser.add_argument("--trigger-speed-threshold", type=float, default=0.3,
                        help="[ball_state_trigger] Min ball speed (m/s) for launch detection.")
    parser.add_argument("--trigger-toward-speed-threshold", type=float, default=0.2,
                        help="[ball_state_trigger] Min velocity toward goal (m/s).")
    parser.add_argument("--trigger-displacement-threshold", type=float, default=0.02,
                        help="[ball_state_trigger] Min ball displacement from init (m).")
    parser.add_argument("--trigger-debounce-steps", type=int, default=2,
                        help="[ball_state_trigger] Consecutive steps trigger condition must hold.")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["A_no_ball", "B_ball", "C_zero_action", "D_fixed_ball"],
                        help="Ablation mode")
    parser.add_argument("--fixed-ball-feat", type=str, default="4.0,0.0,-0.045",
                        help="Comma-separated ball_feature [x,y,z] for ablation D")
    parser.add_argument("--shot-mode", type=int, default=-1,
                        help="Ball shot mode 0-5 (matches training 6 modes), -1=use config shot_init_pos/vel")
    parser.add_argument("--zero-ball-obs", action="store_true",
                        help="Force ball_feature=0 in all obs frames (Task 3B: isolate ball obs effect)")
    parser.add_argument("--debug_zero_action", action="store_true",
                        help="Skip policy inference and keep target at init/default standing pose")
    parser.add_argument("--debug_policy_no_apply", action="store_true",
                        help="Run policy and record actions, but apply init/default standing target")
    parser.add_argument("--debug_apply_isaac_catchstep", action="store_true",
                        help="Apply IsaacGym catchstep > startstep target override to init_dof_pos")
    parser.add_argument("--debug_qacc_log", action="store_true",
                        help="Print a +/-10 physics-step window around the first qacc spike")
    parser.add_argument("--debug_qacc_threshold", type=float, default=None,
                        help="qacc threshold for debug_qacc_log; default=max_abs_qacc")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve config path
    config_path = args.config
    if not os.path.isabs(config_path):
        # Try relative to cwd, then relative to this script
        if not os.path.isfile(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, config_path)

    cfg = read_conf(config_path)

    if args.duration is not None:
        cfg["simulation_duration"] = args.duration

    # Print key config
    print("=" * 60)
    print("Q1 Goalkeeper MuJoCo sim2sim")
    print("=" * 60)
    print(f"  XML:        {cfg['xml_path']}")
    print(f"  Policy:     {cfg['policy_path']}")
    print(f"  sim_dt:     {cfg['simulation_dt']}")
    print(f"  decimation: {cfg['control_decimation']}")
    print(f"  policy_dt:  {cfg['simulation_dt'] * cfg['control_decimation']}")
    print(f"  obs dim:    {cfg['num_obs']} = {cfg['frame_stack']} x {cfg['num_single_obs']}")
    print(f"  action dim: {cfg['num_actions']}")
    print(f"  joints:     {len(cfg['joint_names'])} (first={cfg['joint_names'][0]}, last={cfg['joint_names'][-1]})")
    print(f"  shot pos:   {cfg.get('shot_init_pos')}")
    print(f"  shot vel:   {cfg.get('shot_init_vel')}")
    print()

    if args.sanity_zero_action:
        sanity_zero_action(cfg, args)
    elif args.sanity_policy_dummy:
        sanity_policy_dummy(cfg, args)
    else:
        run_mujoco(cfg, args)


if __name__ == "__main__":
    main()
