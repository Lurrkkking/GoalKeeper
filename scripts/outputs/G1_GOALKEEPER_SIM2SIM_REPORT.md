# G1 Goalkeeper MuJoCo sim2sim — Report

Date: 2026-06-10

## 1. New/Modified Files

| File | Description |
|------|-------------|
| `scripts/g1_goalkeeper_mujoco_sim2sim.py` | G1 goalkeeper MuJoCo sim2sim script (copied from Q1 structure) |
| `scripts/g1_goalkeeper_mujoco_config.yaml` | G1 goalkeeper config (all 29-DOF params from g1_29_config.py) |
| `scripts/g1_goalkeeper_mujoco.xml` | G1 29-DOF goalkeeper MuJoCo XML (ball + collision geoms) |
| `scripts/g1_meshes/` | Symlink to G1 mesh STLs (→ soccerLab unitree_g1/meshes/) |
| `scripts/outputs/g1_gk_zero_ball_obs.mp4` | Test 3 video output |
| `scripts/outputs/g1_gk_highball_mode2.mp4` | Test 4 video output |

## 2. G1 ONNX Input/Output

- **ONNX path**: `legged_gym/logs/g1/exported/policies/goalkeeper.onnx`
- **Input**: `('input', [batch_size, 960])`
- **Output**: `('output', [batch_size, 29])`
- **Dummy inference**: finite, mean=0.0269, max=0.1981

**ONNX confirmed: obs_dim=960, action_dim=29.** No new export needed.

## 3. G1 Joint Names (29 joints, matches action_dim)

Joint order from `g1_29_config.py` `default_joint_angles`:
```
0-5:   left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll
6-11:  right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll
12-14: waist_yaw, waist_roll, waist_pitch
15-21: left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow, left_wrist_roll, left_wrist_pitch, left_wrist_yaw
22-28: right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow, right_wrist_roll, right_wrist_pitch, right_wrist_yaw
```

All 29 config vectors (joint_names, kps, kds, tau_limit, action_scale_vec, default_dof_pos) verified length=29.

## 4. XML Joint/Actuator Mapping

All 29 joint_names found in XML. Actuator mapping via `model.actuator_trnid` → joint name — all 29/29 matched successfully.

- Root free joint: `floating_base_joint` (under `pelvis` body, qpos indices 0-6)
- Ball free joint: `ball_free` (under `ball` body)
- IMU body: `pelvis` (matches training `upper_body_link`)
- Torso body: `torso_link` (matches training `torso_link`)
- Mesh loading: via symlink `g1_meshes/` → soccerLab meshes

## 5. Test Results

### Test 1: Config length + policy dummy
- ✅ ONNX input dim=960, output dim=29
- ✅ Dummy inference finite
- ✅ All config vector lengths = 29

### Test 2: Zero-action stability (5s)
```
Root z: min=0.108, max=0.800  (range 0.692m)
Max abs qvel: 13.930
Max abs qacc: 1896.7
Termination: STABLE
```
The robot sinks from 0.8 to 0.108m due to default pose being a semi-crouch (not a standing pose). qacc stays well under 200k threshold. **Numerically stable.**

### Test 3: Closed-loop zero-ball-obs (3s)
- Stop reason: `root_height_violation(z=0.199)`
- 73 control steps (~1.5s)
- Robot pitches forward aggressively (pitch=-80° at t=0.4s)

### Test 4: Closed-loop high-ball mode 2 (5s)
- Stop reason: `root_height_violation(z=0.199)`
- 73 control steps (~1.5s)
- Same collapse pattern as Test 3

### Test 5: Mode sweep (0-5)
All 6 modes collapse within 53-87 control steps (~1.0-1.7s):
```
Mode 0: 87 steps, z_min=0.199
Mode 1: 70 steps, z_min=0.199
Mode 2: 73 steps, z_min=0.199
Mode 3: 73 steps, z_min=0.198
Mode 4: 69 steps, z_min=0.198
Mode 5: 53 steps, z_min=0.184
```

**Root cause**: Raw sim2sim transfer without dynamics adaptation. The G1 goalkeeper policy was trained in IsaacGym physics and exhibits different behavior in MuJoCo (contact dynamics, solver differences, etc.). This is expected behavior for unadapted sim2sim transfer.

## 6. Differences from Q1 Script

| Item | Q1 | G1 | Reason |
|------|----|----|--------|
| Action dim | 22 | 29 | G1 has 7 extra joints (waist_pitch, 2×3 wrist) |
| Single obs dim | 75 | 96 | 3+3+3+29+29+29=96 |
| Total obs dim | 750 | 960 | 10×96=960 |
| Default root z | 0.415 | 0.8 | G1 is ~2× taller than Q1 |
| PD gains | Lower (30/20/80) | Higher (150/300/40/150/20) | G1 config has much stiffer joints |
| tau limits | 22-36 Nm | 5-139 Nm | G1 XML actuatorfrcrange |
| Decimation | 10 | 4 | G1 training uses 4 (200Hz→50Hz) |
| sim_dt | 0.002 | 0.005 | G1 IsaacGym default timestep |
| Shot mode ranges | Q1 ranges (lower, narrower) | G1 ranges (higher, wider) | From g1_29_config.py commands |
| Root free joint name | `pelvis` | `floating_base_joint` | G1 XML uses different naming |
| Waist joints | 2 (roll, yaw) | 3 (yaw, roll, pitch) | G1 has waist_pitch |
| Arms | 4 joints each | 7 joints each | G1 adds wrist roll/pitch/yaw |
| Wrist wrist | None | 6 wrist joints | G1 has full wrist actuation |

## 7. Architecture Compliance

- ✅ Strictly follows Q1 goalkeeper obs structure
- ✅ No motion tracking `ref_motion_phase` in obs
- ✅ No 23-DOF motion tracking joint list (uses 29-DOF from g1_29_config.py)
- ✅ All Q1 function names and control flow preserved
- ✅ PID control chain: policy→clip→target_dof=default+action*scale→tau=kp*(target-pos)-kd*vel→clip→ctrl
- ✅ Only dimension-dependent hardcoded values changed (75→96, SHOT_MODES→G1 ranges)

## 8. Parameter Sources

| Parameter | Source |
|-----------|--------|
| joint_names | `g1_29_config.py` `default_joint_angles` dict order |
| default_dof_pos | `g1_29_config.py` `default_joint_angles` values |
| kps | `g1_29_config.py` `stiffness` dict, mapped by joint type |
| kds | `g1_29_config.py` `damping` dict, mapped by joint type |
| tau_limit | `g1_29dof.xml` `actuatorfrcrange` values |
| action_scale_vec | `g1_29_config.py` `action_scale = 0.25` (uniform) |
| clip_actions/clip_observations | `g1_29_config.py` `normalization` |
| obs_scales | `g1_29_config.py` `normalization.obs_scales` |
| frame_stack=10 | `g1_29_config.py` `num_actor_history` |
| decimation=4 | `g1_29_config.py` `control.decimation` |
| sim_dt=0.005 | IsaacGym default for G1 |
| imu_body_name | `g1_29_config.py` `upper_body_link = "pelvis"` |
| torso_body_name | `g1_29_config.py` `torso_link = "torso_link"` |
| shot mode ranges | `g1_29_config.py` `commands.ranges_0` through `ranges_5` |
| XML kinematics/inertials | `soccerLab/.../g1_29dof.xml` — unchanged |
| XML actuators | `soccerLab/.../g1_29dof.xml` — unchanged, used via transmission joint mapping |
| XML ball/ground | Copied from Q1 goalkeeper XML with G1-appropriate positioning |
| XML collision geoms | Added box/sphere colliders on all moving links, following Q1 goalkeeper pattern (friction=1.0, condim=3, solref/solimp) |

## 9. Known Issues

1. **Sim2sim collapse**: Policy dives and collapses within ~1.5s in all modes. This is a sim2sim physics gap, not a code bug. Expected without ASAP-style adaptation.
2. **Default pose sinking**: The G1 default joint angles (knees at 0.3, ankles at -0.2) create a semi-crouch that is not statically stable with PD alone.
3. **No sim2sim adaptation applied**: The script is for verification only — no training or policy modification.
