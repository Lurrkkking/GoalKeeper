# G1 Goalkeeper MuJoCo sim2sim — 参数参考

> 用于与其他仿真/真机对齐参数。
> 策略权重: `goalkeeper_official.pt` (iter 25600, 2026-05-26)
> ONNX 导出: `goalkeeper_official.onnx` (input 960, output 29)

---

## 1. 策略 I/O

| 参数 | 值 |
|------|-----|
| 观测维度 (num_obs) | **960** = 10 frame_stack × 96 single_obs |
| 动作维度 (num_actions) | **29** |
| 策略频率 | **50 Hz** (policy_dt = 0.02s) |
| 仿真频率 | **200 Hz** (sim_dt = 0.005s, decimation = 4) |
| clip_observations | ±100.0 |
| clip_actions | ±100.0 |

### Single obs 结构 (96 dim)

```
[0:3]   ball_feature          — base-frame 球相对位置 (未缩放)
[3:6]   ang_vel_base × 0.25   — base-frame 角速度
[6:9]   projected_gravity     — base-frame 重力方向
[9:38]  (dof_pos - default) × 1.0
[38:67] dof_vel × 0.05
[67:96] last_action           — 上一帧的策略输出
```

---

## 2. 关节参数 (29 DOF, 按策略 action 顺序)

| # | 关节名 | default_pos | kp | kd | τ_limit (Nm) | action_scale |
|---|--------|-------------|------|------|------|------|
| 0 | left_hip_pitch_joint | -0.1 | 150 | 2.0 | 88 | 0.25 |
| 1 | left_hip_roll_joint | 0.2 | 150 | 2.0 | 88 | 0.25 |
| 2 | left_hip_yaw_joint | 0.0 | 150 | 2.0 | 88 | 0.25 |
| 3 | left_knee_joint | 0.3 | 300 | 4.0 | 139 | 0.25 |
| 4 | left_ankle_pitch_joint | -0.2 | 40 | 2.0 | 50 | 0.25 |
| 5 | left_ankle_roll_joint | -0.2 | 40 | 2.0 | 50 | 0.25 |
| 6 | right_hip_pitch_joint | -0.1 | 150 | 2.0 | 88 | 0.25 |
| 7 | right_hip_roll_joint | -0.2 | 150 | 2.0 | 88 | 0.25 |
| 8 | right_hip_yaw_joint | 0.0 | 150 | 2.0 | 88 | 0.25 |
| 9 | right_knee_joint | 0.3 | 300 | 4.0 | 139 | 0.25 |
| 10 | right_ankle_pitch_joint | -0.2 | 40 | 2.0 | 50 | 0.25 |
| 11 | right_ankle_roll_joint | 0.2 | 40 | 2.0 | 50 | 0.25 |
| 12 | waist_yaw_joint | 0.0 | 150 | 2.0 | 88 | 0.25 |
| 13 | waist_roll_joint | 0.0 | 150 | 2.0 | 50 | 0.25 |
| 14 | waist_pitch_joint | 0.0 | 150 | 2.0 | 50 | 0.25 |
| 15 | left_shoulder_pitch_joint | 0.0 | 150 | 2.0 | 25 | 0.25 |
| 16 | left_shoulder_roll_joint | 0.5 | 150 | 2.0 | 25 | 0.25 |
| 17 | left_shoulder_yaw_joint | 0.0 | 150 | 2.0 | 25 | 0.25 |
| 18 | left_elbow_joint | 1.2 | 150 | 2.0 | 25 | 0.25 |
| 19 | left_wrist_roll_joint | 0.0 | 20 | 0.5 | 25 | 0.25 |
| 20 | left_wrist_pitch_joint | 0.0 | 20 | 0.5 | 5 | 0.25 |
| 21 | left_wrist_yaw_joint | 0.0 | 20 | 0.5 | 5 | 0.25 |
| 22 | right_shoulder_pitch_joint | 0.0 | 150 | 2.0 | 25 | 0.25 |
| 23 | right_shoulder_roll_joint | -0.5 | 150 | 2.0 | 25 | 0.25 |
| 24 | right_shoulder_yaw_joint | 0.0 | 150 | 2.0 | 25 | 0.25 |
| 25 | right_elbow_joint | 1.2 | 150 | 2.0 | 25 | 0.25 |
| 26 | right_wrist_roll_joint | 0.0 | 20 | 0.5 | 25 | 0.25 |
| 27 | right_wrist_pitch_joint | 0.0 | 20 | 0.5 | 5 | 0.25 |
| 28 | right_wrist_yaw_joint | 0.0 | 20 | 0.5 | 5 | 0.25 |

### 关节分组 kp/kd 来源（`g1_29_config.py` 训练配置）

| 组 | 关节 | kp | kd |
|----|------|------|------|
| hip | hip_pitch/roll/yaw | 150 | 2 |
| knee | knee | 300 | 4 |
| ankle | ankle_pitch/roll | 40 | 2 |
| waist | waist_yaw/roll/pitch | 150 | 2 |
| shoulder | shoulder_pitch/roll/yaw | 150 | 2 |
| elbow | elbow | 150 | 2 |
| wrist | wrist_roll/pitch/yaw | 20 | 0.5 |

---

## 3. MuJoCo XML 关节物理参数

所有数值来自 `g1_goalkeeper_mujoco.xml`（基于 `soccerLab/.../g1_29dof.xml`）。

### Joint defaults（按 motor class）

| class | damping | armature | frictionloss | 适用关节 |
|-------|---------|----------|-------------|---------|
| leg_motor | 0.05 | 0.01 | 0.2 | hip_pitch/roll/yaw, knee |
| ankle_motor | 0.05 | 0.01 | 0.2 | ankle_pitch/roll |
| torso_motor | 0.05 | 0.01 | 0.2 | waist_yaw/roll/pitch |
| arm_motor | 0.05 | 0.01 | 0.2 | shoulder_pitch/roll/yaw, elbow, wrist_roll |
| wrist_motor | 0.05 | 0.01 | 0.1 | wrist_pitch/yaw |

### 关节限位 (range) 和力矩限位 (actuatorfrcrange)

| 关节 | range (rad) | frcrange (Nm) |
|------|-------------|----------------|
| left_hip_pitch | [-2.5307, 2.8798] | ±88 |
| left_hip_roll | [-0.5236, 2.9671] | ±88 |
| left_hip_yaw | [-2.7576, 2.7576] | ±88 |
| left_knee | [-0.0873, 2.8798] | ±139 |
| left_ankle_pitch | [-0.8727, 0.5236] | ±50 |
| left_ankle_roll | [-0.2618, 0.2618] | ±50 |
| right_hip_pitch | [-2.5307, 2.8798] | ±88 |
| right_hip_roll | [-2.9671, 0.5236] | ±88 |
| right_hip_yaw | [-2.7576, 2.7576] | ±88 |
| right_knee | [-0.0873, 2.8798] | ±139 |
| right_ankle_pitch | [-0.8727, 0.5236] | ±50 |
| right_ankle_roll | [-0.2618, 0.2618] | ±50 |
| waist_yaw | [-2.618, 2.618] | ±88 |
| waist_roll | [-0.52, 0.52] | ±50 |
| waist_pitch | [-0.52, 0.52] | ±50 |
| left_shoulder_pitch | [-3.0892, 2.6704] | ±25 |
| left_shoulder_roll | [-1.5882, 2.2515] | ±25 |
| left_shoulder_yaw | [-2.618, 2.618] | ±25 |
| left_elbow | [-1.0472, 2.0944] | ±25 |
| left_wrist_roll | [-1.9722, 1.9722] | ±25 |
| left_wrist_pitch | [-1.6144, 1.6144] | ±5 |
| left_wrist_yaw | [-1.6144, 1.6144] | ±5 |
| right_shoulder_pitch | [-3.0892, 2.6704] | ±25 |
| right_shoulder_roll | [-2.2515, 1.5882] | ±25 |
| right_shoulder_yaw | [-2.618, 2.618] | ±25 |
| right_elbow | [-1.0472, 2.0944] | ±25 |
| right_wrist_roll | [-1.9722, 1.9722] | ±25 |
| right_wrist_pitch | [-1.6144, 1.6144] | ±5 |
| right_wrist_yaw | [-1.6144, 1.6144] | ±5 |

---

## 4. 接触 / 碰撞参数

### 球

| 参数 | 值 | 来源 |
|------|-----|------|
| mass | 0.1 kg | `ball.urdf` |
| radius | 0.1 m | `ball.urdf` |
| friction | [0.5, 0.005, 0.0001] | XML ball_geom |
| contype | 2 | XML |
| conaffinity | 13 | XML |
| condim | 6 | XML |
| solref | [0.012, 1] | XML |
| solimp | [0.75, 0.95, 0.001] | XML |

### 地面

| 参数 | 值 |
|------|-----|
| contype | 4 |
| conaffinity | 11 |
| condim | 3 |

### 机器人碰撞体

| 参数 | 值 |
|------|-----|
| 碰撞 friction | [1.0, 0.005, 0.0001] |
| 碰撞 condim | 3 |
| 碰撞 solref | [0.04, 1] |
| 碰撞 solimp | [0.2, 0.7, 0.001] |
| 脚部碰撞 friction | [1.0, 0.01, 0.001] |
| 脚部碰撞 solref | [0.008, 1] |
| 脚部碰撞 solimp | [0.9, 0.95, 0.001] |

---

## 5. 仿真时间参数

| 参数 | 值 |
|------|-----|
| simulation_dt | 0.005s (200 Hz) |
| control_decimation | 4 |
| policy_dt | 0.02s (50 Hz) |
| solver | Newton (mjSOL_NEWTON) |
| solver_iterations | 100 |
| ls_iterations | 50 |

---
