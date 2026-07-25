# Q1 Goalkeeper MuJoCo sim2sim — 参数参考

> 用于与其他仿真/真机对齐参数。
> 策略权重: `goalkeeper.onnx` (Q1 goalkeeper checkpoint)
> XML: `q1_goalkeeper.xml`

---

## 1. 策略 I/O

| 参数 | 值 |
|------|-----|
| 观测维度 (num_obs) | **750** = 10 frame_stack × 75 single_obs |
| 动作维度 (num_actions) | **22** |
| 策略频率 | **50 Hz** (policy_dt = 0.02s) |
| 仿真频率 | **200 Hz** (sim_dt = 0.005s, decimation = 4，与 IsaacGym 训练对齐) |
| clip_observations | ±100.0 |
| clip_actions | ±100.0 |

### Single obs 结构 (75 dim)

```
[0:3]   ball_feature          — base-frame 球相对位置 (未缩放)
[3:6]   ang_vel_base × 0.25   — base-frame 角速度
[6:9]   projected_gravity     — base-frame 重力方向
[9:31]  (dof_pos - default) × 1.0
[31:53] dof_vel × 0.05
[53:75] last_action           — 上一帧的策略输出
```

---

## 2. 关节参数 (22 DOF, 按策略 action 顺序)

| # | 关节名 | default_pos | kp | kd | τ_limit (Nm) | action_scale |
|---|--------|-------------|------|------|------|------|
| 0 | left_hip_pitch_joint | -0.087 | 30 | 1.5 | 36 | 0.25 |
| 1 | left_hip_roll_joint | 0.0 | 30 | 1.5 | 36 | 0.12 |
| 2 | left_hip_yaw_joint | 0.0 | 30 | 1.5 | 36 | 0.10 |
| 3 | left_knee_joint | 0.175 | 30 | 1.5 | 36 | 0.25 |
| 4 | left_ankle_pitch_joint | -0.087 | 20 | 1.0 | 22 | 0.25 |
| 5 | left_ankle_roll_joint | 0.0 | 20 | 1.0 | 22 | 0.10 |
| 6 | right_hip_pitch_joint | -0.087 | 30 | 1.5 | 36 | 0.25 |
| 7 | right_hip_roll_joint | 0.0 | 30 | 1.5 | 36 | 0.12 |
| 8 | right_hip_yaw_joint | 0.0 | 30 | 1.5 | 36 | 0.10 |
| 9 | right_knee_joint | 0.175 | 30 | 1.5 | 36 | 0.25 |
| 10 | right_ankle_pitch_joint | -0.087 | 20 | 1.0 | 22 | 0.25 |
| 11 | right_ankle_roll_joint | 0.0 | 20 | 1.0 | 22 | 0.10 |
| 12 | waist_roll_joint | 0.0 | 80 | 2.0 | 36 | 0.25 |
| 13 | waist_yaw_joint | 0.0 | 80 | 2.0 | 36 | 0.25 |
| 14 | left_shoulder_pitch_joint | 0.0 | 20 | 1.0 | 22 | 0.50 |
| 15 | left_shoulder_roll_joint | 0.0 | 20 | 1.0 | 22 | 0.50 |
| 16 | left_shoulder_yaw_joint | 0.0 | 20 | 1.0 | 22 | 0.50 |
| 17 | left_elbow_joint | 0.0 | 20 | 1.0 | 22 | 0.50 |
| 18 | right_shoulder_pitch_joint | 0.0 | 20 | 1.0 | 22 | 0.50 |
| 19 | right_shoulder_roll_joint | 0.0 | 20 | 1.0 | 22 | 0.50 |
| 20 | right_shoulder_yaw_joint | 0.0 | 20 | 1.0 | 22 | 0.50 |
| 21 | right_elbow_joint | 0.0 | 20 | 1.0 | 22 | 0.50 |

### 关节分组 kp/kd 来源

| 组 | 关节 | kp | kd |
|----|------|------|------|
| hip | hip_pitch/roll/yaw | 30 | 1.5 |
| knee | knee | 30 | 1.5 |
| ankle | ankle_pitch/roll | 20 | 1.0 |
| waist | waist_roll/yaw | 80 | 2.0 |
| shoulder | shoulder_pitch/roll/yaw | 20 | 1.0 |
| elbow | elbow | 20 | 1.0 |

---

## 3. MuJoCo XML 关节物理参数

所有数值来自 `q1_goalkeeper.xml`。

Q1 XML 使用 per-joint 直接赋值（非 class default），所有关节 **damping=0.52, armature=0.004**。

### 关节限位 (range) 和力矩限位 (actuatorfrcrange)

| 关节 | range (rad) | frcrange (Nm) |
|------|-------------|----------------|
| left_hip_pitch | [-3.0543, 1.5708] | ±36 |
| left_hip_roll | [-0.6981, 1.5708] | ±36 |
| left_hip_yaw | [-1.5708, 1.5708] | ±36 |
| left_knee | [0.0, 2.4435] | ±36 |
| left_ankle_pitch | [-0.7854, 0.4363] | ±22 |
| left_ankle_roll | [-0.3491, 0.3491] | ±22 |
| right_hip_pitch | [-3.0543, 1.5708] | ±36 |
| right_hip_roll | [-1.5708, 0.6981] | ±36 |
| right_hip_yaw | [-1.5708, 1.5708] | ±36 |
| right_knee | [0.0, 2.4435] | ±36 |
| right_ankle_pitch | [-0.7854, 0.4363] | ±22 |
| right_ankle_roll | [-0.3491, 0.3491] | ±22 |
| waist_roll | [-0.2618, 0.2618] | ±36 |
| waist_yaw | [-1.5708, 1.5708] | ±36 |
| left_shoulder_pitch | [-3.1416, 1.5708] | ±22 |
| left_shoulder_roll | [-0.0873, 2.7925] | ±22 |
| left_shoulder_yaw | [-1.5708, 1.5708] | ±22 |
| left_elbow | [-0.8727, 1.6581] | ±22 |
| right_shoulder_pitch | [-3.1416, 1.5708] | ±22 |
| right_shoulder_roll | [-2.7925, 0.0873] | ±22 |
| right_shoulder_yaw | [-1.5708, 1.5708] | ±22 |
| right_elbow | [-0.8727, 1.6581] | ±22 |

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

| 参数 | 身体碰撞 (box) | 脚部碰撞 (sphere) |
|------|--------------|------------------|
| contype | 8 | 1 |
| conaffinity | 6 | 6 |
| condim | 3 | 3 |
| friction | [1.0, 0.005, 0.0001] | [1.0, 0.01, 0.001] |
| solref | [0.04, 1] | [0.008, 1] |
| solimp | [0.2, 0.7, 0.001] | [0.9, 0.95, 0.001] |

---

## 5. 仿真时间参数

| 参数 | 值 |
|------|-----|
| simulation_dt | 0.005s (200 Hz，与 IsaacGym 训练对齐) |
| control_decimation | 4 |
| policy_dt | 0.02s (50 Hz) |
| solver | Newton (mjSOL_NEWTON) |
| solver_iterations | 100 |
| ls_iterations | 50 |
| init_root_pos | [0.0, 0.0, 0.415] |
| init_root_quat | [1.0, 0.0, 0.0, 0.0] (wxyz) |
| imu_body | pelvis |
| torso_body | torso_link |
