# GMR → AMP Motion 格式转换检查报告

**日期**: 2026-06-19  
**GMR 文件**: `/root/autodl-tmp/GMR/unitree_g1_gmr/goalkeep_level1_right.pkl`  
**AMP 参考文件**: `/root/autodl-tmp/Humanoid-Goalkeeper/legged_gym/resources/datasets/goalkeeper/leftjump.pt`

---

## 1. leftjump.pt 完整 Schema

| Key | Shape | Dtype | 单位 | 语义 |
|-----|-------|-------|------|------|
| `base_position` | (254, 3) | float32 | 米 | 全局 base 根位置 (x, y, z)，Z-up |
| `base_pose` | (254, 4) | float32 | quat xyzw | 全局 base 旋转四元数 [qx, qy, qz, qw] |
| `base_velocity` | (254, 3) | float32 | m/s | 全局 base 线速度 |
| `base_angular_velocity` | (254, 3) | float32 | rad/s | 全局 base 角速度 |
| `joint_position` | (254, 21) | float32 | rad | 21 个关节角度 |
| `joint_velocity` | (254, 21) | float32 | rad/s | 21 个关节速度 |
| `link_position` | (254, 17, 3) | float32 | 米 | 17 个 keyframe link 全局位置 |
| `link_orientation` | (254, 17, 4) | float32 | quat xyzw | 17 个 keyframe link 全局旋转 |
| `link_velocity` | (254, 17, 3) | float32 | m/s | 17 个 keyframe link 线速度 |
| `link_angular_velocity` | (254, 17, 3) | float32 | rad/s | 17 个 keyframe link 角速度 |

- **T**: 254 帧
- **FPS**: 30 (from `dataset.frame_rate = 30` in config)
- **DOF 数量**: 21 (与 `joint_id.txt` 一致)
- **Keyframe bodies**: 17 个
- **Quat 格式**: xyzw (scipy 标准)，与 GMR 一致

### 21 关节列表 (joint_id.txt)

```
 0 left_hip_pitch_joint       12 waist_yaw_joint
 1 left_hip_roll_joint        13 left_shoulder_pitch_joint
 2 left_hip_yaw_joint         14 left_shoulder_roll_joint
 3 left_knee_joint            15 left_shoulder_yaw_joint
 4 left_ankle_pitch_joint     16 left_elbow_joint
 5 left_ankle_roll_joint      17 right_shoulder_pitch_joint
 6 right_hip_pitch_joint      18 right_shoulder_roll_joint
 7 right_hip_roll_joint       19 right_shoulder_yaw_joint
 8 right_hip_yaw_joint        20 right_elbow_joint
 9 right_knee_joint
10 right_ankle_pitch_joint
11 right_ankle_roll_joint
```

**缺失的 8 个关节**（G1 有 29 DOF，但 .pt 只有 21）:
- `waist_roll_joint`
- `waist_pitch_joint`
- `left_wrist_roll_joint`, `left_wrist_pitch_joint`, `left_wrist_yaw_joint`
- `right_wrist_roll_joint`, `right_wrist_pitch_joint`, `right_wrist_yaw_joint`

> 这 8 个 DOF 在 MotionLib 中会被保持为零（因为 `joint_id.txt` 中没有对应的映射条目）。

---

## 2. goalkeep_level1_right.pkl 完整 Schema

| Key | Shape | Dtype | 单位 | 语义 |
|-----|-------|-------|------|------|
| `fps` | scalar | float | Hz | 帧率 = 50.0 |
| `root_pos` | (73, 3) | float64 | 米 | 全局 root 位置 (x, y, z)，Z-up |
| `root_rot` | (73, 4) | float64 | quat xyzw | 全局 root 旋转 [qx, qy, qz, qw] |
| `dof_pos` | (73, 29) | float64 | rad | 29 个关节角度 |
| `dof_positions` | (73, 29) | float64 | rad | 同上（别名） |
| `joint_pos` | (73, 29) | float64 | rad | 同上（别名） |
| `joint_vel` | (73, 29) | float32 | rad/s | 29 个关节速度 |
| `joint_names` | (29,) | str | — | 29 个关节名称 |
| `body_pos_w` | (73, 3) | float64 | 米 | body 位置（= root_pos） |
| `body_quat_w` | (73, 4) | float64 | quat xyzw | body 旋转（= root_rot） |

- **T**: 73 帧
- **FPS**: 50
- **Duration**: 1.46 秒
- **DOF 数量**: 29（完整 G1）
- **无 keyframe/link 数据**: 缺少 `link_position`, `link_orientation`, `link_velocity`, `link_angular_velocity`
- **无 base 速度**: 缺少 `base_velocity`, `base_angular_velocity`

### DOF 顺序（与 G1 dive_reach / dive_save config 完全一致）

```
 0-5:   left_leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
 6-11:  right_leg (同上)
12-14:  waist (yaw, roll, pitch)
15-21:  left_arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)
22-28:  right_arm (同上)
```

---

## 3. 字段差异对比

| AMP .pt Key | GMR pkl Key | 匹配？ | 说明 |
|-------------|------------|--------|------|
| `base_position` | `root_pos` | ✅ 重命名 | 语义相同，Z-up 坐标系一致 |
| `base_pose` | `root_rot` | ✅ 重命名 | 都是 xyzw 格式，无需转换 |
| `base_velocity` | — | ❌ 缺失 | 需从 `root_pos` 有限差分计算 |
| `base_angular_velocity` | — | ❌ 缺失 | 需从 `root_rot` 有限差分计算 |
| `joint_position` | `dof_pos` | ✅ 重命名 | GMR 有 29 DOF，.pt 只有 21 |
| `joint_velocity` | `joint_vel` | ✅ 重命名 | 同上 |
| `link_position` | — | ❌ 缺失 | **不需要** — G1 URDF 无 keyframe body |
| `link_orientation` | — | ❌ 缺失 | **不需要** |
| `link_velocity` | — | ❌ 缺失 | **不需要** |
| `link_angular_velocity` | — | ❌ 缺失 | **不需要** |
| — | `joint_names` | ➕ GMR 多 | 仅用于调试，.pt 不需要 |
| — | `body_pos_w` | ➕ GMR 多 | = root_pos，冗余 |
| — | `body_quat_w` | ➕ GMR 多 | = root_rot，冗余 |

---

## 4. GMR pkl 是否是 G1 Motion

**结论：是，已经是完整的 G1 retargeted motion。**

验证清单：

| 检查项 | 结果 |
|--------|------|
| DOF 数量 = 29 | ✅ 与 `num_dofs=29` 一致 |
| DOF 顺序一致 | ✅ 29 个名称完全匹配 config 中的 `default_joint_angles` 顺序 |
| Quat 格式 xyzw | ✅ `euler_from_quaternion` 期望 xyzw，GMR 输出 xyzw |
| 坐标系 Z-up | ✅ IsaacGym 用 Z-up，GMR 输出 Z-up |
| 无地面穿透 | ✅ min Z = 0.42m |
| 初始 root 高度 | ✅ 0.77m（合理守门员站姿） |
| 运动方向 | ✅ positive Y = RIGHT dive（文件名含 `right`） |
| 无 NaN/Inf | ✅ |

**注意事项**:
- FPS: GMR 输出 50fps，AMP dataset config 指定 30fps。MotionLib 使用 config 中的 `frame_rate` 参数来计算速度，不会自动从数据中读取 fps。需要在转换时重采样到 30fps，或者更新 config 为 `frame_rate = 50`。
- 轨迹很短：73 帧 @ 50fps = 1.46s，侧向位移仅 0.16m（更像小范围扑救而非大范围鱼跃）

---

## 5. 能否直接转换

**结论：可以直接转换。**

### 转换路径 A：生成 21-DOF .pt（推荐先用此方案）
- 从 GMR 29 DOF 中提取与 `joint_id.txt` 对应的 21 个关节
- 重命名 key
- 计算 base 速度
- 可选：resample 50→30fps
- 无需生成 link 数据

### 转换路径 B：生成 29-DOF .pt + 更新 joint_id.txt
- 保留全部 29 DOF
- 创建新的 `joint_id.txt`（29 行，0→28 直接映射）
- 更新 `dataset.frame_rate = 50`（或 resample 到 30）

---

## 6. 最小 Converter 脚本方案

```python
#!/usr/bin/env python3
"""Convert GMR motion pkl → Humanoid-Goalkeeper AMP .pt format."""
import joblib, torch, numpy as np
import argparse

def convert_gmr_to_amp(gmr_path, amp_path, target_fps=30, dof_mode='21'):
    """
    Args:
        gmr_path: GMR pkl path
        amp_path: output .pt path
        target_fps: target frame rate (30 for existing config)
        dof_mode: '21' to match existing joint_id.txt, '29' for full G1
    """
    with open(gmr_path, 'rb') as f:
        gmr = joblib.load(f)
    
    src_fps = gmr['fps']
    root_pos = torch.tensor(gmr['root_pos'], dtype=torch.float32)
    root_rot = torch.tensor(gmr['root_rot'], dtype=torch.float32)  # xyzw
    dof_pos_all = torch.tensor(gmr['dof_pos'], dtype=torch.float32)
    dof_vel_all = torch.tensor(gmr['joint_vel'], dtype=torch.float32)
    joint_names = list(gmr['joint_names'])
    
    # Resample if needed
    if src_fps != target_fps:
        ratio = src_fps / target_fps
        T_new = int(root_pos.shape[0] / ratio)
        indices = (torch.arange(T_new) * ratio).long().clamp(0, root_pos.shape[0]-1)
        root_pos = root_pos[indices]
        root_rot = root_rot[indices]
        dof_pos_all = dof_pos_all[indices]
        dof_vel_all = dof_vel_all[indices]
    else:
        T_new = root_pos.shape[0]
    
    # Select DOFs
    if dof_mode == '21':
        # 21 joints matching joint_id.txt
        keep_21 = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
            'waist_yaw_joint',
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
            'left_elbow_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
            'right_elbow_joint',
        ]
        indices_21 = [joint_names.index(n) for n in keep_21]
        joint_pos = dof_pos_all[:, indices_21]
        joint_vel = dof_vel_all[:, indices_21]
    else:
        joint_pos = dof_pos_all
        joint_vel = dof_vel_all
    
    # Compute base velocities via finite differences
    dt = 1.0 / target_fps
    base_vel = torch.zeros_like(root_pos)
    base_ang_vel = torch.zeros_like(root_pos)
    if T_new > 1:
        # linear vel
        base_vel[:-1] = (root_pos[1:] - root_pos[:-1]) / dt
        base_vel[-1] = base_vel[-2]
        # angular vel from quat difference
        from scipy.spatial.transform import Rotation as R
        r0 = R.from_quat(root_rot[:-1].numpy())  # scipy uses xyzw
        r1 = R.from_quat(root_rot[1:].numpy())
        # relative rotation: r1 = delta * r0 → delta = r1 * r0.inv()
        delta = r1 * r0.inv()
        ang_vel = delta.as_rotvec() / dt
        base_ang_vel[:-1] = torch.tensor(ang_vel, dtype=torch.float32)
        base_ang_vel[-1] = base_ang_vel[-2]
    
    # Build .pt dict
    amp_data = {
        'base_position': root_pos,
        'base_pose': root_rot,
        'base_velocity': base_vel,
        'base_angular_velocity': base_ang_vel,
        'joint_position': joint_pos,
        'joint_velocity': joint_vel,
        # link_* fields: NOT needed (G1 URDF has no "keyframe" bodies)
    }
    
    torch.save(amp_data, amp_path)
    print(f'Converted: {gmr_path} → {amp_path}')
    print(f'  Frames: {T_new}, FPS: {target_fps}, DOF: {joint_pos.shape[1]}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gmr_path', required=True)
    parser.add_argument('--amp_path', required=True)
    parser.add_argument('--target_fps', type=int, default=30)
    parser.add_argument('--dof_mode', choices=['21', '29'], default='21')
    args = parser.parse_args()
    convert_gmr_to_amp(args.gmr_path, args.amp_path, args.target_fps, args.dof_mode)
```

---

## 7. 当前还缺什么

| 缺口 | 状态 |
|------|------|
| GMR pkl → .pt 转换脚本 | ✅ 上方已提供方案 |
| 更多的 dive motion 数据 | ⚠️ 目前只有 1 条 1.46s 的 motion，AMP 通常需要多条 |
| `rightjump.pt` / `righthand.pt` 等 | ⚠️ AMP 按 mode (0-5) 使用不同的 motion prior，当前只有 1 条 |
| FPS 对齐 | ⚠️ GMR 输出 50fps，config 设 30fps |
| 大范围鱼跃 motion | ⚠️ 当前 displacement 只有 0.16m，可能不够 "dive" |

---

## 8. 后续接入 g1_dive_save_amp 可复用的文件/类

| 文件/类 | 作用 |
|---------|------|
| `legged_gym/envs/g1/g1_utils.py::MotionLib` | AMP motion prior 管理器，已支持 21-DOF |
| `legged_gym/envs/g1/g1_utils.py::load_imitation_dataset` | 加载整个文件夹的 .pt 文件 |
| `legged_gym/envs/base/legged_robot.py::LeggedRobot` | 在 `_init_mocap` 中初始化 `self.motions` dict |
| `rsl_rl/algorithms/him_ppo.py::HIMPPO` | AMP 判别器训练，从 `motion_buffer` 取 expert obs |
| `rsl_rl/runners/him_on_policy_runner.py` | 将 `self.env.motions` 传入 HIMPPO |
| `legged_gym/envs/g1/g1_dive_save_config.py` | Dive Save 训练 config，AMP 当前 disabled (`amp_coef=0.0`) |
| `legged_gym/resources/datasets/goalkeeper/joint_id.txt` | 21 关节映射（如果改用 29 DOF 需更新） |

**建议接入流程**:
1. 收集多条 GMR retargeted 扑救 motion（左/右/高/低 corner）
2. 用 converter 转为 .pt 格式
3. 按 mode 命名放入 `resources/datasets/goalkeeper/`
4. 在 `g1_dive_save_config.py` 设置 `amp_coef > 0` 启用 AMP
5. 调整 `amp.num_steps` 和 `dataset.frame_rate`
