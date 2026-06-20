# GMR PKL → AMP PT 转换流程

**最后更新**: 2026-06-19

从 GVHMR 视频到 Humanoid-Goalkeeper AMP motion prior 的端到端流程。

---

## 概览

```
视频 (.mp4)
  │  GVHMR demo.py                    # 人体运动捕捉 → SMPL 参数
  ▼
hmr4d_results.pt                      # SMPL (Y-up, 全局坐标)
  │  GMR gvhmr_to_robot.py            # IK retarget → G1 机器人
  │  + dive IK config (smplx_to_g1_dive.json)
  ▼
GMR pkl                               # G1 29 DOF, Z-up, 50fps
  │  convert_gmr_pkl_to_amp_pt.py     # 格式转换 + 重采样
  ▼
AMP .pt                               # 21 DOF, 30fps, AMP MotionLib 格式
  │  verify_motion_direction.py       # IsaacGym kinematic replay
  ▼
right_dive_gmr.pt                     # 就绪，可接入 AMP 训练
```

---

## Step 1: GVHMR 人体运动提取

### 输入

- 视频文件 (`.mp4`)，静态相机，人体做扑救动作

### 命令

```bash
cd /root/autodl-tmp/GVHMR
conda run -n GVHMR python tools/demo/demo.py --video <input.mp4> -s
```

### 输出

```
outputs/demo/<video_name>/hmr4d_results.pt
```

### 关键数据

| Key | Shape | 说明 |
|-----|-------|------|
| `smpl_params_global.transl` | (T, 3) | SMPL pelvis 全局位移 (Y-up: X=lateral, Y=up, Z=depth) |
| `smpl_params_global.global_orient` | (T, 3) | 全局朝向 (axis-angle) |
| `smpl_params_global.body_pose` | (T, 63) | 身体姿态 (axis-angle, 21 joints × 3) |
| `smpl_params_incam.transl` | (T, 3) | 相机坐标系位移 (仅供参考) |

### 检查要点

- 确认 global transl X (lateral) 的 p2p ≥ 1.0m（飞扑幅度足够）
- 确认帧率正确（通常 50fps）
- 确认 global_orient 第一帧 yaw 合理

---

## Step 2: GMR Retarget → G1 机器人

### 输入

- `hmr4d_results.pt` (Step 1 输出)
- IK config JSON（用 dive 专用版）

### IK Config

飞扑专用配置: `general_motion_retargeting/ik_configs/smplx_to_g1_dive.json`

关键参数 (对比默认 `smplx_to_g1.json`):

| 参数 | 默认值 | Dive 版 | 说明 |
|------|--------|---------|------|
| `human_scale_table.pelvis` | 0.9 | **1.3** | 放大 pelvis 位移 |
| `ik_match_table1.pelvis` pos | 100 | **500** | 提高 root 跟踪权重 |
| `ik_match_table1.*_toe` pos | 100 | **5** | 降低脚部约束 |
| `ik_match_table2.*_toe` pos | 100 | **5** | 同上 |
| `ik_match_table1.torso_link` pos | 0 | **20** | 增加躯干跟踪 |

完整配置见 `smplx_to_g1_dive.json`。

### 命令

```bash
cd /root/autodl-tmp/GMR
PYTHONPATH=$PWD:$PWD/third_party xvfb-run -a conda run -n gmr \
  python scripts/gvhmr_to_robot.py \
    --gvhmr_pred_file <hmr4d_results.pt> \
    --src_fps <fps> \
    --robot unitree_g1 \
    --ik_config general_motion_retargeting/ik_configs/smplx_to_g1_dive.json \
    --save_as_pkl True \
    --save_path unitree_g1_gmr/
```

> **注意**: `--ik_config` 参数需要确保 `motion_retarget.py` 和 `gvhmr_to_robot.py` 已支持 (已在本次修改中添加)。

或者使用批处理脚本:

```bash
# 批处理版本 (headless, 无 viewer)
MUJOCO_GL=egl xvfb-run -a conda run -n gmr \
  python /tmp/retarget_gmr_batch.py \
    --gvhmr_pred <hmr4d_results.pt> \
    --ik_config <dive_config.json> \
    --output <output.pkl>
```

### 输出

```
unitree_g1_gmr/<name>.pkl
```

### GMR pkl Schema

| Key | Shape | Dtype | 说明 |
|-----|-------|-------|------|
| `fps` | scalar | float | 帧率 (50) |
| `root_pos` | (T, 3) | float64 | 全局 root 位置 (Z-up: X=前, Y=侧, Z=上) |
| `root_rot` | (T, 4) | float64 | 全局 root 旋转 (xyzw) |
| `dof_pos` | (T, 29) | float64 | 29 个关节角度 (rad) |
| `joint_vel` | (T, 29) | float32 | 29 个关节速度 (rad/s) |
| `joint_names` | (29,) | str | 关节名称列表 |

### 检查要点

- Root Y p2p ≥ 0.8m (飞扑幅度)
- Root Z 起点 ≈ 0.8m (与 dive_save config 一致)
- Quat norm ≈ 1.0
- 无 NaN/Inf
- DOF range 在 [-3.14, 3.14] 内

---

## Step 3: PKL → PT 格式转换

### 输入

- GMR pkl (Step 2 输出)
- `leftjump.pt` (参考 schema)
- `joint_id.txt` (21 DOF joint mapping)

### Converter

```bash
cd /root/autodl-tmp/Humanoid-Goalkeeper
MUJOCO_GL=egl xvfb-run -a conda run -n gmr \
  python tools/convert_gmr_pkl_to_amp_pt.py \
    --pkl <input.pkl> \
    --output legged_gym/resources/datasets/goalkeeper/<output.pt> \
    --joint-mapping legged_gym/resources/datasets/goalkeeper/joint_id.txt \
    --target-fps 30
```

### 转换逻辑

1. **加载 pkl**: joblib 读取 GMR pkl
2. **Quat 验证**: 检查 xyzw vs wxyz，确保一致
3. **FPS 重采样**: 50 → 30fps (quat 使用 SLERP，其他使用线性插值)
4. **DOF 截取**: 29 → 21 DOF (按 `joint_id.txt` 映射)
5. **速度计算**: 有限差分计算 `base_velocity`, `base_angular_velocity`
6. **Key 重命名**:

   | pkl key | pt key |
   |---------|--------|
   | `root_pos` | `base_position` |
   | `root_rot` | `base_pose` |
   | `dof_pos` | `joint_position` |
   | `joint_vel` | `joint_velocity` |
   | (计算) | `base_velocity` |
   | (计算) | `base_angular_velocity` |

7. **不生成 `link_*`**: G1 URDF 无 keyframe body，MotionLib 不读取

### 输出

```
legged_gym/resources/datasets/goalkeeper/<output.pt>
```

### AMP .pt Schema

| Key | Shape | Dtype | 说明 |
|-----|-------|-------|------|
| `base_position` | (T, 3) | float32 | 全局 root 位置 |
| `base_pose` | (T, 4) | float32 | 全局 root 旋转 (xyzw) |
| `base_velocity` | (T, 3) | float32 | 全局 root 线速度 |
| `base_angular_velocity` | (T, 3) | float32 | 全局 root 角速度 |
| `joint_position` | (T, 21) | float32 | 21 个关节角度 |
| `joint_velocity` | (T, 21) | float32 | 21 个关节速度 |

---

## Step 4: 方向验证

### IsaacGym Replay

```bash
cd /root/autodl-tmp/Humanoid-Goalkeeper
conda activate rl
python tools/verify_motion_direction.py \
  --pt legged_gym/resources/datasets/goalkeeper/<output.pt>
```

### 检查项

- Root Y delta 符号与预期一致
- 命名与方向匹配 (right → -Y, left → +Y)
- Hand world Y 在出发和到达帧合理
- IsaacGym kinematic replay 不报错

### 方向约定

```
IsaacGym: +X = forward, +Y = LEFT, -Y = RIGHT

g1_dive_save:
  target_y > 0  → target_side > 0 → left_rubber_hand   (LEFT side target)
  target_y < 0  → target_side < 0 → right_rubber_hand   (RIGHT side target)

Motion naming:
  right_* → dive to -Y (RIGHT)
  left_*  → dive to +Y (LEFT)
```

---

## Step 5: MotionLib Load Test

```bash
cd /root/autodl-tmp/Humanoid-Goalkeeper
conda activate rl
python -c "
import isaacgym, torch, sys
sys.path.insert(0, 'legged_gym')
from legged_gym.envs.g1.g1_utils import load_imitation_dataset, MotionLib

folder = 'legged_gym/resources/datasets/goalkeeper'
multidataset, mapping = load_imitation_dataset(folder, folder + '/joint_id.txt')

dof_names = [...] # 21 joint names
keyframe_names = []

ml = MotionLib(multidataset['<name>'], mapping, dof_names, keyframe_names,
               fps=30, min_dt=0.1, device='cpu', amp_obs_type='dof', num_steps=2)
obs = ml.get_expert_obs(32)
assert obs.shape == (32, 42)
print('OK')
"
```

---

## Step 6: 接入 AMP 训练 (后续)

`right_dive_gmr.pt` 放入 dataset 目录后，AMP loader 自动加载。

在 `g1_dive_save_config.py` 中启用 AMP:

```python
class amp:
    obs_type = 'dof'
    num_obs = 21 * 2    # num_steps=2 → 42 dim expert obs
    amp_coef = 0.1      # 从小权重开始
    num_steps = 2
```

在 `him_ppo.py` 中映射 motion key 到 mode:

```python
# motion_ids → motion_buffer key
# Mode 1 (right targets) → right_dive_gmr
motion_buffer["right_dive_gmr"].get_expert_obs(...)
```

---

## 常见问题

### Q: Root Y 幅度不够

检查 GVHMR source 的 global transl X p2p。如果 source < 1.0m，考虑:
- 用更大幅度的源视频
- 增加 `smplx_to_g1_dive.json` 中 `human_scale_table.pelvis`

### Q: 方向反了

检查 GVHMR source 的 global_orient 第一帧。如果 human 面朝方向与预期不同，可能需要:
- 在 GMR retarget 前旋转 global_orient
- 或镜像处理

### Q: MotionLib 加载报错

常见原因:
- .pt 缺少必需 key → 检查 converter 是否遗漏
- shape 不匹配 → 检查 joint mapping 是否正确
- 帧数太少 (< 3) → MotionLib 的 `min_dt * fps` 过滤

### Q: get_expert_obs shape 不对

确保 `num_steps=2` 且 `num_dof=21` → expert_obs shape = `(batch, 42)`

---

## 已生成的文件

| 文件 | 说明 |
|------|------|
| `right_dive_gmr.pt` | 当前就绪的 right-side dive AMP prior |
| `smplx_to_g1_dive.json` | Dive 专用 IK config |
| `tools/convert_gmr_pkl_to_amp_pt.py` | PKL→PT converter |
| `tools/verify_motion_direction.py` | IsaacGym replay 验证 |
