# Q1 Goalkeeper MuJoCo sim2sim 评估指南

## 当前推荐配置 (2026-06-06 更新)

**最佳组合：B6 XML + contact DR checkpoint + dt=0.002 decimation=10**

| 组件 | 推荐 | 说明 |
|---|---|---|
| XML | `q1_abi_B6_g1_joint.xml` | armature=0.004, damping=0.52, G1 碰撞，默认配置 |
| Config | `q1_goalkeeper_mujoco_config.yaml` | dt=0.002, decimation=10 → policy_dt=0.02 = 50Hz |
| **Checkpoint** | **contact DR** | `stand_urdf_5_014_contact_dr/model_750.pt` ✅ 双边 150 |
| Ball | mass=0.1, radius=0.1, friction=0.5 | 对齐训练 ball.urdf |

## 关键配置：仿真频率对齐

| 参数 | 值 | 说明 |
|---|---|---|
| `simulation_dt` | **0.002** | 500Hz MuJoCo physics（200Hz 下 body-ground contact 不稳定） |
| `control_decimation` | **10** | 每 10 个 physics step 调用一次 policy |
| **policy_dt** | **0.02 = 50Hz** | 对齐 IsaacGym 训练频率 |
| IsaacGym 训练 | dt=0.005, decimation=4 | 也是 0.02 = 50Hz |

**注意**：MuJoCo 用更小的 physics dt（0.002 vs 0.005）来获得更稳定的 body-ground contact 数值积分，通过增大 decimation 保持 policy 频率一致。不要用 dt=0.005 + decimation=4——会导致 zero-action qacc 爆炸。

## Contact DR 验证结果

| iter | mode 0 (左) | mode 1 (右) |
|---|---|---|
| 250 | 250 ✅ | 250 ✅ |
| 450 | 250 ✅ | 250 ✅ |
| 550 | 250 ✅ | 250 ✅ |
| 750 | 250 ✅ | 250 ✅ |
| 1050 | 250 ✅ (6/6) | 250 ✅ |
| 1250 | 250 ✅ (6/6) | 250 ✅ |
| 1350 | 250 ✅ (6/6) | 250 ✅ |

Contact DR 全线双边 150 timeout，是首个在 MuJoCo 双侧稳定的训练配置。

### 训练命令
```bash
cd legged_gym/legged_gym/scripts
./train_q1_contact_dr.sh
```

## 完整流程

```
training checkpoint (.pt)
  → ① ONNX 导出 (tools/export_q1_goalkeeper_onnx.py)
  → ② 诊断验证 (ONNX一致性 / obs对比 / joint mapping)
  → ③ MuJoCo sim2sim 消融 (C / A / B / D)
  → ④ 视频导出
```

## 文件结构

```
scripts/
├── q1_goalkeeper_mujoco_sim2sim.py       # 主 runner (支持 6 种 shot mode)
├── q1_goalkeeper_mujoco_config.yaml       # 默认配置 (dt=0.002, decimation=10)
├── q1_abi_B6_g1_joint.xml                 # ★ 推荐: armature+damping + G1 碰撞
├── q1_22dof_goalkeeper_ball_F1_body_fric10.xml  # F1: 球对齐 + body fric=1.0
├── dump_mujoco_friction.py                # MuJoCo friction dump 工具
└── outputs/                               # 视频输出目录

tools/
├── export_q1_goalkeeper_onnx.py           # ① checkpoint → ONNX
├── diagnose_onnx_vs_torch.py              # ② ONNX vs PyTorch 一致性
├── diagnose_reset_obs.py                  # ② PhysX vs MuJoCo reset obs
└── diagnose_action_joint_mapping.py       # ② Action→Joint 映射验证
```

---

## ① ONNX 导出

```bash
# Non-DR checkpoint
python tools/export_q1_goalkeeper_onnx.py \
    --checkpoint legged_gym/logs/q1/stand_urdf_5_014/model_2000.pt \
    --output legged_gym/logs/q1/exported/policies/goalkeeper_2000.onnx

# DR checkpoint
python tools/export_q1_goalkeeper_onnx.py \
    --checkpoint legged_gym/logs/q1/stand_urdf_5_014_rand_hard/model_600.pt \
    --output legged_gym/logs/q1/exported/policies/goalkeeper_rand_hard_600.onnx
```

---

## ② 诊断工具 (首次验证，后续无需重复)

```bash
python tools/diagnose_onnx_vs_torch.py     # ONNX ≡ PyTorch, diff < 3e-6
MUJOCO_GL=egl python tools/diagnose_reset_obs.py       # 6 段 obs 全部 diff=0
MUJOCO_GL=egl python tools/diagnose_action_joint_mapping.py  # 22/22 OK
```

---

## ③ MuJoCo sim2sim 消融

### 基础命令

```bash
MUJOCO_GL=egl python scripts/q1_goalkeeper_mujoco_sim2sim.py \
    --config scripts/q1_goalkeeper_mujoco_config.yaml \
    --headless
```
默认 duration=3.0s（150 control steps at 50Hz），可加 `--duration 3.0` 覆盖。

### 四种消融模式

| Flag | Policy | 球 | 说明 |
|---|---|---|---|
| `--ablation C_zero_action` | 不加载 | 固定 | PD 锁默认位姿 |
| `--ablation A_no_ball --no-ball-launch` | 加载 | 固定 | policy 纯站立 |
| `--ablation B_ball` | 加载 | 发射 | policy 守门 |
| `--ablation D_fixed_ball --fixed-ball-feat "x,y,z"` | 加载 | 特征冻结 | 排除球状态影响 |

### 6 种射门模式 (--shot-mode)

| mode | 方向 | 高度 | start_y | start_z |
|---|---|---|---|---|
| 0 | 右 | 中低 | +0.455 | 0.52 |
| 1 | 左 | 中低 | -0.455 | 0.52 |
| 2 | 右 | 高 | +0.325 | 0.91 |
| 3 | 左 | 高 | -0.325 | 0.91 |
| 4 | 右 | 低 | +0.455 | 0.135 |
| 5 | 左 | 低 | -0.455 | 0.135 |

```bash
# 指定射门模式
--shot-mode 0
```

### 完整测试模板

```bash
# 生成配置
python3 -c "
import yaml, os
cfg = yaml.safe_load(open('scripts/q1_goalkeeper_mujoco_config.yaml'))
cfg['policy_path'] = 'legged_gym/logs/q1/exported/policies/goalkeeper_rand_hard_600.onnx'
cfg['xml_path'] = 'scripts/q1_abi_B6_g1_joint.xml'
yaml.dump(cfg, open('/tmp/test.yaml','w'))
"

# 跑所有模式 + shot mode 0
for mode in C_zero_action A_no_ball B_ball; do
  case $mode in
    A_no_ball) extra="--no-ball-launch" ;;
    B_ball) extra="--shot-mode 0" ;;
    C_zero_action) extra="--no-ball-launch" ;;
  esac
  MUJOCO_GL=egl python scripts/q1_goalkeeper_mujoco_sim2sim.py \
    --config /tmp/test.yaml --duration 3.0 --headless --ablation $mode $extra
done
```

---

## ④ 视频导出

```bash
MUJOCO_GL=egl python scripts/q1_goalkeeper_mujoco_sim2sim.py \
    --config /tmp/test.yaml --duration 3.0 --headless \
    --ablation B_ball --shot-mode 0 \
    --video-out scripts/outputs/video.mp4
```

---

## B6 XML 配置详解

B6 (`q1_abi_B6_g1_joint.xml`) 包含：

| 特性 | 值 | 来源 |
|---|---|---|
| armature | 0.004 | 对齐训练 URDF |
| damping | 0.52 | G1-style joint damping |
| frictionloss | 0.65 | G1-style friction loss |
| 脚底碰撞 | 8 个 sphere (contype=1) | URDF + G1 style |
| 肩部碰撞 | 4 个 cylinder (contype=1) | G1 style |
| 身体碰撞 | 无 (仅 visual mesh) | G1 style |
| 球 | mass=0.1, radius=0.1, friction=0.5 | 对齐训练 ball.urdf |
| 地面 | condim=3, slide friction=1.0 | 对齐训练 terrain |
| 自碰 | 0 (bitmask contype=8 conaffinity=6) | 对齐训练 self_collisions=1 |

---

## 跨 Checkpoint 趋势 (B6 XML, shot mode 0)

| 测试 | Non-DR 9500 | Hard DR 100 | Hard DR 600 | Hard DR 700 |
|---|---|---|---|---|
| C_zero | 179 | — | 179 | — |
| A_no_ball | 56 | 250 | 250 | — |
| B_ball mode 0 | 58 | 250 | 250 | 250 |
| B_ball mode 1 | 250 | — | 91 | 78 |

## 关键发现总结

1. **ONNX/obs/joint mapping 验证通过**
2. **B6 (armature+damping) 是必要基础修复** — qacc 从 93k 降到 14，zero-action 从 127→179
3. **Hard DR 效果远超 weak DR 和 non-DR** — 宽范围随机化对 sim2sim 迁移至关重要
4. **Policy 有左/右不对称性** — 不同训练阶段偏好不同侧
5. **"150 timeout" 不等于守门成功** — z<0.3 机器人实际已失稳，只是 body-ground 撑着
6. **dt=0.002 + decimation=10 = policy_dt 0.02** — 对齐训练 50Hz，200Hz 下 MuJoCo 稳定性问题通过 500Hz physics 绕过
