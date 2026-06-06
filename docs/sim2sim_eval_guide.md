# Q1 Goalkeeper MuJoCo sim2sim 评估指南

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
├── q1_goalkeeper_mujoco_sim2sim.py     # 主 runner
├── q1_goalkeeper_mujoco_config.yaml     # 默认配置
├── q1_abi_D2.xml                        # D2 碰撞 (soft body, hard sole, bitmask)
├── q1_22dof_goalkeeper_ball_F1_body_fric10.xml  # 推荐碰撞 (球对齐 + body fric=1.0)
├── dump_mujoco_friction.py              # MuJoCo friction dump 工具
└── outputs/                             # 视频输出目录

tools/
├── export_q1_goalkeeper_onnx.py         # ① checkpoint → ONNX
├── diagnose_onnx_vs_torch.py            # ② ONNX vs PyTorch 一致性
├── diagnose_reset_obs.py                # ② PhysX vs MuJoCo reset obs
└── diagnose_action_joint_mapping.py     # ② Action→Joint 映射验证
```

---

## ① ONNX 导出

```bash
python tools/export_q1_goalkeeper_onnx.py \
    --checkpoint legged_gym/logs/q1/stand_urdf_5_014/model_2000.pt \
    --output legged_gym/logs/q1/exported/policies/goalkeeper_2000.onnx
```

支持 DR checkpoint：
```bash
python tools/export_q1_goalkeeper_onnx.py \
    --checkpoint legged_gym/logs/q1/stand_urdf_5_014_rand_weak/model_4000.pt \
    --output legged_gym/logs/q1/exported/policies/goalkeeper_rand_weak_4000.onnx
```

---

## ② 诊断工具 (首次验证，后续无需重复)

### ONNX 一致性测试

```bash
python tools/diagnose_onnx_vs_torch.py
# 预期: max_diff < 3e-6 (正常 obs), < 2e-4 (极端 ±100 obs, float32 精度极限)
```

### Reset obs 对比

```bash
MUJOCO_GL=egl python tools/diagnose_reset_obs.py
# 预期: 6 段 obs dim 全部 diff=0
```

### Action→Joint 映射

```bash
MUJOCO_GL=egl python tools/diagnose_action_joint_mapping.py
# 预期: 22/22 actuator→joint OK
```

---

## ③ MuJoCo sim2sim 消融

### 基础命令

```bash
MUJOCO_GL=egl python scripts/q1_goalkeeper_mujoco_sim2sim.py \
    --config scripts/q1_goalkeeper_mujoco_config.yaml \
    --duration 5.0 --headless
```

### 四种消融模式

| Flag | Policy | 球 | 说明 |
|---|---|---|---|
| `--ablation C_zero_action` | 不加载 | — | PD 锁默认位姿，验证静态站立 |
| `--ablation A_no_ball --no-ball-launch` | 加载 | 固定 | 验证 policy 纯站立 |
| `--ablation B_ball` | 加载 | 发射 | 验证 policy 守门 |
| `--ablation D_fixed_ball --fixed-ball-feat "x,y,z"` | 加载 | 特征冻结 | 排除球状态影响 |

### 切换 ONNX / XML

```bash
# 在 config 中指定，或创建临时 config 覆盖
python3 -c "
import yaml
cfg = yaml.safe_load(open('scripts/q1_goalkeeper_mujoco_config.yaml'))
cfg['xml_path'] = 'scripts/q1_22dof_goalkeeper_ball_F1_body_fric10.xml'
cfg['policy_path'] = 'legged_gym/logs/q1/exported/policies/goalkeeper_2000.onnx'
yaml.dump(cfg, open('/tmp/test.yaml', 'w'))
"
MUJOCO_GL=egl python scripts/q1_goalkeeper_mujoco_sim2sim.py --config /tmp/test.yaml --duration 5.0 --headless --ablation B_ball
```

### 完整消融模板

```bash
MODEL=rand_weak_4000
XML=q1_22dof_goalkeeper_ball_F1_body_fric10.xml
OUTDIR=scripts/outputs/model_${MODEL}
mkdir -p $OUTDIR

for mode in C_zero_action A_no_ball B_ball; do
  case $mode in
    A_no_ball) extra="--no-ball-launch"; out="${MODEL}_A_no_ball.mp4" ;;
    B_ball) extra=""; out="${MODEL}_B_ball.mp4" ;;
    C_zero_action) extra=""; out="${MODEL}_C_zero.mp4" ;;
  esac
  MUJOCO_GL=egl python scripts/q1_goalkeeper_mujoco_sim2sim.py \
    --config /tmp/test.yaml --duration 5.0 --headless \
    --ablation $mode $extra \
    --video-out $OUTDIR/$out
done
```

### 日志解读

```
t=0.00s | z=0.546 | roll=0.0° pitch=0.0° | qacc=30 | ncon=8
```

| 字段 | 含义 |
|---|---|
| `z` | torso (IMU) 高度，正常站立 ~0.55 |
| `roll/pitch` | 躯干倾斜角，正常 < 5° |
| `qacc` | 最大 joint 加速度，正常 < 100 |
| `ncon` | 当前活跃接触点数 |

**正常站立**：z > 0.4, roll/pitch < 10°, qacc < 1000, ncon = 8 (脚底 sphere 接触地面)

**即将摔倒**：z < 0.3, roll/pitch > 30°, qacc > 50000, ncon < 4 (脚离地)

---

## ④ 视频导出

```bash
MUJOCO_GL=egl python scripts/q1_goalkeeper_mujoco_sim2sim.py \
    --config scripts/q1_goalkeeper_mujoco_config.yaml \
    --duration 5.0 --headless \
    --ablation B_ball \
    --video-out scripts/outputs/video.mp4
```

**注意**：必须在 `MUJOCO_GL=egl` 环境下运行 headless。

---

## 碰撞 Bitmask 方案

| 类别 | contype | conaffinity | 含义 |
|---|---|---|---|
| sole (脚底 sphere) | 1 | 6 | 碰 ground(4)+ball(2) |
| ball | 2 | 13 | 碰 sole(1)+ground(4)+body(8) |
| ground | 4 | 11 | 碰 sole(1)+ball(2)+body(8) |
| body (身体 box) | 8 | 6 | 碰 ball(2)+ground(4)，不碰 robot↔robot |

## Friction 审计结论

| Contact Pair | Isaac Gym 训练 | MuJoCo 推荐 | 状态 |
|---|---|---|---|
| foot-ground | f=1.0 (rand [0.6,1.4]) | sole f=1.0 | ✅ |
| body-ground | f=1.0 (rand [0.6,1.4]) | body f=1.0, condim=3 | ✅ F1_fric10 |
| ball-ground | f≈0.5 (PhysX 默认) | ball f=0.5 | ✅ F0 |
| ball mass | 0.1 kg | 0.1 kg | ✅ F0 |
| ball radius | 0.1 m | 0.1 m | ✅ F0 |

## 跨 Checkpoint 趋势 (F1_fric10)

| 测试 | model_100 | DR300 | DR500 | model_2000 | DR4000 | model_9500 |
|---|---|---|---|---|---|---|
| C_zero | 250 | 195 | 250 | 250 | 250 | 250 |
| A_no_ball | 250 | 99 | 107 | 250 | 250 | 238 |
| B_ball | 250 | 162 | 94 | 250 | 250 | 134 |

## 关键发现

1. **ONNX/obs/joint mapping 无误**
2. **早期 checkpoint 更稳定**：model_100 (接近零动作) 5s 全通关
3. **非 DR 训练越久越不稳定**：model_2000 双 250，model_9500 仅在 body-ground 支撑下存活
4. **DR 训练有效**：DR4000 是第一个两种 policy 模式都 250 timeout 的 DR checkpoint
5. **Zero-action 在 F1_fric10 下 qacc=14**：球质量修复 + body friction 对齐是关键
6. **"存活"不等于"守门"**：z<0.3 机器人已倒地，只是 body-ground 撑着不触发 z<0.2 终止
