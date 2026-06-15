# G1/Q1 Goalkeeper MuJoCo Wrapper 说明

## 概述

`q1_goalkeeper_mujoco_sim2sim.py` / `g1_goalkeeper_mujoco_sim2sim.py` 是 goalkeeper 策略在 MuJoCo 中的 sim2sim 验证脚本。Wrapper 是一个状态机，负责在**球运动前让机器人安全站立**，**检测射门后激活 GK 策略**，避免策略在等待阶段空跑导致 OOD。

## 三种激活模式

| 模式 | 参数 | 说明 |
|------|------|------|
| `ball_state_trigger` | `--activation-mode ball_state_trigger` | **默认**。从 ball pos history 差分检测射门，检测到才激活 GK |
| `always_on` | `--activation-mode always_on` | 策略从 t=0 运行（旧行为，用于对比 OOD） |
| 不传 | 同上 | 默认 `ball_state_trigger` |

## 状态机

```
       球静止/缓慢
    ┌──────────────┐
    │              │
    ▼    触发条件满足    │
  WAIT ────────────► ACTIVE ───► RECOVER
   │                  │
   │  PD站立          │  GK policy 运行
   │  不跑策略         │  最长 2s
   │  监控球状态        │
   └──────────────────┘
```

### WAIT 阶段
- 机器人用 default_dof_pos + PD 维持站立
- **不调用 ONNX 策略**
- 不更新 history、last_action、catchstep
- 每步读取 ball_pos，存入 `ball_pos_history`（最多 0.24s）
- 用 5 帧差分 (0.1s) 估计 ball_vel_est
- 方向用 ball → goal_center 计算，不硬编码 x 轴

### ACTIVE 阶段
- 从 GK local step 0 开始运行完整策略 loop
- history / ball_obs_state / last_action 在触发时全部重置
- 最长运行 2s（`gk_local_step > 2.0s` 截止）

### RECOVER 阶段
- PD 保持站立，等待仿真结束

## 触发条件 (ball_state_trigger)

用 ball_pos_history 差分估计速度，避免依赖 MuJoCo 真值：

```
speed = |vel_est|         > 0.3 m/s   (--trigger-speed-threshold)
v_toward_goal             > 0.2 m/s   (--trigger-toward-speed-threshold)
displacement (从初始位置)  > 0.02 m    (--trigger-displacement-threshold)
```

**连续满足 2 步**后才触发（debounce），防止噪声误触。

触发瞬间打印 `[GK_TRIGGER]` + ball_pos、vel_est、speed、预测到达时间。

## 球初始可见性

| 模式 | 参数 | 说明 |
|------|------|------|
| `train_timing` | `--gk-visible-after-trigger train_timing` | 保留训练 catchstep vanish（球激活后 ~8 步不可见） |
| `immediate` | `--gk-visible-after-trigger immediate` | 球立即可见（catchstep 跳到 startstep-1） |

## 球发射延迟

`--ball-launch-delay N`：球在 spawn 位置冻结 N 秒（物理层面，每 substep 重新锁定 qpos/qvel），与 wrapper 逻辑解耦。wrapper 在球开始运动后才检测触发。

## 常用命令

```bash
# 标准测试：1s 延迟 + ball_state_trigger + 球立即可见
python g1_goalkeeper_mujoco_sim2sim.py \
    --config g1_goalkeeper_mujoco_config.yaml \
    --headless --duration 5.0 --shot-mode 0 \
    --activation-mode ball_state_trigger \
    --gk-visible-after-trigger immediate \
    --ball-launch-delay 1.0 \
    --video-out output.mp4

# OOD 对比：always_on
python g1_goalkeeper_mujoco_sim2sim.py \
    --config g1_goalkeeper_mujoco_config.yaml \
    --headless --duration 3.0 --shot-mode 0 \
    --activation-mode always_on \
    --video-out output.mp4
```

## always_on 为何失效

`always_on` 模式下策略从 t=0 运行，但前 ~0.2s obs 中 ball_feature=0（训练 vanish）。这 0.2s 里策略看到 0 球观测但身体在动，history 被污染。等球可见时 history 已经 OOD，策略无法正确响应。

Wrapper 的 `ball_state_trigger` 保证 history 在触发时归零，策略从干净状态开始接球。

## 核心设计原则

- wrapper 不是 shooter→goalkeeper 通信，是 GK wrapper 从 ball state 自治检测
- playground global time 和 GK skill local time 完全解耦
- 不重置物理世界，只重置 policy-side state
