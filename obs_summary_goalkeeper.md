# Goalkeeper Observation Summary

## 1. Code locations

| 文件 | 作用 |
|------|------|
| `legged_gym/envs/base/legged_robot.py:465` | `compute_observations` — 构造每帧 obs |
| `legged_gym/envs/base/legged_robot.py:312` | `reset_idx` — 重置 history/catchstep/ball_last |
| `legged_gym/envs/base/legged_robot.py:1127` | `catchstep` / `ball_last` / `vanish_step` 初始化 |
| `legged_gym/envs/base/base_task.py:72` | `obs_buf` 全零初始化 |
| `legged_gym/envs/q1/q1_goalkeeper_config.py:8` | Q1 obs 维度定义 |
| `legged_gym/envs/g1/g1_29_config.py:1` | G1 obs 维度定义 |
| `scripts/q1_goalkeeper_mujoco_sim2sim.py` | Q1 MuJoCo sim2sim obs 构造 |
| `scripts/g1_goalkeeper_mujoco_sim2sim.py` | G1 MuJoCo sim2sim obs 构造 |

---

## 2. Q1 actor obs

| index | item | dim | scale | frame | source tensor | deploy? |
|-------|------|-----|-------|-------|---------------|---------|
| 0:3 | ball_feature | 3 | ×1.0 | base (pelvis quat) | `ball_states[:,:3] - torso_pos`, rotated by `base_quat` | ✅ |
| 3:6 | base_ang_vel | 3 | ×0.25 | base | `rigid_body_states[pelvis, 10:13]`, rotated by pelvis quat | ✅ |
| 6:9 | projected_gravity | 3 | ×1.0 | base | `quat_rotate_inverse(pelvis_quat, [0,0,-1])` | ✅ |
| 9:31 | dof_pos − default | 22 | ×1.0 | joint | `self.dof_pos` | ✅ |
| 31:53 | dof_vel | 22 | ×0.05 | joint | `self.dof_vel` | ✅ |
| 53:75 | last_action | 22 | ×1.0 | — | `self.actions` (上一帧 policy 输出) | ✅ |

- **单帧维度**: 75
- **帧栈 (history)**: 10 → **actor obs = 750**
- **动作维度**: 22

---

## 3. G1 actor obs

| index | item | dim | scale | frame | source tensor | deploy? |
|-------|------|-----|-------|-------|---------------|---------|
| 0:3 | ball_feature | 3 | ×1.0 | base (pelvis quat) | 同上 | ✅ |
| 3:6 | base_ang_vel | 3 | ×0.25 | base | 同上 | ✅ |
| 6:9 | projected_gravity | 3 | ×1.0 | base | 同上 | ✅ |
| 9:38 | dof_pos − default | 29 | ×1.0 | joint | 同上 | ✅ |
| 38:67 | dof_vel | 29 | ×0.05 | joint | 同上 | ✅ |
| 67:96 | last_action | 29 | ×1.0 | — | 同上 | ✅ |

- **单帧维度**: 96
- **帧栈 (history)**: 10 → **actor obs = 960**
- **动作维度**: 29

---

## 4. History layout

- **初始化**: `obs_buf = zeros(num_envs, num_obs)` (训练) / `zeros(frame_stack, single_obs)` (sim2sim)
- **追加方式**: **oldest → newest**，即 `obs_buf[0]` 最老，`obs_buf[frame_stack-1]` 最新
- **每步更新**: 左移抛弃最老帧，新帧追加到末尾
  ```python
  self.obs_buf = torch.cat((self.obs_buf[:, single_obs:], current_actor_obs), dim=-1)
  ```
- **last_action**: 上一帧 policy 输出 `self.actions`。episode 首次为 0
- **reset**: `last_actions = 0`，`last_dof_vel = 0`，`obs_buf` 在后续步骤中自然被新帧填充（不显式清零）

---

## 5. Ball observation logic

### 5.1 Ball feature 计算

```
ball_feature = quat_rotate_inverse(base_quat, ball_states[:,:3] - torso_pos)
```

- 球世界坐标 − `torso_link` 世界坐标 → 相对位置
- 用 `pelvis` quaternion (`base_quat`) 旋转到 base frame
- **sim2sim 中一致**：`ball_pos_world - torso_pos`, 用 `base_quat` 旋转

### 5.2 可见性控制 (catchstep / startstep / vanish)

训练中的球可见性由三层控制：

| 层 | 变量 | 逻辑 | sim2sim 对应 |
|----|------|------|-------------|
| 初始消失 | `startstep = 50 - randint(3,10)` [40,47] | `catchstep < startstep` 时 `initial_vanish=False`，ball_feature 为零 | ✅ Q1: `vanish_steps=10` 前零化; G1: `compute_ball_feature` 内部 |
| 飞行判断 | `flying` | `end_target[:,0]` 在 [0.05, 3.4] 且 y/z 在范围内，且 `catchstep>0`，且球 x 递减 | ⚠️ G1 sim2sim 部分实现，Q1 未实现 |
| 随机消失 | `vanish_step = randint(0,30)` | `catchstep > vanish_step` 时额外 mask → 球可见窗口随机缩短 | ✅ G1 sim2sim 实现; Q1 未使用 |
| flying × random_vanish | | 训练时 `ball_feature *= flying * random_vanish` | ⚠️ 不一致：sim2sim 未做 flying 过滤 |
| ball_last | `ball_last = end_target_local` | 每帧更新，用于 `flying` 检查中判断球是否在接近 | ✅ G1 sim2sim |

### 5.3 Q1 vs G1 ball 可见性差异

- **Q1**: 训练用 `flying * random_vanish`；sim2sim 用简化的 `vanish_steps=10` + `catchstep/startstep`
- **G1**: 训练用 `flying * random_vanish`；sim2sim 用 `compute_ball_feature` 内部 `initial_vanish` + `flying` + `random_vanish`

### 5.4 球不可见时

训练中球不可见时 `ball_feature` 被置零。sim2sim 中 Q1 用显式 `vanish_steps` 零化，G1 用 `compute_ball_feature` 内部零化。

---

## 6. Critic / privileged obs

Critic 比 actor 多了以下 privileged 信息（从 `current_obs[:, single_obs:]` 读取）：

| item | dim | source | 部署可用? |
|------|-----|--------|----------|
| base_lin_vel × scale | 3 | root states 线速度 | ❌ 真机难获取 |
| end_regions / 3 | 1 | 球落点区域 (0-5, 6种) | ❌ 训练特有标签 |
| end_target local (base frame) | 3 | 球轨迹终点 − torso_pos | ❌ 需知道球终点 |
| ball_vel local × scale | 3 | ball_states 速度，base frame | ⚠️ 真机可估计但不稳定 |
| hand_pos_r (base frame) | 3 | 右手 − torso_pos | ✅ 可从 qpos+FK 算 |
| hand_pos_l (base frame) | 3 | 左手 − torso_pos | ✅ |
| dist | 1 | 球落点到手距离 | ❌ 训练特有 |

**Critic obs维度**: Q1=92, G1=113（= actor_single_obs + 17 privileged）

---

## 7. Sim2sim consistency check

### 一致项 ✅

| 项目 | 训练 | sim2sim |
|------|------|---------|
| ball_feature 构造 | base-frame，相对 torso，用 pelvis quat | ✅ 一致 |
| ang_vel_base | pelvis body 角速度，base frame | ✅ 一致 |
| projected_gravity | pelvis quat 旋转 [0,0,-1] | ✅ 一致 |
| dof_pos − default | joint position | ✅ 一致 |
| dof_vel | joint velocity | ✅ 一致 |
| last_action | 上一帧 action | ✅ 一致 |
| 缩放系数 | ang_vel×0.25, dof_vel×0.05, dof_pos×1.0 | ✅ 一致 |
| history 维度和顺序 | 10 帧, oldest→newest | ✅ 一致 |
| obs_buf 初始化 | 全零 | ✅ 一致 (create_history_buffer) |

### 不一致/可疑项 ⚠️

| 项目 | 训练 | sim2sim | 影响 |
|------|------|---------|------|
| **ball 可见性 (Q1)** | `ball_feature *= flying * random_vanish` (catchstep/startstep/vanish三层) | 简化版: `vanish_steps=10` 零化 + catchstep/startstep | 低 — vanish 期间行为不完全一致，但 vanish 结束后一致 |
| **ball 可见性 (G1)** | `ball_feature *= flying * random_vanish` | `compute_ball_feature` 含 `initial_vanish` + `flying` + `random_vanish` | 中 — `flying` 检查稍后可能过早隐藏球 |
| **privileged obs** | 训练 critic 有 17 维 privilege | sim2sim 不使用 critic | 无影响 |
| **ball_feature 未缩放** | 训练 actor obs 直接 raw ball_feature | 同样 raw | ✅ 一致 |
| **history reset** | 训练 `reset_idx` 置零 `last_actions`，`obs_buf` 不清零 | sim2sim 全部 reset 到 zero | ✅ 更彻底，效果等价 |

---

## 8. Final conclusion

| | Q1 | G1 |
|---|---|---|
| obs_dim (actor) | 750 | 960 |
| action_dim | 22 | 29 |
| single_obs | 75 | 96 |
| frame_stack | 10 | 10 |
| privileged_obs_dim | 92 | 113 |
| actor obs 部署可得 | ✅ 全部 6 项 | ✅ 全部 6 项 |
| sim2sim obs 对齐度 | 高 (vanish逻辑简化但等价) | 中高 (ball可见性有细微差异) |

**最关键的对齐点**:
1. ball_feature 必须用 `torso_link` 而非 `pelvis` 做位置参考 — 已对齐
2. ball_feature 和 projected_gravity 共用 `base_quat` (pelvis 四元数) — 已对齐
3. last_action 是上一帧 policy 输出，部署时需自行追踪 — 已对齐
4. G1 sim2sim 的 `compute_ball_feature` 中 `flying`/`random_vanish` 和训练有细微差异（训练用 `ball_states` 直接算，sim2sim 用 MuJoCo `data.xpos`），可能导致球在边缘位置时不可见时间不同
5. history 帧顺序必须 oldest→newest，部署时保持一致
