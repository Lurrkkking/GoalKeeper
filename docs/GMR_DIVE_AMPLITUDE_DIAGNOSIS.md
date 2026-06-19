# GMR 飞扑幅度压缩诊断报告

**日期**: 2026-06-19  
**GVHMR Source**: `/root/autodl-tmp/GVHMR/outputs/demo/goalkeep_level1_right/hmr4d_results.pt`  
**GMR Output**: `/root/autodl-tmp/GMR/unitree_g1_gmr/goalkeep_level1_right.pkl`

---

## 1. GVHMR Source vs GMR Output 幅度对比

### 关键指标对比表

| 指标 | GVHMR Source (SMPL global) | GMR Output (G1 FK) | 压缩比 |
|------|---------------------------|-------------------|--------|
| Pelvis/Root Y lateral p2p | **1.008 m** | **0.179 m** | **5.6×** |
| Pelvis/Root Z range | 0.418 m (1.25→0.83) | 0.349 m (0.77→0.42) | 1.2× |
| Pelvis/Root lateral vel peak | **2.55 m/s** | **0.85 m/s** | **3.0×** |
| Pelvis/Root vertical vel peak | 1.11 m/s | 0.94 m/s | 1.2× |
| Right ankle Y p2p | — | 0.94 m (world) / 1.08 m (rel) | — |
| Right wrist Y p2p | — | 0.60 m (world) / 0.44 m (rel) | — |
| Root roll range | — | 5.5° → 113.2° (107.7° span) | — |
| Root pitch range | — | -28.9° → 27.8° (56.7° span) | — |
| DOF count | 63 body_pose + 3 global_orient | 29 | — |
| Frames | 74 @ 50fps | 73 @ 50fps | — |
| Duration | 1.48s | 1.46s | — |

> **注**: SMPL Y-up 坐标系与 GMR Z-up 的映射：SMPL X (lateral) → GMR Y (lateral), SMPL Y (up) → GMR Z (up)

### 高度缩放系数

IK config 中 `human_height_assumption = 1.8m`，`pelvis scale = 0.9`。
从 betas 估算 `actual_human_height ≈ 1.62m`，ratio = 1.62/1.8 = 0.9。
综合缩放: `scaled_root = 0.9 × 0.9 × source = 0.81 × source`

**即使考虑缩放，期望的 pelvis Y p2p = 0.81 × 1.008 = 0.82m，实际只有 0.18m — 仅为期望的 22%**。

---

## 2. 诊断结论

### 2.1 幅度损失发生在哪一阶段？

**结论：幅度损失发生在 GMR retarget 阶段，不在 GVHMR 阶段。**

证据：
- GVHMR source pelvis lateral p2p = **1.008 m**，速度峰值 **2.55 m/s** → 源数据幅度足够大
- GMR 输出 root Y p2p = **0.179 m**，速度峰值 **0.85 m/s** → 输出被大幅压缩
- 时序（74帧→73帧）和 Z 幅度（0.42m→0.35m）基本保持 → 只有 lateral 维度被压缩

**诊断类型 E：GMR root lateral 和 body/arm 展开都受到抑制，但 legs 补偿性地做大幅度步态/跳跃。**

### 2.2 根因分析：Pelvis vs Foot 的 IK 冲突

GMR IK config (`smplx_to_g1.json`) 中的权重配置：

**ik_match_table1** (第一轮求解):
| Body | Pos Weight | Rot Weight |
|------|-----------|-----------|
| pelvis | **100** | 10 |
| left_toe (left_foot) | **100** | 10 |
| right_toe (right_foot) | **100** | 10 |
| others | 0 | 10 |

**ik_match_table2** (第二轮求解):
| Body | Pos Weight | Rot Weight |
|------|-----------|-----------|
| pelvis | **100** | 5 |
| left_toe (left_foot) | **100** | **50** |
| right_toe (right_foot) | **100** | **50** |
| others | 10 | 5 |

**问题：Pelvis 和 Feet 位置权重相同（均为 100），在飞扑动作中形成约束冲突。**

当人类飞扑时：
1. Pelvis 侧向移动 **1.01m**，脚也跟随移动并离开地面
2. G1 机器人腿长固定，IK 无法同时满足 "pelvis 在 0.82m 外" 和 "双脚在各自跟踪位置"
3. 由于 feet 在 table2 增加了 rot_weight=50（比 pelvis 的 5 高 10 倍），**foot tracking 实际上主导了 IK**
4. IK 妥协：让 pelvis 只移动 0.18m，让 feet 通过大幅步态/跳跃来匹配源数据脚位置

**FK 证据**：
- Right ankle 世界坐标 Y p2p = **0.94m**（比 pelvis 的 0.18m 大 5.2 倍）
- Right ankle 相对 pelvis Y p2p = **1.08m**（腿跨度极大）
- Right wrist 相对 pelvis Y p2p = **0.44m**（手臂有一定伸展）

### 2.3 其他可能影响因素

| 因素 | 检查结果 | 影响程度 |
|------|---------|---------|
| Root XY 被重置/zero | `ROOT_ORIGIN_OFFSET = True` (line 291) 只是将 frame0 平移到原点，不压缩幅度 | 无影响 |
| 高度缩放过小 | ratio ≈ 0.9，pelvis_scale = 0.9 → 综合 0.81×，不应该导致 5.6× 压缩 | 轻微 |
| GVHMR hip abduction clamp | MAX_HIP_ANGLE=0.6rad 限制髋外展 | 对 lateral dive 可能有轻微影响 |
| GVHMR arm scale | ARM_SCALE=1.0 (未启用) | 无影响 |
| IK solver damping | damping=5e-1 | 可能稍微平滑运动，但不至于压缩 5× |
| Velocity limits | `use_velocity_limit=False` (默认) | 无影响 |
| IK 迭代次数 | max_iter=10 | 充分 |
| Foot contact constraint | GMR 中没有显式的 foot contact / ground constraint | 见下文分析 |

### 2.4 IK config 中 ROOT XY 是否被忽略

**没有被忽略。** Pelvis 的 pos_weight=100 明确追踪 full 3D position（包括 XY）。问题不是 "不追踪"，而是 "同时追踪 pelvis 和 feet 导致冲突"。

### 2.5 Hand/Wrist 约束是否过强

在当前 IK config 中，wrist/shoulder/elbow 的 pos_weight 在 table1 为 0，table2 为 10。相比 feet 的 100，hand tracking 权重较低。**Hand tracking 不是 root 幅度压缩的原因。**

但需要注意：当前权重配置对手臂的 position tracking 为 0（table1）或 10（table2），仅 orientation tracking 在全表启用。对于 AMP prior 用途，这其实是合适的 —— 手不需要精确匹配 demo 轨迹。

---

## 3. 最小修改建议

### 3.1 IK Config 权重调整（推荐方案）

为飞扑类动作创建专用 IK config `smplx_to_g1_dive.json`:

```json
"ik_match_table1": {
    "pelvis": ["pelvis", 200, 20, [0,0,0], [0.5,-0.5,-0.5,-0.5]],
    "torso_link": ["spine3", 50, 20, [0,0,0], [0.5,-0.5,-0.5,-0.5]],
    "left_toe_link": ["left_foot", 10, 10, [0,0.02,0], [0.5,-0.5,-0.5,-0.5]],
    "right_toe_link": ["right_foot", 10, 10, [0,-0.02,0], [0.5,-0.5,-0.5,-0.5]],
    ... (其余 body 保持 pos_weight=0, rot_weight=10)
},
"ik_match_table2": {
    "pelvis": ["pelvis", 200, 10, [0,0,0], [0.5,-0.5,-0.5,-0.5]],
    "torso_link": ["spine3", 20, 10, [0,0,0], [0.5,-0.5,-0.5,-0.5]],
    "left_toe_link": ["left_foot", 10, 10, [0,0,0], [-0.5,0.5,0.5,0.5]],
    "right_toe_link": ["right_foot", 10, 10, [0,0,0], [-0.5,0.5,0.5,0.5]],
    ... (其余 body 保持 pos_weight=10, rot_weight=5)
}
```

**关键变更**:
1. **Pelvis pos_weight: 100 → 200** (提高 root lateral 跟踪优先级)
2. **Feet pos_weight: 100 → 10** (大幅降低脚位置约束)
3. **Torso_link 新增 pos_weight: 0 → 50 (table1)** (增加躯干侧倾跟踪)
4. **Feet rot_weight: 50 → 10 (table2)** (降低脚朝向约束)
5. Hand/wrist 保持低权重（table1 pos=0, table2 pos=10），不精确跟踪手部轨迹

### 3.2 其他可选调整

| 参数 | 当前值 | 建议值 | 原因 |
|------|--------|--------|------|
| `damping` | 5e-1 | 1e-1 | 降低阻尼使动态动作更敏捷 |
| `max_iter` | 10 | 15 | 允许更多迭代收敛 |
| `use_velocity_limit` | False | False | 飞扑需要高速，不限制 |

### 3.3 无需修改的部分

- **不要**修改 `ROOT_ORIGIN_OFFSET` — 仅做 canonicalization
- **不要**修改 `HEIGHT_ADJUST` — 确保脚不穿地
- **不要**修改 `ARM_SCALE` / `MAX_HIP_ANGLE` — 与 lateral 幅度无关
- **不要**在 post-processing 中用 `root_pos[:,1] *= scale` 硬放大 — 会导致 feet 和 pelvis 的相对位置不物理（脚滑、悬空）

---

## 4. Root Lateral Scale 后处理的风险评估

**不推荐**直接对 root lateral 做 scale 后处理。原因：

1. **脚-骨盆相对位置失真**：当前 FK 显示 right_ankle rel Y 范围 [-0.43, 0.66]，如果 root Y 放大 3×，ankle rel Y 会变成 [-1.3, 1.98]，远超 G1 腿长（~0.7m）
2. **速度不连续**：单纯 scale 会导致 root velocity 跳变
3. **IK 内部的妥协仍然存在**：即使后处理放大 root，DOF 姿态并没有对应调整（腿仍然在做"步态"而非"飞扑"）

**正确做法**：回到 IK config 权重调整（3.1），让 IK 在求解时就将 lateral motion 分配到 pelvis 而不是 feet。

---

## 5. 当前 goalkeep_level1_right.pkl 的适用性判断

| 用途 | 是否适合 | 原因 |
|------|---------|------|
| AMP pipeline smoke test | ✅ 适合 | 数据格式完整（29 DOF），FK 合理，无 NaN/穿地 |
| 正式 high-corner dive AMP prior | ❌ 不适合 | Root lateral 仅 0.18m，pelvis dive 被转换为 foot step/jump，policy 会学到"站在原位伸手"而非侧向飞扑 |
| 低幅度 side-step save prior | ⚠️ 勉强可用 | 如果 target 在很近范围内（< 0.5m 侧向），当前 motion 可能够用 |
| 训练代码接入测试 | ✅ 适合 | converter 可用，数据加载正常 |

---

## 6. 优先级总结

```
高优先级：
  ✅ 调整 IK config 权重 — pelvis ↑, feet ↓
  ✅ 增加 torso_link 位置跟踪

中优先级：
  创建专用的 dive IK config
  调整 damping 和 max_iter

低优先级（不需要做）：
  后处理 root lateral scale
  修改 GVHMR clamp 参数
  手部精确轨迹跟踪
  启用 velocity limits
```
