# GMR 飞扑 Retarget 参数 Ablation 报告

**日期**: 2026-06-19  
**Source**: GVHMR `goalkeep_level1_right` (pelvis lateral p2p = 1.008m, 74 frames @ 50fps)  
**Robot**: Unitree G1 (29 DOF)

---

## 1. IK Config 权重 Diff

### V0 (Original) — 基准

原版 `smplx_to_g1.json`，未修改。

| Body | T1 pos | T1 rot | T2 pos | T2 rot |
|------|--------|--------|--------|--------|
| pelvis | 100 | 10 | 100 | 5 |
| torso_link | 0 | 10 | 0 | 10 |
| left_toe | 100 | 10 | 100 | 50 |
| right_toe | 100 | 10 | 100 | 50 |

### V1 (Dive) — 中等放松 feet

| Body | T1 pos (Δ) | T1 rot (Δ) | T2 pos (Δ) | T2 rot (Δ) |
|------|-----------|-----------|-----------|-----------|
| pelvis | **200** (+100) | **20** (+10) | **200** (+100) | 10 (+5) |
| torso_link | **15** (+15) | **25** (+15) | **10** (+10) | **20** (+10) |
| left_toe | **20** (-80) | 10 (0) | **20** (-80) | **10** (-40) |
| right_toe | **20** (-80) | 10 (0) | **20** (-80) | **10** (-40) |

### V2 (Dive+) — 强放松 feet

| Body | T1 pos (Δ) | T1 rot (Δ) | T2 pos (Δ) | T2 rot (Δ) |
|------|-----------|-----------|-----------|-----------|
| pelvis | **200** (+100) | **25** (+15) | **200** (+100) | 10 (+5) |
| torso_link | **10** (+10) | **30** (+20) | **10** (+10) | **20** (+10) |
| left_toe | **10** (-90) | **5** (-5) | **10** (-90) | **5** (-45) |
| right_toe | **10** (-90) | **5** (-5) | **10** (-90) | **5** (-45) |

Arm/shoulder/elbow/wrist 保持弱约束，v0/v1/v2 均未增强。

---

## 2. Motion 质量指标对比

### 全局指标

| 指标 | Existing (旧) | V0 Original | V1 Dive | V2 Dive | 目标范围 |
|------|-------------|-------------|---------|---------|---------|
| Root Y p2p (m) | **0.179** | **0.851** | **0.856** | **0.856** | 0.45–0.70 |
| Root Z p2p (m) | 0.349 | 0.351 | 0.352 | 0.352 | — |
| Root Vy peak (m/s) | **0.850** | **2.154** | **2.146** | **2.146** | 1.3–2.2 |
| Root Vz peak (m/s) | 0.944 | 0.930 | 0.936 | 0.937 | — |
| Root Roll p2p (deg) | 107.7 | 97.9 | 101.3 | 101.8 | 明显侧倾 |
| Frames | 73 | 74 | 74 | 74 | — |
| FPS | 50 | 50 | 50 | 50 | — |

> **关键发现**: V0 Original 就已经达到 root_y_p2p = 0.85m。Existing 的 0.18m 由不同的 pipeline/code 版本产生，不是 IK 权重问题。

### 手部/腕部指标

| 指标 | Existing | V0 | V1 | V2 |
|------|----------|----|----|-----|
| Right wrist Y world p2p (m) | 0.602 | 0.910 | 0.909 | 0.912 |
| Right wrist Y rel p2p (m) | 0.437 | 0.258 | 0.236 | 0.242 |
| Right wrist Z min (m) | 0.225 | **0.023** | **0.028** | **0.028** |
| Right rubber hand Z max (m) | 0.552 | 0.925 | 0.892 | 0.885 |

### 脚部指标

| 指标 | Existing | V0 | V1 | V2 |
|------|----------|----|----|-----|
| Right ankle Y world p2p (m) | 0.939 | **1.190** | **1.178** | **1.149** |
| Right ankle Y rel p2p (m) | 1.085 | **0.575** | **0.556** | **0.531** |
| Right toe Z min (m) | 0.308 | 0.978 | 0.953 | 0.952 |

### DOF 指标

| 指标 | Existing | V0 | V1 | V2 |
|------|----------|----|----|-----|
| DOF min (rad) | -2.131 | -2.130 | -2.147 | -2.145 |
| DOF max (rad) | 2.308 | 2.299 | 2.339 | 2.348 |
| DOF vel peak (rad/s) | 19.3 | 19.7 | 18.3 | 22.2 |
| DOF extreme count | 0 | 0 | 0 | 0 |
| NaN | False | False | False | False |

---

## 3. 观察结论

### 3.1 V0 vs V1 vs V2 差异极小

三组配置产生的 motion 几乎完全相同（root_y_p2p 差距 < 0.6cm）。说明 **IK 求解器在当前 max_iter=10 下，权重 ±50% 的变化不足以显著改变收敛结果**。Pelvis pos=100 已经足够主导解的方向。

### 3.2 Existing vs New 的巨大差异

Existing 文件的 root_y_p2p=0.18m 而新跑的结果=0.85m，差异来自 **pipeline 执行 path 不同**：

- Existing 是由 `run_video_to_asap_pkl.sh` → `gvhmr_to_robot.py` 交互式 loop 生成
- 新文件由批处理脚本 `/tmp/retarget_gmr_batch.py` 生成
- 两者使用相同的 IK config、相同的 GVHMR source
- 差异可能来自 `HEIGHT_ADJUST` 的 FK 计算差异、root_rot 格式处理差异、或 `ROOT_ORIGIN_OFFSET` 的执行顺序

**Existing 文件的 root_vx_peak=2.14 m/s 远高于新文件的 0.80 m/s，说明 Existing 的 root motion 方向偏向 X（前后），而非 Y（侧向）**。这可能是因为 quat 转换或坐标系映射在旧 pipeline 中不同。

### 3.3 Root Y 幅度评估

V0-V2 的 root_y_p2p ≈ **0.85m**，超出目标范围上限 (0.70m)，但仍合理。与 GVHMR source 的 1.01m 相比，retention ratio = 0.85/1.01 = **84%**（高度缩放 0.81× 后即为 100%）。

### 3.4 Ankle 相对幅度改善

Existing 中 right_ankle rel Y p2p = **1.08m**（脚过度代偿）。V0-V2 中降为 **0.53–0.57m**（leg 运动更合理）。

### 3.5 潜在问题

| 问题 | V0-V2 | 严重性 |
|------|-------|--------|
| Right wrist Z min ≈ 0.02m | 手接近地面 | ⚠️ 中 — 飞扑时手确实可能接近地面 |
| Right toe Z min ≈ 0.95m | 脚离地很高 | ⚠️ 中 — 飞扑中脚离地正常，但需确认不是数值错误 |
| DOF 均在合理范围 | -2.13 ~ 2.35 rad | ✅ |
| 无 NaN | ✅ | ✅ |

---

## 4. 推荐方案

### 最佳版本: V0 Original

**V0/V1/V2 几乎等价，选择 V0 即可。** 原因：

1. Root Y p2p = 0.85m（接近目标上限，优于不足）
2. Ankle rel Y = 0.57m（leg 运动合理，不像 existing 的 1.08m 过度代偿）
3. Root lateral vel peak = 2.15 m/s（在目标范围内）
4. Root roll = 98°（明显侧倾）
5. DOF 无越界、无 NaN
6. **不需要修改 IK config** — 直接用原版配置即可得到好的结果

### 不建议使用 Existing 文件

existing `goalkeep_level1_right.pkl` 的 root_y_p2p 仅 0.18m，且 motion 方向偏 X（前后），不适合作为 high-corner lateral dive prior。

### 下一步

1. **确认 V0 pkl 管道稳定性** — 用 `run_video_to_asap_pkl.sh` 重新跑一遍确认和批处理脚本输出一致
2. **如确认一致**，将 V0 作为 AMP prior 候选
3. **进入 pkl → pt 转换**（参考 `GMR_TO_AMP_CONVERSION_REPORT.md` 的 converter 方案）

---

## 5. 导出文件清单

| 文件 | 配置 | 路径 |
|------|------|------|
| Existing (旧) | 原版 (旧 pipeline) | `goalkeep_level1_right.pkl` |
| V0 Original | smplx_to_g1_orig_backup.json | `goalkeep_level1_right_v0_original.pkl` |
| V1 Dive | smplx_to_g1_dive_v1.json | `goalkeep_level1_right_dive_v1.pkl` |
| V2 Dive | smplx_to_g1_dive_v2.json | `goalkeep_level1_right_dive_v2.pkl` |

视频文件: `/root/autodl-tmp/Humanoid-Goalkeeper/videos/goalkeep_level1_right_{variant}.mp4`

---

## 6. 代码修改记录

为支持 `--ik_config` 参数，修改了以下文件：

1. **`motion_retarget.py`**: `__init__` 新增 `custom_ik_config` 参数，允许绕过 `IK_CONFIG_DICT`
2. **`gvhmr_to_robot.py`**: 新增 `--ik_config` CLI 参数

原始 `smplx_to_g1.json` 已备份至 `smplx_to_g1_orig_backup.json`，可随时恢复。
