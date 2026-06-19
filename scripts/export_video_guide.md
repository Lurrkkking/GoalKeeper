# Goalkeeper Sim2Sim 视频导出指南

## 基本命令

```bash
cd /root/autodl-tmp/Humanoid-Goalkeeper

# 1. 导出 ONNX
conda run -n rl python tools/export_q1_goalkeeper_onnx.py \
    --checkpoint legged_gym/logs/q1/q1_gk_v3_amp_v2/model_22000.pt \
    --output legged_gym/logs/q1/exported/policies/q1_22000.onnx

# 2. 切换到目标 ONNX
sed -i 's|goalkeeper.onnx|q1_22000.onnx|' scripts/q1_goalkeeper_mujoco_config.yaml

# 3. 导出单个视频
conda run -n rl python scripts/q1_sim2sim.py \
    --config scripts/q1_goalkeeper_mujoco_config.yaml \
    --headless --duration 3.0 --shot-mode 0 \
    --activation-mode ball_state_trigger --gk-visible-after-trigger immediate \
    --ball-launch-delay 1.5 \
    --video-out outputs/mode0.mp4

# 4. 恢复 ONNX
sed -i 's|q1_22000.onnx|goalkeeper.onnx|' scripts/q1_goalkeeper_mujoco_config.yaml
```

## 批量导出 6 个 Mode

```bash
BASE=/root/autodl-tmp/Humanoid-Goalkeeper
mkdir -p outputs/my_eval

for pair in "0:R" "1:L" "2:R" "3:L" "4:R" "5:L"; do
    mode=$(echo $pair | cut -d: -f1); side=$(echo $pair | cut -d: -f2)
    timeout 20 conda run -n rl python $BASE/scripts/q1_sim2sim.py \
        --config $BASE/scripts/q1_goalkeeper_mujoco_config.yaml \
        --headless --duration 3.0 --shot-mode $mode \
        --activation-mode ball_state_trigger --gk-visible-after-trigger immediate \
        --ball-launch-delay 1.5 \
        --video-out "outputs/my_eval/${side}_mode${mode}.mp4" 2>/dev/null
done
```

## G1 同理

```bash
# ONNX 导出
python tools/export_g1_goalkeeper_onnx.py --checkpoint <path> --output <path>

# 脚本
python scripts/g1_sim2sim.py --config scripts/g1_goalkeeper_mujoco_config.yaml ...
```

## 点球模拟 (7m)

```bash
# 球从 7m 正前方地面射出，射向球门边缘
--y-range="0.7,1.0" --z-range="0.05,1.6" --spawn-x 7.0

# 左侧
--y-range="-1.0,-0.7" --z-range="0.05,1.6" --spawn-x 7.0
```

## 关键参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--shot-mode` | 0-5 射门模式 | 0-5 循环 |
| `--duration` | 仿真时长(s) | 3.0 |
| `--ball-launch-delay` | 球静止时间(s) | 1.5 (Q1), 1.0 (G1) |
| `--activation-mode` | 策略激活方式 | ball_state_trigger |
| `--gk-visible-after-trigger` | 球可见时机 | immediate |
| `--y-range` | 球门横向范围 | 默认脚本范围 |
| `--z-range` | 球门高度范围 | 默认脚本范围 |
| `--spawn-x` | 点球距离(m) | 7.0 |

## Q1 vs G1 脚本对照

| | Q1 | G1 |
|---|---|---|
| 交付版脚本 | `scripts/q1_sim2sim.py` | `scripts/g1_sim2sim.py` |
| 配置 | `scripts/q1_goalkeeper_mujoco_config.yaml` | `scripts/g1_goalkeeper_mujoco_config.yaml` |
| ONNX 尺寸 | 750→22 | 960→29 |
| 推荐 delay | 1.5s | 1.0s |
