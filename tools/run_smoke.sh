#!/bin/bash
# Q1 Goalkeeper Smoke Test Launcher — supports softA3 / midA3 / both
# Run from /root/autodl-tmp/ASAP
#
# Usage:
#   bash ../Humanoid-Goalkeeper/tools/run_smoke.sh softA3 zero_action
#   bash ../Humanoid-Goalkeeper/tools/run_smoke.sh midA3 all
#   bash ../Humanoid-Goalkeeper/tools/run_smoke.sh both all

set -euo pipefail
cd /root/autodl-tmp/ASAP

CONFIG="${1:-softA3}"
MODE="${2:-zero_action}"
STEPS="${3:-300}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROLLOUT_SCRIPT="${SCRIPT_DIR}/q1_goalkeeper_rollout.py"
OUTPUT_BASE="${SCRIPT_DIR}/../outputs"

export PATH="/root/miniconda3/envs/rl/bin:${PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

run_one() {
  local exp_name="$1"
  local mode="$2"
  local steps="$3"
  local out_dir="${OUTPUT_BASE}/${exp_name}/${mode}"

  echo "========================================"
  echo "  ${exp_name} | ${mode} (${steps} steps)"
  echo "========================================"

  HYDRA_FULL_ERROR=1 python "${ROLLOUT_SCRIPT}" \
    "+exp=${exp_name}" \
    project_name=Q1_Goalkeeper_Smoke experiment_name=Smoke_Rollout \
    num_envs=1 headless=True use_wandb=False checkpoint=null auto_load_latest=False \
    "+mode=${mode}" "+steps=${steps}" "+out_dir=${out_dir}" \
    2>&1 | grep -E "^\s|RESET|===|ball_|robot_|goal_|final|contact|first_|Smoke|PASS|FAIL|step|episode|shot|height|before|after|original|new"
}

run_modes_for_exp() {
  local exp_name="$1"
  shift
  local modes=("$@")
  for m in "${modes[@]}"; do
    run_one "$exp_name" "$m" "$STEPS"
    echo ""
  done
}

MODE_LIST=(zero_action random_small scripted_block)
case "$MODE" in
  all)  ;;
  *) MODE_LIST=("$MODE") ;;
esac

case "$CONFIG" in
  softA3)
    run_modes_for_exp q1_goalkeeper_smoke_softA3 "${MODE_LIST[@]}"
    ;;
  midA3)
    run_modes_for_exp q1_goalkeeper_smoke_midA3 "${MODE_LIST[@]}"
    ;;
  both)
    run_modes_for_exp q1_goalkeeper_smoke_softA3 "${MODE_LIST[@]}"
    echo "=============================="
    run_modes_for_exp q1_goalkeeper_smoke_midA3 "${MODE_LIST[@]}"
    ;;
  *)
    echo "Usage: $0 {softA3|midA3|both} [{zero_action|random_small|scripted_block|all}] [steps]"
    exit 1
    ;;
esac
