#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:?usage: robotwin2_run_clean_traces.sh RUN_ROOT TASK_NAME [SEEDS]}"
TASK_NAME="${2:?usage: robotwin2_run_clean_traces.sh RUN_ROOT TASK_NAME [SEEDS]}"
SEEDS="${3:-0-7}"

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/yihao_hyh/robotwin_probe}"
EFV_ROOT="${EFV_ROOT:-/home/yihao_hyh/Executable-Future-Verification}"
CONDA_ENV="${CONDA_ENV:-robotwin2-favc}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean_k5}"
CANDIDATE_PRESET="${CANDIDATE_PRESET:-targeted_energy_matched}"
GPU_ID="${GPU_ID:-0}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"
RUN_ANALYSIS_AFTER="${RUN_ANALYSIS_AFTER:-0}"
DRY_RUN="${DRY_RUN:-0}"

RAW_DIR="$RUN_ROOT/raw/$TASK_NAME"
LOG_DIR="$RUN_ROOT/logs"
LOG_FILE="$LOG_DIR/${TASK_NAME}_${CANDIDATE_PRESET}_seeds_${SEEDS//[^0-9A-Za-z_-]/_}.log"

mkdir -p "$RAW_DIR" "$LOG_DIR"

cmd=(
  python -m umm_reward_evaluator.benchmarks.robotwin2_gripper_aware_trace
  --task-name "$TASK_NAME"
  --task-config "$TASK_CONFIG"
  --seeds "$SEEDS"
  --output-dir "$RAW_DIR"
  --candidate-preset "$CANDIDATE_PRESET"
  --skip-existing
)

echo "RUN_ROOT=$RUN_ROOT"
echo "TASK_NAME=$TASK_NAME"
echo "SEEDS=$SEEDS"
echo "GPU_ID=$GPU_ID"
echo "ROBOTWIN_ROOT=$ROBOTWIN_ROOT"
echo "EFV_ROOT=$EFV_ROOT"
echo "LOG_FILE=$LOG_FILE"
printf 'COMMAND='
printf '%q ' "CUDA_VISIBLE_DEVICES=$GPU_ID" "${cmd[@]}"
printf '\n'

if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

if [ "$WAIT_FOR_GPU" = "1" ]; then
  while [ -n "$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')" ]; do
    echo "$(date -Is) GPU $GPU_ID busy; waiting ${GPU_WAIT_SECONDS}s"
    sleep "$GPU_WAIT_SECONDS"
  done
fi

source /home/yihao_hyh/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export PYTHONPATH="$EFV_ROOT/src:${PYTHONPATH:-}"

cd "$ROBOTWIN_ROOT"
echo "$(date -Is) start $TASK_NAME seeds=$SEEDS preset=$CANDIDATE_PRESET" | tee "$LOG_FILE"
CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
echo "$(date -Is) finish $TASK_NAME seeds=$SEEDS preset=$CANDIDATE_PRESET" | tee -a "$LOG_FILE"

if [ "$RUN_ANALYSIS_AFTER" = "1" ]; then
  cd "$EFV_ROOT"
  PYTHONPATH=src scripts/robotwin2_multitask_analysis.sh "$RUN_ROOT" "$TASK_NAME" 2>&1 | tee -a "$LOG_FILE"
fi
