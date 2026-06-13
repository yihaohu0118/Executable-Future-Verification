#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-/home/yihao_hyh/efv_runs/robotwin2_iclr_window_$(date +%Y%m%d)}"
SEEDS="${SEEDS:-0-7}"
TASKS="${TASKS:-stack_blocks_two stamp_seal open_microwave place_object_basket stack_bowls_two press_stapler}"
GPU_ID="${GPU_ID:-0}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean_k5}"
CANDIDATE_PRESET="${CANDIDATE_PRESET:-targeted_energy_matched}"
EXECUTE="${EXECUTE:-0}"
RUN_ANALYSIS_AFTER="${RUN_ANALYSIS_AFTER:-0}"

echo "RUN_ROOT=$RUN_ROOT"
echo "SEEDS=$SEEDS"
echo "TASKS=$TASKS"
echo "GPU_ID=$GPU_ID"
echo "TASK_CONFIG=$TASK_CONFIG"
echo "CANDIDATE_PRESET=$CANDIDATE_PRESET"
echo "EXECUTE=$EXECUTE"
echo "RUN_ANALYSIS_AFTER=$RUN_ANALYSIS_AFTER"

for task in $TASKS; do
  echo
  echo "=== $task ==="
  if [ "$EXECUTE" = "1" ]; then
    GPU_ID="$GPU_ID" \
    TASK_CONFIG="$TASK_CONFIG" \
    CANDIDATE_PRESET="$CANDIDATE_PRESET" \
    RUN_ANALYSIS_AFTER="$RUN_ANALYSIS_AFTER" \
      scripts/robotwin2_run_clean_traces.sh "$RUN_ROOT" "$task" "$SEEDS"
  else
    DRY_RUN=1 \
    GPU_ID="$GPU_ID" \
    TASK_CONFIG="$TASK_CONFIG" \
    CANDIDATE_PRESET="$CANDIDATE_PRESET" \
    RUN_ANALYSIS_AFTER="$RUN_ANALYSIS_AFTER" \
      scripts/robotwin2_run_clean_traces.sh "$RUN_ROOT" "$task" "$SEEDS"
  fi
done

if [ "$EXECUTE" = "1" ]; then
  echo
  echo "Run analysis after traces finish with:"
  echo "  PYTHONPATH=src scripts/robotwin2_multitask_analysis.sh $RUN_ROOT $TASKS"
fi
