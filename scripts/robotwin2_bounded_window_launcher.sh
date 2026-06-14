#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-/home/yihao_hyh/efv_runs/robotwin2_bounded_window_$(date +%Y%m%d)}"
SEEDS="${SEEDS:-0-7}"
TASKS="${TASKS:-stack_blocks_two stamp_seal place_object_basket stack_bowls_two}"
GPU_ID="${GPU_ID:-auto}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean_k5}"
CANDIDATE_PRESET="${CANDIDATE_PRESET:-targeted_energy_matched}"
EXECUTE="${EXECUTE:-0}"
RUN_ANALYSIS_AFTER="${RUN_ANALYSIS_AFTER:-0}"
RESUME_PARTIAL="${RESUME_PARTIAL:-0}"
REQUIRE_CANDIDATES_PER_CASE="${REQUIRE_CANDIDATES_PER_CASE:-24}"
NUM_SWEEP_SEEDS="${NUM_SWEEP_SEEDS:-10}"
RETRY_TRANSIENT_GPU="${RETRY_TRANSIENT_GPU:-1}"
TRANSIENT_GPU_RETRY_SECONDS="${TRANSIENT_GPU_RETRY_SECONDS:-180}"
TRANSIENT_GPU_MAX_RETRIES="${TRANSIENT_GPU_MAX_RETRIES:-0}"

echo "RUN_ROOT=$RUN_ROOT"
echo "SEEDS=$SEEDS"
echo "TASKS=$TASKS"
echo "GPU_ID=$GPU_ID"
echo "TASK_CONFIG=$TASK_CONFIG"
echo "CANDIDATE_PRESET=$CANDIDATE_PRESET"
echo "EXECUTE=$EXECUTE"
echo "RUN_ANALYSIS_AFTER=$RUN_ANALYSIS_AFTER"
echo "RESUME_PARTIAL=$RESUME_PARTIAL"
echo "REQUIRE_CANDIDATES_PER_CASE=$REQUIRE_CANDIDATES_PER_CASE"
echo "NUM_SWEEP_SEEDS=$NUM_SWEEP_SEEDS"
echo "RETRY_TRANSIENT_GPU=$RETRY_TRANSIENT_GPU"
echo "TRANSIENT_GPU_RETRY_SECONDS=$TRANSIENT_GPU_RETRY_SECONDS"
echo "TRANSIENT_GPU_MAX_RETRIES=$TRANSIENT_GPU_MAX_RETRIES"
echo "MODE=sequential_bounded"

if [ "$EXECUTE" = "1" ]; then
  echo "Sequential bounded mode: one task runs at a time. With GPU_ID=auto, each task starts only if a GPU is already free."
else
  echo "Dry-run mode: set EXECUTE=1 to run traces."
fi

for task in $TASKS; do
  echo
  echo "=== $task ==="
  if [ "$EXECUTE" = "1" ]; then
    transient_attempt=0
    while true; do
      set +e
      GPU_ID="$GPU_ID" \
      TASK_CONFIG="$TASK_CONFIG" \
      CANDIDATE_PRESET="$CANDIDATE_PRESET" \
      RESUME_PARTIAL="$RESUME_PARTIAL" \
      RUN_ANALYSIS_AFTER=0 \
        scripts/robotwin2_run_clean_traces.sh "$RUN_ROOT" "$task" "$SEEDS"
      status="$?"
      set -e
      if [ "$status" -eq 0 ]; then
        break
      fi
      if [ "$status" -ne 75 ] || [ "$RETRY_TRANSIENT_GPU" != "1" ]; then
        exit "$status"
      fi
      transient_attempt=$((transient_attempt + 1))
      if [ "$TRANSIENT_GPU_MAX_RETRIES" -gt 0 ] && [ "$transient_attempt" -gt "$TRANSIENT_GPU_MAX_RETRIES" ]; then
        echo "$(date -Is 2>/dev/null || date +%Y-%m-%dT%H:%M:%S%z) $task hit transient GPU exit 75 after $TRANSIENT_GPU_MAX_RETRIES retries; exiting"
        exit "$status"
      fi
      echo "$(date -Is 2>/dev/null || date +%Y-%m-%dT%H:%M:%S%z) $task hit transient GPU exit 75; retrying in ${TRANSIENT_GPU_RETRY_SECONDS}s (attempt $transient_attempt)"
      sleep "$TRANSIENT_GPU_RETRY_SECONDS"
    done
  else
    DRY_RUN=1 \
    GPU_ID="$GPU_ID" \
    TASK_CONFIG="$TASK_CONFIG" \
    CANDIDATE_PRESET="$CANDIDATE_PRESET" \
    RESUME_PARTIAL="$RESUME_PARTIAL" \
    RUN_ANALYSIS_AFTER=0 \
      scripts/robotwin2_run_clean_traces.sh "$RUN_ROOT" "$task" "$SEEDS"
  fi
done

if [ "$EXECUTE" = "1" ] && [ "$RUN_ANALYSIS_AFTER" = "1" ]; then
  echo
  echo "=== finalize run ==="
  PYTHON_BIN_CMD="${PYTHON_BIN:-python3}"
  PYTHONPATH=src \
  PYTHON_BIN="$PYTHON_BIN_CMD" \
  REQUIRE_CANDIDATES_PER_CASE="$REQUIRE_CANDIDATES_PER_CASE" \
  NUM_SWEEP_SEEDS="$NUM_SWEEP_SEEDS" \
    scripts/robotwin2_finalize_run.sh "$RUN_ROOT" $TASKS
elif [ "$EXECUTE" = "1" ]; then
  echo
  echo "Run analysis after traces finish with:"
  echo "  PYTHONPATH=src PYTHON_BIN=python3 REQUIRE_CANDIDATES_PER_CASE=$REQUIRE_CANDIDATES_PER_CASE NUM_SWEEP_SEEDS=$NUM_SWEEP_SEEDS scripts/robotwin2_finalize_run.sh $RUN_ROOT $TASKS"
fi
