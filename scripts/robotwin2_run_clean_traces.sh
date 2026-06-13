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
GPU_STABLE_SECONDS="${GPU_STABLE_SECONDS:-30}"
GPU_FREE_MAX_MEMORY_MB="${GPU_FREE_MAX_MEMORY_MB:-1024}"
RUN_ANALYSIS_AFTER="${RUN_ANALYSIS_AFTER:-0}"
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_SEED_ERROR="${CONTINUE_ON_SEED_ERROR:-1}"
RESUME_PARTIAL="${RESUME_PARTIAL:-0}"
GPU_CONFLICT_MONITOR="${GPU_CONFLICT_MONITOR:-1}"
GPU_CONFLICT_CHECK_SECONDS="${GPU_CONFLICT_CHECK_SECONDS:-30}"
GPU_CONFLICT_TERM_GRACE_SECONDS="${GPU_CONFLICT_TERM_GRACE_SECONDS:-10}"

RAW_DIR="$RUN_ROOT/raw/$TASK_NAME"
LOG_DIR="$RUN_ROOT/logs"
LOG_FILE="$LOG_DIR/${TASK_NAME}_${CANDIDATE_PRESET}_seeds_${SEEDS//[^0-9A-Za-z_-]/_}.log"
CONFLICT_FILE="$LOG_DIR/.${TASK_NAME}_${CANDIDATE_PRESET}_seeds_${SEEDS//[^0-9A-Za-z_-]/_}.gpu_conflict"

mkdir -p "$RAW_DIR" "$LOG_DIR"

find_free_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  for gpu in $(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null); do
    if gpu_is_free "$gpu"; then
      echo "$gpu"
      return 0
    fi
  done
  return 1
}

gpu_is_free() {
  gpu="$1"
  busy="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  memory_used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  [ -z "$busy" ] && [ -n "$memory_used" ] && [ "$memory_used" -le "$GPU_FREE_MAX_MEMORY_MB" ]
}

foreign_compute_apps() {
  gpu="$1"
  own_pid="$2"
  nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F, -v own_pid="$own_pid" '
      {
        pid=$1
        gsub(/[[:space:]]/, "", pid)
        if (pid != "" && pid != own_pid) {
          print $0
        }
      }
    '
}

if [ "$GPU_ID" = "auto" ] && [ "$DRY_RUN" != "1" ]; then
  selected_gpu="$(find_free_gpu || true)"
  if [ -z "$selected_gpu" ]; then
    echo "$(date -Is) no free GPU found for $TASK_NAME; exiting without starting simulation" >&2
    exit 75
  fi
  if [ "$GPU_STABLE_SECONDS" -gt 0 ]; then
    echo "$(date -Is) GPU $selected_gpu looks free; rechecking after ${GPU_STABLE_SECONDS}s"
    sleep "$GPU_STABLE_SECONDS"
    if ! gpu_is_free "$selected_gpu"; then
      echo "$(date -Is) GPU $selected_gpu no longer free for $TASK_NAME; exiting without starting simulation" >&2
      exit 75
    fi
  fi
  GPU_ID="$selected_gpu"
fi

cmd=(
  python -m umm_reward_evaluator.benchmarks.robotwin2_gripper_aware_trace
  --task-name "$TASK_NAME"
  --task-config "$TASK_CONFIG"
  --seeds "$SEEDS"
  --output-dir "$RAW_DIR"
  --candidate-preset "$CANDIDATE_PRESET"
  --skip-existing
)

if [ "$CONTINUE_ON_SEED_ERROR" = "1" ]; then
  cmd+=(--continue-on-seed-error)
fi
if [ "$RESUME_PARTIAL" = "1" ]; then
  cmd+=(--resume-partial)
fi

echo "RUN_ROOT=$RUN_ROOT"
echo "TASK_NAME=$TASK_NAME"
echo "SEEDS=$SEEDS"
echo "GPU_ID=$GPU_ID"
echo "ROBOTWIN_ROOT=$ROBOTWIN_ROOT"
echo "EFV_ROOT=$EFV_ROOT"
echo "LOG_FILE=$LOG_FILE"
echo "CONTINUE_ON_SEED_ERROR=$CONTINUE_ON_SEED_ERROR"
echo "RESUME_PARTIAL=$RESUME_PARTIAL"
echo "GPU_CONFLICT_MONITOR=$GPU_CONFLICT_MONITOR"
printf 'COMMAND='
printf '%q ' "CUDA_VISIBLE_DEVICES=$GPU_ID" "${cmd[@]}"
printf '\n'

if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

if [ "$WAIT_FOR_GPU" = "1" ]; then
  while ! gpu_is_free "$GPU_ID"; do
    memory_used="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || true)"
    busy="$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || true)"
    echo "$(date -Is) GPU $GPU_ID not free; memory_used_mb=${memory_used:-unknown}; compute_pids=${busy:-none}; waiting ${GPU_WAIT_SECONDS}s"
    sleep "$GPU_WAIT_SECONDS"
  done
  if [ "$GPU_STABLE_SECONDS" -gt 0 ]; then
    echo "$(date -Is) GPU $GPU_ID is free; rechecking after ${GPU_STABLE_SECONDS}s"
    sleep "$GPU_STABLE_SECONDS"
    if ! gpu_is_free "$GPU_ID"; then
      echo "$(date -Is) GPU $GPU_ID no longer free for $TASK_NAME; exiting without starting simulation" >&2
      exit 75
    fi
  fi
fi

source /home/yihao_hyh/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export PYTHONPATH="$EFV_ROOT/src:${PYTHONPATH:-}"

cd "$ROBOTWIN_ROOT"
echo "$(date -Is) start $TASK_NAME seeds=$SEEDS preset=$CANDIDATE_PRESET" | tee "$LOG_FILE"
rm -f "$CONFLICT_FILE"
CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}" > >(tee -a "$LOG_FILE") 2>&1 &
child_pid="$!"
monitor_pid=""
if [ "$GPU_CONFLICT_MONITOR" = "1" ]; then
  (
    while kill -0 "$child_pid" >/dev/null 2>&1; do
      foreign="$(foreign_compute_apps "$GPU_ID" "$child_pid" || true)"
      if [ -n "$foreign" ]; then
        {
          echo "$(date -Is) foreign GPU compute app detected on GPU $GPU_ID while $TASK_NAME is running; terminating own child $child_pid"
          echo "$foreign"
        } | tee -a "$LOG_FILE"
        printf '%s\n' "$foreign" > "$CONFLICT_FILE"
        kill -TERM "$child_pid" >/dev/null 2>&1 || true
        sleep "$GPU_CONFLICT_TERM_GRACE_SECONDS"
        if kill -0 "$child_pid" >/dev/null 2>&1; then
          echo "$(date -Is) own child $child_pid still alive after TERM; sending KILL" | tee -a "$LOG_FILE"
          kill -KILL "$child_pid" >/dev/null 2>&1 || true
        fi
        exit 0
      fi
      sleep "$GPU_CONFLICT_CHECK_SECONDS"
    done
  ) &
  monitor_pid="$!"
fi
set +e
wait "$child_pid"
child_status="$?"
set -e
if [ -n "$monitor_pid" ]; then
  kill "$monitor_pid" >/dev/null 2>&1 || true
  wait "$monitor_pid" >/dev/null 2>&1 || true
fi
if [ -f "$CONFLICT_FILE" ]; then
  echo "$(date -Is) stopped $TASK_NAME because another compute app appeared on GPU $GPU_ID" | tee -a "$LOG_FILE"
  exit 75
fi
if [ "$child_status" -ne 0 ]; then
  echo "$(date -Is) $TASK_NAME failed with exit code $child_status" | tee -a "$LOG_FILE"
  exit "$child_status"
fi
echo "$(date -Is) finish $TASK_NAME seeds=$SEEDS preset=$CANDIDATE_PRESET" | tee -a "$LOG_FILE"

if [ "$RUN_ANALYSIS_AFTER" = "1" ]; then
  cd "$EFV_ROOT"
  PYTHONPATH=src scripts/robotwin2_multitask_analysis.sh "$RUN_ROOT" "$TASK_NAME" 2>&1 | tee -a "$LOG_FILE"
fi
