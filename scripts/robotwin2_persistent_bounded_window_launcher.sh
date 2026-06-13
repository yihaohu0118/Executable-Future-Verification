#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:?usage: robotwin2_persistent_bounded_window_launcher.sh RUN_ROOT}"

RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-300}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"

ts() {
  date -Is 2>/dev/null || date +%Y-%m-%dT%H:%M:%S%z
}

attempt=1
while true; do
  echo "$(ts) persistent attempt=$attempt run_root=$RUN_ROOT tasks=${TASKS:-default} gpu_id=${GPU_ID:-auto}"
  set +e
  scripts/robotwin2_bounded_window_launcher.sh "$RUN_ROOT"
  status="$?"
  set -e
  echo "$(ts) persistent attempt=$attempt status=$status"

  if [ "$status" -eq 0 ]; then
    exit 0
  fi
  if [ "$status" -ne 75 ]; then
    exit "$status"
  fi
  if [ "$MAX_ATTEMPTS" -gt 0 ] && [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
    echo "$(ts) reached MAX_ATTEMPTS=$MAX_ATTEMPTS after retryable GPU-busy/conflict exits" >&2
    exit 75
  fi
  attempt=$((attempt + 1))
  echo "$(ts) retryable GPU-busy/conflict exit; sleeping ${RETRY_SLEEP_SECONDS}s"
  sleep "$RETRY_SLEEP_SECONDS"
done
