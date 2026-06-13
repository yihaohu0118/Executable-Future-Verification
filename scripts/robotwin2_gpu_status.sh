#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found"
  exit 1
fi

GPU_FREE_MAX_MEMORY_MB="${GPU_FREE_MAX_MEMORY_MB:-1024}"

echo "time=$(date -Is)"
echo "gpu_free_max_memory_mb=$GPU_FREE_MAX_MEMORY_MB"
echo "gpu_index,name,memory_used_mb,memory_total_mb,utilization_gpu"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

free_gpus=()
while IFS= read -r gpu; do
  [ -z "$gpu" ] && continue
  busy="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  memory_used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  if [ -z "$busy" ] && [ -n "$memory_used" ] && [ "$memory_used" -le "$GPU_FREE_MAX_MEMORY_MB" ]; then
    free_gpus+=("$gpu")
  fi
done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)

echo
if [ "${#free_gpus[@]}" -eq 0 ]; then
  echo "free_gpus=none"
  echo "recommendation=do_not_start_robotwin2"
else
  echo "free_gpus=${free_gpus[*]}"
  echo "recommendation=robotwin2_auto_launcher_can_start"
fi
