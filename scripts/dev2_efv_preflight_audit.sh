#!/usr/bin/env bash
set -euo pipefail

HOME_ROOT="${HOME_ROOT:-/home/yihao_hyh}"
EFV_RUN_ROOT="${EFV_RUN_ROOT:-$HOME_ROOT/efv_runs}"
EFV_REPO="${EFV_REPO:-$HOME_ROOT/Executable-Future-Verification}"

echo "=== host ==="
hostname || true

echo
echo "=== disk ==="
df -h / "$HOME_ROOT" /tmp 2>/dev/null || true

echo
echo "=== gpu memory ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true

echo
echo "=== compute apps ==="
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true

echo
echo "=== tmux sessions ==="
tmux ls 2>/dev/null || true

echo
echo "=== efv repo ==="
if [ -d "$EFV_REPO/.git" ]; then
  git -C "$EFV_REPO" rev-parse --short HEAD || true
  git -C "$EFV_REPO" status --short || true
else
  echo "missing repo: $EFV_REPO"
fi

echo
echo "=== home top-level usage ==="
du -xhd1 "$HOME_ROOT" 2>/dev/null | sort -h | tail -30 || true

echo
echo "=== checkpoint usage ==="
du -xhd2 "$HOME_ROOT/checkpoints" 2>/dev/null | sort -h | tail -50 || true

echo
echo "=== cache usage ==="
du -xhd2 "$HOME_ROOT/.cache" 2>/dev/null | sort -h | tail -50 || true

echo
echo "=== efv run usage ==="
du -xhd2 "$EFV_RUN_ROOT" 2>/dev/null | sort -h | tail -50 || true

echo
echo "=== recent robotwin2 raw counts ==="
find "$EFV_RUN_ROOT" -path "*/raw/*/seed_*.jsonl" -type f -printf "%p " -exec wc -l {} \; 2>/dev/null | tail -80 || true
