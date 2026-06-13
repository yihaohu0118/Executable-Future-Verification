#!/usr/bin/env bash
set -euo pipefail

HOME_ROOT="${HOME_ROOT:-/home/yihao_hyh}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$HOME_ROOT/checkpoints/evogym}"
TOP_N="${TOP_N:-30}"

echo "=== disk ==="
df -h / "$HOME_ROOT" /tmp 2>/dev/null || true

echo
echo "=== gpu compute apps ==="
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true

echo
echo "=== top checkpoint directories ==="
du -xhd2 "$CHECKPOINT_ROOT" 2>/dev/null | sort -h | tail -"$TOP_N" || true

echo
echo "=== likely cleanup candidates by name ==="
find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d \
  \( -name '*smoke*' -o -name '*partial*' -o -name '*repro120*' \) -print0 2>/dev/null \
  | xargs -0 du -xsh 2>/dev/null \
  | sort -h || true

echo
echo "=== approval template ==="
cat <<'EOF'
This script is read-only. It does not delete anything.

If the owner approves deletion, run explicit rm commands for the selected
directories only. Do not delete active checkpoints or GPU0/GPU1 training output
without checking the owner.

Example shape after approval:
  rm -rf -- /home/yihao_hyh/checkpoints/evogym/APPROVED_DIR_NAME
EOF
