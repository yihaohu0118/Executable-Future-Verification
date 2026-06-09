#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/yihao_hyh/umm-reward-evaluator}"
PY="${PY:-/home/yihao_hyh/miniconda3/envs/umm-reward-evaluator/bin/python}"
BRANCH="${BRANCH:-codex/first-layer-pusht100-world-action}"
OUT="${OUT:-outputs/maniskill_pickcube_brittle_grasp_n20_video}"
STATE_OUT="${STATE_OUT:-outputs/maniskill_pickcube_brittle_grasp_n20}"
TMP_PKG="${TMP_PKG:-/tmp/codex_ms_pkg}"

cd "$ROOT"
git fetch origin "$BRANCH"
rm -rf "$TMP_PKG"
mkdir -p "$TMP_PKG/umm_reward_evaluator/benchmarks"
touch "$TMP_PKG/umm_reward_evaluator/__init__.py" "$TMP_PKG/umm_reward_evaluator/benchmarks/__init__.py"
for file in \
  common.py \
  shuffle_manifest_rows.py \
  train_action_sequence_selector.py \
  train_video_frame_selector.py \
  train_action_video_fusion_selector.py; do
  git show "FETCH_HEAD:src/umm_reward_evaluator/benchmarks/$file" > "$TMP_PKG/umm_reward_evaluator/benchmarks/$file"
done

export PYTHONPATH="$TMP_PKG:src"

BASE_VIDEO="$OUT/PickCube-v1_candidate_manifest.jsonl"
BASE_STATE="$STATE_OUT/PickCube-v1_candidate_manifest.jsonl"

"$PY" "$TMP_PKG/umm_reward_evaluator/benchmarks/train_action_sequence_selector.py" \
  --manifest "$BASE_STATE" \
  --output-dir "$STATE_OUT/action_selector_raw_no_length" \
  --feature-mode raw_no_length \
  --epochs 200

"$PY" "$TMP_PKG/umm_reward_evaluator/benchmarks/shuffle_manifest_rows.py" \
  --manifest "$BASE_VIDEO" \
  --output "$OUT/PickCube-v1_candidate_manifest.shuffled_s7.jsonl" \
  --seed 7

"$PY" "$TMP_PKG/umm_reward_evaluator/benchmarks/train_action_sequence_selector.py" \
  --manifest "$OUT/PickCube-v1_candidate_manifest.shuffled_s7.jsonl" \
  --output-dir "$OUT/action_selector_raw_shuffled_s7" \
  --feature-mode raw \
  --epochs 200

"$PY" "$TMP_PKG/umm_reward_evaluator/benchmarks/train_video_frame_selector.py" \
  --manifest "$OUT/PickCube-v1_candidate_manifest.shuffled_s7.jsonl" \
  --output-dir "$OUT/video_selector_raw_shuffled_s7" \
  --feature-mode raw \
  --epochs 200 \
  --image-size 32 \
  --max-frames 6

"$PY" "$TMP_PKG/umm_reward_evaluator/benchmarks/train_action_video_fusion_selector.py" \
  --base-manifest "$BASE_VIDEO" \
  --action-scored-manifest "$STATE_OUT/action_selector_raw_fixed/scored_manifest.jsonl" \
  --video-scored-manifest "$OUT/video_selector_raw_cached/scored_manifest.jsonl" \
  --output-dir "$OUT/action_video_fusion_raw" \
  --feature-set fusion \
  --epochs 200

"$PY" - <<'PY'
import json
from pathlib import Path

paths = {
    "raw_no_length": Path("outputs/maniskill_pickcube_brittle_grasp_n20/action_selector_raw_no_length/summary.json"),
    "action_shuffled": Path("outputs/maniskill_pickcube_brittle_grasp_n20_video/action_selector_raw_shuffled_s7/summary.json"),
    "video_shuffled": Path("outputs/maniskill_pickcube_brittle_grasp_n20_video/video_selector_raw_shuffled_s7/summary.json"),
    "fusion_raw": Path("outputs/maniskill_pickcube_brittle_grasp_n20_video/action_video_fusion_raw/summary.json"),
}
print(json.dumps({name: json.loads(path.read_text()) for name, path in paths.items()}, indent=2, sort_keys=True))
PY

