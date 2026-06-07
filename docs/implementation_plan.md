# Implementation Plan

## Phase 1: Offline Evaluator Pipeline

Implemented modules:

- `umm_reward_evaluator.manifest`
- `umm_reward_evaluator.exporters.nanowm_planning`
- `umm_reward_evaluator.hard_negatives`
- `umm_reward_evaluator.evaluator.openai_compatible`
- `umm_reward_evaluator.metrics.pixel`
- `umm_reward_evaluator.analysis.correlate`
- `umm_reward_evaluator.analysis.rerank`
- `umm_reward_evaluator.analysis.pairwise_negatives`

Expected flow:

```bash
python -m umm_reward_evaluator.exporters.nanowm_planning \
  --planning-dir /path/to/nanowm/planning_results \
  --output outputs/manifests/pusht_rollouts.jsonl \
  --extract-frames-root outputs/frames/pusht

python -m umm_reward_evaluator.hard_negatives \
  --manifest outputs/manifests/pusht_rollouts.jsonl \
  --output outputs/manifests/pusht_hard_negatives.jsonl \
  --include-originals

python -m umm_reward_evaluator.metrics.pixel \
  --manifest outputs/manifests/pusht_hard_negatives.jsonl \
  --output outputs/scores/pixel_scores.jsonl

python -m umm_reward_evaluator.evaluator.openai_compatible \
  --manifest outputs/manifests/pusht_hard_negatives.jsonl \
  --output outputs/scores/umm_scores.jsonl \
  --api-base http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-VL-72B-Instruct

python -m umm_reward_evaluator.analysis.correlate \
  --scores outputs/scores/umm_scores.jsonl

python -m umm_reward_evaluator.analysis.pairwise_negatives \
  --scores outputs/scores/umm_scores.jsonl

python -m umm_reward_evaluator.analysis.rerank \
  --scores outputs/scores/umm_scores.jsonl
```

## Next Missing Piece

The current exporter converts NanoWM episode-level planning outputs into a
manifest. The next implementation task is candidate-level export from inside
NanoWM CEM planning, so reranking can compare multiple action candidates per
initial state instead of only scoring the final planned episode.

That candidate exporter should initially avoid invasive NanoWM changes. It can:

1. wrap a modified CEM top-k export path in a local adapter, or
2. import NanoWM planning modules and add a local decode/export path.

The episode-level exporter is sufficient for first-pass correlation and
hard-negative experiments.
