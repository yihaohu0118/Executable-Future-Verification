# Implementation Plan

## Phase 1: Offline Evaluator Pipeline

Implemented modules:

- `umm_reward_evaluator.manifest`
- `umm_reward_evaluator.hard_negatives`
- `umm_reward_evaluator.evaluator.openai_compatible`
- `umm_reward_evaluator.analysis.correlate`
- `umm_reward_evaluator.analysis.rerank`
- `umm_reward_evaluator.analysis.pairwise_negatives`

Expected flow:

```bash
python -m umm_reward_evaluator.hard_negatives \
  --manifest outputs/manifests/pusht_rollouts.jsonl \
  --output outputs/manifests/pusht_hard_negatives.jsonl \
  --include-originals

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

The current pipeline expects a rollout manifest. The next implementation task is
a NanoWM exporter that runs or wraps NanoWM planning, decodes candidate rollouts
to frames/videos, and writes `outputs/manifests/pusht_rollouts.jsonl`.

That exporter should not modify NanoWM initially. It should either:

1. call NanoWM scripts and collect their saved outputs, or
2. import NanoWM planning modules and add a local decode/export path.

The safer first choice is script-level wrapping.
