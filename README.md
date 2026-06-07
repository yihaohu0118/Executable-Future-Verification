# UMM Reward Evaluator for World Models

This repository is a research workspace for studying unified multimodal models
(UMMs) as evaluators and reward models for action-conditioned world-model
rollouts.

The initial target project is:

> NanoWM + UMM Reward Evaluator

Use `nano-world-model` as the rollout generator, borrow long-horizon
consistency ideas from `Echo-Memory`, and evaluate whether a multimodal
evaluator can score rollouts more usefully than pixel-level metrics.

## Core Question

Can a multimodal evaluator predict task success, action consistency, temporal
consistency, and memory consistency of generated world-model rollouts well
enough to improve model selection, rollout reranking, or planning?

## Initial Milestones

1. Build a benchmark of positive and negative world-model rollouts.
2. Prompt or adapt UMM/VLM evaluators to score rollout quality.
3. Compare UMM rewards against pixel metrics and oracle task success.
4. Use UMM reward for rollout reranking or MPC/CEM planning.

## Repository Layout

- `docs/proposal.md`: research proposal and method design.
- `docs/experiment_plan.md`: concrete experiments, baselines, and metrics.
- `docs/code_survey.md`: code-level notes from NanoWM and Echo-Memory.
- `docs/implementation_plan.md`: current implementation plan and CLI flow.
- `src/`: evaluator, hard-negative, manifest, and analysis code.
- `configs/`: model and experiment config templates.
- `data/`: local metadata or small synthetic benchmark manifests.

## Quick Start

Install the local package:

```bash
pip install -e .
```

Convert NanoWM planning outputs to a rollout manifest:

```bash
python -m umm_reward_evaluator.exporters.nanowm_planning \
  --planning-dir /path/to/planning_results \
  --output outputs/manifests/pusht_rollouts.jsonl \
  --extract-frames-root outputs/frames/pusht
```

Given a rollout manifest, create hard negatives:

```bash
python -m umm_reward_evaluator.hard_negatives \
  --manifest outputs/manifests/pusht_rollouts.jsonl \
  --output outputs/manifests/pusht_hard_negatives.jsonl \
  --include-originals
```

Compute pixel baselines:

```bash
python -m umm_reward_evaluator.metrics.pixel \
  --manifest outputs/manifests/pusht_hard_negatives.jsonl \
  --output outputs/scores/pixel_scores.jsonl
```

Score rollouts with an OpenAI-compatible multimodal endpoint:

```bash
python -m umm_reward_evaluator.evaluator.openai_compatible \
  --manifest outputs/manifests/pusht_hard_negatives.jsonl \
  --output outputs/scores/umm_scores.jsonl \
  --api-base http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-VL-72B-Instruct
```

Analyze evaluator usefulness:

```bash
python -m umm_reward_evaluator.analysis.correlate --scores outputs/scores/umm_scores.jsonl
python -m umm_reward_evaluator.analysis.pairwise_negatives --scores outputs/scores/umm_scores.jsonl
python -m umm_reward_evaluator.analysis.rerank --scores outputs/scores/umm_scores.jsonl
```

For UMM-vs-VLM comparisons, run the same manifest through multiple evaluator
configs and compare their outputs with the analysis scripts.

## Reference Projects

- Nano World Model: https://github.com/simchowitzlabpublic/nano-world-model
- Echo Memory: https://github.com/Echo-Team-Joy-Future-Academy-JD/Echo-Memory
- World-Env: https://github.com/OpenDriveLab/World-Env
- RoboMeter: https://github.com/robometer/robometer
