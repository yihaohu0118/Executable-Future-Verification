# Executable-Future Verification Manifest Protocol

## Purpose

This protocol turns RoboCasa365, RoboWM-Bench, RoboTwin 2.0, or another modern robotics benchmark into the same candidate-selection problem:

> Given several proposed or imagined futures for the same initial state and instruction, choose the future that is physically executable and completes the task.

It lets the current execution-envelope verifier become benchmark-agnostic. Each benchmark adapter only needs to emit JSONL rows with the fields below.

## Required Row Fields

Each row is one candidate future.

| Field | Meaning |
| --- | --- |
| `benchmark` | Benchmark family, e.g. `robocasa365`, `robowm_bench`, `robotwin2`. |
| `suite` | Split or suite name. |
| `task_name` | Task/environment name. |
| `case_id` | Shared ID for all candidates from the same initial state/instruction. |
| `candidate_id` | Unique candidate ID within the case. |
| `candidate_rank_by_planner` | Rank from the planner/world-model/default scoring rule; rank 0 is the default. |
| `actions` | Executable action sequence, or an extracted/proxy action sequence for generated futures. |
| `oracle_success` | Whether this candidate completes the task under benchmark execution/scoring. |

`case_id` only needs to be unique within a task. All shared tooling groups
candidates by `(task_name, case_id)`, so benchmarks that reuse seed IDs across
tasks do not accidentally mix candidates from different tasks.

Recommended existing fields:

| Field | Meaning |
| --- | --- |
| `instruction` | Natural-language task instruction. |
| `rollout_video_path` | Candidate future video path when available. |
| `rollout_video_layout` | Layout description for multiview videos. |
| `planner_score` | Policy likelihood, visual-realism score, or world-model score. |
| `oracle_return` / `oracle_progress` / `oracle_state_dist` | Extra oracle metrics used only for evaluation/tie-breaking. |

## Recommended Future Metadata

Adapters should put benchmark-specific details under `metadata`:

| Metadata key | Meaning |
| --- | --- |
| `future_source` | `policy_sample`, `demo_replay`, `world_model_video`, `retrieved_video`, `hand_corruption`, etc. |
| `future_representation` | `actions`, `rgb_video`, `state_trace`, `video_to_actions`, etc. |
| `verification_target` | What success means: `task_success`, `physical_executability`, `constraint_satisfaction`, or a benchmark-specific label. |
| `state_trace` | Optional low-dimensional rollout trace for mechanism probes. |
| `generated_video_path` | Optional path to a generated future video. |
| `video_model_name` | Optional world-model identifier. |
| `original_candidate_rank_by_planner` | Useful after candidate/rank randomization controls. |

`validate_future_verification_manifest.py --require-future-metadata` enforces the first three metadata keys.

## Evaluation Contract

All benchmark adapters should support the same core metrics:

- rank0/default success;
- oracle-best success;
- oracle better than rank0;
- selector success;
- recovered rank0 failures;
- oracle match;
- few-shot adaptation curve over target-task cases;
- source-only and no-task-ID transfer controls.

The key comparison is not just "better than rank0". It is whether the verifier improves executable success after shortcut controls:

1. action-magnitude/energy heuristics;
2. candidate-rank/candidate-ID randomization;
3. source-only transfer;
4. no-task-ID versus task-conditioned calibration;
5. task-head versus shared task-conditioned calibration;
6. action-only versus execution-envelope/video/contact evidence.

## Mapping Current RoboCasa365 Results

RoboCasa365 already fits the protocol:

- `benchmark`: `robocasa365`;
- `suite`: target atomic task split;
- `future_source`: demo replay with energy-matched corruptions;
- `future_representation`: actions plus optional low-dimensional state trace;
- `verification_target`: task success in simulation;
- `oracle_success`: RoboCasa `_check_success()`.

The current result is therefore Layer 1 of the world-model story: a controlled executable-future verification mechanism study.

## Mapping RoboWM-Bench

RoboWM-Bench code is accessible at `https://github.com/fffstrong/RoboWM-Bench`. The adapter should emit one row per generated future:

- `future_source`: `world_model_video`;
- `future_representation`: `video_to_actions`;
- `actions`: the action sequence recovered by the RoboWM-Bench video-to-action pipeline;
- `rollout_video_path`: generated world-model future;
- `oracle_success`: execution result in the RoboWM-Bench simulator;
- `planner_score`: visual-realism or world-model likelihood score when available.

The main question becomes:

> Does execution-envelope verification select more physically executable generated futures than visual realism or world-model likelihood?

The concrete code audit and adapter plan is in `docs/robowm_bench_code_audit.md`.

## Mapping RoboTwin 2.0

If RoboWM-Bench is not accessible, RoboTwin 2.0 can be used as the executable-simulation fallback:

- `future_source`: policy proposal, generated expert code rollout, or perturbed expert action;
- `future_representation`: actions and/or RGB rollout;
- `verification_target`: task success in RoboTwin;
- evaluate under domain randomization and task/embodiment shifts.

The main question becomes:

> Does few-shot task/contact calibration still improve future selection under stronger domain randomization and different embodiments?

## Validation

Run:

```bash
python -m umm_reward_evaluator.benchmarks.validate_future_verification_manifest \
  --manifest path/to/candidates.jsonl \
  --require-future-metadata
```

For legacy/current manifests that do not yet contain the recommended metadata, omit `--require-future-metadata` to validate the base candidate-selection schema.

Smoke test on the current RoboCasa365 n16 `PickPlaceCounterToCabinet` hard-negative manifest passes the base schema validator with 128 rows, 16 cases, 8 candidates per case, rank0 success 0/16, and oracle success 16/16.
