# RoboTwin 2.0 Executable-Future Adapter

## Code-Grounded Finding

RoboTwin 2.0 is the most practical second benchmark after RoboCasa365 if RoboWM-Bench code/data remain unavailable. A shallow audit of the official repository shows:

- `README.md` identifies RoboTwin 2.0 as the latest 2025 benchmark branch, with 50 tasks, multiple policy baselines, a public leaderboard, and strong domain randomization.
- `collect_data.sh` calls `script/collect_data.py`, which first searches for expert-valid seeds and then collects trajectories.
- `script/eval_policy.py` also performs an expert-valid seed check before policy evaluation. The policy is evaluated on seeds where `TASK_ENV.plan_success and TASK_ENV.check_success()` already hold for the expert.
- `task_config/demo_randomized.yml` enables cluttered tables, random background, random lighting, random table height, RGB, qpos, endpose, and video logging.
- `envs/_base_task.py` exposes `get_obs()` with qpos/endpose and `take_action(action, action_type=...)`; success is exposed through `TASK_ENV.eval_success` and `TASK_ENV.check_success()`.

This means RoboTwin should not be used as "can our method find feasible expert seeds?" The fair problem is:

> Fix expert-valid initial seeds and instructions, generate several policy/world-model candidate futures for each seed, then select the candidate that executes successfully.

## Why It Fits The Current Story

RoboCasa365 already shows that action-envelope calibration can recover conservative-prior failures, but hard negatives require compact robot execution-envelope/contact evidence. RoboTwin 2.0 adds:

- a 2025 benchmark with public code and leaderboard;
- dual-arm tasks rather than kitchen single-arm tasks;
- stronger domain randomization than the current RoboCasa probe;
- multiple policy baselines that can provide candidate futures.

The intended claim is not real-robot deployment. The intended claim is executable-future verification under modern simulation and world-model style candidate generation.

## Adapter Contract

Instrument RoboTwin evaluation to emit one JSON record per candidate rollout:

```json
{
  "task_name": "open_microwave",
  "task_config": "demo_randomized",
  "seed": 100001,
  "instruction": "open the microwave",
  "policy_name": "DP",
  "ckpt_setting": "baseline",
  "candidate_seed": 0,
  "candidate_rank_by_planner": 0,
  "video_path": "eval_result/.../episode0.mp4",
  "actions": [[0.0, 0.1]],
  "success": false,
  "action_type": "qpos"
}
```

Convert these traces with:

```bash
python -m umm_reward_evaluator.benchmarks.robotwin2_trace_to_manifest \
  --input-dir /path/to/robotwin_candidate_traces \
  --output-manifest /path/to/robotwin2_candidate_manifest.jsonl

python -m umm_reward_evaluator.benchmarks.validate_future_verification_manifest \
  --manifest /path/to/robotwin2_candidate_manifest.jsonl \
  --require-future-metadata
```

The converter fills the shared future metadata:

- `future_source`: `policy_rollout` by default;
- `future_representation`: `actions` by default;
- `verification_target`: `task_success` by default.

## Minimal Instrumentation Point

The least invasive patch is inside RoboTwin `script/eval_policy.py`:

1. Initialize a list before each policy rollout.
2. Wrap or monkey-patch `TASK_ENV.take_action` so every action passed by a policy is appended to that list.
3. After the rollout, write a candidate JSON record with task, seed, instruction, candidate id, video path, actions, success flag, and optional qpos/endpose summaries from `TASK_ENV.now_obs`.

This records the proposal as the policy actually executed it, after the same expert-valid seed filter used by the official benchmark.

Do not record the expert-valid seed filtering trajectory as a candidate. The
candidate pool starts only inside the policy-evaluation loop after a seed has
passed `TASK_ENV.plan_success and TASK_ENV.check_success()`.

### Concrete Patch Skeleton

Add these helpers near the top of `script/eval_policy.py`:

```python
import json
from pathlib import Path

import numpy as np


def _jsonable_array(value, max_items=256):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size > max_items:
        array = array[:max_items]
    return array.tolist()


def _compact_robot_state(task_env):
    obs = getattr(task_env, "now_obs", None)
    if not isinstance(obs, dict):
        return {}
    out = {}
    for key in ("qpos", "endpose", "left_arm_qpos", "right_arm_qpos", "left_endpose", "right_endpose"):
        if key in obs:
            out[key] = _jsonable_array(obs[key])
    return out
```

Then wrap each policy rollout:

```python
candidate_actions = []
candidate_state_trace = []
original_take_action = TASK_ENV.take_action


def traced_take_action(action, action_type="qpos"):
    candidate_actions.append(_jsonable_array(action, max_items=128))
    candidate_state_trace.append(_compact_robot_state(TASK_ENV))
    return original_take_action(action, action_type=action_type)


TASK_ENV.take_action = traced_take_action
try:
    # existing official policy loop:
    # while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
    #     observation = TASK_ENV.get_obs()
    #     eval_func(TASK_ENV, model, observation)
    pass
finally:
    TASK_ENV.take_action = original_take_action
```

After the rollout finishes and `succ` has been computed, write one JSON object:

```python
trace_record = {
    "task_name": args["task_name"],
    "task_config": args.get("task_config", "demo_randomized"),
    "seed": now_seed,
    "instruction": instruction,
    "policy_name": args["policy_name"],
    "ckpt_setting": str(ckpt_setting),
    "candidate_seed": candidate_seed,
    "candidate_rank_by_planner": candidate_rank,
    "video_path": TASK_ENV.eval_video_path,
    "actions": candidate_actions,
    "success": bool(succ),
    "action_type": args.get("action_type", "qpos"),
    "metadata": {
        "future_source": "policy_rollout",
        "future_representation": "actions_and_state_trace",
        "verification_target": "task_success",
        "state_trace": candidate_state_trace,
    },
}
output_path = Path(save_dir) / "umm_candidate_traces.jsonl"
with output_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(trace_record, sort_keys=True) + "\n")
```

`candidate_seed` and `candidate_rank` should be assigned by the candidate
generation wrapper. For stochastic policies, use the sampling seed as
`candidate_seed`; for checkpoint variants, use the checkpoint order as the
planner rank. For pure action-noise candidates, keep rank 0 as the unperturbed
policy and assign larger ranks to perturbations before running any selector.

## First Experiment Matrix

Use 4-6 tasks chosen to stress different contact regimes:

| Task type | Candidate tasks |
| --- | --- |
| articulated object | `open_microwave`, `open_laptop`, `turn_switch` |
| precise contact | `press_stapler`, `click_bell`, `stamp_seal` |
| bimanual handover/coordination | `handover_block`, `handover_mic` |
| stacking/placement | `stack_blocks_two`, `place_object_basket` |

For each task:

- fixed expert-valid seeds: 25 to start, 100 for paper table;
- candidates per seed: 4-8;
- rank0: default policy/checkpoint/planner score;
- candidates: checkpoint variants, stochastic policy samples, action-noise samples, or generated future rollouts if a world-model source is available.

The first smoke should be deliberately small:

1. one task, five expert-valid seeds, four candidates per seed;
2. convert traces with `robotwin2_trace_to_manifest.py`;
3. validate with `--require-future-metadata`;
4. report only rank0, oracle-best, and candidate-count histogram.

Scale only if oracle-best improves over rank0. If the smoke has no headroom,
switch tasks or candidate-generation source before training any selector.

## Required Controls

The benchmark is only useful if oracle-best beats rank0. For each task, report:

- rank0 success;
- oracle-best success;
- action magnitude / smoothness heuristic;
- action-only endpoint-free selector;
- execution-envelope selector using qpos/endpose summaries;
- no-task-ID source-only transfer;
- few-shot task/contact calibration.

If oracle-best does not improve over rank0, that task should be excluded from the main reranking table and kept only as a negative-control task.

## Counterintuitive Claim To Test

The RoboTwin-specific hypothesis should mirror the RoboCasa mechanism but under stronger randomization:

> More visual realism or task-specific heads may not be the bottleneck. A compact few-shot execution-envelope verifier over end-effector/gripper or qpos traces can recover candidate futures across randomized dual-arm tasks, while no-task-ID transfer and action-only statistics remain brittle.

This is publishable only if it survives:

- fixed expert-valid seeds;
- randomized candidate IDs/ranks;
- action-magnitude controls;
- held-out task or held-out domain-randomization splits;
- small K calibration curves.
