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

## Remote Environment Status

Verified on dev2 on 2026-06-10:

- Official RoboTwin checkout: `/tmp/robotwin_probe`, commit `c3ddfa8b97d5519efa828b075999bd0006778e5e`.
- Dedicated conda env: `/home/yihao_hyh/miniconda3/envs/robotwin2-favc`, Python 3.10.
- Installed core runtime: `torch==2.4.1+cu121`, `sapien==3.0.0b1`, `mplib==0.2.1`, `open3d==0.18.0`, `gymnasium==0.29.1`, `curobo==v0.7.8` editable checkout, `warp-lang==1.12.0`.
- Official assets downloaded and extracted under `/tmp/robotwin_probe/assets`.
- `script/test_render.py` passes with `Render Well`.
- Task import passes for `open_microwave` and `click_bell`.
- One-episode clean expert smoke passes:

```bash
CUDA_VISIBLE_DEVICES=7 \
timeout 600 \
/home/yihao_hyh/miniconda3/bin/conda run --no-capture-output \
  -n robotwin2-favc \
  python -u script/collect_data.py click_bell demo_clean_smoke
```

Result:

```text
simulate data episode 0 success! (seed = 0)
Complete simulation, failed 0 times / 1 tries
```

Smoke artifacts:

- `data/click_bell/demo_clean_smoke/seed.txt`: `0`
- `data/click_bell/demo_clean_smoke/_traj_data/episode0.pkl`

The environment still prints a SAPIEN Vulkan ICD warning, but rendering works.
`pytorch3d` is also reported missing during import; this did not block task
import, rendering, or the clean expert smoke.

### Curobo H100 Patch

The official curobo install initially failed during planner warmup on H100:

```text
RuntimeError: CUDA error: an illegal instruction was encountered
...
LBFGScu.apply -> lbfgs_step_cu.forward
```

The failure was isolated to curobo's fused LBFGS CUDA extension, not SAPIEN,
assets, or RoboTwin task code. The working patch in our environment is:

1. keep official curobo v0.7.8 and `warp-lang==1.12.0`;
2. set `LBFGSOptConfig.use_cuda_kernel = False` in
   `envs/curobo/src/curobo/opt/newton/lbfgs.py`;
3. set `use_cuda_kernel: False` in curobo task configs under
   `envs/curobo/src/curobo/content/configs/task/*.yml`.

This disables only the fused LBFGS custom kernel and keeps curobo's planning
path active through the PyTorch/JIT fallback. It is slower but sufficient for
small smoke tests and avoids switching RoboTwin to a different planner.

## Candidate Manifest Smoke

The first end-to-end candidate trace smoke was run on the same
`click_bell/demo_clean_smoke/seed=0` case. It used the expert pre-motion pkl to
construct four compact qpos candidates:

| Candidate | Planner rank | Success |
| --- | ---: | --- |
| `expert_endpoints` | 0 | true |
| `reverse_endpoints` | 1 | true |
| `first_endpoint_only` | 2 | false |
| `noop` | 3 | false |

The raw trace was written on dev2:

```text
/tmp/robotwin_probe/umm_candidate_traces/click_bell_clean_smoke.jsonl
```

Local conversion and validation passed:

```bash
PYTHONPATH=src python -m umm_reward_evaluator.benchmarks.robotwin2_trace_to_manifest \
  --input /private/tmp/robotwin2_smoke/click_bell_clean_smoke.jsonl \
  --output-manifest /private/tmp/robotwin2_smoke/click_bell_clean_smoke_manifest.jsonl

PYTHONPATH=src python -m umm_reward_evaluator.benchmarks.validate_future_verification_manifest \
  --manifest /private/tmp/robotwin2_smoke/click_bell_clean_smoke_manifest.jsonl \
  --require-future-metadata
```

Validation summary:

```json
{
  "rows": 4,
  "cases": 1,
  "candidate_count_histogram": {"4": 1},
  "rank0_success": 1,
  "oracle_success": 1,
  "oracle_better": 0,
  "num_errors": 0
}
```

This is only a pipeline smoke, not benchmark evidence. Because rank0 already
succeeds and the reversed endpoint candidate also succeeds, `click_bell` is too
order-insensitive for the main reranking table. The next RoboTwin task should
be more contact/order-sensitive, such as `press_stapler`, `stamp_seal`,
`open_laptop`, `handover_block`, or `stack_blocks_two`.

## Harder Task Candidate Findings

Two additional clean one-seed probes were run after `click_bell`.

### `press_stapler`

Expert seed collection:

```text
simulate data episode 0 fail! (seed = 0)
Error: target_pose cannot be None for move action.
simulate data episode 0 success! (seed = 1)
Complete simulation, failed 1 times / 2 tries
```

Candidate pool from expert pre-motion endpoints:

| Candidate | Planner rank | Success |
| --- | ---: | --- |
| `first_endpoint_rank0` | 0 | false |
| `expert_endpoints` | 1 | true |
| `drop_last_endpoint` | 2 | false |
| `reverse_endpoints` | 3 | true |
| `noop` | 4 | false |

Manifest validation passed with 5 rows, 1 case, no schema errors:

```json
{
  "rank0_success": 0,
  "oracle_success": 1,
  "oracle_better": 1
}
```

Interpretation: `press_stapler` is useful for under-execution headroom, but not
ideal for temporal-order evidence because the reversed endpoint candidate can
still hit the contact success condition.

### `stack_blocks_two`

Expert seed collection:

```text
simulate data episode 0 success! (seed = 0)
Complete simulation, failed 0 times / 1 tries
```

Endpoint-only candidate pool:

| Candidate | Planner rank | Success |
| --- | ---: | --- |
| `first_endpoint_rank0` | 0 | false |
| `expert_endpoints_const_gripper` | 1 | false |
| `first_half` | 2 | false |
| `drop_last_endpoint` | 3 | false |
| `reverse_endpoints` | 4 | false |
| `noop` | 5 | false |

Manifest validation passed with 6 rows, 1 case, no schema errors, but oracle
success was 0/1.

Interpretation: this is an important negative smoke. The official
`_traj_data/episode0.pkl` stores arm joint paths but not enough gripper
open/close semantics to reconstruct a successful stack candidate from endpoints
alone. For multi-stage placement tasks, the RoboTwin2 candidate source must be
either:

1. real policy `take_action(...)` traces from `script/eval_policy.py`; or
2. a deeper expert instrumentation layer that records gripper-aware qpos actions
   during `Base_Task.move(...)`, not just planned joint endpoints.

This is exactly why `script/eval_policy.py` tracing is the right main path for
RoboTwin2 evidence.

### `stack_blocks_two` Gripper-Aware Trace

A second `stack_blocks_two` probe recorded the endpoint of every
`Base_Task.take_dense_action(control_seq)` call during the expert rollout:

```text
expert_record actions 16 success True
info: {"{A}": "red block", "{B}": "green block", "{a}": "left", "{b}": "left"}
```

Each recorded action is a qpos target with both gripper values:

```text
[left_arm_qpos(6), left_gripper, right_arm_qpos(6), right_gripper]
```

Candidate results:

| Candidate | Planner rank | Success |
| --- | ---: | --- |
| `first_action_rank0` | 0 | false |
| `full_gripper_aware` | 1 | true |
| `first_half` | 2 | false |
| `drop_last` | 3 | true |
| `reverse` | 4 | false |
| `noop` | 5 | false |

Manifest validation passed with 6 rows, 1 case, no schema errors:

```json
{
  "rank0_success": 0,
  "oracle_success": 1,
  "oracle_better": 1
}
```

This is the first RoboTwin2 mechanism result that matches the RoboCasa story:

> Endpoint-only futures fail on a multi-stage task, while a compact
> gripper-aware execution trace recovers a successful future; reversing the
> trace fails.

The `drop_last` success is also useful: some post-success retreat actions are
not necessary, so future corruptions should distinguish task-critical contact
and release phases from redundant cleanup motions.

### Gripper-Aware Multi-Task Smoke

The gripper-aware trace probe was then formalized as
`src/umm_reward_evaluator/benchmarks/robotwin2_gripper_aware_trace.py`. The
script should be copied into an official RoboTwin checkout and run from the
RoboTwin root. It records compact qpos targets during expert `play_once()`:

```text
[left_arm_qpos(6), left_gripper, right_arm_qpos(6), right_gripper]
```

It then replays six default candidate futures:

| Candidate | Meaning |
| --- | --- |
| `first_action_rank0` | under-actuated default first candidate |
| `full_gripper_aware` | full recorded expert trace |
| `first_half` | truncated prefix |
| `drop_last` | remove final action |
| `reverse` | temporal reversal |
| `noop` | zero action |

For multi-seed runs, do not assume expert-valid seeds are contiguous. RoboTwin
`collect_data.py` writes successful seeds to `data/<task>/<config>/seed.txt`,
and failed seed attempts can create gaps. Use the batch mode so replay follows
the official valid-seed list:

```bash
python umm_robotwin2_gripper_aware_trace.py \
  --task-name stamp_seal \
  --task-config demo_clean_k5 \
  --instruction "stamp the seal on the target" \
  --all-seeds \
  --max-seeds 5 \
  --output-dir umm_candidate_traces/stamp_seal_k5 \
  --skip-existing
```

One implementation detail mattered for `open_laptop`: the official
`check_success()` expects `self.arm_tag`, which is set inside expert
`play_once()` but not by `setup_demo()` alone. The adapter therefore restores
`env.arm_tag` from the expert info binding `{a}` before replay. This keeps the
official success function unchanged while reproducing task-local state expected
by replay.

Current one-seed gripper-aware results:

| Task | Task type | Rank0 | Full trace | First half | Drop last | Reverse | Noop | Oracle better |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stack_blocks_two` | multi-stage stack | 0 | 1 | 0 | 1 | 0 | 0 | 1 |
| `stamp_seal` | precise contact | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| `open_laptop` | articulated object | 0 | 1 | 0 | 1 | 0 | 0 | 1 |
| `handover_block` | bimanual handover | 0 | 1 | 0 | 0 | 0 | 0 | 1 |

All four traces converted with `robotwin2_trace_to_manifest.py` and validated
with `validate_future_verification_manifest --require-future-metadata`. Each
validated manifest has 6 rows, 1 case, 0 errors, rank0 success 0/1, oracle
success 1/1, and oracle_better 1/1.

Interpretation:

- RoboTwin2 now passes the first kill-line condition: at least four 2025
  benchmark tasks show oracle headroom under fixed expert-valid seeds.
- `stack_blocks_two` proves endpoint-only reconstruction is insufficient for
  multi-stage placement; gripper-aware traces recover the executable future.
- `stamp_seal` and `handover_block` are stricter than the earlier
  `press_stapler` smoke because reverse, half, drop-last, and noop all fail.
- `open_laptop` shows the effect is not limited to rigid pick/place tasks, but
  its `drop_last` success indicates the final actions are cleanup rather than
  task-critical.
- These are mechanism/protocol smokes, not final statistical results. The next
  paper-quality step is to scale each task to multiple seeds and replace the
  oracle/full-trace check with learned selector comparisons against action-only
  and magnitude/smoothness controls.

### Three-Task K=5 Gripper-Aware Result

The first multi-seed run used fixed expert-valid seeds from the official
`seed.txt` files and six candidates per seed. It intentionally excludes
`handover_block` from this K=5 table because that replay was stopped after the
one-seed mechanism result; the current multi-seed evidence is therefore three
tasks, not four.

Local validated manifest:

```text
/private/tmp/robotwin2_k5/robotwin2_three_task_k5_manifest.jsonl
```

Validation summary:

```json
{
  "rows": 90,
  "cases": 15,
  "tasks": {
    "open_laptop": 5,
    "stack_blocks_two": 5,
    "stamp_seal": 5
  },
  "candidate_count_histogram": {"6": 15},
  "rank0_success": 0,
  "oracle_success": 15,
  "oracle_better": 15,
  "num_errors": 0,
  "num_warnings": 0
}
```

Candidate success counts:

| Task | Rank0 | Full trace | First half | Drop last | Reverse | Noop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `stack_blocks_two` | 0/5 | 5/5 | 0/5 | 5/5 | 0/5 | 0/5 |
| `open_laptop` | 0/5 | 5/5 | 0/5 | 5/5 | 0/5 | 0/5 |
| `stamp_seal` | 0/5 | 5/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| **Total** | **0/15** | **15/15** | **0/15** | **10/15** | **0/15** | **0/15** |

Interpretation:

- The recoverable-future headroom is stable across 15 fixed expert-valid cases:
  default rank0 never succeeds, while the candidate pool always contains a
  successful gripper-aware future.
- Temporal reversal never succeeds, so the result is not explained by final
  pose alone.
- `stamp_seal` is the strictest contact task: removing the last action also
  fails in all five seeds.
- `stack_blocks_two` and `open_laptop` expose a useful counterpoint:
  `drop_last` succeeds in all five seeds, showing that some generated futures
  contain redundant cleanup actions after task success. A verifier should
  identify task-critical contact/release phases rather than require exact
  trajectory reproduction.
- This is still an oracle/full-trace mechanism table. The next paper table must
  add learned selector baselines and action-only/magnitude/smoothness controls
  on the same K=5 protocol.

## K=5 Selector Baseline Smoke

The first scaled RoboTwin2 selector evidence uses the same three-task K=5
manifest above. Pure-numpy selector baselines are implemented in
`src/umm_reward_evaluator/benchmarks/robotwin2_selector_baselines.py`.

The main baseline run is:

```bash
PYTHONPATH=src python3 -m umm_reward_evaluator.benchmarks.robotwin2_selector_baselines \
  --manifest /private/tmp/robotwin2_k5/robotwin2_three_task_k5_manifest.jsonl \
  --output-dir /private/tmp/robotwin2_k5/selectors/pure_numpy_baselines
```

Summary results:

| Selector | Success |
| --- | ---: |
| Rank0 | 0/15 |
| Uniform random expected | 4.17/15 |
| Best simple action heuristic, `smoothness_max` | 6/15 |
| Action distribution nearest-positive, same-task | 8/15 |
| State distribution nearest-positive, same-task | 8/15 |
| State distribution nearest-positive, all-task | 8/15 |
| Gripper distribution nearest-positive, same-task | 11/15 |
| Gripper distribution nearest-positive, all-task | 12/15 |
| Phase-gripper distribution nearest-positive, same-task | 13/15 |
| Phase-gripper distribution nearest-positive, all-task | 12/15 |
| Candidate ID `full_gripper_aware` | 15/15 |

The candidate-ID row is an ID-leak upper-bound sanity check, not a method. It
should not be reported as a learned or deployable selector.

Additional centroid controls were run on the same manifest:

| Selector | Success |
| --- | ---: |
| Gripper positive centroid, same-task | 0/15 |
| Gripper positive centroid, all-task | 0/15 |
| Gripper positive-negative centroid, same-task | 2/15 |
| Gripper positive-negative centroid, all-task | 0/15 |
| State positive-negative centroid, same-task | 2/15 |
| State positive-negative centroid, all-task | 1/15 |

The counterintuitive mechanism is that a tiny nearest-positive memory over only
left/right gripper trace distribution beats action statistics and simple
centroids. Adding three coarse temporal phases improves the best real selector
to 13/15, while phase-aware joint or joint+gripper features stay at 10-12/15.
This suggests the useful signal is not "more robot state is always better";
coarse gripper contact timing is currently the cleanest low-dimensional
execution-envelope signal.

Averaging successful traces is harmful, suggesting successful execution
envelopes are multi-modal or phase-specific rather than described by a single
"success centroid."

The main boundary is `open_laptop`: plain all-task gripper nearest-positive
gets only 2/5 on that articulated-object task, and phase-aware gripper improves
it only to 3/5. This is useful rather than fatal: it says gripper timing alone
is not enough when success depends on contact direction and articulated
geometry. The next selector should add compact EEF/contact-direction trace
features and a K-shot calibration curve instead of claiming universal
gripper-only transfer.

### Candidate-ID And Rank Remap Control

The rank randomizer now groups by `(task_name, case_id)`, not only `case_id`.
This matters for RoboTwin2 because different tasks reuse seed strings such as
`seed=0`; grouping only by seed would mix candidates across tasks.

The following control was run with failing candidates forced to rank0 and all
candidate IDs remapped to anonymous `cand_XX` names:

```bash
PYTHONPATH=src python3 -m umm_reward_evaluator.benchmarks.randomize_planner_rank \
  --manifest /private/tmp/robotwin2_k5/robotwin2_three_task_k5_manifest.jsonl \
  --output /private/tmp/robotwin2_k5/robotwin2_three_task_k5_rankrand_remap_seed0.jsonl \
  --mode failure_rank0_shuffle_rest \
  --seed 0 \
  --remap-candidate-ids
```

Validation still passes with 90 rows, 15 cases, six candidates per case,
rank0 success 0/15, oracle success 15/15, and oracle-better 15/15.

Key selector results after candidate-ID remapping for seed 0:

| Selector | Success |
| --- | ---: |
| Rank0 | 0/15 |
| Uniform random expected | 4.17/15 |
| Candidate ID `full_gripper_aware` | 0/15 |
| Best simple action heuristic, `smoothness_max` | 5/15 |
| Gripper distribution nearest-positive, same-task | 11/15 |
| Gripper distribution nearest-positive, all-task | 9/15 |
| Phase-gripper distribution nearest-positive, same-task | 11/15 |
| Phase-joint distribution nearest-positive, all-task | 12/15 |
| Phase-joint+gripper distribution nearest-positive, all-task | 12/15 |

This control changes the interpretation. The fixed-order table shows the
cleanest gripper-only mechanism, but the anonymous rank-remapped table is the
more reviewer-safe result. It proves the signal is not purely candidate-name
leakage, while also showing that the current nearest-positive selector is still
sensitive to candidate ordering/tie-breaking.

A 10-seed anonymous rank/candidate-ID sweep is now implemented in
`src/umm_reward_evaluator/benchmarks/robotwin2_rank_randomization_sweep.py`:

```bash
PYTHONPATH=src python3 -m umm_reward_evaluator.benchmarks.robotwin2_rank_randomization_sweep \
  --manifest /private/tmp/robotwin2_k5/robotwin2_three_task_k5_manifest.jsonl \
  --output /private/tmp/robotwin2_k5/selectors/rankrand_remap_sweep_seed0_9.json \
  --num-seeds 10 \
  --seed-start 0 \
  --mode failure_rank0_shuffle_rest \
  --remap-candidate-ids
```

Mean selector success over seeds 0-9:

| Selector | Mean success |
| --- | ---: |
| Rank0 | 0.0 +/- 0.0 / 15 |
| Candidate ID `full_gripper_aware` | 0.0 +/- 0.0 / 15 |
| Uniform random expected | 4.17 +/- 0.00 / 15 |
| Best simple action heuristic, `smoothness_max` | 5.3 +/- 0.46 / 15 |
| Action distribution nearest-positive, same-task | 6.8 +/- 0.60 / 15 |
| Gripper distribution nearest-positive, same-task | 11.0 +/- 0.00 / 15 |
| Phase-gripper distribution nearest-positive, same-task | 11.4 +/- 0.49 / 15 |
| Phase-joint distribution nearest-positive, all-task | 12.0 +/- 0.00 / 15 |
| Phase-joint+gripper distribution nearest-positive, all-task | 12.0 +/- 0.00 / 15 |

The reviewer-safe current RoboTwin2 table should therefore cite 12.0/15 under
anonymous rank/candidate-ID randomization rather than the fixed-order 13/15.
The remaining weakness is still `open_laptop`: the best remapped all-task
phase-joint selectors average 3/5 there, while `stack_blocks_two` is 4/5 and
`stamp_seal` is 5/5.

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
