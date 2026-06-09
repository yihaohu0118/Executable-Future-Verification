# ManiSkill Multi-Task Action Critic Diagnostic

## Question

Does a trained action-sequence failure critic transfer across manipulation tasks, or does it need task-specific calibration?

This experiment combines two diagnostic ManiSkill candidate pools:

- `pick`: randomized PickCube grasp pool, 50 cases, 8 candidates per case.
- `stack`: fixed StackCube brittle-stack pool, 50 cases, 6 candidates per case.

All selectors use leave-one-case-out evaluation. Features are `raw_no_length`, so trajectory length is removed as a shortcut.

## Results

| Selector mode | Overall success | Recovered rank0 failures | Oracle match | Pick success | Stack success |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared MLP + task one-hot | 82/100 | 81/99 | 32/100 | 50/50 | 32/50 |
| Shared trunk + per-task head | 97/100 | 96/99 | 47/100 | 50/50 | 47/50 |
| Independent per-task MLP | 100/100 | 99/99 | 48/100 | 50/50 | 50/50 |

Baselines:

| Baseline | Overall success | Pick success | Stack success |
| --- | ---: | ---: | ---: |
| Rank0 planner candidate | 1/100 | 1/50 | 0/50 |
| Oracle-best candidate | 100/100 | 50/50 | 50/50 |

## Mechanism

The shared one-hot critic solves randomized PickCube but fails badly on StackCube:

- Pick: 50/50 success.
- Stack: 32/50 success.
- Stack choices are spread across `low_grasp`, `slow_center`, `xy_offset`, `center`, and even failed `high_grasp`.

Adding only per-task output heads recovers most of the lost StackCube performance:

- Pick remains 50/50.
- Stack improves from 32/50 to 47/50.
- Stack choices concentrate on `slow_center`, the dominant robust candidate family.

The independent per-task selector reaches the oracle success ceiling on both tasks. This means the action features contain enough information for both tasks; the failure is not due to missing features. The failure is caused by shared calibration across different manipulation geometries.

## Interpretation

The counterintuitive result is that adding a task one-hot is not enough. A single shared decision surface still conflates action statistics that mean different things in different contact modes:

- In PickCube, lower grasp height is usually safer than the brittle high-z rank0 action.
- In StackCube, success depends on a coordinated grasp-place-release sequence, and the same raw action statistics must be calibrated against a different failure mode.

This supports a stronger version of FAVC:

> Action-visual critics should be task-gated failure detectors, not globally shared reward models. Shared representations can be useful, but the final failure calibration should be task- or mode-specific.

## Reproducibility

Script:

- `src/umm_reward_evaluator/benchmarks/train_multitask_action_sequence_selector.py`

Remote outputs:

- `outputs/maniskill_multitask_pick_random_stack_fixed/shared_onehot/summary.json`
- `outputs/maniskill_multitask_pick_random_stack_fixed/per_task_head/summary.json`
- `outputs/maniskill_multitask_pick_random_stack_fixed/independent/summary.json`

Command shape:

```bash
python -m umm_reward_evaluator.benchmarks.train_multitask_action_sequence_selector \
  --manifest pick=outputs/maniskill_pickcube_random_grasp_n50_k8/PickCube-v1_candidate_manifest.jsonl \
  --manifest stack=outputs/maniskill_stackcube_brittle_stack_n50/StackCube-v1_candidate_manifest.jsonl \
  --output-dir outputs/maniskill_multitask_pick_random_stack_fixed/shared_onehot \
  --feature-mode raw_no_length \
  --task-mode shared_onehot \
  --epochs 50
```
