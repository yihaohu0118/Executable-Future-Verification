# ManiSkill Action Critic Training-Case Scaling

## Question

How many held-out training cases does the action-sequence critic need before it can reliably recover rank0 failures?

This diagnostic limits the number of training cases available in each leave-one-case-out fold. It uses the same `raw_no_length` action features as the main action-critic experiments.

## Protocol

- Model: action-sequence MLP selector.
- Features: `raw_no_length`.
- Epochs: 30.
- Repeats: 3 random subsamples per training-case count.
- Evaluation: 50 held-out cases per task.

## PickCube Randomized Grasp Pool

| Train cases per fold | Selector success | Range | Oracle match mean |
| ---: | ---: | ---: | ---: |
| 4 | 50.0/50 | 50-50 | 15.0/50 |
| 8 | 50.0/50 | 50-50 | 13.0/50 |
| 16 | 50.0/50 | 50-50 | 17.0/50 |
| 32 | 50.0/50 | 50-50 | 16.3/50 |
| 49 | 50.0/50 | 50-50 | 16.3/50 |

Interpretation:

- PickCube random-grasp failures are extremely low-sample.
- Four training cases are enough to recover all 49 rank0 failures in all three repeats.
- This is a useful result, but also a warning: PickCube likely contains a simple action-geometry signal, mostly grasp-height related.

## StackCube Brittle-Stack Pool

| Train cases per fold | Selector success | Range | Oracle match mean |
| ---: | ---: | ---: | ---: |
| 4 | 43.0/50 | 42-44 | 28.0/50 |
| 8 | 48.0/50 | 47-49 | 31.0/50 |
| 16 | 49.0/50 | 48-50 | 33.3/50 |
| 32 | 49.3/50 | 49-50 | 33.3/50 |
| 49 | 50.0/50 | 50-50 | 33.7/50 |

Interpretation:

- StackCube is meaningfully harder than PickCube under low-data training.
- The critic still reaches 48/50 with only 8 training cases per fold and 49/50 with 16.
- This supports the failure-detection framing: the model does not need hundreds of task-specific cases, but complex contact sequences need enough examples to calibrate the failure boundary.

## Mechanism

The scaling curves support a refined claim:

> Recoverable manipulation failures expose compact action-geometry signatures. Some tasks are almost one-rule failures, while contact-rich tasks need a small amount of task-specific calibration.

This connects directly to the multi-task result: shared features are useful, but task/mode calibration controls reliability.

## Reproducibility

Script:

- `src/umm_reward_evaluator/benchmarks/train_action_sequence_selector_scaling.py`

Remote outputs:

- `outputs/maniskill_pickcube_random_grasp_n50_k8/action_selector_scaling_e30_r3/summary.json`
- `outputs/maniskill_stackcube_brittle_stack_n50/action_selector_scaling_e30_r3/summary.json`
