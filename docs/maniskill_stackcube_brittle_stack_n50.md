# ManiSkill StackCube Brittle-Stack N50

## Purpose

This is the second ManiSkill task for the FAVC evidence chain. It tests whether the trained action critic transfers beyond PickCube to a harder stacking task with multiple failure modes.

## Setup

- Benchmark: ManiSkill3 `StackCube-v1`
- Cases: 50 seeds
- Candidates per case: 6
- Candidate source: privileged diagnostic controller
- Rank0 profile: `brittle_stack`
- Rank0 candidate: `high_grasp`
- Selector: case-heldout action-sequence MLP
- Feature mode: `raw_no_length`
- Training: 50 epochs per held-out case

Remote artifacts:

- `outputs/maniskill_stackcube_brittle_stack_n50/StackCube-v1_candidate_manifest.jsonl`
- `outputs/maniskill_stackcube_brittle_stack_n50/action_selector_raw_no_length_e50/summary.json`

## Results

| Metric | Value |
| --- | ---: |
| Cases | 50 |
| Candidate rows | 300 |
| Rank0 success | 0/50 |
| Oracle-best success | 50/50 |
| Oracle better than rank0 | 50/50 |
| Action critic success | 50/50 |
| Recovered rank0 failures | 50/50 |
| Action critic oracle match | 35/50 |

## Candidate-Level Results

| Candidate | Success | Mean final dist | Mean progress |
| --- | ---: | ---: | ---: |
| `high_grasp` | 0/50 | 0.1825 | -0.0008 |
| `center` | 38/50 | 0.0445 | 0.1371 |
| `low_grasp` | 49/50 | 0.0082 | 0.1735 |
| `xy_offset` | 2/50 | 0.1092 | 0.0724 |
| `no_release` | 0/50 | 0.0486 | 0.1330 |
| `slow_center` | 50/50 | 0.0016 | 0.1801 |

Oracle-best distribution:

| Candidate | Cases |
| --- | ---: |
| `slow_center` | 34 |
| `low_grasp` | 9 |
| `center` | 7 |

Action-critic selected candidates:

| Candidate | Cases |
| --- | ---: |
| `slow_center` | 49 |
| `low_grasp` | 1 |

## Interpretation

StackCube gives a stronger second-task diagnostic than PushCube. PushCube rank0 already solved all cases, but StackCube has clear recoverable failures:

- `high_grasp` fails every seed;
- `no_release` also fails every seed;
- `xy_offset` mostly fails;
- `slow_center` and `low_grasp` are reliable recovery candidates.

The no-length action critic recovers all 50 failures and mostly selects `slow_center`, the strongest candidate family. This supports the claim that FAVC can learn action-geometry failure signals beyond a single PickCube grasp-height artifact.

## Remaining Caveat

This is still a privileged diagnostic candidate pool. The next ICLR-critical step is to replace hand-designed candidate families with policy-generated top-k candidates or randomized candidate-family positions.

