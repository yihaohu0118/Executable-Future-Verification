# ManiSkill PickCube Brittle-Grasp Headroom

## Purpose

This is the first external-benchmark headroom probe for the ICLR path. It is not a final method result. The goal is to verify that a recognized manipulation benchmark contains recoverable candidate-selection failures before training a reranker.

Benchmark:

- ManiSkill3 `PickCube-v1`
- 20 deterministic seeds
- 5 executable candidates per seed
- Oracle: simulator `success`, dense return, and final object-goal distance
- Candidate source: privileged diagnostic controller
- Rank0 profile: `brittle_grasp`

The `brittle_grasp` profile intentionally ranks a high-grasp candidate first. This is a mechanism probe, not a deployable planner baseline.

## Summary

| Metric | Value |
| --- | ---: |
| Cases | 20 |
| Candidate rows | 100 |
| Rank0 success | 4/20 |
| Oracle-best success | 20/20 |
| Oracle better than rank0 | 20/20 |
| Rank0 oracle match | 0/20 |

## Candidate-Level Results

| Candidate | Success | Mean progress | Mean final dist |
| --- | ---: | ---: | ---: |
| `high_grasp` | 4/20 | 0.0622 | 0.1303 |
| `low_grasp` | 20/20 | 0.1833 | 0.0091 |
| `rank0_center` | 20/20 | 0.1778 | 0.0146 |
| `slow_center` | 20/20 | 0.1758 | 0.0167 |
| `x_offset` | 20/20 | 0.1739 | 0.0185 |

Oracle-best distribution:

| Oracle-best candidate | Cases |
| --- | ---: |
| `low_grasp` | 16 |
| `rank0_center` | 3 |
| `slow_center` | 1 |

## Code-Driven Observation

A small grasp-height choice dominates success. The high-grasp candidate is close enough to look plausible but fails in 16/20 seeds, while lower grasp candidates solve every seed.

This gives a concrete reranking problem:

> Can a trained visual/action selector detect that the high-grasp trajectory is a recoverable failure and override it with a lower-grasp candidate?

This is aligned with the PushT finding: the trained component should be a failure override, not a global replacement.

## Next Experiment

Use this manifest to train and test held-out selectors:

1. Static visual progress selector from rendered candidate videos.
2. Action-sequence world critic trained on raw action trajectories.
3. Failure gate: use action-world only when static visual progress predicts a failure.
4. Negative control: shuffled action sequences and candidate-id-free features.

Required success criterion:

- Improve rank0 success from 4/20 toward oracle-best 20/20.
- Report hard-case recovery on the 16 cases where rank0 fails.
- Avoid relying on candidate id or privileged metadata.

Remote artifacts:

- `outputs/maniskill_pickcube_brittle_grasp_n20/PickCube-v1_candidate_manifest.jsonl`
- `outputs/maniskill_pickcube_brittle_grasp_n20/summary.json`

