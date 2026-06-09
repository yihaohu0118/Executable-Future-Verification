# ManiSkill PickCube N100 Action-Critic Stability

## Purpose

This scales the PickCube brittle-grasp state-only diagnostic from 20 cases to 100 cases. The goal is to check whether the action critic still recovers failures after removing trajectory length as a shortcut.

## Setup

- Benchmark: ManiSkill3 `PickCube-v1`
- Cases: 100 seeds
- Candidates per case: 5
- Candidate source: privileged diagnostic controller
- Rank0 profile: `brittle_grasp`
- Selector: case-heldout action-sequence MLP
- Feature mode: `raw_no_length`
- Training: 50 epochs per held-out case

Remote artifacts:

- `outputs/maniskill_pickcube_brittle_grasp_n100/PickCube-v1_candidate_manifest.jsonl`
- `outputs/maniskill_pickcube_brittle_grasp_n100/action_selector_raw_no_length_e50/summary.json`

## Results

| Metric | Value |
| --- | ---: |
| Rank0 success | 17/100 |
| Oracle-best success | 100/100 |
| Oracle better than rank0 | 100/100 |
| Action critic success | 100/100 |
| Recovered rank0 failures | 83/83 |
| Action critic oracle match | 71/100 |

## Interpretation

The n100 result supports the smaller n20 finding. The trained action critic still recovers every brittle-grasp failure after trajectory length is removed from the input features.

This strengthens the claim that the action critic is learning a reusable action-geometry failure signal, not only a row-order shortcut, candidate-count shortcut, or early-termination shortcut.

## Remaining Caveat

The candidate pool is still privileged and diagnostic. The next required step is not more PickCube seeds; it is a less hand-designed candidate source or a second task with heterogeneous failure modes.

