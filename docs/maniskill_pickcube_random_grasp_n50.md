# ManiSkill PickCube Randomized Grasp Pool N50

## Purpose

This is a stronger control than the fixed-family PickCube brittle-grasp pool. Each case has one brittle rank0 candidate plus continuously sampled counterfactual grasp candidates. This reduces the risk that the action critic only learns a fixed candidate-family label.

## Setup

- Benchmark: ManiSkill3 `PickCube-v1`
- Cases: 50 seeds
- Candidates per case: 8
- Candidate source: randomized privileged diagnostic controller
- Rank0: sampled high-grasp candidate
- Non-rank0: continuously sampled grasp height, xy offset, and gain
- Selector: case-heldout action-sequence MLP
- Feature mode: `raw_no_length`
- Training: 50 epochs per held-out case

Remote artifacts:

- `outputs/maniskill_pickcube_random_grasp_n50_k8/PickCube-v1_candidate_manifest.jsonl`
- `outputs/maniskill_pickcube_random_grasp_n50_k8/action_selector_raw_no_length_e50/summary.json`

## Results

| Metric | Value |
| --- | ---: |
| Cases | 50 |
| Candidate rows | 400 |
| Rank0 success | 1/50 |
| Oracle-best success | 50/50 |
| Oracle better than rank0 | 50/50 |
| Action critic success | 50/50 |
| Recovered rank0 failures | 49/49 |
| Action critic oracle match | 14/50 |

## Parameter Diagnostics

| Group | N | Success | Mean grasp z | Min z | Max z | Mean gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rank0 | 50 | 1 | 0.0570 | 0.0481 | 0.0643 | 17.37 |
| Non-rank0 | 350 | 211 | 0.0366 | 0.0240 | 0.0519 | 17.14 |
| Successful candidates | 212 | 212 | 0.0324 | 0.0240 | 0.0536 | 17.31 |
| Failed candidates | 188 | 0 | 0.0467 | 0.0247 | 0.0643 | 17.01 |

Selector candidate-rank distribution:

| Candidate rank token | Selected cases |
| --- | ---: |
| `01` | 21 |
| `05` | 8 |
| `02` | 6 |
| `03` | 5 |
| `04` | 5 |
| `06` | 2 |
| `07` | 2 |
| `00` | 1 |

## Interpretation

The action critic still recovers all failures when candidates are continuously sampled rather than fixed families. The selector does not always pick the same candidate index; it chooses across multiple sampled candidates.

The diagnostic supports the core mechanism:

> recoverable manipulation failures can be detected from action-geometry traces, even when the recovery candidate is not a fixed hand-designed family.

## Remaining Caveat

The candidates are still generated from privileged state. The next step for paper-quality evidence is to produce policy-generated top-k candidates or a non-privileged planner candidate source.

