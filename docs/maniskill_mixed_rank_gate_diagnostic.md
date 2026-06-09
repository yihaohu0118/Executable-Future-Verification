# ManiSkill Mixed-Rank Failure Gate Diagnostic

## Question

Does FAVC need a failure gate, or is global critic reranking enough?

The previous brittle-rank experiments made rank0 nearly always wrong, so the selector could simply override rank0. This diagnostic randomizes planner rank0 within each case, creating a mixed setting where rank0 is sometimes already successful.

## Protocol

1. Reassign `candidate_rank_by_planner` within each case using `randomize_planner_rank.py`.
2. Train a leave-one-case-out action critic using `raw_no_length` features.
3. Select a gate threshold on training folds only.
4. On the held-out case:
   - keep rank0 if rank0 critic score is above the threshold;
   - otherwise override with the highest-scoring candidate.

## Results

### PickCube Randomized Grasp Pool

Manifest:

- `outputs/maniskill_pickcube_random_grasp_n50_k8/PickCube-v1_candidate_manifest.random_rank_s0.jsonl`

| Selector | Success | Recovered rank0 failures | Preserved rank0 |
| --- | ---: | ---: | ---: |
| Randomized rank0 | 24/50 | 0/26 | 50/50 |
| Global action critic | 50/50 | 26/26 | 0/50 |
| Failure-gated action critic | 50/50 | 26/26 | 24/50 |
| Oracle-best | 50/50 | 26/26 | n/a |

Interpretation:

- The gate is conservative without losing success.
- It preserves exactly the successful rank0 cases and overrides all rank0 failures.
- Global reranking is equally successful, but does not preserve planner choices.

### StackCube Brittle-Stack Pool

Manifest:

- `outputs/maniskill_stackcube_brittle_stack_n50/StackCube-v1_candidate_manifest.random_rank_s0.jsonl`

| Selector | Success | Recovered rank0 failures | Preserved rank0 |
| --- | ---: | ---: | ---: |
| Randomized rank0 | 20/50 | 0/30 | 50/50 |
| Global action critic | 50/50 | 30/30 | 0/50 |
| Failure-gated action critic | 49/50 | 29/30 | 18/50 |
| Oracle-best | 50/50 | 30/30 | n/a |

Interpretation:

- The critic itself is strong enough to solve the mixed StackCube diagnostic.
- The gate is slightly too conservative and preserves one failing rank0.
- This is a useful negative result: failure gating is a calibration problem, not a free improvement over global reranking.

## Mechanism

The mixed-rank results refine the FAVC claim:

> Failure-gated critics are useful when preserving a planner's correct choices matters, but the gate must be calibrated. If the candidate critic is already near-oracle, a conservative gate can underperform global reranking.

For paper development, this suggests that the next method component should be uncertainty-aware or task-conditioned gating, not only a stronger candidate score.

## Reproducibility

Scripts:

- `src/umm_reward_evaluator/benchmarks/randomize_planner_rank.py`
- `src/umm_reward_evaluator/benchmarks/train_gated_action_sequence_selector.py`

Remote outputs:

- `outputs/maniskill_pickcube_random_grasp_n50_k8/gated_action_random_rank_s0/summary.json`
- `outputs/maniskill_stackcube_brittle_stack_n50/gated_action_random_rank_s0/summary.json`
