# ManiSkill Learned Grasp Proposal Diagnostic

## Question

Can FAVC improve over a trained candidate source, rather than only over hand-designed candidate families?

This diagnostic trains a high-level PickCube grasp proposal from labeled candidate rollouts, samples candidate grasp parameters on unseen seeds, executes those candidates, and then reranks them with the held-out action critic.

It is not a final policy benchmark: the low-level executor is still the privileged ManiSkill diagnostic controller. The important change is that candidate parameters are sampled from a learned success-conditioned proposal distribution instead of fixed candidate IDs or manually ordered families.

## Proposal Training

Training source:

- `outputs/maniskill_pickcube_random_grasp_n50_k8/PickCube-v1_candidate_manifest.jsonl`

The proposal fits a Gaussian over successful candidate parameters:

- xy offset;
- controller gain;
- grasp height.

Fitted mean:

| Parameter | Mean |
| --- | ---: |
| xy offset x | -0.0004 |
| xy offset y | 0.0004 |
| gain | 17.3126 |
| grasp z | 0.0324 |

Number of successful training parameters: 212.

## Stochastic Proposal Ranking

Evaluation:

- 50 unseen eval seeds: 50-99.
- 8 candidates per case.
- temperature: 2.5.
- rank0: first sampled candidate.

| Selector | Success | Recovered rank0 failures | Oracle match |
| --- | ---: | ---: | ---: |
| First learned proposal sample | 25/50 | 0/25 | 3/50 |
| Zero-action selector control | 25/50 | 0/25 | 3/50 |
| Action critic reranker | 49/50 | 24/25 | 19/50 |
| Oracle-best proposal sample | 50/50 | 25/25 | 50/50 |

Interpretation:

- The learned proposal distribution contains successful candidates for every held-out seed.
- A random first sample is not enough.
- The held-out action critic recovers 24 of 25 failed rank0 samples.
- The zero selector falls back to rank0, so the gain is not class prior or tie-breaking.

## Likelihood-Ranked Proposal

A stronger baseline ranks samples by proposal likelihood before execution.

| Selector | Success | Recovered rank0 failures | Preserved rank0 |
| --- | ---: | ---: | ---: |
| Likelihood-ranked proposal rank0 | 44/50 | 0/6 | 50/50 |
| Global action critic reranker | 49/50 | 6/6 | 0/50 |
| Failure-gated action critic | 50/50 | 6/6 | 43/50 |
| Oracle-best proposal sample | 50/50 | 6/6 | n/a |

Interpretation:

- Proposal likelihood is a strong baseline, but it is not the same as rollout success.
- The global action critic recovers all 6 likelihood-rank0 failures, but changes one successful rank0 into a failure.
- The failure gate closes this gap: it recovers all 6 failures while preserving most successful rank0 choices, reaching 50/50.

## Mechanism

This is the cleanest current FAVC story:

> A trained proposal distribution can contain the right action, and can even rank most cases correctly by likelihood. But likelihood is not calibrated to physical success. A rollout action critic detects the remaining execution failures, and a gate prevents unnecessary overrides.

This directly addresses the earlier weakness that the candidate pool was hand-designed. The proposal is still diagnostic and state/controller based, but the ranking failure is now induced by a trained candidate source.

## Negative Low-Level BC Result

A direct state-BC policy was also tested as a lower-level policy-generated candidate source:

| Variant | Rank0 success | Oracle-best success |
| --- | ---: | ---: |
| Markov state BC, mixed demos | 1/10 | 2/10 |
| Time-conditioned state BC | 1/10 | 1/10 |
| Noisy-demo augmented state BC | 1/10 | 2/10 |
| Low-grasp-only state BC | 1/10 | 3/10 |

This negative result suggests that low-level BC candidate generation is not the right short-term route without recurrent policies, stronger imitation learning, or official demonstrations. The learned high-level proposal is a more reliable intermediate benchmark for the current paper story.

## Reproducibility

Scripts:

- `src/umm_reward_evaluator/benchmarks/maniskill_learned_grasp_proposal_pool.py`
- `src/umm_reward_evaluator/benchmarks/train_action_sequence_selector.py`
- `src/umm_reward_evaluator/benchmarks/train_gated_action_sequence_selector.py`

Remote outputs:

- `outputs/maniskill_pickcube_learned_grasp_proposal_n50_t2p5/summary.json`
- `outputs/maniskill_pickcube_learned_grasp_proposal_n50_t2p5/action_selector_raw_no_length_e50/summary.json`
- `outputs/maniskill_pickcube_learned_grasp_proposal_n50_t2p5/action_selector_zero_e50/summary.json`
- `outputs/maniskill_pickcube_learned_grasp_proposal_n50_t2p5_likelihood_ranked/summary.json`
- `outputs/maniskill_pickcube_learned_grasp_proposal_n50_t2p5_likelihood_ranked/action_selector_raw_no_length_e50/summary.json`
- `outputs/maniskill_pickcube_learned_grasp_proposal_n50_t2p5_likelihood_ranked/gated_action_raw_no_length_e50/summary.json`
