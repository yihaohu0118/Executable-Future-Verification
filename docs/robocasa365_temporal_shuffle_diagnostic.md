# RoboCasa365 Temporal Shuffle Diagnostic

## Purpose

This diagnostic tests a counterintuitive failure mode in the RoboCasa365 action critic: preserving temporal order in a compact action-summary feature can be worse than destroying temporal order.

This is not evidence that robot actions are order-invariant. It is evidence that, under the current small-data held-out selector and sparse action snapshots, ordered temporal features can overfit or miscalibrate failure probability. Distributional action calibration is currently more reliable than the ordered trace.

## Code Change

`train_action_sequence_selector.py` now uses a stable SHA-256 seed for `shuffle_time` instead of Python's process-randomized `hash(case_id)`. This makes shuffle controls reproducible across processes.

New feature modes:

- `phase_no_length`: raw summary plus four equal temporal phase summaries.
- `phase_shuffle_time`: shuffle the action snapshot order first, then compute the same phase summaries.

The phase features include per-phase mean, standard deviation, absolute mean, phase energy, late-energy ratio, and early-late energy delta.

## Single-Task Faucet No-Demo

Manifest:

`/tmp/robocasa365_demo_pool_faucet_random_n8_s11/TurnOnSinkFaucet_no_demo_candidate_manifest.jsonl`

Oracle ceiling: 7/8.

| Feature | Seeds | Selector success |
| --- | --- | --- |
| raw action statistics, no length | 0,1,2,3,4 | 2,2,2,2,2 |
| shuffled-time action statistics | 0,1,2,3,4 | 2,3,3,3,3 |
| phase summaries | 0,1,2,3,4 | 2,2,2,2,2 |
| phase summaries after time shuffle | 0,1,2,3,4 | 4,4,3,3,5 |

The important result is not that phase features solve Faucet. They do not. The surprising result is that shuffling before phase summaries improves over ordered phase summaries on every seed.

## Three-Task No-Demo

Manifests:

- `PickPlaceCounterToCabinet_no_demo_candidate_manifest.jsonl`
- `TurnOnSinkFaucet_no_demo_candidate_manifest.jsonl`
- `OpenCabinet_no_demo_candidate_manifest.jsonl`

Oracle ceiling: 19/24.

| Feature | Seeds | Overall success | Faucet success |
| --- | --- | --- | --- |
| raw action statistics, no length | 0,1,2 | 14,14,13 | 2,3,1 |
| shuffled-time action statistics | 0,1,2 | 17,16,17 | 5,4,5 |
| phase summaries | 0,1,2 | 15,15,14 | 3,3,2 |
| phase summaries after time shuffle | 0,1,2 | 17,16,16 | 5,4,4 |

PickPlaceCounterToCabinet and OpenCabinet already reach their no-demo oracle ceilings. The gain from shuffle controls is concentrated in TurnOnSinkFaucet.

## Interpretation

The current evidence supports this mechanism:

> For small-data action critics on RoboCasa365 replay candidates, temporal order can be an anti-feature: ordered summaries overfit to phase artifacts, while shuffled action-distribution summaries better capture whether a candidate has the right calibration envelope.

This is a useful ICLR-style diagnostic because it contradicts the default assumption that more temporal structure is always better for action-conditioned evaluation.

The next method should not be "always shuffle actions." A safer direction is:

1. learn when temporal order is reliable;
2. use order-invariant calibration features as a conservative failure detector;
3. add contact-conditioned features for Faucet-style interactions where successful candidates exist but compact statistics still miss them.

## Reviewer Caveats

- Candidate generation is still replay perturbation, not a learned policy.
- The action traces are sparse snapshots stored with stride 25, so this diagnostic does not rule out high-frequency temporal information.
- The task count is three tasks and eight episodes per task; the result should be used as a mechanism-finding diagnostic before becoming a headline benchmark table.
- The shuffled controls should be presented as a warning against overclaiming temporal world modeling, not as proof that action order is irrelevant.
