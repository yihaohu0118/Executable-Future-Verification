# RoboCasa365 Temporal Shuffle Diagnostic

## Purpose

This diagnostic tests a counterintuitive failure mode in the RoboCasa365 action critic: preserving temporal order in a compact action-summary feature can be worse than destroying temporal order.

This is not evidence that robot actions are order-invariant. It is evidence that, under the current small-data held-out selector and sparse action snapshots, ordered temporal features can overfit or miscalibrate failure probability. Distributional action calibration is currently more reliable than the ordered trace.

## Code Change

`train_action_sequence_selector.py` now uses a stable SHA-256 seed for `shuffle_time` instead of Python's process-randomized `hash(case_id)`. This makes shuffle controls reproducible across processes.

`train_multiview_action_sequence_selector.py` adds a held-out multiview calibration test. It trains each feature view with the same leave-one-case-out protocol, then combines view scores with either simple score/rank aggregation or a second-stage logistic calibrator over view probabilities, within-case view ranks, and candidate planner rank. The logistic calibrator uses an outer-isolated stacking protocol: for each held-out case, view scores for meta-training are produced by models trained only on the other cases.

New feature modes:

- `bag_no_length`: order-invariant action-envelope moments without first/last endpoints.
- `sampled_endpoints_no_length`: replace true first/last endpoints with one deterministic pseudo-endpoint pair sampled from the trajectory.
- `multi_sampled_endpoints_no_length`: replace true first/last endpoints with four deterministic pseudo-endpoint pairs sampled from the trajectory.
- `permutation_endpoints_no_length`: shuffle each trajectory with a stable per-case seed and use the permuted first/last actions as one unordered pseudo-endpoint pair.
- `multi_permutation_endpoints_no_length`: use four stable per-case permutations and concatenate four unordered pseudo-endpoint pairs.
- `phase_no_length`: raw summary plus four equal temporal phase summaries.
- `phase_shuffle_time`: shuffle the action snapshot order first, then compute the same phase summaries.

The bag features include mean, standard deviation, min, max, absolute mean, and per-action-dimension energy. The pseudo-endpoint features concatenate sampled endpoint pairs with the same bag features. The phase features include per-phase mean, standard deviation, absolute mean, phase energy, late-energy ratio, and early-late energy delta.

## Single-Task Faucet No-Demo

Manifest:

`/tmp/robocasa365_demo_pool_faucet_random_n8_s11/TurnOnSinkFaucet_no_demo_candidate_manifest.jsonl`

Oracle ceiling: 7/8.

| Feature | Seeds | Selector success |
| --- | --- | --- |
| raw action statistics, no length | 0,1,2,3,4 | 2,2,2,2,2 |
| bag action-envelope moments, no length | 0,1,2,3,4 | 4,2,2,2,3 |
| bag action-envelope moments with length | 0,1,2,3,4 | 3,2,2,2,3 |
| one pseudo-endpoint pair, no length | 0,1,2,3,4 | 1,3,1,1,1 |
| one pseudo-endpoint pair with length | 0,1,2,3,4 | 2,2,1,1,1 |
| four pseudo-endpoint pairs, no length | 0,1,2,3,4 | 2,3,3,3,1 |
| four pseudo-endpoint pairs with length | 0,1,2,3,4 | 2,3,3,3,1 |
| one unordered pseudo-endpoint pair, no length | 0,1,2,3,4 | 2,3,2,2,3 |
| four unordered pseudo-endpoint pairs, no length | 0,1,2,3,4 | 4,4,2,3,3 |
| shuffled-time action statistics | 0,1,2,3,4 | 2,3,3,3,3 |
| phase summaries | 0,1,2,3,4 | 2,2,2,2,2 |
| phase summaries after time shuffle | 0,1,2,3,4 | 4,4,3,3,5 |

The important result is not that phase features solve Faucet. They do not. The surprising result is that shuffling before phase summaries improves over ordered phase summaries on every seed. Simple order-invariant bag moments also do not explain the gain; they remain close to the raw ordered baseline. Pseudo-endpoint features are weaker than phase-shuffle on single-task Faucet, but they become useful in the multitask setting below.

## Three-Task No-Demo

Manifests:

- `PickPlaceCounterToCabinet_no_demo_candidate_manifest.jsonl`
- `TurnOnSinkFaucet_no_demo_candidate_manifest.jsonl`
- `OpenCabinet_no_demo_candidate_manifest.jsonl`

Oracle ceiling: 19/24.

| Feature | Seeds | Overall success | Faucet success |
| --- | --- | --- | --- |
| raw action statistics, no length | 0,1,2 | 14,14,13 | 2,3,1 |
| bag action-envelope moments, no length | 0,1,2 | 14,14,14 | 2,2,2 |
| bag action-envelope moments with length | 0,1,2 | 14,14,14 | 2,2,2 |
| one pseudo-endpoint pair, no length | 0,1,2 | 15,16,15 | 3,4,3 |
| one pseudo-endpoint pair with length | 0,1,2 | 15,16,15 | 3,4,3 |
| four pseudo-endpoint pairs, no length | 0,1,2 | 16,16,17 | 4,4,5 |
| four pseudo-endpoint pairs with length | 0,1,2 | 15,16,16 | 3,4,4 |
| shuffled-time action statistics | 0,1,2 | 17,16,17 | 5,4,5 |
| phase summaries | 0,1,2 | 15,15,14 | 3,3,2 |
| phase summaries after time shuffle | 0,1,2 | 17,16,16 | 5,4,4 |

PickPlaceCounterToCabinet and OpenCabinet already reach their no-demo oracle ceilings. The gain from shuffle controls is concentrated in TurnOnSinkFaucet.

## Four-Task No-Demo

The fourth task adds `TurnOnMicrowave`, a second button-style fixture interaction.

Manifests:

- `PickPlaceCounterToCabinet_no_demo_candidate_manifest.jsonl`
- `TurnOnSinkFaucet_no_demo_candidate_manifest.jsonl`
- `OpenCabinet_no_demo_candidate_manifest.jsonl`
- `TurnOnMicrowave_no_demo_candidate_manifest.jsonl`

Oracle ceiling: 25/32.

| Feature | Seeds | Overall success | Faucet success | Microwave success |
| --- | --- | --- | --- | --- |
| raw action statistics, no length | 0,1,2 | 18,17,16 | 2,2,1 | 4,3,3 |
| bag action-envelope moments, no length | 0,1,2 | 19,18,16 | 3,3,2 | 4,4,2 |
| four pseudo-endpoint pairs, no length | 0,1,2 | 19,18,16 | 5,4,3 | 3,2,3 |
| one unordered pseudo-endpoint pair, no length | 0,1,2 | 20,20,20 | 4,4,5 | 4,4,3 |
| one unordered pseudo-endpoint pair with length | 0,1,2 | 20,20,20 | 4,5,5 | 4,3,3 |
| four unordered pseudo-endpoint pairs, no length | 0,1,2 | 17,19,20 | 5,5,5 | 1,3,3 |
| four unordered pseudo-endpoint pairs with length | 0,1,2 | 16,20,19 | 5,6,4 | 1,3,3 |
| shuffled-time action statistics | 0,1,2 | 22,19,19 | 5,5,4 | 5,2,3 |
| multiview meta: raw + one unordered endpoint | 0,1,2 | 20,19,16 | 3,3,2 | 5,4,2 |
| multiview meta: bag + one unordered endpoint | 0,1,2 | 20,19,18 | 3,3,3 | 5,4,3 |
| multiview meta: one unordered endpoint + shuffled-time | 0,1,2 | 21,20,20 | 4,5,5 | 5,3,3 |
| multiview rank aggregation: one unordered endpoint + shuffled-time | 0,1,2 | 20,19,19 | 4,4,4 | 4,3,3 |

Single-task `TurnOnMicrowave` no-demo has oracle ceiling 6/8. Multi-pseudo-endpoints recover 5,3,5/8, compared with raw 3,2,3/8 and shuffled-time 3,4,3/8. In the four-task multitask setting, however, pseudo-endpoints are weaker than shuffled-time overall. This makes endpoint dropout a useful partial mechanism, not yet a complete replacement for shuffle-robust calibration.

Single-task `TurnOnSinkFaucet` remains harder for unordered endpoints: one unordered pair recovers 2,3,2,2,3/8 and four unordered pairs recover 4,4,2,3,3/8 against a 7/8 oracle. Single-task `TurnOnMicrowave` is more favorable: one unordered pair recovers 3,3,5,5,3/8 and four unordered pairs recover 4,4,5,5,4/8 against a 6/8 oracle. The four-task result is therefore not just a single-task feature improvement; task sharing and contact calibration still matter.

The multiview results separate two mechanisms. Simple rank aggregation of unordered endpoints and shuffled-time scores is weak at 20,19,19/32, so the gain is not from naive agreement voting. An outer-isolated logistic calibrator over the same two views reaches 21,20,20/32. This trades away the best shuffled-time seed but reduces the low-seed drop from 19 to 20 and slightly improves over the stable unordered-endpoint view on seed 0. Adding raw or bag views is weaker, which supports the negative finding that ordered/raw temporal summaries can still drag down calibration.

## Interpretation

The current evidence supports this mechanism:

> For small-data action critics on RoboCasa365 replay candidates, temporal order can be an anti-feature: ordered summaries overfit to endpoint or phase artifacts, while deterministic shuffle perturbations act like an anti-overfitting control that preserves candidate-level action calibration.

This is a useful ICLR-style diagnostic because it contradicts the default assumption that more temporal structure is always better for action-conditioned evaluation.

The negative bag result matters. If ordinary order-invariant moments were enough, `bag_no_length` should have matched `shuffle_time`; it did not. The pseudo-endpoint result narrows the mechanism further: replacing brittle true endpoints with deterministic pseudo-endpoints nearly matches shuffled-time performance in the three-task setting. The four-task `TurnOnMicrowave` result adds a boundary: one unordered pseudo-endpoint pair is stable at 20/32, but four unordered pairs are less stable and shuffled-time remains the strongest single-view diagnostic. More pseudo-endpoints are not automatically better.

The strongest current method-shaped result is not "always shuffle." It is a two-view temporal-dropout calibrator: one view keeps a stable unordered endpoint-dropout summary, the other view uses shuffled-time calibration, and an outer-isolated meta selector learns how to combine their within-case ranks. This is still diagnostic-scale and only a small improvement in stability, but it is more defensible than presenting shuffle as a final policy.

The next method should not be "always shuffle actions." A safer direction is:

1. learn when temporal order is reliable;
2. use endpoint-dropout and temporal-dropout views as conservative failure detectors, but keep shuffle-robust controls as a diagnostic baseline;
3. add contact-conditioned features for Faucet-style interactions where successful candidates exist but compact statistics still miss them.

## Reviewer Caveats

- Candidate generation is still replay perturbation, not a learned policy.
- The action traces are sparse snapshots stored with stride 25, so this diagnostic does not rule out high-frequency temporal information.
- The task count is four tasks and eight episodes per task; the result should be used as a mechanism-finding diagnostic before becoming a headline benchmark table.
- The shuffled controls should be presented as a warning against overclaiming temporal world modeling, not as proof that action order is irrelevant.
