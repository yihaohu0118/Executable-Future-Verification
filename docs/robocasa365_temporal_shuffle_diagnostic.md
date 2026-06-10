# RoboCasa365 Temporal Shuffle Diagnostic

## Purpose

This diagnostic tests a counterintuitive failure mode in the RoboCasa365 action critic: preserving temporal order in a compact action-summary feature can be worse than destroying temporal order.

This is not evidence that robot actions are order-invariant. It is evidence that, under the current small-data held-out selector and sparse action snapshots, ordered temporal features can overfit or miscalibrate failure probability. Distributional action calibration is currently more reliable than the ordered trace.

## Code Change

`train_action_sequence_selector.py` now uses a stable SHA-256 seed for `shuffle_time` instead of Python's process-randomized `hash(case_id)`. This makes shuffle controls reproducible across processes.

`train_multiview_action_sequence_selector.py` adds a held-out multiview calibration test. It trains each feature view with the same leave-one-case-out protocol, then combines view scores with either simple score/rank aggregation or a second-stage logistic calibrator over view probabilities, within-case view ranks, and candidate planner rank. The logistic calibrator uses an outer-isolated stacking protocol: for each held-out case, view scores for meta-training are produced by models trained only on the other cases.

New feature modes:

- `stats_no_endpoints_no_length`: remove first/last endpoints and keep only mean, standard deviation, min, max, and absolute mean.
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

## Four-Task No-Demo, 16 Episodes Per Task

The same four tasks were expanded from 8 to 16 target episodes per task by generating only episodes 8-15 with the same random no-demo candidate protocol and merging them with the original n8 manifests.

Oracle ceiling: 41/64. Conservative rank0 remains 0/64.

| Feature | Seeds | Overall success | Mean | Pick mean | Faucet mean | OpenCabinet mean | Microwave mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw action statistics, no length | 0,1,2,3,4 | 24,20,20,22,22 | 21.6 | 5.0 | 3.8 | 7.0 | 5.8 |
| endpoint-free stats, no length | 0,1,2,3,4 | 25,27,27,24,26 | 25.8 | 6.4 | 6.0 | 7.8 | 5.6 |
| bag action-envelope moments, no length | 0,1,2,3,4 | 31,27,27,30,28 | 28.6 | 6.4 | 5.8 | 8.8 | 7.6 |
| one unordered pseudo-endpoint pair, no length | 0,1,2,3,4 | 25,27,26,29,30 | 27.4 | 7.0 | 6.8 | 7.8 | 5.8 |
| shuffled-time action statistics | 0,1,2,3,4 | 26,24,25,25,26 | 25.2 | 5.6 | 7.4 | 7.2 | 5.0 |
| multiview meta: one unordered endpoint + shuffled-time | 0,1,2 | 26,25,24 | 25.0 | 5.7 | 7.3 | 7.0 | 5.0 |

Deterministic heuristic controls on the same n16 manifests:

| Heuristic | Overall success | Pick | Faucet | OpenCabinet | Microwave |
| --- | ---: | ---: | ---: | ---: | ---: |
| max mean absolute action | 28/64 | 6/16 | 8/16 | 8/16 | 6/16 |
| max sum action energy | 27/64 | 6/16 | 7/16 | 7/16 | 7/16 |
| max action energy | 26/64 | 7/16 | 6/16 | 6/16 | 7/16 |
| max planner rank / lowest conservative prior | 26/64 | 7/16 | 5/16 | 6/16 | 8/16 |
| min action energy / rank0-style prior | 0/64 | 0/16 | 0/16 | 0/16 | 0/16 |

The n16 result changes the interpretation. On 32 cases, shuffled-time looked like the strongest diagnostic. On 64 cases, simple action-envelope moments are strongest on average, and one unordered endpoint pair is second. Raw ordered summaries remain much worse. The new endpoint-free stats control shows that removing first/last endpoints explains a large part of the gain: 25.8/64 mean versus 21.6/64 for raw. Adding the full bag energy/envelope moments raises this to 28.6/64.

This creates a stronger counterintuitive claim than the original shuffle diagnostic:

> On expanded RoboCasa365 replay candidates, preserving true temporal endpoints hurts more than it helps; coarse action-envelope calibration is more reliable than ordered first/last summaries and even more reliable than shuffled-time diagnostics.

The heuristic controls make the claim sharper and more dangerous: max mean absolute action reaches 28/64 without training, nearly matching the learned bag mean of 28.6/64. The result is not a single-candidate shortcut, because the selected candidate IDs are spread across ranks. It is a sign that the conservative policy prior is badly miscalibrated for these recoverable failures: it prefers low-action candidates that always fail, while successful candidates often require larger action magnitude.

## Interpretation

The current evidence supports this mechanism:

> For action critics on RoboCasa365 replay candidates, temporal detail can be an anti-feature: ordered first/last summaries overfit to endpoint artifacts, while endpoint-free action-envelope statistics preserve candidate-level action calibration better.

This is a useful ICLR-style diagnostic because it contradicts the default assumption that more temporal structure is always better for action-conditioned evaluation.

The earlier negative bag result matters as a small-sample warning: on 24-32 cases, bag moments did not explain the full shuffle-time gain. After expanding to 64 cases, however, bag moments become the strongest single-view selector. This suggests the original shuffle result was a high-variance diagnostic of endpoint overfitting, while the more scalable mechanism is endpoint-free action-envelope calibration.

The strongest current method-shaped result is not "always shuffle." It is an endpoint-free envelope critic: remove brittle first/last temporal anchors and score candidates from action-distribution statistics. Unordered endpoint dropout remains useful, but the n16 result shows that the simplest envelope moments are the strongest learned baseline and that a no-learning action-magnitude heuristic is already surprisingly competitive.

The next method should not be "always shuffle actions." A safer direction is:

1. learn when temporal order is reliable;
2. use endpoint-free envelope and endpoint-dropout views as conservative failure detectors, but keep shuffle-robust controls as a diagnostic baseline;
3. add energy-matched hard negatives to test whether the envelope critic understands action geometry or mostly corrects an under-actuation prior;
4. add contact-conditioned features for Faucet-style interactions where successful candidates exist but compact statistics still miss them.

## Reviewer Caveats

- Candidate generation is still replay perturbation, not a learned policy.
- The strongest deterministic heuristic is action magnitude, so the current candidate pool must be stress-tested with energy-matched hard negatives before claiming semantic action understanding.
- The action traces are sparse snapshots stored with stride 25, so this diagnostic does not rule out high-frequency temporal information.
- The strongest current table uses four tasks and sixteen episodes per task. It is stronger than the original 32-case diagnostic, but still needs more RoboCasa365 tasks or a learned proposal source before becoming a headline benchmark table.
- The shuffled controls should be presented as a warning against overclaiming temporal world modeling, not as proof that action order is irrelevant.
