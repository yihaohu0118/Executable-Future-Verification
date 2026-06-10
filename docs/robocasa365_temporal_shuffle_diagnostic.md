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

## Four-Task Energy-Matched Hard Negatives

To test whether the n16 action-envelope result is more than an action-magnitude shortcut, a second four-task pool was generated with energy-matched corruptions of the successful demonstration action trace:

- time reverse;
- 25% and 50% temporal roll;
- deterministic time shuffle;
- block swap;
- xyz sign flip;
- gripper sign flip;
- original demonstration action trace.

The original action trace is deliberately placed at `cand_07`, so `planner_rank_max` is a construction check rather than a valid baseline. The fair stress-test question is whether action-only selectors can distinguish the original from corruptions that preserve much of the same magnitude envelope.

Manifest set:

- `/tmp/robocasa365_energy_matched_pick_n4_s17/PickPlaceCounterToCabinet_candidate_manifest.jsonl`
- `/tmp/robocasa365_energy_matched_faucet_n4_s17/TurnOnSinkFaucet_candidate_manifest.jsonl`
- `/tmp/robocasa365_energy_matched_opencab_n4_s17/OpenCabinet_candidate_manifest.jsonl`
- `/tmp/robocasa365_energy_matched_microwave_n4_s17/TurnOnMicrowave_candidate_manifest.jsonl`

Oracle ceiling: 16/16. Conservative rank0: 0/16. All corruptions fail; the original `cand_07` succeeds in 16/16.

Deterministic controls:

| Heuristic | Success |
| --- | ---: |
| max mean absolute action | 0/16 |
| max sum action energy | 0/16 |
| max action energy | 0/16 |
| max action standard deviation | 0/16 |
| max action range | 0/16 |
| max/min smoothness | 0/16 |
| conservative prior variants | 0/16 |
| planner rank max | 16/16, invalid construction check |

Learned case-heldout selectors:

| Feature | Seeds | Overall success | Mean | Pick mean | Faucet mean | OpenCabinet mean | Microwave mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw action statistics, no length | 0,1,2,3,4 | 6,6,5,4,6 | 5.4 | 0.6 | 2.4 | 0.0 | 2.4 |
| bag action-envelope moments, no length | 0,1,2,3,4 | 6,4,5,8,4 | 5.4 | 1.0 | 0.8 | 0.8 | 2.8 |
| endpoint-free stats, no length | 0,1,2,3,4 | 6,7,6,6,6 | 6.2 | 1.8 | 0.8 | 0.8 | 2.8 |
| one unordered pseudo-endpoint pair, no length | 0,1,2,3,4 | 4,5,7,5,5 | 5.2 | 0.8 | 2.0 | 0.2 | 2.2 |
| shuffled-time action statistics | 0,1,2,3,4 | 7,4,5,7,7 | 6.0 | 0.6 | 2.6 | 0.2 | 2.6 |

This is the first hard-negative diagnostic. The magnitude heuristic that nearly matched the learned bag critic on the n16 pool collapses from 28/64 to 0/16 under energy-matched negatives. Learned action-only selectors recover only 5-6/16 on average, so they are not just magnitude heuristics, but they are far from the 16/16 oracle. The bottleneck is now visible: endpoint-free action-envelope calibration fixes under-actuation, but distinguishing correct contact timing and direction from energy-matched corruptions likely requires visual state, contact context, or a stronger temporal model.

## State-Trace Proxy on Energy-Matched Negatives

To test whether the energy-matched gap is genuinely about missing state/contact context, the same four-task hard-negative pool was regenerated with low-dimensional rollout observation traces stored in metadata. The trace uses RoboCasa state and proprioceptive keys such as `object-state`, `robot0_proprio-state`, `obj_pos`, `obj_to_robot0_eef_pos`, `robot0_eef_pos`, and gripper state, sampled every 25 simulator steps. It does not use reward, `_check_success`, or the oracle label as an input feature.

Manifest set:

- `/tmp/robocasa365_energy_state_pick_n4_s17/PickPlaceCounterToCabinet_candidate_manifest.jsonl`
- `/tmp/robocasa365_energy_state_faucet_n4_s17/TurnOnSinkFaucet_candidate_manifest.jsonl`
- `/tmp/robocasa365_energy_state_opencab_n4_s17/OpenCabinet_candidate_manifest.jsonl`
- `/tmp/robocasa365_energy_state_microwave_n4_s17/TurnOnMicrowave_candidate_manifest.jsonl`

Oracle ceiling remains 16/16 and conservative rank0 remains 0/16. Trace coverage is complete, with per-candidate trace lengths ranging from 6 to 20 snapshots depending on task horizon.

Held-out selectors:

| Feature | Seeds | Overall success | Mean | Pick mean | Faucet mean | OpenCabinet mean | Microwave mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| zero state control | 0,1,2,3,4 | 0,0,0,0,0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| best action-only control, endpoint-free stats | 0,1,2,3,4 | 6,7,6,6,6 | 6.2 | 1.8 | 0.8 | 0.8 | 2.8 |
| low-dimensional state trace | 0,1,2,3,4 | 13,15,14,14,15 | 14.2 | 3.8 | 3.4 | 3.0 | 4.0 |
| state trace + endpoint-free action stats | 0,1,2,3,4 | 13,14,14,14,13 | 13.6 | 3.8 | 3.0 | 3.0 | 3.8 |

This converts the hard-negative result from a pure limitation into a method direction. Once action magnitude is controlled, action-only selectors recover only 5-6/16. A low-dimensional rollout-state selector recovers 14.2/16 and is stable across seeds, while the zero control stays at 0/16. Adding action-envelope features to state traces does not improve the result and slightly hurts mean success, so the current bottleneck is not more action summarization. It is whether the rollout state trajectory shows the correct physical interaction.

The remaining errors are concentrated in `OpenCabinet:ep_0000` and `TurnOnSinkFaucet:ep_0002`, which suggests the next useful feature is finer visual/contact evidence rather than a larger action MLP.

### n8 Scale-Up

The state-trace stress test was then doubled to eight target episodes per task by generating episodes 4-7 and merging them with the original n4 manifests:

- `/tmp/robocasa365_energy_state_pick_n8_s17/PickPlaceCounterToCabinet_candidate_manifest.jsonl`
- `/tmp/robocasa365_energy_state_faucet_n8_s17/TurnOnSinkFaucet_candidate_manifest.jsonl`
- `/tmp/robocasa365_energy_state_opencab_n8_s17/OpenCabinet_candidate_manifest.jsonl`
- `/tmp/robocasa365_energy_state_microwave_n8_s17/TurnOnMicrowave_candidate_manifest.jsonl`

Oracle ceiling is 32/32 and conservative rank0 remains 0/32. The original demonstration trace is still placed at `cand_07`, so `planner_rank_max` is only a construction check, not a valid baseline.

Held-out selectors:

| Feature | Seeds | Overall success | Mean | Std | Seed-0 by task |
| --- | --- | ---: | ---: | ---: | --- |
| zero state control | 0,1,2,3,4 | 0,0,0,0,0 | 0.0 | 0.0 | 0/8 each |
| action-only endpoint-free stats | 0,1,2,3,4 | 10,8,7,8,8 | 8.2 | 0.98 | not state-conditioned |
| low-dimensional state trace | 0,1,2,3,4 | 30,31,31,30,30 | 30.4 | 0.49 | Pick 8, Faucet 7, OpenCabinet 7, Microwave 8 |
| state trace + endpoint-free action stats | 0,1,2,3,4 | 30,31,29,30,30 | 30.0 | 0.63 | Pick 8, Faucet 7, OpenCabinet 7, Microwave 8 |

Deterministic action controls on the same n8 state-trace manifests:

| Heuristic | Success |
| --- | ---: |
| max/min action energy variants | 0/32 |
| max mean absolute action | 0/32 |
| max action standard deviation | 0/32 |
| max action range | 0/32 |
| max/min smoothness | 0/32 |
| conservative prior variants | 0/32 |
| planner rank max | 32/32, invalid construction check |

The n8 result strengthens the mechanism. The energy/magnitude heuristic family remains exactly 0/32, action-only learning recovers only 8.2/32, and state traces recover 30.4/32. Adding action stats to the state trace again does not help and is slightly worse. This makes the counterintuitive method target sharper: after magnitude is controlled, the bottleneck is not a richer action summary but contact-conditioned state evidence.

### Rank and Candidate-ID Randomization Control

To test whether the n8 state-trace result depends on the original action always being `cand_07`, the same n8 manifests were postprocessed with a per-case rank/candidate-id shuffle:

- rank0 is forced to remain a failed corruption;
- all other candidates are shuffled;
- `candidate_id` is remapped to the new rank;
- the original candidate ID/rank is stored in metadata for audit.

This is a manifest-level leakage control, not a new physical rollout. The action traces, success labels, and state traces are unchanged, but the visible candidate position of the successful original action is no longer fixed. With seed 23, successful candidates are distributed across nonzero ranks 1, 2, 3, 4, 6, and 7 for each task, while rank0 success remains 0/32.

Held-out selectors on the rank/candidate-id randomized n8 manifests:

| Feature | Seeds | Overall success | Mean | Std |
| --- | --- | ---: | ---: | ---: |
| zero state control | 0,1,2,3,4 | 0,0,0,0,0 | 0.0 | 0.0 |
| action-only endpoint-free stats | 0,1,2,3,4 | 10,9,7,8,8 | 8.4 | 1.02 |
| low-dimensional state trace | 0,1,2,3,4 | 30,31,31,30,30 | 30.4 | 0.49 |
| state trace + endpoint-free action stats | 0,1,2,3,4 | 30,31,29,30,31 | 30.2 | 0.75 |

The state-trace result is unchanged under candidate-position randomization, which directly weakens the fixed-`cand_07` leakage objection. Some deterministic action heuristics become nonzero after rank/candidate-id remapping because score ties now break differently, but they remain weak: energy/magnitude/range variants reach only 4-7/32, smoothness reaches 1-5/32, and planner-rank max reaches 4/32.

### Fully Regenerated Random-Position Pool

The stronger control regenerates the physical rollouts with `--energy-original-placement random_nonzero`, rather than postprocessing an existing manifest. This keeps rank0 as a failed corruption while placing the original action trace at a nonzero rank per episode. With seed 29 and eight episodes per task, the original rank distribution is identical across the four tasks because the same episode-index seed schedule is used: rank 1 for two cases, rank 5 for one case, and rank 7 for five cases. Each task has oracle 8/8 and rank0 0/8, for a combined 32/32 oracle and 0/32 rank0.

Held-out selectors on the fully regenerated random-position n8 manifests:

| Feature | Seeds | Overall success | Mean | Std |
| --- | --- | ---: | ---: | ---: |
| zero state control | 0,1,2,3,4 | 0,0,0,0,0 | 0.0 | 0.0 |
| action-only endpoint-free stats | 0,1,2,3,4 | 10,9,7,8,8 | 8.4 | 1.02 |
| low-dimensional state trace | 0,1,2,3,4 | 30,31,31,30,30 | 30.4 | 0.49 |
| state trace + endpoint-free action stats | 0,1,2,3,4 | 30,31,29,30,30 | 30.0 | 0.63 |

Deterministic action heuristics remain weak on the regenerated pool: energy/magnitude/range variants reach only 1-3/32, smoothness reaches 0-4/32, and conservative-prior variants reach 0-2/32. `planner_rank_max` reaches 20/32 because random_nonzero still places the original at high rank in five of eight episodes per task; this is a construction diagnostic, not a valid action or state baseline. The learned state-trace result is unchanged from both the fixed-position and manifest-randomized controls.

### State-Key Ablation

The next leakage question is whether the state-trace selector is simply reading a privileged object-state success variable. On the fully regenerated random-position n8 pool, the selector was rerun with explicit state-key include/exclude controls:

| State keys | Seeds | Overall success | Mean | Std | Seed-0 by task |
| --- | --- | ---: | ---: | ---: | --- |
| all state keys | 0,1,2,3,4 | 30,31,31,30,30 | 30.4 | 0.49 | Pick 8, Faucet 7, OpenCabinet 7, Microwave 8 |
| exclude `object-state` | 0,1,2,3,4 | 31,32,32,31,31 | 31.4 | 0.49 | Pick 8, Faucet 8, OpenCabinet 7, Microwave 8 |
| robot-only, excluding all object keys | 0,1,2,3,4 | 31,31,31,31,32 | 31.2 | 0.40 | Pick 8, Faucet 8, OpenCabinet 7, Microwave 8 |
| `robot0_proprio-state` only | 0,1,2,3,4 | 31,30,31,30,29 | 30.2 | 0.75 | Pick 7, Faucet 8, OpenCabinet 8, Microwave 8 |
| eef/gripper keys only | 0,1,2,3,4 | 31,31,31,31,29 | 30.6 | 0.80 | Pick 8, Faucet 8, OpenCabinet 7, Microwave 8 |
| object low-dimensional keys only | 0,1,2,3,4 | 16,16,16,16,16 | 16.0 | 0.0 | Pick 8, Faucet 0, OpenCabinet 0, Microwave 8 |
| eef/object relative contact proxy | 0,1,2,3,4 | 29,30,29,30,29 | 29.4 | 0.49 | Pick 8, Faucet 6, OpenCabinet 7, Microwave 8 |

This is the most useful mechanism diagnostic so far. Removing `object-state` does not hurt; it slightly improves the mean. Robot-only traces recover 31.2/32, and even `robot0_proprio-state` alone recovers 30.2/32. In contrast, object low-dimensional keys alone collapse to 16/32, solving Pick and Microwave but failing Faucet and OpenCabinet. The remaining signal is therefore not just privileged object-state leakage. It appears to be encoded in the robot/eef/gripper rollout response, consistent with contact/execution feedback: when an energy-matched action is temporally or directionally wrong, the robot's realized state trajectory differs even if compact action statistics look similar.

### n16 Random-Position Hard-Negative Scale-Up

The fully regenerated random-position pool was then expanded from eight to sixteen target episodes per task by merging episodes 0-7 with newly generated episodes 8-15 under the same `random_nonzero` original-placement protocol. The merged benchmark has 64 hard-negative cases and 512 candidates. Every task has oracle 16/16 and rank0 0/16. The successful original action is always at a nonzero rank, with the same audited rank distribution for each task: rank 1 for three cases, rank 2 for one case, rank 4 for one case, rank 5 for four cases, rank 6 for one case, and rank 7 for six cases.

Held-out selectors on the n16 fully regenerated random-position manifests:

| Feature | Seeds | Overall success | Mean | Std |
| --- | --- | ---: | ---: | ---: |
| zero state control | 0,1,2,3,4 | 0,0,0,0,0 | 0.0 | 0.0 |
| action-only endpoint-free stats | 0,1,2,3,4 | 29,30,29,28,26 | 28.4 | 1.36 |
| object low-dimensional keys only | 0,1,2,3,4 | 31,31,31,31,31 | 31.0 | 0.0 |
| `robot0_proprio-state` only | 0,1,2,3,4 | 62,64,63,62,64 | 63.0 | 0.89 |
| all state keys | 0,1,2,3,4 | 63,63,63,63,63 | 63.0 | 0.0 |
| state trace + endpoint-free action stats | 0,1,2,3,4 | 62,62,62,62,62 | 62.0 | 0.0 |
| robot-only, excluding all object keys | 0,1,2,3,4 | 64,64,64,64,64 | 64.0 | 0.0 |

The per-task split sharpens the mechanism. Object-only traces solve Pick almost perfectly and Microwave nearly perfectly, but get 0/16 on Faucet and OpenCabinet across all seeds. Robot-only traces get 16/16 on every task and every seed. Proprio-only traces are nearly saturated as well: 15-16/16 on Pick, Faucet, and OpenCabinet, and 16/16 on Microwave.

Deterministic heuristics remain far below the state/contact result on the same 64 hard cases. The strongest heuristic is `planner_rank_max` at 24/64, which is a construction diagnostic because the successful original is often placed at high nonzero rank. The next best is `smoothness_min` at 11/64; energy and magnitude variants reach only 4-8/64. This rules out the ordinary no-demo shortcut where max action magnitude nearly matched the learned selector.

### Robot-Key Ablation

The n16 result above still leaves a deployability question: `robot-only` contains both a broad `robot0_proprio-state` vector and explicit end-effector/gripper keys. A second n16 ablation isolates those signals:

| Robot/contact keys | Seeds | Overall success | Mean | Std | Main failure pattern |
| --- | --- | ---: | ---: | ---: | --- |
| EEF/gripper, no proprio | 0,1,2,3,4 | 62,64,62,62,61 | 62.2 | 0.98 | OpenCabinet loses 1-3 cases |
| EEF/gripper + object-to-EEF relative vector | 0,1,2,3,4 | 62,60,62,62,62 | 61.6 | 0.80 | OpenCabinet remains hardest |
| EEF position + gripper | 0,1,2,3,4 | 63,62,62,62,61 | 62.0 | 0.63 | OpenCabinet and Microwave lose a few |
| EEF position only | 0,1,2,3,4 | 60,61,60,61,60 | 60.4 | 0.49 | Pick/Microwave/Faucet lose 1-2 each |
| EEF pose only | 0,1,2,3,4 | 63,60,62,60,60 | 61.0 | 1.26 | Small losses across all tasks |
| gripper only | 0,1,2,3,4 | 45,43,40,46,45 | 43.8 | 2.14 | Microwave and Faucet lose most |

This is the strongest anti-leakage evidence so far. A broad proprio vector is not necessary: explicit EEF and gripper traces without `robot0_proprio-state` still recover 62.2/64, nearly matching full state and state+action. EEF position alone recovers 60.4/64, far above the 28.4/64 action-only selector and the 31.0/64 object-only selector. Adding the object-to-EEF relative vector does not help. The signal therefore looks less like privileged object state and more like a compact execution trace: did the end effector actually move through the contact-relevant path, and did the gripper response match the intended interaction?

### EEF Summary-Statistic Ablation

The full state-trace feature concatenates first, last, delta, mean, standard deviation, min, max, and mean absolute step. To test whether the EEF result is just endpoint success in disguise, the selector was extended with `--state-summary-mode` and rerun on the n16 random-position pool.

For EEF position only (`robot0_base_to_eef_pos`, `robot0_eef_pos`):

| EEF summary | Seeds | Overall success | Mean | Std |
| --- | --- | ---: | ---: | ---: |
| full summary | 0,1,2,3,4 | 60,61,60,61,60 | 60.4 | 0.49 |
| terminal only | 0,1,2,3,4 | 35,37,37,39,36 | 36.8 | 1.33 |
| endpoint first/last/delta | 0,1,2,3,4 | 34,37,28,31,36 | 33.2 | 3.31 |
| delta only | 0,1,2,3,4 | 39,41,44,41,40 | 41.0 | 1.67 |
| distribution only, mean/std/min/max | 0,1,2,3,4 | 63,62,63,62,63 | 62.6 | 0.49 |
| path step only | 0,1,2,3,4 | 33,29,35,32,29 | 31.6 | 2.33 |
| no endpoints, distribution + path | 0,1,2,3,4 | 62,62,63,63,63 | 62.6 | 0.49 |

For EEF position plus gripper (`robot0_base_to_eef_pos`, `robot0_eef_pos`, `robot0_gripper_qpos`, `robot0_gripper_qvel`):

| EEF/gripper summary | Seeds | Overall success | Mean | Std |
| --- | --- | ---: | ---: | ---: |
| full summary | 0,1,2,3,4 | 63,62,62,62,61 | 62.0 | 0.63 |
| terminal only | 0,1,2,3,4 | 35,38,38,35,37 | 36.6 | 1.36 |
| endpoint first/last/delta | 0,1,2,3,4 | 41,41,42,45,44 | 42.6 | 1.62 |
| distribution only, mean/std/min/max | 0,1,2,3,4 | 64,64,64,63,63 | 63.6 | 0.49 |
| path step only | 0,1,2,3,4 | 42,42,39,40,42 | 41.0 | 1.26 |
| no endpoints, distribution + path | 0,1,2,3,4 | 63,62,64,64,64 | 63.4 | 0.80 |

This is a sharper counterintuitive result than the original robot-only table. The successful signal is not final EEF location, start/end displacement, or average step size. Removing endpoints and keeping distributional execution-envelope statistics preserves almost all performance, and for EEF position plus gripper, distribution-only statistics slightly outperform the full summary. The critic appears to detect whether the realized robot trajectory occupied the right contact-relevant workspace, not whether it ended at a particular pose.

## Interpretation

The current evidence supports this mechanism:

> For action critics on RoboCasa365 replay candidates, temporal detail can be an anti-feature in ordinary no-demo pools: ordered first/last summaries overfit to endpoint artifacts, while endpoint-free action-envelope statistics preserve candidate-level action calibration better. But once action magnitude is matched, compact action-only summaries recover only limited signal; on the n16 fully regenerated hard-negative pool, action-only reaches 28.4/64 while robot-only rollout traces reach 64/64 and proprio-only reaches 63.0/64. A finer key ablation shows that EEF/gripper traces without the broad proprio vector still reach 62.2/64, and an EEF summary ablation shows that endpoint-free distributional execution envelopes reach 63.4-63.6/64 while endpoint-only summaries stay near 33-43/64. The next method should condition action adequacy on compact robot/contact execution-envelope feedback rather than action envelope, temporal detail, endpoint state, or object-state shortcuts alone.

This is a useful ICLR-style diagnostic because it contradicts the default assumption that more temporal structure is always better for action-conditioned evaluation.

The earlier negative bag result matters as a small-sample warning: on 24-32 cases, bag moments did not explain the full shuffle-time gain. After expanding to 64 cases, however, bag moments become the strongest single-view selector. This suggests the original shuffle result was a high-variance diagnostic of endpoint overfitting, while the more scalable mechanism is endpoint-free action-envelope calibration.

The strongest current method-shaped result is not "always shuffle." It is a two-stage critic: use endpoint-free envelope calibration to detect under-actuation in ordinary candidate pools, then use state/contact-conditioned rollout evidence to reject energy-matched corruptions. Unordered endpoint dropout remains useful as a diagnostic, but the n16 result shows that the simplest envelope moments are already close to a no-learning action-magnitude heuristic unless the candidate pool is stress-tested.

The next method should not be "always shuffle actions." A safer direction is:

1. learn when temporal order is reliable;
2. use endpoint-free envelope and endpoint-dropout views as conservative failure detectors, but keep shuffle-robust controls as a diagnostic baseline;
3. add energy-matched hard negatives to test whether the envelope critic understands action geometry or mostly corrects an under-actuation prior;
4. add state/contact-conditioned rollout features for Faucet-style and cabinet-contact interactions where successful candidates exist but compact action statistics still miss them.

## Reviewer Caveats

- Candidate generation is still replay perturbation, not a learned policy.
- The strongest deterministic heuristic on ordinary no-demo pools is action magnitude; the energy-matched hard-negative pool shows that this shortcut collapses, and the n16 state-trace proxy shows that nearly all of the remaining gap is recoverable from robot/contact rollout state/context.
- The action traces are sparse snapshots stored with stride 25, so this diagnostic does not rule out high-frequency temporal information.
- The strongest current table uses four tasks and sixteen episodes per task. It is stronger than the original 32-case diagnostic, and the n16 hard-negative result is now saturated for robot-only traces, but it still needs more RoboCasa365 tasks or a learned proposal source before becoming a headline benchmark table.
- The shuffled controls should be presented as a warning against overclaiming temporal world modeling, not as proof that action order is irrelevant.
