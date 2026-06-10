# RoboCasa365 Demo Candidate Probe

## Purpose

This is the first executable RoboCasa365 candidate-selection probe. The goal is not to claim a final method result yet. The goal is to verify that a current 2026 benchmark exposes recoverable action-selection failures under the same rank0-vs-oracle protocol used in PushT and ManiSkill.

## Benchmark And Data

- Benchmark: RoboCasa365
- Tasks: `PickPlaceCounterToCabinet`, `TurnOnSinkFaucet`, `OpenCabinet`, `TurnOnMicrowave`
- Split/source: target human demonstrations
- Dataset path on `dev2`:
  - `/home/yihao_hyh/benchmarks/robocasa/datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot`
  - `/home/yihao_hyh/benchmarks/robocasa/datasets/v1.0/target/atomic/TurnOnSinkFaucet/20250812/lerobot`
  - `/home/yihao_hyh/benchmarks/robocasa/datasets/v1.0/target/atomic/OpenCabinet/20250813/lerobot`
  - `/home/yihao_hyh/benchmarks/robocasa/datasets/v1.0/target/atomic/TurnOnMicrowave/20250813/lerobot`
- Download sizes: 561 MB tarball for `PickPlaceCounterToCabinet`, 398 MB tarball for `TurnOnSinkFaucet`, 788 MB tarball for `OpenCabinet`, 341 MB tarball for `TurnOnMicrowave`
- Observation exposed by smoke adapter:
  - language instruction
  - proprioceptive state
  - three 256x256 RGB views
- Action shape in first five episodes: variable horizon, 12D raw robosuite actions

## Code Path

Script:

```bash
PYTHONPATH=/tmp python -m umm_reward_evaluator.benchmarks.robocasa_demo_candidate_pool \
  --dataset /home/yihao_hyh/benchmarks/robocasa/datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot \
  --task-name PickPlaceCounterToCabinet \
  --suite target-human \
  --num-episodes 5 \
  --action-stride 25 \
  --output-dir /tmp/robocasa365_demo_pool_n5
```

Implementation details:

- Uses official `robocasa.utils.lerobot_utils` to read episode states, XML, metadata, and actions.
- Uses official playback reset logic to restore the same initial simulator state.
- Executes action candidates open-loop in robosuite.
- Uses `env._check_success()` as the oracle success label.
- Writes the shared candidate JSONL schema used by the ManiSkill selectors.
- Uses `filter_candidate_manifest.py` to remove original-demonstration candidates and recompute oracle-best labels for no-demo subsets.

## Candidate Pool

Each case uses five candidates from the same initial state:

| Rank | Candidate | Transform |
| ---: | --- | --- |
| 0 | `cand_00_underactuated_065` | scale all actions by 0.65 |
| 1 | `cand_01_demo_original` | original demonstration actions |
| 2 | `cand_02_underactuated_085` | scale all actions by 0.85 |
| 3 | `cand_03_noisy_003` | add clipped Gaussian action noise, std 0.03 |
| 4 | `cand_04_truncated_080` | zero actions after 80% of the trajectory |

This intentionally creates a brittle rank0. It should be treated as a headroom probe, not as the final planner baseline.

The second probe randomizes the candidate profiles per episode:

- Candidate IDs are reassigned after ranking as `cand_00`, `cand_01`, ...
- The rank is assigned by a non-oracle conservative action-energy prior.
- Each sampled candidate applies a random action scale, optional small noise, and optional truncation.
- The true transform parameters are stored only in metadata.
- The original demonstration can appear in the pool, but it is not rank0 under the conservative prior.

## Results

Two-episode smoke:

| Metric | Value |
| --- | ---: |
| Cases | 2 |
| Rank0 success | 0/2 |
| Oracle-best success | 2/2 |
| Oracle better than rank0 | 2/2 |
| Rank0 oracle match | 0/2 |

Five-episode probe:

| Metric | Value |
| --- | ---: |
| Cases | 5 |
| Rank0 success | 0/5 |
| Oracle-best success | 5/5 |
| Oracle better than rank0 | 5/5 |
| Rank0 oracle match | 0/5 |

Per-candidate success on five episodes:

| Candidate | Success | Oracle-best count |
| --- | ---: | ---: |
| `cand_00_underactuated_065` | 0/5 | 0 |
| `cand_01_demo_original` | 5/5 | 5 |
| `cand_02_underactuated_085` | 0/5 | 0 |
| `cand_03_noisy_003` | 5/5 | 0 |
| `cand_04_truncated_080` | 0/5 | 0 |

Randomized five-episode probe, seed 7, eight candidates per episode:

| Metric | Value |
| --- | ---: |
| Cases | 5 |
| Rank0 success | 0/5 |
| Oracle-best success | 5/5 |
| Oracle better than rank0 | 5/5 |
| Rank0 oracle match | 0/5 |

Held-out action selector on the randomized five-episode manifest:

| Selector | Success | Recovered rank0 failures | Oracle match |
| --- | ---: | ---: | ---: |
| Raw action statistics, no length | 5/5 | 5/5 | 4/5 |
| Zero-feature control | 0/5 | 0/5 | 0/5 |
| Shuffled-time action statistics | 5/5 | 5/5 | 3/5 |
| Failure-gated raw action selector | 5/5 | 5/5 | n/a |

Randomized eight-episode probe, seed 11, six candidates per episode:

| Metric | With demo candidate | No-demo subset |
| --- | ---: | ---: |
| Cases | 8 | 8 |
| Rank0 success | 0/8 | 0/8 |
| Oracle-best success | 8/8 | 6/8 |
| Oracle better than rank0 | 8/8 | 6/8 |
| Rank0 oracle match | 0/8 | 2/8 |

Candidate-rank success in the randomized eight-episode probe:

| Rank under conservative prior | Success |
| ---: | ---: |
| 0 | 0/8 |
| 1 | 0/8 |
| 2 | 2/8 |
| 3 | 3/8 |
| 4 | 6/8 |
| 5 | 8/8 |

The original demonstration is not rank0 under the conservative prior. Its ranks across the eight episodes are 5, 5, 4, 4, 5, 4, 3, and 5.

Perturbed-only candidates are not all failures: 11/40 randomized non-demo candidates succeed. Removing original demonstrations still leaves oracle-best success at 6/8.

Held-out selectors on the randomized eight-episode manifest:

| Manifest | Selector | Success | Recovered rank0 failures | Oracle match |
| --- | --- | ---: | ---: | ---: |
| With demo | Raw action statistics, no length | 8/8 | 8/8 | 5/8 |
| With demo | Zero-feature control | 0/8 | 0/8 | 0/8 |
| With demo | Shuffled-time action statistics | 8/8 | 8/8 | 5/8 |
| With demo | Failure-gated raw action selector | 8/8 | 8/8 | n/a |
| No demo | Raw action statistics, no length | 5/8 | 5/8 | 4/8 |
| No demo | Zero-feature control | 0/8 | 0/8 | 2/8 |
| No demo | Shuffled-time action statistics | 6/8 | 6/8 | 5/8 |
| No demo | Failure-gated raw action selector | 5/8 | 5/8 | n/a |

Randomized eight-episode probe on `TurnOnSinkFaucet`, seed 11, six candidates per episode:

| Metric | With demo candidate | No-demo subset |
| --- | ---: | ---: |
| Cases | 8 | 8 |
| Rank0 success | 0/8 | 0/8 |
| Oracle-best success | 8/8 | 7/8 |
| Oracle better than rank0 | 8/8 | 7/8 |
| Rank0 oracle match | 0/8 | 1/8 |

Held-out selectors on `TurnOnSinkFaucet`:

| Manifest | Selector | Success | Recovered rank0 failures | Oracle match |
| --- | --- | ---: | ---: | ---: |
| With demo | Raw action statistics, no length | 8/8 | 8/8 | 3/8 |
| With demo | Zero-feature control | 0/8 | 0/8 | 0/8 |
| With demo | Shuffled-time action statistics | 8/8 | 8/8 | 3/8 |
| With demo | Failure-gated raw action selector | 8/8 | 8/8 | n/a |
| No demo | Raw action statistics, no length | 2/8 | 2/8 | 2/8 |
| No demo | Zero-feature control | 0/8 | 0/8 | 1/8 |
| No demo | Shuffled-time action statistics | 3/8 | 3/8 | 2/8 |
| No demo | Failure-gated raw action selector | 2/8 | 2/8 | n/a |

Randomized eight-episode probe on `OpenCabinet`, seed 11, six candidates per episode:

| Metric | With demo candidate | No-demo subset |
| --- | ---: | ---: |
| Cases | 8 | 8 |
| Rank0 success | 0/8 | 0/8 |
| Oracle-best success | 8/8 | 6/8 |
| Oracle better than rank0 | 8/8 | 6/8 |
| Rank0 oracle match | 0/8 | 2/8 |

Held-out selectors on `OpenCabinet`:

| Manifest | Selector | Success | Recovered rank0 failures | Oracle match |
| --- | --- | ---: | ---: | ---: |
| With demo | Raw action statistics, no length | 7/8 | 7/8 | 6/8 |
| With demo | Zero-feature control | 0/8 | 0/8 | 0/8 |
| With demo | Shuffled-time action statistics | 8/8 | 8/8 | 5/8 |
| No demo | Raw action statistics, no length | 6/8 | 6/8 | 5/8 |
| No demo | Zero-feature control | 0/8 | 0/8 | 2/8 |
| No demo | Shuffled-time action statistics | 6/8 | 6/8 | 5/8 |

Randomized eight-episode probe on `TurnOnMicrowave`, seed 11, six candidates per episode:

| Metric | With demo candidate | No-demo subset |
| --- | ---: | ---: |
| Cases | 8 | 8 |
| Rank0 success | 0/8 | 0/8 |
| Oracle-best success | 8/8 | 6/8 |
| Oracle better than rank0 | 8/8 | 6/8 |
| Rank0 oracle match | 0/8 | 2/8 |

Held-out selectors on `TurnOnMicrowave` no-demo:

| Selector | Seeds | Success | Oracle match |
| --- | --- | ---: | ---: |
| Raw action statistics, no length | 0,1,2 | 3,2,3 / 8 | 3,2,3 / 8 |
| Shuffled-time action statistics | 0,1,2 | 3,4,3 / 8 | 3,4,3 / 8 |
| Four pseudo-endpoint pairs, no length | 0,1,2 | 5,3,5 / 8 | 5,3,5 / 8 |
| Bag action-envelope moments, no length | 0,1,2 | 4,4,3 / 8 | 4,3,3 / 8 |

Two-task multitask selector results:

| Manifest | Feature | Task mode | Overall success | Pick success | Faucet success | Oracle ceiling | Zero-feature control |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| With demo | Raw action statistics, no length | shared one-hot | 16/16 | 8/8 | 8/8 | 16/16 | 0/16 |
| No demo | Raw action statistics, no length | shared one-hot | 7/16 | 5/8 | 2/8 | 13/16 | n/a |
| No demo | Raw action statistics, no length | per-task head | 8/16 | 5/8 | 3/8 | 13/16 | 0/16 |
| No demo | Raw action statistics, no length | independent per task | 7/16 | 5/8 | 2/8 | 13/16 | n/a |
| No demo | Shuffled-time action statistics | per-task head | 8/16 | 6/8 | 2/8 | 13/16 | n/a |

Three-task multitask selector results:

| Manifest | Feature | Task mode | Seeds | Overall success | Pick success | Faucet success | OpenCabinet success | Oracle ceiling |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| With demo | Raw action statistics, no length | shared one-hot | 0 | 24/24 | 8/8 | 8/8 | 8/8 | 24/24 |
| With demo | Shuffled-time action statistics | shared one-hot | 0 | 24/24 | 8/8 | 8/8 | 8/8 | 24/24 |
| With demo | Zero-feature control | shared one-hot | 0 | 1/24 | 0/8 | 1/8 | 0/8 | 24/24 |
| No demo | Raw action statistics, no length | per-task head | 0,1,2 | 14,14,13 / 24 | 6,5,6 / 8 | 2,3,1 / 8 | 6,6,6 / 8 | 19/24 |
| No demo | Bag action-envelope moments, no length | per-task head | 0,1,2 | 14,14,14 / 24 | 6,6,6 / 8 | 2,2,2 / 8 | 6,6,6 / 8 | 19/24 |
| No demo | Four pseudo-endpoint pairs, no length | per-task head | 0,1,2 | 16,16,17 / 24 | 6,6,6 / 8 | 4,4,5 / 8 | 6,6,6 / 8 | 19/24 |
| No demo | Shuffled-time action statistics | per-task head | 0,1,2 | 17,16,17 / 24 | 6,6,6 / 8 | 5,4,5 / 8 | 6,6,6 / 8 | 19/24 |
| No demo | Phase summaries | per-task head | 0,1,2 | 15,15,14 / 24 | 6,6,6 / 8 | 3,3,2 / 8 | 6,6,6 / 8 | 19/24 |
| No demo | Phase summaries after time shuffle | per-task head | 0,1,2 | 17,16,16 / 24 | 6,6,6 / 8 | 5,4,4 / 8 | 6,6,6 / 8 | 19/24 |

Four-task no-demo multitask selector results:

| Feature | Task mode | Seeds | Overall success | Pick | Faucet | OpenCabinet | Microwave | Oracle ceiling |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw action statistics, no length | per-task head | 0,1,2 | 18,17,16 / 32 | 6,6,6 / 8 | 2,2,1 / 8 | 6,6,6 / 8 | 4,3,3 / 8 | 25/32 |
| Bag action-envelope moments, no length | per-task head | 0,1,2 | 19,18,16 / 32 | 6,6,6 / 8 | 3,3,2 / 8 | 6,5,6 / 8 | 4,4,2 / 8 | 25/32 |
| Four pseudo-endpoint pairs, no length | per-task head | 0,1,2 | 19,18,16 / 32 | 5,6,6 / 8 | 5,4,3 / 8 | 6,6,6 / 8 | 3,2,3 / 8 | 25/32 |
| One unordered pseudo-endpoint pair, no length | per-task head | 0,1,2 | 20,20,20 / 32 | 6,6,6 / 8 | 4,4,5 / 8 | 6,6,6 / 8 | 4,4,3 / 8 | 25/32 |
| Four unordered pseudo-endpoint pairs, no length | per-task head | 0,1,2 | 17,19,20 / 32 | 6,6,6 / 8 | 5,5,5 / 8 | 6,5,6 / 8 | 1,3,3 / 8 | 25/32 |
| Shuffled-time action statistics | per-task head | 0,1,2 | 22,19,19 / 32 | 6,6,6 / 8 | 5,5,4 / 8 | 6,5,6 / 8 | 5,2,3 / 8 | 25/32 |
| Multiview meta: raw + one unordered endpoint | per-task head | 0,1,2 | 20,19,16 / 32 | 6,6,6 / 8 | 3,3,2 / 8 | 6,6,6 / 8 | 5,4,2 / 8 | 25/32 |
| Multiview meta: one unordered endpoint + shuffled-time | per-task head | 0,1,2 | 21,20,20 / 32 | 6,6,6 / 8 | 4,5,5 / 8 | 6,6,6 / 8 | 5,3,3 / 8 | 25/32 |

Expanded four-task no-demo multitask selector results:

The n16 manifests merge the original eight target episodes with newly generated episodes 8-15 under the same random no-demo candidate protocol. Rank0 remains 0/64 and oracle-best is 41/64.

| Feature | Task mode | Seeds | Overall success | Mean | Pick mean | Faucet mean | OpenCabinet mean | Microwave mean | Oracle ceiling |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw action statistics, no length | per-task head | 0,1,2,3,4 | 24,20,20,22,22 / 64 | 21.6 | 5.0 | 3.8 | 7.0 | 5.8 | 41/64 |
| Endpoint-free stats, no length | per-task head | 0,1,2,3,4 | 25,27,27,24,26 / 64 | 25.8 | 6.4 | 6.0 | 7.8 | 5.6 | 41/64 |
| Bag action-envelope moments, no length | per-task head | 0,1,2,3,4 | 31,27,27,30,28 / 64 | 28.6 | 6.4 | 5.8 | 8.8 | 7.6 | 41/64 |
| One unordered pseudo-endpoint pair, no length | per-task head | 0,1,2,3,4 | 25,27,26,29,30 / 64 | 27.4 | 7.0 | 6.8 | 7.8 | 5.8 | 41/64 |
| Shuffled-time action statistics | per-task head | 0,1,2,3,4 | 26,24,25,25,26 / 64 | 25.2 | 5.6 | 7.4 | 7.2 | 5.0 | 41/64 |
| Multiview meta: one unordered endpoint + shuffled-time | per-task head | 0,1,2 | 26,25,24 / 64 | 25.0 | 5.7 | 7.3 | 7.0 | 5.0 | 41/64 |

Deterministic n16 heuristic controls:

| Heuristic | Overall success | Pick | Faucet | OpenCabinet | Microwave |
| --- | ---: | ---: | ---: | ---: | ---: |
| Max mean absolute action | 28/64 | 6/16 | 8/16 | 8/16 | 6/16 |
| Max sum action energy | 27/64 | 6/16 | 7/16 | 7/16 | 7/16 |
| Max action energy | 26/64 | 7/16 | 6/16 | 6/16 | 7/16 |
| Max planner rank / lowest conservative prior | 26/64 | 7/16 | 5/16 | 6/16 | 8/16 |
| Min action energy / rank0-style prior | 0/64 | 0/16 | 0/16 | 0/16 | 0/16 |

## Observation

The surprising part is the action calibration cliff:

- Scaling the full demonstration action sequence by 0.85 fails in all first five target episodes.
- Adding small noise with std 0.03 still succeeds in all first five target episodes.
- Truncating the final 20% fails in all first five target episodes.
- In the randomized pool, a held-out selector recovers all rank0 failures while the zero-feature control recovers none.
- Shuffling action time still performs well, suggesting the current signal is dominated by action calibration statistics rather than fine temporal ordering.
- In the no-demo subset, shuffled-time action statistics reach the 6/8 oracle ceiling while raw ordered statistics reach 5/8. More temporal structure is not automatically better on this diagnostic.
- The same pattern transfers to `TurnOnSinkFaucet` when the original demo is present, but the no-demo subset is much harder: oracle-best remains 7/8 while action-statistic selectors recover only 2-3/8.
- The two-task no-demo result is a useful boundary case. Task-conditioned heads improve slightly over a shared/global critic, but neither closes the 13/16 oracle ceiling. That points to task/contact-conditioned calibration as the next method need.
- `OpenCabinet` adds a third target task and a longer fixture-interaction stress test. In no-demo, raw and shuffled-time selectors both reach the 6/8 oracle ceiling, while zero features recover 0/8. This shows the Faucet failure is not simply because all fixture tasks are impossible for action statistics.
- Across three tasks, with-demo selectors recover all 24 rank0 failures. In no-demo, raw ordered action statistics recover only 13-14/24 against a 19/24 oracle ceiling, while shuffled-time statistics recover 16-17/24. The gain is concentrated in `TurnOnSinkFaucet`.
- Phase features do not fix Faucet when temporal order is preserved. Counterintuitively, phase features after time shuffling perform better than ordered phase features. Simple bag-of-actions moments also fail to match shuffle, so the effect is not explained by generic order-invariant statistics alone.
- Multi-pseudo-endpoint features nearly match shuffled-time statistics on the three-task no-demo setting: 16,16,17/24 vs 17,16,17/24. This makes endpoint-dropout calibration the cleanest method-shaped version of the shuffle diagnostic so far. See `docs/robocasa365_temporal_shuffle_diagnostic.md`.
- `TurnOnMicrowave` adds a second button-style task. It has the same rank0 failure pattern and a 6/8 no-demo oracle ceiling. In single-task training, multi-pseudo-endpoints recover 5,3,5/8, better than raw 3,2,3/8. In the initial 32-case four-task multitask training, shuffled-time remains the strongest and most stable overall feature, while pseudo-endpoints become a partial explanation rather than a complete replacement.
- Unordered pseudo-endpoints sharpen the mechanism. One stable permutation-derived endpoint pair gets 20,20,20/32 in four-task no-demo multitask evaluation, but four such pairs drop to 17,19,20/32. This is a useful negative result: adding more randomized endpoint evidence can hurt stability, so the method should learn when to trust temporal/detail features rather than simply add more of them.
- Multiview calibration partially turns the diagnostic into a method. An outer-isolated logistic calibrator over one unordered endpoint view plus shuffled-time view reaches 21,20,20/32, compared with 22,19,19/32 for shuffled-time alone and 20,20,20/32 for unordered endpoints alone. Simple rank aggregation of the same views stays at 20,19,19/32, so the useful effect is learned stabilization rather than agreement voting.
- The 64-case expansion changes the strongest mechanism. Bag action-envelope moments reach a 28.6/64 mean and one unordered endpoint pair reaches 27.4/64, both above shuffled-time at 25.2/64 and raw ordered summaries at 21.6/64. Endpoint-free stats without energy already reach 25.8/64, so removing brittle first/last endpoints is a major source of the gain; adding energy/envelope moments gives the best current result.
- The deterministic max-absolute-action heuristic reaches 28/64 without training, nearly matching the learned bag critic. This is an important shortcut control, not a failure: it exposes a strong under-actuation bias in the conservative rank0 prior.
- Energy-matched hard negatives break the shortcut. On a four-task n4 stress pool, original demo actions succeed in 16/16, all time-reverse/roll/shuffle/block-swap/sign-flip corruptions fail, and magnitude/energy/smoothness heuristics fall to 0/16. Learned action-only selectors recover only 5-6/16 on average. This means the ordinary no-demo pool mostly tests under-actuation calibration, while the energy-matched pool exposes the remaining contact-timing and direction bottleneck.
- Low-dimensional rollout state traces mostly close the energy-matched gap. The first n4 stress protocol gave zero 0/16, best action-only 6.2/16, and state-trace 13,15,14,14,15/16 across five seeds. After scaling to n8, zero remains 0/32, deterministic action energy/magnitude/smoothness heuristics remain weak, action-only endpoint-free stats average 8.2/32, and state traces recover 30,31,31,30,30/32. Adding endpoint-free action stats to state traces is again slightly worse at 30.0/32 mean. A manifest-level rank/candidate-id randomization control keeps rank0 failed, moves the successful original action across nonzero ranks, and leaves state-trace performance unchanged at 30.4/32 while action-only remains 8.4/32. A fully regenerated random-position rollout pool repeats the same result: zero 0/32, action-only 8.4/32, state trace 30.4/32, and state+action 30.0/32. Key ablations show this is not merely privileged `object-state`: excluding `object-state` reaches 31.4/32, robot-only reaches 31.2/32, proprio-only reaches 30.2/32, while object low-dimensional keys alone get only 16/32. Scaling that regenerated random-position pool to n16 gives 64/64 oracle and 0/64 rank0. Action-only endpoint-free stats average 28.4/64, object-only remains 31.0/64, proprio-only reaches 63.0/64, full state reaches 63.0/64, state+action is worse at 62.0/64, and robot-only traces reach 64.0/64 across all five seeds. A finer n16 robot-key ablation shows that EEF/gripper traces without the broad proprio vector still reach 62.2/64, EEF position+gripper reaches 62.0/64, EEF position alone reaches 60.4/64, and gripper alone reaches 43.8/64. A summary-statistic ablation shows that terminal-only and endpoint summaries stay near 33-43/64, while endpoint-free EEF position distribution statistics reach 62.6/64 and EEF+gripper distribution statistics reach 63.6/64. A non-neural same-task nearest-positive prototype over EEF+gripper distribution features reaches 59/64, while all-task prototypes fall to 21-48/64. Case-heldout independent/shared/task-head MLPs all stay high at 62.4-63.6/64, but leave-one-task-out drops to 25.6/64. Few-shot target-task calibration recovers quickly: two cases reach 46.8/64, four reach 56.8/64, and eight reach 61.0/64. The next method should prioritize few-shot task/contact-conditioned EEF execution-envelope feedback rather than bigger action summaries or zero-shot cross-task claims.

This suggests that for RoboCasa-style long-horizon manipulation, the failure mode may be less about generic action noise and more about systematic action-scale or temporal-completion errors. That is aligned with the failure-gated critic story: a useful critic should detect physically plausible but under-executed candidates, not just visually plausible endpoints.

## Reviewer Risk

This probe is not yet sufficient as a final benchmark result:

- Rank0 is intentionally brittle.
- The fixed probe has candidate identity shortcuts; the randomized probe reduces but does not eliminate action-energy shortcut concerns.
- `demo_original` is an oracle-like candidate source if presented as a policy output.
- The largest randomized selector result so far uses sixteen episodes per task on four RoboCasa tasks. It is stronger than the original direction check, but still not enough for a final benchmark table without more tasks or a learned proposal source.
- Current ranking prior is intentionally conservative and non-oracle; the next version should compare against an actual learned policy likelihood or BC proposal.
- The no-demo Faucet result shows the current compact action statistic critic is not enough for articulated-fixture interaction. This is a weakness, but it is also the strongest evidence that the final method needs task-conditioned or contact-conditioned failure modeling.
- The max-absolute-action heuristic is a strong baseline on the n16 ordinary no-demo pool. Energy-matched candidates beat that shortcut: on the n16 regenerated random-position hard-negative pool, action-only selectors still leave a large oracle gap at 28.4/64, deterministic heuristics top out at 24/64, and robot/proprio traces close nearly all of it at 64.0/64 and 63.0/64. Endpoint-free EEF+gripper distribution statistics reach 63.6/64, so future results should target visual/contact prediction of execution envelopes, not just more action summaries.

## Next Experiment

The next fairer RoboCasa365 experiment should keep the same replay infrastructure and scale the randomized probe:

1. Run more target episodes and multiple random seeds.
2. Use original demo actions only as supervision or an oracle upper bound.
3. Generate rank0 from a non-oracle policy score, likelihood score, or noisy BC model.
4. Scale the fully regenerated random-position hard-negative pool beyond four tasks and convert the robot/proprio proxy into visual/contact-conditioned rollout critics.
5. Train/evaluate task-conditioned, visual-conditioned, and contact-conditioned rollout critics under held-out episodes.
6. Report:
   - rank0 success
   - oracle-best success
   - global action critic success
   - failure-gated action critic success
   - recovered rank0 failures

The target publishable claim remains:

> In current household manipulation benchmarks, small action-calibration errors can dominate success even when candidates are close to demonstrations; a failure-gated action critic can recover these cases better than blindly trusting rank0 policy likelihood or visual plausibility.
