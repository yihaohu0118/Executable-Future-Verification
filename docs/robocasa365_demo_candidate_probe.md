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
- `TurnOnMicrowave` adds a second button-style task. It has the same rank0 failure pattern and a 6/8 no-demo oracle ceiling. In single-task training, multi-pseudo-endpoints recover 5,3,5/8, better than raw 3,2,3/8. In four-task multitask training, shuffled-time remains the strongest and most stable overall feature, while pseudo-endpoints become a partial explanation rather than a complete replacement.
- Unordered pseudo-endpoints sharpen the mechanism. One stable permutation-derived endpoint pair gets 20,20,20/32 in four-task no-demo multitask evaluation, but four such pairs drop to 17,19,20/32. This is a useful negative result: adding more randomized endpoint evidence can hurt stability, so the method should learn when to trust temporal/detail features rather than simply add more of them.

This suggests that for RoboCasa-style long-horizon manipulation, the failure mode may be less about generic action noise and more about systematic action-scale or temporal-completion errors. That is aligned with the failure-gated critic story: a useful critic should detect physically plausible but under-executed candidates, not just visually plausible endpoints.

## Reviewer Risk

This probe is not yet sufficient as a final benchmark result:

- Rank0 is intentionally brittle.
- The fixed probe has candidate identity shortcuts; the randomized probe reduces but does not eliminate action-energy shortcut concerns.
- `demo_original` is an oracle-like candidate source if presented as a policy output.
- The largest randomized selector result so far uses eight episodes per task on four RoboCasa tasks, so it should be treated as a direction check, not a final benchmark table.
- Current ranking prior is intentionally conservative and non-oracle; the next version should compare against an actual learned policy likelihood or BC proposal.
- The no-demo Faucet result shows the current compact action statistic critic is not enough for articulated-fixture interaction. This is a weakness, but it is also the strongest evidence that the final method needs task-conditioned or contact-conditioned failure modeling.

## Next Experiment

The next fairer RoboCasa365 experiment should keep the same replay infrastructure and scale the randomized probe:

1. Run more target episodes and multiple random seeds.
2. Use original demo actions only as supervision or an oracle upper bound.
3. Generate rank0 from a non-oracle policy score, likelihood score, or noisy BC model.
4. Add harder controls that match action energy while changing direction or phase.
5. Train/evaluate task-conditioned and contact-conditioned action critics under held-out episodes.
6. Report:
   - rank0 success
   - oracle-best success
   - global action critic success
   - failure-gated action critic success
   - recovered rank0 failures

The target publishable claim remains:

> In current household manipulation benchmarks, small action-calibration errors can dominate success even when candidates are close to demonstrations; a failure-gated action critic can recover these cases better than blindly trusting rank0 policy likelihood or visual plausibility.
