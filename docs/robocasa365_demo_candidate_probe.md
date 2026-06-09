# RoboCasa365 Demo Candidate Probe

## Purpose

This is the first executable RoboCasa365 candidate-selection probe. The goal is not to claim a final method result yet. The goal is to verify that a current 2026 benchmark exposes recoverable action-selection failures under the same rank0-vs-oracle protocol used in PushT and ManiSkill.

## Benchmark And Data

- Benchmark: RoboCasa365
- Task: `PickPlaceCounterToCabinet`
- Split/source: target human demonstrations
- Dataset path on `dev2`:
  `/home/yihao_hyh/benchmarks/robocasa/datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot`
- Download size: 561 MB tarball
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

## Observation

The surprising part is the action calibration cliff:

- Scaling the full demonstration action sequence by 0.85 fails in all first five target episodes.
- Adding small noise with std 0.03 still succeeds in all first five target episodes.
- Truncating the final 20% fails in all first five target episodes.

This suggests that for RoboCasa-style long-horizon manipulation, the failure mode may be less about generic action noise and more about systematic action-scale or temporal-completion errors. That is aligned with the failure-gated critic story: a useful critic should detect physically plausible but under-executed candidates, not just visually plausible endpoints.

## Reviewer Risk

This probe is not yet sufficient as a final benchmark result:

- Rank0 is intentionally brittle.
- The candidate profiles are fixed, so a selector could exploit candidate identity or simple action-energy shortcuts.
- `demo_original` is an oracle-like candidate source if presented as a policy output.

## Next Experiment

The next fairer RoboCasa365 experiment should keep the same replay infrastructure but change the candidate source:

1. Use original demo actions only as supervision or an oracle upper bound.
2. Generate rank0 from a non-oracle policy score, likelihood score, or noisy BC model.
3. Randomize scale, noise, truncation, and temporal warp per episode so candidate identity is not enough.
4. Train/evaluate the action critic under held-out episodes.
5. Report:
   - rank0 success
   - oracle-best success
   - global action critic success
   - failure-gated action critic success
   - recovered rank0 failures

The target publishable claim remains:

> In current household manipulation benchmarks, small action-calibration errors can dominate success even when candidates are close to demonstrations; a failure-gated action critic can recover these cases better than blindly trusting rank0 policy likelihood or visual plausibility.
