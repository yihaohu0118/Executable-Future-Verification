# ICLR Idea: Failure-Gated Action-Visual Critics for Recoverable Manipulation Planning Failures

## One-Sentence Idea

Train a lightweight action/video critic to detect recoverable planning failures and use it as a gated override, rather than replacing the planner or visual-progress heuristic globally.

## Why This Is Not Training-Free

The core component is trained case-heldout:

- action-sequence MLP selector
- video-frame MLP selector
- future gate over visual-progress and action/video critic confidence

The method is not a static heuristic, prompt, hand-coded score, or training-free reranker.

## Code-Driven Origin

This idea came from implementation and benchmark behavior, not from paper-level recombination.

### PushT-100 Observation

The PushT-100 first layer showed:

| Selector | Mean state dist | Oracle match |
| --- | ---: | ---: |
| NanoWM rank0 | 190.9580 | 21/100 |
| Static DINO progress | 181.1198 | 41/100 |
| ActionWorld e200 | 178.8655 | 39/100 |
| Static DINO progress + ActionWorld failure gate | 177.3960 | 43/100 |
| Oracle-best CEM | 157.4419 | 100/100 |

The important mechanism was not "world model score always wins." The useful behavior was gated override: keep the static visual-progress anchor unless the trained action-world critic predicts failure.

### ManiSkill PickCube Observation

The ManiSkill adapter exposed a concrete recoverable failure:

- `high_grasp` looks plausible but fails frequently.
- `low_grasp` solves the same initial states.
- A small grasp-height candidate ordering detail creates a large success gap.

20-case headroom:

| Metric | Value |
| --- | ---: |
| Rank0 brittle-grasp success | 4/20 |
| Oracle-best success | 20/20 |
| Oracle better than rank0 | 20/20 |

Candidate-level result:

| Candidate | Success |
| --- | ---: |
| `high_grasp` | 4/20 |
| `low_grasp` | 20/20 |
| `rank0_center` | 20/20 |
| `slow_center` | 20/20 |
| `x_offset` | 20/20 |

## Implemented Evidence

### Action-Sequence Selector

Case-heldout MLP over raw action-sequence statistics:

| Selector | Success | Recovered rank0 failures | Oracle match |
| --- | ---: | ---: | ---: |
| Rank0 brittle grasp | 4/20 | 0/16 | 0/20 |
| Raw action-sequence MLP | 20/20 | 16/16 | 13/20 |
| Raw action MLP without trajectory length | 20/20 | 16/16 | 13/20 |
| Raw action MLP on shuffled manifest rows | 20/20 | 16/16 | 13/20 |
| Zero-action negative control | 4/20 | 0/16 | 0/20 |
| Shuffle-time action control | 20/20 | 16/16 | 8/20 |

N100 stability check:

| Selector | Success | Recovered rank0 failures | Oracle match |
| --- | ---: | ---: | ---: |
| Rank0 brittle grasp | 17/100 | 0/83 | 0/100 |
| Raw action MLP without trajectory length | 100/100 | 83/83 | 71/100 |
| Oracle-best candidate | 100/100 | 83/83 | 100/100 |

Interpretation:

- The learned critic recovers every rank0 failure.
- The zero control proves the result is not only class prior or candidate ordering.
- The no-length control proves the result is not explained by early termination or trajectory length.
- Shuffling manifest rows does not change the result, so JSONL row order is not the shortcut.
- Shuffle-time staying strong means this slice is mostly action-distribution/stage-geometry driven, not precise temporal-order driven.
- The n100 stability check shows the action-critic result is not a small-seed accident.

### Video-Frame Selector

Case-heldout MLP over rendered RGB frame features:

| Selector | Success | Recovered rank0 failures | Oracle match |
| --- | ---: | ---: | ---: |
| Rank0 brittle grasp | 4/20 | 0/16 | 0/20 |
| Raw video-frame MLP | 20/20 | 16/16 | 13/20 |
| Raw video-frame MLP on shuffled manifest rows | 20/20 | 16/16 | 13/20 |
| Zero-video negative control | 4/20 | 0/16 | 0/20 |
| Shuffle-time video control | 20/20 | 16/16 | 10/20 |

Interpretation:

- The failure is visible from rollout frames.
- Zero-video control falls back to rank0.
- Shuffling manifest rows does not change the result, so row order is not the shortcut.
- Shuffle-time staying strong again warns against overclaiming temporal reasoning on this slice.

### Action-Video Fusion Selector

Case-heldout MLP over action and video critic scores:

| Selector | Success | Recovered rank0 failures | Oracle match |
| --- | ---: | ---: | ---: |
| Action-video fusion MLP | 20/20 | 16/16 | 2/20 |

Interpretation:

- Fusion recovers all failures but has low oracle match because many successful candidates are near-equivalent under the binary success metric.
- For this diagnostic slice, success recovery is the primary metric; oracle match is less informative because `rank0_center`, `low_grasp`, `x_offset`, and `slow_center` often all succeed.

## Concrete Method Direction

Name:

**FAVC: Failure-Gated Action-Visual Critic**

Core pipeline:

1. A planner or policy proposes K candidates.
2. A static visual-progress anchor selects the default candidate.
3. A trained action/video critic scores candidate failure likelihood.
4. A learned gate overrides the anchor only when the anchor is likely to fail.
5. The final selector is evaluated on task success and hard-case recovery.

The key claim should be:

> Recoverable planning failures are often easier to identify than to solve from scratch; a trained action-visual critic can detect these failures and selectively override brittle planner choices.

## Why This Is Potentially Novel

The common framing is to build a stronger planner, larger world model, or globally better reward. Our evidence points to a narrower and more counterintuitive mechanism:

- global replacement is not always best;
- a simple trained critic is valuable as a failure detector;
- the critic can recover failures caused by small action-geometry mistakes;
- action/video signals can expose failure even when the candidate appears plausible;
- in the current slice, temporal order is less important than expected.

## Current Weaknesses

These must be addressed before this is paper-ready:

1. ManiSkill candidate source is privileged diagnostic, not a policy-generated baseline.
2. PickCube brittle-grasp profile is intentionally constructed.
3. Current video/action selectors may exploit candidate-family regularities.
4. Current phenomenon does not prove temporal world modeling because shuffle-time controls remain strong.
5. Need a second task or policy-generated candidate source for external validity.
6. Need a harder fusion benchmark because current action/video critics each solve the slice independently.

## Next Experiments

Priority order:

1. Randomize candidate family positions, not only JSONL row order.
2. Add a second ManiSkill task with genuine success headroom.
3. Train a gate that combines visual-progress anchor and action/video critic on heterogeneous failures.
4. Replace privileged candidates with BC/diffusion-policy top-k samples.
5. Move to LIBERO once the ManiSkill gate is stable.

## Implemented Files

- `src/umm_reward_evaluator/benchmarks/maniskill_candidate_pool.py`
- `src/umm_reward_evaluator/benchmarks/train_action_sequence_selector.py`
- `src/umm_reward_evaluator/benchmarks/train_video_frame_selector.py`
- `src/umm_reward_evaluator/benchmarks/train_action_video_fusion_selector.py`
- `src/umm_reward_evaluator/benchmarks/shuffle_manifest_rows.py`
- `docs/maniskill_pickcube_brittle_grasp_headroom.md`
- `docs/maniskill_pickcube_action_selector_results.md`
- `docs/maniskill_pickcube_video_selector_results.md`
- `docs/maniskill_pickcube_n100_action_stability.md`

## Newly Added Verification Hooks

The implementation now includes reviewer-oriented controls:

1. `raw_no_length` action features: removes trajectory length as a possible success shortcut while preserving action statistics.
2. Candidate-row shuffling: tests whether results are invariant to JSONL order. Selector tie-breaking now prefers planner rank0 when scores are equal, so zero controls are well-defined and not row-order dependent.
3. Action-video fusion: tests whether the FAVC critic can combine independently trained action and visual critics in a held-out protocol.

The implementation also includes `train_action_video_fusion_selector.py`, a case-heldout fusion critic over action and video selector scores. This is the first minimal FAVC implementation beyond separate action-only and video-only diagnostics.
