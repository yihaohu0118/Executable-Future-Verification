# ManiSkill PickCube Video-Frame Selector Results

## Setup

Input manifest:

- `outputs/maniskill_pickcube_brittle_grasp_n20_video/PickCube-v1_candidate_manifest.jsonl`
- 20 cases, 5 candidates per case
- Short rendered rollout videos for every candidate
- Rank0 profile: `brittle_grasp`

Selector:

- Case-heldout MLP.
- Input is sampled RGB rollout frames.
- Frames are downsampled to `32x32`, six frames per candidate.
- Features are projected with a fixed random projection inside each fold.
- No action sequence, candidate id, metadata, oracle state, or privileged object pose is used.

## Results

| Selector | Success | Recovered rank0 failures | Oracle match |
| --- | ---: | ---: | ---: |
| Rank0 brittle grasp | 4/20 | 0/16 | 0/20 |
| Raw video-frame MLP | 20/20 | 16/16 | 13/20 |
| Zero-video negative control | 4/20 | 0/16 | 0/20 |
| Shuffle-time video control | 20/20 | 16/16 | 10/20 |
| Oracle-best candidate | 20/20 | 16/16 | 20/20 |

Raw video selector choices:

| Candidate | Selected cases |
| --- | ---: |
| `low_grasp` | 16 |
| `x_offset` | 3 |
| `slow_center` | 1 |

Zero-video control choices:

| Candidate | Selected cases |
| --- | ---: |
| `high_grasp` | 20 |

## Interpretation

The visual selector recovers every brittle-grasp failure using only rendered video frames. The zero-video control falls back to rank0, so the result is not explained by case ordering or a constant class prior.

The shuffle-time control also succeeds. This mirrors the action-sequence selector result and suggests that the visible failure signal is mostly stage appearance or final configuration, not precise frame order. This is useful for mechanism discovery but weakens any claim that this particular slice requires temporal reasoning.

## Current ICLR Evidence Chain

The current external-benchmark evidence is:

1. ManiSkill PickCube has recoverable candidate-selection headroom: rank0 `4/20`, oracle-best `20/20`.
2. A held-out action-sequence selector recovers `16/16` rank0 failures.
3. A held-out video-frame selector also recovers `16/16` rank0 failures.
4. Corrected zero controls fail, so the selectors use nontrivial action/video signals.
5. Shuffle-time controls remain strong, so the current phenomenon is not primarily temporal-order dependent.

This supports a training-based failure-detection claim, but the next benchmark slice must be harder: either a task where temporal order matters or a policy-generated candidate source where candidate families are less hand-designed.

## Next Required Experiment

Move from diagnostic candidates to one of:

- A larger PickCube sweep with randomized candidate order and no family-specific rank pattern.
- A second ManiSkill task with true success headroom.
- A BC/diffusion-policy top-k candidate source.

For the paper story, the strongest next result would be:

> visual progress alone is strong, action critic alone is strong, but a learned gate is more robust when candidate failures are heterogeneous.

