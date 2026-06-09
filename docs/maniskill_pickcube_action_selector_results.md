# ManiSkill PickCube Action-Sequence Selector Results

## Setup

Input manifest:

- `outputs/maniskill_pickcube_brittle_grasp_n20/PickCube-v1_candidate_manifest.jsonl`
- 20 cases, 5 candidates per case
- Rank0 profile: `brittle_grasp`
- Rank0 is the high-grasp candidate

Selector:

- Case-heldout MLP.
- Each fold holds out one seed/case.
- Input features are summary statistics over raw action sequences.
- No candidate id, metadata, oracle state, or privileged object pose is used by the selector.

## Results

| Selector | Success | Recovered rank0 failures | Oracle match |
| --- | ---: | ---: | ---: |
| Rank0 brittle grasp | 4/20 | 0/16 | 0/20 |
| Raw action-sequence MLP | 20/20 | 16/16 | 13/20 |
| Raw action MLP without trajectory length | 20/20 | 16/16 | 13/20 |
| Raw action MLP on shuffled manifest rows | 20/20 | 16/16 | 13/20 |
| Zero-action negative control | 4/20 | 0/16 | 0/20 |
| Shuffle-time action control | 20/20 | 16/16 | 8/20 |
| Oracle-best candidate | 20/20 | 16/16 | 20/20 |

## Interpretation

The learned action selector fully recovers the brittle grasp failures on held-out cases. The corrected zero-action control falls back to rank0, so the result is not explained by class prior or candidate ordering.

The no-length control also recovers all failures, so the action result is not explained by early termination or trajectory length. Shuffling JSONL row order does not change the result, so row order is not the shortcut either.

The shuffle-time control still succeeds. This is a useful diagnostic: for this PickCube slice, the key signal is mostly in the action distribution and stage geometry, not precise temporal order. That makes the first external benchmark result simpler but also creates a reviewer risk. We should not overclaim temporal world modeling from this result alone.

## ICLR-Relevant Claim

The current defensible claim is:

> On ManiSkill PickCube, a trained action-sequence critic can detect a brittle grasp-height failure and recover all rank0 failures in a held-out candidate-selection protocol.

This is stronger than a training-free reranker, but still not enough for a final paper. The next required experiment is to add visual-progress and action-world failure gating on rendered rollouts, then move from privileged diagnostic candidates to a policy-generated candidate source.

## Next Required Ablations

1. Render rollout videos and test static visual progress.
2. Train a video/action selector that excludes action length as a shortcut.
3. Candidate-id and candidate-order randomization.
4. Larger `PickCube-v1` seed sweep.
5. A second ManiSkill task where rank0 has genuine success headroom.
6. Replace privileged diagnostic candidates with BC/diffusion-policy top-k candidates.
