# ManiSkill PickCube FAVC Controls

## Purpose

This records the reviewer-facing controls for the PickCube brittle-grasp slice. The goal is to rule out easy shortcuts before building the next benchmark layer.

Manifest:

- `outputs/maniskill_pickcube_brittle_grasp_n20/PickCube-v1_candidate_manifest.jsonl`
- `outputs/maniskill_pickcube_brittle_grasp_n20_video/PickCube-v1_candidate_manifest.jsonl`
- 20 cases, 5 candidates per case
- Rank0 profile: `brittle_grasp`

## Results

| Experiment | Success | Recovered rank0 failures | Oracle match |
| --- | ---: | ---: | ---: |
| Rank0 brittle grasp | 4/20 | 0/16 | 0/20 |
| Raw action MLP | 20/20 | 16/16 | 13/20 |
| Raw action MLP without trajectory length | 20/20 | 16/16 | 13/20 |
| Raw action MLP on shuffled manifest rows | 20/20 | 16/16 | 13/20 |
| Raw video-frame MLP | 20/20 | 16/16 | 13/20 |
| Raw video-frame MLP on shuffled manifest rows | 20/20 | 16/16 | 13/20 |
| Action-video fusion MLP | 20/20 | 16/16 | 2/20 |
| Zero-action control | 4/20 | 0/16 | 0/20 |
| Zero-video control | 4/20 | 0/16 | 0/20 |
| Oracle-best candidate | 20/20 | 16/16 | 20/20 |

## Interpretation

The action and video critics recover all rank0 failures even after removing trajectory length and shuffling JSONL row order. This rules out two major shortcut explanations:

- the model is not simply using early termination / action count;
- the model is not relying on manifest row order.

The fusion critic also recovers all failures, but its oracle match is low. This is acceptable for this slice because multiple non-rank0 candidates succeed; binary success is the primary metric here.

## Remaining Risk

This still does not prove the full ICLR story. The current candidate pool is diagnostic and hand-designed. The next necessary control is candidate-family position randomization or a policy-generated candidate source.

Remote runner:

- `scripts/run_maniskill_favc_controls.sh`

