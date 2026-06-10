# RoboTwin2 Anti-Template K=5 Results

Date: 2026-06-10 UTC

This run uses the `--candidate-preset anti_template` RoboTwin2 trace adapter on
three tasks:

- `stack_blocks_two`, seeds `0 1 2 3 4`;
- `open_laptop`, seeds `0 1 2 3 4`;
- `stamp_seal`, seeds `0 1 3 4 6`.

The run was executed on dev2 with `CUDA_VISIBLE_DEVICES=5`. GPU5 was the only
idle GPU at the time; earlier attempts on GPU7 collided with an unrelated Ray
training job and caused cuRobo OOM / camera-buffer failures.

## Remote Artifacts

- Raw traces: `/tmp/robotwin2_antitemplate_k5/raw/`
- Per-task manifests: `/tmp/robotwin2_antitemplate_k5/manifests/*_antitemplate_k5_manifest.jsonl`
- Combined manifest: `/tmp/robotwin2_antitemplate_k5/manifests/robotwin2_three_task_antitemplate_k5_manifest.jsonl`
- Diagnostics: `/tmp/robotwin2_antitemplate_k5/selectors/*_antitemplate_diagnostics.{json,md}`
- Selector sweeps: `/tmp/robotwin2_antitemplate_k5/selectors/*_rankrand_sweep.json`
- K-shot sweep: `/tmp/robotwin2_antitemplate_k5/selectors/three_task_kshot_sweep.json`

## Anti-Template Diagnostics

| Split | Cases | Rank0 | Oracle | Diverse non-full success | Matched low-DTW negative |
| --- | ---: | ---: | ---: | ---: | ---: |
| stack_blocks_two | 5 | 0/5 | 5/5 | 5/5 | 0/5 |
| open_laptop | 5 | 0/5 | 5/5 | 5/5 | 2/5 |
| stamp_seal | 5 | 0/5 | 5/5 | 4/5 | 4/5 |
| combined | 15 | 0/15 | 15/15 | 14/15 | 6/15 |

Compared with the earlier pool, this is a real improvement: diverse
non-full-expert successes rise from 0/15 to 14/15. The hard-negative side is
still incomplete: matched low-DTW failures appear in 6/15 cases, concentrated
in `open_laptop` and `stamp_seal`.

## Candidate-Source Success

| Candidate source | Success |
| --- | ---: |
| full_expert_trace | 15/15 |
| time_warp_hard_positive_probe | 24/30 |
| matched_contact_direction_negative_probe | 13/15 |
| matched_gripper_timing_negative_probe | 19/30 |
| suffix_truncation | 10/15 |
| prefix_truncation | 0/15 |
| first_action | 0/15 |
| time_reverse | 0/15 |
| noop | 0/15 |

By task:

| Task | suffix_truncation | time_warp | gripper timing | contact perturb |
| --- | ---: | ---: | ---: | ---: |
| open_laptop | 5/5 | 10/10 | 10/10 | 5/5 |
| stack_blocks_two | 5/5 | 5/10 | 5/10 | 5/5 |
| stamp_seal | 0/5 | 9/10 | 4/10 | 3/5 |

The main mechanism signal is task dependence. `open_laptop` accepts almost all
anti-template perturbations, `stack_blocks_two` is sensitive to timing, and
`stamp_seal` is sensitive to truncation and gripper/contact perturbations.

## Anonymous Rank/ID Selector Sweep

Mean success over 10 anonymous candidate-ID/rank remap seeds:

| Selector | Success |
| --- | ---: |
| rank0 | 0.0/15 |
| candidate-ID full-trace lookup | 0.0/15 |
| random expected | 7.36/15 |
| energy_sum_max heuristic | 10.0/15 |
| length_max heuristic | 10.0/15 |
| smoothness_max heuristic | 9.0/15 |
| action distribution nearest-positive | 7.0/15 |
| gripper distribution nearest-positive, all-task | 12.0 +/- 0.77 / 15 |
| gripper distribution nearest-positive, same-task | 13.3 +/- 0.64 / 15 |
| phase-gripper nearest-positive, same-task | 13.4 +/- 0.66 / 15 |
| phase-joint nearest-positive, all-task | 11.0/15 |
| phase-joint+gripper nearest-positive, all-task | 11.0/15 |
| DTW action nearest-positive | 13.0/15 |
| DTW gripper nearest-positive | 13.5 +/- 0.50 / 15 |
| DTW joint nearest-positive | 13.0/15 |
| DTW joint+gripper nearest-positive | 13.0/15 |

Interpretation:

- The anti-template pool removes candidate-ID and rank shortcuts.
- Simple action heuristics are no longer enough, but they are not trivial:
  9-10/15.
- DTW is no longer oracle, but it remains a strong baseline at 13-13.5/15.
- Phase/gripper prototype selectors are competitive with DTW but do not yet
  clearly beat it.

## K-Shot Calibration

Mean success over 10 rank seeds and 5 support seeds:

| Feature | K=0 | K=1 | K=2 | K=4 |
| --- | ---: | ---: | ---: | ---: |
| gripper distribution | 2.3/15 | 5.32/15 | 8.54/15 | 12.0/15 |
| phase-gripper distribution | 4.0/15 | 4.8/15 | 6.9/15 | 10.2/15 |
| phase-joint distribution | 1.0/15 | 3.2/15 | 6.4/15 | 11.0/15 |
| phase-joint+gripper distribution | 2.0/15 | 4.2/15 | 7.4/15 | 11.0/15 |

The few-shot story survives the harder pool: source-only transfer is weak, and
target-task support substantially improves selection. The claim should remain
"few-shot task/contact calibration", not universal zero-shot verification.

## Current Interpretation

This result is stronger than the previous RoboTwin2 smoke because it creates
real diverse successes and some matched failures. It is still not a finished
main-table result because DTW and gripper-prototype baselines remain very
strong. The next experiment should create harder matched negatives where
joint/gripper DTW and gripper timing are close to successful trajectories but
contact or task completion fails.
