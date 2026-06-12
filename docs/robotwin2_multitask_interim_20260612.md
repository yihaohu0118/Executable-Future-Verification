# RoboTwin2 Multitask Targeted-Energy-Matched Interim

Date: 2026-06-12 UTC

This is an interim task-selection note, not a main result. It uses only the
currently complete 24-candidate official-planner seeds copied from:

`/home/yihao_hyh/efv_runs/robotwin2_targeted_energy_matched_multitask_official_20260612`

Interim analysis root:

`/home/yihao_hyh/efv_runs/robotwin2_targeted_energy_matched_multitask_interim_complete_20260612_1539`

The converter was run with `--require-candidates-per-case 24`, so partial or
terminated seeds are excluded.

## Completed Cases

| Task | Complete cases | Rank0 | Oracle | Diverse non-full success | Matched low-DTW negative |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stack_blocks_two` | 2 | 0/2 | 2/2 | 2/2 | 2/2 |
| `open_laptop` | 2 | 0/2 | 2/2 | 2/2 | 2/2 |
| `handover_block` | 0 | - | - | - | - |
| `press_stapler` | 0 | - | - | - | - |

## Selector Signal

Mean success over 10 anonymous rank/candidate-ID remaps:

| Task | Random expected | Energy max | Length max | Smoothness max | Gripper proto | DTW gripper | DTW joint+gripper |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stack_blocks_two` | 0.92/2 | 0.0/2 | 0.2/2 | 0.0/2 | 0.0/2 | 0.0/2 | 0.0/2 |
| `open_laptop` | 1.58/2 | 0.0/2 | 0.0/2 | 2.0/2 | 2.0/2 | 2.0/2 | 0.0/2 |

## Interpretation

`stack_blocks_two` is the more valuable next benchmark task. It has recoverable
headroom, diverse hard positives, and matched low-DTW negatives, but every
current simple selector collapses on the interim two-case table. This is a
stronger diagnostic than `stamp_seal`: the same gripper-envelope shortcut that
is oracle on `stamp_seal` is not sufficient here.

The failure rows show why. On `stack_blocks_two`, gripper-distribution and
DTW-gripper nearest-positive selectors choose energy-matched gripper/contact
negative probes and fail in 20/20 anonymized selections. Smoothness selects
time-warp candidates and also fails in 20/20. This suggests the next verifier
must include task-phase/contact direction or terminal task-state cues; pure
gripper timing is too brittle for stacking.

`open_laptop` is useful as a counterexample but weak as a headline task. It has
rank0 headroom, but the candidate pool is permissive: random expected is
1.58/2 and smoothness, gripper-distribution, and DTW-gripper all reach 2/2.
This task can show that some benchmarks are easy for shortcut selectors, but it
should not be used as the main evidence for a stronger verifier.

`handover_block` and `press_stapler` currently have no complete 24-candidate
cases in the interim copy. Their partial traces should remain diagnostics until
the full run or replacement seeds produce complete cases.

## Immediate Use

- Prioritize `stack_blocks_two` for method development and full K=5 analysis.
- Treat `open_laptop` as a stress-test counterexample for benchmark
  permissiveness.
- Do not include incomplete `handover_block` or `press_stapler` seeds in main
  tables.
- If `stack_blocks_two` remains hard at K=5, it becomes the strongest
  anti-template result because it breaks the current gripper-only mechanism.
