# RoboTwin2 Current Gate Status

Date: 2026-06-13

## GPU / Run Status

At the latest dev2 check, all 8 H100 GPUs were occupied by user
`ray::WorkerDict` train/rollout/checkpoint workers. EFV/RoboTwin2 jobs were
stopped when they began sharing GPUs with those workers. No ray or training
process was stopped.

The attempted clean stack run directory
`/home/yihao_hyh/efv_runs/robotwin2_stack_clean_energy_matched_gpu0_20260613`
contains only `logs/driver.log`; it terminated while waiting for GPU0.

A later clean-window attempt was launched under:

```text
/home/yihao_hyh/efv_runs/robotwin2_iclr_clean_20260613_0905
```

It briefly collected partial traces across eight RoboTwin2 tasks, but user
Ray workers reclaimed all GPUs before any case reached the required full
candidate count. The run was stopped to protect training. See
`docs/robotwin2_iclr_clean_run_20260613_0905.md`.

## Latest CPU-Only Analysis

The latest CPU-only analysis was run on existing raw files under:

```text
/home/yihao_hyh/efv_runs/robotwin2_targeted_energy_matched_multitask_official_20260612
```

No simulation or GPU execution was used for this analysis.

## Readiness Summary

| Task | Cases | Rank0 | Oracle | Base gate | Relation gate | Relation coverage |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| `stack_blocks_two` | 2 | 0/2 | 2/2 | pass | fail | 0.00 |
| `open_laptop` | 2 | 0/2 | 2/2 | pass | fail | 0.00 |

Both tasks have clean oracle headroom after dropping incomplete or
candidate-error cases. Neither task can support object-relation claims because
the old traces do not contain `actor_pose_vector` or
`actor_pairwise_distances`.

## Selector Summary

| Task | Rank0 | Random | Energy | Smooth | Gripper | DTW gripper | Relation coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stack_blocks_two` | 0.0/2 | 0.9/2 | 0.0/2 | 0.0/2 | 0.0/2 | 0.0/2 | 0.00 |
| `open_laptop` | 0.0/2 | 1.6/2 | 0.0/2 | 2.0/2 | 2.0/2 | 2.0/2 | 0.00 |

Interpretation:

- `stack_blocks_two` is the useful mechanism task: simple selectors fail under
  shortcut-controlled negatives, but relation evidence is still missing.
- `open_laptop` is a permissive counterexample: smoothness, gripper, and
  DTW-gripper all succeed, so it should not carry the main method claim.

## Paper Readiness Gate

The latest `robotwin2_paper_readiness_gate.py` result fails:

| Check | Status |
| --- | --- |
| base-ready tasks | fail: 2 / min 4 |
| relation-ready tasks | fail: 0 / min 1 |
| non-template success tasks | pass: 2 / min 2 |
| matched-negative tasks | fail: 2 / min 3 |
| strong-envelope tasks | fail: 0 / min 3 |
| relation-rescue tasks | fail: 0 / min 1 |

This is the correct current conclusion: RoboTwin2 is useful evidence in
progress, but it is not ready as the second main ICLR benchmark.

## Next Safe Action

The latest clean-window run produced only partial candidate pools. CPU-only
posthoc analysis with `REQUIRE_CANDIDATES_PER_CASE=24` dropped all cases for
`candidate_count_mismatch`, so the paper gate remains fail with zero usable
cases from this run.

Do not start another RoboTwin2 GPU window while the current Ray training jobs
occupy the GPUs. The next safe action is either:

- wait until training finishes and rerun a bounded 4-task window first; or
- use CPU-only/documentation work to improve gates, summaries, and benchmark
  selection logic.

For the next rerun, prefer the bounded sequential launcher:

```bash
cd /home/yihao_hyh/Executable-Future-Verification
EXECUTE=1 RUN_ANALYSIS_AFTER=1 GPU_ID=auto SEEDS=0-7 \
  scripts/robotwin2_bounded_window_launcher.sh \
  /home/yihao_hyh/efv_runs/robotwin2_bounded_window_YYYYMMDD
```

The bounded launcher defaults to four tasks:
`stack_blocks_two stamp_seal place_object_basket stack_bowls_two`. It runs one
task at a time, uses the same GPU guard/conflict monitor as the lower-level
trace script, and performs multitask analysis only after the trace jobs finish.

With `GPU_ID=auto`, a task starts only if a GPU is already free; if all GPUs are
occupied, it exits without waiting, killing, or preempting processes. Do not
run with an explicit GPU while the current Ray training jobs occupy the same
GPUs.

Check whether it is safe to start with:

```bash
cd /home/yihao_hyh/Executable-Future-Verification
scripts/robotwin2_gpu_status.sh
```
