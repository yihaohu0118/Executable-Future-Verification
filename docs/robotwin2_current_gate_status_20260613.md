# RoboTwin2 Current Gate Status

Date: 2026-06-13

## GPU / Run Status

At the latest dev2 check, all 8 H100 GPUs were occupied by `ray::WorkerDict`
processes using roughly 53 GB each, with 84-86% utilization. No RoboTwin2
benchmark run was started, and no training process was stopped.

The attempted clean stack run directory
`/home/yihao_hyh/efv_runs/robotwin2_stack_clean_energy_matched_gpu0_20260613`
contains only `logs/driver.log`; it terminated while waiting for GPU0.

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

When GPUs are free, run the six-task primary window:

```bash
cd /home/yihao_hyh/Executable-Future-Verification
EXECUTE=1 GPU_ID=0 SEEDS=0-7 scripts/robotwin2_iclr_window_launcher.sh \
  /home/yihao_hyh/efv_runs/robotwin2_iclr_window_YYYYMMDD
```

The launcher waits for the selected GPU and does not kill existing processes.
Do not run it while the current Ray training jobs occupy all GPUs.
