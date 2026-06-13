# dev2 Disk Blocker, 2026-06-14

This note records why the RoboTwin2 pressure-closure run could not continue on
dev2 after the first dry-run-validated launch.

## Status

- EFV local branch: `4098760 Wait for stable auto GPU selection`
- dev2 checkout: `ab40a9f Add world-model bridge gate`
- dev2 could not fast-forward to `4098760` because `git fetch` from a local
  bundle failed with `No space left on device`.
- `/home` returned to effectively zero free space after briefly showing about
  37GB available.
- The EFV run directory is not the cause: `/home/yihao_hyh/efv_runs` is about
  34MB.
- `/tmp/ray` is not the cause: about 41MB.

## What Happened

The pressure-closure launcher started:

```text
/home/yihao_hyh/efv_runs/robotwin2_pressure_closure_20260614_execute.sh
```

It used the intended protected GPU pool:

```text
AUTO_GPU_IDS=2 3 4 5 6 7
```

The first resume target, `handover_block seed 0`, was already complete:

```text
resume partial .../handover_block/seed_0.jsonl: reusable=24 missing=0
skip complete existing .../handover_block/seed_0.jsonl
```

The launcher then attempted `place_object_basket seed 1`, but stopped because
GPU2 became occupied during the stability check. That was the correct safety
behavior; it did not touch GPU0/1, where the active training processes were
running.

The launcher revealed a robustness issue in the old auto-GPU logic: a transient
GPU occupancy during the recheck caused the whole sequential script to exit.
This was fixed locally in `4098760` by making `GPU_ID=auto` retry while
`WAIT_FOR_GPU=1`.

## Current Disk Culprit

Recent large files are checkpoint files under:

```text
/home/yihao_hyh/checkpoints/evogym
```

Most relevant recent directories:

| Directory | Size | Note |
| --- | ---: | --- |
| `qwen3_8b_tau_action_reminder_smoke_retry_20260614_dev2` | 37G | recent files around 20:28 |
| `qwen3_8b_api_bank_action_reminder_smoke2_20260614_dev2` | 3.8G | recent files around 19:18 |
| `qwen3_8b_tau_bench_partial_smoke_20260614_dev2` | 77G | recent files around 18:40 |

Largest checkpoint directories observed:

| Directory | Size |
| --- | ---: |
| `qwen3_8b_session_modebalanced_strict_repro120_rawbase_20260611_dev2` | 184G |
| `qwen3_8b_tau_bench_intent_dense10_20260613_dev2` | 184G |
| `qwen3_8b_bfcl_partial_gold64_smoke_small_20260612_dev2` | 107G |
| `qwen3_8b_tau_bench_bridge_dense5_20260613_dev2` | 107G |
| `qwen3_8b_tau_bench_trajectory_grpo_20260612` | 107G |
| `qwen3_8b_tau_bench_disclosure_smoke_20260613_dev2` | 92G |
| `qwen3_8b_tau_bench_readonly_smoke_20260613_dev2` | 92G |
| `qwen3_8b_tau_bench_reward_smoke_20260613_dev2` | 92G |
| `qwen3_8b_tau_bench_term_smoke_20260613_dev2` | 92G |
| `qwen3_8b_session_officialstrict_modebalanced_rawbase_8gpu_ep10_filtered_20260611_dev2` | 77G |
| `qwen3_8b_tau_bench_partial_smoke_20260614_dev2` | 77G |
| `qwen3_8b_tau_action_reminder_smoke_retry_20260614_dev2` | 37G |

## Safe Next Step

Do not delete checkpoints without explicit approval.

Minimum unblock target:

- Free at least 20GB so the dev2 repo can fast-forward to `4098760` and the
  pressure-closure launcher can wait safely for GPU2-7.

Recommended approval candidate if these are disposable smoke checkpoints:

```bash
rm -rf /home/yihao_hyh/checkpoints/evogym/qwen3_8b_tau_action_reminder_smoke_retry_20260614_dev2
```

This should free about 37GB and is the smallest recent checkpoint directory that
appears large enough to unblock EFV. Larger deletions should be considered only
if the active training workflow no longer needs them.

