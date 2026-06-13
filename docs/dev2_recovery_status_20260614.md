# dev2 Recovery Status, 2026-06-14

This is an operational status note for the RoboTwin2 evidence window. It is not
paper evidence by itself.

## Current Finding

dev2 has usable H100 capacity for EFV but cannot safely launch RoboTwin2 because
the root filesystem is full.

Read-only preflight at `2026-06-14`:

| Resource | Status |
| --- | --- |
| `/`, `/home`, `/tmp` | `24T` size, `23T` used, `0` available, `100%` |
| GPU0 | active training process, about `74090 / 81559` MiB |
| GPU1 | active training process, about `74090 / 81559` MiB |
| GPU2-7 | idle, about `1 / 81559` MiB each |
| EFV remote repo | `/home/yihao_hyh/Executable-Future-Verification` at `10a8efb` |
| latest pushed EFV repo | GitHub `main` at `7ddb899` |
| EFV run directory size | about `34M`, not a cleanup target |

The current blocker is disk, not GPU. Do not start persistent RoboTwin2 waiters
until free space is restored and the remote EFV checkout is updated.

Follow-up read-only check after GitHub `b5b9e8e`:

- `/`, `/home`, and `/tmp` are still `100%` full with `0` available;
- GPU0/1 still host active `evogym-train` Python processes at about `74080`
  MiB each;
- GPU2-7 are still idle at about `1` MiB each;
- `/home/yihao_hyh/Executable-Future-Verification` is still at `10a8efb`;
- no deletion or process kill was performed.

Use `scripts/dev2_checkpoint_cleanup_audit.sh` on dev2 for a read-only cleanup
report. The script prints disk status, active GPU compute apps, large checkpoint
directories, and an approval template. It intentionally contains no `rm`.

## Large Disk Consumers

Read-only `du` shows the main pressure is under
`/home/yihao_hyh/checkpoints`, not EFV traces or caches.

Top checkpoint candidates observed:

| Path | Size |
| --- | ---: |
| `/home/yihao_hyh/checkpoints/evogym/qwen3_8b_session_modebalanced_strict_repro120_rawbase_20260611_dev2` | `184G` |
| `/home/yihao_hyh/checkpoints/evogym/qwen3_8b_tau_bench_intent_dense10_20260613_dev2` | `184G` |
| `/home/yihao_hyh/checkpoints/evogym/qwen3_8b_bfcl_partial_gold64_smoke_small_20260612_dev2` | `107G` |
| `/home/yihao_hyh/checkpoints/evogym/qwen3_8b_tau_bench_bridge_dense5_20260613_dev2` | `107G` |
| `/home/yihao_hyh/checkpoints/evogym/qwen3_8b_tau_bench_trajectory_grpo_20260612` | `107G` |
| `/home/yihao_hyh/checkpoints/evogym/qwen3_8b_tau_bench_disclosure_smoke_20260613_dev2` | `92G` |
| `/home/yihao_hyh/checkpoints/evogym/qwen3_8b_tau_bench_readonly_smoke_20260613_dev2` | `92G` |
| `/home/yihao_hyh/checkpoints/evogym/qwen3_8b_tau_bench_reward_smoke_20260613_dev2` | `92G` |
| `/home/yihao_hyh/checkpoints/evogym/qwen3_8b_tau_bench_term_smoke_20260613_dev2` | `92G` |
| `/home/yihao_hyh/checkpoints/evogym/qwen3_8b_session_officialstrict_modebalanced_rawbase_8gpu_ep10_filtered_20260611_dev2` | `77G` |
| `/home/yihao_hyh/checkpoints/evogym/qwen3_8b_tau_bench_partial_smoke_20260614_dev2` | `77G` |
| `/home/yihao_hyh/checkpoints/evogym/qwen3_8b_api_bank_action_reminder_smoke_20260614_dev2` | `34G` |
| `/home/yihao_hyh/checkpoints/evogym/qwen3_8b_api_bank_action_reminder_smoke2_20260614_dev2` | `3.8G` |
| `/home/yihao_hyh/checkpoints/evogym/qwen3_8b_api_bank_partial_smoke_20260614_dev2` | `2.6G` |

Do not delete these without explicit user approval. The active training
processes on GPU0/1 should also be left untouched.

## Safe Recovery Sequence

After space is freed:

```bash
cd /home/yihao_hyh/Executable-Future-Verification
git pull --ff-only
bash -n scripts/robotwin2_run_clean_traces.sh scripts/robotwin2_persistent_bounded_window_launcher.sh
```

Then use the new default GPU pool. `GPU_ID=auto` now searches only
`AUTO_GPU_IDS="2 3 4 5 6 7"` unless explicitly overridden.

Example persistent task window:

```bash
EXECUTE=1 GPU_ID=auto AUTO_GPU_IDS="2 3 4 5 6 7" \
  WAIT_FOR_GPU=1 GPU_WAIT_SECONDS=300 GPU_STABLE_SECONDS=120 \
  TASK_CONFIG=demo_clean_k5 CANDIDATE_PRESET=targeted_energy_matched \
  TASKS=handover_block SEEDS=0-3 RESUME_PARTIAL=1 \
  scripts/robotwin2_persistent_bounded_window_launcher.sh \
  /home/yihao_hyh/efv_runs/robotwin2_evidence_window_20260613_k4
```

The trace runner checks `MIN_FREE_DISK_MB` before and after GPU waiting. The
default is `2048` MB; insufficient disk exits with code `76` and should not be
treated as a retryable GPU-busy condition.

## Research Implication

This does not change the scientific status. RoboTwin2 remains the highest-value
next benchmark because it is the missing second executable layer for the ICLR
story. The immediate next evidence target is still:

1. at least four base-ready RoboTwin2 tasks with rank0 failure and oracle
   headroom;
2. hard positives that are not exact expert traces;
3. matched low-DTW negatives;
4. a supported contact/envelope selector that beats action heuristics and DTW
   template baselines on at least some nontrivial tasks.
