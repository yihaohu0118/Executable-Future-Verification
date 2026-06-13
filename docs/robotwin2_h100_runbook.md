# RoboTwin2 H100 Runbook

Date: 2026-06-12 UTC

This note records the current RoboTwin2 execution setup for executable-future
verification runs on the GCP dev boxes.

## Completed Run

The completed `stamp_seal` `targeted_energy_matched` K=5 run is on dev2:

- root: `/home/yihao_hyh/efv_runs/robotwin2_targeted_energy_matched_k5_official_20260612`
- raw traces: `raw/stamp_seal/seed_*.jsonl`
- logs: `logs/*.log`
- official RoboTwin checkout: `/home/yihao_hyh/robotwin_probe`
- EFV checkout: `/home/yihao_hyh/Executable-Future-Verification`
- conda env: `/home/yihao_hyh/miniconda3/envs/robotwin2-favc`

The run used the default adapter path, which builds the official replay planner
for every candidate. This is slow but is the only currently trusted path for
main-table results. It completed five valid seeds: `0, 3, 4, 6, 9`.

## H100 cuRobo Fallback

Official cuRobo v0.7.8 triggers CUDA illegal-instruction failures on the dev2
H100s in several custom/fused kernels. The working remote checkout applies a
local fallback patch under:

- `/home/yihao_hyh/robotwin_probe/envs/curobo/src/curobo/opt/newton/newton_base.py`
- `/home/yihao_hyh/robotwin_probe/envs/curobo/src/curobo/opt/newton/lbfgs.py`
- `/home/yihao_hyh/robotwin_probe/envs/curobo/src/curobo/util/torch_utils.py`

The patch disables CUDA graph capture, torch compile/JIT decorators, fused
line-search/update-best kernels, and the custom LBFGS CUDA kernel. Expert
recording and official replay then run through slower PyTorch fallback paths.

## Invalid Fast Replay Path

The adapter exposes `--skip-replay-planner` only as an experimental diagnostic.
It should not be used for headline results.

Seed-0 evidence:

- default official replay path: earlier smoke kept rank0 `0/1` and oracle
  `1/1`;
- `--skip-replay-planner` with a lightweight gripper planner made
  `full_gripper_aware` fail on `stamp_seal` seed 0.

Therefore the fast path changes the replay semantics. The main experiments must
use the default official planner path despite the runtime cost.

## Seed Handling

For `stamp_seal`, seeds `1, 2, 5, 7, 8` failed expert rollout in the official
K=5 run. Seed `9` was the successful replacement, and seed `6` completed on
GPU0 after seed `0`.

Some RoboTwin2 task configs have empty `data/<task>/<config>/seed.txt` files on
dev2. Use explicit seed lists for clean reruns instead of relying on
`--all-seeds`. Prefer the repository launcher so the run root, logs, explicit
seeds, GPU wait, and `--skip-existing` behavior are consistent:

```bash
cd /home/yihao_hyh/Executable-Future-Verification
DRY_RUN=1 GPU_ID=auto AUTO_GPU_IDS="2 3 4 5 6 7" \
  TASK_CONFIG=demo_clean_k5 CANDIDATE_PRESET=targeted_energy_matched \
  scripts/robotwin2_run_clean_traces.sh \
  /home/yihao_hyh/efv_runs/robotwin2_stack_clean_energy_matched_YYYYMMDD \
  stack_blocks_two \
  0-7
```

Remove `DRY_RUN=1` to execute. By default `GPU_ID=auto` searches only
`AUTO_GPU_IDS="2 3 4 5 6 7"`, leaving GPU0/1 available for active training.
Override `AUTO_GPU_IDS` only after checking ownership. The launcher waits for
the selected GPU to be empty by default but does not kill or stop any existing
process. "Empty" means both no compute process and
`memory.used <= GPU_FREE_MAX_MEMORY_MB` after a stability recheck. Do not treat
low utilization alone as safe; Ray or another training job can hold tens of GB
while utilization is temporarily near zero.
During execution, `GPU_CONFLICT_MONITOR=1` is enabled by default. If another
compute app appears on the same GPU after the trace job starts, the wrapper
terminates only its own RoboTwin2 child process and exits with code `75`.
It does not kill the foreign process.

For long evidence windows on shared machines, wrap the bounded launcher with the
persistent retry wrapper instead of hand-writing a tmux `while` loop:

```bash
cd /home/yihao_hyh/Executable-Future-Verification
tmux new-session -d -s efv_pwait_stack_blocks_gpu5 \
  "EXECUTE=1 GPU_ID=5 WAIT_FOR_GPU=1 GPU_WAIT_SECONDS=300 GPU_STABLE_SECONDS=120 \
   TASK_CONFIG=demo_clean_k5 CANDIDATE_PRESET=targeted_energy_matched \
   TASKS=stack_blocks_two SEEDS=0-3 RESUME_PARTIAL=1 \
   scripts/robotwin2_persistent_bounded_window_launcher.sh \
   /home/yihao_hyh/efv_runs/robotwin2_parallel_YYYYMMDD/stack_blocks_two \
   > /home/yihao_hyh/efv_runs/robotwin2_parallel_YYYYMMDD/stack_blocks_two/logs/pwait_gpu5.log 2>&1"
```

The persistent wrapper retries only exit code `75`, which is reserved for
"GPU busy" or "foreign GPU process appeared" conditions. Exit code `0` stops
the loop as success; any other nonzero code stops the loop as a real failure.
This keeps EFV from occupying a GPU while another user's training or Ray job is
active, while still allowing the evidence window to resume once the card is
actually free.

Check disk space before launching persistent waiters. The wrapper can wait for
GPUs, but it cannot recover from a full filesystem if logs or atomic raw temp
files cannot be written. `robotwin2_run_clean_traces.sh` now checks
`MIN_FREE_DISK_MB` before and after GPU waiting; the default is `2048` MB. If
the filesystem is below that threshold, it exits with code `76`, which the
persistent wrapper treats as a real failure rather than a retryable GPU-busy
condition. The first check happens before creating `raw/` or `logs/`; if the
run root does not exist yet, the script checks the nearest existing parent
directory. On dev2, recent `checkpoints/evogym/qwen3_8b_*` training checkpoints
can consume tens of GB within minutes; do not delete those without explicit
approval. Clear only reproducible caches or `/tmp` experiment scratch when
space is needed.

Seed files are published atomically. A seed writes to a hidden temporary file
first and replaces `raw/<task>/seed_<n>.jsonl` only after the full candidate
pool finishes. Interrupted runs therefore should not create official partial
JSONL files in `raw/`; hidden temp files can be inspected for debugging but
must not be converted into manifests for paper tables.
Before manifest conversion, run `robotwin2_raw_integrity_report.py` on
`RUN_ROOT/raw`. The bounded launcher runs this automatically when
`RUN_ANALYSIS_AFTER=1`; a failed audit means the raw directory is not paper-table
ready.
Do not use `--skip-replay-planner` for main-table data. Set
`RUN_ANALYSIS_AFTER=1` or run `scripts/robotwin2_finalize_run.sh` afterward.
The finalize script runs the raw audit, selector analysis, paper-readiness gate,
and registry-entry proposal. Require the generated relation gate to pass before
using object-relation selector numbers in a paper table.
