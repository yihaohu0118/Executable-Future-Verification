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
DRY_RUN=1 GPU_ID=0 TASK_CONFIG=demo_clean_k5 CANDIDATE_PRESET=targeted_energy_matched \
  scripts/robotwin2_run_clean_traces.sh \
  /home/yihao_hyh/efv_runs/robotwin2_stack_clean_energy_matched_YYYYMMDD \
  stack_blocks_two \
  0-7
```

Remove `DRY_RUN=1` to execute. The launcher waits for the selected GPU to be
empty by default but does not kill or stop any existing process. "Empty" means
both no compute process and `memory.used <= GPU_FREE_MAX_MEMORY_MB` after a
stability recheck. Do not treat low utilization alone as safe; Ray or another
training job can hold tens of GB while utilization is temporarily near zero.
During execution, `GPU_CONFLICT_MONITOR=1` is enabled by default. If another
compute app appears on the same GPU after the trace job starts, the wrapper
terminates only its own RoboTwin2 child process and exits with code `75`.
It does not kill the foreign process.

Seed files are published atomically. A seed writes to a hidden temporary file
first and replaces `raw/<task>/seed_<n>.jsonl` only after the full candidate
pool finishes. Interrupted runs therefore should not create official partial
JSONL files in `raw/`; hidden temp files can be inspected for debugging but
must not be converted into manifests for paper tables.
Do not use `--skip-replay-planner` for main-table data. Set
`RUN_ANALYSIS_AFTER=1` or run `scripts/robotwin2_multitask_analysis.sh`
afterward, and require the generated relation gate to pass before using
object-relation selector numbers in a paper table.
