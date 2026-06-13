# RoboTwin2 ICLR Clean Run 20260613 0905

## Purpose

This run is the next evidence-gathering window for the Executable-Future
Verification project. The goal is to turn RoboTwin2 from a smoke diagnostic
into a second benchmark layer by collecting enough clean, shortcut-controlled
tasks to test whether execution-envelope selectors generalize beyond
RoboCasa365.

The run does not modify or stop existing training processes. It only starts
new RoboTwin2 trace jobs on GPUs that pass the repository guard:

- no active compute PID on the selected GPU;
- `memory.used <= 1024` MB;
- the same condition still holds after a 30 second stability recheck.

## Remote Location

- host: `dev2`
- repository: `/home/yihao_hyh/Executable-Future-Verification`
- repo head at launch: `067ce7e`
- run root: `/home/yihao_hyh/efv_runs/robotwin2_iclr_clean_20260613_0905`
- driver PID at launch: `2434584`

## Task Matrix

| Task | GPU | Seeds | Candidate preset | Role |
| --- | ---: | --- | --- | --- |
| `stack_blocks_two` | 0 | `0-7` | `targeted_energy_matched` | multi-stage stacking / order sensitivity |
| `stamp_seal` | 1 | `0-7` | `targeted_energy_matched` | contact timing and gripper/phase sensitivity |
| `open_microwave` | 2 | `0-7` | `targeted_energy_matched` | articulated-object opening |
| `place_object_basket` | 3 | `0-7` | `targeted_energy_matched` | pick-place relation and placement tolerance |
| `stack_bowls_two` | 4 | `0-7` | `targeted_energy_matched` | stacking with contact/pose tolerance |
| `press_stapler` | 5 | `0-7` | `targeted_energy_matched` | button/press contact control |

## Early Task Failures And Replacements

Early monitoring found that `place_object_basket` and `press_stapler` failed
inside the official expert/demo initialization path, before producing usable
candidate traces:

```text
AssertionError: target_pose cannot be None for move action.
```

This is treated as a task-generation failure, not as evidence against the EFV
selector. To avoid wasting the GPU window, two diagnostic replacements were
started under the same run root:

| Replacement task | GPU | Seeds | Candidate preset | Reason |
| --- | ---: | --- | --- | --- |
| `open_laptop` | 3 | `0-7` | `targeted_energy_matched` | previously showed clean headroom but permissive smoothness/gripper baselines |
| `handover_block` | 5 | `0-7` | `targeted_energy_matched` | bimanual handoff/contact diagnostic; useful if primary relation tasks remain weak |

The original driver posthoc command only includes the primary-window task list.
After replacement traces finish, rerun multitask analysis manually over all
tasks that produced usable raw JSONL files.

## Bad-Seed Continuation Fix

The first clean window exposed a sampler robustness issue: one failed seed could
abort the entire task before later seeds were attempted. This is especially bad
for RoboTwin2 because some official expert initializations fail for individual
seeds even when the task is otherwise usable.

The trace generator was updated after the run started:

- commit: `958a7ba`
- Python flag: `--continue-on-seed-error`
- shell default: `CONTINUE_ON_SEED_ERROR=1`

New trace jobs skip only the failed seed and keep collecting later seeds. This
does not convert failed expert initialization into a negative example; it only
prevents one bad seed from discarding the rest of the task.

After syncing `958a7ba` to dev2, supplementary jobs were launched for tasks
that failed or stalled on seed 0:

| Supplement task | GPU | Seeds | Reason |
| --- | ---: | --- | --- |
| `open_microwave` | 2 | `1-7` | seed 0 expert initialization failed |
| `stack_bowls_two` | 4 | `1-7` | seed 0 expert rollout did not succeed |
| `place_object_basket` | 6 | `1-7` | seed 0 expert initialization failed |
| `press_stapler` | 7 | `1-7` | seed 0 expert initialization failed |

## Resource Conflict Handling

At `2026-06-13T09:26Z`, GPUs 0-3 acquired large
`ray::WorkerDict.actor_rollout_update_actor` processes, each using about 72-76
GB. These processes were treated as user training/rollout work and were not
modified.

To avoid interfering with the user's training jobs, only the EFV/RoboTwin2
processes sharing those GPUs were terminated:

| Stopped PID | Task | GPU | Reason |
| ---: | --- | ---: | --- |
| `2435959` | `stack_blocks_two` | 0 | shared GPU with user ray worker |
| `2436036` | `stamp_seal` | 1 | shared GPU with user ray worker |
| `2476425` | `open_microwave` | 2 | shared GPU with user ray worker |
| `2456348` | `open_laptop` | 3 | shared GPU with user ray worker |

The following EFV tasks continued because their GPUs had no user ray worker:

| Continuing task | GPU |
| --- | ---: |
| `handover_block` | 5 |
| `stack_bowls_two` | 4 |
| `place_object_basket` | 6 |
| `press_stapler` | 7 |

Partial traces from stopped tasks may still be useful for debugging, but they
should not be treated as complete ICLR evidence unless rerun later on free GPUs.

The interrupted partial JSONL files were moved, not deleted, to keep the final
`raw/` directory from mixing complete and interrupted candidate pools:

```text
interrupted_raw/stack_blocks_two_seed_0.jsonl.interrupted_092847
interrupted_raw/stamp_seal_seed_0.jsonl.interrupted_092847
interrupted_raw/open_laptop_seed_0.jsonl.interrupted_092847
interrupted_raw/open_microwave_seed_1.jsonl.interrupted_092847
```

When GPUs 0-3 became free again, the interrupted tasks were restarted with the
same guard and bad-seed continuation enabled:

| Restart task | GPU | Seeds |
| --- | ---: | --- |
| `stack_blocks_two` | 0 | `0-7` |
| `stamp_seal` | 1 | `0-7` |
| `open_microwave` | 2 | `0-7` |
| `open_laptop` | 3 | `0-7` |

At `2026-06-13T09:35Z`, all eight GPUs were again occupied by user
`ray::WorkerDict.ref_compute_ref_log_prob` / `actor_rollout_*` workers. To
avoid interfering with the user's training job, all EFV/RoboTwin2 Python
processes were terminated. No ray or training process was modified.

The final raw directory therefore contains partial candidate pools only. These
partials are useful for debugging the collection pipeline, but not for a main
benchmark table:

| Task seed file | Candidate rows | Success rows | Status |
| --- | ---: | ---: | --- |
| `handover_block/seed_0.jsonl` | 9 | 2 | partial |
| `open_laptop/seed_0.jsonl` | 1 | 0 | partial |
| `place_object_basket/seed_1.jsonl` | 8 | 2 | partial |
| `press_stapler/seed_1.jsonl` | 8 | 5 | partial |
| `stack_blocks_two/seed_0.jsonl` | 0 | 0 | interrupted |
| `stack_bowls_two/seed_1.jsonl` | 7 | 3 | partial |
| `stamp_seal/seed_0.jsonl` | 1 | 0 | partial |

CPU-only posthoc analysis with `REQUIRE_CANDIDATES_PER_CASE=24` dropped all
cases for `candidate_count_mismatch`; the paper readiness gate correctly failed
with zero usable cases. This run should therefore be cited only as an
operational/debugging window, not as RoboTwin2 evidence.

## Launch Command

The run was started as a background driver. Each task is launched through
`scripts/robotwin2_run_clean_traces.sh`, then the driver runs one multitask
analysis after all trace jobs finish:

```bash
RUN_ROOT=/home/yihao_hyh/efv_runs/robotwin2_iclr_clean_20260613_0905
TASKS=(stack_blocks_two stamp_seal open_microwave place_object_basket stack_bowls_two press_stapler)
GPUS=(0 1 2 3 4 5)
SEEDS=0-7

for each task/gpu:
  GPU_ID=$gpu WAIT_FOR_GPU=1 GPU_STABLE_SECONDS=30 GPU_FREE_MAX_MEMORY_MB=1024 \
    scripts/robotwin2_run_clean_traces.sh "$RUN_ROOT" "$task" "$SEEDS"

PYTHONPATH=src PYTHON_BIN=python3 REQUIRE_CANDIDATES_PER_CASE=24 NUM_SWEEP_SEEDS=10 \
  scripts/robotwin2_multitask_analysis.sh "$RUN_ROOT" "${TASKS[@]}"
```

## Initial Health Check

After the 30 second stability recheck, the run had one Python compute process
on each of GPUs 0-5 and no compute process on GPUs 6-7. Early logs showed the
six tasks entering SAPIEN/RoboTwin2 initialization. The SAPIEN Vulkan ICD
warning is present, as in earlier RoboTwin2 runs, but the processes had not
failed at the time of the initial health check.

## Paper Gate Criteria

This run should not be used as an ICLR main table unless the posthoc
`robotwin2_paper_readiness_gate` passes or the failure is explicitly used as a
negative diagnostic. The current target pass line remains:

- at least 4 base-ready RoboTwin2 tasks;
- at least 3 matched-negative tasks;
- at least 3 strong-envelope tasks;
- at least 1 relation-rescue task;
- evidence that rank0 and simple energy/magnitude heuristics are not enough;
- evidence that the selector is not merely candidate-ID or nearest-expert
  template matching.

## Monitoring

```bash
RUN=/home/yihao_hyh/efv_runs/robotwin2_iclr_clean_20260613_0905
pgrep -af "robotwin2_iclr_clean_20260613_0905|robotwin2_gripper_aware_trace"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
tail -80 "$RUN/logs/driver.log"
tail -80 "$RUN/logs/posthoc_multitask_analysis.log"
```
