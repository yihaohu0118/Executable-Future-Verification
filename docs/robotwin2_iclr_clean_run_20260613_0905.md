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
