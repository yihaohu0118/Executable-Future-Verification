# RoboTwin2 ICLR Window Run

Date: 2026-06-13

## Run Root

```text
/home/yihao_hyh/efv_runs/robotwin2_iclr_window_20260613_0802
```

## Launch Status

The RoboTwin2 six-task primary window was started on dev2 after the GPUs became
free.

Launcher:

```bash
cd /home/yihao_hyh/Executable-Future-Verification
EXECUTE=1 GPU_ID=auto SEEDS=0-7 RUN_ANALYSIS_AFTER=0 \
  scripts/robotwin2_iclr_window_launcher.sh \
  /home/yihao_hyh/efv_runs/robotwin2_iclr_window_20260613_0802
```

Observed remote processes after launch:

| PID | Role |
| ---: | --- |
| 2268025 | `robotwin2_iclr_window_launcher.sh` |
| 2268026 | current `robotwin2_run_clean_traces.sh` task wrapper |
| 2268139 | current `robotwin2_gripper_aware_trace` Python process |
| 2292529 | posthoc CPU-only analysis watcher |

The launcher selected GPU0 for the first task, `stack_blocks_two`.

## Logs

```text
/home/yihao_hyh/efv_runs/robotwin2_iclr_window_20260613_0802/logs/launcher.log
/home/yihao_hyh/efv_runs/robotwin2_iclr_window_20260613_0802/logs/stack_blocks_two_targeted_energy_matched_seeds_0-7.log
/home/yihao_hyh/efv_runs/robotwin2_iclr_window_20260613_0802/logs/posthoc_analysis_after_launcher.log
```

## Analysis Watcher

A CPU-only watcher is running. It waits for launcher PID `2268025` to finish,
then runs:

```bash
PYTHONPATH=src scripts/robotwin2_multitask_analysis.sh \
  /home/yihao_hyh/efv_runs/robotwin2_iclr_window_20260613_0802 \
  stack_blocks_two stamp_seal open_microwave place_object_basket stack_bowls_two press_stapler
```

The watcher does not use GPU and does not stop any process.

## Monitoring Commands

```bash
ssh -F /Users/huyihao/Desktop/evo_gym-main/ssh_config dev2 \
  'pgrep -af "robotwin2_iclr_window_20260613_0802|robotwin2_gripper_aware_trace" || true'

ssh -F /Users/huyihao/Desktop/evo_gym-main/ssh_config dev2 \
  'tail -80 /home/yihao_hyh/efv_runs/robotwin2_iclr_window_20260613_0802/logs/launcher.log'
```

## Current Interpretation

This run is the first attempt to close the RoboTwin2 paper-readiness gap:

- target: six primary tasks;
- seed range: `0-7`;
- candidate preset: `targeted_energy_matched`;
- required outcome: at least four base-ready tasks, at least one relation-ready
  task, and at least one relation-rescue mechanism.

Do not count this run as evidence until the posthoc readiness gates are
generated.
