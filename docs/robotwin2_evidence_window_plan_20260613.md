# RoboTwin2 Evidence Window Plan

- run root: `/home/yihao_hyh/efv_runs/robotwin2_evidence_window_20260613_k4`
- execute: `false`
- gpu id: `auto`
- task config: `demo_clean_k5`
- candidate preset: `targeted_energy_matched`
- required candidates per case: `24`
- rank-randomization sweep seeds: `10`

## Task Window

| Task | Role | Planned cases | Seeds | Keep rule |
| --- | --- | ---: | --- | --- |
| `handover_block` | main anti-template/contact task | 4 | `0-3` | keep |
| `place_object_basket` | spatial constraint task | 4 | `0-3` | keep |
| `stack_bowls_two` | multistage gripper/contact task | 4 | `0-3` | keep if oracle headroom appears |
| `stack_blocks_two` | multistage endpoint-vs-trace stress task | 4 | `0-3` | keep if gripper-aware trace works |
| `press_stapler` | permissive counterexample | 3 | `0-2` | diagnostic only; keep as negative control |

## Trace Commands

### handover_block

```bash
EXECUTE=0 GPU_ID=auto TASK_CONFIG=demo_clean_k5 CANDIDATE_PRESET=targeted_energy_matched TASKS=handover_block SEEDS=0-3 RUN_ANALYSIS_AFTER=0 RESUME_PARTIAL=0 REQUIRE_CANDIDATES_PER_CASE=24 NUM_SWEEP_SEEDS=10 scripts/robotwin2_bounded_window_launcher.sh /home/yihao_hyh/efv_runs/robotwin2_evidence_window_20260613_k4
```

### place_object_basket

```bash
EXECUTE=0 GPU_ID=auto TASK_CONFIG=demo_clean_k5 CANDIDATE_PRESET=targeted_energy_matched TASKS=place_object_basket SEEDS=0-3 RUN_ANALYSIS_AFTER=0 RESUME_PARTIAL=0 REQUIRE_CANDIDATES_PER_CASE=24 NUM_SWEEP_SEEDS=10 scripts/robotwin2_bounded_window_launcher.sh /home/yihao_hyh/efv_runs/robotwin2_evidence_window_20260613_k4
```

### stack_bowls_two

```bash
EXECUTE=0 GPU_ID=auto TASK_CONFIG=demo_clean_k5 CANDIDATE_PRESET=targeted_energy_matched TASKS=stack_bowls_two SEEDS=0-3 RUN_ANALYSIS_AFTER=0 RESUME_PARTIAL=0 REQUIRE_CANDIDATES_PER_CASE=24 NUM_SWEEP_SEEDS=10 scripts/robotwin2_bounded_window_launcher.sh /home/yihao_hyh/efv_runs/robotwin2_evidence_window_20260613_k4
```

### stack_blocks_two

```bash
EXECUTE=0 GPU_ID=auto TASK_CONFIG=demo_clean_k5 CANDIDATE_PRESET=targeted_energy_matched TASKS=stack_blocks_two SEEDS=0-3 RUN_ANALYSIS_AFTER=0 RESUME_PARTIAL=0 REQUIRE_CANDIDATES_PER_CASE=24 NUM_SWEEP_SEEDS=10 scripts/robotwin2_bounded_window_launcher.sh /home/yihao_hyh/efv_runs/robotwin2_evidence_window_20260613_k4
```

### press_stapler

```bash
EXECUTE=0 GPU_ID=auto TASK_CONFIG=demo_clean_k5 CANDIDATE_PRESET=targeted_energy_matched TASKS=press_stapler SEEDS=0-2 RUN_ANALYSIS_AFTER=0 RESUME_PARTIAL=0 REQUIRE_CANDIDATES_PER_CASE=24 NUM_SWEEP_SEEDS=10 scripts/robotwin2_bounded_window_launcher.sh /home/yihao_hyh/efv_runs/robotwin2_evidence_window_20260613_k4
```

## Finalize Command

```bash
PYTHONPATH=src PYTHON_BIN=python3 REQUIRE_CANDIDATES_PER_CASE=24 NUM_SWEEP_SEEDS=10 scripts/robotwin2_finalize_run.sh /home/yihao_hyh/efv_runs/robotwin2_evidence_window_20260613_k4 handover_block place_object_basket stack_bowls_two stack_blocks_two press_stapler
```

## Candidate Pool Requirements

- planner/rank0 failure when possible
- full gripper-aware expert candidate
- hard positives that are not full-template copies
- energy-matched negatives
- contact-direction or gripper-timing negatives
- low-DTW negatives near the expert trace
- reverse/shuffle/block-swap/action-axis controls
- candidate-ID and rank remapping in selector sweeps

## Readiness Gates

1. at least four base-ready tasks have rank0 failure and oracle success
2. at least three tasks have matched negative cases
3. at least two tasks have diverse non-template successes
4. at least two tasks have matched low-DTW failures near the expert trace
5. at least three tasks show a supported envelope selector beating the strongest simple/template baseline
6. at least one task shows relation/contact-aware rescue over gripper-only or template-distance selectors
7. no main-table method column is unsupported by held-out calibration data

## Stop Rule

Downgrade to diagnostic/workshop scope if 4 tasks x 4 cases keep showing oracle headroom but no supported selector margin over DTW/action/heuristic baselines.
