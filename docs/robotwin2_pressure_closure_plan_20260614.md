# RoboTwin2 Pressure Closure Plan

- fresh run root: `/home/yihao_hyh/efv_runs/robotwin2_pressure_closure_20260614`
- execute: `false`
- gpu id: `auto`
- auto gpu ids: `2 3 4 5 6 7`
- task config: `demo_clean_k5`
- required candidates per case: `24`
- rank-randomization sweep seeds: `10`

## Why This Plan Exists

This plan is designed to close the expert-template objection, not to maximize raw success. Promote RoboTwin2 only if the generated gates show at least two DTW-breaking pressured tasks.

## Commands

| Phase | Task | Seeds | Diagnostic | Keep rule |
| --- | --- | --- | --- | --- |
| `resume_relation_partial` | `handover_block` | `0` | false | keep if relation gate passes after completing all 24 candidates |
| `resume_relation_partial` | `place_object_basket` | `1` | false | keep if relation gate passes and simple heuristics do not solve it |
| `resume_relation_partial` | `press_stapler` | `1` | true | diagnostic only; use to show where EFV is not needed |
| `fresh_pressure` | `stack_blocks_two` | `2-7` | false | keep only if EFV-family selector continues to beat DTW/template baselines |
| `fresh_pressure` | `stack_bowls_two` | `0-5` | false | keep if rank0 fails, oracle succeeds, and DTW/template is not near oracle |
| `fresh_pressure` | `handover_block` | `2-5` | false | keep if relation/contact traces rescue cases beyond gripper-only |
| `fresh_pressure` | `place_object_basket` | `2-5` | false | keep if relation/contact traces beat smoothness and DTW baselines |
| `fresh_pressure` | `press_stapler` | `2-4` | true | diagnostic only; do not count if energy/smoothness solves it |

### resume_relation_partial: handover_block

Role: complete high-value partial with mixed success/failure and object-state rows

```bash
EXECUTE=0 GPU_ID=auto AUTO_GPU_IDS='2 3 4 5 6 7' WAIT_FOR_GPU=1 TASK_CONFIG=demo_clean_k5 CANDIDATE_PRESET=targeted_energy_matched TASKS=handover_block SEEDS=0 RUN_ANALYSIS_AFTER=0 RESUME_PARTIAL=1 REQUIRE_CANDIDATES_PER_CASE=24 NUM_SWEEP_SEEDS=10 scripts/robotwin2_bounded_window_launcher.sh /home/yihao_hyh/efv_runs/robotwin2_iclr_clean_20260613_0905
```

### resume_relation_partial: place_object_basket

Role: complete high-value partial with mixed success/failure and object-state rows

```bash
EXECUTE=0 GPU_ID=auto AUTO_GPU_IDS='2 3 4 5 6 7' WAIT_FOR_GPU=1 TASK_CONFIG=demo_clean_k5 CANDIDATE_PRESET=targeted_energy_matched TASKS=place_object_basket SEEDS=1 RUN_ANALYSIS_AFTER=0 RESUME_PARTIAL=1 REQUIRE_CANDIDATES_PER_CASE=24 NUM_SWEEP_SEEDS=10 scripts/robotwin2_bounded_window_launcher.sh /home/yihao_hyh/efv_runs/robotwin2_iclr_clean_20260613_0905
```

### resume_relation_partial: press_stapler

Role: complete permissive partial as a negative control

```bash
EXECUTE=0 GPU_ID=auto AUTO_GPU_IDS='2 3 4 5 6 7' WAIT_FOR_GPU=1 TASK_CONFIG=demo_clean_k5 CANDIDATE_PRESET=targeted_energy_matched TASKS=press_stapler SEEDS=1 RUN_ANALYSIS_AFTER=0 RESUME_PARTIAL=1 REQUIRE_CANDIDATES_PER_CASE=24 NUM_SWEEP_SEEDS=10 scripts/robotwin2_bounded_window_launcher.sh /home/yihao_hyh/efv_runs/robotwin2_iclr_clean_20260613_0905
```

### fresh_pressure: stack_blocks_two

Role: known clean mechanism target; extend beyond the 2 complete seeds

```bash
EXECUTE=0 GPU_ID=auto AUTO_GPU_IDS='2 3 4 5 6 7' WAIT_FOR_GPU=1 TASK_CONFIG=demo_clean_k5 CANDIDATE_PRESET=targeted_energy_matched TASKS=stack_blocks_two SEEDS=2-7 RUN_ANALYSIS_AFTER=0 RESUME_PARTIAL=0 REQUIRE_CANDIDATES_PER_CASE=24 NUM_SWEEP_SEEDS=10 scripts/robotwin2_bounded_window_launcher.sh /home/yihao_hyh/efv_runs/robotwin2_pressure_closure_20260614
```

### fresh_pressure: stack_bowls_two

Role: new multistage contact task for a second DTW-breaking result

```bash
EXECUTE=0 GPU_ID=auto AUTO_GPU_IDS='2 3 4 5 6 7' WAIT_FOR_GPU=1 TASK_CONFIG=demo_clean_k5 CANDIDATE_PRESET=targeted_energy_matched TASKS=stack_bowls_two SEEDS=0-5 RUN_ANALYSIS_AFTER=0 RESUME_PARTIAL=0 REQUIRE_CANDIDATES_PER_CASE=24 NUM_SWEEP_SEEDS=10 scripts/robotwin2_bounded_window_launcher.sh /home/yihao_hyh/efv_runs/robotwin2_pressure_closure_20260614
```

### fresh_pressure: handover_block

Role: relation-ready bimanual transfer candidate

```bash
EXECUTE=0 GPU_ID=auto AUTO_GPU_IDS='2 3 4 5 6 7' WAIT_FOR_GPU=1 TASK_CONFIG=demo_clean_k5 CANDIDATE_PRESET=targeted_energy_matched TASKS=handover_block SEEDS=2-5 RUN_ANALYSIS_AFTER=0 RESUME_PARTIAL=0 REQUIRE_CANDIDATES_PER_CASE=24 NUM_SWEEP_SEEDS=10 scripts/robotwin2_bounded_window_launcher.sh /home/yihao_hyh/efv_runs/robotwin2_pressure_closure_20260614
```

### fresh_pressure: place_object_basket

Role: spatial constraint candidate with object-state coverage

```bash
EXECUTE=0 GPU_ID=auto AUTO_GPU_IDS='2 3 4 5 6 7' WAIT_FOR_GPU=1 TASK_CONFIG=demo_clean_k5 CANDIDATE_PRESET=targeted_energy_matched TASKS=place_object_basket SEEDS=2-5 RUN_ANALYSIS_AFTER=0 RESUME_PARTIAL=0 REQUIRE_CANDIDATES_PER_CASE=24 NUM_SWEEP_SEEDS=10 scripts/robotwin2_bounded_window_launcher.sh /home/yihao_hyh/efv_runs/robotwin2_pressure_closure_20260614
```

### fresh_pressure: press_stapler

Role: permissive negative-control task

```bash
EXECUTE=0 GPU_ID=auto AUTO_GPU_IDS='2 3 4 5 6 7' WAIT_FOR_GPU=1 TASK_CONFIG=demo_clean_k5 CANDIDATE_PRESET=targeted_energy_matched TASKS=press_stapler SEEDS=2-4 RUN_ANALYSIS_AFTER=0 RESUME_PARTIAL=0 REQUIRE_CANDIDATES_PER_CASE=24 NUM_SWEEP_SEEDS=10 scripts/robotwin2_bounded_window_launcher.sh /home/yihao_hyh/efv_runs/robotwin2_pressure_closure_20260614
```

## Finalize Commands

### Fresh pressure run

```bash
PYTHONPATH=src PYTHON_BIN=python3 REQUIRE_CANDIDATES_PER_CASE=24 NUM_SWEEP_SEEDS=10 scripts/robotwin2_finalize_run.sh /home/yihao_hyh/efv_runs/robotwin2_pressure_closure_20260614 stack_blocks_two stack_bowls_two handover_block place_object_basket press_stapler
```

### Resume root 1

```bash
PYTHONPATH=src PYTHON_BIN=python3 REQUIRE_CANDIDATES_PER_CASE=24 NUM_SWEEP_SEEDS=10 scripts/robotwin2_finalize_run.sh /home/yihao_hyh/efv_runs/robotwin2_iclr_clean_20260613_0905 handover_block place_object_basket press_stapler
```

## Readiness Requirements

1. at least four base-ready RoboTwin2 tasks with rank0 below oracle
2. at least two pressured tasks where EFV-family selectors beat the best DTW/template baseline
3. at least three strong-envelope tasks after candidate-ID/rank remap and simple heuristic controls
4. at least one relation/contact rescue task with object-relation trace coverage
5. no headline task where energy, smoothness, length, or DTW reaches oracle

## Kill Rules

- downgrade RoboTwin2 to diagnostic if DTW/template stays within one success of EFV on all new pressured tasks
- do not count tasks with fewer than two complete cases in the main table
- do not count relation selectors unless the relation gate reports nonzero coverage for every claimed case
- do not use GPU0/1 unless the owner explicitly releases active training jobs
