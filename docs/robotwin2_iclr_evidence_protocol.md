# RoboTwin2 ICLR Evidence Protocol

Date: 2026-06-13

## Purpose

This protocol defines the next RoboTwin2 evidence window for the
Executable-Future Verification project. The goal is not to maximize a single
selector number. The goal is to decide whether the ICLR story is defensible:

> Generated futures often contain successful candidates, but default ranking
> and simple action heuristics fail. Compact execution envelopes recover
> executable futures only when the candidate pool blocks rank, ID, energy,
> smoothness, and expert-template shortcuts.

## Non-Negotiable Boundaries

- Do not use simulator errors, CUDA OOM rows, or incomplete candidate files as
  physical failures.
- Do not use `--skip-replay-planner` for headline RoboTwin2 results.
- Do not claim object/contact-relation gains unless the relation gate passes.
- Do not claim general executability if successful candidates are mostly full
  expert traces.
- Do not stop unrelated user training jobs to get these results.

## Main-Table Entry Criteria

Each task must pass the per-task main-table gate before it can appear in a
paper table:

```bash
python -m umm_reward_evaluator.benchmarks.robotwin2_main_table_gate \
  --manifest RUN_ROOT/manifests/TASK_targeted_energy_matched_manifest.jsonl \
  --required-candidates-per-case 24 \
  --min-cases 1 \
  --min-oracle-better-cases 1
```

Object-relation claims additionally require:

```bash
python -m umm_reward_evaluator.benchmarks.robotwin2_main_table_gate \
  --manifest RUN_ROOT/manifests/TASK_targeted_energy_matched_manifest.jsonl \
  --required-candidates-per-case 24 \
  --min-cases 1 \
  --min-oracle-better-cases 1 \
  --require-feature object_relation_distribution \
  --require-feature phase_object_relation_joint_gripper_distribution
```

The multitask analysis script runs both gates automatically.

## Paper-Level Gate

After multitask analysis, run:

```bash
python -m umm_reward_evaluator.benchmarks.robotwin2_paper_readiness_gate \
  --run-root RUN_ROOT \
  --output-json RUN_ROOT/selectors/robotwin2_paper_readiness_gate.json \
  --output-md RUN_ROOT/selectors/robotwin2_paper_readiness_gate.md
```

Default thresholds are intentionally strict:

| Requirement | Default |
| --- | ---: |
| Base-ready tasks with oracle headroom | 4 |
| Relation-ready tasks | 1 |
| Tasks with non-template successful candidates | 2 |
| Tasks with matched/energy-controlled negative candidates | 3 |
| Tasks with DTW-diverse non-full-expert successes | 2 |
| Tasks with low-DTW failed negatives near the expert trace | 2 |
| Tasks where an envelope selector beats the strongest simple baseline | 3 |
| Tasks where relation features rescue gripper/DTW-gripper failure | 1 |

Passing this gate means the RoboTwin2 result set is strong enough to support
the main ICLR evidence package. The DTW-diverse success and low-DTW negative
requirements are specifically there to block the "nearest expert template"
failure mode. DTW nearest-positive selectors, including joint+gripper DTW, are
treated as template-matching baselines rather than as mechanism evidence. Failing
the gate means the result can still be a diagnostic, but the paper should not
rely on it as the second benchmark.

## Task Roles

The next window should prioritize tasks by role, not by convenience.

| Role | Candidate task | Desired phenomenon |
| --- | --- | --- |
| Contact-timing positive control | `stamp_seal` | Gripper/contact envelope beats energy and smoothness controls. |
| Multi-stage relation task | `stack_blocks_two` | Gripper-only fails; object/contact relation recovers. |
| Permissiveness counterexample | `open_laptop` | Smoothness or gripper may work because the benchmark is forgiving. |
| Extra hard-contact task | `press_stapler` or similar | Headroom survives beyond one button-like task. |
| Extra spatial task | a pick/place or stack variant | Relation features matter outside one stack task. |

At least one task should intentionally be a counterexample. A credible paper is
stronger if it explains where compact gripper envelopes fail.

## Candidate Pool Requirements

Each clean task run should include:

- rank0/default future;
- full expert or gripper-aware positive candidate;
- hard positives that are not full expert templates, such as time-warped or
  route-diverse successful traces;
- matched negatives with similar energy, length, or gripper timing but failed
  contact direction or object relation;
- energy-matched negatives so action magnitude cannot explain the result;
- candidate/rank randomization and candidate-ID remap analysis.
- learned ridge-probe baselines over action-only and execution-envelope
  features.

The current `targeted_energy_matched` preset already emits the source labels
needed by the paper-level gate:

- `time_warp_hard_positive_probe`;
- `matched_gripper_timing_negative_probe`;
- `matched_contact_direction_negative_probe`;
- `energy_matched_*_negative_probe`.

If a task produces no successful non-template candidates, it should be reported
as a limitation or diagnostic, not as proof that the verifier generalizes.
Successful candidates with unknown source labels are also not counted as
non-template evidence until they are explicitly labeled or confirmed by the DTW
anti-template diagnostic.

The default rank-randomization sweep includes a small pure-numpy learned
verifier baseline: `linear_probe:*:ridge_l2_1`. Action-only linear probes are
treated as strong baselines. Gripper/relation linear probes are reported as
envelope-family verifiers, but they still must beat DTW nearest-positive
template baselines to support the anti-template claim.

## Decision Rule

Continue toward a full ICLR submission if RoboTwin2 satisfies the paper-level
gate and RoboCasa365 remains unchanged under the same shortcut controls.

Downgrade to a diagnostic/workshop paper if:

- fewer than four RoboTwin2 tasks have clean oracle headroom;
- relation features do not rescue any gripper-only failure;
- successful candidates remain mostly full expert traces;
- DTW nearest-positive stays within one success of the best verifier;
- many apparent failures come from system errors or incomplete traces.

## Suggested Next Command, When Ready To Run

Use the bounded-window launcher first. It defaults to dry-run mode and prints
the four-task command matrix without executing simulations:

```bash
cd /home/yihao_hyh/Executable-Future-Verification
GPU_ID=auto SEEDS=0-7 scripts/robotwin2_bounded_window_launcher.sh \
  /home/yihao_hyh/efv_runs/robotwin2_bounded_window_YYYYMMDD
```

Set `EXECUTE=1` only when the GPU is free. With `GPU_ID=auto`, the launcher
starts only if a GPU is already idle, uses at most
`GPU_FREE_MAX_MEMORY_MB=1024` MB, and remains free after a short stability
check; otherwise it exits without waiting, killing, or preempting processes.
If training jobs are actively restarting or reclaiming GPUs, do not start the
window even if a GPU appears briefly idle.
With a fixed `GPU_ID`, the lower-level run script still waits for the same
memory-and-process free condition before starting. Low GPU utilization is not
enough; a card with tens of GB allocated is not considered free.

The lower-level run script also protects training jobs after launch. With the
default `GPU_CONFLICT_MONITOR=1`, it periodically checks the selected GPU and
terminates only its own RoboTwin2 child if another compute app appears. The
exit code is `75`, matching the "not safe to run now" path.

Raw seed files are atomically published. A seed is first written to a hidden
temporary file and is moved into `raw/<task>/seed_<n>.jsonl` only after all
candidates for that seed are complete. This keeps interrupted windows from
producing partial JSONL files that later fail the manifest gate with
`candidate_count_mismatch`.

Before converting raw traces, audit the raw directory:

```bash
python -m umm_reward_evaluator.benchmarks.robotwin2_raw_integrity_report \
  --raw-root RUN_ROOT/raw \
  --required-candidates-per-case 24 \
  --output-json RUN_ROOT/selectors/robotwin2_raw_integrity_report.json \
  --output-md RUN_ROOT/selectors/robotwin2_raw_integrity_report.md
```

The bounded launcher runs this audit automatically before multitask analysis
when `RUN_ANALYSIS_AFTER=1`. If the audit fails, do not convert the raw traces
into a paper-table manifest.

For a completed run root, use the finalize script instead of calling analysis
pieces by hand:

```bash
PYTHONPATH=src PYTHON_BIN=python3 REQUIRE_CANDIDATES_PER_CASE=24 NUM_SWEEP_SEEDS=10 \
  scripts/robotwin2_finalize_run.sh RUN_ROOT \
  stack_blocks_two stamp_seal place_object_basket stack_bowls_two
```

This script performs the raw integrity audit, multitask selector analysis,
paper-readiness gate, and a conservative registry-entry proposal. It does not
edit `docs/iclr_evidence_stack_registry.json`; inspect
`RUN_ROOT/selectors/robotwin2_registry_entry_proposal.json` first.

Use the full six-task `robotwin2_iclr_window_launcher.sh` only after the
bounded window confirms complete candidate pools and at least three base-ready
tasks.
