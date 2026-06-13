# ICLR Next Evidence Plan

Date: 2026-06-13

## Current Claim Boundary

The current defensible claim is:

> Under shortcut-controlled candidate pools, compact EEF/gripper/proprio
> execution envelopes can recover executable futures that rank0 and simple
> action heuristics miss.

This is supported on RoboCasa365, but it is not yet an ICLR-strength
multi-benchmark claim. The project should not claim general physical
executability, world-model understanding, or real-robot transfer until the
missing evidence below is closed.

## What Is Already Strong

- RoboCasa365 has clean oracle headroom: rank0 fails, oracle succeeds, and the
  envelope selector is close to oracle.
- Shortcut controls are already meaningful: energy, magnitude, smoothness,
  action-only, and object-only signals do not explain the main result.
- Few-shot calibration is a better story than zero-shot universality. The
  method should be framed as task/contact-conditioned verification.

## Main Weakness To Fix

The largest reviewer risk is:

> The verifier may be selecting the trajectory closest to the expert template,
> not selecting a genuinely executable future.

The next experiments must therefore test non-template successes and
near-template failures before expanding the benchmark list. More samples of the
same candidate construction will not resolve this risk.

Current RoboTwin2 artifacts already expose the sharper version of this risk.
See `docs/robotwin2_current_evidence_audit_20260613.md`: the official two-task
run has anti-template pressure, but the EFV-family selector does not yet beat
DTW/template or smoothness shortcuts. This makes the next milestone a selector
mechanism improvement under pressure, not just a larger rerun.

## Required Evidence Before A Strong Submission

### 1. RoboTwin2 Second Benchmark

RoboTwin2 must become the second executable benchmark. A usable result requires:

- at least four tasks with complete raw candidate pools;
- clean oracle headroom in those tasks;
- EEF/gripper/proprio selectors beating rank0, random, magnitude, energy, and
  smoothness baselines;
- candidate ID and rank remap controls;
- at least one multi-stage task where endpoint-only or gripper-only fails but a
  temporal/contact-aware trace feature helps.

Do not count partial runs, CUDA/system failures, or incomplete JSONL files as
physical failures. The raw integrity audit must pass before manifest
conversion.

### 2. Anti-Template-Matching Controls

Each serious RoboTwin2 task should include:

- hard positives: successful futures that are not full expert traces;
- matched hard negatives: failures with similar EEF path, action energy, or
  gripper timing;
- DTW-near failed negatives: trajectories close to the expert trace but
  unsuccessful;
- route-diverse or time-warped successes when the task permits them.

The key table should show whether the verifier can prefer diverse successful
futures over near-template failures. DTW nearest-positive selectors are treated
as template baselines; they do not count as evidence that the proposed execution
envelope mechanism has beaten the expert-template shortcut.

The dedicated RoboTwin2 anti-template pressure report should be inspected before
any headline table is trusted:

```bash
python -m umm_reward_evaluator.benchmarks.robotwin2_antitemplate_pressure_gate \
  --run-root RUN_ROOT \
  --output-json RUN_ROOT/selectors/robotwin2_antitemplate_pressure_gate.json \
  --output-md RUN_ROOT/selectors/robotwin2_antitemplate_pressure_gate.md
```

Passing this report means the method wins specifically on tasks where the
candidate pool contains both successful non-template futures and failed
low-DTW futures. Failing it is actionable: either the pool lacks anti-template
pressure, or DTW/expert-template matching still explains the result.

### 3. One World-Model Diagnostic Layer

The paper needs one diagnostic layer that connects the method to current
world-model benchmarks. This should not be the main result unless the official
benchmark path is stable.

Preferred order:

1. MiraBench, if public artifacts expose usable judgments or model-score
   proxies.
2. RoboTrustBench, as adapter validation unless oracle/headroom labels are
   strong enough.
3. RoboWM-Bench, only as conditional robustness evidence until the official GT
   replay/evaluator mismatch is resolved.

The diagnostic only counts if EFV beats a visual/model-score proxy under the
same candidate groups.

## Stop Or Downgrade Rules

Downgrade to a diagnostic or workshop-style paper if any of these remain true
after the next RoboTwin2 window:

- fewer than four RoboTwin2 tasks have clean oracle headroom;
- successful candidates are mostly full expert traces;
- DTW nearest-positive matches the best learned/envelope verifier;
- successful candidates have unknown source labels and cannot be separated from
  full-template variants;
- relation/contact features never rescue gripper-only or endpoint-only failure;
- many failures are simulator, reset, CUDA, or incomplete-run artifacts.

Continue toward ICLR only if RoboTwin2 closes the second-benchmark gap and at
least one diagnostic world-model layer gives a credible external check.

## Next Execution Order

1. Keep the current GPU state untouched while user training occupies dev2.
2. When a clean window exists, run the bounded RoboTwin2 launcher with
   `RUN_ANALYSIS_AFTER=1` so raw integrity audit runs before analysis.
3. Inspect the paper-readiness gate and anti-template pressure gate before
   looking at headline selector accuracy.
4. If RoboTwin2 passes, freeze RoboCasa365 and RoboTwin2 tables, then instantiate
   one world-model diagnostic manifest.
5. If RoboTwin2 fails, do not expand blindly. First inspect whether the failure
   is no headroom, no non-template positives, or verifier weakness under matched
   negatives.

## Paper Framing If Evidence Closes

The strongest framing remains:

> Future generation is increasingly cheap; future selection remains brittle.
> Compact, task-calibrated execution envelopes provide a surprisingly strong
> verifier for selecting executable robot futures under shortcut-controlled
> candidate pools.

Avoid claiming a new world model, a universal reward model, or real-robot
validity. The contribution should be a verifier framework plus a diagnostic
evidence package showing which signals survive shortcut controls.
