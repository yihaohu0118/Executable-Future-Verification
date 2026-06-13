# RoboTwin2 Current Evidence Audit

Date: 2026-06-13

Audited run root on dev2:

```text
/home/yihao_hyh/efv_runs/robotwin2_targeted_energy_matched_multitask_official_20260612
```

This is a CPU-only audit of already-generated artifacts. No new simulation,
training, GPU job, or user process was started.

## Summary

The current RoboTwin2 evidence is useful, but it is not yet paper-ready.

The run has two base-ready tasks with oracle headroom:

| Task | Cases | Rank0 | Oracle | Non-template success | Matched negative | DTW-diverse success | Low-DTW negative |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `open_laptop` | 2 | 0/2 | 2/2 | 2 | 2 | 2 | 2 |
| `stack_blocks_two` | 2 | 0/2 | 2/2 | 2 | 2 | 2 | 2 |

This is enough to show anti-template pressure exists in the current candidate
pool. It is not enough to support the ICLR claim because the selector does not
beat strong shortcut baselines.

## Paper-Readiness Gate

`robotwin2_paper_readiness_gate.json` fails.

| Check | Status | Detail |
| --- | --- | --- |
| Base-ready tasks | fail | 2/4 tasks: `open_laptop`, `stack_blocks_two` |
| Relation-ready tasks | fail | 0/1 |
| Non-template success tasks | pass | 2/2 |
| Matched negative tasks | fail | 2/3 |
| DTW-diverse success tasks | pass | 2/2 |
| Low-DTW negative tasks | pass | 2/2 |
| Strong envelope tasks | fail | 0/3 |
| Relation rescue tasks | fail | 0/1 |

## Anti-Template Pressure Gate

The newly generated pressure report also fails:

| Check | Status | Detail |
| --- | --- | --- |
| Anti-template pressure tasks | pass | `open_laptop`, `stack_blocks_two` |
| Method beats template tasks | fail | 0/2 |
| No template-oracle risk | fail | `open_laptop` |

Per-task risk:

| Task | Best EFV-family method | Best DTW template | Best simple baseline | Risk |
| --- | ---: | ---: | ---: | --- |
| `open_laptop` | gripper 2.0/2 | DTW-gripper 2.0/2 | smoothness 2.0/2 | DTW/template and smoothness explain the result |
| `stack_blocks_two` | gripper 0.0/2 | DTW-gripper 0.0/2 | random 0.9/2 | current envelope features fail under pressure |

## Interpretation

This is a valuable negative result. The candidate pool already contains the
right diagnostic pressure: successful non-template futures and failed low-DTW
futures. The failure is not merely "we need more probes"; the current selector
family is not strong enough on RoboTwin2.

The right next experiment is therefore not a bigger version of the same table.
It should test a more contact/relation-aware verifier against the same pressure:

- add relation coverage to the RoboTwin2 traces;
- explicitly score gripper timing relative to object/contact events;
- compare against DTW-joint/gripper/relation and smoothness as first-class
  baselines;
- keep `open_laptop` as a permissiveness counterexample, not a headline win;
- make `stack_blocks_two` the main mechanism target because it currently breaks
  gripper-only and DTW-gripper shortcuts.

## Current Decision

Do not count RoboTwin2 as the second benchmark yet.

The strongest current statement is:

> RoboTwin2 confirms that anti-template pressure can be constructed, but the
> present EFV selector does not yet beat DTW/template or smoothness shortcuts.

The next paper-relevant milestone is at least four base-ready RoboTwin2 tasks,
with at least two pressured tasks where an EFV-family verifier beats the best
DTW template baseline by the anti-template pressure gate.
