# RoboTwin2 Current Evidence Audit

Date: 2026-06-13

Audited source run root on dev2:

```text
/home/yihao_hyh/efv_runs/robotwin2_targeted_energy_matched_multitask_official_20260612
```

Latest CPU-only two-task reanalysis run root on dev2:

```text
/home/yihao_hyh/efv_runs/robotwin2_official_reanalysis_latest_20260613
```

Latest CPU-only existing-artifact mechanism reanalysis run root on dev2:

```text
/home/yihao_hyh/efv_runs/robotwin2_existing_mechanism_reanalysis_20260613
```

Latest partial raw rescue plan on dev2:

```text
/home/yihao_hyh/efv_runs/robotwin2_iclr_clean_20260613_0905/selectors/robotwin2_partial_raw_rescue_plan.md
```

This is a CPU-only audit of already-generated artifacts. No new simulation,
training, GPU job, or user process was started.

## Summary

The current RoboTwin2 evidence is useful, but it is not yet paper-ready.

The official multitask run has two base-ready tasks with oracle headroom, and
the existing stamp run adds one more base-ready task:

| Task | Cases | Rank0 | Oracle | Non-template success | Matched negative | DTW-diverse success | Low-DTW negative |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `open_laptop` | 2 | 0/2 | 2/2 | 2 | 2 | 2 | 2 |
| `stack_blocks_two` | 2 | 0/2 | 2/2 | 2 | 2 | 2 | 2 |
| `stamp_seal` | 5 | 0/5 | 5/5 | 5 | 5 | 5 | 5 |

This is enough to show anti-template pressure exists in the current candidate
pool. The latest selector reanalysis also reveals a useful mechanism signal:
`stack_blocks_two` is solved by a learned phase-gripper verifier even though
DTW-action, DTW-gripper, DTW-joint+gripper, smoothness, energy, action-only,
and gripper prototype selectors fail. The result is still not paper-ready
because there are only three base-ready tasks, no relation coverage, and
`open_laptop` and `stamp_seal` are fully explained by gripper/DTW or smoothness
shortcuts.

## Paper-Readiness Gate

`robotwin2_paper_readiness_gate.json` fails.

| Check | Status | Detail |
| --- | --- | --- |
| Base-ready tasks | fail | 3/4 tasks: `open_laptop`, `stack_blocks_two`, `stamp_seal` |
| Relation-ready tasks | fail | 0/1 |
| Non-template success tasks | pass | 3/2 |
| Matched negative tasks | pass | 3/3 |
| DTW-diverse success tasks | pass | 3/2 |
| Low-DTW negative tasks | pass | 3/2 |
| Strong envelope tasks | fail | 1/3: `stack_blocks_two` |
| Relation rescue tasks | fail | 0/1 |

## Anti-Template Pressure Gate

The newly generated pressure report also fails:

| Check | Status | Detail |
| --- | --- | --- |
| Anti-template pressure tasks | pass | `open_laptop`, `stack_blocks_two`, `stamp_seal` |
| Method beats template tasks | fail | 1/2: `stack_blocks_two` |
| No template-oracle risk | fail | `open_laptop`, `stamp_seal` |

Per-task risk:

| Task | Best EFV-family method | Best DTW template | Best simple baseline | Risk |
| --- | ---: | ---: | ---: | --- |
| `open_laptop` | gripper 2.0/2 | DTW-gripper 2.0/2 | smoothness 2.0/2 | DTW/template and smoothness explain the result |
| `stack_blocks_two` | linear phase-gripper 2.0/2 | best DTW 0.0/2 | random 0.9/2 | no current pressure-gate risk |
| `stamp_seal` | gripper 5.0/5 | DTW-gripper 5.0/5 | linear action 3.0/5 | DTW/template explains the result |

## Interpretation

This is a mixed but useful result. The candidate pool already contains the
right diagnostic pressure: successful non-template futures and failed low-DTW
futures. On `stack_blocks_two`, the result is now a genuine positive mechanism
signal: phase-aware gripper statistics beat DTW/template and simple action
shortcuts. On `open_laptop` and `stamp_seal`, the benchmark remains too
permissive because gripper/DTW or smoothness reach oracle.

The right next experiment is therefore not a bigger version of the old
gripper-only table. It should test whether the phase-gripper mechanism survives
more tasks and whether relation/contact traces can rescue tasks where gripper
alone is ambiguous:

- add relation coverage to the RoboTwin2 traces;
- explicitly score gripper timing relative to object/contact events;
- compare against DTW-joint/gripper/relation and smoothness as first-class
  baselines;
- keep `open_laptop` and `stamp_seal` as permissiveness counterexamples, not
  headline wins;
- make `stack_blocks_two` the main mechanism target because it breaks
  action-only, smoothness, and DTW shortcuts while rewarding phase-gripper
  timing.

## Current Decision

Do not count RoboTwin2 as the second benchmark yet.

The strongest current statement is:

> RoboTwin2 confirms that anti-template pressure can be constructed. On
> `stack_blocks_two`, a phase-aware gripper verifier recovers executable
> futures where DTW/template and smoothness shortcuts fail. On `open_laptop`,
> the same result is not meaningful because simple shortcuts already solve the
> task. `stamp_seal` adds more headroom but not mechanism evidence because
> DTW-gripper also reaches oracle.

The next paper-relevant milestone is at least four base-ready RoboTwin2 tasks,
with at least two pressured tasks where an EFV-family verifier beats the best
DTW template baseline by the anti-template pressure gate.

## Next Raw Completion Targets

The partial raw rescue plan for
`/home/yihao_hyh/efv_runs/robotwin2_iclr_clean_20260613_0905/raw` identifies
three high-value partial seeds:

| Task | Seed | Existing candidates | Missing candidates | Success | Failure | Object-state rows | Priority |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `handover_block` | 0 | 9 | 15 | 2 | 7 | 9 | high-value partial |
| `place_object_basket` | 1 | 8 | 16 | 2 | 6 | 8 | high-value partial |
| `press_stapler` | 1 | 8 | 16 | 5 | 3 | 8 | high-value partial |

These are the best next GPU targets because they already contain mixed
success/failure candidates and object-state traces. Completing them to 24
candidates per seed can test both the phase-gripper mechanism and the missing
relation/contact verifier claim. By contrast, the partial `open_laptop` and
`stamp_seal` seeds only contain one failed candidate each and should not be
prioritized.
