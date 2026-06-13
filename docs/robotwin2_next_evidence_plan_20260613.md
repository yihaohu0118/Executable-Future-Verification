# RoboTwin2 Next Evidence Plan

Date: 2026-06-13

## Current Judgment

The project should continue, but the next step should be evidence-gated rather
than open-ended. RoboCasa365 already supports the core EFV mechanism: rank0 can
miss all successful candidates, oracle selection shows large headroom, and
compact robot execution-envelope features recover most of that headroom under
shortcut-controlled negatives.

RoboTwin2 is not yet paper-ready. The latest completed priority-1 resume run
improved the raw trace pool but still only gives one completed case per task in
the filtered analysis root. That is useful for checking candidate construction
and anti-template pressure, but it is not enough to evaluate few-shot or
same-task calibrated selectors. In that setting, a same-task verifier has no
held-out training cases, so zero selector scores mean "unsupported", not
"method failed."

## Claim Boundary

Use this claim:

> Generated futures often contain executable successes, but default ordering and
> simple action heuristics fail to select them. Few-shot task/contact-calibrated
> execution envelopes can recover executable futures under shortcut-controlled
> candidate pools.

Avoid these claims until further evidence exists:

- universal zero-shot verifier;
- physical understanding from RGB/video alone;
- real-robot executability;
- world-model improvement rather than generator-agnostic future selection;
- RoboTwin2 main-table success from single-case smoke tests.

## What The Latest RoboTwin2 Evidence Shows

Useful signals:

- `handover_block`, `place_object_basket`, and `press_stapler` have complete
  24-candidate raw traces in the priority-1 filtered run.
- All three have rank0 failure and oracle success, so candidate-pool headroom is
  real.
- `handover_block` is the strongest anti-template case so far: it has non-full
  successes and hard matched failures.
- `place_object_basket` is useful as a spatial negative-heavy task.
- `press_stapler` is too permissive: many perturbations succeed, and simple
  heuristics can look artificially strong. Treat it as a diagnostic
  counterexample, not as a main mechanism task.

Current gaps:

- only three base-ready tasks in the filtered root, below the four-task gate;
- one case per task, so same-task calibrated selectors are unsupported;
- no robust method-vs-template margin on RoboTwin2 yet;
- matched negative evidence is still below the paper-readiness threshold;
- relation-rescue is not proven.

## Required Code/Report Behavior Before More Runs

Selector reports must distinguish three states:

- supported and successful;
- supported and failed;
- unsupported due to insufficient held-out calibration data.

Single-case same-task selectors should not be interpreted as verifier failures.
The selector table should display unsupported calibrated selectors as
`unsup(rate)`, and the paper-readiness gate should not count unsupported
selector values toward method margins.

## Next Experiment Window

Target a small, high-information RoboTwin2 window before expanding:

| Task | Role | Minimum cases | Keep? |
| --- | --- | ---: | --- |
| `handover_block` | main anti-template/contact task | 4 | yes |
| `place_object_basket` | spatial constraint task | 4 | yes |
| `stack_bowls_two` | multistage gripper/contact task | 4 | yes if oracle headroom appears |
| `stack_blocks_two` | multistage endpoint-vs-trace stress task | 4 | yes if gripper-aware trace works |
| `press_stapler` | permissive counterexample | 3 | diagnostic only |

Minimum candidate pool per case:

- one planner/rank0 failure when possible;
- one full gripper-aware expert candidate;
- at least two hard positives that are not full-template copies;
- energy-matched negatives;
- contact-direction or gripper-timing negatives;
- low-DTW negatives near the expert trace;
- reverse/shuffle/block-swap/action-axis controls;
- candidate-ID and rank remapping for selector sweeps.

## Readiness Gates

Do not promote RoboTwin2 to main ICLR evidence until all of these pass:

1. At least four base-ready tasks have rank0 failure and oracle success.
2. At least three tasks have matched negative cases.
3. At least two tasks have diverse non-template successes.
4. At least two tasks have matched low-DTW failures near the expert trace.
5. At least three tasks show a supported envelope selector beating the strongest
   simple/template baseline by the configured margin.
6. At least one task shows relation or contact-aware features rescuing a failure
   of gripper-only or template-distance selectors.
7. Selector support metadata shows no main-table method column is unsupported.

## Main Reviewer Risks To Attack

Expert-template matching:

- Add hard positives that succeed while deviating from the full expert trace.
- Add low-DTW failures that are close to expert but unsuccessful.
- Report DTW nearest-positive as a baseline, not as part of EFV.

Shortcut detection:

- Keep magnitude, energy, smoothness, length, action-only, and candidate-ID
  baselines.
- Candidate-ID remapping must remain enabled in the main selector sweeps.
- Use press_stapler as a negative control showing that permissive tasks can make
  shortcuts look good.

Object-state leakage:

- Keep robot-only, EEF/gripper, object-only, and relation feature ablations.
- Prefer claims around robot execution envelope, not privileged object-state
  access.

No real robot:

- Do not overclaim deployment.
- Position RoboTwin2 and RoboCasa365 as executable simulation evidence.
- Add a world-model diagnostic layer only after RoboTwin2 closes the core gate.

## Stop/Continue Rule

Continue toward an ICLR-style paper if the next RoboTwin2 window closes the
method-margin and anti-template gates above.

Downgrade to a diagnostic/workshop-style paper if RoboTwin2 keeps showing
oracle headroom but no supported selector margin over DTW/action/heuristic
baselines after 4 tasks x 4 cases.

Stop expanding benchmarks if new tasks mostly behave like `press_stapler`, where
many corrupted futures still succeed and simple shortcuts match oracle.
