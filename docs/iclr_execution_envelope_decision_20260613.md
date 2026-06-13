# ICLR Direction Decision: Executable-Future Verification

Date: 2026-06-13

## Decision

This direction is worth continuing, but only inside a bounded evidence window.
The project should not keep expanding benchmarks until the RoboTwin2 mechanism
is cleaner.

The paper should be framed as:

> Future generation is increasingly cheap, but future selection is brittle.
> Under shortcut-controlled candidate pools, compact task/contact-conditioned
> execution envelopes can recover executable futures that default rankers,
> action heuristics, candidate IDs, and visual plausibility proxies miss.

The paper should not claim:

> The method learns general physical executability or replaces robot execution.

The current best claim is narrower and more defensible:

> In executable manipulation benchmarks, recoverable futures often exist, but
> default ordering and simple shortcuts fail. Few-shot execution-envelope
> verification is a useful diagnostic and selector, provided the candidate pool
> breaks expert-template shortcuts.

## Why The External Advice Is Mostly Correct

The advice correctly identifies the main value:

- the problem is future selection, not another future generator;
- the method boundary should be generator-agnostic candidate verification;
- RoboCasa365 is already the strongest mechanism evidence;
- RoboTwin2 is the right second executable benchmark layer;
- real-robot absence is acceptable only if the paper stays in the
  world-action / simulated executable-future framing;
- the strongest reviewer risk is that the selector is learning construction
  artifacts or expert-trace similarity rather than executability.

The advice also correctly warns against uncontrolled benchmark expansion.
Adding many weak benchmark probes would not solve the core critique. The next
evidence must show that the signal survives hard positives and matched hard
negatives on multiple RoboTwin2 tasks.

## Current Evidence

### RoboCasa365

RoboCasa365 currently supports the main mechanism:

- rank0 conservative prior: 0/64;
- oracle-best: 64/64;
- action-only endpoint-free selector: 28.4/64;
- object-only trace selector: 31.0/64;
- EEF/gripper distribution-only selector: 63.6/64;
- source-only no-task-ID transfer: 25.6/64;
- four-shot target calibration: 59.2/64;
- eight-shot target calibration: 62.2/64.

This supports a few-shot task/contact-conditioned execution-envelope story. It
also weakens object-leakage and action-magnitude explanations.

### RoboTwin2

RoboTwin2 is still mixed.

Strong signal:

- `stamp_seal` targeted-energy-matched K=5 controls rank, candidate ID, action
  energy, length, smoothness, and action-distribution shortcuts.
- rank0 is 0/5 and oracle is 5/5.
- energy, length, smoothness, and action-distribution baselines fail or remain
  weak.
- gripper execution-envelope features recover 5/5.

Main risk:

- `stamp_seal` is one task.
- gripper timing may itself be a task-specific construction shortcut.
- DTW and gripper-template baselines remain strong on some pools.

Most useful counter-signal:

- interim `stack_blocks_two` cases have rank0 0/2 and oracle 2/2, but
  gripper-only and DTW-gripper selectors collapse to 0/2.
- This is not bad for the paper. It suggests a stronger mechanism:
  gripper timing can be sufficient on contact-button tasks but fails on
  multi-stage spatial stacking, where contact direction and object relation
  state matter.

## Integrity Rule Added On 2026-06-13

System-error candidates must not be interpreted as physical failures.

The RoboTwin2 converter now supports:

```bash
--drop-cases-with-candidate-error
```

The multitask analysis script enables this flag by default. Any case containing
`candidate_error`, including CUDA OOM rows, is dropped from main analysis even
if it has the expected number of candidates. This prevents OOM or simulator
exceptions from becoming false hard negatives.

## Bounded Next Window

The next focused window should produce one of two outcomes.

### Continue Toward ICLR If

At least four RoboTwin2 tasks satisfy:

- rank0 is clearly below oracle;
- oracle headroom is not caused by system-error rows;
- candidate-ID and rank remap controls remain at chance or fail;
- energy, length, smoothness, and action-distribution heuristics do not explain
  the gain;
- hard positives exist that are not just the full expert trace;
- matched hard negatives exist that are close under action or joint/gripper DTW
  but fail official task success;
- the proposed verifier beats random, rank0, simple heuristics, and at least
  one strong template baseline.

At least one multi-stage task should show that gripper-only verification fails
and a richer contact/object-relation execution envelope is required.

### Downgrade If

The direction should be downgraded to a RoboCasa365 diagnostic or workshop paper
if:

- RoboTwin2 cannot produce three to four tasks with stable oracle headroom;
- successful candidates remain mostly full-expert or DTW-nearest expert traces;
- DTW nearest-positive stays within one or two successes of the best verifier;
- gains disappear after candidate source labels, rank, and full-expert traces
  are removed;
- many apparent hard negatives are actually simulator errors, OOMs, or
  incomplete candidate files.

## Immediate Engineering Plan

1. Re-run clean RoboTwin2 analysis with `--drop-cases-with-candidate-error`.
2. Treat existing incomplete or OOM-contaminated stack seeds as diagnostics
   only.
3. When GPU capacity is free, generate clean `stack_blocks_two` K=5 or K=8
   targeted-energy-matched data with object-relation traces enabled.
4. Add selectors that explicitly test the `stack_blocks_two` failure mode:
   gripper-only versus joint+gripper versus object-pair/contact-relation
   features.
5. Keep `open_laptop` as a permissiveness counterexample unless a harder
   candidate pool makes random and smoothness baselines weak.
6. Do not stop user training jobs to obtain these results.

## Paper Story After This Update

The strongest current story is not "gripper timing solves executable future
verification." That is too narrow and too easy to attack.

The stronger story is:

> Different tasks expose different shortcut failures. On some contact tasks,
> gripper execution envelopes are enough once action-energy shortcuts are
> removed. On multi-stage stacking, the same gripper-only verifier fails,
> revealing that executable-future verification must be task-phase and
> contact/object-relation conditioned. The contribution is the controlled
> evaluation framework plus the evidence that compact execution envelopes,
> calibrated with few target examples, recover futures that default generation
> rankings miss.

