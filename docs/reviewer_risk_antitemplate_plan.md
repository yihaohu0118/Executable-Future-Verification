# Reviewer Risk and Anti-Template Experiment Plan

## Current Reviewer Risk

The project has a promising signal, but the current RoboTwin2 evidence is not
yet enough for a strong paper claim.

The main risk is expert-template matching:

> The selector may be choosing the candidate that is closest to known successful
> expert traces, rather than learning physical executability.

This risk is no longer hypothetical. On the current three-task RoboTwin2 K=5
manifest, anonymous candidate-ID/rank remap gives:

| Selector | Success |
| --- | ---: |
| Rank0 | 0.0/15 |
| Candidate-ID full-trace lookup | 0.0/15 |
| Uniform random expected | 4.17/15 |
| Best action heuristic, smoothness max | 5.3 +/- 0.46 / 15 |
| Phase-joint+gripper prototype, all-task | 12.0 +/- 0.00 / 15 |
| DTW joint+gripper nearest-positive, all-task | 14.0 +/- 0.00 / 15 |

The DTW row is a strong baseline and a warning. The current RoboTwin2 pool can
mostly be solved by nearest-positive trajectory similarity. This table should
be reported as a diagnostic control, not as the final evidence that the verifier
understands executability.

An additional anti-template diagnostic makes this stricter. The current
manifest has 10/15 nominal non-full-expert success cases, but all of them are
DTW-zero to the full expert executed trace under joint+gripper distance. Under
the diverse-success criterion, the current manifest has 0/15 hard positive
cases and 0/15 matched low-DTW negative cases.

## Revised Claim Boundary

Safe current claim:

> Compact robot execution traces expose a recoverable future-selection signal,
> but the current RoboTwin2 candidate pool still admits an expert-template
> shortcut.

Unsafe current claim:

> The verifier has learned physical executability beyond expert imitation.

Target claim after the next experiments:

> Under candidate pools with non-expert successful futures and matched
> trajectory-close failures, task/contact-conditioned execution-envelope
> verification selects executable futures better than rank0, action heuristics,
> candidate-ID controls, and nearest-expert DTW.

## Why This Matters for Novelty

Recent action-level test-time verifier work such as RoVer already frames robot
reward models as test-time verifiers for VLA candidate actions. To avoid a weak
"another robot verifier" story, this project must emphasize:

1. Future-level or trajectory-level verification, not single-action scoring.
2. Generator-agnostic candidate futures from policies, planners, world models,
   corrupted demonstrations, or retrieval.
3. Mechanism diagnostics showing when execution envelopes work and when they
   collapse to expert-template matching.
4. Anti-shortcut candidate pools that break candidate ID, fixed rank, action
   magnitude, object-state leakage, and nearest-expert distance.

## Required Anti-Template Experiments

### 1. Hard Positives

Goal: include successful futures that are not the full expert trace.

Candidate sources:

- time-warped expert actions that still succeed;
- successful truncated or cleaned-up variants such as `drop_last` when valid;
- cross-seed action traces replayed from another demonstration and accepted
  only if the simulator success check passes;
- small joint-space perturbations around successful traces, filtered by
  execution success;
- alternative gripper timing around the contact phase, filtered by execution
  success.

Acceptance criterion:

- at least 3 RoboTwin2 tasks have successful non-full-expert candidates;
- at least 50 percent of cases have more than one successful candidate source;
- at least 50 percent of cases have a successful non-full-expert candidate with
  nonzero executed-trace DTW from the full expert candidate;
- DTW-to-expert is no longer a near-oracle selector.

### 2. Matched Hard Negatives

Goal: create failures that look close to successful traces under action,
joint, and gripper DTW.

Candidate sources:

- preserve joint path but shift gripper close/open timing;
- preserve gripper sequence but perturb contact approach direction;
- swap or mirror a late contact segment while keeping action energy matched;
- replay a successful trace under a nearby but incompatible object or target
  configuration;
- execute low-DTW cross-seed traces that fail under the held-out initial state.

Acceptance criterion:

- hard negatives have action and joint+gripper DTW distances comparable to
  hard positives;
- DTW nearest-positive drops substantially;
- the proposed verifier recovers a meaningful gap over DTW.

### 3. Strong Baseline Table

Every future RoboTwin2 table should include:

- rank0;
- uniform random expected;
- action magnitude, energy, smoothness, and length heuristics;
- candidate-ID lookup under anonymous remap;
- distribution prototype features;
- DTW action, joint, gripper, and joint+gripper nearest-positive baselines;
- simple learned binary classifier;
- contrastive nearest-positive or metric-learning baseline;
- RoVer-style scalar process-reward baseline if VLA/action candidates are
  available.

### 4. Kill Criteria

Pause the full-paper push and downgrade to a diagnostic note if:

- DTW nearest-positive remains within 1-2 successes of the best verifier after
  anti-template pool construction;
- non-expert successful candidates cannot be generated reliably on at least 3
  tasks;
- most gains disappear when candidate source labels, rank, and full expert
  traces are removed;
- improvements appear only on RoboCasa and do not transfer to a second modern
  executable benchmark.

## Immediate Implementation Priority

The next engineering target is not another feature extractor. It is a new
RoboTwin2 candidate-generation pass that explicitly records:

- candidate source;
- whether the successful candidate is full expert, perturbed expert,
  cross-seed replay, time-warped replay, or truncated cleanup;
- action DTW to nearest successful training trace;
- joint+gripper DTW to nearest successful training trace;
- oracle success under official task checks.

Only after this anti-template manifest exists should EEF/contact-direction
features be added and evaluated.

Implementation hook now available:

```bash
PYTHONPATH=/path/to/Executable-Future-Verification/src python -m umm_reward_evaluator.benchmarks.robotwin2_gripper_aware_trace \
  --task-name stack_blocks_two \
  --task-config demo_clean_smoke \
  --all-seeds \
  --max-seeds 5 \
  --output-dir /tmp/robotwin2_antitemplate/stack_blocks_two \
  --candidate-preset anti_template
```

The preset adds time-warp probes, gripper-timing shifts, and a contact-segment
joint perturbation. These are not assumed to be good candidates; they are
designed to produce an empirical pool from which real hard positives and
matched hard negatives can be filtered by official success labels and DTW.
