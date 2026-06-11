# Proposal: Executable-Future Verification for Robot World-Action Candidates

## Working Thesis

Future generation is becoming cheap, but future selection is still brittle.
Given multiple candidate robot futures for the same initial state and
instruction, the useful question is not which future looks most plausible, but
which future is physically executable and task-completing.

The current hypothesis is:

> Generated or proposed futures often contain a successful candidate, but
> default ranking and simple action heuristics fail to identify it. A compact
> task/contact-conditioned execution-envelope verifier over robot traces can
> recover executable futures, while shortcut controls expose when apparent
> gains come from action magnitude, candidate IDs, or fixed rank ordering.

## Current Method Boundary

This project is not currently proposing a new world model, a new robot policy,
or a real-robot deployment system.

It is a generator-agnostic candidate verifier:

- input: several candidate futures for the same task case;
- candidate representations: action sequences, robot state traces, gripper
  traces, optional videos or world-model outputs;
- output: a selected future expected to execute successfully;
- evaluation target: benchmark task success or executable-future label.

This framing keeps the contribution compatible with policy samples, demo
retrieval, world-model video-to-action rollouts, planner rollouts, or corrupted
expert futures.

## Core Counterintuitive Claims

1. More action detail is not always better.
   On hard controls, action-only selectors and action magnitude heuristics can
   fail while compact robot execution traces work.

2. Object state is not the main signal in the strongest RoboCasa result.
   Robot-only, proprio-only, and EEF/gripper traces recover most successes,
   while object-only traces are much weaker.

3. Averaging successful futures can erase contact signal.
   On RoboTwin2, nearest-positive prototype memory works much better than
   positive or positive-negative centroids.

4. Gripper timing can beat richer action statistics.
   A tiny nearest-positive selector over low-dimensional gripper trace
   distributions beats action distribution features and simple action
   heuristics on the current RoboTwin2 K=5 smoke.

5. Fixed candidate naming and rank order are dangerous.
   Candidate-ID and rank remap controls are required before any selector result
   is safe enough for a main table.

## Evidence So Far

### RoboCasa365

RoboCasa365 is the primary 2026 benchmark layer. The current strongest result
is the regenerated random-position n16 hard-negative pool:

| Setting | Success |
| --- | ---: |
| Rank0 conservative replay prior | 0/64 |
| Oracle-best | 64/64 |
| Action-only endpoint-free selector, mean over 5 seeds | 28.4/64 |
| Object-only trace selector | 31.0/64 |
| Proprio-only selector | 63.0/64 |
| Robot-only selector | 64.0/64 |
| EEF+gripper distribution-only selector | 63.6/64 |
| Same-task nearest-positive EEF+gripper prototype | 59/64 |
| Source-only no-task-ID transfer | 25.6/64 |
| Full-source plus one-shot target calibration | 46.0/64 |
| Full-source plus four-shot target calibration | 59.2/64 |
| Full-source plus eight-shot target calibration | 62.2/64 |

Interpretation: the useful signal is a few-shot task/contact-conditioned robot
execution envelope, not a universal zero-shot verifier, object-state leakage,
or an action magnitude shortcut.

### RoboTwin2

RoboTwin2 is the second executable benchmark layer because it is a 2025
dual-arm manipulation benchmark with official task success checks.

Current three-task K=5 mechanism table:

| Setting | Success |
| --- | ---: |
| Rank0 | 0/15 |
| Oracle-best | 15/15 |
| Full gripper-aware trace | 15/15 |
| First action | 0/15 |
| First half | 0/15 |
| Reverse | 0/15 |
| Noop | 0/15 |
| Drop last | 10/15 |

Current fixed-order selector table:

| Selector | Success |
| --- | ---: |
| Uniform random expected | 4.17/15 |
| Best action heuristic, smoothness max | 6/15 |
| Action distribution nearest-positive | 8/15 |
| Gripper distribution nearest-positive, all-task | 12/15 |
| Phase-gripper nearest-positive, same-task | 13/15 |

Anonymous candidate-ID/rank remap control, averaged over 10 seeds:

| Selector | Success |
| --- | ---: |
| Rank0 | 0/15 |
| Candidate ID full-trace lookup | 0/15 |
| Uniform random expected | 4.17/15 |
| Best action heuristic, smoothness max | 5.3 +/- 0.46 / 15 |
| Action distribution nearest-positive, same-task | 6.8 +/- 0.60 / 15 |
| Gripper distribution nearest-positive, same-task | 11.0 +/- 0.00 / 15 |
| Phase-gripper nearest-positive, same-task | 11.4 +/- 0.49 / 15 |
| Phase-joint nearest-positive, all-task | 12.0 +/- 0.00 / 15 |
| Phase-joint+gripper nearest-positive, all-task | 12.0 +/- 0.00 / 15 |
| DTW action nearest-positive, same-task | 12.0 +/- 0.00 / 15 |
| DTW joint nearest-positive, all-task | 12.0 +/- 0.00 / 15 |
| DTW joint+gripper nearest-positive, all-task | 14.0 +/- 0.00 / 15 |

Interpretation: the signal survives candidate-ID removal, but the fixed-order
13/15 result is not the reviewer-safe headline. The safer current RoboTwin2
claim is 12.0/15 under anonymous candidate-ID/rank randomization, with rank0
and candidate-ID lookup both at 0/15.

The new DTW control changes the interpretation. A strong nearest-expert
trajectory-distance baseline reaches 14/15, which means the current RoboTwin2
pool is not sufficient evidence for executability beyond expert-template
matching. This is a useful diagnostic, not a final main-table result. The next
RoboTwin2 pool must include successful futures that are not simply the full
expert trace and matched negatives that are close in joint/gripper DTW but fail
because of contact or task constraints.

Anti-template diagnostics make the issue sharper. The current pool has 10/15
nominal non-full successes, but their executed joint+gripper trace is DTW-zero
to the full expert trace, likely because execution stops as soon as the task
success flag is reached. Under the stricter criterion, there are 0/15 diverse
non-full success cases and 0/15 matched low-DTW negative cases. This confirms
that the current RoboTwin2 result is a template-confound detector, not yet a
hard executability benchmark.

The first anti-template K=5 rerun improves this substantially. Across
`stack_blocks_two`, `open_laptop`, and `stamp_seal`, rank0 remains 0/15 and
oracle remains 15/15, but diverse non-full-expert successes increase to 14/15
and matched low-DTW negatives appear in 6/15 cases. The result is still not a
finished method table: DTW and gripper-based nearest-positive baselines remain
strong at 13-13.5/15, while the best compact phase/gripper prototype reaches
13.4/15. The useful paper claim is therefore not "we beat DTW"; it is that the
anti-template pool exposes task-dependent contact/timing sensitivity and
requires few-shot task/contact calibration.

A new `targeted_hard` candidate preset is now the next reviewer-risk test. It
adds contact-phase time warps, stronger/offset contact perturbations, and
gripper-contact pulses around the failure sources found by selector analysis.
On a one-case `stamp_seal` seed-0 smoke, it keeps rank0 at 0/1 and oracle at
1/1 while producing both diverse non-full success and matched low-DTW failure
cases. The useful observation is not the score itself; it is that very nearby
trace edits split into successes and failures, which is the setting needed to
test whether a verifier is doing more than expert-template matching.

The first `stamp_seal` K=5 targeted-hard table confirms the near-neighbor part
but also reveals a new shortcut: `energy_sum_max` and `length_max` are still
5/5 because the longest contact-repeat candidate succeeds in every case. A
new `targeted_energy_matched` preset adds longer failed gripper/contact/reverse
probes specifically to test whether the selector advantage survives when failed
futures are at least as long and high-energy as the successful time-warp
futures.

The seed-0 `targeted_energy_matched` smoke gives the desired first signal:
`energy_sum_max` selects a long reverse-contact failure, and `length_max`
selects a long gripper-contact-pulse failure. This turns the previous
energy/length shortcut into an explicit negative control, but it still needs
the full K=5 result before it can support the main claim.

K-shot target-task calibration under the same anonymous remap protocol:

| Selector | K=0 | K=1 | K=2 | K=4 |
| --- | ---: | ---: | ---: | ---: |
| Gripper distribution | 3.0 | 6.0 | 8.3 | 9.7 |
| Phase-gripper distribution | 2.4 | 3.8 | 5.46 | 8.1 |
| Phase-joint distribution | 0.0 | 2.6 | 6.2 | 12.0 |
| Phase-joint+gripper distribution | 2.0 | 3.4 | 6.8 | 12.0 |

Each cell is mean success out of 15 over 10 anonymous rank seeds and 5 support
seeds. K=0 uses source tasks only; K=4 uses all other target-task cases. The
curve supports the same boundary seen in RoboCasa: this is not a universal
zero-shot verifier, but a task/contact-calibrated verifier that improves as
target-task support appears.

## Target Paper Story

Provisional title:

> Executable-Future Verification for Robot World-Action Candidates

Main claim:

> Generated futures often contain executable candidates, but default ranking,
> visual plausibility proxies, action magnitude, and averaged success
> prototypes are brittle. Few-shot execution-envelope verification over compact
> robot traces recovers executable futures under shortcut-controlled negatives
> only when the candidate pool breaks expert-template shortcuts.

Novelty boundary relative to action-level test-time verifiers such as RoVer:
this project should not be written as a generic robot reward model. The safer
claim is trajectory/future-level selection over candidate futures, generator
agnostic inputs, and explicit diagnostics for template matching, candidate-ID
leakage, rank leakage, action magnitude, object-state leakage, and weak
baselines.

Recommended contribution shape:

1. A benchmark-agnostic executable-future manifest protocol.
2. A controlled RoboCasa365 mechanism study showing action shortcut failure and
   robot execution-envelope recovery.
3. A RoboTwin2 2025 benchmark study showing the same recoverable-future problem
   under dual-arm task success checks.
4. Diagnostic controls: action magnitude, candidate-ID/rank randomization,
   object-state ablation, centroid-vs-nearest-positive prototypes, and K-shot
   calibration.

## Main Risks

- The current RoboTwin2 table is small: 15 cases across three tasks.
- Candidate pools still contain an obvious full expert trace; this must be
  replaced or supplemented with less nameable successful futures.
- A strong DTW nearest-positive baseline reaches 14/15 on the current RoboTwin2
  table. Until hard positives and matched hard negatives are added, the result
  is vulnerable to the critique that the selector is expert-trace matching.
- Current nominal non-full successes are not diverse under executed-trace DTW.
  The anti-template diagnostic reports 0/15 diverse non-full successes and
  0/15 matched low-DTW failures.
- The new anti-template K=5 pool fixes most hard-positive coverage but not the
  strongest baseline problem. DTW is no longer oracle, but it is still close to
  the best selector.
- The `targeted_hard` preset has only a one-case smoke so far. It must be
  scaled before any claim about beating template matching is safe.
- The `stamp_seal` targeted-hard K=5 run exposes an energy/length shortcut:
  `energy_sum_max` and `length_max` are 5/5. This must be controlled before the
  result can be used as a reviewer-safe method win.
- We have no real robot; the paper must be framed as executable-future
  verification in modern simulated/world-model benchmarks, not deployment.
- RoboWM-Bench remains conceptually ideal, but current public-code friction
  makes it risky as the main table until the evaluator ceiling is clarified.

## Immediate Next Experiments

1. Build a RoboTwin2 anti-template pool where at least one successful candidate
   per case is not the full expert trace.
2. Add hard positives: successful trajectories with different timing,
   intermediate joint path, or contact timing than the expert trace.
3. Add matched hard negatives: low DTW distance to successful traces but failed
   contact direction, gripper timing, or task completion.
4. Re-run rank0, random, action heuristic, phase prototype, DTW nearest-expert,
   and learned binary/contrastive selector baselines.
5. Keep `handover_block` as a one-seed bimanual mechanism example unless a
   fourth K=5 task is needed for breadth.
6. Scale `--candidate-preset targeted_hard` from the current `stamp_seal`
   seed-0 smoke to K=5, then decide whether it should replace or supplement the
   anti-template main table.
7. Run `--candidate-preset targeted_energy_matched` on `stamp_seal` to verify
   that energy/length heuristics collapse when long failed futures are added.
   Seed 0 has passed this gate; K=5 is the current remote run.

Implementation status: `robotwin2_gripper_aware_trace.py` now has an
`--candidate-preset anti_template` mode that adds time-warp, gripper-timing,
and contact-segment perturbation probes with explicit `candidate_source`
metadata. It also has an experimental `--candidate-preset targeted_hard` mode
for the near-neighbor success/failure pairs needed to test template-matching
baselines, plus a `targeted_energy_matched` mode for the stricter
length/energy-shortcut control.

Latest remote result: see `docs/robotwin2_antitemplate_k5_results.md`. The next
candidate-generation pass should make matched negatives harder, especially by
keeping joint/gripper DTW close while changing contact direction or terminal
task completion.

## Legacy Direction

The earlier UMM/NanoWM reward-evaluator proposal is preserved in
`docs/umm_reward_evaluator_proposal.md`. It is no longer the active ICLR main
story, although video/world-model evaluation can still become a later diagnostic
layer if RoboWM-Bench, MiraBench, or RoboTrustBench data become available.
