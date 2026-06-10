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

Interpretation: the signal survives candidate-ID removal, but the fixed-order
13/15 result is not the reviewer-safe headline. The safer current RoboTwin2
claim is 12.0/15 under anonymous candidate-ID/rank randomization, with rank0
and candidate-ID lookup both at 0/15.

## Target Paper Story

Provisional title:

> Executable-Future Verification for Robot World-Action Candidates

Main claim:

> Generated futures often contain executable candidates, but default ranking,
> visual plausibility proxies, action magnitude, and averaged success
> prototypes are brittle. Few-shot execution-envelope verification over compact
> robot traces recovers executable futures under shortcut-controlled negatives.

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
- We have no real robot; the paper must be framed as executable-future
  verification in modern simulated/world-model benchmarks, not deployment.
- RoboWM-Bench remains conceptually ideal, but current public-code friction
  makes it risky as the main table until the evaluator ceiling is clarified.

## Immediate Next Experiments

1. Add compact EEF/contact-direction features to address the `open_laptop`
   boundary.
2. Build a candidate pool where success is not always exactly the full expert
   trace.
3. Add K-shot calibration curves on RoboTwin2 and compare source-only,
   no-task-ID, same-task prototype, and task-conditioned variants.
4. Keep `handover_block` as a one-seed bimanual mechanism example unless a
   fourth K=5 task is needed for breadth.

## Legacy Direction

The earlier UMM/NanoWM reward-evaluator proposal is preserved in
`docs/umm_reward_evaluator_proposal.md`. It is no longer the active ICLR main
story, although video/world-model evaluation can still become a later diagnostic
layer if RoboWM-Bench, MiraBench, or RoboTrustBench data become available.
