# Benchmark Expansion Roadmap

## Current Claim

The active ICLR evidence is restricted to 2025-2026 robotics benchmarks. Older PushT and ManiSkill diagnostics are now archived as mechanism-discovery notes only; they should not be used as headline benchmark evidence.

The current RoboCasa365 claim is sharper and more defensible:

> A conservative manipulation prior is often under-actuated. Endpoint-free action-envelope calibration recovers many failures, but its deterministic magnitude shortcut collapses under energy-matched hard negatives; the useful method must therefore separate action adequacy from action magnitude.

Current 2026 RoboCasa365 numbers:

Ordinary no-demo replay pool:

| Setting | Success |
| --- | ---: |
| Rank0 conservative replay prior | 0/64 |
| Oracle-best no-demo candidate | 41/64 |
| Raw ordered action selector, mean over 5 seeds | 21.6/64 |
| Shuffled-time action selector, mean over 5 seeds | 25.2/64 |
| Endpoint-free stats selector, mean over 5 seeds | 25.8/64 |
| One unordered endpoint pair, mean over 5 seeds | 27.4/64 |
| Bag action-envelope selector, mean over 5 seeds | 28.6/64 |
| Deterministic max mean-absolute-action heuristic | 28/64 |

Energy-matched hard-negative pool with rollout state traces:

| Setting | Success |
| --- | ---: |
| Rank0 conservative replay prior | 0/32 |
| Oracle-best original action trace | 32/32 |
| Deterministic action energy/magnitude/smoothness heuristics | 0/32 |
| Action-only endpoint-free stats selector, mean over 5 seeds | 8.2/32 |
| Low-dimensional state-trace selector, mean over 5 seeds | 30.4/32 |
| State trace + endpoint-free action stats, mean over 5 seeds | 30.0/32 |
| Rank/candidate-id randomized state-trace selector, mean over 5 seeds | 30.4/32 |

## 2025-2026 External-Benchmark Plan

Scope gate:

- Active benchmark evidence must come from 2025-2026 benchmarks.
- Legacy suites such as LIBERO, CALVIN, D4RL, PointMaze, and older tabletop-only diagnostics are no longer acceptable as main ICLR evidence.
- Recently installed benchmark attempts that do not pass this scope gate should be marked inactive rather than kept as active side tracks.
- VideoZeroBench is not part of the active ICLR benchmark stack because it is outside the current robotics benchmark scope. Do not delete its training/data cache when it already exists; keep it available for unrelated video-reasoning work or later restoration.
- The active remote robotics benchmark setup is RoboCasa365 plus its required robosuite dependency.

### 1. RoboCasa365 First

RoboCasa365 is the primary benchmark because it is a 2026 benchmark for generalist household manipulation and substantially closer to modern Physical AI/VLA evaluation than legacy tabletop suites. It covers hundreds of kitchen manipulation tasks and supports Gymnasium-style environment construction through the official RoboCasa/robosuite stack.

Official sources:

- RoboCasa365 paper: https://arxiv.org/abs/2603.04356
- RoboCasa project: https://robocasa.ai/
- RoboCasa code: https://github.com/robocasa/robocasa
- robosuite code: https://github.com/ARISE-Initiative/robosuite

Local finding on the remote machine:

- No legacy benchmark install remains under `/home/yihao_hyh/benchmarks`; only `robocasa` and its required `robosuite` dependency are present.
- The old LIBERO clone was removed from `/home/yihao_hyh/LIBERO`; no `libero-favc` conda env was present.
- Official `robosuite` and `robocasa` repositories were cloned under `/home/yihao_hyh/benchmarks/`.
- A separate `robocasa-favc` conda environment was created with Python 3.11.
- Editable installs for official robosuite and robocasa succeeded.
- Import smoke passed with `robocasa==1.0.1`, `robosuite==1.5.2`, `gymnasium==0.29.1`, and CUDA-enabled `torch==2.7.1+cu126`.
- Gymnasium registers 396 `robocasa/*` environments.
- Full kitchen/object assets were downloaded and extracted.
- Four target-split smoke tasks reset and step successfully: `PickPlaceCounterToCabinet`, `PickPlaceCounterToSink`, `CloseCabinet`, and `TurnOnSinkFaucet`.
- The RoboCasa Gym wrapper exposes language instructions, proprioceptive state, and three 256x256 RGB camera streams per observation.
- Code finding: top-level `import robocasa` does not register Gymnasium IDs; the adapter must import `robocasa.wrappers.gym_wrapper`.
- First demo replay headroom probe on `PickPlaceCounterToCabinet` target-human data: rank0 under-actuated replay success is 0/5, oracle-best success is 5/5, and oracle is better than rank0 in 5/5 cases.
- Randomized replay probe on the same task: conservative-prior rank0 is 0/8, oracle-best is 8/8, held-out action selector is 8/8, and zero-feature control is 0/8. Removing original demo candidates leaves oracle-best 6/8; shuffled-time action statistics reach 6/8.
- Cross-task randomized replay probe on `TurnOnSinkFaucet`: conservative-prior rank0 is 0/8, oracle-best is 8/8 with original demos and 7/8 without original demos. This gives a harder articulated-fixture case where simple no-demo action statistics recover only 2-3/8, so the story cannot collapse to a single pick-place shortcut.
- Third-task randomized replay probe on `OpenCabinet`: conservative-prior rank0 is 0/8, oracle-best is 8/8 with original demos and 6/8 without original demos. No-demo raw and shuffled action-statistic selectors both reach 6/8, which makes the Faucet failures a targeted contact-calibration boundary rather than a generic fixture-task failure.
- Three-task result on 24 target cases: with original demo candidates, shared action-statistic selectors recover 24/24 rank0 failures; in no-demo subsets, selectors recover 14/24 against a 19/24 oracle ceiling, with the remaining gap concentrated in `TurnOnSinkFaucet`.
- Temporal-shuffle diagnostic: after making shuffle controls deterministic, no-demo shuffled-time action statistics recover 16-17/24 across three tasks, compared with 13-14/24 for ordered raw statistics. Simple bag-of-actions moments stay at 14/24, so the gain is not explained by ordinary order-invariant moments alone. Multi-pseudo-endpoint features recover 16,16,17/24, nearly matching shuffle while giving a cleaner endpoint-dropout interpretation.
- Fourth-task randomized replay probe on `TurnOnMicrowave`: conservative-prior rank0 is 0/8, oracle-best is 8/8 with original demos and 6/8 without original demos. In the initial 32-case four-task no-demo multitask evaluation, raw ordered statistics recover 16-18/32 against a 25/32 oracle ceiling, shuffled-time recovers 19-22/32, and multi-pseudo-endpoints recover 16-19/32. This made shuffled-time the strongest small-sample diagnostic while showing endpoint-dropout was only a partial method explanation.
- Permutation-endpoint counterfactual: a single stable unordered pseudo-endpoint pair recovers 20,20,20/32 in four-task no-demo multitask evaluation. Using four unordered pairs is less stable at 17,19,20/32 without length and 16,20,19/32 with length. This gives a sharper story than "more temporal features help": a small amount of endpoint dropout helps calibration, but extra pseudo-temporal evidence can overfit.
- Multiview temporal-dropout calibrator: an outer-isolated meta selector over one unordered endpoint-dropout view plus a shuffled-time view recovers 21,20,20/32, compared with 22,19,19/32 for shuffled-time alone and 20,20,20/32 for unordered endpoints alone. Simple rank aggregation of the same two views stays at 20,19,19/32, so the effect is learned stabilization rather than naive agreement voting.
- Expanded n16 result: four tasks were expanded to sixteen target episodes per task, giving 64 no-demo cases with oracle-best 41/64 and rank0 0/64. Across five seeds, raw ordered statistics recover 21.6/64 on average, shuffled-time recovers 25.2/64, endpoint-free stats recover 25.8/64, one unordered endpoint pair recovers 27.4/64, and bag action-envelope moments recover 28.6/64. This supersedes the small-sample shuffle story: the stronger mechanism is endpoint-free action-envelope calibration.
- Deterministic heuristic control: max mean absolute action recovers 28/64 without training, nearly matching the learned bag critic. This exposes the main shortcut risk and the main paper opportunity: the conservative policy prior is under-actuated, and action-envelope calibration fixes that prior surprisingly well. The next evidence layer must use energy-matched hard negatives to show whether the critic understands more than action magnitude.
- Energy-matched hard-negative control: on four tasks with four target episodes each, original demonstration actions succeed in 16/16 while time-reverse, temporal-roll, time-shuffle, block-swap, xyz-flip, and gripper-flip corruptions all fail. Magnitude/energy/smoothness heuristics collapse to 0/16. Learned action-only selectors recover only 5-6/16 on average, with endpoint-free stats at 6.2/16 and shuffled-time at 6.0/16. This is the strongest current evidence that action-envelope calibration is useful for under-actuation but insufficient for contact-timing correctness without visual/contact context.
- State-trace proxy control: the same energy-matched pool was regenerated with low-dimensional RoboCasa rollout observation traces and then scaled to eight target episodes per task. On 32 hard cases, a zero control remains 0/32, deterministic energy/magnitude/smoothness heuristics remain 0/32, and the action-only endpoint-free selector averages 8.2/32. A state-trace selector recovers 30,31,31,30,30/32 across five seeds (mean 30.4/32), while adding action stats to state traces is slightly worse at 30.0/32. After per-case rank/candidate-id randomization, the state-trace selector remains 30.4/32 and action-only remains 8.4/32, so the result is not explained by fixed `cand_07` placement. This makes the method direction clearer: action-envelope calibration detects under-actuation, but hard-negative discrimination comes from rollout state/contact evidence.

First RoboCasa365 milestone:

1. Replace the intentionally brittle replay rank0 with a non-oracle policy score, likelihood score, or noisy BC proposal.
2. Randomize scale, noise, truncation, and temporal warp per episode so candidate identity is not sufficient.
3. Add endpoint-free action-envelope, endpoint-dropout, shuffle-robust, and task/contact-conditioned calibration features that can separate Faucet-style fixture interaction from easier pick-place and cabinet-opening action statistics.
4. Add energy-matched hard negatives so max-action-magnitude cannot solve the benchmark.
5. Train compact state/contact-conditioned rollout critics, with action-only and zero controls, on the same hard-negative protocol.
6. Report rank0 success, oracle-best success, deterministic magnitude heuristic success, action-only success, state/contact-conditioned success, gated success, and hard-case recovery on target split tasks.

### 2. Newer Complementary Benchmarks Only

Do not spend more setup time on legacy LIBERO, CALVIN, D4RL, PointMaze, PushT, or ManiSkill for the main ICLR evidence. They can be cited as background or internal mechanism probes, but they are not benchmark targets for the current story.

Use newer benchmark layers only after RoboCasa365 has a stronger table:

- RoboTwin 2.0 / recent manipulation benchmark-audit settings for shortcut and statistical-significance stress tests.
- RoboMIND 2.0 if we need a 2025 multi-embodiment dataset layer rather than executable sim.
- 2026 robotic world-model diagnostics such as RoboWM-Bench, MiraBench, and RoboTrustBench if we need a world-model-specific reliability table.

## Reviewer-Facing Minimum Bar

The paper-quality result should not be framed as "we improve PushT" or "we solve a legacy tabletop diagnostic." The target claim should be:

1. RoboCasa365 shows a recoverable failure mode on a current 2026 kitchen-manipulation benchmark.
2. Energy-matched hard negatives show that action magnitude is a real shortcut; after that shortcut is removed, low-dimensional rollout state traces recover 30.4/32 while action-only selectors recover only 8.2/32, and the state-trace result is unchanged under rank/candidate-id randomization.
3. Any second benchmark must pass the 2025-2026 scope gate and add stress-test value beyond RoboCasa365, rather than serving as an easier legacy control.

The key ablation is whether the trained action-world critic helps only when used as a gated override. If global ActionWorld is worse than static progress but the gate is better, that is a stronger and more counterintuitive story.
