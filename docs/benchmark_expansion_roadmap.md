# Benchmark Expansion Roadmap

## Current Claim

The strongest PushT-100 result is not "a learned world score is globally better." The more defensible claim is:

> Temporal visual progress is a strong anchor, and a trained action-world critic helps when it is used as a failure override rather than as a global replacement.

Current PushT-100 numbers:

| Setting | Mean state dist | Oracle match |
| --- | ---: | ---: |
| NanoWM rank0 | 190.9580 | 21/100 |
| Static DINO mean | 179.5808 | 33/100 |
| Static DINO progress | 181.1198 | 41/100 |
| ActionWorld e200 | 178.8655 | 39/100 |
| Static DINO progress + ActionWorld failure gate | 177.3960 | 43/100 |
| Oracle-best CEM | 157.4419 | 100/100 |

## 2025-2026 External-Benchmark Plan

Scope gate:

- Active benchmark evidence must come from 2025-2026 benchmarks.
- Legacy suites such as LIBERO, CALVIN, D4RL, PointMaze, and older tabletop-only diagnostics are no longer acceptable as main ICLR evidence.
- Recently installed benchmark attempts that do not pass this scope gate should be removed from the remote machine rather than kept as side tracks.
- As of the latest remote cleanup, the newly installed VideoZeroBench environment, project directory, and downloaded data shards were removed. The active remote benchmark setup is RoboCasa365 plus its required robosuite dependency.

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
- Fourth-task randomized replay probe on `TurnOnMicrowave`: conservative-prior rank0 is 0/8, oracle-best is 8/8 with original demos and 6/8 without original demos. In four-task no-demo multitask evaluation, raw ordered statistics recover 16-18/32 against a 25/32 oracle ceiling, shuffled-time recovers 19-22/32, and multi-pseudo-endpoints recover 16-19/32. This keeps shuffled-time as the strongest diagnostic while showing endpoint-dropout is only a partial method explanation.
- Permutation-endpoint counterfactual: a single stable unordered pseudo-endpoint pair recovers 20,20,20/32 in four-task no-demo multitask evaluation. Using four unordered pairs is less stable at 17,19,20/32 without length and 16,20,19/32 with length. This gives a sharper story than "more temporal features help": a small amount of endpoint dropout helps calibration, but extra pseudo-temporal evidence can overfit.

First RoboCasa365 milestone:

1. Replace the intentionally brittle replay rank0 with a non-oracle policy score, likelihood score, or noisy BC proposal.
2. Randomize scale, noise, truncation, and temporal warp per episode so candidate identity is not sufficient.
3. Add shuffle-robust, endpoint-dropout, and task/contact-conditioned calibration features that can separate Faucet-style fixture interaction from easier pick-place and cabinet-opening action statistics.
4. Train the same compact action critic and failure gate used in the diagnostic pipeline.
5. Report rank0 success, oracle-best success, gated success, and hard-case recovery on target split tasks.

### 2. Newer Complementary Benchmarks Only

Do not spend more setup time on legacy LIBERO, CALVIN, D4RL, or PointMaze for the main ICLR evidence. They can be cited as background, but they are not the benchmark target for the current story.

Use newer benchmark layers only after RoboCasa365 has a stronger table:

- RoboTwin 2.0 / recent manipulation benchmark-audit settings for shortcut and statistical-significance stress tests.
- RoboMIND 2.0 if we need a 2025 multi-embodiment dataset layer rather than executable sim.
- 2026 robotic world-model diagnostics such as RoboWM-Bench, MiraBench, and RoboTrustBench if we need a world-model-specific reliability table.

## Reviewer-Facing Minimum Bar

The paper-quality result should not be framed as "we improve PushT." The target claim should be:

1. PushT-100 shows the mechanism under NanoWM/CEM planning.
2. RoboCasa365 shows the same failure-gated mechanism transfers to a current kitchen-manipulation benchmark with strong 2025-2026 relevance.
3. Any second benchmark must pass the 2025-2026 scope gate and add stress-test value beyond RoboCasa365, rather than serving as an easier legacy control.

The key ablation is whether the trained action-world critic helps only when used as a gated override. If global ActionWorld is worse than static progress but the gate is better, that is a stronger and more counterintuitive story.
