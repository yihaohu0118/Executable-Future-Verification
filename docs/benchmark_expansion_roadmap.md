# Benchmark Expansion Roadmap

## Current Claim

The active ICLR evidence is restricted to 2025-2026 robotics benchmarks. Older PushT and ManiSkill diagnostics are now archived as mechanism-discovery notes only; they should not be used as headline benchmark evidence.

The current RoboCasa365 claim is sharper and more defensible:

> A conservative manipulation prior is often under-actuated. Endpoint-free action-envelope calibration recovers many failures, but its deterministic magnitude shortcut collapses under energy-matched hard negatives; the useful method must therefore separate action adequacy from action magnitude.

Because we do not have physical robot access, the submission should be framed as a world-model / executable-future verification paper rather than a real-robot deployment paper. The detailed claim boundary is in `docs/no_real_robot_world_model_strategy.md`.

All future benchmark adapters should target the shared candidate JSONL contract in `docs/future_verification_manifest_protocol.md`, then pass `validate_future_verification_manifest.py` before method evaluation.

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
| Fully regenerated random-position state-trace selector, mean over 5 seeds | 30.4/32 |
| Fully regenerated random-position robot-only state-trace selector, mean over 5 seeds | 31.2/32 |
| Fully regenerated random-position n16 action-only endpoint-free selector, mean over 5 seeds | 28.4/64 |
| Fully regenerated random-position n16 proprio-only selector, mean over 5 seeds | 63.0/64 |
| Fully regenerated random-position n16 EEF/gripper without proprio, mean over 5 seeds | 62.2/64 |
| Fully regenerated random-position n16 EEF position only, mean over 5 seeds | 60.4/64 |
| Fully regenerated random-position n16 EEF position distribution-only, mean over 5 seeds | 62.6/64 |
| Fully regenerated random-position n16 EEF+gripper distribution-only, mean over 5 seeds | 63.6/64 |
| Fully regenerated random-position n16 same-task nearest-positive EEF+gripper prototype | 59/64 |
| Fully regenerated random-position n16 leave-one-task-out EEF+gripper distribution MLP | 25.6/64 |
| Fully regenerated random-position n16 balanced shared-onehot one-shot EEF+gripper calibration | 41.2/64 |
| Fully regenerated random-position n16 full-source no-task-ID source-only transfer | 25.6/64 |
| Fully regenerated random-position n16 full-source no-task-ID one-shot target calibration | 42.2/64 |
| Fully regenerated random-position n16 full-source plus one-shot target EEF+gripper calibration | 46.0/64 |
| Fully regenerated random-position n16 full-source plus four-shot target EEF+gripper calibration | 59.2/64 |
| Fully regenerated random-position n16 full-source plus eight-shot target EEF+gripper calibration | 62.2/64 |
| Fully regenerated random-position n16 independent four-shot EEF+gripper calibration | 56.8/64 |
| Fully regenerated random-position n16 independent eight-shot EEF+gripper calibration | 61.0/64 |
| Fully regenerated random-position n16 robot-only selector, mean over 5 seeds | 64.0/64 |

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
- The local downloaded `datasets/v1.0/target/atomic` tree currently contains only four demonstration tasks: `PickPlaceCounterToCabinet`, `TurnOnSinkFaucet`, `OpenCabinet`, and `TurnOnMicrowave`. There are no local composite/non-atomic RoboCasa365 dataset splits under `datasets/v1.0`. Scaling the same evidence beyond four tasks therefore requires downloading additional RoboCasa365 demonstrations or moving to a second 2025-2026 benchmark.
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
- State-trace proxy control: the same energy-matched pool was regenerated with low-dimensional RoboCasa rollout observation traces and then scaled to eight target episodes per task. On 32 hard cases, a zero control remains 0/32, deterministic energy/magnitude/smoothness heuristics remain weak, and the action-only endpoint-free selector averages 8.2/32. A state-trace selector recovers 30,31,31,30,30/32 across five seeds (mean 30.4/32), while adding action stats to state traces is slightly worse at 30.0/32. After per-case rank/candidate-id randomization, the state-trace selector remains 30.4/32 and action-only remains 8.4/32. A fully regenerated random-position rollout pool gives the same pattern: zero 0/32, action-only 8.4/32, state trace 30.4/32, and state+action 30.0/32. Key ablations make the result sharper: excluding `object-state` improves to 31.4/32, robot-only traces reach 31.2/32, proprio-only reaches 30.2/32, while object low-dimensional keys alone stay at 16/32. Scaling the regenerated random-position pool to sixteen episodes per task gives 64/64 oracle and 0/64 rank0. Action-only endpoint-free stats average only 28.4/64, object-only traces stay at 31.0/64, proprio-only reaches 63.0/64, full state reaches 63.0/64, state+action drops to 62.0/64, and robot-only traces reach 64.0/64 across all five seeds. A finer n16 robot-key ablation shows that EEF/gripper traces without the broad proprio vector still reach 62.2/64, EEF position+gripper reaches 62.0/64, EEF position alone reaches 60.4/64, and gripper alone is only 43.8/64. A summary-statistic ablation is even sharper: terminal-only and endpoint summaries stay near 33-43/64, while EEF position distribution-only reaches 62.6/64 and EEF+gripper distribution-only reaches 63.6/64. A non-neural same-task nearest-positive prototype over the same EEF+gripper distribution features reaches 59/64, while all-task prototypes fall to 21-48/64. Case-heldout independent/shared/task-head MLPs all stay high at 62.4-63.6/64, but leave-one-task-out and no-task-ID source-only transfer stay at 25.6/64. Few-shot adaptation exposes the boundary: full-source no-task-ID one-shot reaches 42.2/64, shared-onehot one-shot reaches 46.0/64, four-shot reaches 59.2/64, and eight-shot reaches 62.2/64; task-specific heads do not improve over shared one-hot. This makes the method direction clearer: action-envelope calibration detects under-actuation, but hard-negative discrimination comes from few-shot task/contact-conditioned robot execution-envelope feedback rather than privileged object-state leakage, zero-shot cross-task generalization, or mandatory task-specific heads.

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

- RoboTwin 2.0 is the best next executable benchmark candidate if setup cost is manageable: it is a 2025 dual-arm manipulation benchmark/data generator with 50 tasks, five robot embodiments, and strong domain randomization. Project/code links from the paper: https://robotwin-platform.github.io/ and https://github.com/robotwin-Platform/robotwin/.
- RoboWM-Bench is the best 2026 world-model-specific candidate if we pivot from execution-envelope probes to evaluating whether generated manipulation futures translate into executable robot actions: https://arxiv.org/abs/2604.19092.
- RoboTrustBench is useful if the paper needs a 2026 trustworthy video-world-model diagnostic under normal, constraint-sensitive, counterfactual, and adversarial instructions, but it is less directly tied to action reranking unless we build a video-world-model evaluation layer: https://arxiv.org/abs/2606.01600.
- RoboMIND 2.0 is relevant if we need a 2025 multi-embodiment real-data layer; it is a dataset/system paper rather than an immediately executable simulator benchmark, so it is a lower-priority next step for the current reranking experiments: https://arxiv.org/abs/2512.24653.

## Reviewer-Facing Minimum Bar

The paper-quality result should not be framed as "we improve PushT" or "we solve a legacy tabletop diagnostic." The target claim should be:

1. RoboCasa365 shows a recoverable failure mode on a current 2026 kitchen-manipulation benchmark.
2. Energy-matched hard negatives show that action magnitude is a real shortcut; after that shortcut is removed, action-only selectors recover only 28.4/64 on the n16 regenerated random-position pool, while proprio-only reaches 63.0/64 and robot-only reaches 64.0/64. The state-trace result is unchanged under both manifest-level and fully regenerated rank/candidate-position randomization. Fine-grained ablations show that compact endpoint-free EEF/gripper execution-envelope statistics recover 63.4-63.6/64 without object state or a broad proprio vector, and a non-neural same-task prototype recovers 59/64, proving most of the signal is an interpretable execution-envelope similarity effect. Leave-one-task-out and no-task-ID source-only transfer stay at 25.6/64. Adding one target-task case reaches 46.0/64 with shared one-hot, four cases reach 59.2/64, and eight cases reach 62.2/64, so the claim should be few-shot task/contact-conditioned rather than zero-shot cross-task.
3. Any second benchmark must pass the 2025-2026 scope gate and add stress-test value beyond RoboCasa365, rather than serving as an easier legacy control.

The key ablation is whether the trained action-world critic helps only when used as a gated override. If global ActionWorld is worse than static progress but the gate is better, that is a stronger and more counterintuitive story.
