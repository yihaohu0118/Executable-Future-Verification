# RoboCasa365 Benchmark Plan

## Why This Benchmark

RoboCasa365 is the current high-impact target for the ICLR story because it is a 2026 benchmark for generalist household manipulation rather than a legacy tabletop suite. The reported benchmark scale is 365 everyday tasks across 2,500 kitchen environments, with large human and synthetic demonstration corpora.

This makes it a better fit for our current claim than older LIBERO-style evaluation:

- It is current for the 2025-2026 benchmark cycle.
- It tests household mobile manipulation, not only small tabletop primitives.
- It has enough task diversity to test whether the failure-gated action critic is a mechanism or a one-task artifact.
- It still exposes simulator success labels, which lets us measure rank0, oracle-best, and gated recovery without a real robot.

## Current Remote Setup

Remote host: `dev2`

Environment:

- Conda env: `robocasa-favc`
- Python: 3.11
- Official repos:
  - `/home/yihao_hyh/benchmarks/robosuite`
  - `/home/yihao_hyh/benchmarks/robocasa`
- Import smoke:
  - `robocasa==1.0.1`
  - `robosuite==1.5.2`
  - `gymnasium==0.29.1`
  - `torch==2.7.1+cu126`
  - CUDA available
  - 396 registered `robocasa/*` env IDs

Status:

- LIBERO clone was removed from the remote machine.
- A fresh check found no legacy benchmark installs under `/home/yihao_hyh/benchmarks`; only RoboCasa365 and its required robosuite dependency remain.
- VideoZeroBench is inactive for this project because the current benchmark evidence is restricted to 2025-2026 robotics benchmarks. Its data/cache should not be deleted when present; keep it separate from the active RoboCasa365 setup.
- RoboCasa kitchen/object assets are being downloaded.
- RoboCasa kitchen/object assets finished downloading and extracting.
- Reset/step smoke succeeds for `PickPlaceCounterToCabinet`, `PickPlaceCounterToSink`, `CloseCabinet`, and `TurnOnSinkFaucet` on `split=target`.
- The wrapper returns language text, proprioceptive state, and three 256x256 RGB camera views.
- `robocasa/CloseDoor` is registered but requires an extra `fixture_id`, so it is not a good default smoke task.
- Implementation detail: Gymnasium registration requires importing `robocasa.wrappers.gym_wrapper`; top-level `import robocasa` is not sufficient.
- Implementation detail: run the smoke script from `/tmp` or another neutral path. Running a standalone script from `/home/yihao_hyh/benchmarks` can interact badly with the sibling `robocasa/` repo directory on `sys.path`.
- A first target-human demo candidate probe was run for `PickPlaceCounterToCabinet`: rank0 under-actuated replay failed in 0/5 episodes, while oracle-best recovered 5/5. See `docs/robocasa365_demo_candidate_probe.md`.
- A randomized candidate probe reduces fixed candidate-ID shortcuts: with eight target-human episodes and six candidates per episode, conservative-prior rank0 gets 0/8, oracle-best gets 8/8, and a held-out action selector gets 8/8. Removing original demo candidates leaves oracle-best at 6/8 and shuffled-time action statistics reach 6/8.
- A second randomized probe on `TurnOnSinkFaucet` confirms cross-task headroom: conservative-prior rank0 gets 0/8 and oracle-best gets 8/8 with original demos. Removing original demos leaves oracle-best at 7/8, while simple action-statistic selectors recover only 2-3/8. This is a useful harder case because it shows the no-demo setting needs task-conditioned calibration rather than a single global action-energy shortcut.
- A third randomized probe on `OpenCabinet` adds a longer fixture-interaction task. Conservative-prior rank0 gets 0/8 and oracle-best gets 8/8 with original demos. Removing original demos leaves oracle-best at 6/8, and raw/shuffled action-statistic selectors both reach 6/8.
- Three-task multitask result: with original demo candidates, shared action-statistic selectors recover 24/24 rank0 failures while zero-feature control recovers 1/24. In no-demo subsets, the selector recovers 14/24 against a 19/24 oracle ceiling; the missing cases are concentrated in `TurnOnSinkFaucet`.
- A temporal-shuffle diagnostic strengthens the counterintuitive mechanism: in no-demo subsets, deterministic shuffled-time action statistics recover 16-17/24 across three seeds, compared with 13-14/24 for ordered raw statistics. Simple bag-of-actions moments stay at 14/24, so the effect is not explained by generic order-invariant moments alone. Multi-pseudo-endpoint features recover 16,16,17/24, nearly matching shuffle while providing a cleaner endpoint-dropout interpretation.
- `TurnOnMicrowave` adds a fourth target task and a second button-style fixture interaction. Conservative-prior rank0 gets 0/8 and oracle-best gets 8/8 with original demos; removing original demos leaves oracle-best at 6/8. In four-task no-demo multitask evaluation, raw ordered features recover 16-18/32 against a 25/32 oracle ceiling, shuffled-time recovers 19-22/32, and multi-pseudo-endpoints recover 16-19/32.
- Unordered permutation-endpoint features add a cleaner counterfactual to the shuffle diagnostic. A single stable unordered endpoint pair recovers 20,20,20/32 in four-task no-demo multitask evaluation, while four unordered pairs recover 17,19,20/32 without length and 16,20,19/32 with length. This suggests endpoint dropout is useful, but adding more pseudo-endpoints can destabilize small-data calibration.
- A multiview temporal-dropout calibrator gives the cleanest method-shaped four-task no-demo result so far: one unordered endpoint view plus shuffled-time view, combined by an outer-isolated logistic calibrator, recovers 21,20,20/32. Simple rank aggregation of the same views recovers only 20,19,19/32, so the effect is learned stabilization rather than naive voting.
- The expanded n16 table is now the strongest evidence layer. With sixteen episodes per task, oracle-best is 41/64 and rank0 is 0/64. Over five seeds, raw ordered summaries recover 21.6/64 on average, shuffled-time recovers 25.2/64, endpoint-free stats recover 25.8/64, one unordered endpoint pair recovers 27.4/64, and bag action-envelope moments recover 28.6/64. This makes endpoint-free action-envelope calibration the current best method-shaped result.
- A deterministic max-absolute-action heuristic recovers 28/64 on the same n16 pool without training. This is nearly tied with the learned bag critic and shows the current candidate pool is partly an under-actuation diagnostic. Future tables must include this heuristic and add energy-matched hard negatives.

## Smoke Command

Run after assets finish:

```bash
python -m umm_reward_evaluator.benchmarks.robocasa365_smoke \
  --list-envs \
  --list-limit 20 \
  --tasks \
    robocasa/PickPlaceCounterToCabinet \
    robocasa/PickPlaceCounterToSink \
    robocasa/CloseCabinet \
    robocasa/TurnOnSinkFaucet \
  --split target \
  --seed 0 \
  --num-steps 1 \
  --output runs/robocasa365_smoke_seed0.json
```

## First Experimental Layer

The first layer should not attempt full VLA/diffusion-policy training immediately. The minimum publishable diagnostic is candidate-selection headroom:

1. Generate multiple action candidates per initial state.
2. Execute each candidate in RoboCasa.
3. Measure:
   - planner rank0 success
   - oracle-best success
   - oracle-better-than-rank0 cases
   - failure-gated action critic success
4. Train the same compact action critic used in the diagnostic pipeline.
5. Report hard-case recovery where rank0 fails but at least one candidate succeeds.

## Candidate Families

Start with tasks where success is object/contact sensitive but not too long-horizon:

- `PickPlaceCounterToCabinet`
- `OpenCabinet`
- `TurnOnSinkFaucet`
- `TurnOnMicrowave`
- `PickPlaceCounterToStove`
- `PickPlaceSinkToCounter`

`PickPlaceCounterToSink` is useful as an environment smoke task but currently has no target-human dataset in the official RoboCasa365 registry, so it should not be used for the target-split demo replay table.

Then expand to longer or more compositional families only after headroom is visible.

## 2025-2026 Complementary Benchmarks

These are useful for the second layer but should not block RoboCasa365. They must be verified before installation; anything outside this window should stay out of the active benchmark stack.

- RoboTwin 2.0 / 2026 manipulation benchmark-audit settings: good for shortcut, statistical-significance, and data-source-dependence stress tests.
- RoboMIND 2.0: 2025 multi-embodiment manipulation data layer; useful if we need dataset-scale evaluation without real-robot execution.
- RoboWM-Bench: 2026, world-model execution validation for manipulation. Good for comparing action-conditioned visual prediction to executable behavior.
- MiraBench: 2026, action-conditioned reliability and optimism-bias diagnostics for robotic world models. Good for framing why visual fidelity is insufficient.
- RoboTrustBench: 2026, trustworthiness of video world models for robotic manipulation. Good for stress tests and failure taxonomy.

## Target Claim

The target claim is:

> In current manipulation benchmarks, the useful role of an action-conditioned critic is not to globally replace visual/planner scores, but to identify planner failures and rerank executable alternatives.

The key evidence is a consistent gap:

- rank0 is brittle;
- oracle-best shows real candidate-set headroom;
- ordered learned scoring can overfit or mis-rank;
- shuffle-robust calibration can be stronger than preserving true temporal endpoints in the current small-data regime;
- endpoint-free action-envelope calibration is stronger than raw ordered summaries, shuffled-time diagnostics, and endpoint dropout on the expanded 64-case table;
- endpoint-dropout remains a useful second-best mechanism and supports the claim that brittle first/last anchors are harmful;
- deterministic action magnitude is a strong baseline, so the next benchmark layer must test energy-matched alternatives rather than only low-energy rank0 failures;
- a failure-gated critic recovers hard cases with less damage to easy cases.
