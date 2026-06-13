# 2025-2026 Benchmark Portfolio For ICLR Evidence

## Goal

The paper should not depend on a single simulator. The target evidence stack is:

1. executable manipulation benchmark with strong current results;
2. second executable manipulation benchmark with different embodiments/tasks;
3. world-model diagnostic benchmark that directly tests action-conditioned reliability;
4. optional trust/counterfactual diagnostic benchmark;
5. RoboWM-Bench only after its public evaluator path is clarified.

The unifying claim is:

> Visual plausibility or action likelihood is not enough. Few-shot robot execution-envelope verification can select executable futures under shortcut-controlled candidate pools.

## Active Portfolio

| Priority | Benchmark | Year | Role | Current status | Main metric |
| --- | --- | ---: | --- | --- | --- |
| P0 | RoboCasa365 | 2026 | Primary executable manipulation evidence | Running, strong n16 hard-negative results | task success / oracle recovery |
| P1 | RoboTwin 2.0 | 2025 | Second executable manipulation benchmark | Dedicated dev2 env works; clean expert smoke passed; needs candidate tracing | task success under randomized dual-arm tasks |
| P2 | MiraBench | 2026 | Action-conditioned world-model reliability diagnostic | Paper public; code/data release not yet confirmed in search | action fidelity / optimism-bias labels |
| P3 | RoboTrustBench | 2026 | Trust/counterfactual/adversarial video-world-model diagnostic | Project page and 40-sample HuggingFace subset public; full dataset coming soon | trustworthiness criteria |
| Conditional | RoboWM-Bench | 2026 | Embodied world-model-to-action benchmark | Environment works after Vulkan/EGL fix; public eval has Pick reset mismatch | simulator success after official clarification |

Source links:

- RoboTwin 2.0: https://arxiv.org/abs/2506.18088 and https://github.com/robotwin-Platform/robotwin/
- RoboWM-Bench: https://arxiv.org/abs/2604.19092 and https://github.com/fffstrong/RoboWM-Bench
- MiraBench: https://arxiv.org/abs/2605.29360
- RoboTrustBench: https://arxiv.org/abs/2606.01600

## Why These Benchmarks Fit

### RoboCasa365

RoboCasa365 remains the main table because it already exposes the mechanism:

- rank0 conservative replay is 0/64 on the regenerated random-position hard-negative pool;
- oracle is 64/64;
- action-only endpoint-free statistics recover 28.4/64;
- robot-only traces recover 64.0/64;
- EEF/gripper distribution-only reaches 63.6/64;
- leave-one-task-out and no-task-ID source-only transfer fail at 25.6/64;
- few-shot target calibration recovers the signal.

This is the cleanest current evidence that the method is few-shot task/contact-conditioned execution-envelope verification, not a generic action magnitude heuristic.

### RoboTwin 2.0

RoboTwin 2.0 is the next executable benchmark because it adds:

- bimanual manipulation instead of RoboCasa's kitchen single-arm setup;
- multiple robot embodiments;
- 50 tasks and strong domain randomization;
- expert-valid seed filtering in the official evaluation pipeline.

The fair protocol is not to test whether we can find feasible seeds. The fair protocol is:

1. fix expert-valid seeds from the official eval path;
2. generate several candidate futures per seed;
3. execute all candidates under the official success function;
4. ask whether an execution-envelope verifier selects the successful candidate.

The existing converter is `robotwin2_trace_to_manifest.py`.

Current dev2 status:

- dedicated `robotwin2-favc` conda environment created;
- official assets downloaded under `/tmp/robotwin_probe/assets`;
- SAPIEN render smoke passes;
- curobo fused LBFGS custom kernel fails on H100 with illegal instruction, but
  disabling that fused kernel and using curobo's PyTorch/JIT fallback fixes
  planner warmup;
- `click_bell` clean expert smoke succeeds with seed 0.
- a four-candidate `click_bell` trace converts to the shared manifest and
  validates with no schema errors, but it has no reranking headroom because
  rank0 already succeeds and the reversed endpoint candidate also succeeds.

The next evidence step is not another environment smoke. It is candidate
generation and trace emission on 1-2 harder, order-sensitive tasks.

### MiraBench

MiraBench directly targets action-conditioned reliability in robotic world models. Its reported levels are:

- Physics Adherence;
- Action-Following Fidelity;
- Optimism Bias Detection.

This is not necessarily an executable simulator benchmark, but it is almost exactly the diagnostic claim we need: visual fidelity is a poor proxy for action fidelity, and models can be over-optimistic under failure-inducing actions.

Use `world_model_diagnostic_to_manifest.py` once data/judgment JSON is available.

### RoboTrustBench

RoboTrustBench tests video world models under:

- Normal;
- Constraint-Sensitive;
- Counterfactual;
- Adversarial.

This can support the safety/trust version of the claim: generated videos can look coherent while violating constraints, counterfactual grounding, physical interaction, or unsafe-instruction suppression.

Use the same diagnostic adapter as MiraBench.

Current public-data status:

- project page: `https://huiqiongli.github.io/RoboTrustBench/`;
- HuggingFace dataset: `Huiqiong0124/RoboTrustBench_Dataset`;
- current release contains a 40-sample subset across Normal,
  Constraint-sensitive, Counterfactual, and Adversarial categories;
- the dataset card states that the full dataset is coming soon.

Therefore RoboTrustBench is useful now for adapter validation and prompt-level
diagnostic plumbing, but it should not be marked as a passed paper-level
benchmark until we have a multi-candidate judgment manifest and the diagnostic
gate passes.

### RoboWM-Bench

RoboWM-Bench is conceptually the closest benchmark, but it should remain conditional for main results until the official evaluator question is resolved.

Current verified facts:

- IsaacSim 5.1 + IsaacLab v2.3.0 can run on dev2 inside `robowmbench_env`.
- The container required a NVIDIA Vulkan ICD JSON pointing at `libEGL_nvidia.so.0` plus GLVND EGL packages.
- Official GitHub `main` has `eval_franka.py` calling `env.reset(pose_name=...)`, while `Task00_Pick/pick.py` only implements `_reset_idx(...)`.
- With a minimal reset shim and camera-enabled replay, the first 10 `GT/pick` episodes score 7/10.

This should be treated as a current-public-evaluator audit finding, not as final evidence against the benchmark.

## Minimum Paper-Quality Evidence Stack

The minimum acceptable main paper table should contain:

| Layer | Benchmark | Required result |
| --- | --- | --- |
| Executable main | RoboCasa365 | existing hard-negative n16 table |
| Executable second | RoboTwin 2.0 | oracle-best beats rank0 on at least 4 tasks; execution-envelope selector beats action-only controls |
| World-model diagnostic | MiraBench | verifier/reward signal aligns with action-conditioned reliability better than visual/model rank |
| Robustness diagnostic | RoboTrustBench or RoboWM-Bench | either trust/counterfactual result or official-confirmed embodied world-model replay |

## Immediate Work Plan

1. RoboTwin 2.0:
   - patch official `script/eval_policy.py` to record actions, qpos/endpose summaries, video path, and success;
   - run a one-task/five-seed/four-candidate candidate-generation smoke;
   - convert traces with `robotwin2_trace_to_manifest.py` and validate with `--require-future-metadata`.

2. MiraBench:
   - monitor official code/data release;
   - if only paper examples are available, prepare the manifest adapter and synthetic schema validation now;
   - once data is available, import human/MLLM judgments as diagnostic oracle labels.

3. RoboTrustBench:
   - same diagnostic adapter path as MiraBench;
   - prioritize only if it has public data before MiraBench.

4. RoboWM-Bench:
   - open issue asking about the reset path and expected GT replay ceiling;
   - do not spend more large-scale evaluation time until the official intended evaluator path is confirmed.

## Adapter Commands

RoboTwin executable traces:

```bash
python -m umm_reward_evaluator.benchmarks.robotwin2_trace_to_manifest \
  --input-dir /path/to/robotwin_candidate_traces \
  --output-manifest /path/to/robotwin2_manifest.jsonl
```

MiraBench / RoboTrustBench diagnostic judgments:

```bash
python -m umm_reward_evaluator.benchmarks.world_model_diagnostic_to_manifest \
  --input /path/to/judgments.jsonl \
  --default-benchmark mirabench \
  --default-suite action_conditioned_reliability \
  --default-verification-target action_conditioned_reliability \
  --output-manifest /path/to/mirabench_manifest.jsonl
```

Then validate:

```bash
python -m umm_reward_evaluator.benchmarks.validate_future_verification_manifest \
  --manifest /path/to/manifest.jsonl \
  --require-future-metadata
```
