# ICLR Idea: Failure-Gated Action-Visual Critics for Recoverable Manipulation Planning Failures

## One-Sentence Idea

Train a lightweight action/video critic to detect recoverable planning failures and use it as a gated override, rather than replacing the planner or visual-progress heuristic globally.

## Why This Is Not Training-Free

The core component is trained case-heldout:

- action-sequence MLP selector
- video-frame MLP selector
- future gate over visual-progress and action/video critic confidence

The method is not a static heuristic, prompt, hand-coded score, or training-free reranker.

## Code-Driven Origin

This idea came from implementation and benchmark behavior, not from paper-level recombination.

### PushT-100 Observation

The PushT-100 first layer showed:

| Selector | Mean state dist | Oracle match |
| --- | ---: | ---: |
| NanoWM rank0 | 190.9580 | 21/100 |
| Static DINO progress | 181.1198 | 41/100 |
| ActionWorld e200 | 178.8655 | 39/100 |
| Static DINO progress + ActionWorld failure gate | 177.3960 | 43/100 |
| Oracle-best CEM | 157.4419 | 100/100 |

The important mechanism was not "world model score always wins." The useful behavior was gated override: keep the static visual-progress anchor unless the trained action-world critic predicts failure.

### ManiSkill PickCube Observation

The ManiSkill adapter exposed a concrete recoverable failure:

- `high_grasp` looks plausible but fails frequently.
- `low_grasp` solves the same initial states.
- A small grasp-height candidate ordering detail creates a large success gap.

20-case headroom:

| Metric | Value |
| --- | ---: |
| Rank0 brittle-grasp success | 4/20 |
| Oracle-best success | 20/20 |
| Oracle better than rank0 | 20/20 |

Candidate-level result:

| Candidate | Success |
| --- | ---: |
| `high_grasp` | 4/20 |
| `low_grasp` | 20/20 |
| `rank0_center` | 20/20 |
| `slow_center` | 20/20 |
| `x_offset` | 20/20 |

## Implemented Evidence

### Action-Sequence Selector

Case-heldout MLP over raw action-sequence statistics:

| Selector | Success | Recovered rank0 failures | Oracle match |
| --- | ---: | ---: | ---: |
| Rank0 brittle grasp | 4/20 | 0/16 | 0/20 |
| Raw action-sequence MLP | 20/20 | 16/16 | 13/20 |
| Raw action MLP without trajectory length | 20/20 | 16/16 | 13/20 |
| Raw action MLP on shuffled manifest rows | 20/20 | 16/16 | 13/20 |
| Zero-action negative control | 4/20 | 0/16 | 0/20 |
| Shuffle-time action control | 20/20 | 16/16 | 8/20 |

N100 stability check:

| Selector | Success | Recovered rank0 failures | Oracle match |
| --- | ---: | ---: | ---: |
| Rank0 brittle grasp | 17/100 | 0/83 | 0/100 |
| Raw action MLP without trajectory length | 100/100 | 83/83 | 71/100 |
| Oracle-best candidate | 100/100 | 83/83 | 100/100 |

Interpretation:

- The learned critic recovers every rank0 failure.
- The zero control proves the result is not only class prior or candidate ordering.
- The no-length control proves the result is not explained by early termination or trajectory length.
- Shuffling manifest rows does not change the result, so JSONL row order is not the shortcut.
- Shuffle-time staying strong means this slice is mostly action-distribution/stage-geometry driven, not precise temporal-order driven.
- The n100 stability check shows the action-critic result is not a small-seed accident.

Second-task StackCube check:

| Selector | Success | Recovered rank0 failures | Oracle match |
| --- | ---: | ---: | ---: |
| StackCube rank0 high-grasp | 0/50 | 0/50 | 0/50 |
| StackCube raw action MLP without trajectory length | 50/50 | 50/50 | 35/50 |
| StackCube oracle-best candidate | 50/50 | 50/50 | 50/50 |

StackCube is a stronger second-task diagnostic than PushCube. PushCube rank0 already succeeded on 100/100 cases, while StackCube exposes multiple recoverable failure types: high grasp, no release, and lateral placement error.

Randomized PickCube candidate-pool check:

| Selector | Success | Recovered rank0 failures | Oracle match |
| --- | ---: | ---: | ---: |
| Random-grasp rank0 | 1/50 | 0/49 | 0/50 |
| Random-grasp raw action MLP without trajectory length | 50/50 | 49/49 | 14/50 |
| Random-grasp oracle-best candidate | 50/50 | 49/49 | 50/50 |

This control reduces the fixed-family shortcut concern. Non-rank0 candidates are sampled continuously over grasp height, xy offset, and gain; the selector chooses across multiple candidate ranks rather than always selecting one fixed index.

Multi-task action-critic diagnostic:

| Selector mode | Overall success | Recovered rank0 failures | Oracle match | Pick success | Stack success |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared MLP + task one-hot | 82/100 | 81/99 | 32/100 | 50/50 | 32/50 |
| Shared trunk + per-task head | 97/100 | 96/99 | 47/100 | 50/50 | 47/50 |
| Independent per-task MLP | 100/100 | 99/99 | 48/100 | 50/50 | 50/50 |

Interpretation:

- A task one-hot is not enough to make a globally shared action critic reliable.
- The action features contain enough information, because independent per-task models reach 100/100.
- Per-task output heads recover most of the shared-critic failure, so the bottleneck is task-specific failure calibration rather than representation alone.

Mixed-rank failure-gate diagnostic:

| Task | Rank0 success | Global critic success | Gated critic success | Gated preserved rank0 |
| --- | ---: | ---: | ---: | ---: |
| PickCube random-grasp | 24/50 | 50/50 | 50/50 | 24/50 |
| StackCube brittle-stack | 20/50 | 50/50 | 49/50 | 18/50 |

Interpretation:

- PickCube supports the conservative-gate claim: the gate preserves correct rank0 choices while recovering all failures.
- StackCube exposes the calibration risk: when the global critic is already near-oracle, a conservative gate can preserve a failing rank0.
- Therefore the paper should not claim that gating is automatically better than global reranking. The stronger claim is that failure detection and override calibration are separate problems.

Training-case scaling diagnostic:

| Task | 4 train cases | 8 train cases | 16 train cases | 32 train cases | All train cases |
| --- | ---: | ---: | ---: | ---: | ---: |
| PickCube random-grasp | 50.0/50 | 50.0/50 | 50.0/50 | 50.0/50 | 50.0/50 |
| StackCube brittle-stack | 43.0/50 | 48.0/50 | 49.0/50 | 49.3/50 | 50.0/50 |

Interpretation:

- PickCube exposes a very low-sample failure signature, likely dominated by grasp-height/action-geometry.
- StackCube is harder but reaches 48/50 with only 8 training cases and 49/50 with 16.
- This supports the claim that failure detection can be data-efficient, while also showing that contact-rich tasks need task-specific calibration.

Learned proposal diagnostic:

| Candidate source / selector | Success | Recovered rank0 failures | Notes |
| --- | ---: | ---: | --- |
| First learned proposal sample | 25/50 | 0/25 | stochastic sample from trained success-conditioned proposal |
| Action critic on learned samples | 49/50 | 24/25 | zero-action control stays 25/50 |
| Likelihood-ranked proposal | 44/50 | 0/6 | strong proposal-likelihood baseline |
| Global action critic on likelihood-ranked samples | 49/50 | 6/6 | recovers failures but harms one successful rank0 |
| Failure-gated action critic on likelihood-ranked samples | 50/50 | 6/6 | preserves 43 rank0 choices and reaches oracle success |

Interpretation:

- This reduces the hand-designed candidate-family concern: candidates come from a trained high-level grasp proposal fit to successful rollouts.
- Proposal likelihood is not equivalent to physical success.
- The full FAVC mechanism is visible: action critic recovers proposal failures, while the gate prevents unnecessary harmful overrides.
- A second proposal sampling seed gives the same rank0 baseline, 44/50, and the gate reaches that pool's oracle ceiling, 49/50.

### Video-Frame Selector

Case-heldout MLP over rendered RGB frame features:

| Selector | Success | Recovered rank0 failures | Oracle match |
| --- | ---: | ---: | ---: |
| Rank0 brittle grasp | 4/20 | 0/16 | 0/20 |
| Raw video-frame MLP | 20/20 | 16/16 | 13/20 |
| Raw video-frame MLP on shuffled manifest rows | 20/20 | 16/16 | 13/20 |
| Zero-video negative control | 4/20 | 0/16 | 0/20 |
| Shuffle-time video control | 20/20 | 16/16 | 10/20 |

Interpretation:

- The failure is visible from rollout frames.
- Zero-video control falls back to rank0.
- Shuffling manifest rows does not change the result, so row order is not the shortcut.
- Shuffle-time staying strong again warns against overclaiming temporal reasoning on this slice.

### Action-Video Fusion Selector

Case-heldout MLP over action and video critic scores:

| Selector | Success | Recovered rank0 failures | Oracle match |
| --- | ---: | ---: | ---: |
| Action-video fusion MLP | 20/20 | 16/16 | 2/20 |

Interpretation:

- Fusion recovers all failures but has low oracle match because many successful candidates are near-equivalent under the binary success metric.
- For this diagnostic slice, success recovery is the primary metric; oracle match is less informative because `rank0_center`, `low_grasp`, `x_offset`, and `slow_center` often all succeed.

## Concrete Method Direction

Name:

**FAVC: Failure-Gated Action-Visual Critic**

Core pipeline:

1. A planner or policy proposes K candidates.
2. A static visual-progress anchor selects the default candidate.
3. A trained action/video critic scores candidate failure likelihood.
4. A task- or mode-conditioned head calibrates failure probability for the current contact regime.
5. An uncertainty-aware gate overrides the anchor only when the anchor is likely to fail and the critic is sufficiently calibrated.
6. The final selector is evaluated on task success and hard-case recovery.

The key claim should be:

> Recoverable planning failures are often easier to identify than to solve from scratch; a trained action-visual critic can detect these failures and selectively override brittle planner choices.

## Why This Is Potentially Novel

The common framing is to build a stronger planner, larger world model, or globally better reward. Our evidence points to a narrower and more counterintuitive mechanism:

- global replacement is not always best;
- a simple trained critic is valuable as a failure detector;
- a globally shared critic can fail even with a task one-hot, and simple per-task heads are still insufficient for Faucet-style contact calibration;
- failure gating is a calibration tradeoff, not a free replacement for global reranking;
- failure detection is data-efficient on these diagnostics, but the sample complexity differs sharply across contact regimes;
- a trained proposal can rank likely samples well while still missing physical execution failures that a rollout critic detects;
- the critic can recover failures caused by small action-geometry mistakes;
- action/video signals can expose failure even when the candidate appears plausible;
- in the current slice, temporal order is not just less important than expected; on the expanded 64-case RoboCasa365 table, endpoint-free action-envelope moments outperform raw ordered summaries, shuffled-time diagnostics, and endpoint-dropout views, while a deterministic max-absolute-action heuristic nearly matches the learned critic.
- when action magnitude is controlled by energy-matched corruptions, deterministic energy/magnitude/smoothness shortcuts collapse to 0/32 and learned action-only selectors recover only 8.2/32;
- low-dimensional rollout state traces recover 30.4/32 on the same energy-matched stress test, while adding action stats to state traces is slightly worse at 30.0/32; the state-trace result stays 30.4/32 after both manifest-level randomization and fully regenerated random-position rollouts, exposing a sharper bottleneck: action adequacy must be conditioned on rollout state/contact context rather than action envelope alone.

## Current Weaknesses

These must be addressed before this is paper-ready:

1. Current RoboCasa365 candidates still use a conservative replay prior rather than a learned policy likelihood.
2. Current video/action selectors may exploit candidate-family regularities.
3. Deterministic action magnitude is a strong baseline on ordinary no-demo pools: max mean absolute action reaches 28/64 without training, nearly matching the learned bag critic.
4. Energy-matched hard negatives remove that shortcut but current action-only selectors still recover only 8.2/32 against a 32/32 oracle.
5. Low-dimensional rollout state traces recover 30.4/32 on the same stress test and remain 30.4/32 after fully regenerated random-position rollouts, but this is still a state/proprio proxy rather than a deployable RGB-only visual critic.
6. Current phenomenon does not prove temporal world modeling; it proves an under-actuation shortcut first, then exposes the need for visual/contact-conditioned discrimination.
7. Need a full low-level policy-generated candidate source or official benchmark demonstrations for external validity.
8. Need a harder fusion benchmark because current action/video critics each solve the slice independently.
9. Need uncertainty-aware gate calibration; mixed-rank settings can make a conservative gate preserve a failing rank0.
10. The benchmark stack must stay inside the 2025-2026 robotics window; VideoZeroBench should remain inactive for this ICLR evidence chain, but existing VideoZeroBench data/cache should be preserved for unrelated video-reasoning work.

## Next Experiments

Priority order:

1. Make RoboCasa365 the headline benchmark layer and scale beyond the current four-task probe.
2. Scale the endpoint-free action-envelope critic for the Faucet/Microwave gap: in the expanded 64-case no-demo table, oracle-best is 41/64, raw ordered summaries average 21.6/64, shuffled-time averages 25.2/64, endpoint-free stats average 25.8/64, one unordered endpoint pair averages 27.4/64, bag action-envelope moments average 28.6/64, and max mean absolute action reaches 28/64 without training.
3. Scale the fully regenerated random-position energy-matched pool beyond the current four-task probe.
4. Convert the state-trace result into a deployable visual/contact-conditioned selector: the target is to preserve the 30.4/32 state-proxy gain without privileged state leakage.
5. Replace the diagnostic replay prior with BC/diffusion-policy top-k samples where a 2025-2026 benchmark provides usable policy or demonstration sources.
6. Add uncertainty-aware calibration to the gate and evaluate under mixed proposal qualities.
7. Use only verified 2025-2026 robotics benchmarks as the next layer, such as RoboTwin 2.0, RoboMIND 2.0, or 2026 robotic world-model diagnostics; do not make legacy LIBERO/CALVIN/D4RL or unrelated side tracks part of the main evidence.

## Implemented Files

- `src/umm_reward_evaluator/benchmarks/maniskill_candidate_pool.py`
- `src/umm_reward_evaluator/benchmarks/train_action_sequence_selector.py`
- `src/umm_reward_evaluator/benchmarks/train_multitask_action_sequence_selector.py`
- `src/umm_reward_evaluator/benchmarks/train_multiview_action_sequence_selector.py`
- `src/umm_reward_evaluator/benchmarks/score_action_heuristic_selector.py`
- `src/umm_reward_evaluator/benchmarks/train_gated_action_sequence_selector.py`
- `src/umm_reward_evaluator/benchmarks/train_action_sequence_selector_scaling.py`
- `src/umm_reward_evaluator/benchmarks/train_video_frame_selector.py`
- `src/umm_reward_evaluator/benchmarks/train_action_video_fusion_selector.py`
- `src/umm_reward_evaluator/benchmarks/maniskill_bc_policy_pool.py`
- `src/umm_reward_evaluator/benchmarks/maniskill_learned_grasp_proposal_pool.py`
- `src/umm_reward_evaluator/benchmarks/shuffle_manifest_rows.py`
- `src/umm_reward_evaluator/benchmarks/merge_candidate_manifests.py`
- `src/umm_reward_evaluator/benchmarks/robocasa365_smoke.py`
- `src/umm_reward_evaluator/benchmarks/robocasa_demo_candidate_pool.py`
- `src/umm_reward_evaluator/benchmarks/train_multitask_state_trace_selector.py`
- `src/umm_reward_evaluator/benchmarks/filter_candidate_manifest.py`
- `docs/maniskill_pickcube_brittle_grasp_headroom.md`
- `docs/maniskill_pickcube_action_selector_results.md`
- `docs/maniskill_pickcube_video_selector_results.md`
- `docs/maniskill_pickcube_n100_action_stability.md`
- `docs/maniskill_stackcube_brittle_stack_n50.md`
- `docs/maniskill_pickcube_random_grasp_n50.md`
- `docs/maniskill_multitask_action_critic.md`
- `docs/maniskill_mixed_rank_gate_diagnostic.md`
- `docs/maniskill_action_selector_scaling.md`
- `docs/maniskill_learned_grasp_proposal.md`

## Newly Added Verification Hooks

The implementation now includes reviewer-oriented controls:

1. `raw_no_length` action features: removes trajectory length as a possible success shortcut while preserving action statistics.
2. Candidate-row shuffling: tests whether results are invariant to JSONL order. Selector tie-breaking now prefers planner rank0 when scores are equal, so zero controls are well-defined and not row-order dependent.
3. Action-video fusion: tests whether the FAVC critic can combine independently trained action and visual critics in a held-out protocol.
4. RoboCasa low-dimensional state-trace selector: tests whether energy-matched failures require rollout state/contact context after action magnitude shortcuts are removed. On the current n8 hard-negative pool, action-only endpoint-free stats average 8.2/32 while state traces recover 30.4/32; after fully regenerated random-position rollouts, action-only is 8.4/32 and state traces remain 30.4/32.

The implementation also includes `train_action_video_fusion_selector.py`, a case-heldout fusion critic over action and video selector scores. This is the first minimal FAVC implementation beyond separate action-only and video-only diagnostics.

The implementation now also includes `train_multitask_action_sequence_selector.py`, which tests whether action critics should share one global head, use task-specific heads, or remain fully independent. The first result is a concrete negative/positive pair: shared one-hot underperforms on StackCube, while per-task heads recover most of the gap.

The implementation also includes `randomize_planner_rank.py` and `train_gated_action_sequence_selector.py`, which create mixed-planner-rank diagnostics and evaluate held-out failure gates. These scripts expose a useful negative result: gating preserves good planner choices on PickCube but slightly underperforms global reranking on StackCube due to calibration.

The implementation also includes `train_action_sequence_selector_scaling.py`, which measures data efficiency by limiting the number of training cases per held-out fold. The first scaling result shows a sharp task difference: PickCube solves with 4 training cases, while StackCube needs roughly 8-16 cases to approach its full-data ceiling.

The implementation now includes `maniskill_learned_grasp_proposal_pool.py`, which creates a trained high-level grasp proposal from successful candidate rollouts. This gives the strongest current story: a likelihood-ranked learned proposal reaches 44/50, a global action critic reaches 49/50, and the failure-gated critic reaches 50/50.
