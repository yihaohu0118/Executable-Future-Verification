# NanoWM + UMM Reward Evaluator Proposal

## Working Title

WorldReward: Unified Multimodal Evaluation for Action-Conditioned World Model Rollouts

## Core Question

Can a unified multimodal model (UMM) evaluate world-model rollouts more reliably than pixel-level metrics, and can that evaluator improve rollout selection or planning?

The target setting is action-conditioned world modeling. Given an initial observation, task instruction, candidate action sequence, and generated rollout video, the evaluator scores whether the rollout is task-relevant, action-consistent, temporally coherent, physically plausible, and stable over long horizons.

## Motivation

World models are usually evaluated with reconstruction or video-quality metrics such as PSNR, LPIPS, FVD, or frame prediction loss. These metrics are weak proxies for interactive usefulness. A rollout can look visually plausible while failing the task, ignoring the action, drifting object identity, or violating contact dynamics.

UMM-based evaluation introduces semantic and task-level scoring. Instead of asking whether generated pixels match a reference, it asks whether the predicted future is a useful and believable consequence of the proposed actions.

## Proposed System

### Inputs

- Task instruction or goal description.
- Initial observation or context frames.
- Candidate action sequence.
- Generated rollout video from a world model.
- Optional goal image or final-state reference.

### Outputs

- Scalar reward score.
- Sub-scores:
  - task success
  - task progress
  - action consistency
  - temporal consistency
  - physical plausibility
  - identity and layout consistency
  - evaluator confidence
- Optional text critique explaining failure modes.

## Initial Experimental Platform

### World Model Backbone

Use `nano-world-model` as the rollout backbone.

Reasons:

- It is a small, inspectable codebase.
- It supports action-conditioned video rollout.
- It includes planning-oriented workflows such as MPC/CEM.
- It is easier to modify than large video diffusion or VLA systems.

### Consistency Inspiration

Use ideas from `Echo-Memory` to design hard negatives and long-horizon consistency tests.

Important negative cases:

- Object identity drift.
- Layout drift after leaving and revisiting a scene.
- Temporally plausible but semantically inconsistent futures.
- Revisit rollouts that change persistent scene state.

## Method

### Stage 1: Evaluator-Only Benchmark

Generate candidate rollouts from NanoWM and evaluate them with several scoring methods:

- Pixel-level metrics: LPIPS, PSNR, FVD where applicable.
- Task oracle: environment reward or success label.
- UMM evaluator: closed-source or open-source multimodal judge.
- Human preference labels on a small validation set.

The key measurement is correlation with oracle success and human preference.

### Stage 2: Hard Negative Construction

For each rollout, create controlled corruptions:

- Action shuffle: keep video plausible but mismatch actions.
- Temporal shuffle: permute or reverse frames.
- Goal swap: use a different goal instruction or goal image.
- Object swap: replace or perturb target object identity where feasible.
- Layout drift: alter persistent object positions across long rollouts.
- False success: final frame appears near success, but intermediate dynamics are implausible.
- No-op hallucination: large action sequence causes little or no state change.

The evaluator should assign lower scores to these negatives than to valid rollouts.

### Stage 3: Reward-Guided Planning

Use the evaluator score inside a planning loop:

1. Sample candidate action sequences.
2. Roll out each sequence with NanoWM.
3. Score each generated rollout with the UMM evaluator.
4. Execute the highest-scoring action sequence in the real or simulated environment.
5. Compare task success against baseline planners.

Baselines:

- Random action selection.
- Original NanoWM planning score.
- Pixel-metric-based rollout selection.
- Oracle reward upper bound, when available.

## Evaluation Metrics

### Evaluator Quality

- Spearman correlation with true environment reward.
- AUROC for success versus failure rollout classification.
- Pairwise preference accuracy on valid rollout versus hard negative.
- Calibration error between evaluator confidence and correctness.
- Failure-mode detection accuracy for action mismatch, temporal mismatch, and identity drift.

### Planning Utility

- Task success rate.
- Average environment reward.
- Number of model rollouts needed per successful episode.
- Robustness under longer horizons.
- Performance drop under distribution shift.

## Candidate Tasks

Start simple, then increase realism:

1. PointMaze or low-dimensional control with rendered observations.
2. PushT-style manipulation.
3. RT-1-style robot video tasks if data and runtime permit.
4. Long-horizon revisit scenarios inspired by Echo-Memory.

## Model Choices

### Teacher Evaluators

Use strong UMMs/VLMs for initial scoring and label generation:

- GPT-4o or equivalent multimodal judge.
- Gemini multimodal judge.
- Qwen2.5-VL or similar open-source VLM.
- Existing robotic reward models such as RoboReward or ROBOMETER if compatible.

### Student Evaluator

After collecting pairwise preference data, distill to a smaller open-source VLM reward model:

- Input: frames/video, instruction, actions, optional goal.
- Training objective: pairwise preference loss or scalar reward regression.
- Output: scalar reward plus diagnostic heads.

## Main Hypotheses

1. UMM reward correlates better with true task success than pixel metrics.
2. UMM reward detects action mismatch and temporal inconsistency better than frame-level video metrics.
3. UMM reward improves rollout reranking and planning success when used inside NanoWM.
4. Memory-inspired hard negatives expose evaluator weaknesses that ordinary video-quality metrics miss.

## Risks

### Reward Hacking

The evaluator may reward videos that look successful while hiding impossible dynamics or action mismatch.

Mitigation:

- Include action-shuffle and false-success negatives.
- Require sub-score explanations.
- Evaluate against true environment execution, not only generated videos.

### Weak Temporal Reasoning

Many VLMs are strong on single images but weak on action-conditioned temporal causality.

Mitigation:

- Use multi-frame or video input.
- Test frame order sensitivity.
- Include explicit before/after and action-consistency prompts.

### Cost

Closed-source UMM scoring can be expensive for planner loops.

Mitigation:

- Use closed-source UMM only as a teacher.
- Cache rollout scores.
- Distill a smaller student evaluator.
- Use UMM reranking only for top-k candidates from a cheaper heuristic.

### Dataset Bias

Evaluator may learn visual priors rather than task causality.

Mitigation:

- Use counterfactual negatives.
- Keep action and instruction mismatches balanced.
- Report per-failure-mode performance.

## Minimal Viable Experiment

1. Set up NanoWM on one simple task.
2. Generate 100-500 candidate rollouts with known environment rewards.
3. Build hard negatives from those rollouts.
4. Score rollouts with one strong UMM/VLM evaluator.
5. Compute correlation, AUROC, and pairwise preference accuracy.
6. Run a small reranking experiment: choose the best action sequence according to UMM reward and compare against baseline selection.

Success criterion:

- UMM reward has higher correlation with environment reward than pixel metrics.
- UMM reward identifies at least action-shuffle and temporal-shuffle negatives reliably.
- UMM-reranked action sequences improve task success or average reward over baseline rollout selection.

## Expected Contribution

This project reframes world-model evaluation from pixel prediction quality to interactive usefulness. The contribution is not a new large world model, but a reward/evaluation layer that can diagnose, select, and eventually train better world-model rollouts.

The most publishable angle is the combination of:

- A rollout evaluator benchmark with hard negatives.
- A UMM-based reward model for world-model outputs.
- Evidence that evaluator-guided reranking improves planning.

## Next Steps

1. Clone and run NanoWM on the smallest supported task.
2. Identify the easiest task with both generated rollout video and oracle reward.
3. Define the evaluator prompt and output schema.
4. Generate the first valid/negative rollout dataset.
5. Run initial UMM scoring and correlation analysis.
