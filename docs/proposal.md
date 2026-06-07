# Proposal: UMM as Reward Evaluator for World Models

## Motivation

Action-conditioned world models are usually evaluated with visual metrics such
as PSNR, LPIPS, FVD, or next-frame prediction loss. These metrics are useful for
low-level reconstruction quality, but they often fail to measure the properties
that matter for planning and robot control:

- whether the rollout completes the task;
- whether the generated state changes are consistent with the action sequence;
- whether physical interactions are plausible;
- whether identity and layout remain stable over long horizons;
- whether generated futures are useful for selecting actions.

Unified multimodal models provide a natural evaluator because they can jointly
condition on language instructions, visual observations, action descriptions,
goal images, and generated rollout videos.

## Research Hypothesis

A UMM-based reward evaluator can score world-model rollouts in ways that better
correlate with downstream task success than pixel-level metrics, and the same
score can improve rollout selection or planning.

## System Definition

Given:

- task instruction;
- initial observation or context frames;
- optional goal image;
- action sequence;
- generated rollout video from a world model;

the evaluator predicts:

- scalar reward;
- task success score;
- progress score;
- action consistency score;
- temporal consistency score;
- physical plausibility score;
- memory or identity consistency score;
- optional natural-language critique.

## Base Implementation

Use `nano-world-model` as the initial rollout generator because it already
supports action-conditioned video rollout and planning-style evaluation.

Use ideas from `Echo-Memory` to design consistency stress tests:

- revisit consistency;
- object identity persistence;
- layout persistence;
- view or camera consistency;
- long-horizon drift.

The first implementation does not need to train a new UMM. It can use prompted
closed-source or open-source VLM/UMM evaluators as teachers, then optionally
distill them into a smaller reward model.

## Method

1. Generate candidate rollouts from NanoWM for the same initial state and task.
2. Create positive and hard-negative rollout pairs.
3. Score rollouts with candidate UMM evaluators.
4. Compare evaluator scores with oracle success and human preference labels.
5. Use evaluator rewards for reranking candidate actions or rollouts.

## Hard Negatives

The benchmark should include rollouts that look visually plausible but are bad
for planning:

- action shuffle: the video is plausible, but mismatched with the action;
- temporal shuffle: frames are reordered;
- object swap: target identity changes;
- layout drift: scene geometry changes over time;
- false success: final frame looks close to goal but the process is invalid;
- no-op hallucination: actions are large but state barely changes;
- contact violation: object moves before contact or ignores contact.

## Contributions

Potential paper contributions:

1. A benchmark for semantic and action-aware evaluation of world-model rollouts.
2. A UMM reward evaluator with interpretable sub-scores.
3. Evidence that UMM reward correlates better with task success than pixel
   metrics.
4. A planning or reranking demonstration showing improved downstream behavior.

## Risks

### Reward Hacking

The evaluator may reward videos that look successful but are not physically or
causally valid. Hard negatives and action mismatch tests are essential.

### Weak Temporal Reasoning

Many VLMs are stronger on static images than videos. The benchmark should test
multi-frame and action-conditioned reasoning explicitly.

### Cost

Closed-source UMMs are expensive for large-scale rollout scoring. A practical
system may require caching, pairwise scoring, or distillation.

### Domain Gap

UMM scores may work on visually rich data but fail on simple simulated states.
This should be tested across PushT, PointMaze, and a real-robot-like dataset
such as RT-1 if feasible.

## Recommended First Target

Start with a small benchmark and offline correlation study:

> Can UMM reward predict true task success of NanoWM rollouts better than PSNR,
> LPIPS, or FVD?

If this works, move to reward-guided rollout reranking and MPC/CEM planning.
