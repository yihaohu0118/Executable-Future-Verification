# Experiment Plan

## Phase 0: Repository Setup

Create a lightweight workspace with:

- documentation;
- dataset manifest format;
- evaluator prompt templates;
- scoring scripts;
- analysis notebooks or scripts.

No model training is required in this phase.

## Phase 1: Offline Rollout Evaluation

### Goal

Measure whether UMM rewards correlate with real task success better than visual
metrics.

### Tasks

Start with one simple NanoWM environment:

- PushT if available;
- otherwise PointMaze or another low-cost environment;
- optionally RT-1 for real-robot-like video once the pipeline works.

### Data

For each task instance:

- initial observation;
- goal or instruction;
- action sequence;
- generated rollout video;
- true environment reward or success;
- optional pixel metric scores.

### Evaluators

Candidate evaluators:

- GPT-4o or other strong closed-source multimodal model as teacher;
- Qwen2.5-VL or similar open-source VLM;
- RoboMeter/RoboReward-style robotic reward model if accessible;
- simple rule or CLIP-style baselines.

### Metrics

Report:

- Spearman correlation with oracle success;
- AUROC for success/failure classification;
- pairwise preference accuracy;
- calibration error;
- cost per scored rollout;
- latency per scored rollout.

### Baselines

Compare against:

- PSNR;
- LPIPS;
- FVD if enough videos are available;
- final-frame distance to goal;
- random score;
- task-specific oracle features if available.

## Phase 2: Hard Negative Benchmark

### Goal

Test whether the evaluator catches failures that pixel metrics miss.

### Negative Types

- action mismatch;
- temporal frame shuffle;
- identity swap;
- layout drift;
- false final-state success;
- contact or causality violation;
- no-op hallucination.

### Evaluation

For each positive rollout and corrupted negative rollout:

- ask evaluator to choose the better rollout;
- record pairwise accuracy;
- inspect failure cases with natural-language critiques.

## Phase 3: Reward-Guided Reranking

### Goal

Use UMM reward as a decision signal.

### Procedure

For each initial state:

1. sample multiple candidate action sequences;
2. generate NanoWM rollouts;
3. score each rollout with the evaluator;
4. choose the highest-scoring candidate;
5. execute or evaluate the chosen action sequence in the environment.

### Comparisons

Compare:

- original planner score;
- pixel-metric-based selection;
- UMM reward reranking;
- oracle selection upper bound.

### Metrics

Report:

- task success rate;
- average environment reward;
- regret relative to oracle selection;
- number of candidate rollouts needed;
- evaluation compute cost.

## Phase 4: Distilled Reward Model

### Goal

Replace expensive UMM calls with a smaller local reward model.

### Data

Use teacher-labeled rollout pairs:

- positive vs negative;
- high oracle reward vs low oracle reward;
- UMM-preferred vs UMM-rejected.

### Training Objective

Start with pairwise preference loss:

`L = -log sigmoid(r_good - r_bad)`

Then optionally add scalar regression to oracle rewards and sub-score
supervision.

## Minimal Deliverable

A convincing first version should contain:

1. A rollout manifest format.
2. At least one task with candidate rollouts and oracle success labels.
3. Prompted UMM scoring.
4. Correlation analysis against pixel metrics.
5. Hard-negative pairwise evaluation.

Planning integration can be the second milestone if the offline results are
strong.
