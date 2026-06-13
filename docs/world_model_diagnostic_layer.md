# World-Model Diagnostic Layer

Date checked: 2026-06-13

Detailed availability snapshot:
`docs/world_model_benchmark_availability_20260613.md`.

## Current Public Status

### MiraBench

MiraBench is the best conceptual fit for the third evidence layer because it
targets action-conditioned reliability directly. The paper reports:

- Physics Adherence;
- Action-Following Fidelity;
- Optimism Bias Detection;
- 906 generated videos;
- 16,704 structured human annotation decisions.

Current blocker: the arXiv HTML page does not expose an official code, project,
or dataset link. Treat MiraBench as the preferred diagnostic target once the
annotation corpus or result tables are publicly accessible.

### RoboTrustBench

RoboTrustBench currently has a public project page and a HuggingFace dataset
release:

- Project page: `https://huiqiongli.github.io/RoboTrustBench/`
- HuggingFace dataset: `Huiqiong0124/RoboTrustBench_Dataset`

The current HuggingFace release is a 40-sample subset, not the full benchmark.
It covers four scenario categories:

- Normal;
- Constraint-sensitive;
- Counterfactual;
- Adversarial.

The dataset card says the full dataset is coming soon. The 40-sample subset can
be used to validate our adapter and prompt-level evaluation pipeline, but it
should not be counted as a passed paper-level diagnostic benchmark.

Convert the prompt metadata into generation/evaluation requests with:

```bash
python -m umm_reward_evaluator.benchmarks.robotrustbench_requests \
  --metadata /path/to/RoboTrustBench_Dataset/metadata.jsonl \
  --image-root /path/to/RoboTrustBench_Dataset \
  --output-requests RUN_ROOT/robotrustbench_generation_requests.jsonl \
  --output-summary RUN_ROOT/robotrustbench_generation_requests.summary.json
```

This request file is not a result table. It is the input for generating
candidate futures and collecting human/MLLM reliability judgments. Only those
judgment rows should be converted into the shared diagnostic manifest.

## Required Diagnostic Manifest

Use `world_model_diagnostic_to_manifest.py` only after generated videos or
judgment rows are available. Each case should contain multiple candidate
futures for the same initial image/instruction, for example different video
models, decoding seeds, or verifier-selected candidates.

Required properties:

- at least two candidates per case;
- human, expert, or benchmark-provided reliability labels mapped to
  `oracle_success`;
- `planner_score` or `model_score` for the visual/model-score baseline;
- metadata category keys such as `metadata.scenario`, `metadata.criterion`,
  `metadata.dimension`, or `metadata.failure_category`;
- rank/candidate-ID controls if model order or generation ID could leak labels.

Then run:

```bash
python -m umm_reward_evaluator.benchmarks.world_model_diagnostic_gate \
  --manifest RUN_ROOT/manifests/diagnostic_manifest.jsonl \
  --category-key metadata.scenario \
  --category-key metadata.failure_category \
  --require-metadata-key metadata.scenario \
  --require-metadata-key metadata.verification_target \
  --output-json RUN_ROOT/selectors/world_model_diagnostic_gate.json \
  --output-md RUN_ROOT/selectors/world_model_diagnostic_gate.md
```

By default this gate requires the `planner_score`/`model_score` proxy to be
present for every case, to fail on at least one case, and to trail oracle-best
judgment by at least one successful case. This is intentional: if the visual or
model-score proxy already matches oracle judgment, the diagnostic does not
support the EFV claim that future selection is brittle.

Only update `docs/iclr_evidence_stack_registry.json` after this diagnostic gate
passes and the selector result beats the planner-score or visual-proxy baseline.

Build that selector table with:

```bash
python -m umm_reward_evaluator.benchmarks.world_model_diagnostic_selector_table \
  --manifest RUN_ROOT/manifests/diagnostic_manifest.jsonl \
  --verifier-score-key metadata.efv_score \
  --output-json RUN_ROOT/selectors/world_model_diagnostic_selector_table.json \
  --output-md RUN_ROOT/selectors/world_model_diagnostic_selector_table.md
```

The table reports rank0, random-expected, planner/model-score proxy,
`metadata.efv_score`, and oracle-best judgment success. The EFV score must beat
the planner/model proxy before the diagnostic benchmark should be counted in
the ICLR evidence stack.

## How This Fits The EFV Story

RoboCasa365 and RoboTwin2 test executable action candidates in simulation.
MiraBench/RoboTrustBench test the same failure mode at the video-world-model
diagnostic level:

> generated futures can look plausible, but still be unfaithful to the action,
> physical constraint, counterfactual state, or safety condition.

The diagnostic layer is not a replacement for executable simulation. Its role
is to show that the same selection/verification problem appears in mainstream
world-model evaluation, where visual plausibility or model rank is not enough.
