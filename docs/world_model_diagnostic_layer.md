# World-Model Diagnostic Layer

Date checked: 2026-06-13

Detailed availability snapshot:
`docs/world_model_benchmark_availability_20260613.md`.

Machine-readable closure plan:
`docs/world_model_diagnostic_closure_plan_current.md`.

Regenerate it with:

```bash
python -m umm_reward_evaluator.benchmarks.world_model_diagnostic_closure_plan \
  --evidence-json docs/iclr_evidence_stack_registry.json \
  --output-json docs/world_model_diagnostic_closure_plan_current.json \
  --output-md docs/world_model_diagnostic_closure_plan_current.md
```

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

If benchmark prompt metadata and candidate judgments are stored separately,
merge them first. This is the expected path for RoboTrustBench-style releases
where the prompt/image metadata and model evaluation rows may be separate files:

```bash
python -m umm_reward_evaluator.benchmarks.world_model_diagnostic_merge_judgments \
  --requests RUN_ROOT/robotrustbench_generation_requests.jsonl \
  --judgments RUN_ROOT/judgments/model_outputs_with_scores.jsonl \
  --output-records RUN_ROOT/manifests/diagnostic_records.jsonl \
  --output-manifest RUN_ROOT/manifests/diagnostic_manifest.jsonl \
  --output-summary RUN_ROOT/manifests/diagnostic_manifest.summary.json \
  --default-benchmark RoboTrustBench \
  --default-suite trustworthiness_subset \
  --default-verification-target trustworthiness
```

The judgment rows should contain one row per candidate future with fields such
as `sample_id`, `candidate_id`, `video_path`, `label` or `judgment`,
`visual_quality_score` or `model_score`, and non-label verifier features such
as `motion_consistency`, `action_following_score`, or `constraint_score`.

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

If the manifest does not already contain `metadata.efv_score`, generate it with
the leave-one-case-out diagnostic calibrator. The calibrator uses labels from
other cases only, writes a verifier score back into candidate metadata, and
reports whether the calibrated selector beats rank0 before the final gates:

```bash
python -m umm_reward_evaluator.benchmarks.world_model_diagnostic_calibrate_verifier \
  --manifest RUN_ROOT/manifests/diagnostic_manifest.jsonl \
  --output-manifest RUN_ROOT/manifests/diagnostic_calibrated_manifest.jsonl \
  --output-summary-json RUN_ROOT/selectors/world_model_diagnostic_calibrated_verifier.json \
  --output-summary-md RUN_ROOT/selectors/world_model_diagnostic_calibrated_verifier.md \
  --score-key metadata.efv_score \
  --feature-key metadata.motion_consistency \
  --feature-key metadata.action_following_score \
  --categorical-key metadata.scenario \
  --no-action-stats
```

Do not use label-derived fields such as `label`, `judgment`, `verdict`,
`oracle_success`, or human annotation scores as verifier features. The calibrator
rejects label-like feature names by default because that would turn the
diagnostic layer into label leakage rather than future verification.

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

Then bind the manifest gate and selector table with the paper-level diagnostic
readiness gate:

```bash
python -m umm_reward_evaluator.benchmarks.world_model_diagnostic_readiness_gate \
  --diagnostic-gate-json RUN_ROOT/selectors/world_model_diagnostic_gate.json \
  --selector-table-json RUN_ROOT/selectors/world_model_diagnostic_selector_table.json \
  --verifier-selector verifier_score:metadata.efv_score \
  --output-json RUN_ROOT/selectors/world_model_diagnostic_readiness_gate.json \
  --output-md RUN_ROOT/selectors/world_model_diagnostic_readiness_gate.md
```

Only if this readiness gate passes should the registry proposal be generated
with `--diagnostic-readiness-json`. A passed manifest gate alone is not enough
for the diagnostic layer to count as the third benchmark.

For a completed diagnostic manifest, use the finalize script to run the full
CPU-only evidence chain:

```bash
PYTHONPATH=src PYTHON_BIN=python3 \
  CALIBRATE_EFV_SCORE=1 \
  CALIBRATION_FEATURE_KEYS="metadata.motion_consistency metadata.action_following_score" \
  CALIBRATION_CATEGORICAL_KEYS="metadata.scenario" \
  CALIBRATION_NO_ACTION_STATS=1 \
  scripts/world_model_diagnostic_finalize_run.sh \
  RUN_ROOT \
  RUN_ROOT/manifests/diagnostic_manifest.jsonl \
  MiraBench \
  2026 \
  world_model_diagnostic \
  metadata.efv_score \
  metadata.scenario metadata.failure_category
```

The script writes:

- `RUN_ROOT/selectors/world_model_diagnostic_gate.{json,md}`;
- `RUN_ROOT/selectors/world_model_diagnostic_selector_table.{json,md}`;
- `RUN_ROOT/selectors/world_model_diagnostic_readiness_gate.{json,md}`;
- `RUN_ROOT/selectors/world_model_diagnostic_registry_entry_proposal.{json,md}`;
- `RUN_ROOT/selectors/world_model_diagnostic_evidence_card_proposal.json`;
- `RUN_ROOT/selectors/world_model_diagnostic_evidence_card_validation.{json,md}`.

It does not edit `docs/iclr_evidence_stack_registry.json`. Inspect the
readiness gate, registry proposal, and evidence-card validation before counting
the diagnostic layer.

## How This Fits The EFV Story

RoboCasa365 and RoboTwin2 test executable action candidates in simulation.
MiraBench/RoboTrustBench test the same failure mode at the video-world-model
diagnostic level:

> generated futures can look plausible, but still be unfaithful to the action,
> physical constraint, counterfactual state, or safety condition.

The diagnostic layer is not a replacement for executable simulation. Its role
is to show that the same selection/verification problem appears in mainstream
world-model evaluation, where visual plausibility or model rank is not enough.
