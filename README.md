# Executable-Future Verification

This repository is a research workspace for executable-future verification in
robot world-action benchmarks.

The active question is:

> Given several proposed futures for the same robot task, which future is
> physically executable and task-completing?

The project started as a UMM reward-evaluator workspace, so the Python package
is still named `umm_reward_evaluator` for compatibility. The current research
direction is broader and more concrete: generator-agnostic candidate selection
over actions, robot traces, gripper/contact envelopes, optional videos, and
world-model outputs.

## Current Thesis

Future generation is becoming cheap, but future selection remains brittle.
Default rankers, action magnitude, visual plausibility, and averaged success
prototypes can all fail under shortcut-controlled negatives.

The current evidence supports a narrower claim:

> Few-shot task/contact-conditioned execution-envelope verification over compact
> robot traces can recover executable futures that rank0 and simple action
> heuristics miss.

## Active Benchmark Evidence

### RoboCasa365

RoboCasa365 is the main 2026 benchmark layer. Current n16 hard-negative results
show:

- rank0 conservative prior: 0/64;
- oracle-best: 64/64;
- action-only endpoint-free selector: 28.4/64;
- object-only trace selector: 31.0/64;
- EEF+gripper distribution-only selector: 63.6/64;
- four-shot target calibration: 59.2/64;
- eight-shot target calibration: 62.2/64.

This supports the mechanism that robot execution envelopes, not object-state
leakage or action magnitude, carry the useful signal.

### RoboTwin2

RoboTwin2 is the current 2025 executable manipulation layer. The three-task K=5
smoke has 15 cases and six candidates per case:

- rank0: 0/15;
- oracle-best: 15/15;
- uniform random expected: 4.17/15;
- best simple action heuristic: 6/15 in fixed order, 5/15 after anonymous
  candidate-ID/rank remap;
- nearest-positive gripper/phase/joint trace selectors: up to 13/15 in fixed
  order and 12.0/15 mean over 10 anonymous remap seeds.
- K-shot calibration on anonymous remap shows a clear boundary: phase-joint
  source-only is 0.0/15, K=2 reaches 6.2/15, and K=4 reaches 12.0/15.
- DTW nearest-positive trajectory-distance control reaches 14.0/15 with
  joint+gripper traces under the same anonymous remap protocol.
- Anti-template diagnostics show the current pool has 10/15 nominal non-full
  successes, but 0/15 diverse non-full successes under joint+gripper DTW and
  0/15 matched low-DTW negative cases.
- A new anti-template K=5 run fixes the hard-positive side: combined
  `stack_blocks_two`, `open_laptop`, and `stamp_seal` now have rank0 0/15,
  oracle 15/15, diverse non-full successes 14/15, and matched low-DTW
  negatives 6/15. DTW is no longer oracle but remains strong at 13-13.5/15.
- A follow-up `targeted_hard` preset adds contact-phase time warps,
  contact-offset perturbations, and gripper-contact pulses. A `stamp_seal`
  seed-0 smoke produced both nearby successes and nearby failures, which is the
  setting needed to test whether selectors beat expert-template matching.
- The `stamp_seal` K=5 targeted-hard run keeps rank0 at 0/5 and oracle at 5/5,
  with 5/5 diverse non-full successes and 5/5 matched low-DTW negatives, but
  energy/length heuristics still reach 5/5. The next preset,
  `targeted_energy_matched`, adds long failed probes to remove that shortcut.
- The `targeted_energy_matched` seed-0 smoke confirms the intended control:
  `energy_sum_max` and `length_max` both select long failed probes rather than
  the successful contact-repeat trajectory.
- The current official multitask `targeted_energy_matched` analysis now drops
  any case containing `candidate_error` rows, so CUDA OOM or simulator errors
  are not counted as physical hard negatives. On the clean complete subset,
  `stack_blocks_two` has rank0 0/2 and oracle 2/2, while energy, smoothness,
  action-distribution, gripper nearest-positive, DTW-gripper, and
  DTW-joint+gripper selectors all stay at 0/2. This is the strongest current
  diagnostic that gripper-only verification is not sufficient for multi-stage
  stacking.
- On the same clean protocol, `open_laptop` has rank0 0/2 and oracle 2/2, but
  smoothness, gripper distribution, and DTW-gripper reach 2/2. This task is
  useful as a permissiveness counterexample, not a headline benchmark result.

The important control is that candidate-ID lookup collapses to 0/15 after
anonymous remapping, while trace-based selectors remain above rank0 and simple
heuristics.

The DTW control is also a warning: the current RoboTwin2 pool can largely be
solved by expert-trajectory similarity. It is useful as a mechanism diagnostic,
but it is not yet strong enough as a main-table claim that the verifier has
learned executability beyond template matching.

## Repository Layout

- `docs/proposal.md`: active paper proposal and current evidence.
- `docs/iclr_execution_envelope_decision_20260613.md`: bounded ICLR direction
  decision after reviewer-risk reassessment.
- `docs/benchmark_expansion_roadmap.md`: benchmark status, results, and next
  experiments.
- `docs/robotwin2_executable_future_adapter.md`: RoboTwin2 setup, trace
  adapter, K=5 results, selector controls.
- `docs/robotwin2_antitemplate_k5_results.md`: latest anti-template RoboTwin2
  K=5 results and interpretation.
- `docs/reviewer_risk_antitemplate_plan.md`: reviewer-risk assessment and the
  next anti-template experiments required for a stronger paper claim.
- `docs/future_verification_manifest_protocol.md`: shared candidate JSONL
  protocol.
- `docs/repository_reorganization_plan.md`: staged rename and architecture
  plan.
- `docs/umm_reward_evaluator_proposal.md`: archived legacy UMM/NanoWM proposal.
- `src/umm_reward_evaluator/benchmarks/`: current benchmark adapters,
  validators, selector baselines, and controls.
- `tests/`: lightweight tests for manifest conversion, controls, and selector
  baselines.

## Quick Start

Install the local package:

```bash
pip install -e .
```

Validate a candidate-future manifest:

```bash
PYTHONPATH=src python -m umm_reward_evaluator.benchmarks.validate_future_verification_manifest \
  --manifest /path/to/candidates.jsonl \
  --require-future-metadata
```

Run RoboTwin2 pure-numpy selector baselines:

```bash
PYTHONPATH=src python -m umm_reward_evaluator.benchmarks.robotwin2_selector_baselines \
  --manifest /path/to/robotwin2_manifest.jsonl \
  --output-dir /path/to/selector_outputs
```

The selector suite includes action, gripper, joint, object-pose, and
object-relation features. The relation features are intended for the
`stack_blocks_two` failure mode where gripper-only and DTW-gripper selectors
can fail even though oracle headroom exists:

```bash
PYTHONPATH=src python -m umm_reward_evaluator.benchmarks.robotwin2_selector_baselines \
  --manifest /path/to/robotwin2_manifest.jsonl \
  --output-dir /path/to/relation_selector_outputs \
  --prototype-feature object_relation_distribution \
  --prototype-feature phase_object_relation_distribution \
  --prototype-feature phase_object_relation_joint_gripper_distribution \
  --trace-distance-feature dtw_object_relation \
  --trace-distance-feature dtw_object_relation_joint_gripper
```

Prototype and trace-distance outputs include `feature_coverage`. Do not use an
object or object-relation selector in a main table unless all main-table cases
have complete required trace keys.

Check whether a RoboTwin2 manifest is ready for a main table:

```bash
PYTHONPATH=src python -m umm_reward_evaluator.benchmarks.robotwin2_main_table_gate \
  --manifest /path/to/robotwin2_manifest.jsonl \
  --required-candidates-per-case 24 \
  --min-cases 5 \
  --min-oracle-better-cases 5 \
  --require-feature object_relation_distribution \
  --require-feature phase_object_relation_joint_gripper_distribution
```

`scripts/robotwin2_multitask_analysis.sh` writes these gate reports
automatically for each task:

- `selectors/<task>_targeted_energy_matched_main_table_gate.json`
- `selectors/<task>_targeted_energy_matched_relation_gate.json`
- `selectors/robotwin2_readiness_report.{json,md}`

Run the multi-seed anonymous rank/candidate-ID sweep:

```bash
PYTHONPATH=src python -m umm_reward_evaluator.benchmarks.robotwin2_rank_randomization_sweep \
  --manifest /path/to/robotwin2_manifest.jsonl \
  --output /path/to/rankrand_sweep.json \
  --num-seeds 10 \
  --mode failure_rank0_shuffle_rest \
  --remap-candidate-ids
```

Use repeated `--heuristic` and `--prototype-config feature:scope:mode` flags to
override the default selector list for diagnostic sweeps.

Randomize rank0 and anonymize candidate IDs for shortcut controls:

```bash
PYTHONPATH=src python -m umm_reward_evaluator.benchmarks.randomize_planner_rank \
  --manifest /path/to/robotwin2_manifest.jsonl \
  --output /path/to/robotwin2_rankrand_remap.jsonl \
  --mode failure_rank0_shuffle_rest \
  --seed 0 \
  --remap-candidate-ids
```

Run the K-shot target-task calibration sweep:

```bash
PYTHONPATH=src python -m umm_reward_evaluator.benchmarks.robotwin2_kshot_calibration_sweep \
  --manifest /path/to/robotwin2_manifest.jsonl \
  --output /path/to/kshot_sweep.json \
  --num-rank-seeds 10 \
  --num-support-seeds 5 \
  --remap-candidate-ids
```

Run anti-template diagnostics:

```bash
PYTHONPATH=src python -m umm_reward_evaluator.benchmarks.robotwin2_antitemplate_diagnostics \
  --manifest /path/to/robotwin2_manifest.jsonl \
  --output-json /path/to/antitemplate_diagnostics.json \
  --output-md /path/to/antitemplate_diagnostics.md \
  --feature-mode dtw_joint_gripper
```

Generate RoboTwin2 traces with anti-template candidate probes from an official
RoboTwin checkout:

```bash
PYTHONPATH=/path/to/Executable-Future-Verification/src python -m umm_reward_evaluator.benchmarks.robotwin2_gripper_aware_trace \
  --task-name stack_blocks_two \
  --task-config demo_clean_smoke \
  --seeds 0-4 \
  --max-seeds 5 \
  --output-dir /tmp/robotwin2_antitemplate/stack_blocks_two \
  --candidate-preset anti_template
```

Use `--candidate-preset targeted_hard` for the newer near-neighbor
success/failure probes. Use `--candidate-preset targeted_energy_matched` when
testing whether length and energy shortcuts survive longer failed futures.

Run the local test suite:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Current Next Steps

1. Run RoboTwin2 anonymous candidate-ID/rank randomization over multiple seeds.
2. Build candidate pools where the successful future is not always the full
   expert trace.
3. Add hard positives: successful but non-expert-like futures with different
   timing, intermediate poses, or contact strategies.
4. Add matched hard negatives: similar joint/gripper traces that fail because
   of contact direction, closing timing, or task constraints.
5. Scale the RoboTwin2 `targeted_hard` preset beyond the current one-case smoke
   and compare against DTW/template baselines.
6. Extend clean `targeted_energy_matched` RoboTwin2 coverage only when GPU
   capacity is available without stopping user training jobs. Main analysis
   must drop incomplete and `candidate_error` cases.
7. Keep RoboWM-Bench as a world-model-specific diagnostic layer until its
   public evaluator ceiling is clarified.

## Rename Status

Recommended future repository name:

```text
Executable-Future-Verification
```

The package name remains `umm_reward_evaluator` for now so existing scripts,
remote commands, and pushed experiment code keep working. See
`docs/repository_reorganization_plan.md` for the staged migration.
