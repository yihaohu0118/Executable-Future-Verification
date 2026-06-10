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

The important control is that candidate-ID lookup collapses to 0/15 after
anonymous remapping, while trace-based selectors remain above rank0 and simple
heuristics.

## Repository Layout

- `docs/proposal.md`: active paper proposal and current evidence.
- `docs/benchmark_expansion_roadmap.md`: benchmark status, results, and next
  experiments.
- `docs/robotwin2_executable_future_adapter.md`: RoboTwin2 setup, trace
  adapter, K=5 results, selector controls.
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

Run the multi-seed anonymous rank/candidate-ID sweep:

```bash
PYTHONPATH=src python -m umm_reward_evaluator.benchmarks.robotwin2_rank_randomization_sweep \
  --manifest /path/to/robotwin2_manifest.jsonl \
  --output /path/to/rankrand_sweep.json \
  --num-seeds 10 \
  --mode failure_rank0_shuffle_rest \
  --remap-candidate-ids
```

Randomize rank0 and anonymize candidate IDs for shortcut controls:

```bash
PYTHONPATH=src python -m umm_reward_evaluator.benchmarks.randomize_planner_rank \
  --manifest /path/to/robotwin2_manifest.jsonl \
  --output /path/to/robotwin2_rankrand_remap.jsonl \
  --mode failure_rank0_shuffle_rest \
  --seed 0 \
  --remap-candidate-ids
```

Run the local test suite:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Current Next Steps

1. Run RoboTwin2 anonymous candidate-ID/rank randomization over multiple seeds.
2. Add compact EEF/contact-direction features for the `open_laptop` boundary.
3. Build candidate pools where the successful future is not always the full
   expert trace.
4. Add RoboTwin2 K-shot calibration curves.
5. Keep RoboWM-Bench as a world-model-specific diagnostic layer until its
   public evaluator ceiling is clarified.

## Rename Status

Recommended future repository name:

```text
Executable-Future-Verification
```

The package name remains `umm_reward_evaluator` for now so existing scripts,
remote commands, and pushed experiment code keep working. See
`docs/repository_reorganization_plan.md` for the staged migration.
