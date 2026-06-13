# ICLR Evidence Stack Gate

This gate tracks whether the project has enough current-benchmark evidence to
claim a main-conference executable-future verification story.

Run:

```bash
python -m umm_reward_evaluator.benchmarks.iclr_evidence_stack_gate \
  --evidence-json docs/iclr_evidence_stack_registry.json \
  --output-json docs/iclr_evidence_stack_gate_result.json \
  --output-md docs/iclr_evidence_stack_gate_result.md
```

The default gate requires:

| Requirement | Default |
| --- | ---: |
| Passed 2025-2026 benchmarks | 3 |
| Passed executable manipulation benchmarks | 2 |
| Passed world-model/trust diagnostic benchmark | 1 |
| Cases per passed benchmark | 16 |
| Tasks per passed executable benchmark | 4 |
| Method margin over best non-oracle baseline | 1.0 |

Required shortcut controls per passed benchmark:

- `rank0`;
- `random`;
- `energy_or_magnitude`;
- `action_only`;
- `candidate_id_or_rank_remap`.

## Current Expected Status

The gate should fail right now.

That is intentional. Current evidence is strong on RoboCasa365, promising but
not yet broad enough on RoboTwin2, and not yet instantiated on a public
world-model diagnostic benchmark. The next work that can make this gate pass is:

1. Complete the RoboTwin2 six-task primary window and pass
   `robotwin2_paper_readiness_gate.py`.
2. Import at least one public diagnostic layer, preferably MiraBench first,
   through `world_model_diagnostic_to_manifest.py`.
3. Validate the diagnostic manifest with `world_model_diagnostic_gate.py`.
4. Update `docs/iclr_evidence_stack_registry.json` only with result files that
   passed their own benchmark-level gates.

Until this gate passes, the project should be described as:

> strong RoboCasa365 evidence plus RoboTwin2/world-model evidence in progress

not as:

> validated across multiple mainstream benchmarks.
