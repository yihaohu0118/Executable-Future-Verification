# ICLR Evidence Stack Gate

This gate tracks whether the project has enough current-benchmark evidence to
claim a main-conference executable-future verification story.

Run:

```bash
python -m umm_reward_evaluator.benchmarks.iclr_evidence_stack_gate \
  --evidence-json docs/iclr_evidence_stack_registry.json \
  --require-evidence-cards \
  --output-json docs/iclr_evidence_stack_gate_result.json \
  --output-md docs/iclr_evidence_stack_gate_result.md
```

Generate the claim-level report with:

```bash
python -m umm_reward_evaluator.benchmarks.iclr_claim_report \
  --evidence-json docs/iclr_evidence_stack_registry.json \
  --output-json docs/iclr_claim_report_result.json \
  --output-md docs/iclr_claim_report_result.md
```

The claim report is the safer artifact to read before writing paper text: it
translates gate status into allowed claims, prohibited claims, and the next
missing evidence.

Generate the current one-page status report with:

```bash
python -m umm_reward_evaluator.benchmarks.iclr_status_report \
  --evidence-json docs/iclr_evidence_stack_registry.json \
  --require-evidence-cards \
  --output-json docs/iclr_status_report_current.json \
  --output-md docs/iclr_status_report_current.md
```

This is the best artifact to send to collaborators because it combines the
stack gate, claim guard, benchmark rows, and evidence-card status.

For a Chinese decision brief, use:

```text
docs/iclr_boss_brief_zh_20260613.md
```

After a benchmark-level gate passes, generate a registry-entry proposal instead
of editing `docs/iclr_evidence_stack_registry.json` by hand:

```bash
python -m umm_reward_evaluator.benchmarks.iclr_registry_proposal robotwin2 \
  --readiness-json RUN_ROOT/selectors/robotwin2_readiness_report.json \
  --selector-table-json RUN_ROOT/selectors/robotwin2_selector_table.json \
  --paper-gate-json RUN_ROOT/selectors/robotwin2_paper_readiness_gate.json \
  --output-json RUN_ROOT/selectors/robotwin2_registry_entry_proposal.json
```

For world-model diagnostics:

```bash
python -m umm_reward_evaluator.benchmarks.iclr_registry_proposal diagnostic \
  --benchmark MiraBench \
  --year 2026 \
  --layer world_model_diagnostic \
  --diagnostic-gate-json RUN_ROOT/selectors/world_model_diagnostic_gate.json \
  --selector-table-json RUN_ROOT/selectors/world_model_diagnostic_selector_table.json \
  --verifier-selector verifier_score:metadata.efv_score \
  --shortcut-control energy_or_magnitude \
  --shortcut-control action_only \
  --shortcut-control candidate_id_or_rank_remap \
  --output-json RUN_ROOT/selectors/mirabench_registry_entry_proposal.json
```

The proposal tool remains conservative: failed gates, missing diagnostic
controls, or verifier scores that do not beat the visual/model-score proxy
produce `status: pending`.

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

World-model/trust diagnostic benchmarks additionally require:

- `oracle_judgment_labels`;
- `proxy_or_rank0_failure`;
- `visual_or_model_score_proxy`.

These extra controls mean the diagnostic layer must show a real selection gap:
visual/model-score ranking or rank0 must fail on some cases, and benchmark or
human judgment labels must expose an oracle candidate that can recover them.

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
