# EFV ICLR Gap Report

- evidence stack passed: `false`
- total benchmarks: 1 / 3
- executable layers: 1 / 2
- diagnostic layers: 0 / 1
- failed checks: total_passed_benchmarks, executable_layers, diagnostic_layers, robotwin2_second_layer_present

## Benchmark Gaps

| Benchmark | Layer | Gate | Priority | Blockers | Next action |
| --- | --- | --- | --- | --- | --- |
| RoboCasa365 | executable_primary | pass | none | - | Keep this benchmark frozen unless upstream result files change. |
| RoboTwin2 | executable_second | fail | high | registry_status, cases, tasks, selector_margin, evidence_card | Run the bounded 4-task RoboTwin2 window and update the registry only after the paper-readiness gate passes. |
| MiraBench | world_model_diagnostic | fail | high | registry_status, cases, tasks, oracle_headroom, selector_margin, shortcut_controls, evidence_card | Instantiate a public diagnostic manifest with oracle judgments, visual/model-score proxy failures, and verifier margin. |
| RoboTrustBench | trust_diagnostic | fail | high | registry_status, oracle_headroom, selector_margin, shortcut_controls, evidence_card | Instantiate a public diagnostic manifest with oracle judgments, visual/model-score proxy failures, and verifier margin. |
| RoboWM-Bench | robustness_diagnostic | fail | high | registry_status, cases, oracle_headroom, selector_margin, shortcut_controls, evidence_card | Instantiate a public diagnostic manifest with oracle judgments, visual/model-score proxy failures, and verifier margin. |

## Next Actions

- First close RoboTwin2 with the bounded sequential launcher; this is the missing second executable layer.
- Then close one 2026 diagnostic layer by converting public judgments into a multi-candidate EFV selector table.
