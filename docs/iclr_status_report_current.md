# EFV ICLR Status Report

- claim level: `single_benchmark_mechanism`
- evidence stack passed: `false`
- passed benchmarks: RoboCasa365
- failed checks: total_passed_benchmarks, executable_layers, diagnostic_layers, robotwin2_second_layer_present

## Benchmark Status

| Benchmark | Year | Layer | Registry | Gate | Cases | Tasks | Rank0 | Oracle | Method | Baseline | Card |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RoboCasa365 | 2026 | executable_primary | passed | pass | 64 | 4 | 0 | 64 | 63.6 | 31.0 | valid |
| RoboTwin2 | 2025 | executable_second | pending | fail | 9 | 3 | 0 | 9 | 0 | 0 | missing_or_invalid |
| MiraBench | 2026 | world_model_diagnostic | pending | fail | 0 | 0 | 0 | 0 | 0 | 0 | missing_or_invalid |
| RoboTrustBench | 2026 | trust_diagnostic | pending | fail | 40 | 4 | 0 | 0 | 0 | 0 | missing_or_invalid |
| RoboWM-Bench | 2026 | robustness_diagnostic | pending | fail | 10 | 1 | 7 | 7 | 0 | 0 | missing_or_invalid |

## Allowed Claims

- RoboCasa365 supports a strong mechanism claim: shortcut-controlled execution-envelope features recover futures that rank0, action-only, and object-only controls miss.
- The project should be described as strong RoboCasa365 evidence with RoboTwin2 and world-model diagnostic evidence still in progress.

## Prohibited Claims

- Do not claim real-robot deployment or sim-to-real validation.
- Do not claim a new world model or robot policy; the contribution is candidate future verification.
- Do not claim validated performance across multiple mainstream benchmarks until RoboTwin2 and a diagnostic layer pass their gates.
- Do not use RoboTwin2 as the second main executable benchmark yet.
- Do not claim validation on world-model/trust diagnostics yet.

## Next Actions

- Close the RoboTwin2 paper-readiness gate: at least four base-ready tasks, anti-template successes, matched low-DTW negatives, selector margins, and one relation-rescue task.
- Instantiate one public diagnostic layer with multi-candidate judgments, proxy-score failure, and an EFV selector table beating the visual/model-score proxy.
- Update the evidence registry only after benchmark-level gates pass from current result files.
