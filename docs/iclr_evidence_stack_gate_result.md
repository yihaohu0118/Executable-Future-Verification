# ICLR Evidence Stack Gate

| Check | Status | Detail |
| --- | --- | --- |
| `modern_scope` | pass | - |
| `total_passed_benchmarks` | fail | RoboCasa365 / min 3 |
| `executable_layers` | fail | RoboCasa365 / min 2 |
| `diagnostic_layers` | fail | - / min 1 |
| `primary_robocasa_present` | pass | RoboCasa365 |
| `robotwin2_second_layer_present` | fail | RoboTwin2 |

| Benchmark | Year | Layer | Status | Cases | Tasks | Rank0 | Oracle | Method | Best baseline | Margin | Missing controls | Evidence card |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| RoboCasa365 | 2026 | executable_primary | pass | 64 | 4 | 0.0 | 64.0 | 63.6 | 31.0 | 32.6 | - | valid |
| RoboTwin2 | 2025 | executable_second | fail | 9 | 3 | 0.0 | 9.0 | 0.0 | 0.0 | 0.0 | - | missing |
| MiraBench | 2026 | world_model_diagnostic | fail | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | action_only, candidate_id_or_rank_remap, energy_or_magnitude, oracle_judgment_labels, proxy_or_rank0_failure, random, rank0, visual_or_model_score_proxy | missing |
| RoboTrustBench | 2026 | trust_diagnostic | fail | 40 | 4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | action_only, candidate_id_or_rank_remap, energy_or_magnitude, oracle_judgment_labels, proxy_or_rank0_failure, random, rank0, visual_or_model_score_proxy | missing |
| RoboWM-Bench | 2026 | robustness_diagnostic | fail | 10 | 1 | 7.0 | 7.0 | 0.0 | 0.0 | 0.0 | action_only, candidate_id_or_rank_remap, energy_or_magnitude, oracle_judgment_labels, proxy_or_rank0_failure, random, rank0, visual_or_model_score_proxy | missing |

Interpretation:

- This is a paper-level gate, not a selector metric.
- A benchmark can be useful diagnostically while still failing this gate.
- The ICLR claim should not be phrased as multi-benchmark evidence until this gate passes with current result files.
