# World-Model Bridge Gate

- passed: `false`
- claim level: `executable_with_pending_world_model_bridge`
- allowed claim: EFV has executable-benchmark evidence and a pending world-model diagnostic bridge; do not call it validated on world-model benchmarks yet.

## Checks

| Check | Status | Detail |
| --- | --- | --- |
| `primary_executable_mechanism` | pass | RoboCasa365 / min 1 |
| `second_executable_pressure` | pass | RoboTwin2 |
| `diagnostic_artifact_pipeline` | fail | - / min 1 |
| `diagnostic_registry` | fail | - / min 1 |
| `world_model_candidate_fields` | fail | - |

## Diagnostic Artifact Status

| Benchmark | Status | Pipeline | Registry | Missing |
| --- | --- | --- | --- | --- |
| MiraBench | blocked_public_artifacts | no | no | cases, tasks, oracle_headroom, method_margin, shortcut_controls, passed_registry_proposal, public_multi_candidate_records, judgment_labels, model_score_proxy |
| RoboTrustBench | adapter_or_subset_only | no | no | oracle_headroom, method_margin, shortcut_controls, passed_registry_proposal, full_or_paper_scale_release, multi_candidate_judgments |
| RoboWM-Bench | adapter_or_subset_only | no | no | oracle_headroom, method_margin, shortcut_controls, passed_registry_proposal, pinned_official_replay_patch, gt_replay_ceiling |

## Next Actions

- Obtain or construct multi-candidate judgment/proxy artifacts for MiraBench, RoboTrustBench, RoboWM-Bench.
- Run world_model_diagnostic_pipeline and require EFV to beat the visual/model-score proxy with full coverage.
- Require diagnostic manifests to include generator/model name, model score or visual proxy, and oracle judgment labels.

## Prohibited Claims

- Do not claim a new world model or policy.
- Do not claim real-robot validation.
- Do not title the paper as a world-model benchmark result yet.
- Do not claim validation on MiraBench, RoboTrustBench, or RoboWM-Bench until a diagnostic registry entry passes.
- Do not use visual plausibility or model-score-only artifacts as oracle success labels.
