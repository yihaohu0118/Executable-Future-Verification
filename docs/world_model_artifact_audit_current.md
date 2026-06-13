# World-Model Artifact Audit

- diagnostic benchmarks: 3
- pipeline-ready: 0
- registry-ready: 0
- blocked on public artifacts: 1
- paper rule: Do not count a world-model diagnostic benchmark unless registry_ready is true and the evidence card validates.

| Benchmark | Status | Note | Cases | Pipeline | Registry | Missing | Next action |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| MiraBench | blocked_public_artifacts | public_judgments_not_found | 0 | no | no | cases, tasks, oracle_headroom, method_margin, shortcut_controls, passed_registry_proposal, public_multi_candidate_records, judgment_labels, model_score_proxy | Watch for official annotation/result artifacts; run world_model_diagnostic_pipeline once public records exist. |
| RoboTrustBench | adapter_or_subset_only | prompt_subset_only | 40 | no | no | oracle_headroom, method_margin, shortcut_controls, passed_registry_proposal, full_or_paper_scale_release, multi_candidate_judgments | Use the public subset only to validate request generation; wait for full benchmark or collect candidate judgments before claiming results. |
| RoboWM-Bench | adapter_or_subset_only | conditional_unstable_replay | 10 | no | no | oracle_headroom, method_margin, shortcut_controls, passed_registry_proposal, pinned_official_replay_patch, gt_replay_ceiling | Keep as conditional robustness diagnostic until official replay/evaluator stability is pinned. |
