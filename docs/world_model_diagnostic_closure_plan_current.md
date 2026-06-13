# World-Model Diagnostic Closure Plan

- recommended order: MiraBench, RoboTrustBench, RoboWM-Bench
- paper rule: Do not count a diagnostic benchmark until its gate passes and the EFV selector beats the visual/model-score proxy.

| Priority | Benchmark | Status | Cases | Missing | Next action |
| ---: | --- | --- | ---: | --- | --- |
| 1 | MiraBench | blocked_public_artifacts | 0 | cases, action_only, candidate_id_or_rank_remap, energy_or_magnitude, oracle_judgment_labels, proxy_or_rank0_failure, random, rank0, visual_or_model_score_proxy | Wait for public judgment/result artifacts, then convert them into a multi-candidate diagnostic manifest. |
| 2 | RoboTrustBench | adapter_validation_only | 40 | action_only, candidate_id_or_rank_remap, energy_or_magnitude, oracle_judgment_labels, proxy_or_rank0_failure, random, rank0, visual_or_model_score_proxy, oracle_headroom | Use the public subset only to validate request generation and manifest plumbing; do not count it as a paper-level diagnostic. |
| 3 | RoboWM-Bench | conditional_robustness_diagnostic | 10 | action_only, candidate_id_or_rank_remap, energy_or_magnitude, oracle_judgment_labels, proxy_or_rank0_failure, random, rank0, visual_or_model_score_proxy, oracle_headroom | Keep this as a robustness diagnostic until GT replay/evaluator stability is resolved under a pinned official patch. |
