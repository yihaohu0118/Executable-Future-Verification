# EFV Reviewer Risk Audit

- verdict: `not_ready_for_strong_claim`
- open high risks: 0
- partial high risks: 3
- open risks: 3
- partially defended risks: 4

| Risk | Severity | Status | Evidence | Next action |
| --- | --- | --- | --- | --- |
| expert_template_matching | high | partially_defended | RoboTwin2 has oracle headroom, but anti-template pressure is not closed.<br>non_template_success_tasks=0, low_dtw_failed_negative_tasks=0, dtw_template_beaten_tasks=0 | Prioritize non-template successes and low-DTW failed negatives before adding easier tasks. |
| single_benchmark_overclaim | high | partially_defended | RoboCasa365 passes, but the second executable benchmark is still pending. | Do not claim multi-benchmark effectiveness until RoboTwin2 passes the paper-readiness gate. |
| weak_or_shortcut_baselines | high | partially_defended | RoboCasa365 includes rank0/random/energy/action-only/rank-remap controls. | Carry the same controls plus DTW nearest-positive into RoboTwin2 and diagnostics. |
| no_real_robot | medium | open | Only RoboCasa365 currently passes; no real robot or second executable benchmark is closed. | State the sim-only boundary explicitly; do not claim real-robot transfer without hardware evidence. |
| visual_plausibility_proxy | medium | open | No passed diagnostic currently shows EFV beating a visual/model-score proxy. | Report visual/model-score proxy selectors next to EFV on MiraBench/RoboTrustBench/RoboWM-style diagnostics. |
| world_model_relevance | medium | open | No world-model/trust diagnostic layer currently passes with oracle labels and proxy baseline controls. | Convert one 2025-2026 public diagnostic benchmark into a multi-candidate selector table. |
| system_artifact_or_partial_run | medium | partially_defended | Primary RoboCasa365 evidence card validates. | Require raw-integrity audits before manifest conversion and evidence-card validation before registry promotion. |
| counterintuitive_signal_strength | low | defended | RoboCasa365 method-vs-best-baseline margin=32.6 | Keep the counterintuitive claim tied to rank0/oracle headroom and shortcut-controlled negatives. |
