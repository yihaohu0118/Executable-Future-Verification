# EFV ICLR Boss Dashboard

- verdict: **Worth continuing in a bounded window: the mechanism is strong, but the multi-benchmark evidence is not closed.**
- maturity: `focused_push`
- evidence score: 40/100
- claim level: `single_benchmark_mechanism`
- evidence stack passed: `false`

## Core Signal

- status: `strong`
- summary: Rank0 fails while oracle succeeds, and compact execution envelopes beat the strongest non-oracle control.
- RoboCasa365 numbers: rank0=0.0, oracle=64.0, method=63.6, best baseline=31.0, margin=32.6

## Benchmark Coverage

- total: 1 / 3
- executable: 1 / 2
- diagnostic: 0 / 1
- passed: RoboCasa365

## Top Blockers

| Benchmark | Layer | Priority | Blockers | Next action |
| --- | --- | --- | --- | --- |
| RoboTwin2 | executable_second | high | registry_status, cases, tasks, selector_margin, evidence_card | Run the bounded 4-task RoboTwin2 window and update the registry only after the paper-readiness gate passes. |
| MiraBench | world_model_diagnostic | high | registry_status, cases, tasks, oracle_headroom, selector_margin, shortcut_controls, evidence_card | Instantiate a public diagnostic manifest with oracle judgments, visual/model-score proxy failures, and verifier margin. |
| RoboTrustBench | trust_diagnostic | high | registry_status, oracle_headroom, selector_margin, shortcut_controls, evidence_card | Instantiate a public diagnostic manifest with oracle judgments, visual/model-score proxy failures, and verifier margin. |

## Next Priority

- Close RoboTwin2 first; it is the missing second executable benchmark.

## Allowed Claims

- RoboCasa365 supports a strong mechanism claim: shortcut-controlled execution-envelope features recover futures that rank0, action-only, and object-only controls miss.
- The project should be described as strong RoboCasa365 evidence with RoboTwin2 and world-model diagnostic evidence still in progress.

## Prohibited Claims

- Do not claim real-robot deployment or sim-to-real validation.
- Do not claim a new world model or robot policy; the contribution is candidate future verification.
- Do not claim validated performance across multiple mainstream benchmarks until RoboTwin2 and a diagnostic layer pass their gates.
- Do not use RoboTwin2 as the second main executable benchmark yet.
- Do not claim validation on world-model/trust diagnostics yet.

## Kill Or Downgrade Triggers

- RoboTwin2 has fewer than four clean oracle-headroom tasks.
- DTW nearest-positive stays within one success of the best envelope verifier.
- Successful candidates remain mostly full expert traces or unknown-source variants.
- No world-model/trust diagnostic layer can produce a public multi-candidate judgment table.
