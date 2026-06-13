# World-Model Benchmark Availability Snapshot

Date checked: 2026-06-13. Artifact audit refreshed on 2026-06-14.

This note records what is actually usable for the third EFV evidence layer.
The goal is to avoid counting a benchmark name before its public artifacts can
support a reproducible selector table.

Current machine-readable audit:

- `docs/world_model_artifact_audit_current.json`
- `docs/world_model_artifact_audit_current.md`

Regenerate it with:

```bash
python -m umm_reward_evaluator.benchmarks.world_model_artifact_audit \
  --evidence-json docs/iclr_evidence_stack_registry.json \
  --output-json docs/world_model_artifact_audit_current.json \
  --output-md docs/world_model_artifact_audit_current.md
```

As of the 2026-06-14 audit, no world-model diagnostic layer is paper-countable:
MiraBench is blocked on public judgment artifacts, RoboTrustBench is still a
prompt-subset / adapter-validation path, and RoboWM-Bench remains conditional
on official replay/evaluator stability.

## MiraBench

- Paper: `https://arxiv.org/abs/2605.29360`
- Status: conceptually best fit, but not yet runnable from public artifacts.
- The paper reports a human-annotated corpus with 906 generated videos and
  16,704 structured annotation decisions across physics, action-following, and
  optimism-bias modules.
- The current arXiv page does not expose an official project, code, or dataset
  link that can be converted into our manifest.

Decision: keep MiraBench as the preferred third-layer target, but do not count
it in the ICLR evidence stack until the annotation corpus or official result
tables are public and pass `world_model_diagnostic_gate.py`.

## RoboTrustBench

- Paper: `https://arxiv.org/abs/2606.01600`
- Project: `https://huiqiongli.github.io/RoboTrustBench/`
- Dataset: `https://huggingface.co/datasets/Huiqiong0124/RoboTrustBench_Dataset`
- Status: public subset available, full benchmark not yet available.
- The project/paper describe 1,207 expert-validated instruction-image pairs
  and a six-dimensional evaluation protocol with 13 fine-grained criteria.
- The current Hugging Face release is a 40-row `test` subset with four
  scenario categories: Normal, Constraint-sensitive, Counterfactual, and
  Adversarial.
- The dataset card states that this is not the full dataset and that the
  complete RoboTrustBench dataset will be released later.

Decision: use the 40-row release only to validate adapters, request generation,
and prompt/evaluator plumbing. It should not be counted as a passed
paper-level diagnostic benchmark.

## RoboWM-Bench

- Paper: `https://arxiv.org/abs/2604.19092`
- Status: highly relevant but not a stable main-result target for the current
  window.
- It is the closest embodiment-grounded world-model benchmark, but our audit
  found evaluator/API fragility and non-perfect GT replay under the local reset
  shim. Treat it as a robustness diagnostic until the official evaluator issue
  is clarified.

## Current Recommendation

The third evidence layer should proceed in this order:

1. Keep RoboTrustBench subset support working, but label it as adapter
   validation only.
2. Watch MiraBench for public annotation/result release; convert it first if
   available because it directly targets action-conditioned reliability.
3. Use RoboTrustBench full release if it arrives before MiraBench artifacts,
   but require multi-candidate judgments and a visual/model-score proxy gap.
4. Keep RoboWM-Bench as a conditional robustness layer, not as the primary
   third benchmark.

Do not update `docs/iclr_evidence_stack_registry.json` to `passed` for any
diagnostic benchmark until:

- the benchmark-level diagnostic gate passes;
- the selector table shows EFV beating the visual/model-score proxy;
- required diagnostic controls include `oracle_judgment_labels`,
  `proxy_or_rank0_failure`, and `visual_or_model_score_proxy`.
