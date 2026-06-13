"""Run the full world-model diagnostic evidence pipeline.

The pipeline is deliberately conservative: it converts public diagnostic
judgments into the shared candidate-future manifest, optionally adds a
leave-one-case-out verifier score, then runs the manifest gate, selector table,
readiness gate, registry-entry proposal, and evidence-card validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.common import write_jsonl
from umm_reward_evaluator.benchmarks.evidence_card_validator import (
    render_markdown as render_card_validation_markdown,
    validate_card,
)
from umm_reward_evaluator.benchmarks.iclr_registry_proposal import propose_diagnostic_entry, render_markdown as render_registry_markdown
from umm_reward_evaluator.benchmarks.world_model_diagnostic_calibrate_verifier import (
    calibrate_scores,
    render_markdown as render_calibration_markdown,
)
from umm_reward_evaluator.benchmarks.world_model_diagnostic_evidence_card import build_card
from umm_reward_evaluator.benchmarks.world_model_diagnostic_gate import (
    evaluate_diagnostic_gate,
    render_markdown as render_gate_markdown,
)
from umm_reward_evaluator.benchmarks.world_model_diagnostic_readiness_gate import (
    evaluate_readiness,
    render_markdown as render_readiness_markdown,
)
from umm_reward_evaluator.benchmarks.world_model_diagnostic_selector_table import (
    evaluate_selectors,
    render_markdown as render_selector_markdown,
)
from umm_reward_evaluator.benchmarks.world_model_diagnostic_to_manifest import _load_records, convert_records


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_pipeline(
    *,
    records: list[dict[str, Any]],
    run_root: Path,
    benchmark: str,
    year: int,
    layer: str,
    suite: str,
    verification_target: str,
    score_key: str | None = None,
    threshold: float | None = None,
    verifier_score_key: str = "metadata.efv_score",
    proxy_score_key: str = "__planner_or_model__",
    calibrate_verifier: bool = True,
    feature_keys: list[str] | None = None,
    categorical_keys: list[str] | None = None,
    calibration_method: str = "ridge",
    include_action_stats: bool = True,
    include_planner_score_in_verifier: bool = False,
    include_rank_in_verifier: bool = False,
    same_task_only: bool = False,
    l2: float = 1.0,
    allow_label_like_features: bool = False,
    min_cases: int = 16,
    min_tasks: int = 1,
    min_oracle_better_cases: int = 1,
    min_candidates_per_case: int = 2,
    category_keys: list[str] | None = None,
    min_categories: int = 1,
    required_metadata_keys: list[str] | None = None,
    min_planner_score_oracle_gap: int = 1,
    min_planner_score_failures: int = 1,
    min_verifier_proxy_margin: float = 1.0,
    shortcut_controls: list[str] | None = None,
    evidence: str | None = None,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    manifests_dir = run_root / "manifests"
    selectors_dir = run_root / "selectors"
    cards_dir = run_root / "evidence_cards"
    manifest_path = manifests_dir / "world_model_diagnostic_manifest.jsonl"
    scored_manifest_path = manifests_dir / "world_model_diagnostic_scored_manifest.jsonl"

    rows = convert_records(
        records,
        default_benchmark=benchmark,
        default_suite=suite,
        default_verification_target=verification_target,
        score_key=score_key,
        threshold=threshold,
    )
    write_jsonl(manifest_path, rows)

    calibration_summary: dict[str, Any] | None = None
    final_rows = rows
    final_manifest_path = manifest_path
    if calibrate_verifier:
        final_rows, calibration_summary = calibrate_scores(
            rows,
            score_key=verifier_score_key,
            feature_keys=list(feature_keys or []),
            categorical_keys=list(categorical_keys or []),
            method=calibration_method,
            include_action_stats=include_action_stats,
            include_planner_score=include_planner_score_in_verifier,
            include_rank=include_rank_in_verifier,
            same_task_only=same_task_only,
            l2=l2,
            allow_label_like_features=allow_label_like_features,
        )
        write_jsonl(scored_manifest_path, final_rows)
        _write_json(selectors_dir / "world_model_diagnostic_calibration_summary.json", calibration_summary)
        (selectors_dir / "world_model_diagnostic_calibration_summary.md").write_text(
            render_calibration_markdown(calibration_summary),
            encoding="utf-8",
        )
        final_manifest_path = scored_manifest_path

    diagnostic_gate = evaluate_diagnostic_gate(
        final_rows,
        min_cases=min_cases,
        min_tasks=min_tasks,
        min_oracle_better_cases=min_oracle_better_cases,
        min_candidates_per_case=min_candidates_per_case,
        category_keys=list(category_keys or []),
        min_categories=min_categories,
        required_metadata_keys=list(required_metadata_keys or []),
        require_planner_score_baseline=True,
        min_planner_score_oracle_gap=min_planner_score_oracle_gap,
        min_planner_score_failures=min_planner_score_failures,
    )
    _write_json(selectors_dir / "world_model_diagnostic_gate.json", diagnostic_gate)
    (selectors_dir / "world_model_diagnostic_gate.md").write_text(render_gate_markdown(diagnostic_gate), encoding="utf-8")

    selector_table = evaluate_selectors(
        final_rows,
        verifier_score_key=verifier_score_key,
        proxy_score_key=proxy_score_key,
    )
    selector_payload = {"manifest": str(final_manifest_path), **selector_table}
    _write_json(selectors_dir / "world_model_diagnostic_selector_table.json", selector_payload)
    (selectors_dir / "world_model_diagnostic_selector_table.md").write_text(
        render_selector_markdown(selector_table),
        encoding="utf-8",
    )

    verifier_selector = f"verifier_score:{verifier_score_key}"
    readiness_gate = evaluate_readiness(
        diagnostic_gate=diagnostic_gate,
        selector_table=selector_payload,
        verifier_selector=verifier_selector,
        proxy_selector="planner_or_model_score",
        min_verifier_proxy_margin=min_verifier_proxy_margin,
    )
    _write_json(selectors_dir / "world_model_diagnostic_readiness_gate.json", readiness_gate)
    (selectors_dir / "world_model_diagnostic_readiness_gate.md").write_text(
        render_readiness_markdown(readiness_gate),
        encoding="utf-8",
    )

    registry_entry = propose_diagnostic_entry(
        benchmark=benchmark,
        year=year,
        layer=layer,
        diagnostic_gate=diagnostic_gate,
        selector_table=selector_payload,
        verifier_selector=verifier_selector,
        readiness_gate=readiness_gate,
        extra_controls=list(shortcut_controls or []),
        evidence=evidence,
    )
    _write_json(selectors_dir / "world_model_diagnostic_registry_entry_proposal.json", registry_entry)
    (selectors_dir / "world_model_diagnostic_registry_entry_proposal.md").write_text(
        render_registry_markdown(registry_entry),
        encoding="utf-8",
    )

    card = build_card(
        registry_entry=registry_entry,
        diagnostic_gate=diagnostic_gate,
        selector_table=selector_payload,
        readiness_gate=readiness_gate,
        verifier_selector=verifier_selector,
        run_root=run_root,
    )
    card_path = cards_dir / f"{benchmark.lower().replace(' ', '_')}_world_model_diagnostic.json"
    _write_json(card_path, card)
    card_validation = validate_card(card, base_dir=repo_root)
    _write_json(card_path.with_suffix(".validation.json"), card_validation)
    card_path.with_suffix(".validation.md").write_text(
        render_card_validation_markdown(card_validation),
        encoding="utf-8",
    )

    summary = {
        "benchmark": benchmark,
        "run_root": str(run_root),
        "manifest": str(manifest_path),
        "scored_manifest": str(scored_manifest_path) if calibrate_verifier else None,
        "final_manifest": str(final_manifest_path),
        "calibration_summary": calibration_summary,
        "diagnostic_gate_passed": bool(diagnostic_gate.get("passed")),
        "readiness_gate_passed": bool(readiness_gate.get("passed")),
        "registry_status": registry_entry.get("status"),
        "evidence_card_validation_passed": bool(card_validation.get("passed")),
        "verifier_selector": verifier_selector,
        "proxy_selector": "planner_or_model_score",
        "outputs": {
            "diagnostic_gate": str(selectors_dir / "world_model_diagnostic_gate.json"),
            "selector_table": str(selectors_dir / "world_model_diagnostic_selector_table.json"),
            "readiness_gate": str(selectors_dir / "world_model_diagnostic_readiness_gate.json"),
            "registry_entry": str(selectors_dir / "world_model_diagnostic_registry_entry_proposal.json"),
            "evidence_card": str(card_path),
        },
    }
    _write_json(run_root / "world_model_diagnostic_pipeline_summary.json", summary)
    return summary


def render_markdown(summary: dict[str, Any], *, title: str = "World-Model Diagnostic Pipeline Summary") -> str:
    lines = [
        f"# {title}",
        "",
        f"- benchmark: {summary['benchmark']}",
        f"- run root: `{summary['run_root']}`",
        f"- final manifest: `{summary['final_manifest']}`",
        f"- verifier selector: `{summary['verifier_selector']}`",
        f"- proxy selector: `{summary['proxy_selector']}`",
        f"- diagnostic gate passed: `{str(summary['diagnostic_gate_passed']).lower()}`",
        f"- readiness gate passed: `{str(summary['readiness_gate_passed']).lower()}`",
        f"- registry status: `{summary['registry_status']}`",
        f"- evidence card validation passed: `{str(summary['evidence_card_validation_passed']).lower()}`",
        "",
        "| Output | Path |",
        "| --- | --- |",
    ]
    for name, path in summary["outputs"].items():
        lines.append(f"| `{name}` | `{path}` |")
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- A passed diagnostic gate means the external world-model judgment manifest has enough multi-candidate headroom.",
            "- A passed readiness gate means the EFV verifier beats the visual/model-score proxy with full coverage.",
            "- The registry status remains `pending` unless both gates and required shortcut controls support a paper-level claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full EFV world-model diagnostic pipeline.")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--layer", required=True, choices=["world_model_diagnostic", "trust_diagnostic", "robustness_diagnostic"])
    parser.add_argument("--suite", default="diagnostic")
    parser.add_argument("--verification-target", default="action_conditioned_reliability")
    parser.add_argument("--score-key")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--verifier-score-key", default="metadata.efv_score")
    parser.add_argument("--proxy-score-key", default="__planner_or_model__")
    parser.add_argument("--no-calibrate-verifier", action="store_true")
    parser.add_argument("--feature-key", action="append", default=[])
    parser.add_argument("--categorical-key", action="append", default=[])
    parser.add_argument("--calibration-method", choices=["ridge", "nearest_pos_neg", "pos_neg_centroid"], default="ridge")
    parser.add_argument("--no-action-stats", action="store_true")
    parser.add_argument("--include-planner-score-in-verifier", action="store_true")
    parser.add_argument("--include-rank-in-verifier", action="store_true")
    parser.add_argument("--same-task-only", action="store_true")
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--allow-label-like-features", action="store_true")
    parser.add_argument("--min-cases", type=int, default=16)
    parser.add_argument("--min-tasks", type=int, default=1)
    parser.add_argument("--min-oracle-better-cases", type=int, default=1)
    parser.add_argument("--min-candidates-per-case", type=int, default=2)
    parser.add_argument("--category-key", action="append", default=[])
    parser.add_argument("--min-categories", type=int, default=1)
    parser.add_argument("--require-metadata-key", action="append", default=[])
    parser.add_argument("--min-planner-score-oracle-gap", type=int, default=1)
    parser.add_argument("--min-planner-score-failures", type=int, default=1)
    parser.add_argument("--min-verifier-proxy-margin", type=float, default=1.0)
    parser.add_argument("--shortcut-control", action="append", default=[])
    parser.add_argument("--evidence")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero unless diagnostic and readiness gates pass.")
    args = parser.parse_args()

    if args.input is None and args.input_dir is None:
        raise SystemExit("provide --input and/or --input-dir")
    if (args.score_key is None) != (args.threshold is None):
        raise SystemExit("--score-key and --threshold must be provided together")

    records = _load_records(args.input, args.input_dir)
    summary = run_pipeline(
        records=records,
        run_root=args.run_root,
        benchmark=args.benchmark,
        year=args.year,
        layer=args.layer,
        suite=args.suite,
        verification_target=args.verification_target,
        score_key=args.score_key,
        threshold=args.threshold,
        verifier_score_key=args.verifier_score_key,
        proxy_score_key=args.proxy_score_key,
        calibrate_verifier=not args.no_calibrate_verifier,
        feature_keys=list(args.feature_key),
        categorical_keys=list(args.categorical_key),
        calibration_method=args.calibration_method,
        include_action_stats=not args.no_action_stats,
        include_planner_score_in_verifier=args.include_planner_score_in_verifier,
        include_rank_in_verifier=args.include_rank_in_verifier,
        same_task_only=args.same_task_only,
        l2=args.l2,
        allow_label_like_features=args.allow_label_like_features,
        min_cases=args.min_cases,
        min_tasks=args.min_tasks,
        min_oracle_better_cases=args.min_oracle_better_cases,
        min_candidates_per_case=args.min_candidates_per_case,
        category_keys=list(args.category_key),
        min_categories=args.min_categories,
        required_metadata_keys=list(args.require_metadata_key),
        min_planner_score_oracle_gap=args.min_planner_score_oracle_gap,
        min_planner_score_failures=args.min_planner_score_failures,
        min_verifier_proxy_margin=args.min_verifier_proxy_margin,
        shortcut_controls=list(args.shortcut_control),
        evidence=args.evidence,
        repo_root=args.repo_root,
    )
    markdown = render_markdown(summary)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    if args.strict and not (summary["diagnostic_gate_passed"] and summary["readiness_gate_passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
