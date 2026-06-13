"""Build a compact current-status report for the EFV ICLR evidence stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.iclr_claim_report import build_claim_report
from umm_reward_evaluator.benchmarks.iclr_evidence_stack_gate import _load_json, evaluate_evidence_stack


def _card_status(entry: dict[str, Any]) -> str:
    card = entry.get("evidence_card_validation") or {}
    if card.get("valid") is True:
        return "valid"
    if card.get("valid") is False and card.get("required"):
        return "missing_or_invalid"
    if card.get("present"):
        return "present_unvalidated"
    return "not_required"


def build_status_report(gate_result: dict[str, Any], claim_report: dict[str, Any]) -> dict[str, Any]:
    benchmark_rows: list[dict[str, Any]] = []
    for entry in gate_result.get("benchmarks", []):
        benchmark_rows.append(
            {
                "benchmark": entry.get("benchmark"),
                "year": entry.get("year"),
                "layer": entry.get("layer"),
                "registry_status": entry.get("status"),
                "gate_passed": bool(entry.get("gate_passed")),
                "cases": entry.get("cases"),
                "tasks": entry.get("tasks"),
                "rank0_success": entry.get("rank0_success"),
                "oracle_success": entry.get("oracle_success"),
                "method_success": entry.get("method_success"),
                "best_non_oracle_baseline_success": entry.get("best_non_oracle_baseline_success"),
                "selector_margin": entry.get("selector_margin"),
                "missing_controls": entry.get("missing_controls") or [],
                "evidence_card": entry.get("evidence_card"),
                "evidence_card_status": _card_status(entry),
                "evidence": entry.get("evidence", ""),
            }
        )

    return {
        "claim_level": claim_report["claim_level"],
        "evidence_stack_passed": bool(gate_result.get("passed")),
        "passed_benchmarks": claim_report["passed_benchmarks"],
        "failed_checks": claim_report["failed_checks"],
        "allowed_claims": claim_report["allowed_claims"],
        "prohibited_claims": claim_report["prohibited_claims"],
        "next_actions": claim_report["next_actions"],
        "benchmarks": benchmark_rows,
        "thresholds": gate_result.get("thresholds", {}),
    }


def render_markdown(report: dict[str, Any], *, title: str = "EFV ICLR Status Report") -> str:
    lines = [
        f"# {title}",
        "",
        f"- claim level: `{report['claim_level']}`",
        f"- evidence stack passed: `{str(report['evidence_stack_passed']).lower()}`",
        f"- passed benchmarks: {', '.join(report['passed_benchmarks']) or '-'}",
        f"- failed checks: {', '.join(report['failed_checks']) or '-'}",
        "",
        "## Benchmark Status",
        "",
        "| Benchmark | Year | Layer | Registry | Gate | Cases | Tasks | Rank0 | Oracle | Method | Baseline | Card |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["benchmarks"]:
        lines.append(
            "| {benchmark} | {year} | {layer} | {registry} | {gate} | {cases} | {tasks} | {rank0} | {oracle} | {method} | {baseline} | {card} |".format(
                benchmark=row.get("benchmark", "-"),
                year=row.get("year", "-"),
                layer=row.get("layer", "-"),
                registry=row.get("registry_status", "-"),
                gate="pass" if row.get("gate_passed") else "fail",
                cases=row.get("cases", 0),
                tasks=row.get("tasks", 0),
                rank0=row.get("rank0_success", 0),
                oracle=row.get("oracle_success", 0),
                method=row.get("method_success", 0),
                baseline=row.get("best_non_oracle_baseline_success", 0),
                card=row.get("evidence_card_status", "-"),
            )
        )
    lines.extend(["", "## Allowed Claims", ""])
    lines.extend(f"- {claim}" for claim in report["allowed_claims"])
    lines.extend(["", "## Prohibited Claims", ""])
    lines.extend(f"- {claim}" for claim in report["prohibited_claims"])
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {action}" for action in report["next_actions"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--require-evidence-cards", action="store_true")
    parser.add_argument("--evidence-card-root", type=Path, default=Path("."))
    parser.add_argument("--min-total-passed", type=int, default=3)
    parser.add_argument("--min-executable-passed", type=int, default=2)
    parser.add_argument("--min-diagnostic-passed", type=int, default=1)
    parser.add_argument("--min-cases-per-passed-benchmark", type=int, default=16)
    parser.add_argument("--min-tasks-per-passed-executable", type=int, default=4)
    parser.add_argument("--min-selector-margin", type=float, default=1.0)
    args = parser.parse_args()

    payload = _load_json(args.evidence_json)
    entries = payload.get("benchmarks")
    if not isinstance(entries, list):
        raise SystemExit("--evidence-json must contain a 'benchmarks' list")
    gate_result = evaluate_evidence_stack(
        entries,
        min_total_passed=args.min_total_passed,
        min_executable_passed=args.min_executable_passed,
        min_diagnostic_passed=args.min_diagnostic_passed,
        min_cases_per_passed_benchmark=args.min_cases_per_passed_benchmark,
        min_tasks_per_passed_executable=args.min_tasks_per_passed_executable,
        min_selector_margin=args.min_selector_margin,
        require_evidence_cards=args.require_evidence_cards,
        evidence_card_root=args.evidence_card_root,
    )
    report = build_status_report(gate_result, build_claim_report(gate_result))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
