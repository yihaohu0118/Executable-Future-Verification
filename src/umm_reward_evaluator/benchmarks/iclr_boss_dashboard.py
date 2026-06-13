"""Boss-facing dashboard for EFV ICLR readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.iclr_claim_report import build_claim_report
from umm_reward_evaluator.benchmarks.iclr_evidence_stack_gate import (
    DIAGNOSTIC_LAYERS,
    EXECUTABLE_LAYERS,
    _as_float,
    _as_int,
    _load_json,
    evaluate_evidence_stack,
)
from umm_reward_evaluator.benchmarks.iclr_gap_report import build_gap_report


def _passed_entries(gate_result: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in gate_result.get("benchmarks", []) if entry.get("gate_passed")]


def _entry_by_name(gate_result: dict[str, Any], name: str) -> dict[str, Any] | None:
    for entry in gate_result.get("benchmarks", []):
        if entry.get("benchmark") == name:
            return entry
    return None


def _evidence_score(gate_result: dict[str, Any]) -> int:
    passed = _passed_entries(gate_result)
    executable = [entry for entry in passed if entry.get("layer") in EXECUTABLE_LAYERS]
    diagnostic = [entry for entry in passed if entry.get("layer") in DIAGNOSTIC_LAYERS]
    score = 0
    if _entry_by_name(gate_result, "RoboCasa365") and _entry_by_name(gate_result, "RoboCasa365").get("gate_passed"):
        score += 35
    if _entry_by_name(gate_result, "RoboTwin2") and _entry_by_name(gate_result, "RoboTwin2").get("gate_passed"):
        score += 30
    if diagnostic:
        score += 20
    score += min(len(passed), 3) * 5
    if len(executable) >= 2 and diagnostic:
        score += 10
    return min(score, 100)


def _maturity(score: int, gate_result: dict[str, Any], claim_report: dict[str, Any]) -> str:
    if gate_result.get("passed"):
        return "full_paper_ready"
    if claim_report.get("claim_level") == "single_benchmark_mechanism" and score >= 40:
        return "focused_push"
    if _passed_entries(gate_result):
        return "diagnostic_candidate"
    return "not_ready"


def _main_verdict(maturity: str) -> str:
    if maturity == "full_paper_ready":
        return "Ready to write the full ICLR story, subject to paper drafting and final reruns."
    if maturity == "focused_push":
        return "Worth continuing in a bounded window: the mechanism is strong, but the multi-benchmark evidence is not closed."
    if maturity == "diagnostic_candidate":
        return "Keep as a diagnostic project unless a second executable benchmark and one diagnostic layer close."
    return "Not ready for a paper-level claim."


def _counterintuitive_signal(gate_result: dict[str, Any]) -> dict[str, Any]:
    robocasa = _entry_by_name(gate_result, "RoboCasa365")
    if not robocasa:
        return {"status": "missing", "summary": "RoboCasa365 evidence is missing."}
    rank0 = _as_float(robocasa.get("rank0_success"))
    oracle = _as_float(robocasa.get("oracle_success"))
    method = _as_float(robocasa.get("method_success"))
    baseline = _as_float(robocasa.get("best_non_oracle_baseline_success"))
    return {
        "status": "strong" if robocasa.get("gate_passed") else "not_closed",
        "summary": (
            "Rank0 fails while oracle succeeds, and compact execution envelopes beat the strongest non-oracle control."
        ),
        "numbers": {
            "rank0_success": rank0,
            "oracle_success": oracle,
            "method_success": method,
            "best_non_oracle_baseline_success": baseline,
            "method_margin": method - baseline,
        },
    }


def _top_blockers(gap_report: dict[str, Any]) -> list[dict[str, Any]]:
    priority_order = {"high": 0, "medium": 1, "low": 2, "none": 3}
    benchmark_order = {"RoboTwin2": 0, "MiraBench": 1, "RoboTrustBench": 2, "RoboWM-Bench": 3}
    blockers = [gap for gap in gap_report.get("benchmark_gaps", []) if not gap.get("gate_passed")]
    return sorted(
        blockers,
        key=lambda gap: (
            priority_order.get(str(gap.get("priority")), 9),
            benchmark_order.get(str(gap.get("benchmark")), 99),
            str(gap.get("benchmark")),
        ),
    )


def _next_priority(top_blockers: list[dict[str, Any]], gap_report: dict[str, Any]) -> str:
    names = {str(gap.get("benchmark")) for gap in top_blockers}
    if "RoboTwin2" in names:
        return "Close RoboTwin2 first; it is the missing second executable benchmark."
    global_gaps = gap_report.get("global_gaps", {})
    if int(global_gaps.get("missing_diagnostic", 0) or 0) > 0:
        return "Close one world-model/trust diagnostic layer with public multi-candidate judgments."
    return "Regenerate claim/status reports and freeze the evidence tables."


def build_dashboard(gate_result: dict[str, Any]) -> dict[str, Any]:
    claim_report = build_claim_report(gate_result)
    gap_report = build_gap_report(gate_result)
    score = _evidence_score(gate_result)
    maturity = _maturity(score, gate_result, claim_report)
    passed = _passed_entries(gate_result)
    executable = [entry for entry in passed if entry.get("layer") in EXECUTABLE_LAYERS]
    diagnostic = [entry for entry in passed if entry.get("layer") in DIAGNOSTIC_LAYERS]
    top_blockers = _top_blockers(gap_report)
    thresholds = gate_result.get("thresholds", {})

    return {
        "verdict": _main_verdict(maturity),
        "maturity": maturity,
        "evidence_score": score,
        "claim_level": claim_report["claim_level"],
        "evidence_stack_passed": bool(gate_result.get("passed")),
        "counterintuitive_signal": _counterintuitive_signal(gate_result),
        "benchmark_coverage": {
            "passed_total": len(passed),
            "required_total": int(thresholds.get("min_total_passed", 3)),
            "passed_executable": len(executable),
            "required_executable": int(thresholds.get("min_executable_passed", 2)),
            "passed_diagnostic": len(diagnostic),
            "required_diagnostic": int(thresholds.get("min_diagnostic_passed", 1)),
            "passed_benchmarks": [str(entry.get("benchmark")) for entry in passed],
        },
        "top_blockers": top_blockers[:3],
        "next_priority": _next_priority(top_blockers, gap_report),
        "allowed_claims": claim_report["allowed_claims"],
        "prohibited_claims": claim_report["prohibited_claims"],
        "next_actions": claim_report["next_actions"],
        "kill_or_downgrade_triggers": [
            "RoboTwin2 has fewer than four clean oracle-headroom tasks.",
            "DTW nearest-positive stays within one success of the best envelope verifier.",
            "Successful candidates remain mostly full expert traces or unknown-source variants.",
            "No world-model/trust diagnostic layer can produce a public multi-candidate judgment table.",
        ],
    }


def render_markdown(report: dict[str, Any], *, title: str = "EFV ICLR Boss Dashboard") -> str:
    coverage = report["benchmark_coverage"]
    signal = report["counterintuitive_signal"]
    numbers = signal.get("numbers") or {}
    lines = [
        f"# {title}",
        "",
        f"- verdict: **{report['verdict']}**",
        f"- maturity: `{report['maturity']}`",
        f"- evidence score: {report['evidence_score']}/100",
        f"- claim level: `{report['claim_level']}`",
        f"- evidence stack passed: `{str(report['evidence_stack_passed']).lower()}`",
        "",
        "## Core Signal",
        "",
        f"- status: `{signal['status']}`",
        f"- summary: {signal['summary']}",
        (
            "- RoboCasa365 numbers: rank0={rank0:.1f}, oracle={oracle:.1f}, method={method:.1f}, "
            "best baseline={baseline:.1f}, margin={margin:.1f}"
        ).format(
            rank0=_as_float(numbers.get("rank0_success")),
            oracle=_as_float(numbers.get("oracle_success")),
            method=_as_float(numbers.get("method_success")),
            baseline=_as_float(numbers.get("best_non_oracle_baseline_success")),
            margin=_as_float(numbers.get("method_margin")),
        ),
        "",
        "## Benchmark Coverage",
        "",
        f"- total: {coverage['passed_total']} / {coverage['required_total']}",
        f"- executable: {coverage['passed_executable']} / {coverage['required_executable']}",
        f"- diagnostic: {coverage['passed_diagnostic']} / {coverage['required_diagnostic']}",
        f"- passed: {', '.join(coverage['passed_benchmarks']) or '-'}",
        "",
        "## Top Blockers",
        "",
        "| Benchmark | Layer | Priority | Blockers | Next action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for blocker in report["top_blockers"]:
        lines.append(
            "| {benchmark} | {layer} | {priority} | {blockers} | {next_action} |".format(
                benchmark=blocker.get("benchmark", "-"),
                layer=blocker.get("layer", "-"),
                priority=blocker.get("priority", "-"),
                blockers=", ".join(blocker.get("blockers") or []) or "-",
                next_action=blocker.get("next_action", "-"),
            )
        )
    lines.extend(["", "## Next Priority", "", f"- {report['next_priority']}", "", "## Allowed Claims", ""])
    lines.extend(f"- {claim}" for claim in report["allowed_claims"])
    lines.extend(["", "## Prohibited Claims", ""])
    lines.extend(f"- {claim}" for claim in report["prohibited_claims"])
    lines.extend(["", "## Kill Or Downgrade Triggers", ""])
    lines.extend(f"- {trigger}" for trigger in report["kill_or_downgrade_triggers"])
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
    report = build_dashboard(gate_result)
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
