"""Generate actionable gap accounting for the EFV ICLR evidence stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.iclr_evidence_stack_gate import (
    DIAGNOSTIC_LAYERS,
    EXECUTABLE_LAYERS,
    _as_float,
    _as_int,
    _load_json,
    evaluate_evidence_stack,
)


def _failed_checks(gate_result: dict[str, Any]) -> list[dict[str, Any]]:
    return [check for check in gate_result.get("checks", []) if not check.get("passed")]


def _passed_entries(gate_result: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in gate_result.get("benchmarks", []) if entry.get("gate_passed")]


def _entry_thresholds(entry: dict[str, Any], thresholds: dict[str, Any]) -> tuple[int, int, float]:
    layer = str(entry.get("layer", "unknown"))
    min_cases = int(thresholds.get("min_cases_per_passed_benchmark", 16))
    min_tasks = int(thresholds.get("min_tasks_per_passed_executable", 4)) if layer in EXECUTABLE_LAYERS else 1
    min_margin = float(thresholds.get("min_selector_margin", 1.0))
    return min_cases, min_tasks, min_margin


def _benchmark_gaps(entry: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    min_cases, min_tasks, min_margin = _entry_thresholds(entry, thresholds)
    cases = _as_int(entry.get("cases"))
    tasks = _as_int(entry.get("tasks"), 1)
    rank0 = _as_float(entry.get("rank0_success"))
    oracle = _as_float(entry.get("oracle_success"))
    method = _as_float(entry.get("method_success"))
    baseline = _as_float(entry.get("best_non_oracle_baseline_success"))
    margin = method - baseline
    card = entry.get("evidence_card_validation") or {}
    blockers: list[str] = []
    missing: dict[str, Any] = {}

    if entry.get("gate_passed"):
        return {
            "benchmark": entry.get("benchmark"),
            "layer": entry.get("layer"),
            "gate_passed": True,
            "priority": "none",
            "blockers": [],
            "missing": {},
            "next_action": "Keep this benchmark frozen unless upstream result files change.",
        }

    if entry.get("status") != "passed":
        blockers.append("registry_status")
        missing["registry_status"] = {"current": entry.get("status"), "required": "passed"}
    if not entry.get("modern_year", False):
        blockers.append("modern_year")
        missing["modern_year"] = {"current": entry.get("year"), "required": "2025_or_2026"}
    if cases < min_cases:
        blockers.append("cases")
        missing["cases"] = {"current": cases, "required": min_cases, "delta": min_cases - cases}
    if tasks < min_tasks:
        blockers.append("tasks")
        missing["tasks"] = {"current": tasks, "required": min_tasks, "delta": min_tasks - tasks}
    if oracle <= rank0:
        blockers.append("oracle_headroom")
        missing["oracle_headroom"] = {"rank0": rank0, "oracle": oracle, "required": "oracle > rank0"}
    if margin < min_margin:
        blockers.append("selector_margin")
        missing["selector_margin"] = {"current": margin, "required": min_margin, "delta": min_margin - margin}
    missing_controls = entry.get("missing_controls") or []
    if missing_controls:
        blockers.append("shortcut_controls")
        missing["shortcut_controls"] = missing_controls
    if card.get("required") and not card.get("valid"):
        blockers.append("evidence_card")
        missing["evidence_card"] = {"present": bool(card.get("present")), "errors": card.get("errors") or []}

    benchmark = str(entry.get("benchmark", "unknown"))
    layer = str(entry.get("layer", "unknown"))
    if benchmark == "RoboTwin2":
        priority = "high"
        next_action = "Run the bounded 4-task RoboTwin2 window and update the registry only after the paper-readiness gate passes."
    elif layer in DIAGNOSTIC_LAYERS:
        priority = "high"
        next_action = "Instantiate a public diagnostic manifest with oracle judgments, visual/model-score proxy failures, and verifier margin."
    elif layer in EXECUTABLE_LAYERS:
        priority = "medium"
        next_action = "Add complete shortcut-controlled executable cases or demote this benchmark from the main evidence stack."
    else:
        priority = "low"
        next_action = "Clarify benchmark layer/status before relying on it."

    return {
        "benchmark": entry.get("benchmark"),
        "layer": entry.get("layer"),
        "gate_passed": False,
        "priority": priority,
        "blockers": blockers,
        "missing": missing,
        "next_action": next_action,
    }


def build_gap_report(gate_result: dict[str, Any]) -> dict[str, Any]:
    thresholds = gate_result.get("thresholds", {})
    passed = _passed_entries(gate_result)
    executable_passed = [entry for entry in passed if entry.get("layer") in EXECUTABLE_LAYERS]
    diagnostic_passed = [entry for entry in passed if entry.get("layer") in DIAGNOSTIC_LAYERS]
    min_total = int(thresholds.get("min_total_passed", 3))
    min_executable = int(thresholds.get("min_executable_passed", 2))
    min_diagnostic = int(thresholds.get("min_diagnostic_passed", 1))

    global_gaps = {
        "passed_total": len(passed),
        "required_total": min_total,
        "missing_total": max(0, min_total - len(passed)),
        "passed_executable": len(executable_passed),
        "required_executable": min_executable,
        "missing_executable": max(0, min_executable - len(executable_passed)),
        "passed_diagnostic": len(diagnostic_passed),
        "required_diagnostic": min_diagnostic,
        "missing_diagnostic": max(0, min_diagnostic - len(diagnostic_passed)),
        "failed_checks": [check["name"] for check in _failed_checks(gate_result)],
    }
    benchmark_gaps = [_benchmark_gaps(entry, thresholds) for entry in gate_result.get("benchmarks", [])]

    next_actions: list[str] = []
    if any(gap["benchmark"] == "RoboTwin2" and not gap["gate_passed"] for gap in benchmark_gaps):
        next_actions.append(
            "First close RoboTwin2 with the bounded sequential launcher; this is the missing second executable layer."
        )
    if global_gaps["missing_diagnostic"] > 0:
        next_actions.append(
            "Then close one 2026 diagnostic layer by converting public judgments into a multi-candidate EFV selector table."
        )
    if not next_actions:
        next_actions.append("Evidence stack gaps are closed; regenerate the claim and status reports before writing the paper.")

    return {
        "evidence_stack_passed": bool(gate_result.get("passed")),
        "global_gaps": global_gaps,
        "benchmark_gaps": benchmark_gaps,
        "next_actions": next_actions,
    }


def render_markdown(report: dict[str, Any], *, title: str = "EFV ICLR Gap Report") -> str:
    global_gaps = report["global_gaps"]
    lines = [
        f"# {title}",
        "",
        f"- evidence stack passed: `{str(report['evidence_stack_passed']).lower()}`",
        f"- total benchmarks: {global_gaps['passed_total']} / {global_gaps['required_total']}",
        f"- executable layers: {global_gaps['passed_executable']} / {global_gaps['required_executable']}",
        f"- diagnostic layers: {global_gaps['passed_diagnostic']} / {global_gaps['required_diagnostic']}",
        f"- failed checks: {', '.join(global_gaps['failed_checks']) or '-'}",
        "",
        "## Benchmark Gaps",
        "",
        "| Benchmark | Layer | Gate | Priority | Blockers | Next action |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for gap in report["benchmark_gaps"]:
        blockers = ", ".join(gap.get("blockers") or []) or "-"
        lines.append(
            "| {benchmark} | {layer} | {gate} | {priority} | {blockers} | {next_action} |".format(
                benchmark=gap.get("benchmark", "-"),
                layer=gap.get("layer", "-"),
                gate="pass" if gap.get("gate_passed") else "fail",
                priority=gap.get("priority", "-"),
                blockers=blockers,
                next_action=gap.get("next_action", "-"),
            )
        )
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
    report = build_gap_report(gate_result)
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
