"""Generate claim-level guidance from the ICLR evidence stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.iclr_evidence_stack_gate import (
    DIAGNOSTIC_LAYERS,
    EXECUTABLE_LAYERS,
    _load_json,
    evaluate_evidence_stack,
)


def _passed_benchmarks(gate_result: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in gate_result.get("benchmarks", []) if entry.get("gate_passed")]


def _failed_check_names(gate_result: dict[str, Any]) -> list[str]:
    return [str(check["name"]) for check in gate_result.get("checks", []) if not check.get("passed")]


def _benchmark_names(entries: list[dict[str, Any]]) -> list[str]:
    return [str(entry.get("benchmark")) for entry in entries]


def claim_level(gate_result: dict[str, Any]) -> str:
    if gate_result.get("passed"):
        return "multi_benchmark_ready"
    passed = _passed_benchmarks(gate_result)
    if any(entry.get("benchmark") == "RoboCasa365" for entry in passed):
        return "single_benchmark_mechanism"
    return "not_ready"


def build_claim_report(gate_result: dict[str, Any]) -> dict[str, Any]:
    passed = _passed_benchmarks(gate_result)
    executable_passed = [entry for entry in passed if entry.get("layer") in EXECUTABLE_LAYERS]
    diagnostic_passed = [entry for entry in passed if entry.get("layer") in DIAGNOSTIC_LAYERS]
    level = claim_level(gate_result)

    allowed_claims: list[str] = []
    prohibited_claims: list[str] = [
        "Do not claim real-robot deployment or sim-to-real validation.",
        "Do not claim a new world model or robot policy; the contribution is candidate future verification.",
    ]

    if level == "multi_benchmark_ready":
        allowed_claims.append(
            "The EFV story is supported across the required 2025-2026 evidence stack: two executable manipulation layers plus one world-model/trust diagnostic layer."
        )
        allowed_claims.append(
            "It is safe to frame the main result as multi-benchmark executable-future selection, while still avoiding real-robot deployment claims."
        )
    elif level == "single_benchmark_mechanism":
        allowed_claims.append(
            "RoboCasa365 supports a strong mechanism claim: shortcut-controlled execution-envelope features recover futures that rank0, action-only, and object-only controls miss."
        )
        allowed_claims.append(
            "The project should be described as strong RoboCasa365 evidence with RoboTwin2 and world-model diagnostic evidence still in progress."
        )
        prohibited_claims.append(
            "Do not claim validated performance across multiple mainstream benchmarks until RoboTwin2 and a diagnostic layer pass their gates."
        )
    else:
        allowed_claims.append(
            "Use the current results only as internal diagnostics; the evidence stack does not yet support a paper-level claim."
        )
        prohibited_claims.append("Do not present the current result set as a main-conference-ready EFV story.")

    if not any(entry.get("benchmark") == "RoboTwin2" for entry in passed):
        prohibited_claims.append("Do not use RoboTwin2 as the second main executable benchmark yet.")
    if not diagnostic_passed:
        prohibited_claims.append("Do not claim validation on world-model/trust diagnostics yet.")

    next_actions: list[str] = []
    if not any(entry.get("benchmark") == "RoboTwin2" for entry in passed):
        next_actions.append(
            "Close the RoboTwin2 paper-readiness gate: at least four base-ready tasks, anti-template successes, matched low-DTW negatives, selector margins, and one relation-rescue task."
        )
    if not diagnostic_passed:
        next_actions.append(
            "Instantiate one public diagnostic layer with multi-candidate judgments, proxy-score failure, and an EFV selector table beating the visual/model-score proxy."
        )
    if len(passed) < int(gate_result.get("thresholds", {}).get("min_total_passed", 3)):
        next_actions.append("Update the evidence registry only after benchmark-level gates pass from current result files.")

    return {
        "claim_level": level,
        "evidence_stack_passed": bool(gate_result.get("passed")),
        "passed_benchmarks": _benchmark_names(passed),
        "passed_executable_benchmarks": _benchmark_names(executable_passed),
        "passed_diagnostic_benchmarks": _benchmark_names(diagnostic_passed),
        "failed_checks": _failed_check_names(gate_result),
        "allowed_claims": allowed_claims,
        "prohibited_claims": prohibited_claims,
        "next_actions": next_actions,
    }


def render_markdown(report: dict[str, Any], *, title: str = "ICLR Claim Report") -> str:
    lines = [
        f"# {title}",
        "",
        f"- claim level: `{report['claim_level']}`",
        f"- evidence stack passed: `{str(report['evidence_stack_passed']).lower()}`",
        f"- passed benchmarks: {', '.join(report['passed_benchmarks']) or '-'}",
        f"- failed checks: {', '.join(report['failed_checks']) or '-'}",
        "",
        "## Allowed Claims",
        "",
    ]
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
    )
    report = build_claim_report(gate_result)
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
