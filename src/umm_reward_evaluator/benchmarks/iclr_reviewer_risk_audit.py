"""Reviewer-risk audit for the EFV ICLR evidence stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.iclr_evidence_stack_gate import (
    DIAGNOSTIC_LAYERS,
    EXECUTABLE_LAYERS,
    REQUIRED_CONTROLS,
    REQUIRED_DIAGNOSTIC_CONTROLS,
    _as_float,
    _as_int,
    _load_json,
    evaluate_evidence_stack,
)


def _entry_by_name(gate_result: dict[str, Any], name: str) -> dict[str, Any] | None:
    for entry in gate_result.get("benchmarks", []):
        if entry.get("benchmark") == name:
            return entry
    return None


def _passed_entries(gate_result: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in gate_result.get("benchmarks", []) if entry.get("gate_passed")]


def _passed_diagnostics(gate_result: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in _passed_entries(gate_result) if entry.get("layer") in DIAGNOSTIC_LAYERS]


def _passed_executable(gate_result: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in _passed_entries(gate_result) if entry.get("layer") in EXECUTABLE_LAYERS]


def _has_controls(entry: dict[str, Any] | None, controls: set[str]) -> bool:
    if entry is None:
        return False
    return controls.issubset(set(entry.get("shortcut_controls") or []))


def _bool_field(entry: dict[str, Any] | None, name: str) -> bool:
    return bool(entry and entry.get(name))


def _int_field(entry: dict[str, Any] | None, name: str) -> int:
    if entry is None:
        return 0
    return _as_int(entry.get(name))


def _risk(
    name: str,
    severity: str,
    status: str,
    evidence: list[str],
    next_action: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "severity": severity,
        "status": status,
        "evidence": evidence,
        "next_action": next_action,
    }


def _risk_sort_key(risk: dict[str, Any]) -> tuple[int, int, str]:
    severity_order = {"high": 0, "medium": 1, "low": 2}
    status_order = {"open": 0, "partially_defended": 1, "defended": 2}
    return (
        severity_order.get(str(risk.get("severity")), 9),
        status_order.get(str(risk.get("status")), 9),
        str(risk.get("name")),
    )


def _expert_template_risk(robotwin2: dict[str, Any] | None) -> dict[str, Any]:
    pressure_passed = _bool_field(robotwin2, "anti_template_pressure_passed")
    paper_gate_passed = _bool_field(robotwin2, "paper_readiness_passed") or bool(
        robotwin2 and robotwin2.get("gate_passed")
    )
    non_template_tasks = _int_field(robotwin2, "non_template_success_tasks")
    low_dtw_failed_tasks = _int_field(robotwin2, "low_dtw_failed_negative_tasks")
    dtw_beaten_tasks = _int_field(robotwin2, "dtw_template_beaten_tasks")
    if pressure_passed or (paper_gate_passed and non_template_tasks >= 2 and low_dtw_failed_tasks >= 2 and dtw_beaten_tasks >= 3):
        return _risk(
            "expert_template_matching",
            "medium",
            "defended",
            [
                "RoboTwin2 anti-template pressure is closed.",
                f"non_template_success_tasks={non_template_tasks}, low_dtw_failed_negative_tasks={low_dtw_failed_tasks}, dtw_template_beaten_tasks={dtw_beaten_tasks}",
            ],
            "Keep DTW nearest-positive in the main baseline table.",
        )
    if robotwin2 and _as_int(robotwin2.get("oracle_success")) > _as_int(robotwin2.get("rank0_success")):
        return _risk(
            "expert_template_matching",
            "high",
            "partially_defended",
            [
                "RoboTwin2 has oracle headroom, but anti-template pressure is not closed.",
                f"non_template_success_tasks={non_template_tasks}, low_dtw_failed_negative_tasks={low_dtw_failed_tasks}, dtw_template_beaten_tasks={dtw_beaten_tasks}",
            ],
            "Prioritize non-template successes and low-DTW failed negatives before adding easier tasks.",
        )
    return _risk(
        "expert_template_matching",
        "high",
        "open",
        ["No second executable benchmark currently proves that EFV beats nearest-expert/template matching."],
        "Complete RoboTwin2 partial seeds and run the anti-template pressure gate.",
    )


def build_risk_audit(gate_result: dict[str, Any]) -> dict[str, Any]:
    robocasa = _entry_by_name(gate_result, "RoboCasa365")
    robotwin2 = _entry_by_name(gate_result, "RoboTwin2")
    passed_executable = _passed_executable(gate_result)
    passed_diagnostics = _passed_diagnostics(gate_result)
    diagnostic_names = [str(entry.get("benchmark")) for entry in passed_diagnostics]
    risks: list[dict[str, Any]] = []

    risks.append(_expert_template_risk(robotwin2))

    if len(passed_executable) >= 2:
        risks.append(
            _risk(
                "single_benchmark_overclaim",
                "high",
                "defended",
                [f"passed_executable={len(passed_executable)}"],
                "Frame RoboCasa365 and RoboTwin2 as the executable evidence backbone.",
            )
        )
    elif robocasa and robocasa.get("gate_passed"):
        risks.append(
            _risk(
                "single_benchmark_overclaim",
                "high",
                "partially_defended",
                ["RoboCasa365 passes, but the second executable benchmark is still pending."],
                "Do not claim multi-benchmark effectiveness until RoboTwin2 passes the paper-readiness gate.",
            )
        )
    else:
        risks.append(
            _risk(
                "single_benchmark_overclaim",
                "high",
                "open",
                ["No executable benchmark currently passes the evidence gate."],
                "Recover a passed RoboCasa365 or equivalent executable benchmark before writing claims.",
            )
        )

    if robocasa and robocasa.get("gate_passed") and _has_controls(robocasa, REQUIRED_CONTROLS):
        risks.append(
            _risk(
                "weak_or_shortcut_baselines",
                "high",
                "partially_defended" if not (robotwin2 and robotwin2.get("gate_passed")) else "defended",
                ["RoboCasa365 includes rank0/random/energy/action-only/rank-remap controls."],
                "Carry the same controls plus DTW nearest-positive into RoboTwin2 and diagnostics.",
            )
        )
    else:
        risks.append(
            _risk(
                "weak_or_shortcut_baselines",
                "high",
                "open",
                ["Required shortcut controls are missing from the primary executable evidence."],
                "Add rank0, random, energy/magnitude, action-only, and candidate-ID/rank remap controls.",
            )
        )

    if passed_diagnostics:
        risks.append(
            _risk(
                "world_model_relevance",
                "medium",
                "defended",
                [f"passed diagnostics: {', '.join(diagnostic_names)}"],
                "Use the diagnostic layer to motivate future-selection rather than new world-model generation.",
            )
        )
    else:
        risks.append(
            _risk(
                "world_model_relevance",
                "medium",
                "open",
                ["No world-model/trust diagnostic layer currently passes with oracle labels and proxy baseline controls."],
                "Convert one 2025-2026 public diagnostic benchmark into a multi-candidate selector table.",
            )
        )

    diagnostic_controls_closed = any(_has_controls(entry, REQUIRED_DIAGNOSTIC_CONTROLS) for entry in passed_diagnostics)
    risks.append(
        _risk(
            "visual_plausibility_proxy",
            "medium",
            "defended" if diagnostic_controls_closed else "open",
            (
                ["A passed diagnostic layer includes oracle judgments and a visual/model-score proxy baseline."]
                if diagnostic_controls_closed
                else ["No passed diagnostic currently shows EFV beating a visual/model-score proxy."]
            ),
            "Report visual/model-score proxy selectors next to EFV on MiraBench/RoboTrustBench/RoboWM-style diagnostics.",
        )
    )

    if len(passed_executable) >= 2 and passed_diagnostics:
        real_robot_status = "partially_defended"
        real_robot_evidence = ["Two modern simulation benchmarks plus one diagnostic layer can support a sim-only claim boundary."]
    elif robocasa and robocasa.get("gate_passed"):
        real_robot_status = "open"
        real_robot_evidence = ["Only RoboCasa365 currently passes; no real robot or second executable benchmark is closed."]
    else:
        real_robot_status = "open"
        real_robot_evidence = ["No real robot evidence and no complete executable stack."]
    risks.append(
        _risk(
            "no_real_robot",
            "medium",
            real_robot_status,
            real_robot_evidence,
            "State the sim-only boundary explicitly; do not claim real-robot transfer without hardware evidence.",
        )
    )

    raw_integrity = _bool_field(robotwin2, "raw_integrity_passed") or _bool_field(robotwin2, "paper_readiness_passed")
    card = robocasa.get("evidence_card_validation") if robocasa else None
    primary_card_valid = isinstance(card, dict) and card.get("valid") is True
    artifact_status = "defended" if raw_integrity and primary_card_valid else "partially_defended" if primary_card_valid else "open"
    artifact_evidence = []
    if primary_card_valid:
        artifact_evidence.append("Primary RoboCasa365 evidence card validates.")
    if raw_integrity:
        artifact_evidence.append("RoboTwin2 raw integrity/paper readiness is marked as passed.")
    if not artifact_evidence:
        artifact_evidence.append("Evidence cards or raw-integrity markers are missing from the current registry/gate output.")
    risks.append(
        _risk(
            "system_artifact_or_partial_run",
            "medium",
            artifact_status,
            artifact_evidence,
            "Require raw-integrity audits before manifest conversion and evidence-card validation before registry promotion.",
        )
    )

    method_margin = _as_float(robocasa.get("method_success") if robocasa else 0.0) - _as_float(
        robocasa.get("best_non_oracle_baseline_success") if robocasa else 0.0
    )
    risks.append(
        _risk(
            "counterintuitive_signal_strength",
            "low",
            "defended" if method_margin >= 10 and robocasa and robocasa.get("gate_passed") else "open",
            [f"RoboCasa365 method-vs-best-baseline margin={method_margin:.1f}"],
            "Keep the counterintuitive claim tied to rank0/oracle headroom and shortcut-controlled negatives.",
        )
    )

    risks = sorted(risks, key=_risk_sort_key)
    open_risks = [risk for risk in risks if risk["status"] == "open"]
    partial_risks = [risk for risk in risks if risk["status"] == "partially_defended"]
    high_open = [risk for risk in open_risks if risk["severity"] == "high"]
    high_partial = [risk for risk in partial_risks if risk["severity"] == "high"]
    verdict = "paper_ready" if not open_risks and not high_open and gate_result.get("passed") else "bounded_push"
    if high_open or high_partial:
        verdict = "not_ready_for_strong_claim"

    return {
        "verdict": verdict,
        "open_high_risks": len(high_open),
        "partial_high_risks": len(high_partial),
        "open_risks": len(open_risks),
        "partially_defended_risks": len(partial_risks),
        "risks": risks,
    }


def render_markdown(report: dict[str, Any], *, title: str = "EFV Reviewer Risk Audit") -> str:
    lines = [
        f"# {title}",
        "",
        f"- verdict: `{report['verdict']}`",
        f"- open high risks: {report['open_high_risks']}",
        f"- partial high risks: {report['partial_high_risks']}",
        f"- open risks: {report['open_risks']}",
        f"- partially defended risks: {report['partially_defended_risks']}",
        "",
        "| Risk | Severity | Status | Evidence | Next action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for risk in report["risks"]:
        evidence = "<br>".join(str(item) for item in risk.get("evidence") or []) or "-"
        lines.append(
            "| {name} | {severity} | {status} | {evidence} | {next_action} |".format(
                name=risk.get("name", "-"),
                severity=risk.get("severity", "-"),
                status=risk.get("status", "-"),
                evidence=evidence,
                next_action=risk.get("next_action", "-"),
            )
        )
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
    report = build_risk_audit(gate_result)
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
