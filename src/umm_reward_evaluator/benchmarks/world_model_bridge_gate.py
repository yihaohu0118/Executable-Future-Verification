"""Gate the EFV story against world-model benchmark connectivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.iclr_evidence_stack_gate import (
    DIAGNOSTIC_LAYERS,
    EXECUTABLE_LAYERS,
    _load_json,
)
from umm_reward_evaluator.benchmarks.world_model_artifact_audit import build_audit


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def _entry_margin(entry: dict[str, Any]) -> float:
    return _as_float(entry.get("method_success")) - _as_float(entry.get("best_non_oracle_baseline_success"))


def _passed_executable(entry: dict[str, Any], *, min_cases: int, min_tasks: int, min_margin: float) -> bool:
    return (
        entry.get("layer") in EXECUTABLE_LAYERS
        and entry.get("status") == "passed"
        and _as_int(entry.get("cases")) >= min_cases
        and _as_int(entry.get("tasks"), 1) >= min_tasks
        and _as_float(entry.get("oracle_success")) > _as_float(entry.get("rank0_success"))
        and _entry_margin(entry) >= min_margin
    )


def _diagnostic_audit_by_name(artifact_audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("benchmark")): row for row in artifact_audit.get("diagnostic_artifacts", [])}


def _diagnostic_entry_ready(entry: dict[str, Any], audit_row: dict[str, Any] | None, *, min_margin: float) -> bool:
    if entry.get("layer") not in DIAGNOSTIC_LAYERS:
        return False
    if audit_row is not None and not audit_row.get("registry_ready"):
        return False
    return (
        entry.get("status") == "passed"
        and _as_int(entry.get("cases")) > 0
        and _as_float(entry.get("oracle_success")) > _as_float(entry.get("rank0_success"))
        and _entry_margin(entry) >= min_margin
    )


def _check(name: str, passed: bool, detail: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _next_actions(checks: list[dict[str, Any]], artifact_rows: list[dict[str, Any]]) -> list[str]:
    failed = {check["name"] for check in checks if not check["passed"]}
    actions: list[str] = []
    if "second_executable_pressure" in failed:
        actions.append("Finish the RoboTwin2 pressure-closure run and require at least two DTW-breaking pressured tasks.")
    if "diagnostic_artifact_pipeline" in failed:
        blocked = [row for row in artifact_rows if not row.get("pipeline_ready")]
        names = ", ".join(str(row.get("benchmark")) for row in blocked) or "a public diagnostic benchmark"
        actions.append(f"Obtain or construct multi-candidate judgment/proxy artifacts for {names}.")
    if "diagnostic_registry" in failed:
        actions.append("Run world_model_diagnostic_pipeline and require EFV to beat the visual/model-score proxy with full coverage.")
    if "world_model_candidate_fields" in failed:
        actions.append("Require diagnostic manifests to include generator/model name, model score or visual proxy, and oracle judgment labels.")
    if not actions:
        actions.append("The bridge is ready; freeze evidence cards and write the claim as world-model future selection, not a new world model.")
    return actions


def evaluate_bridge(
    entries: list[dict[str, Any]],
    *,
    artifact_audit: dict[str, Any] | None = None,
    min_primary_cases: int = 16,
    min_primary_tasks: int = 4,
    min_selector_margin: float = 1.0,
    require_second_executable: bool = True,
) -> dict[str, Any]:
    audit = artifact_audit or build_audit(entries)
    audit_by_name = _diagnostic_audit_by_name(audit)

    executable_entries = [entry for entry in entries if entry.get("layer") in EXECUTABLE_LAYERS]
    diagnostic_entries = [entry for entry in entries if entry.get("layer") in DIAGNOSTIC_LAYERS]
    primary_ready = [
        entry
        for entry in executable_entries
        if _passed_executable(
            entry,
            min_cases=min_primary_cases,
            min_tasks=min_primary_tasks,
            min_margin=min_selector_margin,
        )
    ]
    second_candidates = [
        entry
        for entry in executable_entries
        if entry.get("benchmark") != "RoboCasa365"
        and _as_int(entry.get("cases")) > 0
        and _as_float(entry.get("oracle_success")) > _as_float(entry.get("rank0_success"))
    ]
    diagnostic_pipeline_ready = [
        row for row in audit.get("diagnostic_artifacts", []) if row.get("pipeline_ready")
    ]
    diagnostic_registry_ready = [
        entry
        for entry in diagnostic_entries
        if _diagnostic_entry_ready(
            entry,
            audit_by_name.get(str(entry.get("benchmark"))),
            min_margin=min_selector_margin,
        )
    ]
    diagnostic_controls = {
        str(entry.get("benchmark")): set(entry.get("shortcut_controls") or [])
        for entry in diagnostic_entries
    }
    required_candidate_fields = {
        "oracle_judgment_labels",
        "visual_or_model_score_proxy",
        "proxy_or_rank0_failure",
    }
    field_ready = any(required_candidate_fields.issubset(controls) for controls in diagnostic_controls.values())

    checks = [
        _check(
            "primary_executable_mechanism",
            bool(primary_ready),
            {"benchmarks": [entry.get("benchmark") for entry in primary_ready], "minimum": 1},
        ),
        _check(
            "second_executable_pressure",
            (not require_second_executable) or bool(second_candidates),
            {"benchmarks": [entry.get("benchmark") for entry in second_candidates], "required": require_second_executable},
        ),
        _check(
            "diagnostic_artifact_pipeline",
            bool(diagnostic_pipeline_ready),
            {"benchmarks": [row.get("benchmark") for row in diagnostic_pipeline_ready], "minimum": 1},
        ),
        _check(
            "diagnostic_registry",
            bool(diagnostic_registry_ready),
            {"benchmarks": [entry.get("benchmark") for entry in diagnostic_registry_ready], "minimum": 1},
        ),
        _check(
            "world_model_candidate_fields",
            field_ready,
            {
                "required": sorted(required_candidate_fields),
                "benchmarks_with_required_fields": [
                    benchmark
                    for benchmark, controls in diagnostic_controls.items()
                    if required_candidate_fields.issubset(controls)
                ],
            },
        ),
    ]
    passed = all(check["passed"] for check in checks)
    if passed:
        claim_level = "world_model_connected_evidence"
    elif primary_ready and second_candidates:
        claim_level = "executable_with_pending_world_model_bridge"
    elif primary_ready:
        claim_level = "single_executable_mechanism_only"
    else:
        claim_level = "insufficient_mechanism_evidence"

    return {
        "passed": passed,
        "claim_level": claim_level,
        "checks": checks,
        "summary": {
            "primary_ready": [entry.get("benchmark") for entry in primary_ready],
            "second_executable_candidates": [entry.get("benchmark") for entry in second_candidates],
            "diagnostic_pipeline_ready": [row.get("benchmark") for row in diagnostic_pipeline_ready],
            "diagnostic_registry_ready": [entry.get("benchmark") for entry in diagnostic_registry_ready],
            "artifact_summary": audit.get("summary", {}),
        },
        "diagnostic_artifacts": audit.get("diagnostic_artifacts", []),
        "allowed_claim": _allowed_claim(passed, bool(primary_ready), bool(second_candidates)),
        "prohibited_claims": _prohibited_claims(passed),
        "next_actions": _next_actions(checks, audit.get("diagnostic_artifacts", [])),
    }


def _allowed_claim(passed: bool, primary_ready: bool, second_ready: bool) -> str:
    if passed:
        return (
            "EFV is supported as a world-model future-selection verifier on executable and diagnostic benchmarks."
        )
    if primary_ready and second_ready:
        return (
            "EFV has executable-benchmark evidence and a pending world-model diagnostic bridge; do not call it validated on world-model benchmarks yet."
        )
    if primary_ready:
        return (
            "EFV has a strong single-benchmark executable mechanism result; the world-model connection remains a planned diagnostic layer."
        )
    return "EFV is still a hypothesis; more executable evidence is required before making world-model claims."


def _prohibited_claims(passed: bool) -> list[str]:
    claims = [
        "Do not claim a new world model or policy.",
        "Do not claim real-robot validation.",
    ]
    if not passed:
        claims.extend(
            [
                "Do not title the paper as a world-model benchmark result yet.",
                "Do not claim validation on MiraBench, RoboTrustBench, or RoboWM-Bench until a diagnostic registry entry passes.",
                "Do not use visual plausibility or model-score-only artifacts as oracle success labels.",
            ]
        )
    return claims


def render_markdown(result: dict[str, Any], *, title: str = "World-Model Bridge Gate") -> str:
    lines = [
        f"# {title}",
        "",
        f"- passed: `{str(result['passed']).lower()}`",
        f"- claim level: `{result['claim_level']}`",
        f"- allowed claim: {result['allowed_claim']}",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in result["checks"]:
        detail = check["detail"]
        values = detail.get("benchmarks")
        if values is None:
            values = detail.get("benchmarks_with_required_fields", detail.get("required", "-"))
        if isinstance(values, list):
            text = ", ".join(str(item) for item in values) or "-"
        else:
            text = str(values)
        if "minimum" in detail:
            text = f"{text} / min {detail['minimum']}"
        lines.append(f"| `{check['name']}` | {'pass' if check['passed'] else 'fail'} | {text} |")

    lines.extend(
        [
            "",
            "## Diagnostic Artifact Status",
            "",
            "| Benchmark | Status | Pipeline | Registry | Missing |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in result["diagnostic_artifacts"]:
        lines.append(
            "| {benchmark} | {status} | {pipeline} | {registry} | {missing} |".format(
                benchmark=row.get("benchmark"),
                status=row.get("artifact_status"),
                pipeline="yes" if row.get("pipeline_ready") else "no",
                registry="yes" if row.get("registry_ready") else "no",
                missing=", ".join(row.get("missing") or []) or "-",
            )
        )
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in result["next_actions"])
    lines.extend(["", "## Prohibited Claims", ""])
    lines.extend(f"- {item}" for item in result["prohibited_claims"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-json", type=Path, required=True)
    parser.add_argument("--artifact-audit-json", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--min-primary-cases", type=int, default=16)
    parser.add_argument("--min-primary-tasks", type=int, default=4)
    parser.add_argument("--min-selector-margin", type=float, default=1.0)
    parser.add_argument("--no-require-second-executable", action="store_true")
    args = parser.parse_args()

    payload = _load_json(args.evidence_json)
    entries = payload.get("benchmarks")
    if not isinstance(entries, list):
        raise SystemExit("--evidence-json must contain a 'benchmarks' list")
    artifact_audit = _load_json(args.artifact_audit_json) if args.artifact_audit_json else None
    result = evaluate_bridge(
        entries,
        artifact_audit=artifact_audit,
        min_primary_cases=args.min_primary_cases,
        min_primary_tasks=args.min_primary_tasks,
        min_selector_margin=args.min_selector_margin,
        require_second_executable=not args.no_require_second_executable,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(result)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

