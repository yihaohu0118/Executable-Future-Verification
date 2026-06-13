"""Audit public artifact readiness for world-model diagnostic benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.iclr_evidence_stack_gate import DIAGNOSTIC_LAYERS, _load_json


REQUIRED_DIAGNOSTIC_CONTROLS = {
    "rank0",
    "random",
    "energy_or_magnitude",
    "action_only",
    "candidate_id_or_rank_remap",
    "oracle_judgment_labels",
    "proxy_or_rank0_failure",
    "visual_or_model_score_proxy",
}


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def _missing_controls(entry: dict[str, Any]) -> list[str]:
    return sorted(REQUIRED_DIAGNOSTIC_CONTROLS - set(entry.get("shortcut_controls") or []))


def _benchmark_note(benchmark: str, cases: int) -> tuple[str, list[str]]:
    if benchmark == "MiraBench" and cases <= 0:
        return "public_judgments_not_found", ["public_multi_candidate_records", "judgment_labels", "model_score_proxy"]
    if benchmark == "RoboTrustBench" and 0 < cases < 100:
        return "prompt_subset_only", ["full_or_paper_scale_release", "multi_candidate_judgments"]
    if benchmark == "RoboWM-Bench":
        return "conditional_unstable_replay", ["pinned_official_replay_patch", "gt_replay_ceiling"]
    return "registry_entry_only", []


def audit_entry(entry: dict[str, Any]) -> dict[str, Any]:
    benchmark = str(entry.get("benchmark", "unknown"))
    cases = _as_int(entry.get("cases"))
    tasks = _as_int(entry.get("tasks"))
    rank0 = _as_float(entry.get("rank0_success"))
    oracle = _as_float(entry.get("oracle_success"))
    method = _as_float(entry.get("method_success"))
    baseline = _as_float(entry.get("best_non_oracle_baseline_success"))
    missing_controls = _missing_controls(entry)
    note, missing_artifacts = _benchmark_note(benchmark, cases)

    checks = {
        "has_public_case_count": cases > 0,
        "has_task_count": tasks > 0,
        "has_oracle_headroom": oracle > rank0,
        "has_method_margin": method > baseline,
        "has_required_controls": not missing_controls,
        "is_registry_passed": entry.get("status") == "passed",
    }

    missing: list[str] = []
    if not checks["has_public_case_count"]:
        missing.append("cases")
    if not checks["has_task_count"]:
        missing.append("tasks")
    if not checks["has_oracle_headroom"]:
        missing.append("oracle_headroom")
    if not checks["has_method_margin"]:
        missing.append("method_margin")
    if missing_controls:
        missing.append("shortcut_controls")
    if entry.get("status") != "passed":
        missing.append("passed_registry_proposal")
    missing.extend(item for item in missing_artifacts if item not in missing)

    pipeline_ready = (
        checks["has_public_case_count"]
        and checks["has_task_count"]
        and "multi_candidate_judgments" not in missing
        and "pinned_official_replay_patch" not in missing
        and "gt_replay_ceiling" not in missing
    )
    registry_ready = all(checks.values())

    if registry_ready:
        status = "ready_for_evidence_stack"
    elif pipeline_ready:
        status = "ready_for_pipeline_or_gate"
    elif cases > 0:
        status = "adapter_or_subset_only"
    else:
        status = "blocked_public_artifacts"

    return {
        "benchmark": benchmark,
        "year": entry.get("year"),
        "layer": entry.get("layer"),
        "cases": cases,
        "tasks": tasks,
        "artifact_note": note,
        "artifact_status": status,
        "pipeline_ready": pipeline_ready,
        "registry_ready": registry_ready,
        "missing": missing,
        "missing_controls": missing_controls,
        "source_urls": entry.get("source_urls") or [],
        "next_action": _next_action(benchmark, status, missing),
    }


def _next_action(benchmark: str, status: str, missing: list[str]) -> str:
    if status == "ready_for_evidence_stack":
        return "Freeze the evidence card unless upstream artifacts change."
    if benchmark == "MiraBench":
        return "Watch for official annotation/result artifacts; run world_model_diagnostic_pipeline once public records exist."
    if benchmark == "RoboTrustBench":
        if "multi_candidate_judgments" in missing:
            return "Use the public subset only to validate request generation; wait for full benchmark or collect candidate judgments before claiming results."
        return "Run world_model_diagnostic_pipeline and require verifier margin over the visual/model-score proxy."
    if benchmark == "RoboWM-Bench":
        return "Keep as conditional robustness diagnostic until official replay/evaluator stability is pinned."
    return "Convert public multi-candidate judgments with world_model_diagnostic_pipeline."


def build_audit(entries: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [audit_entry(entry) for entry in entries if entry.get("layer") in DIAGNOSTIC_LAYERS]
    priority = {"MiraBench": 1, "RoboTrustBench": 2, "RoboWM-Bench": 3}
    rows.sort(key=lambda row: (priority.get(str(row["benchmark"]), 9), str(row["benchmark"])))
    return {
        "diagnostic_artifacts": rows,
        "summary": {
            "diagnostic_benchmarks": len(rows),
            "pipeline_ready": sum(1 for row in rows if row["pipeline_ready"]),
            "registry_ready": sum(1 for row in rows if row["registry_ready"]),
            "blocked_public_artifacts": sum(1 for row in rows if row["artifact_status"] == "blocked_public_artifacts"),
        },
        "paper_rule": "Do not count a world-model diagnostic benchmark unless registry_ready is true and the evidence card validates.",
    }


def render_markdown(audit: dict[str, Any], *, title: str = "World-Model Artifact Audit") -> str:
    summary = audit["summary"]
    lines = [
        f"# {title}",
        "",
        f"- diagnostic benchmarks: {summary['diagnostic_benchmarks']}",
        f"- pipeline-ready: {summary['pipeline_ready']}",
        f"- registry-ready: {summary['registry_ready']}",
        f"- blocked on public artifacts: {summary['blocked_public_artifacts']}",
        f"- paper rule: {audit['paper_rule']}",
        "",
        "| Benchmark | Status | Note | Cases | Pipeline | Registry | Missing | Next action |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for row in audit["diagnostic_artifacts"]:
        lines.append(
            "| {benchmark} | {status} | {note} | {cases} | {pipeline} | {registry} | {missing} | {next_action} |".format(
                benchmark=row["benchmark"],
                status=row["artifact_status"],
                note=row["artifact_note"],
                cases=row["cases"],
                pipeline="yes" if row["pipeline_ready"] else "no",
                registry="yes" if row["registry_ready"] else "no",
                missing=", ".join(row["missing"]) or "-",
                next_action=row["next_action"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    payload = _load_json(args.evidence_json)
    entries = payload.get("benchmarks")
    if not isinstance(entries, list):
        raise SystemExit("--evidence-json must contain a 'benchmarks' list")
    audit = build_audit(entries)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(audit)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
