"""Build conservative evidence-registry entry proposals from gate outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_EXECUTABLE_CONTROLS = [
    "rank0",
    "random",
    "energy_or_magnitude",
    "action_only",
    "candidate_id_or_rank_remap",
]
DEFAULT_DIAGNOSTIC_CONTROLS = [
    "rank0",
    "random",
    "oracle_judgment_labels",
    "proxy_or_rank0_failure",
    "visual_or_model_score_proxy",
]
REQUIRED_DIAGNOSTIC_STACK_CONTROLS = set(DEFAULT_EXECUTABLE_CONTROLS) | set(DEFAULT_DIAGNOSTIC_CONTROLS)
ENVELOPE_COLUMNS = ("gripper", "phase_gripper", "relation", "phase_relation_robot", "dtw_relation")
BASELINE_COLUMNS = ("rank0", "random", "energy", "smooth", "action", "dtw_gripper")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def _max_numeric(row: dict[str, Any], keys: tuple[str, ...]) -> float:
    values = [_as_float(row.get(key), 0.0) for key in keys if row.get(key) is not None]
    return max(values) if values else 0.0


def _selector_by_name(selector_table: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("selector")): row for row in selector_table.get("selectors", [])}


def _tasks_by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("task_name")): row for row in payload.get("tasks", [])}


def propose_robotwin2_entry(
    *,
    readiness_report: dict[str, Any],
    selector_table: dict[str, Any],
    paper_gate: dict[str, Any],
    evidence: str | None = None,
) -> dict[str, Any]:
    readiness_by_task = _tasks_by_name(readiness_report)
    selector_by_task = _tasks_by_name(selector_table)
    task_names = sorted(set(readiness_by_task) | set(selector_by_task))
    cases = sum(_as_int(readiness_by_task.get(task, {}).get("cases"), _as_int(selector_by_task.get(task, {}).get("cases"))) for task in task_names)
    rank0_success = sum(_as_float(readiness_by_task.get(task, {}).get("rank0_success"), _as_float(selector_by_task.get(task, {}).get("rank0"))) for task in task_names)
    oracle_success = sum(_as_float(readiness_by_task.get(task, {}).get("oracle_success"), _as_float(selector_by_task.get(task, {}).get("cases"))) for task in task_names)
    method_success = sum(_max_numeric(selector_by_task.get(task, {}), ENVELOPE_COLUMNS) for task in task_names)
    best_baseline_success = sum(_max_numeric(selector_by_task.get(task, {}), BASELINE_COLUMNS) for task in task_names)

    return {
        "benchmark": "RoboTwin2",
        "year": 2025,
        "layer": "executable_second",
        "status": "passed" if paper_gate.get("passed") else "pending",
        "cases": cases,
        "tasks": len([task for task in task_names if _as_int(readiness_by_task.get(task, {}).get("cases"), _as_int(selector_by_task.get(task, {}).get("cases"))) > 0]),
        "rank0_success": rank0_success,
        "oracle_success": oracle_success,
        "method_success": method_success,
        "best_non_oracle_baseline_success": best_baseline_success,
        "shortcut_controls": list(DEFAULT_EXECUTABLE_CONTROLS),
        "evidence": evidence
        or (
            "Generated from RoboTwin2 readiness, selector-table, and paper-readiness gate outputs. "
            f"Paper gate passed={bool(paper_gate.get('passed'))}."
        ),
    }


def propose_diagnostic_entry(
    *,
    benchmark: str,
    year: int,
    layer: str,
    diagnostic_gate: dict[str, Any],
    selector_table: dict[str, Any],
    verifier_selector: str,
    readiness_gate: dict[str, Any] | None = None,
    extra_controls: list[str] | None = None,
    evidence: str | None = None,
) -> dict[str, Any]:
    selectors = _selector_by_name(selector_table)
    rank0 = selectors.get("rank0", {})
    random_expected = selectors.get("random_expected", {})
    proxy = selectors.get("planner_or_model_score", {})
    oracle = selectors.get("oracle", {})
    verifier = selectors.get(verifier_selector, {})
    summary = selector_table.get("summary") or {}
    cases = _as_int(summary.get("cases"), _as_int(diagnostic_gate.get("summary", {}).get("cases")))
    tasks = len(summary.get("tasks") or diagnostic_gate.get("summary", {}).get("tasks") or [])

    rank0_success = _as_float(rank0.get("selector_success"))
    oracle_success = _as_float(oracle.get("selector_success"), _as_float(diagnostic_gate.get("summary", {}).get("oracle_success")))
    verifier_success = _as_float(verifier.get("selector_success"))
    best_baseline = max(
        _as_float(rank0.get("selector_success")),
        _as_float(random_expected.get("selector_success")),
        _as_float(proxy.get("selector_success")),
    )
    controls = list(dict.fromkeys(DEFAULT_DIAGNOSTIC_CONTROLS + list(extra_controls or [])))
    selector_beats_proxy = verifier_success > _as_float(proxy.get("selector_success"))
    has_required_controls = REQUIRED_DIAGNOSTIC_STACK_CONTROLS.issubset(set(controls))
    readiness_passed = bool(readiness_gate.get("passed")) if readiness_gate is not None else None
    status = (
        "passed"
        if diagnostic_gate.get("passed")
        and selector_beats_proxy
        and has_required_controls
        and (readiness_passed is not False)
        else "pending"
    )

    return {
        "benchmark": benchmark,
        "year": int(year),
        "layer": layer,
        "status": status,
        "cases": cases,
        "tasks": tasks,
        "rank0_success": rank0_success,
        "oracle_success": oracle_success,
        "method_success": verifier_success,
        "best_non_oracle_baseline_success": best_baseline,
        "shortcut_controls": controls,
        "evidence": evidence
        or (
            "Generated from world-model diagnostic gate and selector-table outputs. "
            f"Diagnostic gate passed={bool(diagnostic_gate.get('passed'))}; verifier beats proxy={selector_beats_proxy}; "
            f"required controls present={has_required_controls}; diagnostic readiness passed={readiness_passed}."
        ),
    }


def render_markdown(entry: dict[str, Any], *, title: str = "ICLR Registry Entry Proposal") -> str:
    lines = [
        f"# {title}",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for key in (
        "benchmark",
        "year",
        "layer",
        "status",
        "cases",
        "tasks",
        "rank0_success",
        "oracle_success",
        "method_success",
        "best_non_oracle_baseline_success",
    ):
        lines.append(f"| `{key}` | {entry.get(key)} |")
    lines.extend(
        [
            f"| `shortcut_controls` | {', '.join(entry.get('shortcut_controls') or []) or '-'} |",
            "",
            "Evidence:",
            "",
            str(entry.get("evidence", "")),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    robotwin2 = subparsers.add_parser("robotwin2")
    robotwin2.add_argument("--readiness-json", type=Path, required=True)
    robotwin2.add_argument("--selector-table-json", type=Path, required=True)
    robotwin2.add_argument("--paper-gate-json", type=Path, required=True)
    robotwin2.add_argument("--evidence")
    robotwin2.add_argument("--output-json", type=Path)
    robotwin2.add_argument("--output-md", type=Path)

    diagnostic = subparsers.add_parser("diagnostic")
    diagnostic.add_argument("--benchmark", required=True)
    diagnostic.add_argument("--year", type=int, required=True)
    diagnostic.add_argument("--layer", required=True, choices=["world_model_diagnostic", "trust_diagnostic", "robustness_diagnostic"])
    diagnostic.add_argument("--diagnostic-gate-json", type=Path, required=True)
    diagnostic.add_argument("--selector-table-json", type=Path, required=True)
    diagnostic.add_argument("--diagnostic-readiness-json", type=Path)
    diagnostic.add_argument("--verifier-selector", required=True)
    diagnostic.add_argument("--shortcut-control", action="append", default=[])
    diagnostic.add_argument("--evidence")
    diagnostic.add_argument("--output-json", type=Path)
    diagnostic.add_argument("--output-md", type=Path)

    args = parser.parse_args()
    if args.mode == "robotwin2":
        entry = propose_robotwin2_entry(
            readiness_report=_load_json(args.readiness_json),
            selector_table=_load_json(args.selector_table_json),
            paper_gate=_load_json(args.paper_gate_json),
            evidence=args.evidence,
        )
    else:
        entry = propose_diagnostic_entry(
            benchmark=args.benchmark,
            year=args.year,
            layer=args.layer,
            diagnostic_gate=_load_json(args.diagnostic_gate_json),
            selector_table=_load_json(args.selector_table_json),
            readiness_gate=_load_json(args.diagnostic_readiness_json) if args.diagnostic_readiness_json else None,
            verifier_selector=args.verifier_selector,
            extra_controls=list(args.shortcut_control),
            evidence=args.evidence,
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(entry)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
