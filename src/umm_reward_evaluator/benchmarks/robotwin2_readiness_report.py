"""Summarize RoboTwin2 readiness gate outputs across tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _task_name_from_gate_path(path: Path) -> str:
    name = path.name
    for suffix in (
        "_targeted_energy_matched_main_table_gate.json",
        "_targeted_energy_matched_relation_gate.json",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _check_status(gate: dict[str, Any], check_name: str) -> bool | None:
    for check in gate.get("checks", []):
        if check.get("name") == check_name:
            return bool(check.get("passed"))
    return None


def _min_feature_coverage(gate: dict[str, Any]) -> float | None:
    coverages = gate.get("feature_coverages") or []
    if not coverages:
        return None
    return min(float(item.get("case_coverage_rate", 0.0)) for item in coverages)


def collect_reports(selectors_dir: Path) -> list[dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for path in sorted(selectors_dir.glob("*_targeted_energy_matched_main_table_gate.json")):
        task = _task_name_from_gate_path(path)
        tasks.setdefault(task, {})["base_gate_path"] = str(path)
        tasks[task]["base_gate"] = _load_json(path)
    for path in sorted(selectors_dir.glob("*_targeted_energy_matched_relation_gate.json")):
        task = _task_name_from_gate_path(path)
        tasks.setdefault(task, {})["relation_gate_path"] = str(path)
        tasks[task]["relation_gate"] = _load_json(path)

    rows: list[dict[str, Any]] = []
    for task, payload in sorted(tasks.items()):
        base = payload.get("base_gate")
        relation = payload.get("relation_gate")
        summary = (base or relation or {}).get("summary", {})
        rows.append(
            {
                "task_name": task,
                "cases": int(summary.get("cases", 0)),
                "rank0_success": int(summary.get("rank0_success", 0)),
                "oracle_success": int(summary.get("oracle_success", 0)),
                "oracle_better": int(summary.get("oracle_better", 0)),
                "base_gate_passed": bool(base.get("passed")) if isinstance(base, dict) else None,
                "relation_gate_passed": bool(relation.get("passed")) if isinstance(relation, dict) else None,
                "candidate_error_free": _check_status(base, "candidate_error_free") if isinstance(base, dict) else None,
                "relation_min_case_coverage": _min_feature_coverage(relation) if isinstance(relation, dict) else None,
                "base_gate_path": payload.get("base_gate_path"),
                "relation_gate_path": payload.get("relation_gate_path"),
            }
        )
    return rows


def render_markdown(rows: list[dict[str, Any]], *, title: str = "RoboTwin2 Readiness Report") -> str:
    lines = [
        f"# {title}",
        "",
        "| Task | Cases | Rank0 | Oracle | Oracle better | Base gate | Relation gate | Relation coverage |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for row in rows:
        cases = int(row["cases"])
        relation_coverage = row["relation_min_case_coverage"]
        coverage_text = "-" if relation_coverage is None else f"{relation_coverage:.2f}"
        lines.append(
            "| {task} | {cases} | {rank0}/{cases} | {oracle}/{cases} | {better}/{cases} | {base} | {relation} | {coverage} |".format(
                task=row["task_name"],
                cases=cases,
                rank0=row["rank0_success"],
                oracle=row["oracle_success"],
                better=row["oracle_better"],
                base="pass" if row["base_gate_passed"] else "fail",
                relation="pass" if row["relation_gate_passed"] else "fail",
                coverage=coverage_text,
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `Base gate` must pass before a task can be used for any main-table selector result.",
            "- `Relation gate` must pass before object-relation selector numbers can be claimed.",
            "- A task with base pass but relation fail can still be used as a gripper/action diagnostic.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selectors-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--title", default="RoboTwin2 Readiness Report")
    args = parser.parse_args()

    rows = collect_reports(args.selectors_dir)
    payload = {"selectors_dir": str(args.selectors_dir), "tasks": rows}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(rows, title=args.title)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
