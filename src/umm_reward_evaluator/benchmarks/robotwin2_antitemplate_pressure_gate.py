"""Gate RoboTwin2 evidence against nearest-template shortcut explanations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.robotwin2_paper_readiness_gate import (
    ENVELOPE_COLUMNS,
    collect_antitemplate_evidence,
    collect_manifest_evidence,
)


TEMPLATE_BASELINE_COLUMNS = (
    "dtw_action",
    "dtw_gripper",
    "dtw_joint",
    "dtw_joint_gripper",
    "dtw_relation",
    "dtw_relation_joint_gripper",
)

SIMPLE_BASELINE_COLUMNS = (
    "rank0",
    "random",
    "energy",
    "smooth",
    "length",
    "action",
    "linear_action",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _rows_by_task(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = _load_json(path)
    return {str(row["task_name"]): row for row in payload.get("tasks", [])}


def _numeric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    return None if value is None else float(value)


def _best(row: dict[str, Any], columns: tuple[str, ...]) -> tuple[str | None, float | None]:
    values = [(column, _numeric(row, column)) for column in columns]
    values = [(column, value) for column, value in values if value is not None]
    if not values:
        return None, None
    return max(values, key=lambda item: item[1])


def _check(name: str, passed: bool, detail: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def evaluate_pressure_gate(
    *,
    selector_rows: dict[str, dict[str, Any]],
    manifest_evidence: dict[str, dict[str, Any]],
    antitemplate_evidence: dict[str, dict[str, Any]],
    min_pressure_tasks: int = 2,
    min_method_advantage_tasks: int = 2,
    min_diverse_success_cases_per_task: int = 1,
    min_low_dtw_negative_cases_per_task: int = 1,
    min_method_vs_template_margin: float = 0.5,
    template_oracle_tolerance: float = 0.25,
) -> dict[str, Any]:
    all_tasks = sorted(set(selector_rows) | set(manifest_evidence) | set(antitemplate_evidence))
    task_rows: list[dict[str, Any]] = []
    for task in all_tasks:
        selector = selector_rows.get(task, {})
        manifest = manifest_evidence.get(task, {})
        antitemplate = antitemplate_evidence.get(task, {})
        cases = int(selector.get("cases", manifest.get("cases", antitemplate.get("cases", 0))) or 0)
        non_template_success_cases = int(manifest.get("non_template_success_cases", 0) or 0)
        matched_negative_cases = int(manifest.get("matched_negative_cases", 0) or 0)
        diverse_success_cases = int(antitemplate.get("diverse_non_full_success_cases", 0) or 0)
        low_dtw_negative_cases = int(antitemplate.get("matched_low_dtw_negative_cases", 0) or 0)
        best_method_name, best_method = _best(selector, ENVELOPE_COLUMNS)
        best_template_name, best_template = _best(selector, TEMPLATE_BASELINE_COLUMNS)
        best_simple_name, best_simple = _best(selector, SIMPLE_BASELINE_COLUMNS)

        has_pressure = (
            diverse_success_cases >= min_diverse_success_cases_per_task
            and low_dtw_negative_cases >= min_low_dtw_negative_cases_per_task
        )
        method_beats_template = (
            has_pressure
            and best_method is not None
            and best_template is not None
            and best_method >= best_template + min_method_vs_template_margin
        )
        template_near_oracle = (
            cases > 0
            and best_template is not None
            and best_template >= max(float(cases) - template_oracle_tolerance, 0.0)
        )
        simple_shortcut_risk = (
            best_simple is not None
            and best_method is not None
            and best_simple >= best_method - min_method_vs_template_margin
        )
        risk_reasons = []
        if not has_pressure:
            risk_reasons.append("missing_anti_template_pressure")
        if has_pressure and not method_beats_template:
            risk_reasons.append("dtw_template_not_beaten")
        if has_pressure and template_near_oracle:
            risk_reasons.append("dtw_template_near_oracle")
        if simple_shortcut_risk:
            risk_reasons.append("simple_shortcut_close_to_method")

        task_rows.append(
            {
                "task_name": task,
                "cases": cases,
                "non_template_success_cases": non_template_success_cases,
                "matched_negative_cases": matched_negative_cases,
                "diverse_success_cases": diverse_success_cases,
                "low_dtw_negative_cases": low_dtw_negative_cases,
                "has_pressure": has_pressure,
                "best_method": best_method,
                "best_method_name": best_method_name,
                "best_template": best_template,
                "best_template_name": best_template_name,
                "best_simple": best_simple,
                "best_simple_name": best_simple_name,
                "method_beats_template": method_beats_template,
                "template_near_oracle": template_near_oracle,
                "simple_shortcut_risk": simple_shortcut_risk,
                "risk_reasons": risk_reasons,
            }
        )

    pressure_tasks = [row["task_name"] for row in task_rows if row["has_pressure"]]
    method_advantage_tasks = [row["task_name"] for row in task_rows if row["method_beats_template"]]
    template_oracle_risk_tasks = [row["task_name"] for row in task_rows if row["has_pressure"] and row["template_near_oracle"]]
    simple_shortcut_risk_tasks = [row["task_name"] for row in task_rows if row["simple_shortcut_risk"]]
    checks = [
        _check(
            "anti_template_pressure_tasks",
            len(pressure_tasks) >= min_pressure_tasks,
            {
                "tasks": pressure_tasks,
                "minimum": min_pressure_tasks,
                "meaning": "task has both diverse successes and low-DTW failures",
            },
        ),
        _check(
            "method_beats_template_tasks",
            len(method_advantage_tasks) >= min_method_advantage_tasks,
            {
                "tasks": method_advantage_tasks,
                "minimum": min_method_advantage_tasks,
                "min_margin": min_method_vs_template_margin,
            },
        ),
        _check(
            "no_template_oracle_risk",
            not template_oracle_risk_tasks,
            {
                "tasks": template_oracle_risk_tasks,
                "template_oracle_tolerance": template_oracle_tolerance,
                "meaning": "nearest-template DTW should not already solve pressured tasks",
            },
        ),
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "tasks": task_rows,
        "risk_summary": {
            "pressure_tasks": pressure_tasks,
            "method_advantage_tasks": method_advantage_tasks,
            "template_oracle_risk_tasks": template_oracle_risk_tasks,
            "simple_shortcut_risk_tasks": simple_shortcut_risk_tasks,
        },
        "thresholds": {
            "min_pressure_tasks": min_pressure_tasks,
            "min_method_advantage_tasks": min_method_advantage_tasks,
            "min_diverse_success_cases_per_task": min_diverse_success_cases_per_task,
            "min_low_dtw_negative_cases_per_task": min_low_dtw_negative_cases_per_task,
            "min_method_vs_template_margin": min_method_vs_template_margin,
            "template_oracle_tolerance": template_oracle_tolerance,
        },
    }


def render_markdown(result: dict[str, Any], *, title: str = "RoboTwin2 Anti-Template Pressure Gate") -> str:
    lines = [
        f"# {title}",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in result["checks"]:
        detail = check["detail"]
        tasks = ", ".join(detail.get("tasks", [])) or "-"
        minimum = detail.get("minimum")
        if minimum is None:
            lines.append(f"| `{check['name']}` | {'pass' if check['passed'] else 'fail'} | {tasks} |")
        else:
            lines.append(f"| `{check['name']}` | {'pass' if check['passed'] else 'fail'} | {tasks} / min {minimum} |")

    lines.extend(
        [
            "",
            "| Task | Cases | Diverse succ | Low-DTW neg | Best method | Best DTW template | Best simple | Risk |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in result["tasks"]:
        risk = ", ".join(row["risk_reasons"]) or "none"
        best_method = "-" if row["best_method"] is None else f"{row['best_method_name']}={row['best_method']:.1f}"
        best_template = "-" if row["best_template"] is None else f"{row['best_template_name']}={row['best_template']:.1f}"
        best_simple = "-" if row["best_simple"] is None else f"{row['best_simple_name']}={row['best_simple']:.1f}"
        lines.append(
            "| {task} | {cases} | {diverse} | {low_dtw} | {method} | {template} | {simple} | {risk} |".format(
                task=row["task_name"],
                cases=row["cases"],
                diverse=row["diverse_success_cases"],
                low_dtw=row["low_dtw_negative_cases"],
                method=best_method,
                template=best_template,
                simple=best_simple,
                risk=risk,
            )
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- This gate targets the strongest reviewer objection: EFV may only be selecting the trajectory closest to an expert template.",
            "- A task only creates anti-template pressure when it contains both successful non-template futures and failed low-DTW futures.",
            "- Passing this gate means the envelope/relation verifier beats DTW-template baselines on pressured tasks; it does not replace the main paper-readiness gate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--selectors-dir", type=Path)
    parser.add_argument("--manifests-dir", type=Path)
    parser.add_argument("--selector-table-json", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--min-pressure-tasks", type=int, default=2)
    parser.add_argument("--min-method-advantage-tasks", type=int, default=2)
    parser.add_argument("--min-diverse-success-cases-per-task", type=int, default=1)
    parser.add_argument("--min-low-dtw-negative-cases-per-task", type=int, default=1)
    parser.add_argument("--min-method-vs-template-margin", type=float, default=0.5)
    parser.add_argument("--template-oracle-tolerance", type=float, default=0.25)
    parser.add_argument("--fail-on-risk", action="store_true")
    args = parser.parse_args()

    if args.run_root is None and (args.selectors_dir is None or args.manifests_dir is None):
        raise SystemExit("provide --run-root, or both --selectors-dir and --manifests-dir")
    selectors_dir = args.selectors_dir or args.run_root / "selectors"
    manifests_dir = args.manifests_dir or args.run_root / "manifests"
    selector_table_json = args.selector_table_json or selectors_dir / "robotwin2_selector_table.json"
    result = evaluate_pressure_gate(
        selector_rows=_rows_by_task(selector_table_json),
        manifest_evidence=collect_manifest_evidence(manifests_dir),
        antitemplate_evidence=collect_antitemplate_evidence(selectors_dir),
        min_pressure_tasks=args.min_pressure_tasks,
        min_method_advantage_tasks=args.min_method_advantage_tasks,
        min_diverse_success_cases_per_task=args.min_diverse_success_cases_per_task,
        min_low_dtw_negative_cases_per_task=args.min_low_dtw_negative_cases_per_task,
        min_method_vs_template_margin=args.min_method_vs_template_margin,
        template_oracle_tolerance=args.template_oracle_tolerance,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(result)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    if args.fail_on_risk and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
