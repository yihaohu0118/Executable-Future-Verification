"""Paper-level readiness gate for RoboTwin2 executable-future evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.common import load_jsonl


FULL_TEMPLATE_SOURCES = {
    "full_expert_trace",
    "full_gripper_aware",
    "rank0",
}
FULL_TEMPLATE_MARKERS = ("full_expert", "full_gripper_aware", "demo_original", "rank0", "first_action")
HARD_POSITIVE_MARKERS = ("hard_positive", "positive_probe", "time_warp", "suffix_truncation")
MATCHED_NEGATIVE_MARKERS = ("matched_", "energy_matched", "negative_probe")
ENVELOPE_COLUMNS = (
    "gripper",
    "phase_gripper",
    "relation",
    "phase_relation_robot",
)
BASELINE_COLUMNS = (
    "rank0",
    "random",
    "energy",
    "smooth",
    "length",
    "action",
    "dtw_action",
    "dtw_gripper",
    "dtw_joint",
    "dtw_joint_gripper",
    "dtw_relation",
    "dtw_relation_joint_gripper",
)
RELATION_COLUMNS = (
    "relation",
    "phase_relation_robot",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _candidate_source(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("candidate_source", metadata.get("original_candidate_id", row.get("candidate_id", ""))))


def _is_full_template_source(source: str) -> bool:
    lowered = source.lower()
    return source in FULL_TEMPLATE_SOURCES or any(marker in lowered for marker in FULL_TEMPLATE_MARKERS)


def _is_explicit_hard_positive_source(source: str) -> bool:
    lowered = source.lower()
    return any(marker in lowered for marker in HARD_POSITIVE_MARKERS)


def _is_non_template_success(row: dict[str, Any]) -> bool:
    source = _candidate_source(row)
    if not bool(row.get("oracle_success")):
        return False
    if _is_full_template_source(source):
        return False
    return _is_explicit_hard_positive_source(source)


def _is_matched_negative(row: dict[str, Any]) -> bool:
    source = _candidate_source(row)
    return (not bool(row.get("oracle_success"))) and any(marker in source for marker in MATCHED_NEGATIVE_MARKERS)


def _case_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["task_name"]), str(row["case_id"])


def collect_manifest_evidence(manifests_dir: Path) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for path in sorted(manifests_dir.glob("*_targeted_energy_matched_manifest.jsonl")):
        rows = load_jsonl(path)
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[_case_key(row)].append(row)

        if rows:
            task_name = str(rows[0]["task_name"])
        else:
            task_name = path.name.removesuffix("_targeted_energy_matched_manifest.jsonl")

        non_template_success_cases = 0
        matched_negative_cases = 0
        energy_matched_negative_cases = 0
        unknown_success_cases = 0
        full_template_success_cases = 0
        source_counts: dict[str, int] = defaultdict(int)
        success_source_counts: dict[str, int] = defaultdict(int)
        failure_source_counts: dict[str, int] = defaultdict(int)
        for case_rows in grouped.values():
            if any(_is_non_template_success(row) for row in case_rows):
                non_template_success_cases += 1
            if any(_is_matched_negative(row) for row in case_rows):
                matched_negative_cases += 1
            if any((not bool(row.get("oracle_success"))) and "energy_matched" in _candidate_source(row) for row in case_rows):
                energy_matched_negative_cases += 1
            if any(bool(row.get("oracle_success")) and _is_full_template_source(_candidate_source(row)) for row in case_rows):
                full_template_success_cases += 1
            if any(
                bool(row.get("oracle_success"))
                and not _is_full_template_source(_candidate_source(row))
                and not _is_explicit_hard_positive_source(_candidate_source(row))
                for row in case_rows
            ):
                unknown_success_cases += 1
            for row in case_rows:
                source = _candidate_source(row)
                source_counts[source] += 1
                if row.get("oracle_success"):
                    success_source_counts[source] += 1
                else:
                    failure_source_counts[source] += 1

        evidence[task_name] = {
            "manifest_path": str(path),
            "cases": len(grouped),
            "non_template_success_cases": non_template_success_cases,
            "matched_negative_cases": matched_negative_cases,
            "energy_matched_negative_cases": energy_matched_negative_cases,
            "full_template_success_cases": full_template_success_cases,
            "unknown_success_cases": unknown_success_cases,
            "source_counts": dict(sorted(source_counts.items())),
            "success_source_counts": dict(sorted(success_source_counts.items())),
            "failure_source_counts": dict(sorted(failure_source_counts.items())),
        }
    return evidence


def collect_antitemplate_evidence(selectors_dir: Path) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for path in sorted(selectors_dir.glob("*_targeted_energy_matched_diagnostics.json")):
        payload = _load_json(path)
        by_task = payload.get("by_task") or {}
        if by_task:
            for task_name, counts in by_task.items():
                evidence[str(task_name)] = {
                    "diagnostics_path": str(path),
                    "cases": int(counts.get("cases", 0) or 0),
                    "diverse_non_full_success_cases": int(counts.get("diverse_non_full_success_cases", 0) or 0),
                    "matched_low_dtw_negative_cases": int(counts.get("matched_negative_cases", 0) or 0),
                }
        else:
            task_name = path.name.removesuffix("_targeted_energy_matched_diagnostics.json")
            overall = payload.get("overall") or {}
            evidence[task_name] = {
                "diagnostics_path": str(path),
                "cases": int(overall.get("cases", 0) or 0),
                "diverse_non_full_success_cases": int(overall.get("diverse_non_full_success_cases", 0) or 0),
                "matched_low_dtw_negative_cases": int(overall.get("matched_negative_cases", 0) or 0),
            }
    return evidence


def _rows_by_task(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = _load_json(path)
    return {str(row["task_name"]): row for row in payload.get("tasks", [])}


def _value(row: dict[str, Any] | None, key: str) -> float | None:
    if not row:
        return None
    value = row.get(key)
    return None if value is None else float(value)


def _max_value(row: dict[str, Any] | None, keys: tuple[str, ...]) -> float | None:
    values = [_value(row, key) for key in keys]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _check(name: str, passed: bool, detail: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def evaluate_paper_readiness(
    *,
    readiness_rows: dict[str, dict[str, Any]],
    selector_rows: dict[str, dict[str, Any]],
    manifest_evidence: dict[str, dict[str, Any]],
    antitemplate_evidence: dict[str, dict[str, Any]] | None = None,
    min_base_ready_tasks: int = 4,
    min_relation_ready_tasks: int = 1,
    min_non_template_success_tasks: int = 2,
    min_matched_negative_tasks: int = 3,
    min_diverse_antitemplate_tasks: int = 2,
    min_low_dtw_negative_tasks: int = 2,
    min_strong_envelope_tasks: int = 3,
    min_relation_rescue_tasks: int = 1,
    min_relation_coverage: float = 1.0,
    min_selector_margin: float = 0.5,
) -> dict[str, Any]:
    antitemplate_evidence = antitemplate_evidence or {}
    all_tasks = sorted(set(readiness_rows) | set(selector_rows) | set(manifest_evidence) | set(antitemplate_evidence))
    task_rows: list[dict[str, Any]] = []
    for task in all_tasks:
        ready = readiness_rows.get(task, {})
        selector = selector_rows.get(task, {})
        manifest = manifest_evidence.get(task, {})
        antitemplate = antitemplate_evidence.get(task, {})
        best_envelope = _max_value(selector, ENVELOPE_COLUMNS)
        strongest_baseline = _max_value(selector, BASELINE_COLUMNS)
        best_relation = _max_value(selector, RELATION_COLUMNS)
        gripper = _value(selector, "gripper")
        dtw_gripper = _value(selector, "dtw_gripper")
        relation_coverage = _value(selector, "relation_min_coverage")

        strong_envelope = (
            best_envelope is not None
            and strongest_baseline is not None
            and best_envelope >= strongest_baseline + min_selector_margin
        )
        relation_rescue = (
            best_relation is not None
            and gripper is not None
            and relation_coverage is not None
            and relation_coverage >= min_relation_coverage
            and best_relation >= gripper + min_selector_margin
            and (dtw_gripper is None or best_relation >= dtw_gripper + min_selector_margin)
        )
        task_rows.append(
            {
                "task_name": task,
                "cases": int(ready.get("cases", selector.get("cases", manifest.get("cases", 0)) or 0)),
                "base_gate_passed": bool(ready.get("base_gate_passed")),
                "relation_gate_passed": bool(ready.get("relation_gate_passed")),
                "oracle_better": int(ready.get("oracle_better", 0) or 0),
                "non_template_success_cases": int(manifest.get("non_template_success_cases", 0) or 0),
                "full_template_success_cases": int(manifest.get("full_template_success_cases", 0) or 0),
                "unknown_success_cases": int(manifest.get("unknown_success_cases", 0) or 0),
                "matched_negative_cases": int(manifest.get("matched_negative_cases", 0) or 0),
                "energy_matched_negative_cases": int(manifest.get("energy_matched_negative_cases", 0) or 0),
                "diverse_antitemplate_success_cases": int(antitemplate.get("diverse_non_full_success_cases", 0) or 0),
                "matched_low_dtw_negative_cases": int(antitemplate.get("matched_low_dtw_negative_cases", 0) or 0),
                "best_envelope": best_envelope,
                "strongest_baseline": strongest_baseline,
                "strong_envelope": strong_envelope,
                "best_relation": best_relation,
                "gripper": gripper,
                "dtw_gripper": dtw_gripper,
                "relation_min_coverage": relation_coverage,
                "relation_rescue": relation_rescue,
            }
        )

    base_ready_tasks = [row["task_name"] for row in task_rows if row["base_gate_passed"] and row["oracle_better"] > 0]
    relation_ready_tasks = [row["task_name"] for row in task_rows if row["relation_gate_passed"]]
    non_template_tasks = [row["task_name"] for row in task_rows if row["non_template_success_cases"] > 0]
    matched_negative_tasks = [row["task_name"] for row in task_rows if row["matched_negative_cases"] > 0]
    diverse_antitemplate_tasks = [row["task_name"] for row in task_rows if row["diverse_antitemplate_success_cases"] > 0]
    low_dtw_negative_tasks = [row["task_name"] for row in task_rows if row["matched_low_dtw_negative_cases"] > 0]
    strong_envelope_tasks = [row["task_name"] for row in task_rows if row["strong_envelope"]]
    relation_rescue_tasks = [row["task_name"] for row in task_rows if row["relation_rescue"]]

    checks = [
        _check("base_ready_tasks", len(base_ready_tasks) >= min_base_ready_tasks, {"tasks": base_ready_tasks, "minimum": min_base_ready_tasks}),
        _check(
            "relation_ready_tasks",
            len(relation_ready_tasks) >= min_relation_ready_tasks,
            {"tasks": relation_ready_tasks, "minimum": min_relation_ready_tasks},
        ),
        _check(
            "non_template_success_tasks",
            len(non_template_tasks) >= min_non_template_success_tasks,
            {"tasks": non_template_tasks, "minimum": min_non_template_success_tasks},
        ),
        _check(
            "matched_negative_tasks",
            len(matched_negative_tasks) >= min_matched_negative_tasks,
            {"tasks": matched_negative_tasks, "minimum": min_matched_negative_tasks},
        ),
        _check(
            "diverse_antitemplate_success_tasks",
            len(diverse_antitemplate_tasks) >= min_diverse_antitemplate_tasks,
            {
                "tasks": diverse_antitemplate_tasks,
                "minimum": min_diverse_antitemplate_tasks,
                "meaning": "successful candidates are not just the full expert trace under DTW",
            },
        ),
        _check(
            "matched_low_dtw_negative_tasks",
            len(low_dtw_negative_tasks) >= min_low_dtw_negative_tasks,
            {
                "tasks": low_dtw_negative_tasks,
                "minimum": min_low_dtw_negative_tasks,
                "meaning": "failures exist near the expert trace, so nearest-template matching is not enough",
            },
        ),
        _check(
            "strong_envelope_tasks",
            len(strong_envelope_tasks) >= min_strong_envelope_tasks,
            {
                "tasks": strong_envelope_tasks,
                "minimum": min_strong_envelope_tasks,
                "min_selector_margin": min_selector_margin,
            },
        ),
        _check(
            "relation_rescue_tasks",
            len(relation_rescue_tasks) >= min_relation_rescue_tasks,
            {
                "tasks": relation_rescue_tasks,
                "minimum": min_relation_rescue_tasks,
                "min_selector_margin": min_selector_margin,
                "min_relation_coverage": min_relation_coverage,
            },
        ),
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "tasks": task_rows,
        "thresholds": {
            "min_base_ready_tasks": min_base_ready_tasks,
            "min_relation_ready_tasks": min_relation_ready_tasks,
            "min_non_template_success_tasks": min_non_template_success_tasks,
            "min_matched_negative_tasks": min_matched_negative_tasks,
            "min_diverse_antitemplate_tasks": min_diverse_antitemplate_tasks,
            "min_low_dtw_negative_tasks": min_low_dtw_negative_tasks,
            "min_strong_envelope_tasks": min_strong_envelope_tasks,
            "min_relation_rescue_tasks": min_relation_rescue_tasks,
            "min_relation_coverage": min_relation_coverage,
            "min_selector_margin": min_selector_margin,
        },
    }


def render_markdown(result: dict[str, Any], *, title: str = "RoboTwin2 Paper Readiness Gate") -> str:
    lines = [
        f"# {title}",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in result["checks"]:
        detail = check["detail"]
        tasks = ", ".join(detail.get("tasks", [])) or "-"
        lines.append(f"| `{check['name']}` | {'pass' if check['passed'] else 'fail'} | {tasks} / min {detail.get('minimum')} |")

    lines.extend(
        [
            "",
            "| Task | Cases | Base | Relation | Oracle better | Non-template succ | Unknown succ | Matched neg | Diverse anti-template succ | Low-DTW neg | Best env | Strong base | Relation rescue |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in result["tasks"]:
        lines.append(
            "| {task} | {cases} | {base} | {relation} | {oracle} | {positive} | {unknown} | {negative} | {diverse} | {low_dtw} | {env} | {baseline} | {rescue} |".format(
                task=row["task_name"],
                cases=row["cases"],
                base="pass" if row["base_gate_passed"] else "fail",
                relation="pass" if row["relation_gate_passed"] else "fail",
                oracle=row["oracle_better"],
                positive=row["non_template_success_cases"],
                unknown=row["unknown_success_cases"],
                negative=row["matched_negative_cases"],
                diverse=row["diverse_antitemplate_success_cases"],
                low_dtw=row["matched_low_dtw_negative_cases"],
                env="-" if row["best_envelope"] is None else f"{row['best_envelope']:.1f}",
                baseline="-" if row["strongest_baseline"] is None else f"{row['strongest_baseline']:.1f}",
                rescue="yes" if row["relation_rescue"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- This gate is deliberately stricter than the per-task main-table gate.",
            "- Passing it means the RoboTwin2 evidence covers clean headroom, shortcut-controlled candidates, distance-based anti-template controls, selector margins, and at least one relation-rescue mechanism task.",
            "- Failing it does not invalidate diagnostics; it means the result set should not yet be used as the main ICLR evidence package.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--selectors-dir", type=Path)
    parser.add_argument("--manifests-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--min-base-ready-tasks", type=int, default=4)
    parser.add_argument("--min-relation-ready-tasks", type=int, default=1)
    parser.add_argument("--min-non-template-success-tasks", type=int, default=2)
    parser.add_argument("--min-matched-negative-tasks", type=int, default=3)
    parser.add_argument("--min-diverse-antitemplate-tasks", type=int, default=2)
    parser.add_argument("--min-low-dtw-negative-tasks", type=int, default=2)
    parser.add_argument("--min-strong-envelope-tasks", type=int, default=3)
    parser.add_argument("--min-relation-rescue-tasks", type=int, default=1)
    parser.add_argument("--min-relation-coverage", type=float, default=1.0)
    parser.add_argument("--min-selector-margin", type=float, default=0.5)
    args = parser.parse_args()

    if args.run_root is None and (args.selectors_dir is None or args.manifests_dir is None):
        raise SystemExit("provide --run-root, or both --selectors-dir and --manifests-dir")
    selectors_dir = args.selectors_dir or args.run_root / "selectors"
    manifests_dir = args.manifests_dir or args.run_root / "manifests"
    result = evaluate_paper_readiness(
        readiness_rows=_rows_by_task(selectors_dir / "robotwin2_readiness_report.json"),
        selector_rows=_rows_by_task(selectors_dir / "robotwin2_selector_table.json"),
        manifest_evidence=collect_manifest_evidence(manifests_dir),
        antitemplate_evidence=collect_antitemplate_evidence(selectors_dir),
        min_base_ready_tasks=args.min_base_ready_tasks,
        min_relation_ready_tasks=args.min_relation_ready_tasks,
        min_non_template_success_tasks=args.min_non_template_success_tasks,
        min_matched_negative_tasks=args.min_matched_negative_tasks,
        min_diverse_antitemplate_tasks=args.min_diverse_antitemplate_tasks,
        min_low_dtw_negative_tasks=args.min_low_dtw_negative_tasks,
        min_strong_envelope_tasks=args.min_strong_envelope_tasks,
        min_relation_rescue_tasks=args.min_relation_rescue_tasks,
        min_relation_coverage=args.min_relation_coverage,
        min_selector_margin=args.min_selector_margin,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    markdown = render_markdown(result)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
