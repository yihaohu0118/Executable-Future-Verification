"""Build compact RoboTwin2 selector tables from rank-randomization sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SELECTOR_COLUMNS = (
    ("rank0", "rank0"),
    ("random", "random_expected"),
    ("energy", "heuristic:energy_sum_max"),
    ("smooth", "heuristic:smoothness_max"),
    ("length", "heuristic:length_max"),
    ("action", "prototype:action_distribution:same_task:nearest_positive"),
    ("linear_action", "linear_probe:action_distribution:same_task:ridge_l2_1"),
    ("gripper", "prototype:gripper_distribution:same_task:nearest_positive"),
    ("phase_gripper", "prototype:phase_gripper_distribution:same_task:nearest_positive"),
    ("linear_gripper", "linear_probe:gripper_distribution:same_task:ridge_l2_1"),
    ("linear_phase_gripper", "linear_probe:phase_gripper_distribution:same_task:ridge_l2_1"),
    ("linear_phase_joint_gripper", "linear_probe:phase_joint_gripper_distribution:all_tasks:ridge_l2_1"),
    ("dtw_action", "trace_distance:dtw_action:same_task:nearest_positive"),
    ("dtw_gripper", "trace_distance:dtw_gripper:same_task:nearest_positive"),
    ("dtw_joint", "trace_distance:dtw_joint:all_tasks:nearest_positive"),
    ("dtw_joint_gripper", "trace_distance:dtw_joint_gripper:all_tasks:nearest_positive"),
    ("relation", "prototype:object_relation_distribution:same_task:nearest_positive"),
    ("phase_relation_robot", "prototype:phase_object_relation_joint_gripper_distribution:same_task:nearest_pos_neg"),
    ("linear_phase_relation_robot", "linear_probe:phase_object_relation_joint_gripper_distribution:same_task:ridge_l2_1"),
    ("dtw_relation", "trace_distance:dtw_object_relation:same_task:nearest_positive"),
    ("dtw_relation_joint_gripper", "trace_distance:dtw_object_relation_joint_gripper:same_task:nearest_positive"),
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _task_name_from_sweep_path(path: Path) -> str:
    suffix = "_targeted_energy_matched_rankrand_sweep.json"
    return path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem


def _case_count(sweep: dict[str, Any]) -> int:
    seed_results = sweep.get("seed_results") or []
    if not seed_results:
        return 0
    selectors = seed_results[0].get("selectors") or []
    if not selectors:
        return 0
    return int(selectors[0].get("cases", 0))


def _aggregate_by_selector(sweep: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["selector"]): row for row in (sweep.get("aggregate") or {}).get("selectors", [])}


def _selector_success(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    return float(row.get("mean_success", 0.0))


def _selector_coverage(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    if "min_feature_case_coverage" not in row:
        return None
    return float(row["min_feature_case_coverage"])


def collect_selector_rows(selectors_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(selectors_dir.glob("*_targeted_energy_matched_rankrand_sweep.json")):
        sweep = _load_json(path)
        by_selector = _aggregate_by_selector(sweep)
        item: dict[str, Any] = {
            "task_name": _task_name_from_sweep_path(path),
            "cases": _case_count(sweep),
            "sweep_path": str(path),
        }
        relation_coverages: list[float] = []
        for column, selector in SELECTOR_COLUMNS:
            selector_row = by_selector.get(selector)
            item[column] = _selector_success(selector_row)
            coverage = _selector_coverage(selector_row)
            if coverage is not None and "relation" in column:
                relation_coverages.append(coverage)
        item["relation_min_coverage"] = min(relation_coverages) if relation_coverages else None
        rows.append(item)
    return rows


def _fmt(value: float | int | None, cases: int | None = None) -> str:
    if value is None:
        return "-"
    if cases is not None:
        return f"{value:.1f}/{cases}"
    return f"{value:.2f}"


def render_markdown(rows: list[dict[str, Any]], *, title: str = "RoboTwin2 Selector Table") -> str:
    lines = [
        f"# {title}",
        "",
        "| Task | Cases | Rank0 | Random | Energy | Smooth | Length | Lin action | Gripper | Lin grip | Lin phase grip | Lin phase J+G | DTW J+G | Relation+robot | Lin rel+robot | Rel cov |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        cases = int(row.get("cases", 0))
        lines.append(
            "| {task} | {cases} | {rank0} | {random} | {energy} | {smooth} | {length} | {linear_action} | {gripper} | {linear_gripper} | {linear_phase_gripper} | {linear_phase_joint_gripper} | {dtw_joint_gripper} | {phase_relation_robot} | {linear_phase_relation_robot} | {coverage} |".format(
                task=row["task_name"],
                cases=cases,
                rank0=_fmt(row.get("rank0"), cases),
                random=_fmt(row.get("random"), cases),
                energy=_fmt(row.get("energy"), cases),
                smooth=_fmt(row.get("smooth"), cases),
                length=_fmt(row.get("length"), cases),
                linear_action=_fmt(row.get("linear_action"), cases),
                gripper=_fmt(row.get("gripper"), cases),
                linear_gripper=_fmt(row.get("linear_gripper"), cases),
                linear_phase_gripper=_fmt(row.get("linear_phase_gripper"), cases),
                linear_phase_joint_gripper=_fmt(row.get("linear_phase_joint_gripper"), cases),
                dtw_joint_gripper=_fmt(row.get("dtw_joint_gripper"), cases),
                phase_relation_robot=_fmt(row.get("phase_relation_robot"), cases),
                linear_phase_relation_robot=_fmt(row.get("linear_phase_relation_robot"), cases),
                coverage=_fmt(row.get("relation_min_coverage")),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- Values are mean successes over anonymous rank/candidate-ID remap seeds.",
            "- DTW columns are nearest-positive template baselines, not evidence that template matching has been ruled out.",
            "- Relation columns require relation coverage; `Rel cov` below 1.00 means the relation number is diagnostic only.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selectors-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--title", default="RoboTwin2 Selector Table")
    args = parser.parse_args()

    rows = collect_selector_rows(args.selectors_dir)
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
