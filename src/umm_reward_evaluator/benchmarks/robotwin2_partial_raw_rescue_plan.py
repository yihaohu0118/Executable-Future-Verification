"""Plan which partial RoboTwin2 raw traces are worth completing."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad_json = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            if isinstance(payload, dict):
                rows.append(payload)
            else:
                bad_json += 1
    return rows, bad_json


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    return metadata if isinstance(metadata, dict) else {}


def _candidate_source(row: dict[str, Any]) -> str:
    return str(_metadata(row).get("candidate_source", row.get("candidate_id", "")))


def _has_candidate_error(row: dict[str, Any]) -> bool:
    return bool(row.get("candidate_error") or _metadata(row).get("candidate_error"))


def _has_state_trace(row: dict[str, Any]) -> bool:
    trace = _metadata(row).get("state_trace")
    return isinstance(trace, list) and bool(trace)


def _has_object_state(row: dict[str, Any]) -> bool:
    trace = _metadata(row).get("state_trace")
    if not isinstance(trace, list):
        return False
    keys: set[str] = set()
    for snapshot in trace:
        if isinstance(snapshot, dict):
            keys.update(str(key) for key in snapshot)
    return {"actor_pose_vector", "actor_pairwise_distances"}.issubset(keys)


def _seed_from_path(path: Path) -> str:
    stem = path.stem
    return stem.removeprefix("seed_")


def _classify_file(rows: list[dict[str, Any]], *, bad_json: int, required_candidates: int) -> str:
    if bad_json:
        return "invalid_json"
    if not rows:
        return "empty"
    if any(_has_candidate_error(row) for row in rows):
        return "candidate_error"
    if len(rows) < required_candidates:
        return "partial"
    if len(rows) == required_candidates:
        return "complete"
    return "oversized"


def _rescue_priority(item: dict[str, Any]) -> tuple[int, str]:
    if item["status"] != "partial":
        return 99, "not_partial"
    if item["candidate_error_rows"]:
        return 99, "candidate_errors"
    if item["success_rows"] == 0:
        return 40, "no_success_yet"
    if item["failure_rows"] == 0:
        return 35, "no_failure_yet"
    if item["state_trace_rows"] < item["rows"]:
        return 30, "missing_state_trace"
    if item["rows"] >= 8:
        return 1, "high_value_partial"
    return 5, "short_partial"


def build_rescue_plan(raw_root: Path, *, required_candidates_per_case: int = 24) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    task_summary: dict[str, Counter[str]] = defaultdict(Counter)
    for path in sorted(p for p in raw_root.glob("*/*.jsonl") if not p.name.startswith(".")):
        rows, bad_json = _load_jsonl(path)
        status = _classify_file(rows, bad_json=bad_json, required_candidates=required_candidates_per_case)
        source_counts: Counter[str] = Counter(_candidate_source(row) for row in rows)
        success_source_counts: Counter[str] = Counter(_candidate_source(row) for row in rows if row.get("success") or row.get("oracle_success"))
        item = {
            "path": str(path.relative_to(raw_root)),
            "task_name": path.parent.name,
            "seed": _seed_from_path(path),
            "rows": len(rows),
            "missing_candidates": max(required_candidates_per_case - len(rows), 0),
            "success_rows": sum(1 for row in rows if row.get("success") or row.get("oracle_success")),
            "failure_rows": sum(1 for row in rows if not (row.get("success") or row.get("oracle_success"))),
            "candidate_error_rows": sum(1 for row in rows if _has_candidate_error(row)),
            "bad_json_lines": bad_json,
            "state_trace_rows": sum(1 for row in rows if _has_state_trace(row)),
            "object_state_rows": sum(1 for row in rows if _has_object_state(row)),
            "source_counts": dict(sorted(source_counts.items())),
            "success_source_counts": dict(sorted(success_source_counts.items())),
            "status": status,
        }
        priority, reason = _rescue_priority(item)
        item["rescue_priority"] = priority
        item["rescue_reason"] = reason
        files.append(item)
        task_summary[path.parent.name][status] += 1

    rescue_files = sorted(
        [item for item in files if item["status"] == "partial"],
        key=lambda item: (int(item["rescue_priority"]), str(item["task_name"]), str(item["seed"])),
    )
    task_recommendations = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in rescue_files:
        by_task[item["task_name"]].append(item)
    for task, items in sorted(by_task.items()):
        best_priority = min(int(item["rescue_priority"]) for item in items)
        task_recommendations.append(
            {
                "task_name": task,
                "partial_files": len(items),
                "best_priority": best_priority,
                "best_reason": next(item["rescue_reason"] for item in items if int(item["rescue_priority"]) == best_priority),
                "seeds": [item["seed"] for item in items],
                "total_existing_rows": sum(int(item["rows"]) for item in items),
                "total_missing_candidates": sum(int(item["missing_candidates"]) for item in items),
                "success_rows": sum(int(item["success_rows"]) for item in items),
                "failure_rows": sum(int(item["failure_rows"]) for item in items),
                "object_state_rows": sum(int(item["object_state_rows"]) for item in items),
            }
        )
    task_recommendations.sort(key=lambda item: (int(item["best_priority"]), str(item["task_name"])))

    return {
        "raw_root": str(raw_root),
        "required_candidates_per_case": required_candidates_per_case,
        "files": files,
        "task_summary": {task: dict(sorted(counter.items())) for task, counter in sorted(task_summary.items())},
        "rescue_files": rescue_files,
        "task_recommendations": task_recommendations,
    }


def render_markdown(plan: dict[str, Any], *, title: str = "RoboTwin2 Partial Raw Rescue Plan") -> str:
    lines = [
        f"# {title}",
        "",
        f"- raw root: `{plan['raw_root']}`",
        f"- required candidates per case: {plan['required_candidates_per_case']}",
        "",
        "## Task Recommendations",
        "",
        "| Task | Partials | Seeds | Existing rows | Missing candidates | Success | Failure | Object-state rows | Priority reason |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in plan["task_recommendations"]:
        lines.append(
            "| {task} | {partials} | {seeds} | {rows} | {missing} | {success} | {failure} | {object_state} | {reason} |".format(
                task=row["task_name"],
                partials=row["partial_files"],
                seeds=", ".join(row["seeds"]),
                rows=row["total_existing_rows"],
                missing=row["total_missing_candidates"],
                success=row["success_rows"],
                failure=row["failure_rows"],
                object_state=row["object_state_rows"],
                reason=row["best_reason"],
            )
        )
    lines.extend(
        [
            "",
            "## Partial Files",
            "",
            "| File | Rows | Missing | Success | Failure | Trace rows | Object rows | Reason |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in plan["rescue_files"]:
        lines.append(
            "| `{path}` | {rows} | {missing} | {success} | {failure} | {trace} | {object_state} | {reason} |".format(
                path=row["path"],
                rows=row["rows"],
                missing=row["missing_candidates"],
                success=row["success_rows"],
                failure=row["failure_rows"],
                trace=row["state_trace_rows"],
                object_state=row["object_state_rows"],
                reason=row["rescue_reason"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- High-value partials already contain both successes and failures and only need candidate completion before manifest conversion.",
            "- Object-state rows must be nonzero before relation/contact verifier claims can be tested.",
            "- Partial rows should not be counted in paper tables until they are completed to the required candidate count.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--required-candidates-per-case", type=int, default=24)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    plan = build_rescue_plan(args.raw_root, required_candidates_per_case=args.required_candidates_per_case)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(plan)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
