"""Audit RoboTwin2 raw trace directories before manifest conversion."""

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


def _candidate_error_count(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        metadata = row.get("metadata") or {}
        if row.get("candidate_error") or metadata.get("candidate_error"):
            count += 1
    return count


def _file_status(row_count: int, *, bad_json: int, candidate_errors: int, required_candidates: int) -> str:
    if bad_json:
        return "invalid_json"
    if row_count == 0:
        return "empty"
    if candidate_errors:
        return "candidate_error"
    if row_count != required_candidates:
        return "partial"
    return "complete"


def audit_raw_root(raw_root: Path, *, required_candidates_per_case: int = 24) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    task_summary: dict[str, Counter[str]] = defaultdict(Counter)
    raw_files = sorted(path for path in raw_root.glob("*/*.jsonl") if not path.name.startswith("."))
    temp_files = sorted(raw_root.glob("**/.*.tmp.*"))

    for path in raw_files:
        rows, bad_json = _load_jsonl(path)
        candidate_errors = _candidate_error_count(rows)
        status = _file_status(
            len(rows),
            bad_json=bad_json,
            candidate_errors=candidate_errors,
            required_candidates=required_candidates_per_case,
        )
        task = path.parent.name
        task_summary[task][status] += 1
        files.append(
            {
                "path": str(path.relative_to(raw_root)),
                "task_name": task,
                "rows": len(rows),
                "success_rows": sum(1 for row in rows if row.get("success") or row.get("oracle_success")),
                "candidate_error_rows": candidate_errors,
                "bad_json_lines": bad_json,
                "required_candidates_per_case": required_candidates_per_case,
                "status": status,
            }
        )

    status_counts = Counter(str(item["status"]) for item in files)
    problem_files = [item for item in files if item["status"] != "complete"]
    ready = bool(files) and not problem_files and not temp_files
    return {
        "raw_root": str(raw_root),
        "required_candidates_per_case": required_candidates_per_case,
        "ready_for_manifest": ready,
        "num_files": len(files),
        "num_temp_files": len(temp_files),
        "status_counts": dict(sorted(status_counts.items())),
        "task_summary": {task: dict(sorted(counter.items())) for task, counter in sorted(task_summary.items())},
        "problem_files": problem_files,
        "temp_files": [str(path.relative_to(raw_root)) for path in temp_files],
        "files": files,
    }


def render_markdown(report: dict[str, Any], *, title: str = "RoboTwin2 Raw Integrity Report") -> str:
    lines = [
        f"# {title}",
        "",
        f"- raw root: `{report['raw_root']}`",
        f"- required candidates per case: {report['required_candidates_per_case']}",
        f"- ready for manifest: `{str(report['ready_for_manifest']).lower()}`",
        f"- files: {report['num_files']}",
        f"- temp files: {report['num_temp_files']}",
        "",
        "## Task Summary",
        "",
        "| Task | Complete | Partial | Empty | Candidate error | Invalid JSON |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, counts in report["task_summary"].items():
        lines.append(
            "| {task} | {complete} | {partial} | {empty} | {candidate_error} | {invalid_json} |".format(
                task=task,
                complete=counts.get("complete", 0),
                partial=counts.get("partial", 0),
                empty=counts.get("empty", 0),
                candidate_error=counts.get("candidate_error", 0),
                invalid_json=counts.get("invalid_json", 0),
            )
        )
    lines.extend(["", "## Problem Files", ""])
    if report["problem_files"]:
        lines.extend(
            "- `{path}`: {status}, rows={rows}, success={success_rows}, candidate_errors={candidate_error_rows}, bad_json={bad_json_lines}".format(
                **item
            )
            for item in report["problem_files"]
        )
    else:
        lines.append("- none")
    lines.extend(["", "## Temporary Files", ""])
    if report["temp_files"]:
        lines.extend(f"- `{path}`" for path in report["temp_files"])
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--required-candidates-per-case", type=int, default=24)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    report = audit_raw_root(args.raw_root, required_candidates_per_case=args.required_candidates_per_case)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    if not args.no_fail and not report["ready_for_manifest"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
