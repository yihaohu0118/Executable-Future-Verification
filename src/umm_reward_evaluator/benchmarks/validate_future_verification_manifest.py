"""Validate candidate manifests for executable-future verification experiments."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.common import load_jsonl, oracle_key


REQUIRED_FIELDS = (
    "benchmark",
    "suite",
    "task_name",
    "case_id",
    "candidate_id",
    "candidate_rank_by_planner",
    "actions",
    "oracle_success",
)

RECOMMENDED_METADATA_FIELDS = (
    "future_source",
    "future_representation",
    "verification_target",
)


def _case_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("task_label", row.get("task_name", "<missing_task>"))), str(row.get("case_id", "<missing_case>"))


def validate_rows(rows: list[dict[str, Any]], *, require_metadata: bool) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        for field in REQUIRED_FIELDS:
            if field not in row:
                errors.append(f"row {idx} missing required field {field!r}")
        if "candidate_rank_by_planner" in row:
            try:
                int(row["candidate_rank_by_planner"])
            except Exception:
                errors.append(f"row {idx} has non-integer candidate_rank_by_planner")
        actions = row.get("actions")
        if not isinstance(actions, list):
            errors.append(f"row {idx} actions must be a list")
        elif not actions:
            warnings.append(f"row {idx} actions is empty")
        if "oracle_success" in row and not isinstance(row["oracle_success"], bool):
            errors.append(f"row {idx} oracle_success must be boolean")

        metadata = row.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            errors.append(f"row {idx} metadata must be an object when present")
            metadata = {}
        if require_metadata:
            for field in RECOMMENDED_METADATA_FIELDS:
                if field not in metadata:
                    errors.append(f"row {idx} metadata missing {field!r}")
        if "task_name" in row and "case_id" in row:
            grouped[_case_key(row)].append(row)

    cases = 0
    rank0_success = 0
    oracle_success = 0
    oracle_better = 0
    candidate_counts: Counter[int] = Counter()
    task_counts: Counter[str] = Counter()
    for (task, case_id), case_rows in sorted(grouped.items()):
        cases += 1
        task_counts[task] += 1
        candidate_counts[len(case_rows)] += 1
        ranks = [int(row.get("candidate_rank_by_planner", 999999)) for row in case_rows]
        if len(ranks) != len(set(ranks)):
            errors.append(f"case {task}/{case_id} has duplicate candidate ranks")
        candidate_ids = [str(row.get("candidate_id")) for row in case_rows]
        if len(candidate_ids) != len(set(candidate_ids)):
            errors.append(f"case {task}/{case_id} has duplicate candidate IDs")
        rank0_rows = [row for row in case_rows if int(row.get("candidate_rank_by_planner", 999999)) == 0]
        if len(rank0_rows) != 1:
            errors.append(f"case {task}/{case_id} has {len(rank0_rows)} rank0 candidates")
            rank0 = min(case_rows, key=lambda row: int(row.get("candidate_rank_by_planner", 999999)))
        else:
            rank0 = rank0_rows[0]
        oracle = max(case_rows, key=oracle_key)
        rank0_success += int(bool(rank0.get("oracle_success")))
        oracle_success += int(bool(oracle.get("oracle_success")))
        oracle_better += int(oracle_key(oracle) > oracle_key(rank0))

    summary = {
        "rows": len(rows),
        "cases": cases,
        "tasks": dict(sorted(task_counts.items())),
        "candidate_count_histogram": dict(sorted(candidate_counts.items())),
        "rank0_success": rank0_success,
        "oracle_success": oracle_success,
        "oracle_better": oracle_better,
        "rank0_success_rate": rank0_success / cases if cases else 0.0,
        "oracle_success_rate": oracle_success / cases if cases else 0.0,
        "warnings": warnings[:50],
        "num_warnings": len(warnings),
        "num_errors": len(errors),
    }
    return summary, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-future-metadata", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    summary, errors = validate_rows(rows, require_metadata=args.require_future_metadata)
    payload = {"summary": summary, "errors": errors[:100]}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
