"""Audit RoboTwin2 manifest state-trace fields before selector evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.common import load_jsonl
from umm_reward_evaluator.benchmarks.robotwin2_selector_baselines import case_key, state_trace


def snapshot_keys(row: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for snapshot in state_trace(row):
        if isinstance(snapshot, dict):
            keys.update(str(key) for key in snapshot)
    return keys


def audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[case_key(row)].append(row)

    key_counts: Counter[str] = Counter()
    candidate_source_counts: Counter[str] = Counter()
    case_rows = []
    for (task_name, case_id), candidates in sorted(grouped.items()):
        candidate_key_counts: Counter[str] = Counter()
        candidates_with_object_state = 0
        candidates_with_state_trace = 0
        for row in candidates:
            metadata = row.get("metadata") or {}
            source = str(metadata.get("candidate_source", row.get("candidate_id", "")))
            candidate_source_counts[source] += 1
            keys = snapshot_keys(row)
            for key in keys:
                key_counts[key] += 1
                candidate_key_counts[key] += 1
            if state_trace(row):
                candidates_with_state_trace += 1
            if {"actor_pose_vector", "actor_pairwise_distances"}.issubset(keys):
                candidates_with_object_state += 1
        case_rows.append(
            {
                "task_name": task_name,
                "case_id": case_id,
                "candidates": len(candidates),
                "candidates_with_state_trace": candidates_with_state_trace,
                "candidates_with_object_state": candidates_with_object_state,
                "object_state_complete": candidates_with_object_state == len(candidates),
                "state_trace_keys": dict(sorted(candidate_key_counts.items())),
            }
        )

    return {
        "cases": len(grouped),
        "candidates": len(rows),
        "state_trace_key_candidate_counts": dict(sorted(key_counts.items())),
        "candidate_source_counts": dict(sorted(candidate_source_counts.items())),
        "cases_with_complete_object_state": sum(int(row["object_state_complete"]) for row in case_rows),
        "case_rows": case_rows,
    }


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# RoboTwin2 Trace Field Audit",
        "",
        f"- cases: {summary['cases']}",
        f"- candidates: {summary['candidates']}",
        f"- cases with complete object state: {summary['cases_with_complete_object_state']}/{summary['cases']}",
        "",
        "## State Trace Keys",
        "",
        "| Key | Candidate count |",
        "| --- | ---: |",
    ]
    for key, count in summary["state_trace_key_candidate_counts"].items():
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## Cases", "", "| Task | Case | Candidates | Object-state candidates | Complete |", "| --- | --- | ---: | ---: | --- |"])
    for row in summary["case_rows"]:
        complete = "yes" if row["object_state_complete"] else "no"
        lines.append(
            f"| `{row['task_name']}` | `{row['case_id']}` | {row['candidates']} | "
            f"{row['candidates_with_object_state']} | {complete} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    summary = audit_rows(load_jsonl(args.manifest))
    summary["manifest"] = str(args.manifest)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("cases", "candidates", "cases_with_complete_object_state")}, indent=2))


if __name__ == "__main__":
    main()
