"""Audit RoboTwin2 manifest state-trace fields before selector evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.common import load_jsonl
from umm_reward_evaluator.benchmarks.robotwin2_selector_baselines import case_key, state_trace

STATIC_ACTOR_NAMES = {"robot", "scene", "viewer", "table", "wall", "ground"}


def snapshot_keys(row: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for snapshot in state_trace(row):
        if isinstance(snapshot, dict):
            keys.update(str(key) for key in snapshot)
    return keys


def actor_names(row: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for snapshot in state_trace(row):
        if not isinstance(snapshot, dict):
            continue
        value = snapshot.get("actor_names")
        if isinstance(value, list):
            names.update(str(item) for item in value)
    return names


def audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[case_key(row)].append(row)

    key_counts: Counter[str] = Counter()
    actor_name_counts: Counter[str] = Counter()
    candidate_source_counts: Counter[str] = Counter()
    case_rows = []
    for (task_name, case_id), candidates in sorted(grouped.items()):
        candidate_key_counts: Counter[str] = Counter()
        case_actor_names: set[str] = set()
        candidates_with_object_state = 0
        candidates_with_state_trace = 0
        candidates_with_actor_names = 0
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
            names = actor_names(row)
            if names:
                candidates_with_actor_names += 1
                case_actor_names.update(names)
                for name in names:
                    actor_name_counts[name] += 1
            if {"actor_pose_vector", "actor_pairwise_distances"}.issubset(keys):
                candidates_with_object_state += 1
        static_actor_names = sorted(name for name in case_actor_names if name in STATIC_ACTOR_NAMES)
        case_rows.append(
            {
                "task_name": task_name,
                "case_id": case_id,
                "candidates": len(candidates),
                "candidates_with_state_trace": candidates_with_state_trace,
                "candidates_with_object_state": candidates_with_object_state,
                "candidates_with_actor_names": candidates_with_actor_names,
                "object_state_complete": candidates_with_object_state == len(candidates),
                "actor_names": sorted(case_actor_names),
                "static_actor_names": static_actor_names,
                "static_actor_pollution": bool(static_actor_names),
                "state_trace_keys": dict(sorted(candidate_key_counts.items())),
            }
        )

    return {
        "cases": len(grouped),
        "candidates": len(rows),
        "state_trace_key_candidate_counts": dict(sorted(key_counts.items())),
        "actor_name_candidate_counts": dict(sorted(actor_name_counts.items())),
        "candidate_source_counts": dict(sorted(candidate_source_counts.items())),
        "cases_with_complete_object_state": sum(int(row["object_state_complete"]) for row in case_rows),
        "cases_with_actor_names": sum(int(row["candidates_with_actor_names"] == row["candidates"]) for row in case_rows),
        "cases_with_static_actor_pollution": sum(int(row["static_actor_pollution"]) for row in case_rows),
        "case_rows": case_rows,
    }


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# RoboTwin2 Trace Field Audit",
        "",
        f"- cases: {summary['cases']}",
        f"- candidates: {summary['candidates']}",
        f"- cases with complete object state: {summary['cases_with_complete_object_state']}/{summary['cases']}",
        f"- cases with complete actor names: {summary['cases_with_actor_names']}/{summary['cases']}",
        f"- cases with static actor pollution: {summary['cases_with_static_actor_pollution']}/{summary['cases']}",
        "",
        "## State Trace Keys",
        "",
        "| Key | Candidate count |",
        "| --- | ---: |",
    ]
    for key, count in summary["state_trace_key_candidate_counts"].items():
        lines.append(f"| `{key}` | {count} |")
    lines.extend(["", "## Actor Names", "", "| Actor name | Candidate count |", "| --- | ---: |"])
    for name, count in summary["actor_name_candidate_counts"].items():
        lines.append(f"| `{name}` | {count} |")
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Task | Case | Candidates | Object-state candidates | Actor-name candidates | Static actors | Complete |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in summary["case_rows"]:
        complete = "yes" if row["object_state_complete"] else "no"
        static_actors = ", ".join(f"`{name}`" for name in row["static_actor_names"]) or "-"
        lines.append(
            f"| `{row['task_name']}` | `{row['case_id']}` | {row['candidates']} | "
            f"{row['candidates_with_object_state']} | {row['candidates_with_actor_names']} | {static_actors} | {complete} |"
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
