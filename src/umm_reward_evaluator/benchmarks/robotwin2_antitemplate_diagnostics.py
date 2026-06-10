"""Anti-template diagnostics for RoboTwin2 executable-future manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from umm_reward_evaluator.benchmarks.common import load_jsonl, oracle_key
from umm_reward_evaluator.benchmarks.robotwin2_selector_baselines import (
    dtw_distance,
    group_cases,
    normalize_sequences,
    state_dims,
    trace_sequence,
)


DEFAULT_FULL_EXPERT_IDS = ("full_gripper_aware", "full_expert", "demo_original")


def original_candidate_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("original_candidate_id", row.get("candidate_id", "")))


def source_label(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    for key in ("candidate_source", "future_source", "policy_name"):
        if key in metadata:
            return str(metadata[key])
    return str(row.get("policy_name", row.get("future_source", "")))


def is_full_expert(row: dict[str, Any], full_expert_ids: tuple[str, ...]) -> bool:
    candidate = original_candidate_id(row).lower()
    source = source_label(row).lower()
    return any(token.lower() in candidate or token.lower() in source for token in full_expert_ids)


def case_dtw_to_full(
    case_rows: list[dict[str, Any]],
    *,
    feature_mode: str,
    full_expert_ids: tuple[str, ...],
) -> dict[str, float | None]:
    dims = state_dims(case_rows, ["joint_action_vector", "left_gripper", "right_gripper"])
    sequences = [trace_sequence(row, feature_mode, dims=dims) for row in case_rows]
    full_indexes = [index for index, row in enumerate(case_rows) if is_full_expert(row, full_expert_ids)]
    if not full_indexes:
        return {str(row["candidate_id"]): None for row in case_rows}
    sequences, _ = normalize_sequences(sequences, [])
    full_sequences = [sequences[index] for index in full_indexes]
    distances: dict[str, float | None] = {}
    for row, sequence in zip(case_rows, sequences, strict=True):
        distances[str(row["candidate_id"])] = min(dtw_distance(sequence, full_sequence) for full_sequence in full_sequences)
    return distances


def summarize_distance(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def diagnose_manifest(
    rows: list[dict[str, Any]],
    *,
    feature_mode: str = "dtw_joint_gripper",
    full_expert_ids: tuple[str, ...] = DEFAULT_FULL_EXPERT_IDS,
    equivalence_eps: float = 1e-6,
) -> dict[str, Any]:
    grouped = group_cases(rows)
    cases = []
    global_counts: Counter[str] = Counter()
    by_task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    distances_by_bucket: dict[str, list[float]] = defaultdict(list)

    for key, case_rows in sorted(grouped.items()):
        task, case_id = key
        full_rows = [row for row in case_rows if is_full_expert(row, full_expert_ids)]
        success_rows = [row for row in case_rows if row.get("oracle_success")]
        non_full_success_rows = [row for row in success_rows if not is_full_expert(row, full_expert_ids)]
        failure_rows = [row for row in case_rows if not row.get("oracle_success")]
        oracle = max(case_rows, key=oracle_key)
        rank0 = min(case_rows, key=lambda row: int(row.get("candidate_rank_by_planner", 999999)))
        distances = case_dtw_to_full(case_rows, feature_mode=feature_mode, full_expert_ids=full_expert_ids)

        full_success = any(row.get("oracle_success") for row in full_rows)
        non_full_success = bool(non_full_success_rows)
        failure_distances = [float(distances[str(row["candidate_id"])]) for row in failure_rows if distances[str(row["candidate_id"])] is not None]
        non_full_success_distances = [
            float(distances[str(row["candidate_id"])])
            for row in non_full_success_rows
            if distances[str(row["candidate_id"])] is not None
        ]
        diverse_non_full_success_rows = [
            row
            for row in non_full_success_rows
            if distances[str(row["candidate_id"])] is not None
            and float(distances[str(row["candidate_id"])]) > equivalence_eps
        ]
        close_threshold = max(non_full_success_distances) if non_full_success_distances else None
        matched_negative_rows = []
        if close_threshold is not None:
            matched_negative_rows = [
                row
                for row in failure_rows
                if distances[str(row["candidate_id"])] is not None
                and float(distances[str(row["candidate_id"])]) <= float(close_threshold)
            ]

        global_counts["cases"] += 1
        global_counts["rank0_success"] += int(bool(rank0.get("oracle_success")))
        global_counts["oracle_success"] += int(bool(oracle.get("oracle_success")))
        global_counts["full_expert_present"] += int(bool(full_rows))
        global_counts["full_expert_success_cases"] += int(full_success)
        global_counts["non_full_success_cases"] += int(non_full_success)
        global_counts["diverse_non_full_success_cases"] += int(bool(diverse_non_full_success_rows))
        global_counts["multi_success_cases"] += int(len(success_rows) > 1)
        global_counts["matched_negative_cases"] += int(bool(matched_negative_rows))

        by_task_counts[task]["cases"] += 1
        by_task_counts[task]["rank0_success"] += int(bool(rank0.get("oracle_success")))
        by_task_counts[task]["oracle_success"] += int(bool(oracle.get("oracle_success")))
        by_task_counts[task]["non_full_success_cases"] += int(non_full_success)
        by_task_counts[task]["diverse_non_full_success_cases"] += int(bool(diverse_non_full_success_rows))
        by_task_counts[task]["matched_negative_cases"] += int(bool(matched_negative_rows))

        for row in non_full_success_rows:
            distance = distances[str(row["candidate_id"])]
            if distance is not None:
                distances_by_bucket["non_full_success_to_full"].append(float(distance))
        for row in diverse_non_full_success_rows:
            distance = distances[str(row["candidate_id"])]
            if distance is not None:
                distances_by_bucket["diverse_non_full_success_to_full"].append(float(distance))
        for row in failure_rows:
            distance = distances[str(row["candidate_id"])]
            if distance is not None:
                distances_by_bucket["failure_to_full"].append(float(distance))
        for row in matched_negative_rows:
            distance = distances[str(row["candidate_id"])]
            if distance is not None:
                distances_by_bucket["matched_negative_to_full"].append(float(distance))

        cases.append(
            {
                "task_name": task,
                "case_id": case_id,
                "rank0_success": bool(rank0.get("oracle_success")),
                "oracle_success": bool(oracle.get("oracle_success")),
                "oracle_candidate_id": oracle.get("candidate_id"),
                "full_expert_candidates": [row.get("candidate_id") for row in full_rows],
                "success_candidates": [
                    {
                        "candidate_id": row.get("candidate_id"),
                        "original_candidate_id": original_candidate_id(row),
                        "is_full_expert": is_full_expert(row, full_expert_ids),
                        "dtw_to_full": distances[str(row["candidate_id"])],
                    }
                    for row in success_rows
                ],
                "matched_negative_candidates": [
                    {
                        "candidate_id": row.get("candidate_id"),
                        "original_candidate_id": original_candidate_id(row),
                        "dtw_to_full": distances[str(row["candidate_id"])],
                    }
                    for row in matched_negative_rows
                ],
                "non_full_success_count": len(non_full_success_rows),
                "diverse_non_full_success_count": len(diverse_non_full_success_rows),
                "matched_negative_count": len(matched_negative_rows),
                "failure_distance_summary": summarize_distance(failure_distances),
                "non_full_success_distance_summary": summarize_distance(non_full_success_distances),
            }
        )

    total_cases = max(int(global_counts["cases"]), 1)
    summary = {
        "feature_mode": feature_mode,
        "full_expert_ids": list(full_expert_ids),
        "equivalence_eps": equivalence_eps,
        "overall": dict(global_counts),
        "rates": {
            "rank0_success_rate": global_counts["rank0_success"] / total_cases,
            "oracle_success_rate": global_counts["oracle_success"] / total_cases,
            "non_full_success_case_rate": global_counts["non_full_success_cases"] / total_cases,
            "diverse_non_full_success_case_rate": global_counts["diverse_non_full_success_cases"] / total_cases,
            "matched_negative_case_rate": global_counts["matched_negative_cases"] / total_cases,
        },
        "by_task": {task: dict(counts) for task, counts in sorted(by_task_counts.items())},
        "distance_summaries": {
            bucket: summarize_distance(values)
            for bucket, values in sorted(distances_by_bucket.items())
        },
        "cases": cases,
    }
    return summary


def markdown_report(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    rates = summary["rates"]
    lines = [
        "# RoboTwin2 Anti-Template Diagnostics",
        "",
        f"- feature mode: `{summary['feature_mode']}`",
        f"- cases: {overall.get('cases', 0)}",
        f"- rank0 success: {overall.get('rank0_success', 0)}/{overall.get('cases', 0)}",
        f"- oracle success: {overall.get('oracle_success', 0)}/{overall.get('cases', 0)}",
        f"- non-full-expert success cases: {overall.get('non_full_success_cases', 0)}/{overall.get('cases', 0)} ({rates['non_full_success_case_rate']:.3f})",
        f"- diverse non-full-expert success cases: {overall.get('diverse_non_full_success_cases', 0)}/{overall.get('cases', 0)} ({rates['diverse_non_full_success_case_rate']:.3f})",
        f"- matched low-DTW negative cases: {overall.get('matched_negative_cases', 0)}/{overall.get('cases', 0)} ({rates['matched_negative_case_rate']:.3f})",
        "",
        "## Distance Summaries",
        "",
        "| Bucket | Count | Mean | Median | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for bucket, stats in summary["distance_summaries"].items():
        fmt = lambda value: "NA" if value is None else f"{float(value):.4f}"
        lines.append(
            f"| {bucket} | {stats['count']} | {fmt(stats['mean'])} | {fmt(stats['median'])} | {fmt(stats['min'])} | {fmt(stats['max'])} |"
        )
    lines.extend(
        [
            "",
            "## By Task",
            "",
            "| Task | Cases | Non-full success cases | Diverse non-full success cases | Matched negative cases |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for task, counts in summary["by_task"].items():
        lines.append(
            f"| {task} | {counts.get('cases', 0)} | {counts.get('non_full_success_cases', 0)} | {counts.get('diverse_non_full_success_cases', 0)} | {counts.get('matched_negative_cases', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A strong paper table needs both non-full-expert successes and matched low-DTW failures.",
            "If non-full successes exist but matched negatives are rare, nearest-expert trajectory matching remains a plausible shortcut.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--feature-mode", default="dtw_joint_gripper", choices=["dtw_action", "dtw_joint", "dtw_gripper", "dtw_joint_gripper"])
    parser.add_argument("--full-expert-id", action="append")
    parser.add_argument("--equivalence-eps", type=float, default=1e-6)
    args = parser.parse_args()

    full_expert_ids = tuple(args.full_expert_id) if args.full_expert_id else DEFAULT_FULL_EXPERT_IDS
    summary = diagnose_manifest(
        load_jsonl(args.manifest),
        feature_mode=args.feature_mode,
        full_expert_ids=full_expert_ids,
        equivalence_eps=args.equivalence_eps,
    )
    summary["manifest"] = str(args.manifest)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("feature_mode", "overall", "rates")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
