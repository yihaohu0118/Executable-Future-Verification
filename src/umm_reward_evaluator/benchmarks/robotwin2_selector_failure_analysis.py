"""Analyze RoboTwin2 selector choices by candidate source and failure mode."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.common import load_jsonl
from umm_reward_evaluator.benchmarks.randomize_planner_rank import randomize_manifest_rows
from umm_reward_evaluator.benchmarks.robotwin2_rank_randomization_sweep import (
    DEFAULT_HEURISTICS,
    DEFAULT_PROTOTYPES,
    DEFAULT_TRACE_DISTANCES,
    evaluate_selectors,
)


def candidate_source(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("candidate_source", metadata.get("original_candidate_id", row.get("candidate_id", ""))))


def original_candidate_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("original_candidate_id", row.get("candidate_id", "")))


def row_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(row["task_name"]), str(row["case_id"]), str(row["candidate_id"])): row
        for row in rows
    }


def compact_selector_name(selector: str) -> str:
    return selector.replace("prototype:", "proto:").replace("trace_distance:", "dtw:")


def analyze_seed(rows: list[dict[str, Any]], *, seed: int, mode: str, remap_candidate_ids: bool) -> list[dict[str, Any]]:
    randomized = randomize_manifest_rows(rows, seed=seed, mode=mode, remap_candidate_ids=remap_candidate_ids)
    lookup = row_lookup(randomized)
    selector_results = evaluate_selectors(
        randomized,
        heuristics=DEFAULT_HEURISTICS,
        prototypes=DEFAULT_PROTOTYPES,
        trace_distances=DEFAULT_TRACE_DISTANCES,
    )
    analysis_rows = []
    for result in selector_results:
        selector = str(result["selector"])
        if selector == "random_expected":
            continue
        for selection in result["selections"]:
            key = (
                str(selection["task_name"]),
                str(selection["case_id"]),
                str(selection["selector_candidate_id"]),
            )
            selected = lookup[key]
            analysis_rows.append(
                {
                    "seed": seed,
                    "selector": selector,
                    "selector_short": compact_selector_name(selector),
                    "task_name": selection["task_name"],
                    "case_id": selection["case_id"],
                    "selected_candidate_id": selection["selector_candidate_id"],
                    "selected_original_candidate_id": original_candidate_id(selected),
                    "selected_candidate_source": candidate_source(selected),
                    "selected_success": bool(selected["oracle_success"]),
                    "rank0_success": bool(selection["rank0_success"]),
                    "oracle_success": bool(selection["oracle_success"]),
                }
            )
    return analysis_rows


def summarize_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_selector: dict[str, dict[str, Any]] = {}
    by_selector_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for selector in sorted({str(row["selector"]) for row in rows}):
        subset = [row for row in rows if row["selector"] == selector]
        cases = len(subset)
        successes = sum(int(row["selected_success"]) for row in subset)
        source_counts = Counter(str(row["selected_candidate_source"]) for row in subset)
        failure_sources = Counter(str(row["selected_candidate_source"]) for row in subset if not row["selected_success"])
        original_counts = Counter(str(row["selected_original_candidate_id"]) for row in subset)
        by_selector[selector] = {
            "runs": cases,
            "success": successes,
            "success_rate": successes / cases if cases else None,
            "source_counts": dict(sorted(source_counts.items())),
            "failure_source_counts": dict(sorted(failure_sources.items())),
            "original_candidate_counts": dict(sorted(original_counts.items())),
        }
        for task in sorted({str(row["task_name"]) for row in subset}):
            task_subset = [row for row in subset if row["task_name"] == task]
            task_cases = len(task_subset)
            task_success = sum(int(row["selected_success"]) for row in task_subset)
            by_selector_task[selector][task] = {
                "runs": task_cases,
                "success": task_success,
                "success_rate": task_success / task_cases if task_cases else None,
                "source_counts": dict(sorted(Counter(str(row["selected_candidate_source"]) for row in task_subset).items())),
                "failure_source_counts": dict(
                    sorted(Counter(str(row["selected_candidate_source"]) for row in task_subset if not row["selected_success"]).items())
                ),
            }

    hard_cases = [
        row
        for row in rows
        if not row["selected_success"]
        and row["selected_candidate_source"]
        in {
            "matched_gripper_timing_negative_probe",
            "matched_contact_direction_negative_probe",
            "suffix_truncation",
        }
    ]
    return {
        "by_selector": by_selector,
        "by_selector_task": by_selector_task,
        "hard_failure_rows": hard_cases,
    }


def markdown_report(summary: dict[str, Any], *, selectors: list[str] | None = None) -> str:
    selected = selectors or sorted(summary["by_selector"])
    lines = [
        "# RoboTwin2 Selector Failure Analysis",
        "",
        "| Selector | Success | Top selected sources | Failure sources |",
        "| --- | ---: | --- | --- |",
    ]
    for selector in selected:
        if selector not in summary["by_selector"]:
            continue
        item = summary["by_selector"][selector]

        def top(counter: dict[str, int], *, limit: int = 3) -> str:
            ordered = sorted(counter.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
            return ", ".join(f"{key} {value}" for key, value in ordered) if ordered else "-"

        lines.append(
            f"| `{compact_selector_name(selector)}` | {item['success']}/{item['runs']} | {top(item['source_counts'])} | {top(item['failure_source_counts'])} |"
        )
    lines.extend(["", "## Hard Failure Rows", ""])
    for row in summary["hard_failure_rows"][:50]:
        lines.append(
            f"- `{compact_selector_name(row['selector'])}` task={row['task_name']} case={row['case_id']} "
            f"selected={row['selected_original_candidate_id']} source={row['selected_candidate_source']}"
        )
    return "\n".join(lines) + "\n"


def run_analysis(
    rows: list[dict[str, Any]],
    *,
    seeds: list[int],
    mode: str,
    remap_candidate_ids: bool,
) -> dict[str, Any]:
    analysis_rows = []
    for seed in seeds:
        analysis_rows.extend(analyze_seed(rows, seed=seed, mode=mode, remap_candidate_ids=remap_candidate_ids))
    summary = summarize_analysis(analysis_rows)
    summary.update(
        {
            "mode": mode,
            "remap_candidate_ids": remap_candidate_ids,
            "seeds": seeds,
            "num_rows": len(analysis_rows),
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["random", "prefer_success", "prefer_failure", "failure_rank0_shuffle_rest"],
        default="failure_rank0_shuffle_rest",
    )
    parser.add_argument("--remap-candidate-ids", action="store_true")
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    summary = run_analysis(
        load_jsonl(args.manifest),
        seeds=seeds,
        mode=args.mode,
        remap_candidate_ids=args.remap_candidate_ids,
    )
    summary["manifest"] = str(args.manifest)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        focus = [
            "heuristic:energy_sum_max",
            "heuristic:smoothness_max",
            "prototype:phase_gripper_distribution:same_task:nearest_positive",
            "prototype:phase_gripper_distribution:same_task:nearest_pos_neg",
            "prototype:gripper_distribution:same_task:nearest_positive",
            "prototype:gripper_distribution:same_task:nearest_pos_neg",
            "trace_distance:dtw_gripper:same_task:nearest_positive",
            "trace_distance:dtw_joint_gripper:all_tasks:nearest_positive",
        ]
        args.output_md.write_text(markdown_report(summary, selectors=focus), encoding="utf-8")
    compact = {
        selector: {
            "success": item["success"],
            "runs": item["runs"],
            "failure_source_counts": item["failure_source_counts"],
        }
        for selector, item in summary["by_selector"].items()
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
