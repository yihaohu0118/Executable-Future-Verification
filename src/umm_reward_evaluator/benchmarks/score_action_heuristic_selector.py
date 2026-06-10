"""Evaluate deterministic action-statistic candidate selectors."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from umm_reward_evaluator.benchmarks.common import oracle_key
from umm_reward_evaluator.benchmarks.train_multitask_action_sequence_selector import load_labeled_rows


def action_array(row: dict[str, Any]) -> np.ndarray:
    actions = np.asarray(row.get("actions") or [], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] == 0:
        return np.zeros((1, 7), dtype=np.float32)
    return actions


def action_stats(row: dict[str, Any]) -> dict[str, float]:
    actions = action_array(row)
    diff = np.diff(actions, axis=0) if len(actions) > 1 else np.zeros_like(actions)
    metadata = row.get("metadata") or {}
    prior = metadata.get("conservative_prior_score")
    return {
        "energy_mean": float(np.mean(np.square(actions))),
        "energy_sum": float(np.sum(np.square(actions))),
        "abs_mean": float(np.mean(np.abs(actions))),
        "std_mean": float(np.mean(np.std(actions, axis=0))),
        "range_mean": float(np.mean(np.max(actions, axis=0) - np.min(actions, axis=0))),
        "smoothness": float(np.mean(np.square(diff))),
        "length": float(len(actions)),
        "planner_rank": float(row.get("candidate_rank_by_planner", 0)),
        "conservative_prior_score": float(prior) if prior is not None else 0.0,
    }


def heuristic_score(row: dict[str, Any], heuristic: str) -> float:
    stats = action_stats(row)
    if heuristic.endswith("_min"):
        return -stats[heuristic[: -len("_min")]]
    if heuristic.endswith("_max"):
        return stats[heuristic[: -len("_max")]]
    raise ValueError(f"Unknown heuristic {heuristic}")


HEURISTICS = (
    "energy_mean_max",
    "energy_mean_min",
    "energy_sum_max",
    "abs_mean_max",
    "std_mean_max",
    "range_mean_max",
    "smoothness_max",
    "smoothness_min",
    "planner_rank_max",
    "planner_rank_min",
    "conservative_prior_score_max",
    "conservative_prior_score_min",
)


def select_case(case_rows: list[dict[str, Any]], *, heuristic: str) -> dict[str, Any]:
    return max(
        case_rows,
        key=lambda row: (
            heuristic_score(row, heuristic),
            -int(row.get("candidate_rank_by_planner", 999)),
        ),
    )


def evaluate_heuristic(rows: list[dict[str, Any]], *, heuristic: str) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["task_label"]), str(row["case_id"])), []).append(row)

    by_task: dict[str, dict[str, Any]] = {}
    choice_counter: dict[str, Counter[str]] = {}
    selections: list[dict[str, Any]] = []
    for (task, case_id), case_rows in sorted(grouped.items()):
        rank0 = min(case_rows, key=lambda row: int(row.get("candidate_rank_by_planner", 999)))
        selected = select_case(case_rows, heuristic=heuristic)
        oracle = max(case_rows, key=oracle_key)
        task_summary = by_task.setdefault(
            task,
            {
                "cases": 0,
                "rank0_success": 0,
                "heuristic_success": 0,
                "oracle_success": 0,
                "heuristic_oracle_match": 0,
            },
        )
        task_summary["cases"] += 1
        task_summary["rank0_success"] += int(bool(rank0["oracle_success"]))
        task_summary["heuristic_success"] += int(bool(selected["oracle_success"]))
        task_summary["oracle_success"] += int(bool(oracle["oracle_success"]))
        task_summary["heuristic_oracle_match"] += int(selected["candidate_id"] == oracle["candidate_id"])
        choice_counter.setdefault(task, Counter())[str(selected["candidate_id"])] += 1
        selections.append(
            {
                "task_label": task,
                "case_id": case_id,
                "rank0_candidate_id": rank0["candidate_id"],
                "heuristic_candidate_id": selected["candidate_id"],
                "oracle_candidate_id": oracle["candidate_id"],
                "rank0_success": bool(rank0["oracle_success"]),
                "heuristic_success": bool(selected["oracle_success"]),
                "oracle_success": bool(oracle["oracle_success"]),
                "heuristic_score": heuristic_score(selected, heuristic),
            }
        )

    for task, summary in by_task.items():
        cases = int(summary["cases"])
        summary["heuristic_success_rate"] = summary["heuristic_success"] / cases if cases else 0.0
        summary["oracle_success_rate"] = summary["oracle_success"] / cases if cases else 0.0
        summary["choice_counts"] = dict(choice_counter.get(task, Counter()))

    overall = {
        "cases": sum(item["cases"] for item in by_task.values()),
        "rank0_success": sum(item["rank0_success"] for item in by_task.values()),
        "heuristic_success": sum(item["heuristic_success"] for item in by_task.values()),
        "oracle_success": sum(item["oracle_success"] for item in by_task.values()),
        "heuristic_oracle_match": sum(item["heuristic_oracle_match"] for item in by_task.values()),
    }
    cases = int(overall["cases"])
    overall["heuristic_success_rate"] = overall["heuristic_success"] / cases if cases else 0.0
    overall["oracle_success_rate"] = overall["oracle_success"] / cases if cases else 0.0
    return {
        "heuristic": heuristic,
        "overall": overall,
        "by_task": by_task,
        "selections": selections,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True, help="Path or task_label=path. Can repeat.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heuristic", action="append", choices=list(HEURISTICS))
    args = parser.parse_args()

    rows = load_labeled_rows(args.manifest)
    heuristics = args.heuristic or list(HEURISTICS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for heuristic in heuristics:
        result = evaluate_heuristic(rows, heuristic=heuristic)
        summaries.append({key: value for key, value in result.items() if key != "selections"})
        (args.output_dir / f"{heuristic}_selections.json").write_text(
            json.dumps(result["selections"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    payload = {"heuristics": summaries}
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
