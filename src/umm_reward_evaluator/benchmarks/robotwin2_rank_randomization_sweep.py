"""Multi-seed rank/candidate-ID randomization sweeps for RoboTwin2 selectors."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from umm_reward_evaluator.benchmarks.common import load_jsonl
from umm_reward_evaluator.benchmarks.randomize_planner_rank import randomize_manifest_rows
from umm_reward_evaluator.benchmarks.robotwin2_selector_baselines import (
    HEURISTICS,
    PROTOTYPE_FEATURES,
    PROTOTYPE_MODES,
    PROTOTYPE_SCOPES,
    evaluate_candidate_id,
    evaluate_heuristic,
    evaluate_prototype,
    evaluate_random_expected,
    evaluate_rank0,
)


DEFAULT_HEURISTICS = ("smoothness_max", "energy_sum_max", "length_max")
DEFAULT_PROTOTYPES = (
    ("action_distribution", "same_task", "nearest_positive"),
    ("gripper_distribution", "same_task", "nearest_positive"),
    ("gripper_distribution", "all_tasks", "nearest_positive"),
    ("phase_gripper_distribution", "same_task", "nearest_positive"),
    ("phase_gripper_distribution", "all_tasks", "nearest_positive"),
    ("phase_joint_distribution", "all_tasks", "nearest_positive"),
    ("phase_joint_gripper_distribution", "all_tasks", "nearest_positive"),
)


def parse_prototype_config(value: str) -> tuple[str, str, str]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("prototype config must be feature:scope:mode")
    feature_mode, scope, prototype_mode = parts
    if feature_mode not in PROTOTYPE_FEATURES:
        raise argparse.ArgumentTypeError(f"unknown prototype feature: {feature_mode}")
    if scope not in PROTOTYPE_SCOPES:
        raise argparse.ArgumentTypeError(f"unknown prototype scope: {scope}")
    if prototype_mode not in PROTOTYPE_MODES:
        raise argparse.ArgumentTypeError(f"unknown prototype mode: {prototype_mode}")
    return feature_mode, scope, prototype_mode


def selector_value(result: dict[str, Any]) -> float:
    overall = result["overall"]
    if result["selector"] == "random_expected":
        return float(overall["tie_expected_success"])
    return float(overall["selector_success"])


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "selector": result["selector"],
        "success": selector_value(result),
        "cases": int(result["overall"]["cases"]),
        "success_rate": selector_value(result) / float(result["overall"]["cases"]),
        "by_task_success": {task: selector_value({"selector": result["selector"], "overall": summary}) for task, summary in result["by_task"].items()},
    }


def evaluate_selectors(
    rows: list[dict[str, Any]],
    *,
    heuristics: tuple[str, ...],
    prototypes: tuple[tuple[str, str, str], ...],
) -> list[dict[str, Any]]:
    results = [
        evaluate_rank0(rows),
        evaluate_random_expected(rows),
        evaluate_candidate_id(rows, "full_gripper_aware"),
    ]
    results.extend(evaluate_heuristic(rows, heuristic) for heuristic in heuristics)
    for feature_mode, scope, prototype_mode in prototypes:
        results.append(
            evaluate_prototype(
                rows,
                feature_mode=feature_mode,
                scope=scope,
                prototype_mode=prototype_mode,
            )
        )
    return results


def aggregate_seed_results(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_selector: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seed_result in seed_results:
        for result in seed_result["selectors"]:
            by_selector[result["selector"]].append(result)

    aggregate = []
    for selector, results in sorted(by_selector.items()):
        successes = np.asarray([float(result["success"]) for result in results], dtype=np.float32)
        rates = np.asarray([float(result["success_rate"]) for result in results], dtype=np.float32)
        by_task_values: dict[str, list[float]] = defaultdict(list)
        for result in results:
            for task, value in result["by_task_success"].items():
                by_task_values[task].append(float(value))
        aggregate.append(
            {
                "selector": selector,
                "seeds": len(results),
                "mean_success": float(successes.mean()),
                "std_success": float(successes.std(ddof=0)),
                "min_success": float(successes.min()),
                "max_success": float(successes.max()),
                "mean_success_rate": float(rates.mean()),
                "by_task_mean_success": {
                    task: float(np.asarray(values, dtype=np.float32).mean())
                    for task, values in sorted(by_task_values.items())
                },
            }
        )
    return {"selectors": aggregate}


def run_sweep(
    rows: list[dict[str, Any]],
    *,
    seeds: list[int],
    mode: str,
    remap_candidate_ids: bool,
    heuristics: tuple[str, ...] = DEFAULT_HEURISTICS,
    prototypes: tuple[tuple[str, str, str], ...] = DEFAULT_PROTOTYPES,
) -> dict[str, Any]:
    seed_results = []
    for seed in seeds:
        randomized = randomize_manifest_rows(
            rows,
            seed=seed,
            mode=mode,
            remap_candidate_ids=remap_candidate_ids,
        )
        seed_results.append(
            {
                "seed": seed,
                "selectors": [
                    compact_result(result)
                    for result in evaluate_selectors(randomized, heuristics=heuristics, prototypes=prototypes)
                ],
            }
        )
    aggregate = aggregate_seed_results(seed_results)
    return {
        "mode": mode,
        "remap_candidate_ids": remap_candidate_ids,
        "seeds": seeds,
        "num_seeds": len(seeds),
        "heuristics": list(heuristics),
        "prototypes": [":".join(config) for config in prototypes],
        "seed_results": seed_results,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["random", "prefer_success", "prefer_failure", "failure_rank0_shuffle_rest"],
        default="failure_rank0_shuffle_rest",
    )
    parser.add_argument("--remap-candidate-ids", action="store_true")
    parser.add_argument("--heuristic", action="append", choices=list(HEURISTICS))
    parser.add_argument(
        "--prototype-config",
        action="append",
        type=parse_prototype_config,
        help="Prototype selector as feature:scope:mode. Repeat to override the default prototype list.",
    )
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    heuristics = tuple(args.heuristic) if args.heuristic else DEFAULT_HEURISTICS
    prototypes = tuple(args.prototype_config) if args.prototype_config else DEFAULT_PROTOTYPES
    summary = run_sweep(
        load_jsonl(args.manifest),
        seeds=seeds,
        mode=args.mode,
        remap_candidate_ids=args.remap_candidate_ids,
        heuristics=heuristics,
        prototypes=prototypes,
    )
    summary["manifest"] = str(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
