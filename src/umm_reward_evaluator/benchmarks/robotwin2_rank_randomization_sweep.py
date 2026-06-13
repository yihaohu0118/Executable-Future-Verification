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
    TRACE_DISTANCE_FEATURES,
    TRACE_DISTANCE_SCOPES,
    evaluate_candidate_id,
    evaluate_heuristic,
    evaluate_prototype,
    evaluate_random_expected,
    evaluate_rank0,
    evaluate_linear_probe,
    evaluate_trace_distance,
    LINEAR_PROBE_FEATURES,
    LINEAR_PROBE_SCOPES,
)


DEFAULT_HEURISTICS = ("smoothness_max", "energy_sum_max", "length_max")
DEFAULT_PROTOTYPES = (
    ("action_distribution", "same_task", "nearest_positive"),
    ("object_relation_distribution", "same_task", "nearest_positive"),
    ("gripper_distribution", "same_task", "nearest_positive"),
    ("gripper_distribution", "all_tasks", "nearest_positive"),
    ("gripper_distribution", "same_task", "nearest_pos_neg"),
    ("contact_envelope", "same_task", "nearest_positive"),
    ("contact_envelope", "same_task", "nearest_pos_neg"),
    ("phase_gripper_distribution", "same_task", "nearest_positive"),
    ("phase_gripper_distribution", "all_tasks", "nearest_positive"),
    ("phase_gripper_distribution", "same_task", "nearest_pos_neg"),
    ("phase_object_relation_distribution", "same_task", "nearest_positive"),
    ("phase_joint_distribution", "all_tasks", "nearest_positive"),
    ("phase_joint_gripper_distribution", "all_tasks", "nearest_positive"),
    ("phase_joint_gripper_distribution", "same_task", "nearest_pos_neg"),
    ("phase_object_relation_joint_gripper_distribution", "same_task", "nearest_pos_neg"),
)
DEFAULT_TRACE_DISTANCES = (
    ("dtw_action", "same_task"),
    ("dtw_object_relation", "same_task"),
    ("dtw_gripper", "same_task"),
    ("dtw_contact_envelope", "same_task"),
    ("dtw_joint", "all_tasks"),
    ("dtw_joint_gripper", "all_tasks"),
    ("dtw_object_relation_joint_gripper", "same_task"),
)
DEFAULT_LINEAR_PROBES = (
    ("action_distribution", "same_task"),
    ("gripper_distribution", "same_task"),
    ("contact_envelope", "same_task"),
    ("phase_gripper_distribution", "same_task"),
    ("phase_joint_gripper_distribution", "all_tasks"),
    ("phase_object_relation_joint_gripper_distribution", "same_task"),
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


def parse_trace_distance_config(value: str) -> tuple[str, str]:
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("trace-distance config must be feature:scope")
    feature_mode, scope = parts
    if feature_mode not in TRACE_DISTANCE_FEATURES:
        raise argparse.ArgumentTypeError(f"unknown trace-distance feature: {feature_mode}")
    if scope not in TRACE_DISTANCE_SCOPES:
        raise argparse.ArgumentTypeError(f"unknown trace-distance scope: {scope}")
    return feature_mode, scope


def parse_linear_probe_config(value: str) -> tuple[str, str]:
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("linear-probe config must be feature:scope")
    feature_mode, scope = parts
    if feature_mode not in LINEAR_PROBE_FEATURES:
        raise argparse.ArgumentTypeError(f"unknown linear-probe feature: {feature_mode}")
    if scope not in LINEAR_PROBE_SCOPES:
        raise argparse.ArgumentTypeError(f"unknown linear-probe scope: {scope}")
    return feature_mode, scope


def selector_value(result: dict[str, Any]) -> float:
    overall = result["overall"]
    if result["selector"] == "random_expected":
        return float(overall["tie_expected_success"])
    return float(overall["selector_success"])


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "selector": result["selector"],
        "success": selector_value(result),
        "cases": int(result["overall"]["cases"]),
        "success_rate": selector_value(result) / float(result["overall"]["cases"]),
        "by_task_success": {task: selector_value({"selector": result["selector"], "overall": summary}) for task, summary in result["by_task"].items()},
    }
    if "feature_coverage" in result:
        payload["feature_coverage"] = result["feature_coverage"]
    if "calibration_support" in result:
        payload["calibration_support"] = result["calibration_support"]
    return payload


def evaluate_selectors(
    rows: list[dict[str, Any]],
    *,
    heuristics: tuple[str, ...],
    prototypes: tuple[tuple[str, str, str], ...],
    trace_distances: tuple[tuple[str, str], ...],
    linear_probes: tuple[tuple[str, str], ...],
    linear_probe_l2: float,
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
    for feature_mode, scope in trace_distances:
        results.append(evaluate_trace_distance(rows, feature_mode=feature_mode, scope=scope))
    for feature_mode, scope in linear_probes:
        results.append(evaluate_linear_probe(rows, feature_mode=feature_mode, scope=scope, l2=linear_probe_l2))
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
        coverage_values = []
        support_rates = []
        min_train_rows = []
        min_positive_train_rows = []
        min_negative_train_rows = []
        unsupported_cases = []
        for result in results:
            for task, value in result["by_task_success"].items():
                by_task_values[task].append(float(value))
            if "feature_coverage" in result:
                coverage_values.append(float(result["feature_coverage"].get("case_coverage_rate", 0.0)))
            if "calibration_support" in result:
                support = result["calibration_support"]
                support_rates.append(float(support.get("support_rate", 0.0)))
                min_train_rows.append(int(support.get("min_train_rows", 0)))
                min_positive_train_rows.append(int(support.get("min_positive_train_rows", 0)))
                min_negative_train_rows.append(int(support.get("min_negative_train_rows", 0)))
                unsupported_cases.append(int(support.get("unsupported_cases", 0)))
        item = {
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
        if coverage_values:
            coverage = np.asarray(coverage_values, dtype=np.float32)
            item["min_feature_case_coverage"] = float(coverage.min())
            item["mean_feature_case_coverage"] = float(coverage.mean())
        if support_rates:
            support = np.asarray(support_rates, dtype=np.float32)
            item["min_calibration_support_rate"] = float(support.min())
            item["mean_calibration_support_rate"] = float(support.mean())
            item["min_train_rows"] = int(min(min_train_rows))
            item["min_positive_train_rows"] = int(min(min_positive_train_rows))
            item["min_negative_train_rows"] = int(min(min_negative_train_rows))
            item["max_unsupported_cases"] = int(max(unsupported_cases))
        aggregate.append(
            item
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
    trace_distances: tuple[tuple[str, str], ...] = DEFAULT_TRACE_DISTANCES,
    linear_probes: tuple[tuple[str, str], ...] = DEFAULT_LINEAR_PROBES,
    linear_probe_l2: float = 1.0,
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
                    for result in evaluate_selectors(
                        randomized,
                        heuristics=heuristics,
                        prototypes=prototypes,
                        trace_distances=trace_distances,
                        linear_probes=linear_probes,
                        linear_probe_l2=linear_probe_l2,
                    )
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
        "trace_distances": [":".join(config) for config in trace_distances],
        "linear_probes": [":".join(config) for config in linear_probes],
        "linear_probe_l2": linear_probe_l2,
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
    parser.add_argument(
        "--trace-distance-config",
        action="append",
        type=parse_trace_distance_config,
        help="Trace-distance selector as feature:scope. Repeat to override the default trace-distance list.",
    )
    parser.add_argument(
        "--linear-probe-config",
        action="append",
        type=parse_linear_probe_config,
        help="Linear learned verifier as feature:scope. Repeat to override the default linear-probe list.",
    )
    parser.add_argument("--linear-probe-l2", type=float, default=1.0)
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    heuristics = tuple(args.heuristic) if args.heuristic else DEFAULT_HEURISTICS
    prototypes = tuple(args.prototype_config) if args.prototype_config else DEFAULT_PROTOTYPES
    trace_distances = tuple(args.trace_distance_config) if args.trace_distance_config else DEFAULT_TRACE_DISTANCES
    linear_probes = tuple(args.linear_probe_config) if args.linear_probe_config else DEFAULT_LINEAR_PROBES
    summary = run_sweep(
        load_jsonl(args.manifest),
        seeds=seeds,
        mode=args.mode,
        remap_candidate_ids=args.remap_candidate_ids,
        heuristics=heuristics,
        prototypes=prototypes,
        trace_distances=trace_distances,
        linear_probes=linear_probes,
        linear_probe_l2=args.linear_probe_l2,
    )
    summary["manifest"] = str(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
