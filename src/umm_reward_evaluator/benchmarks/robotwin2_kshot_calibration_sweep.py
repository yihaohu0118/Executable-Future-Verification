"""K-shot target-task calibration sweeps for RoboTwin2 prototype selectors."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from umm_reward_evaluator.benchmarks.common import load_jsonl
from umm_reward_evaluator.benchmarks.randomize_planner_rank import randomize_manifest_rows
from umm_reward_evaluator.benchmarks.robotwin2_rank_randomization_sweep import parse_prototype_config
from umm_reward_evaluator.benchmarks.robotwin2_selector_baselines import (
    PROTOTYPE_FEATURES,
    group_cases,
    feature_vector,
    normalize_train_test,
    prototype_scores,
    select_by_score,
    state_dims,
    summarize_selections,
)


DEFAULT_FEATURES = (
    "gripper_distribution",
    "phase_gripper_distribution",
    "phase_joint_distribution",
    "phase_joint_gripper_distribution",
)


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    metadata = row.get("metadata") or {}
    original_id = metadata.get("original_candidate_id", row["candidate_id"])
    return str(row["task_name"]), str(row["case_id"]), str(original_id)


def feature_matrix(rows: list[dict[str, Any]], feature_mode: str) -> dict[tuple[str, str, str], np.ndarray]:
    dims = state_dims(rows, ["joint_action_vector", "left_gripper", "right_gripper"])
    return {
        row_key(row): feature_vector(row, feature_mode, dims=dims).astype(np.float32)
        for row in rows
    }


def support_keys_for_case(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    heldout_key: tuple[str, str],
    *,
    k: int,
    support_seed: int,
) -> list[tuple[str, str]]:
    if k <= 0:
        return []
    task, _case_id = heldout_key
    candidates = sorted(key for key in grouped if key != heldout_key and key[0] == task)
    if not candidates:
        return []
    offset = support_seed % len(candidates)
    rotated = candidates[offset:] + candidates[:offset]
    return rotated[: min(k, len(rotated))]


def evaluate_kshot_feature(
    rows: list[dict[str, Any]],
    *,
    feature_mode: str,
    k: int,
    support_seed: int,
    include_source_tasks: bool,
    features: dict[tuple[str, str, str], np.ndarray] | None = None,
) -> dict[str, Any]:
    grouped = group_cases(rows)
    feature_by_row = features or feature_matrix(rows, feature_mode)
    selected = {}
    support_by_case = {}
    for heldout_key, test_rows in sorted(grouped.items()):
        task, _case_id = heldout_key
        train_rows: list[dict[str, Any]] = []
        if include_source_tasks:
            train_rows.extend(row for key, case_rows in grouped.items() if key[0] != task for row in case_rows)
        support_keys = support_keys_for_case(grouped, heldout_key, k=k, support_seed=support_seed)
        train_rows.extend(row for key in support_keys for row in grouped[key])
        support_by_case[heldout_key] = [f"{key[0]}:{key[1]}" for key in support_keys]

        if not train_rows:
            scores = np.zeros((len(test_rows),), dtype=np.float32)
        else:
            x_train = np.stack([feature_by_row[row_key(row)] for row in train_rows]).astype(np.float32)
            x_test = np.stack([feature_by_row[row_key(row)] for row in test_rows]).astype(np.float32)
            x_train, x_test = normalize_train_test(x_train, x_test)
            y_train = np.asarray([1.0 if row["oracle_success"] else 0.0 for row in train_rows], dtype=np.float32)
            scores = prototype_scores(x_train, y_train, x_test, "nearest_positive")
        selected[heldout_key] = select_by_score(test_rows, scores.tolist())

    mode = "source_plus_target" if include_source_tasks else "target_only"
    result = summarize_selections(
        rows,
        selector_name=f"kshot:{feature_mode}:{mode}:k{k}",
        selected_by_case=selected,
    )
    result["support_by_case"] = {f"{task}:{case_id}": support for (task, case_id), support in support_by_case.items()}
    return result


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    overall = result["overall"]
    return {
        "selector": result["selector"],
        "success": float(overall["selector_success"]),
        "cases": int(overall["cases"]),
        "success_rate": float(overall["selector_success"]) / float(overall["cases"]),
        "by_task_success": {
            task: float(summary["selector_success"])
            for task, summary in result["by_task"].items()
        },
    }


def aggregate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_selector: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_selector[result["selector"]].append(result)
    out = []
    for selector, rows in sorted(by_selector.items()):
        successes = np.asarray([row["success"] for row in rows], dtype=np.float32)
        task_values: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            for task, value in row["by_task_success"].items():
                task_values[task].append(float(value))
        out.append(
            {
                "selector": selector,
                "runs": len(rows),
                "mean_success": float(successes.mean()),
                "std_success": float(successes.std(ddof=0)),
                "min_success": float(successes.min()),
                "max_success": float(successes.max()),
                "mean_success_rate": float(successes.mean() / rows[0]["cases"]),
                "by_task_mean_success": {
                    task: float(np.asarray(values, dtype=np.float32).mean())
                    for task, values in sorted(task_values.items())
                },
            }
        )
    return out


def run_kshot_sweep(
    rows: list[dict[str, Any]],
    *,
    rank_seeds: list[int],
    support_seeds: list[int],
    k_values: list[int],
    feature_modes: tuple[str, ...],
    mode: str,
    remap_candidate_ids: bool,
    include_source_tasks: bool,
) -> dict[str, Any]:
    run_results = []
    for rank_seed in rank_seeds:
        randomized = randomize_manifest_rows(
            rows,
            seed=rank_seed,
            mode=mode,
            remap_candidate_ids=remap_candidate_ids,
        )
        feature_cache = {
            feature_mode: feature_matrix(randomized, feature_mode)
            for feature_mode in feature_modes
        }
        for support_seed in support_seeds:
            for k in k_values:
                for feature_mode in feature_modes:
                    result = compact_result(
                        evaluate_kshot_feature(
                            randomized,
                            feature_mode=feature_mode,
                            k=k,
                            support_seed=support_seed,
                            include_source_tasks=include_source_tasks,
                            features=feature_cache[feature_mode],
                        )
                    )
                    result["rank_seed"] = rank_seed
                    result["support_seed"] = support_seed
                    result["k"] = k
                    result["feature_mode"] = feature_mode
                    run_results.append(result)
    return {
        "rank_seeds": rank_seeds,
        "support_seeds": support_seeds,
        "k_values": k_values,
        "feature_modes": list(feature_modes),
        "mode": mode,
        "remap_candidate_ids": remap_candidate_ids,
        "include_source_tasks": include_source_tasks,
        "runs": run_results,
        "aggregate": {"selectors": aggregate(run_results)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-rank-seeds", type=int, default=10)
    parser.add_argument("--rank-seed-start", type=int, default=0)
    parser.add_argument("--num-support-seeds", type=int, default=5)
    parser.add_argument("--support-seed-start", type=int, default=0)
    parser.add_argument("--k", action="append", type=int)
    parser.add_argument("--feature", action="append", choices=list(PROTOTYPE_FEATURES))
    parser.add_argument(
        "--mode",
        choices=["random", "prefer_success", "prefer_failure", "failure_rank0_shuffle_rest"],
        default="failure_rank0_shuffle_rest",
    )
    parser.add_argument("--remap-candidate-ids", action="store_true")
    parser.add_argument("--target-only", action="store_true")
    args = parser.parse_args()

    # Reuse validation logic for feature names if users pass a full prototype config by mistake.
    if args.feature:
        for feature in args.feature:
            parse_prototype_config(f"{feature}:same_task:nearest_positive")

    rank_seeds = list(range(args.rank_seed_start, args.rank_seed_start + args.num_rank_seeds))
    support_seeds = list(range(args.support_seed_start, args.support_seed_start + args.num_support_seeds))
    summary = run_kshot_sweep(
        load_jsonl(args.manifest),
        rank_seeds=rank_seeds,
        support_seeds=support_seeds,
        k_values=args.k or [0, 1, 2, 4],
        feature_modes=tuple(args.feature) if args.feature else DEFAULT_FEATURES,
        mode=args.mode,
        remap_candidate_ids=args.remap_candidate_ids,
        include_source_tasks=not args.target_only,
    )
    summary["manifest"] = str(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
