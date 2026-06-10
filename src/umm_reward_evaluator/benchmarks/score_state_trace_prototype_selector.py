"""Evaluate non-neural prototype selectors over rollout state-trace features."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from umm_reward_evaluator.benchmarks.common import oracle_key
from umm_reward_evaluator.benchmarks.train_multitask_action_sequence_selector import load_labeled_rows
from umm_reward_evaluator.benchmarks.train_multitask_state_trace_selector import (
    STATE_SUMMARY_MODES,
    build_state_spec,
    feature_vector,
)

PROTOTYPE_MODES = ("pos_centroid", "pos_neg_centroid", "nearest_positive")
TRAIN_SCOPES = ("same_task", "all_tasks")


def normalize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    return (x_train - mean) / std, (x_test - mean) / std


def row_features(
    rows: list[dict[str, Any]],
    *,
    state_spec: Any,
    summary_mode: str,
) -> np.ndarray:
    return np.stack(
        [
            feature_vector(
                row,
                mode="state",
                spec=state_spec,
                action_mode="stats_no_endpoints_no_length",
                summary_mode=summary_mode,
            )
            for row in rows
        ]
    ).astype(np.float32)


def score_from_prototype(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    prototype_mode: str,
) -> np.ndarray:
    pos = x_train[y_train > 0.5]
    neg = x_train[y_train <= 0.5]
    if pos.size == 0:
        return np.zeros((x_test.shape[0],), dtype=np.float32)
    pos_centroid = pos.mean(axis=0, keepdims=True)
    pos_dist = np.linalg.norm(x_test - pos_centroid, axis=1)
    if prototype_mode == "pos_centroid":
        return -pos_dist
    if prototype_mode == "pos_neg_centroid":
        if neg.size == 0:
            return -pos_dist
        neg_centroid = neg.mean(axis=0, keepdims=True)
        neg_dist = np.linalg.norm(x_test - neg_centroid, axis=1)
        return neg_dist - pos_dist
    if prototype_mode == "nearest_positive":
        dists = np.linalg.norm(x_test[:, None, :] - pos[None, :, :], axis=2)
        return -dists.min(axis=1)
    raise ValueError(f"Unknown prototype mode {prototype_mode}")


def attach_scores(rows: list[dict[str, Any]], scores: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for row, score in zip(rows, scores):
        payload = dict(row)
        payload["state_trace_prototype_score"] = float(score)
        out.append(payload)
    return out


def evaluate_grouped(scored: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in scored:
        grouped.setdefault((str(row["task_label"]), str(row["case_id"])), []).append(row)

    by_task: dict[str, dict[str, Any]] = {}
    choice_counter: dict[str, Counter[str]] = {}
    selections: list[dict[str, Any]] = []
    for (task, case_id), case_rows in sorted(grouped.items()):
        rank0 = min(case_rows, key=lambda row: int(row.get("candidate_rank_by_planner", 999)))
        selected = max(
            case_rows,
            key=lambda row: (
                row["state_trace_prototype_score"],
                -int(row.get("candidate_rank_by_planner", 999)),
            ),
        )
        oracle = max(case_rows, key=oracle_key)
        task_summary = by_task.setdefault(
            task,
            {
                "cases": 0,
                "rank0_success": 0,
                "prototype_success": 0,
                "oracle_success": 0,
                "prototype_oracle_match": 0,
            },
        )
        task_summary["cases"] += 1
        task_summary["rank0_success"] += int(bool(rank0["oracle_success"]))
        task_summary["prototype_success"] += int(bool(selected["oracle_success"]))
        task_summary["oracle_success"] += int(bool(oracle["oracle_success"]))
        task_summary["prototype_oracle_match"] += int(selected["candidate_id"] == oracle["candidate_id"])
        choice_counter.setdefault(task, Counter())[str(selected["candidate_id"])] += 1
        selections.append(
            {
                "task_label": task,
                "case_id": case_id,
                "rank0_candidate_id": rank0["candidate_id"],
                "prototype_candidate_id": selected["candidate_id"],
                "oracle_candidate_id": oracle["candidate_id"],
                "rank0_success": bool(rank0["oracle_success"]),
                "prototype_success": bool(selected["oracle_success"]),
                "oracle_success": bool(oracle["oracle_success"]),
                "prototype_score": selected["state_trace_prototype_score"],
            }
        )

    for task, summary in by_task.items():
        cases = int(summary["cases"])
        summary["prototype_success_rate"] = summary["prototype_success"] / cases if cases else 0.0
        summary["oracle_success_rate"] = summary["oracle_success"] / cases if cases else 0.0
        summary["choice_counts"] = dict(choice_counter.get(task, Counter()))

    overall = {
        "cases": sum(item["cases"] for item in by_task.values()),
        "rank0_success": sum(item["rank0_success"] for item in by_task.values()),
        "prototype_success": sum(item["prototype_success"] for item in by_task.values()),
        "oracle_success": sum(item["oracle_success"] for item in by_task.values()),
        "prototype_oracle_match": sum(item["prototype_oracle_match"] for item in by_task.values()),
    }
    cases = int(overall["cases"])
    overall["prototype_success_rate"] = overall["prototype_success"] / cases if cases else 0.0
    overall["oracle_success_rate"] = overall["oracle_success"] / cases if cases else 0.0
    return {"overall": overall, "by_task": by_task}, selections


def evaluate(
    rows: list[dict[str, Any]],
    *,
    include_state_keys: set[str] | None,
    exclude_state_keys: set[str] | None,
    summary_mode: str,
    prototype_mode: str,
    train_scope: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    state_spec = build_state_spec(rows, include_keys=include_state_keys, exclude_keys=exclude_state_keys)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["task_label"]), str(row["case_id"])), []).append(row)

    scored: list[dict[str, Any]] = []
    for task, case_id in sorted(grouped):
        test_rows = grouped[(task, case_id)]
        if train_scope == "same_task":
            train_rows = [
                row
                for (other_task, other_case_id), case_rows in grouped.items()
                if other_task == task and other_case_id != case_id
                for row in case_rows
            ]
        elif train_scope == "all_tasks":
            train_rows = [row for other_key, case_rows in grouped.items() if other_key != (task, case_id) for row in case_rows]
        else:
            raise ValueError(f"Unknown train scope {train_scope}")
        x_train = row_features(train_rows, state_spec=state_spec, summary_mode=summary_mode)
        x_test = row_features(test_rows, state_spec=state_spec, summary_mode=summary_mode)
        x_train, x_test = normalize_train_test(x_train, x_test)
        y_train = np.asarray([1.0 if row["oracle_success"] else 0.0 for row in train_rows], dtype=np.float32)
        scores = score_from_prototype(x_train, y_train, x_test, prototype_mode=prototype_mode)
        scored.extend(attach_scores(test_rows, scores))

    summary, selections = evaluate_grouped(scored)
    summary["state_summary_mode"] = summary_mode
    summary["prototype_mode"] = prototype_mode
    summary["train_scope"] = train_scope
    summary["state_keys"] = list(state_spec.keys)
    summary["state_dims"] = state_spec.dims
    summary["include_state_keys"] = sorted(include_state_keys) if include_state_keys is not None else None
    summary["exclude_state_keys"] = sorted(exclude_state_keys) if exclude_state_keys is not None else None
    return summary, selections, scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True, help="Path or task_label=path. Can repeat.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-summary-mode", default="distribution", choices=list(STATE_SUMMARY_MODES))
    parser.add_argument("--prototype-mode", default="pos_neg_centroid", choices=list(PROTOTYPE_MODES))
    parser.add_argument("--train-scope", default="same_task", choices=list(TRAIN_SCOPES))
    parser.add_argument("--state-key", action="append", help="Restrict state features to this key. Can repeat.")
    parser.add_argument("--exclude-state-key", action="append", help="Drop this state key. Can repeat.")
    args = parser.parse_args()

    rows = load_labeled_rows(args.manifest)
    summary, selections, scored = evaluate(
        rows,
        include_state_keys=set(args.state_key) if args.state_key else None,
        exclude_state_keys=set(args.exclude_state_key) if args.exclude_state_key else None,
        summary_mode=args.state_summary_mode,
        prototype_mode=args.prototype_mode,
        train_scope=args.train_scope,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "selections.json").write_text(
        json.dumps(selections, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "scored_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
