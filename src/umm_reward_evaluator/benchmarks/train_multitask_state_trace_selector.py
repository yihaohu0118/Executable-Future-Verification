"""Train held-out selectors from RoboCasa low-dimensional rollout state traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from umm_reward_evaluator.benchmarks.common import load_jsonl, oracle_key
from umm_reward_evaluator.benchmarks.train_action_sequence_selector import SelectorMLP, action_features
from umm_reward_evaluator.benchmarks.train_multitask_action_sequence_selector import (
    TaskHeadMLP,
    load_labeled_rows,
    normalize_train_test,
)

STATE_FEATURE_MODES = ("state", "state_action", "zero")
STATE_SUMMARY_MODES = ("full", "endpoint", "terminal", "delta", "distribution", "path", "no_endpoint")


@dataclass(frozen=True)
class StateFeatureSpec:
    keys: tuple[str, ...]
    dims: dict[str, int]


def state_trace(row: dict[str, Any]) -> list[dict[str, list[float]]]:
    trace = row.get("metadata", {}).get("state_trace", [])
    return trace if isinstance(trace, list) else []


def build_state_spec(
    rows: list[dict[str, Any]],
    *,
    include_keys: set[str] | None = None,
    exclude_keys: set[str] | None = None,
) -> StateFeatureSpec:
    dims: dict[str, int] = {}
    for row in rows:
        for snapshot in state_trace(row):
            if not isinstance(snapshot, dict):
                continue
            for key, values in snapshot.items():
                if include_keys is not None and key not in include_keys:
                    continue
                if exclude_keys is not None and key in exclude_keys:
                    continue
                arr = np.asarray(values, dtype=np.float32).reshape(-1)
                dims[key] = max(dims.get(key, 0), int(arr.size))
    return StateFeatureSpec(keys=tuple(sorted(dims)), dims=dims)


def padded_snapshot(snapshot: dict[str, list[float]], key: str, dim: int) -> np.ndarray:
    arr = np.asarray(snapshot.get(key, []), dtype=np.float32).reshape(-1)
    out = np.zeros((dim,), dtype=np.float32)
    keep = min(dim, int(arr.size))
    if keep:
        out[:keep] = np.nan_to_num(arr[:keep], nan=0.0, posinf=0.0, neginf=0.0)
    return out


def per_key_trace_features(
    trace: list[dict[str, list[float]]],
    key: str,
    dim: int,
    *,
    summary_mode: str,
) -> np.ndarray:
    if not trace:
        x = np.zeros((1, dim), dtype=np.float32)
    else:
        x = np.stack([padded_snapshot(snapshot, key, dim) for snapshot in trace]).astype(np.float32)
    first = x[0]
    last = x[-1]
    delta = last - first
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    amin = x.min(axis=0)
    amax = x.max(axis=0)
    path = np.abs(np.diff(x, axis=0)).mean(axis=0) if x.shape[0] > 1 else np.zeros((dim,), dtype=np.float32)
    if summary_mode == "full":
        parts = [first, last, delta, mean, std, amin, amax, path]
    elif summary_mode == "endpoint":
        parts = [first, last, delta]
    elif summary_mode == "terminal":
        parts = [last]
    elif summary_mode == "delta":
        parts = [delta]
    elif summary_mode == "distribution":
        parts = [mean, std, amin, amax]
    elif summary_mode == "path":
        parts = [path]
    elif summary_mode == "no_endpoint":
        parts = [mean, std, amin, amax, path]
    else:
        raise ValueError(f"Unknown state summary mode {summary_mode}")
    return np.concatenate(parts).astype(np.float32)


def state_features(row: dict[str, Any], spec: StateFeatureSpec, *, summary_mode: str) -> np.ndarray:
    trace = state_trace(row)
    if not spec.keys:
        return np.zeros((1,), dtype=np.float32)
    parts = [per_key_trace_features(trace, key, spec.dims[key], summary_mode=summary_mode) for key in spec.keys]
    return np.concatenate(parts).astype(np.float32)


def feature_vector(
    row: dict[str, Any],
    *,
    mode: str,
    spec: StateFeatureSpec,
    action_mode: str,
    summary_mode: str,
) -> np.ndarray:
    if mode == "zero":
        return np.zeros((1,), dtype=np.float32)
    if mode == "state":
        return state_features(row, spec, summary_mode=summary_mode)
    if mode == "state_action":
        return np.concatenate(
            [state_features(row, spec, summary_mode=summary_mode), action_features(row, mode=action_mode)]
        ).astype(np.float32)
    raise ValueError(f"Unknown state feature mode {mode}")


def feature_matrix(
    rows: list[dict[str, Any]],
    *,
    feature_mode: str,
    state_spec: StateFeatureSpec,
    action_mode: str,
    summary_mode: str,
    task_mode: str,
    tasks: list[str],
) -> np.ndarray:
    xs = [
        feature_vector(row, mode=feature_mode, spec=state_spec, action_mode=action_mode, summary_mode=summary_mode)
        for row in rows
    ]
    if task_mode == "shared_onehot":
        task_to_idx = {task: idx for idx, task in enumerate(tasks)}
        with_task = []
        for row, x in zip(rows, xs):
            onehot = np.zeros((len(tasks),), dtype=np.float32)
            onehot[task_to_idx[str(row["task_label"])]] = 1.0
            with_task.append(np.concatenate([x, onehot]).astype(np.float32))
        xs = with_task
    return np.stack(xs).astype(np.float32)


def attach_scores(rows: list[dict[str, Any]], scores: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for row, score in zip(rows, scores):
        payload = dict(row)
        payload["state_trace_selector_score"] = float(score)
        out.append(payload)
    return out


def train_shared_fold(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    feature_mode: str,
    action_mode: str,
    summary_mode: str,
    task_mode: str,
    tasks: list[str],
    state_spec: StateFeatureSpec,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
) -> list[dict[str, Any]]:
    if not train_rows:
        raise ValueError("state-trace selector needs at least two held-out cases; got an empty training fold")
    torch.manual_seed(seed)
    x_train = feature_matrix(
        train_rows,
        feature_mode=feature_mode,
        state_spec=state_spec,
        action_mode=action_mode,
        summary_mode=summary_mode,
        task_mode=task_mode,
        tasks=tasks,
    )
    x_test = feature_matrix(
        test_rows,
        feature_mode=feature_mode,
        state_spec=state_spec,
        action_mode=action_mode,
        summary_mode=summary_mode,
        task_mode=task_mode,
        tasks=tasks,
    )
    y_train = np.asarray([1.0 if row["oracle_success"] else 0.0 for row in train_rows], dtype=np.float32)
    x_train, x_test = normalize_train_test(x_train, x_test)

    model = SelectorMLP(x_train.shape[1], hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    x = torch.from_numpy(x_train)
    y = torch.from_numpy(y_train)
    pos = float(y.sum().item())
    neg = float(y.numel() - pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32))
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        scores = torch.sigmoid(model(torch.from_numpy(x_test))).cpu().numpy()
    return attach_scores(test_rows, scores)


def train_task_head_fold(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    feature_mode: str,
    action_mode: str,
    summary_mode: str,
    tasks: list[str],
    state_spec: StateFeatureSpec,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
) -> list[dict[str, Any]]:
    if not train_rows:
        raise ValueError("state-trace selector needs at least two held-out cases; got an empty training fold")
    torch.manual_seed(seed)
    task_to_idx = {task: idx for idx, task in enumerate(tasks)}
    x_train = feature_matrix(
        train_rows,
        feature_mode=feature_mode,
        state_spec=state_spec,
        action_mode=action_mode,
        summary_mode=summary_mode,
        task_mode="per_task_head",
        tasks=tasks,
    )
    x_test = feature_matrix(
        test_rows,
        feature_mode=feature_mode,
        state_spec=state_spec,
        action_mode=action_mode,
        summary_mode=summary_mode,
        task_mode="per_task_head",
        tasks=tasks,
    )
    y_train = np.asarray([1.0 if row["oracle_success"] else 0.0 for row in train_rows], dtype=np.float32)
    task_train = torch.tensor([task_to_idx[str(row["task_label"])] for row in train_rows], dtype=torch.long)
    task_test = torch.tensor([task_to_idx[str(row["task_label"])] for row in test_rows], dtype=torch.long)
    x_train, x_test = normalize_train_test(x_train, x_test)

    model = TaskHeadMLP(x_train.shape[1], hidden, tasks)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    x = torch.from_numpy(x_train)
    y = torch.from_numpy(y_train)
    pos = float(y.sum().item())
    neg = float(y.numel() - pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32))
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x, task_train, tasks), y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        scores = torch.sigmoid(model(torch.from_numpy(x_test), task_test, tasks)).cpu().numpy()
    return attach_scores(test_rows, scores)


def evaluate_grouped(scored: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in scored:
        grouped.setdefault((str(row["task_label"]), str(row["case_id"])), []).append(row)

    by_task: dict[str, dict[str, Any]] = {}
    choice_counter: dict[str, Counter[str]] = {}
    selections: list[dict[str, Any]] = []
    for (task, case_id), case_rows in grouped.items():
        rank0 = min(case_rows, key=lambda row: row["candidate_rank_by_planner"])
        selector = max(
            case_rows,
            key=lambda row: (
                row["state_trace_selector_score"],
                -int(row.get("candidate_rank_by_planner", 999)),
            ),
        )
        oracle = max(case_rows, key=oracle_key)
        task_summary = by_task.setdefault(
            task,
            {
                "cases": 0,
                "rank0_success": 0,
                "selector_success": 0,
                "oracle_success": 0,
                "selector_oracle_match": 0,
                "rank0_failures": 0,
                "recovered_rank0_fail": 0,
            },
        )
        task_summary["cases"] += 1
        task_summary["rank0_success"] += int(bool(rank0["oracle_success"]))
        task_summary["selector_success"] += int(bool(selector["oracle_success"]))
        task_summary["oracle_success"] += int(bool(oracle["oracle_success"]))
        task_summary["selector_oracle_match"] += int(selector["candidate_id"] == oracle["candidate_id"])
        if not rank0["oracle_success"]:
            task_summary["rank0_failures"] += 1
            task_summary["recovered_rank0_fail"] += int(bool(selector["oracle_success"]))
        choice_counter.setdefault(task, Counter())[str(selector["candidate_id"])] += 1
        selections.append(
            {
                "task_label": task,
                "case_id": case_id,
                "rank0_candidate_id": rank0["candidate_id"],
                "selector_candidate_id": selector["candidate_id"],
                "oracle_candidate_id": oracle["candidate_id"],
                "rank0_success": bool(rank0["oracle_success"]),
                "selector_success": bool(selector["oracle_success"]),
                "oracle_success": bool(oracle["oracle_success"]),
                "selector_score": selector["state_trace_selector_score"],
            }
        )

    for task, task_summary in by_task.items():
        cases = int(task_summary["cases"])
        task_summary["rank0_success_rate"] = task_summary["rank0_success"] / cases if cases else 0.0
        task_summary["selector_success_rate"] = task_summary["selector_success"] / cases if cases else 0.0
        task_summary["oracle_success_rate"] = task_summary["oracle_success"] / cases if cases else 0.0
        task_summary["selector_choice_counts"] = dict(choice_counter.get(task, Counter()))

    overall: dict[str, Any] = {
        "cases": sum(item["cases"] for item in by_task.values()),
        "rank0_success": sum(item["rank0_success"] for item in by_task.values()),
        "selector_success": sum(item["selector_success"] for item in by_task.values()),
        "oracle_success": sum(item["oracle_success"] for item in by_task.values()),
        "selector_oracle_match": sum(item["selector_oracle_match"] for item in by_task.values()),
        "rank0_failures": sum(item["rank0_failures"] for item in by_task.values()),
        "recovered_rank0_fail": sum(item["recovered_rank0_fail"] for item in by_task.values()),
    }
    cases = int(overall["cases"])
    overall["rank0_success_rate"] = overall["rank0_success"] / cases if cases else 0.0
    overall["selector_success_rate"] = overall["selector_success"] / cases if cases else 0.0
    overall["oracle_success_rate"] = overall["oracle_success"] / cases if cases else 0.0
    return {"overall": overall, "by_task": by_task}, selections


def evaluate(
    rows: list[dict[str, Any]],
    *,
    feature_mode: str,
    action_mode: str,
    summary_mode: str,
    task_mode: str,
    include_state_keys: set[str] | None,
    exclude_state_keys: set[str] | None,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    state_spec = build_state_spec(rows, include_keys=include_state_keys, exclude_keys=exclude_state_keys)
    tasks = sorted({str(row["task_label"]) for row in rows})
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["task_label"]), str(row["case_id"])), []).append(row)

    scored: list[dict[str, Any]] = []
    for fold, (task, case_id) in enumerate(sorted(grouped)):
        test_rows = grouped[(task, case_id)]
        if task_mode == "independent":
            train_rows = [
                row
                for (other_task, other_case_id), case_rows in grouped.items()
                if other_task == task and other_case_id != case_id
                for row in case_rows
            ]
        else:
            train_rows = [row for other_key, case_rows in grouped.items() if other_key != (task, case_id) for row in case_rows]
        if task_mode == "per_task_head":
            scored.extend(
                train_task_head_fold(
                    train_rows,
                    test_rows,
                    feature_mode=feature_mode,
                    action_mode=action_mode,
                    summary_mode=summary_mode,
                    tasks=tasks,
                    state_spec=state_spec,
                    hidden=hidden,
                    epochs=epochs,
                    lr=lr,
                    seed=seed + fold,
                )
            )
        else:
            scored.extend(
                train_shared_fold(
                    train_rows,
                    test_rows,
                    feature_mode=feature_mode,
                    action_mode=action_mode,
                    summary_mode=summary_mode,
                    task_mode="shared_onehot" if task_mode == "shared_onehot" else "independent",
                    tasks=tasks,
                    state_spec=state_spec,
                    hidden=hidden,
                    epochs=epochs,
                    lr=lr,
                    seed=seed + fold,
                )
            )

    summary, selections = evaluate_grouped(scored)
    summary["feature_mode"] = feature_mode
    summary["action_mode"] = action_mode
    summary["state_summary_mode"] = summary_mode
    summary["task_mode"] = task_mode
    summary["state_keys"] = list(state_spec.keys)
    summary["state_dims"] = state_spec.dims
    summary["include_state_keys"] = sorted(include_state_keys) if include_state_keys is not None else None
    summary["exclude_state_keys"] = sorted(exclude_state_keys) if exclude_state_keys is not None else None
    summary["tasks"] = tasks
    return summary, selections, scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True, help="Path or task_label=path. Can repeat.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-mode", default="state", choices=list(STATE_FEATURE_MODES))
    parser.add_argument("--action-mode", default="stats_no_endpoints_no_length")
    parser.add_argument("--state-summary-mode", default="full", choices=list(STATE_SUMMARY_MODES))
    parser.add_argument("--task-mode", default="per_task_head", choices=["shared_onehot", "per_task_head", "independent"])
    parser.add_argument("--state-key", action="append", help="Restrict state features to this key. Can repeat.")
    parser.add_argument("--exclude-state-key", action="append", help="Drop this state key. Can repeat.")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = load_labeled_rows(args.manifest)
    summary, selections, scored = evaluate(
        rows,
        feature_mode=args.feature_mode,
        action_mode=args.action_mode,
        summary_mode=args.state_summary_mode,
        task_mode=args.task_mode,
        include_state_keys=set(args.state_key) if args.state_key else None,
        exclude_state_keys=set(args.exclude_state_key) if args.exclude_state_key else None,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "selections.json").write_text(
        json.dumps(selections, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "scored_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
