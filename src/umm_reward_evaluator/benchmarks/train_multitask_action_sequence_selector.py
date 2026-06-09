"""Train held-out action-sequence selectors across multiple benchmark manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from umm_reward_evaluator.benchmarks.common import load_jsonl, oracle_key
from umm_reward_evaluator.benchmarks.train_action_sequence_selector import (
    ACTION_FEATURE_MODES,
    SelectorMLP,
    action_features,
)


class TaskHeadMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, tasks: list[str]):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict({task: nn.Linear(hidden, 1) for task in tasks})

    def forward(self, x: torch.Tensor, task_idx: torch.Tensor, tasks: list[str]) -> torch.Tensor:
        z = self.trunk(x)
        out = torch.empty((x.shape[0],), dtype=z.dtype, device=z.device)
        for idx, task in enumerate(tasks):
            mask = task_idx == idx
            if bool(mask.any()):
                out[mask] = self.heads[task](z[mask]).squeeze(-1)
        return out


def parse_manifest_arg(value: str) -> tuple[str | None, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label, Path(path)
    return None, Path(value)


def load_labeled_rows(manifest_args: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_arg in manifest_args:
        explicit_label, path = parse_manifest_arg(manifest_arg)
        for row in load_jsonl(path):
            payload = dict(row)
            label = explicit_label or str(payload.get("task_name") or payload.get("suite") or path.stem)
            payload["task_label"] = label
            payload["source_manifest"] = str(path)
            rows.append(payload)
    return rows


def feature_matrix(
    rows: list[dict[str, Any]],
    *,
    feature_mode: str,
    task_mode: str,
    tasks: list[str],
) -> np.ndarray:
    xs = [action_features(row, mode=feature_mode) for row in rows]
    if task_mode == "shared_onehot":
        task_to_idx = {task: idx for idx, task in enumerate(tasks)}
        onehots = []
        for row in rows:
            onehot = np.zeros((len(tasks),), dtype=np.float32)
            onehot[task_to_idx[str(row["task_label"])]] = 1.0
            onehots.append(onehot)
        xs = [np.concatenate([x, onehot]).astype(np.float32) for x, onehot in zip(xs, onehots)]
    return np.stack(xs).astype(np.float32)


def normalize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    return (x_train - mean) / std, (x_test - mean) / std


def train_shared_fold(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    feature_mode: str,
    task_mode: str,
    tasks: list[str],
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
) -> list[dict[str, Any]]:
    torch.manual_seed(seed)
    x_train = feature_matrix(train_rows, feature_mode=feature_mode, task_mode=task_mode, tasks=tasks)
    x_test = feature_matrix(test_rows, feature_mode=feature_mode, task_mode=task_mode, tasks=tasks)
    y_train = np.asarray([1.0 if row["oracle_success"] else 0.0 for row in train_rows], dtype=np.float32)
    x_train, x_test = normalize_train_test(x_train, x_test)

    model = SelectorMLP(x_train.shape[1], hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    y = torch.from_numpy(y_train)
    pos = float(y.sum().item())
    neg = float(y.numel() - pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32))
    x = torch.from_numpy(x_train)
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
    tasks: list[str],
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
) -> list[dict[str, Any]]:
    torch.manual_seed(seed)
    task_to_idx = {task: idx for idx, task in enumerate(tasks)}
    x_train = feature_matrix(train_rows, feature_mode=feature_mode, task_mode="per_task_head", tasks=tasks)
    x_test = feature_matrix(test_rows, feature_mode=feature_mode, task_mode="per_task_head", tasks=tasks)
    y_train = np.asarray([1.0 if row["oracle_success"] else 0.0 for row in train_rows], dtype=np.float32)
    task_train = torch.tensor([task_to_idx[str(row["task_label"])] for row in train_rows], dtype=torch.long)
    task_test = torch.tensor([task_to_idx[str(row["task_label"])] for row in test_rows], dtype=torch.long)
    x_train, x_test = normalize_train_test(x_train, x_test)

    model = TaskHeadMLP(x_train.shape[1], hidden, tasks)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    y = torch.from_numpy(y_train)
    pos = float(y.sum().item())
    neg = float(y.numel() - pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32))
    x = torch.from_numpy(x_train)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x, task_train, tasks), y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        scores = torch.sigmoid(model(torch.from_numpy(x_test), task_test, tasks)).cpu().numpy()
    return attach_scores(test_rows, scores)


def attach_scores(rows: list[dict[str, Any]], scores: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for row, score in zip(rows, scores):
        payload = dict(row)
        payload["multitask_action_sequence_selector_score"] = float(score)
        out.append(payload)
    return out


def evaluate_grouped(scored: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in scored:
        grouped.setdefault((str(row["task_label"]), str(row["case_id"])), []).append(row)

    summary_by_task: dict[str, dict[str, Any]] = {}
    selections: list[dict[str, Any]] = []
    choice_counter: dict[str, Counter[str]] = {}
    for (task, case_id), case_rows in grouped.items():
        rank0 = min(case_rows, key=lambda row: row["candidate_rank_by_planner"])
        selector = max(
            case_rows,
            key=lambda row: (
                row["multitask_action_sequence_selector_score"],
                -int(row.get("candidate_rank_by_planner", 999)),
            ),
        )
        oracle = max(case_rows, key=oracle_key)
        task_summary = summary_by_task.setdefault(
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
                "selector_score": selector["multitask_action_sequence_selector_score"],
            }
        )

    for task, task_summary in summary_by_task.items():
        cases = int(task_summary["cases"])
        task_summary["rank0_success_rate"] = task_summary["rank0_success"] / cases if cases else 0.0
        task_summary["selector_success_rate"] = task_summary["selector_success"] / cases if cases else 0.0
        task_summary["oracle_success_rate"] = task_summary["oracle_success"] / cases if cases else 0.0
        task_summary["selector_choice_counts"] = dict(choice_counter.get(task, Counter()))

    overall: dict[str, Any] = {
        "cases": sum(item["cases"] for item in summary_by_task.values()),
        "rank0_success": sum(item["rank0_success"] for item in summary_by_task.values()),
        "selector_success": sum(item["selector_success"] for item in summary_by_task.values()),
        "oracle_success": sum(item["oracle_success"] for item in summary_by_task.values()),
        "selector_oracle_match": sum(item["selector_oracle_match"] for item in summary_by_task.values()),
        "rank0_failures": sum(item["rank0_failures"] for item in summary_by_task.values()),
        "recovered_rank0_fail": sum(item["recovered_rank0_fail"] for item in summary_by_task.values()),
    }
    cases = int(overall["cases"])
    overall["rank0_success_rate"] = overall["rank0_success"] / cases if cases else 0.0
    overall["selector_success_rate"] = overall["selector_success"] / cases if cases else 0.0
    overall["oracle_success_rate"] = overall["oracle_success"] / cases if cases else 0.0
    return {"overall": overall, "by_task": summary_by_task}, selections


def evaluate(
    rows: list[dict[str, Any]],
    *,
    feature_mode: str,
    task_mode: str,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = sorted({str(row["task_label"]) for row in rows})
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["task_label"]), str(row["case_id"])), []).append(row)

    scored: list[dict[str, Any]] = []
    folds = sorted(grouped)
    for fold, (task, case_id) in enumerate(folds):
        test_rows = grouped[(task, case_id)]
        if task_mode == "independent":
            train_rows = [
                row
                for (other_task, other_case_id), case_rows in grouped.items()
                if other_task == task and other_case_id != case_id
                for row in case_rows
            ]
        else:
            train_rows = [
                row
                for other_key, case_rows in grouped.items()
                if other_key != (task, case_id)
                for row in case_rows
            ]
        if task_mode == "per_task_head":
            scored.extend(
                train_task_head_fold(
                    train_rows,
                    test_rows,
                    feature_mode=feature_mode,
                    tasks=tasks,
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
                    task_mode="shared_onehot" if task_mode == "shared_onehot" else "independent",
                    tasks=tasks,
                    hidden=hidden,
                    epochs=epochs,
                    lr=lr,
                    seed=seed + fold,
                )
            )

    summary, selections = evaluate_grouped(scored)
    summary["feature_mode"] = feature_mode
    summary["task_mode"] = task_mode
    summary["tasks"] = tasks
    return summary, selections, scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True, help="Path or task_label=path. Can repeat.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-mode", default="raw_no_length", choices=list(ACTION_FEATURE_MODES))
    parser.add_argument("--task-mode", default="shared_onehot", choices=["shared_onehot", "per_task_head", "independent"])
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = load_labeled_rows(args.manifest)
    summary, selections, scored = evaluate(
        rows,
        feature_mode=args.feature_mode,
        task_mode=args.task_mode,
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
