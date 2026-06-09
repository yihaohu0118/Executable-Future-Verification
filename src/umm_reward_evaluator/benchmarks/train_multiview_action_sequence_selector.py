"""Train held-out multitask selectors for multiple action-feature views."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from umm_reward_evaluator.benchmarks.train_action_sequence_selector import ACTION_FEATURE_MODES
from umm_reward_evaluator.benchmarks.train_multitask_action_sequence_selector import (
    evaluate as evaluate_multitask,
)
from umm_reward_evaluator.benchmarks.train_multitask_action_sequence_selector import (
    evaluate_grouped,
    load_labeled_rows,
    train_shared_fold,
    train_task_head_fold,
)


SCORE_KEY = "multitask_action_sequence_selector_score"


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["task_label"]), str(row["case_id"]), str(row["candidate_id"])


def case_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["task_label"]), str(row["case_id"])


def per_case_rank_scores(scored_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], float]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        grouped[case_key(row)].append(row)

    ranks: dict[tuple[str, str, str], float] = {}
    for case_rows in grouped.values():
        ordered = sorted(
            case_rows,
            key=lambda row: (
                float(row[SCORE_KEY]),
                -int(row.get("candidate_rank_by_planner", 999)),
            ),
            reverse=True,
        )
        denom = max(len(ordered) - 1, 1)
        for rank, row in enumerate(ordered):
            ranks[row_key(row)] = 1.0 - (rank / denom)
    return ranks


def combine_view_scores(
    base_rows: list[dict[str, Any]],
    *,
    scored_views: list[list[dict[str, Any]]],
    combine: str,
    risk_alpha: float,
) -> list[dict[str, Any]]:
    score_maps = [{row_key(row): float(row[SCORE_KEY]) for row in scored_rows} for scored_rows in scored_views]
    rank_maps = [per_case_rank_scores(scored_rows) for scored_rows in scored_views]

    combined: list[dict[str, Any]] = []
    for row in base_rows:
        key = row_key(row)
        view_scores = np.asarray([score_map[key] for score_map in score_maps], dtype=np.float32)
        view_ranks = np.asarray([rank_map[key] for rank_map in rank_maps], dtype=np.float32)
        if combine == "score_mean":
            score = float(view_scores.mean())
        elif combine == "score_min":
            score = float(view_scores.min())
        elif combine == "score_product":
            score = float(np.prod(view_scores))
        elif combine == "rank_mean":
            score = float(view_ranks.mean())
        elif combine == "rank_min":
            score = float(view_ranks.min())
        elif combine == "risk_adjusted_rank":
            score = float(view_ranks.mean() - risk_alpha * view_ranks.std())
        else:
            raise ValueError(f"Unknown combine mode {combine}")
        payload = dict(row)
        payload[SCORE_KEY] = score
        payload["multiview_selector_view_scores"] = [float(item) for item in view_scores]
        payload["multiview_selector_view_ranks"] = [float(item) for item in view_ranks]
        combined.append(payload)
    return combined


def build_meta_features(
    base_rows: list[dict[str, Any]],
    *,
    scored_views: list[list[dict[str, Any]]],
) -> dict[tuple[str, str, str], np.ndarray]:
    score_maps = [{row_key(row): float(row[SCORE_KEY]) for row in scored_rows} for scored_rows in scored_views]
    rank_maps = [per_case_rank_scores(scored_rows) for scored_rows in scored_views]
    features: dict[tuple[str, str, str], np.ndarray] = {}
    for row in base_rows:
        key = row_key(row)
        view_scores = np.asarray([score_map[key] for score_map in score_maps], dtype=np.float32)
        view_ranks = np.asarray([rank_map[key] for rank_map in rank_maps], dtype=np.float32)
        planner_rank = float(row.get("candidate_rank_by_planner", 0))
        planner_feature = np.asarray([planner_rank / 10.0], dtype=np.float32)
        features[key] = np.concatenate([view_scores, view_ranks, planner_feature]).astype(np.float32)
    return features


def score_view_once(
    train_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    *,
    feature_mode: str,
    task_mode: str,
    tasks: list[str],
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
) -> list[dict[str, Any]]:
    if task_mode == "per_task_head":
        return train_task_head_fold(
            train_rows,
            score_rows,
            feature_mode=feature_mode,
            tasks=tasks,
            hidden=hidden,
            epochs=epochs,
            lr=lr,
            seed=seed,
        )
    if task_mode == "shared_onehot":
        return train_shared_fold(
            train_rows,
            score_rows,
            feature_mode=feature_mode,
            task_mode="shared_onehot",
            tasks=tasks,
            hidden=hidden,
            epochs=epochs,
            lr=lr,
            seed=seed,
        )

    scored: list[dict[str, Any]] = []
    for task in tasks:
        task_train = [row for row in train_rows if str(row["task_label"]) == task]
        task_score = [row for row in score_rows if str(row["task_label"]) == task]
        if not task_score:
            continue
        scored.extend(
            train_shared_fold(
                task_train,
                task_score,
                feature_mode=feature_mode,
                task_mode="independent",
                tasks=tasks,
                hidden=hidden,
                epochs=epochs,
                lr=lr,
                seed=seed,
            )
        )
    return scored


def meta_logistic_scores(
    base_rows: list[dict[str, Any]],
    *,
    feature_modes: list[str],
    task_mode: str,
    tasks: list[str],
    hidden: int,
    view_epochs: int,
    view_lr: float,
    meta_epochs: int,
    meta_lr: float,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in base_rows:
        grouped[case_key(row)].append(row)

    scored: list[dict[str, Any]] = []
    for fold, heldout_case in enumerate(sorted(grouped)):
        train_rows = [row for other_case, rows in grouped.items() if other_case != heldout_case for row in rows]
        test_rows = grouped[heldout_case]
        score_rows = train_rows + test_rows
        scored_views = [
            score_view_once(
                train_rows,
                score_rows,
                feature_mode=feature_mode,
                task_mode=task_mode,
                tasks=tasks,
                hidden=hidden,
                epochs=view_epochs,
                lr=view_lr,
                seed=seed + fold,
            )
            for feature_mode in feature_modes
        ]
        features = build_meta_features(score_rows, scored_views=scored_views)
        x_train = np.stack([features[row_key(row)] for row in train_rows])
        y_train = np.asarray([1.0 if row["oracle_success"] else 0.0 for row in train_rows], dtype=np.float32)
        x_test = np.stack([features[row_key(row)] for row in test_rows])
        mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True) + 1e-6
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std

        torch.manual_seed(seed + fold)
        model = nn.Linear(x_train.shape[1], 1)
        opt = torch.optim.AdamW(model.parameters(), lr=meta_lr, weight_decay=1e-3)
        y = torch.from_numpy(y_train)
        pos = float(y.sum().item())
        neg = float(y.numel() - pos)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32))
        x = torch.from_numpy(x_train)
        for _ in range(meta_epochs):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x).squeeze(-1), y)
            loss.backward()
            opt.step()

        with torch.no_grad():
            scores = torch.sigmoid(model(torch.from_numpy(x_test)).squeeze(-1)).cpu().numpy()
        for row, score in zip(test_rows, scores):
            payload = dict(row)
            payload[SCORE_KEY] = float(score)
            payload["multiview_selector_meta_features"] = [float(item) for item in features[row_key(row)]]
            scored.append(payload)
    return scored


def evaluate(
    rows: list[dict[str, Any]],
    *,
    feature_modes: list[str],
    combine: str,
    task_mode: str,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
    risk_alpha: float,
    meta_epochs: int,
    meta_lr: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    view_summaries = []
    scored_views = []
    for feature_mode in feature_modes:
        view_summary, _, scored = evaluate_multitask(
            rows,
            feature_mode=feature_mode,
            task_mode=task_mode,
            hidden=hidden,
            epochs=epochs,
            lr=lr,
            seed=seed,
        )
        view_summaries.append(view_summary)
        scored_views.append(scored)

    if combine == "meta_logistic":
        combined_scored = meta_logistic_scores(
            rows,
            feature_modes=feature_modes,
            task_mode=task_mode,
            tasks=sorted({str(row["task_label"]) for row in rows}),
            hidden=hidden,
            view_epochs=epochs,
            view_lr=lr,
            meta_epochs=meta_epochs,
            meta_lr=meta_lr,
            seed=seed,
        )
    else:
        combined_scored = combine_view_scores(
            rows,
            scored_views=scored_views,
            combine=combine,
            risk_alpha=risk_alpha,
        )
    summary, selections = evaluate_grouped(combined_scored)
    summary["feature_modes"] = feature_modes
    summary["combine"] = combine
    summary["task_mode"] = task_mode
    summary["risk_alpha"] = risk_alpha
    summary["meta_epochs"] = meta_epochs
    summary["meta_lr"] = meta_lr
    summary["view_summaries"] = view_summaries
    return summary, selections, combined_scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True, help="Path or task_label=path. Can repeat.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-mode", action="append", required=True, choices=list(ACTION_FEATURE_MODES))
    parser.add_argument(
        "--combine",
        default="rank_mean",
        choices=[
            "score_mean",
            "score_min",
            "score_product",
            "rank_mean",
            "rank_min",
            "risk_adjusted_rank",
            "meta_logistic",
        ],
    )
    parser.add_argument("--risk-alpha", type=float, default=0.5)
    parser.add_argument("--meta-epochs", type=int, default=200)
    parser.add_argument("--meta-lr", type=float, default=1e-2)
    parser.add_argument("--task-mode", default="per_task_head", choices=["shared_onehot", "per_task_head", "independent"])
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = load_labeled_rows(args.manifest)
    summary, selections, scored = evaluate(
        rows,
        feature_modes=args.feature_mode,
        combine=args.combine,
        task_mode=args.task_mode,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        risk_alpha=args.risk_alpha,
        meta_epochs=args.meta_epochs,
        meta_lr=args.meta_lr,
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
