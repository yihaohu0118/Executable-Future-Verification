"""Train a held-out action critic and use it as a failure-gated override."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from umm_reward_evaluator.benchmarks.common import load_jsonl, oracle_key
from umm_reward_evaluator.benchmarks.train_action_sequence_selector import SelectorMLP, action_features


def train_and_score(
    train_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    *,
    feature_mode: str,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
) -> list[dict[str, Any]]:
    torch.manual_seed(seed)
    x_train = np.stack([action_features(row, mode=feature_mode) for row in train_rows])
    y_train = np.asarray([1.0 if row["oracle_success"] else 0.0 for row in train_rows], dtype=np.float32)
    x_score = np.stack([action_features(row, mode=feature_mode) for row in score_rows])
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_score = (x_score - mean) / std

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
        scores = torch.sigmoid(model(torch.from_numpy(x_score))).cpu().numpy()
    out = []
    for row, score in zip(score_rows, scores):
        payload = dict(row)
        payload["gated_action_sequence_selector_score"] = float(score)
        out.append(payload)
    return out


def group_by_case(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), []).append(row)
    return grouped


def pick_global(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        case_rows,
        key=lambda row: (
            row["gated_action_sequence_selector_score"],
            -int(row.get("candidate_rank_by_planner", 999)),
        ),
    )


def pick_gated(case_rows: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    rank0 = min(case_rows, key=lambda row: int(row["candidate_rank_by_planner"]))
    if float(rank0["gated_action_sequence_selector_score"]) >= threshold:
        return rank0
    return pick_global(case_rows)


def choose_threshold(scored_train_rows: list[dict[str, Any]]) -> float:
    grouped = group_by_case(scored_train_rows)
    rank0_scores = [
        float(min(case_rows, key=lambda row: int(row["candidate_rank_by_planner"]))["gated_action_sequence_selector_score"])
        for case_rows in grouped.values()
    ]
    thresholds = [-float("inf"), float("inf")]
    for score in sorted(set(rank0_scores)):
        thresholds.append(float(score))

    best = (-1, -1, -float("inf"))
    best_threshold = float("inf")
    for threshold in thresholds:
        successes = 0
        preserved = 0
        for case_rows in grouped.values():
            rank0 = min(case_rows, key=lambda row: int(row["candidate_rank_by_planner"]))
            selected = pick_gated(case_rows, threshold=threshold)
            successes += int(bool(selected["oracle_success"]))
            preserved += int(selected["candidate_id"] == rank0["candidate_id"])
        key = (successes, preserved, -threshold)
        if key > best:
            best = key
            best_threshold = threshold
    return best_threshold


def evaluate(rows: list[dict[str, Any]], *, feature_mode: str, hidden: int, epochs: int, lr: float, seed: int):
    cases = group_by_case(rows)
    scored_test: list[dict[str, Any]] = []
    selections = []
    thresholds = []
    for fold, case_id in enumerate(sorted(cases)):
        train_rows = [row for other_id in cases if other_id != case_id for row in cases[other_id]]
        test_rows = cases[case_id]
        scored_train = train_and_score(
            train_rows,
            train_rows,
            feature_mode=feature_mode,
            hidden=hidden,
            epochs=epochs,
            lr=lr,
            seed=seed + fold,
        )
        threshold = choose_threshold(scored_train)
        thresholds.append(threshold)
        scored_case = train_and_score(
            train_rows,
            test_rows,
            feature_mode=feature_mode,
            hidden=hidden,
            epochs=epochs,
            lr=lr,
            seed=seed + fold,
        )
        scored_test.extend(scored_case)

        rank0 = min(scored_case, key=lambda row: int(row["candidate_rank_by_planner"]))
        global_selected = pick_global(scored_case)
        gated_selected = pick_gated(scored_case, threshold=threshold)
        oracle = max(scored_case, key=oracle_key)
        selections.append(
            {
                "case_id": case_id,
                "threshold": threshold,
                "rank0_candidate_id": rank0["candidate_id"],
                "global_candidate_id": global_selected["candidate_id"],
                "gated_candidate_id": gated_selected["candidate_id"],
                "oracle_candidate_id": oracle["candidate_id"],
                "rank0_success": bool(rank0["oracle_success"]),
                "global_success": bool(global_selected["oracle_success"]),
                "gated_success": bool(gated_selected["oracle_success"]),
                "oracle_success": bool(oracle["oracle_success"]),
                "gated_preserved_rank0": gated_selected["candidate_id"] == rank0["candidate_id"],
            }
        )

    num_cases = len(selections)
    rank0_success = sum(int(row["rank0_success"]) for row in selections)
    global_success = sum(int(row["global_success"]) for row in selections)
    gated_success = sum(int(row["gated_success"]) for row in selections)
    oracle_success = sum(int(row["oracle_success"]) for row in selections)
    rank0_failures = num_cases - rank0_success
    summary = {
        "cases": num_cases,
        "feature_mode": feature_mode,
        "rank0_success": rank0_success,
        "global_selector_success": global_success,
        "gated_selector_success": gated_success,
        "oracle_success": oracle_success,
        "rank0_failures": rank0_failures,
        "global_recovered_rank0_fail": sum(
            int((not row["rank0_success"]) and row["global_success"]) for row in selections
        ),
        "gated_recovered_rank0_fail": sum(int((not row["rank0_success"]) and row["gated_success"]) for row in selections),
        "gated_preserved_rank0": sum(int(row["gated_preserved_rank0"]) for row in selections),
        "rank0_success_rate": rank0_success / num_cases if num_cases else 0.0,
        "global_selector_success_rate": global_success / num_cases if num_cases else 0.0,
        "gated_selector_success_rate": gated_success / num_cases if num_cases else 0.0,
        "oracle_success_rate": oracle_success / num_cases if num_cases else 0.0,
        "threshold_mean": float(np.mean(thresholds)) if thresholds else 0.0,
    }
    return summary, selections, scored_test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-mode", default="raw_no_length", choices=["raw", "raw_no_length", "shuffle_time", "zero"])
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary, selections, scored = evaluate(
        load_jsonl(args.manifest),
        feature_mode=args.feature_mode,
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
