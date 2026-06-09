"""Train a held-out action-sequence selector on benchmark manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from umm_reward_evaluator.benchmarks.common import load_jsonl, oracle_key


def action_features(row: dict[str, Any], *, mode: str) -> np.ndarray:
    actions = np.asarray(row["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] == 0:
        actions = np.zeros((1, 7), dtype=np.float32)
    zero_features = mode == "zero"
    drop_length = mode == "raw_no_length"
    if mode == "shuffle_time":
        rng = np.random.default_rng(abs(hash(row["case_id"])) % (2**32))
        actions = actions[rng.permutation(actions.shape[0])]
    elif mode not in {"raw", "raw_no_length", "zero"}:
        raise ValueError(f"Unknown feature mode {mode}")

    length_feature = 0.0 if drop_length else actions.shape[0] / 200.0
    first = actions[0]
    last = actions[-1]
    mean = actions.mean(axis=0)
    std = actions.std(axis=0)
    amin = actions.min(axis=0)
    amax = actions.max(axis=0)
    abs_mean = np.abs(actions).mean(axis=0)
    feature = np.concatenate(
        [
            np.array([length_feature], dtype=np.float32),
            first,
            last,
            mean,
            std,
            amin,
            amax,
            abs_mean,
        ]
    )
    if zero_features:
        feature = np.zeros_like(feature)
    return feature.astype(np.float32)


class SelectorMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_fold(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
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
    x_test = np.stack([action_features(row, mode=feature_mode) for row in test_rows])

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    model = SelectorMLP(x_train.shape[1], hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    y = torch.from_numpy(y_train)
    pos = float(y.sum().item())
    neg = float(y.numel() - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    x = torch.from_numpy(x_train)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        scores = torch.sigmoid(model(torch.from_numpy(x_test))).cpu().numpy()
    out = []
    for row, score in zip(test_rows, scores):
        payload = dict(row)
        payload["action_sequence_selector_score"] = float(score)
        out.append(payload)
    return out


def evaluate(rows: list[dict[str, Any]], *, feature_mode: str, hidden: int, epochs: int, lr: float, seed: int):
    cases: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cases.setdefault(str(row["case_id"]), []).append(row)

    scored: list[dict[str, Any]] = []
    case_ids = sorted(cases)
    for fold, case_id in enumerate(case_ids):
        train_rows = [row for other_id in case_ids if other_id != case_id for row in cases[other_id]]
        test_rows = cases[case_id]
        scored.extend(
            train_fold(
                train_rows,
                test_rows,
                feature_mode=feature_mode,
                hidden=hidden,
                epochs=epochs,
                lr=lr,
                seed=seed + fold,
            )
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in scored:
        grouped.setdefault(str(row["case_id"]), []).append(row)

    rank0_success = 0
    selector_success = 0
    oracle_success = 0
    selector_oracle_match = 0
    recovered_rank0_fail = 0
    rank0_failures = 0
    selections = []
    for case_id, case_rows in grouped.items():
        rank0 = min(case_rows, key=lambda row: row["candidate_rank_by_planner"])
        selector = max(case_rows, key=lambda row: row["action_sequence_selector_score"])
        oracle = max(case_rows, key=oracle_key)
        rank0_success += int(bool(rank0["oracle_success"]))
        selector_success += int(bool(selector["oracle_success"]))
        oracle_success += int(bool(oracle["oracle_success"]))
        selector_oracle_match += int(selector["candidate_id"] == oracle["candidate_id"])
        if not rank0["oracle_success"]:
            rank0_failures += 1
            recovered_rank0_fail += int(bool(selector["oracle_success"]))
        selections.append(
            {
                "case_id": case_id,
                "rank0_candidate_id": rank0["candidate_id"],
                "selector_candidate_id": selector["candidate_id"],
                "oracle_candidate_id": oracle["candidate_id"],
                "rank0_success": bool(rank0["oracle_success"]),
                "selector_success": bool(selector["oracle_success"]),
                "oracle_success": bool(oracle["oracle_success"]),
                "selector_score": selector["action_sequence_selector_score"],
            }
        )

    num_cases = len(grouped)
    summary = {
        "cases": num_cases,
        "feature_mode": feature_mode,
        "rank0_success": rank0_success,
        "selector_success": selector_success,
        "oracle_success": oracle_success,
        "selector_oracle_match": selector_oracle_match,
        "rank0_failures": rank0_failures,
        "recovered_rank0_fail": recovered_rank0_fail,
        "rank0_success_rate": rank0_success / num_cases if num_cases else 0.0,
        "selector_success_rate": selector_success / num_cases if num_cases else 0.0,
        "oracle_success_rate": oracle_success / num_cases if num_cases else 0.0,
    }
    return summary, selections, scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-mode", default="raw", choices=["raw", "raw_no_length", "shuffle_time", "zero"])
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    summary, selections, scored = evaluate(
        rows,
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
