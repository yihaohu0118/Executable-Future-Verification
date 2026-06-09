"""Train a held-out fusion selector from action and video critic scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from umm_reward_evaluator.benchmarks.common import load_jsonl, oracle_key


def row_id(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["case_id"]), str(row["candidate_id"])


def score_map(rows: list[dict[str, Any]], field: str) -> dict[tuple[str, str], float]:
    out = {}
    for row in rows:
        key = row_id(row)
        if field not in row:
            raise KeyError(f"{field} missing for {key}")
        out[key] = float(row[field])
    return out


def feature_vector(
    row: dict[str, Any],
    action_scores: dict[tuple[str, str], float],
    video_scores: dict[tuple[str, str], float],
    *,
    feature_set: str,
) -> np.ndarray:
    key = row_id(row)
    rank = float(row.get("candidate_rank_by_planner", 0) or 0.0)
    rank_norm = rank / 4.0
    action = action_scores[key]
    video = video_scores[key]
    if feature_set == "action":
        values = [action, rank_norm]
    elif feature_set == "video":
        values = [video, rank_norm]
    elif feature_set == "fusion":
        values = [action, video, 0.5 * (action + video), action - video, abs(action - video), rank_norm]
    elif feature_set == "zero":
        values = [0.0]
    else:
        raise ValueError(f"Unknown feature set {feature_set}")
    return np.asarray(values, dtype=np.float32)


class FusionMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        if hidden <= 0:
            self.net = nn.Linear(dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_fold(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    action_scores: dict[tuple[str, str], float],
    video_scores: dict[tuple[str, str], float],
    *,
    feature_set: str,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
) -> list[dict[str, Any]]:
    torch.manual_seed(seed)
    x_train = np.stack(
        [feature_vector(row, action_scores, video_scores, feature_set=feature_set) for row in train_rows]
    )
    x_test = np.stack([feature_vector(row, action_scores, video_scores, feature_set=feature_set) for row in test_rows])
    y_train = np.asarray([1.0 if row["oracle_success"] else 0.0 for row in train_rows], dtype=np.float32)

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    model = FusionMLP(x_train.shape[1], hidden)
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
    out = []
    for row, score in zip(test_rows, scores):
        payload = dict(row)
        payload["action_video_fusion_score"] = float(score)
        out.append(payload)
    return out


def evaluate(
    base_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    video_rows: list[dict[str, Any]],
    *,
    action_field: str,
    video_field: str,
    feature_set: str,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    action_scores = score_map(action_rows, action_field)
    video_scores = score_map(video_rows, video_field)
    base_keys = {row_id(row) for row in base_rows}
    if base_keys != set(action_scores) or base_keys != set(video_scores):
        raise ValueError("base/action/video manifests do not contain the same case/candidate keys")

    cases: dict[str, list[dict[str, Any]]] = {}
    for row in base_rows:
        cases.setdefault(str(row["case_id"]), []).append(row)

    case_ids = sorted(cases)
    scored: list[dict[str, Any]] = []
    for fold, case_id in enumerate(case_ids):
        train_rows = [row for other_id in case_ids if other_id != case_id for row in cases[other_id]]
        scored.extend(
            train_fold(
                train_rows,
                cases[case_id],
                action_scores,
                video_scores,
                feature_set=feature_set,
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
    fusion_success = 0
    oracle_success = 0
    fusion_oracle_match = 0
    rank0_failures = 0
    recovered_rank0_fail = 0
    selections = []
    for case_id, case_rows in grouped.items():
        rank0 = min(case_rows, key=lambda row: row["candidate_rank_by_planner"])
        selected = max(
            case_rows,
            key=lambda row: (
                row["action_video_fusion_score"],
                -int(row.get("candidate_rank_by_planner", 999)),
            ),
        )
        oracle = max(case_rows, key=oracle_key)
        rank0_success += int(bool(rank0["oracle_success"]))
        fusion_success += int(bool(selected["oracle_success"]))
        oracle_success += int(bool(oracle["oracle_success"]))
        fusion_oracle_match += int(selected["candidate_id"] == oracle["candidate_id"])
        if not rank0["oracle_success"]:
            rank0_failures += 1
            recovered_rank0_fail += int(bool(selected["oracle_success"]))
        selections.append(
            {
                "case_id": case_id,
                "rank0_candidate_id": rank0["candidate_id"],
                "fusion_candidate_id": selected["candidate_id"],
                "oracle_candidate_id": oracle["candidate_id"],
                "rank0_success": bool(rank0["oracle_success"]),
                "fusion_success": bool(selected["oracle_success"]),
                "oracle_success": bool(oracle["oracle_success"]),
                "fusion_score": selected["action_video_fusion_score"],
            }
        )

    n = len(grouped)
    summary = {
        "cases": n,
        "feature_set": feature_set,
        "rank0_success": rank0_success,
        "fusion_success": fusion_success,
        "oracle_success": oracle_success,
        "fusion_oracle_match": fusion_oracle_match,
        "rank0_failures": rank0_failures,
        "recovered_rank0_fail": recovered_rank0_fail,
        "rank0_success_rate": rank0_success / n if n else 0.0,
        "fusion_success_rate": fusion_success / n if n else 0.0,
        "oracle_success_rate": oracle_success / n if n else 0.0,
    }
    return summary, selections, scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--action-scored-manifest", type=Path, required=True)
    parser.add_argument("--video-scored-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action-field", default="action_sequence_selector_score")
    parser.add_argument("--video-field", default="video_frame_selector_score")
    parser.add_argument("--feature-set", default="fusion", choices=["action", "video", "fusion", "zero"])
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary, selections, scored = evaluate(
        load_jsonl(args.base_manifest),
        load_jsonl(args.action_scored_manifest),
        load_jsonl(args.video_scored_manifest),
        action_field=args.action_field,
        video_field=args.video_field,
        feature_set=args.feature_set,
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

