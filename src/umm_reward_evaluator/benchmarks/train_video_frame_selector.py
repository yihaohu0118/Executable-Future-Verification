"""Train a held-out video-frame selector on benchmark manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from umm_reward_evaluator.benchmarks.common import load_jsonl, oracle_key


def video_path(row: dict[str, Any]) -> str:
    path = row.get("rollout_video_path") or row.get("rollout_video")
    if not path:
        raise ValueError(f"row {row.get('case_id')}/{row.get('candidate_id')} has no rollout video path")
    return str(path)


def sample_video(path: str | Path, max_frames: int, image_size: int) -> np.ndarray:
    import imageio.v3 as iio

    frames = list(iio.imiter(path))
    if not frames:
        raise ValueError(f"video has no frames: {path}")
    idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
    arrs = []
    for i in idx:
        img = Image.fromarray(np.asarray(frames[int(i)], dtype=np.uint8)).convert("RGB")
        img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
        arrs.append(np.asarray(img, dtype=np.float32) / 255.0)
    return np.stack(arrs, axis=0)


def frame_features(row: dict[str, Any], *, max_frames: int, image_size: int, mode: str) -> np.ndarray:
    if mode == "zero":
        return np.zeros((max_frames * image_size * image_size * 3,), dtype=np.float32)
    frames = sample_video(video_path(row), max_frames=max_frames, image_size=image_size)
    if mode == "shuffle_time":
        rng = np.random.default_rng(abs(hash(row["case_id"])) % (2**32))
        frames = frames[rng.permutation(frames.shape[0])]
    elif mode != "raw":
        raise ValueError(f"Unknown feature mode {mode}")
    return frames.reshape(-1).astype(np.float32)


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
    train_features: np.ndarray,
    test_features: np.ndarray,
    *,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
) -> list[dict[str, Any]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x_train = train_features
    x_test = test_features
    y_train = np.asarray([1.0 if row["oracle_success"] else 0.0 for row in train_rows], dtype=np.float32)

    # Fixed random projection keeps the visual baseline cheap and reduces pixel shortcut capacity.
    proj_dim = min(256, x_train.shape[1])
    projection = rng.normal(0.0, 1.0 / np.sqrt(x_train.shape[1]), size=(x_train.shape[1], proj_dim)).astype(np.float32)
    x_train = x_train @ projection
    x_test = x_test @ projection
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

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
    out = []
    for row, score in zip(test_rows, scores):
        payload = dict(row)
        payload["video_frame_selector_score"] = float(score)
        out.append(payload)
    return out


def evaluate(
    rows: list[dict[str, Any]],
    *,
    feature_mode: str,
    max_frames: int,
    image_size: int,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
):
    features = np.stack(
        [frame_features(row, max_frames=max_frames, image_size=image_size, mode=feature_mode) for row in rows]
    )
    cases: dict[str, list[dict[str, Any]]] = {}
    case_indices: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        cases.setdefault(str(row["case_id"]), []).append(row)
        case_indices.setdefault(str(row["case_id"]), []).append(idx)

    scored: list[dict[str, Any]] = []
    case_ids = sorted(cases)
    for fold, case_id in enumerate(case_ids):
        train_indices = [idx for other_id in case_ids if other_id != case_id for idx in case_indices[other_id]]
        test_indices = case_indices[case_id]
        train_rows = [rows[idx] for idx in train_indices]
        scored.extend(
            train_fold(
                train_rows,
                cases[case_id],
                features[train_indices],
                features[test_indices],
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
        selector = max(case_rows, key=lambda row: row["video_frame_selector_score"])
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
                "selector_score": selector["video_frame_selector_score"],
            }
        )

    num_cases = len(grouped)
    summary = {
        "cases": num_cases,
        "feature_mode": feature_mode,
        "max_frames": max_frames,
        "image_size": image_size,
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
    parser.add_argument("--feature-mode", default="raw", choices=["raw", "shuffle_time", "zero"])
    parser.add_argument("--max-frames", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    summary, selections, scored = evaluate(
        rows,
        feature_mode=args.feature_mode,
        max_frames=args.max_frames,
        image_size=args.image_size,
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
