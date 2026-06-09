from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Dinov2Model

from umm_reward_evaluator.manifest import read_jsonl
from umm_reward_evaluator.training.ggp_umm_encoder_reward import split_prediction_goal_image
from umm_reward_evaluator.training.ggp_umm_pairwise_reward import is_expert, row_frame_arrays

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_tensor(img: Image.Image, image_size: int) -> torch.Tensor:
    img = img.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1)
    mean_t = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (x - mean_t) / std_t


def collect_images(rows: list[dict[str, Any]], max_frames: int) -> tuple[list[Image.Image], list[tuple[int, int, int]]]:
    images: list[Image.Image] = []
    index: list[tuple[int, int, int]] = []
    for row_idx, row in enumerate(rows):
        frames = row_frame_arrays(row, max_frames=max_frames)
        for frame in frames:
            pred, goal = split_prediction_goal_image(np.asarray(frame))
            pred_idx = len(images)
            images.append(pred)
            goal_idx = len(images)
            images.append(goal)
            index.append((row_idx, pred_idx, goal_idx))
    return images, index


@torch.inference_mode()
def encode_patch_tokens(
    images: list[Image.Image],
    model_name: str,
    image_size: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    model = Dinov2Model.from_pretrained(model_name).to(device).eval()
    chunks = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack([image_tensor(img, image_size=image_size) for img in images[start : start + batch_size]]).to(device)
        outputs = model(pixel_values=batch)
        tokens = F.normalize(outputs.last_hidden_state[:, 1:].float(), dim=-1)
        chunks.append(tokens.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def load_or_compute_row_tokens(
    rows: list[dict[str, Any]],
    cache_path: str,
    model_name: str,
    image_size: int,
    max_frames: int,
    batch_size: int,
    device: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    cache = Path(cache_path)
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        pred = data["pred_tokens"]
        goal = data["goal_tokens"]
        counts = data["counts"]
        out = []
        offset = 0
        for count in counts.tolist():
            out.append((pred[offset : offset + count], goal[offset : offset + count]))
            offset += count
        return out

    images, image_index = collect_images(rows, max_frames=max_frames)
    tokens = encode_patch_tokens(images, model_name=model_name, image_size=image_size, batch_size=batch_size, device=device)
    pred_lists: list[list[np.ndarray]] = [[] for _ in rows]
    goal_lists: list[list[np.ndarray]] = [[] for _ in rows]
    for row_idx, pred_idx, goal_idx in image_index:
        pred_lists[row_idx].append(tokens[pred_idx])
        goal_lists[row_idx].append(tokens[goal_idx])
    row_tokens = []
    pred_chunks = []
    goal_chunks = []
    counts = []
    for pred_list, goal_list in zip(pred_lists, goal_lists):
        pred_arr = np.asarray(pred_list, dtype=np.float32)
        goal_arr = np.asarray(goal_list, dtype=np.float32)
        row_tokens.append((pred_arr, goal_arr))
        pred_chunks.append(pred_arr)
        goal_chunks.append(goal_arr)
        counts.append(len(pred_arr))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        pred_tokens=np.concatenate(pred_chunks, axis=0),
        goal_tokens=np.concatenate(goal_chunks, axis=0),
        counts=np.asarray(counts, dtype=np.int32),
        model_name=np.asarray([model_name], dtype="U128"),
        image_size=np.asarray([image_size], dtype=np.int32),
    )
    return row_tokens


def safe_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    xs = np.arange(len(values), dtype=np.float64)
    return float(np.polyfit(xs, values.astype(np.float64), 1)[0])


def patch_features(pred: np.ndarray, goal: np.ndarray) -> dict[str, float]:
    if len(pred) == 0:
        return {}
    patch_cos = 1.0 - np.mean(np.sum(pred * goal, axis=-1), axis=-1)
    patch_l2 = np.mean(np.linalg.norm(pred - goal, axis=-1), axis=-1)
    token_motion = 1.0 - np.mean(np.sum(pred[1:] * pred[:-1], axis=-1), axis=-1) if len(pred) > 1 else np.asarray([], dtype=np.float32)
    first = float(patch_cos[0])
    final = float(patch_cos[-1])
    best = float(np.min(patch_cos))
    return {
        "patch_cos_first": first,
        "patch_cos_mean": float(np.mean(patch_cos)),
        "patch_cos_final": final,
        "patch_cos_min": best,
        "patch_cos_progress": first - final,
        "patch_cos_best_progress": first - best,
        "patch_cos_slope": safe_slope(patch_cos),
        "patch_l2_final": float(patch_l2[-1]),
        "patch_l2_mean": float(np.mean(patch_l2)),
        "patch_motion_mean": float(np.mean(token_motion)) if len(token_motion) else 0.0,
    }


def summarize(rows: list[dict[str, Any]], metric: str, reverse: bool) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(row)
    selected = []
    for case_id, items in sorted(by_case.items()):
        top = max(items, key=lambda r: float(r[metric])) if reverse else min(items, key=lambda r: float(r[metric]))
        rank0 = min(items, key=lambda r: int(r.get("candidate_rank_by_planner", 999)))
        oracle = min(items, key=lambda r: float(r.get("oracle_state_dist", 1e9)))
        selected.append(
            {
                "case_id": case_id,
                "candidate_id": top["candidate_id"],
                "state_dist": float(top["oracle_state_dist"]),
                "rank0_state_dist": float(rank0["oracle_state_dist"]),
                "oracle_best_cem_id": oracle["candidate_id"],
                "oracle_best_cem_state_dist": float(oracle["oracle_state_dist"]),
            }
        )
    dists = [x["state_dist"] for x in selected]
    rank0_dists = [x["rank0_state_dist"] for x in selected]
    return {
        "metric": metric,
        "selection": "max" if reverse else "min",
        "state_dist": {"mean": float(mean(dists)), "median": float(median(dists)), "min": float(min(dists)), "max": float(max(dists))},
        "better_than_rank0_count": int(sum(a < b for a, b in zip(dists, rank0_dists))),
        "matched_oracle_best_cem_count": int(sum(x["candidate_id"] == x["oracle_best_cem_id"] for x in selected)),
        "selected": selected,
    }


def write_report(path: str, args: argparse.Namespace, summary: list[dict[str, Any]]) -> None:
    lines = []
    n_cases = len(summary[0]["selected"]) if summary else 0
    for row in summary:
        d = row["state_dist"]
        lines.append(
            f"| {row['metric']} ({row['selection']}) | {d['mean']:.4f} | {d['median']:.4f} | "
            f"{row['better_than_rank0_count']}/{n_cases} | {row['matched_oracle_best_cem_count']}/{n_cases} |"
        )
    report = f"""# DINOv2 Patch-Token Similarity Baselines

## Setup

- Manifest: `{args.manifest}`
- Encoder: `{args.model_name}`
- Image size: {args.image_size}
- Patch-token cache: `{args.cache}`

These baselines compare spatially aligned DINOv2 patch tokens between prediction and goal frames, matching NanoWM's WebDINO-style dense encoder interface more closely than global pooled embeddings.

| Selector | Mean state dist | Median state dist | Better than rank0 | Matched oracle-best CEM |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(lines)}
"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dense DINOv2 patch-token similarity for CEM reranking.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--model-name", default="facebook/dinov2-small")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-frames", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    row_tokens = load_or_compute_row_tokens(
        rows=rows,
        cache_path=args.cache,
        model_name=args.model_name,
        image_size=args.image_size,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        device=args.device,
    )
    cem_rows = []
    for row, tokens in zip(rows, row_tokens):
        if is_expert(row):
            continue
        out = dict(row)
        out.update(patch_features(*tokens))
        cem_rows.append(out)
    metrics = [
        ("patch_cos_final", False),
        ("patch_cos_mean", False),
        ("patch_cos_min", False),
        ("patch_cos_progress", True),
        ("patch_cos_best_progress", True),
        ("patch_cos_slope", False),
        ("patch_l2_final", False),
        ("patch_l2_mean", False),
    ]
    summary = [summarize(cem_rows, metric, reverse) for metric, reverse in metrics]
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(args.report_output, args, summary)
    for row in summary:
        d = row["state_dist"]
        n_cases = len(row["selected"])
        print(
            f"[dinov2_patch] {row['metric']} {row['selection']} "
            f"mean={d['mean']:.4f} median={d['median']:.4f} "
            f"better={row['better_than_rank0_count']}/{n_cases} oracle={row['matched_oracle_best_cem_count']}/{n_cases}"
        )
    print(f"[dinov2_patch] report={args.report_output}")


if __name__ == "__main__":
    main()
