from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from umm_reward_evaluator.manifest import read_jsonl, write_jsonl
from umm_reward_evaluator.media import discover_frame_paths


def load_rgb(path: str | Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape == reference.shape:
        return image
    from PIL import Image

    h, w = reference.shape[:2]
    pil = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).resize((w, h))
    return np.asarray(pil, dtype=np.float32)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    value = mse(a, b)
    if value <= 1e-12:
        return 100.0
    return float(10.0 * math.log10((255.0 ** 2) / value))


def ssim_global(a: np.ndarray, b: np.ndarray) -> float:
    """Small dependency-free SSIM approximation over whole RGB image."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a = a.mean()
    mu_b = b.mean()
    var_a = a.var()
    var_b = b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)))


def score_row(row: dict[str, Any]) -> dict[str, float] | None:
    frames = discover_frame_paths(row)
    goal_frame = row.get("goal_frame") or row.get("goal_image")
    if not frames or not goal_frame:
        return None
    final = load_rgb(frames[-1])
    goal = resize_like(load_rgb(goal_frame), final)
    initial = load_rgb(frames[0])
    initial = resize_like(initial, final)
    return {
        "final_goal_mse": mse(final, goal),
        "final_goal_psnr": psnr(final, goal),
        "final_goal_ssim": ssim_global(final, goal),
        "initial_final_mse": mse(initial, final),
        "num_metric_frames": float(len(frames)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute simple pixel baselines for rollout manifests.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--drop-missing", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    out_rows = []
    missing = 0
    for row in rows:
        metrics = score_row(row)
        if metrics is None:
            missing += 1
            if args.drop_missing:
                continue
            row = dict(row)
            row["pixel_metrics"] = None
        else:
            row = dict(row)
            row["pixel_metrics"] = metrics
            row.update(metrics)
        out_rows.append(row)
    write_jsonl(args.output, out_rows)
    print(f"[pixel] wrote {len(out_rows)} rows -> {args.output}; missing={missing}")


if __name__ == "__main__":
    main()
