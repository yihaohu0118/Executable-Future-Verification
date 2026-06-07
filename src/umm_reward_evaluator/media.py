from __future__ import annotations

import base64
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def image_to_data_url(path: str | Path) -> str:
    p = Path(path)
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(p.suffix.lower(), "image/png")
    payload = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def discover_frame_paths(row: dict[str, Any]) -> list[str]:
    if row.get("frame_paths"):
        return [str(p) for p in row["frame_paths"]]
    if row.get("frames_dir"):
        root = Path(row["frames_dir"])
        return [str(p) for p in sorted(root.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    return []


def sample_frames(paths: list[str], num_frames: int) -> list[str]:
    if num_frames <= 0 or len(paths) <= num_frames:
        return paths
    if num_frames == 1:
        return [paths[len(paths) // 2]]
    idxs = [
        round(i * (len(paths) - 1) / (num_frames - 1))
        for i in range(num_frames)
    ]
    return [paths[i] for i in idxs]


def summarize_actions(row: dict[str, Any], max_values: int = 24) -> str:
    actions = row.get("actions")
    if actions is None:
        if row.get("actions_path"):
            return f"Actions are stored at {row['actions_path']}."
        return "No explicit action sequence was provided."
    flat: list[Any] = []
    for item in actions:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    clipped = flat[:max_values]
    suffix = " ..." if len(flat) > max_values else ""
    return f"Action sequence values: {clipped}{suffix}"
