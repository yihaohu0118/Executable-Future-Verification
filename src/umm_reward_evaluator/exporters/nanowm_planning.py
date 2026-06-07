from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from umm_reward_evaluator.manifest import write_jsonl


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_task(results: dict[str, Any], fallback: str) -> str:
    cfg = results.get("planning_config") or {}
    return str(cfg.get("env_name") or fallback)


def default_instruction(task: str) -> str:
    if task == "pusht":
        return "Push the T-shaped block to match the goal pose."
    if task == "point_maze":
        return "Move the point mass from the initial position to the goal position."
    return f"Reach the goal state in the {task} environment."


def extract_video_frames(
    video_path: Path,
    out_dir: Path,
    max_frames: int,
    split_side_by_side: bool,
) -> tuple[list[str], str | None, str | None]:
    """Extract sampled frames from NanoWM planning videos.

    NanoWM planning videos concatenate current rollout frame and goal frame
    horizontally. When `split_side_by_side` is true, this saves only the left
    half as rollout frames and writes the right half from the first frame as
    `goal.png`.
    """
    import imageio.v3 as iio
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = list(iio.imiter(video_path))
    if not frames:
        return [], None, None
    if max_frames > 0 and len(frames) > max_frames:
        idxs = [round(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)]
        frames = [frames[i] for i in idxs]

    frame_paths: list[str] = []
    initial_frame: str | None = None
    goal_frame: str | None = None
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        if split_side_by_side:
            w, h = img.size
            left = img.crop((0, 0, w // 2, h))
            right = img.crop((w // 2, 0, w, h))
            if goal_frame is None:
                goal_path = out_dir / "goal.png"
                right.save(goal_path)
                goal_frame = str(goal_path)
            img = left
        out_path = out_dir / f"frame_{i:04d}.png"
        img.save(out_path)
        if initial_frame is None:
            initial_frame = str(out_path)
        frame_paths.append(str(out_path))
    return frame_paths, initial_frame, goal_frame


def build_manifest_rows(
    planning_dir: Path,
    output_frames_root: Path | None,
    max_frames: int,
    split_side_by_side: bool,
    task: str | None,
    instruction: str | None,
    include_missing_videos: bool,
) -> list[dict[str, Any]]:
    results_path = planning_dir / "planning_results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"missing NanoWM planning results: {results_path}")
    results = load_json(results_path)
    targets_path = planning_dir / "planning_targets.json"
    targets = load_json(targets_path) if targets_path.is_file() else []
    targets_by_episode = {
        int(row["episode"]): row
        for row in targets
        if isinstance(row, dict) and row.get("episode") is not None
    }

    task_name = task or infer_task(results, planning_dir.name)
    instr = instruction or default_instruction(task_name)
    rows: list[dict[str, Any]] = []
    for episode in results.get("per_episode", []):
        ep_idx = int(episode["episode"])
        video_path = planning_dir / f"episode_{ep_idx:03d}.mp4"
        if not video_path.is_file() and not include_missing_videos:
            continue
        row: dict[str, Any] = {
            "case_id": f"{task_name}_episode_{ep_idx:03d}",
            "candidate_id": "nanowm_planned",
            "task": task_name,
            "instruction": instr,
            "oracle_reward": float(episode.get("success", 0.0)),
            "oracle_success": bool(episode.get("success", 0.0)),
            "source": "nanowm_planning",
            "planning_metrics": episode,
        }
        if video_path.is_file():
            row["rollout_video"] = str(video_path)
        target = targets_by_episode.get(ep_idx)
        if target is not None:
            row["planning_target"] = target
        if output_frames_root is not None and video_path.is_file():
            frames_dir = output_frames_root / f"episode_{ep_idx:03d}"
            frame_paths, initial_frame, goal_frame = extract_video_frames(
                video_path=video_path,
                out_dir=frames_dir,
                max_frames=max_frames,
                split_side_by_side=split_side_by_side,
            )
            row["frames_dir"] = str(frames_dir)
            row["frame_paths"] = frame_paths
            if initial_frame:
                row["initial_frame"] = initial_frame
            if goal_frame:
                row["goal_frame"] = goal_frame
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NanoWM planning_results into rollout_manifest.jsonl.")
    parser.add_argument("--planning-dir", required=True, help="Directory containing planning_results.json and episode_*.mp4.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--extract-frames-root", default=None)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--no-split-side-by-side", action="store_true")
    parser.add_argument("--task", default=None)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--include-missing-videos", action="store_true")
    args = parser.parse_args()

    rows = build_manifest_rows(
        planning_dir=Path(args.planning_dir),
        output_frames_root=Path(args.extract_frames_root) if args.extract_frames_root else None,
        max_frames=args.max_frames,
        split_side_by_side=not args.no_split_side_by_side,
        task=args.task,
        instruction=args.instruction,
        include_missing_videos=args.include_missing_videos,
    )
    write_jsonl(args.output, rows)
    print(f"[nanowm_planning] wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
