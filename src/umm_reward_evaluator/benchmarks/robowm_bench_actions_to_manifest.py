"""Convert RoboWM-Bench action/evaluation outputs into candidate manifests."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.common import annotate_oracle_best, summarize_headroom, write_jsonl


EPISODE_RE = re.compile(r"(?:episode_)?(\d+)$")
REPLAY_RE = re.compile(r"Replay result:\s+episode=(\S+)\s+success=(True|False)")
PART_RE = re.compile(r"part_scores:\s+episode=(\S+)\s+part=(\S+)\s+scores=(\S+)")


def _episode_index(stem: str) -> int:
    match = EPISODE_RE.fullmatch(stem)
    if not match:
        raise ValueError(f"cannot parse episode index from {stem!r}")
    return int(match.group(1))


def _episode_name(index: int) -> str:
    return f"episode_{index:06d}"


def _episode_keys(name_or_index: str | int) -> set[str]:
    index = int(name_or_index) if isinstance(name_or_index, int) else _episode_index(str(name_or_index))
    return {str(index), _episode_name(index)}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"true", "1", "yes"}:
            return True
        if value.lower() in {"false", "0", "no"}:
            return False
    raise ValueError(f"cannot parse boolean from {value!r}")


def _parse_float(value: str) -> float | None:
    if value in {"None", "nan", "NaN"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _read_actions(path: Path) -> list[list[float]]:
    actions: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, list):
                raise ValueError(f"{path}:{line_no} action row must be a JSON list")
            actions.append([float(item) for item in payload])
    return actions


def _parse_candidate_spec(spec: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in spec.split(","):
        if "=" not in item:
            raise ValueError(f"candidate spec item must be key=value: {item!r}")
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()
    required = {"id", "rank", "path"}
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"candidate spec missing keys: {missing}")
    return result


def _load_eval_jsonl(paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                candidate_id = str(row["candidate_id"])
                episode = str(row.get("episode", row.get("episode_name", row.get("case_id"))))
                if episode.startswith("episode_") or episode.isdigit():
                    for key in _episode_keys(episode):
                        results[(candidate_id, key)] = row
                else:
                    results[(candidate_id, episode)] = row
    return results


def _load_eval_log(path: Path, candidate_id: str) -> dict[tuple[str, str], dict[str, Any]]:
    results: dict[tuple[str, str], dict[str, Any]] = {}
    parts: dict[str, tuple[float | None, float | None]] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in PART_RE.finditer(text):
        episode, part, score = match.groups()
        for key in _episode_keys(episode):
            parts[key] = (_parse_float(part), _parse_float(score))
    for match in REPLAY_RE.finditer(text):
        episode, success = match.groups()
        for key in _episode_keys(episode):
            part, score = parts.get(key, (None, None))
            results[(candidate_id, key)] = {
                "candidate_id": candidate_id,
                "episode": key,
                "success": _parse_bool(success),
                "oracle_progress": part,
                "oracle_return": score,
                "eval_log_path": str(path),
            }
    return results


def _instruction(task_name: str, episode: str, instruction_root: Path | None) -> tuple[str | None, str | None]:
    if instruction_root is None:
        return None, None
    path = instruction_root / task_name / f"{episode}.txt"
    if not path.exists() and task_name == "push_button":
        path = instruction_root / "push_button" / f"{episode}.txt"
    if not path.exists():
        return None, None
    return path.read_text(encoding="utf-8").strip(), str(path)


def _video_path(video_root: str | None, episode: str, candidate_id: str) -> str:
    if not video_root:
        return ""
    root = Path(video_root)
    candidates = [
        root / f"{episode}.mp4",
        root / f"{episode}_{candidate_id}.mp4",
        root / candidate_id / f"{episode}.mp4",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def convert_candidates(
    candidate_specs: list[str],
    *,
    task_name: str,
    suite: str,
    instruction_root: Path | None,
    eval_results: list[Path],
    episode_indices: set[int] | None,
) -> list[dict[str, Any]]:
    eval_by_candidate_episode = _load_eval_jsonl(eval_results)
    rows: list[dict[str, Any]] = []

    for spec_text in candidate_specs:
        spec = _parse_candidate_spec(spec_text)
        candidate_id = spec["id"]
        rank = int(spec["rank"])
        action_root = Path(spec["path"])
        if not action_root.exists():
            raise FileNotFoundError(f"candidate action directory does not exist: {action_root}")
        if "log" in spec:
            eval_by_candidate_episode.update(_load_eval_log(Path(spec["log"]), candidate_id))

        for action_path in sorted(action_root.glob("*.json")):
            if action_path.name == "pose.jsonl":
                continue
            try:
                episode_idx = _episode_index(action_path.stem)
            except ValueError:
                continue
            episode = _episode_name(episode_idx)
            if episode_indices is not None and episode_idx not in episode_indices:
                continue
            result = None
            for key in _episode_keys(episode_idx):
                result = eval_by_candidate_episode.get((candidate_id, key))
                if result is not None:
                    break
            if result is None:
                raise ValueError(
                    f"missing eval result for candidate={candidate_id!r} episode={episode!r}; "
                    "provide --eval-results or log=... in --candidate"
                )

            instruction, instruction_path = _instruction(task_name, episode, instruction_root)
            metadata = {
                "future_source": spec.get("source", "world_model_video"),
                "future_representation": spec.get("representation", "video_to_actions"),
                "verification_target": "task_success",
                "robowm_task": task_name,
                "action_json_path": str(action_path),
            }
            if instruction_path is not None:
                metadata["instruction_path"] = instruction_path
            if "eval_log_path" in result:
                metadata["eval_log_path"] = result["eval_log_path"]
            if "model" in spec:
                metadata["video_model_name"] = spec["model"]
            if "generated_video_root" in spec:
                metadata["generated_video_root"] = spec["generated_video_root"]

            rows.append(
                {
                    "benchmark": "robowm_bench",
                    "suite": suite,
                    "task_name": task_name,
                    "case_id": f"{task_name}:{episode}",
                    "candidate_id": f"{candidate_id}:{episode}",
                    "candidate_rank_by_planner": rank,
                    "rollout_video_path": _video_path(spec.get("video_root"), episode, candidate_id),
                    "rollout_video_layout": "world_model_or_replay_rollout",
                    "actions": _read_actions(action_path),
                    "oracle_success": _parse_bool(result.get("oracle_success", result.get("success"))),
                    "instruction": instruction,
                    "planner_score": result.get("planner_score", spec.get("planner_score")),
                    "oracle_return": result.get("oracle_return", result.get("score")),
                    "oracle_progress": result.get("oracle_progress", result.get("part")),
                    "metadata": metadata,
                }
            )

    return annotate_oracle_best(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Comma-separated spec: id=veo,rank=0,path=/actions,log=/eval.log,source=world_model_video",
    )
    parser.add_argument("--instruction-root", type=Path)
    parser.add_argument("--eval-results", action="append", type=Path, default=[])
    parser.add_argument("--episode-index", action="append", type=int)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path)
    args = parser.parse_args()

    rows = convert_candidates(
        args.candidate,
        task_name=args.task_name,
        suite=args.suite,
        instruction_root=args.instruction_root,
        eval_results=args.eval_results,
        episode_indices=set(args.episode_index) if args.episode_index else None,
    )
    write_jsonl(args.output_manifest, rows)

    summary = summarize_headroom(rows)
    summary.update(
        {
            "benchmark": "robowm_bench",
            "task_name": args.task_name,
            "suite": args.suite,
            "output_manifest": str(args.output_manifest),
        }
    )
    summary_path = args.output_summary or args.output_manifest.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
