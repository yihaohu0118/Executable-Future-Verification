from __future__ import annotations

import argparse
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

from umm_reward_evaluator.manifest import group_by_case, read_jsonl, validate_row, write_jsonl
from umm_reward_evaluator.media import discover_frame_paths


def with_suffix(row: dict[str, Any], negative_type: str) -> dict[str, Any]:
    out = deepcopy(row)
    out["source_candidate_id"] = row.get("candidate_id")
    out["candidate_id"] = f"{row.get('candidate_id')}_{negative_type}"
    out["negative_type"] = negative_type
    out["is_hard_negative"] = True
    return out


def temporal_reverse(row: dict[str, Any]) -> dict[str, Any]:
    out = with_suffix(row, "temporal_reverse")
    frames = discover_frame_paths(row)
    if frames:
        out["frame_paths"] = list(reversed(frames))
        out.pop("frames_dir", None)
        out.pop("rollout_video", None)
    return out


def temporal_shuffle(row: dict[str, Any], seed: int) -> dict[str, Any]:
    out = with_suffix(row, "temporal_shuffle")
    frames = discover_frame_paths(row)
    if frames:
        rng = random.Random(f"{seed}:{row.get('case_id')}:{row.get('candidate_id')}")
        shuffled = frames[:]
        rng.shuffle(shuffled)
        out["frame_paths"] = shuffled
        out.pop("frames_dir", None)
        out.pop("rollout_video", None)
    return out


def action_shuffle(row: dict[str, Any], seed: int) -> dict[str, Any]:
    out = with_suffix(row, "action_shuffle")
    actions = row.get("actions")
    if isinstance(actions, list):
        rng = random.Random(f"{seed}:actions:{row.get('case_id')}:{row.get('candidate_id')}")
        shuffled = actions[:]
        rng.shuffle(shuffled)
        out["actions"] = shuffled
    else:
        out["action_corruption_note"] = "actions were not inline; evaluator should treat action/rollout match as uncertain"
    return out


def goal_swap(rows_by_case: dict[str, list[dict[str, Any]]], row: dict[str, Any]) -> dict[str, Any] | None:
    if not row.get("goal_frame") and not row.get("goal_image") and not row.get("instruction"):
        return None
    case_ids = sorted(k for k in rows_by_case if k != row.get("case_id"))
    if not case_ids:
        return None
    donor = rows_by_case[case_ids[0]][0]
    out = with_suffix(row, "goal_swap")
    for key in ("goal_frame", "goal_image", "instruction"):
        if donor.get(key):
            out[key] = donor[key]
    out["goal_swap_source_case_id"] = donor.get("case_id")
    return out


def build_negatives(rows: list[dict[str, Any]], types: list[str], seed: int) -> list[dict[str, Any]]:
    for row in rows:
        validate_row(row)
    rows_by_case = group_by_case(rows)
    negatives: list[dict[str, Any]] = []
    for row in rows:
        if "temporal_reverse" in types:
            negatives.append(temporal_reverse(row))
        if "temporal_shuffle" in types:
            negatives.append(temporal_shuffle(row, seed))
        if "action_shuffle" in types:
            negatives.append(action_shuffle(row, seed))
        if "goal_swap" in types:
            neg = goal_swap(rows_by_case, row)
            if neg is not None:
                negatives.append(neg)
    return negatives


def main() -> None:
    parser = argparse.ArgumentParser(description="Create hard-negative rollout manifest rows.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--types",
        default="temporal_reverse,temporal_shuffle,action_shuffle,goal_swap",
        help="Comma-separated negative types.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-originals", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    types = [x.strip() for x in args.types.split(",") if x.strip()]
    negatives = build_negatives(rows, types, args.seed)
    out_rows = rows + negatives if args.include_originals else negatives
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, out_rows)
    print(f"[hard_negatives] originals={len(rows)} negatives={len(negatives)} -> {args.output}")


if __name__ == "__main__":
    main()
