"""Create mixed-rank benchmark manifests by reassigning rank0 within each case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from umm_reward_evaluator.benchmarks.common import load_jsonl, write_jsonl


def randomize_case_ranks(
    case_rows: list[dict[str, Any]],
    *,
    seed: int,
    mode: str,
    remap_candidate_ids: bool,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    original = sorted(case_rows, key=lambda row: int(row.get("candidate_rank_by_planner", 999)))
    if mode == "failure_rank0_shuffle_rest":
        failed = [idx for idx, row in enumerate(original) if not bool(row.get("oracle_success"))]
        chosen_idx = int(rng.choice(failed)) if failed else int(rng.integers(0, len(original)))
        remaining = [idx for idx in range(len(original)) if idx != chosen_idx]
        rng.shuffle(remaining)
        assigned_indices = [chosen_idx, *remaining]
    else:
        if mode == "random":
            chosen_idx = int(rng.integers(0, len(original)))
        elif mode == "prefer_success":
            successful = [idx for idx, row in enumerate(original) if bool(row.get("oracle_success"))]
            chosen_idx = int(rng.choice(successful)) if successful else int(rng.integers(0, len(original)))
        elif mode == "prefer_failure":
            failed = [idx for idx, row in enumerate(original) if not bool(row.get("oracle_success"))]
            chosen_idx = int(rng.choice(failed)) if failed else int(rng.integers(0, len(original)))
        else:
            raise ValueError(f"Unknown mode {mode}")
        assigned_indices = [chosen_idx, *[idx for idx in range(len(original)) if idx != chosen_idx]]

    reranked = []
    for new_rank, original_idx in enumerate(assigned_indices):
        row = original[original_idx]
        payload = dict(row)
        metadata = dict(payload.get("metadata") or {})
        metadata["original_candidate_rank_by_planner"] = int(payload.get("candidate_rank_by_planner", 999))
        metadata["original_candidate_id"] = payload.get("candidate_id")
        metadata["rank_randomizer_mode"] = mode
        metadata["remap_candidate_ids"] = remap_candidate_ids
        payload["metadata"] = metadata
        payload["candidate_rank_by_planner"] = new_rank
        if remap_candidate_ids:
            payload["candidate_id"] = f"cand_{new_rank:02d}"
        reranked.append(payload)
    return sorted(reranked, key=lambda row: int(row["candidate_rank_by_planner"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["random", "prefer_success", "prefer_failure", "failure_rank0_shuffle_rest"],
        default="random",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--remap-candidate-ids", action="store_true")
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(args.manifest):
        grouped.setdefault(str(row["case_id"]), []).append(row)

    rows = []
    for case_offset, case_id in enumerate(sorted(grouped)):
        rows.extend(
            randomize_case_ranks(
                grouped[case_id],
                seed=args.seed + case_offset,
                mode=args.mode,
                remap_candidate_ids=args.remap_candidate_ids,
            )
        )
    write_jsonl(args.output, rows)

    randomized_grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        randomized_grouped.setdefault(str(row["case_id"]), []).append(row)
    rank0_success = 0
    for case_rows in randomized_grouped.values():
        rank0 = min(case_rows, key=lambda row: int(row["candidate_rank_by_planner"]))
        rank0_success += int(bool(rank0.get("oracle_success")))
    success_rank_hist: dict[str, int] = {}
    for row in rows:
        if bool(row.get("oracle_success")):
            rank = str(int(row.get("candidate_rank_by_planner", 999)))
            success_rank_hist[rank] = success_rank_hist.get(rank, 0) + 1
    print(
        json.dumps(
            {
                "cases": len(grouped),
                "mode": args.mode,
                "rank0_success": rank0_success,
                "remap_candidate_ids": args.remap_candidate_ids,
                "success_rank_hist": success_rank_hist,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
