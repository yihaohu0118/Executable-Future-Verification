"""Shuffle candidate rows within each case for order-leakage controls."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from umm_reward_evaluator.benchmarks.common import load_jsonl, write_jsonl


def shuffle_rows(rows: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(row)

    shuffled: list[dict] = []
    for case_id in sorted(by_case):
        items = [dict(row) for row in by_case[case_id]]
        order = rng.permutation(len(items))
        shuffled.extend([items[int(i)] for i in order])
    return shuffled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    write_jsonl(args.output, shuffle_rows(rows, args.seed))
    print(f"[shuffle_manifest_rows] wrote {args.output} rows={len(rows)} seed={args.seed}")


if __name__ == "__main__":
    main()

