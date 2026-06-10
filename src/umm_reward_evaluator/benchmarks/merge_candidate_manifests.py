"""Merge candidate-selection manifests and recompute oracle annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.common import (
    annotate_oracle_best,
    load_jsonl,
    summarize_headroom,
    write_jsonl,
)


def merge_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        for row in load_jsonl(path):
            key = (str(row["task_name"]), str(row["case_id"]), str(row["candidate_id"]))
            if key in seen:
                raise ValueError(f"duplicate manifest row key {key}")
            seen.add(key)
            rows.append(row)
    return annotate_oracle_best(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()

    rows = merge_rows(args.manifest)
    write_jsonl(args.output_manifest, rows)
    summary = summarize_headroom(rows)
    if rows:
        first = rows[0]
        summary.update(
            {
                "benchmark": first.get("benchmark"),
                "suite": first.get("suite"),
                "task_name": first.get("task_name"),
                "merged_manifests": [str(path) for path in args.manifest],
            }
        )
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
