"""Filter candidate-selection manifests and recompute oracle annotations."""

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


def metadata_value(row: dict[str, Any], key: str) -> Any:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return None
    return metadata.get(key)


def compact_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), []).append(dict(row))

    compacted: list[dict[str, Any]] = []
    for case_rows in grouped.values():
        ordered = sorted(case_rows, key=lambda row: int(row.get("candidate_rank_by_planner", 999999)))
        for new_rank, row in enumerate(ordered):
            row["candidate_rank_by_planner"] = new_rank
            compacted.append(row)
    return compacted


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    exclude_metadata_key: str | None,
    exclude_metadata_value: str | None,
    exclude_candidate_ids: set[str] | None = None,
    preserve_ranks: bool,
) -> list[dict[str, Any]]:
    excluded_ids = exclude_candidate_ids or set()
    filtered = []
    for row in rows:
        if str(row.get("candidate_id")) in excluded_ids:
            continue
        if exclude_metadata_key is not None and exclude_metadata_value is not None:
            if str(metadata_value(row, exclude_metadata_key)) == exclude_metadata_value:
                continue
        filtered.append(dict(row))
    if not preserve_ranks:
        filtered = compact_ranks(filtered)
    return annotate_oracle_best(filtered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--exclude-metadata-key", default="profile")
    parser.add_argument("--exclude-metadata-value", default="original")
    parser.add_argument("--exclude-candidate-id", action="append", default=[])
    parser.add_argument("--preserve-ranks", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    exclude_metadata_key = args.exclude_metadata_key
    exclude_metadata_value = args.exclude_metadata_value
    if args.exclude_metadata_key == "" or args.exclude_metadata_value == "":
        exclude_metadata_key = None
        exclude_metadata_value = None
    filtered = filter_rows(
        rows,
        exclude_metadata_key=exclude_metadata_key,
        exclude_metadata_value=exclude_metadata_value,
        exclude_candidate_ids=set(args.exclude_candidate_id),
        preserve_ranks=args.preserve_ranks,
    )
    write_jsonl(args.output, filtered)

    summary = summarize_headroom(filtered)
    summary.update(
        {
            "source_manifest": str(args.manifest),
            "output_manifest": str(args.output),
            "excluded_metadata_key": args.exclude_metadata_key,
            "excluded_metadata_value": args.exclude_metadata_value,
            "excluded_candidate_ids": sorted(set(args.exclude_candidate_id)),
            "input_rows": len(rows),
            "output_rows": len(filtered),
            "preserve_ranks": args.preserve_ranks,
        }
    )
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"manifest={args.output}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
