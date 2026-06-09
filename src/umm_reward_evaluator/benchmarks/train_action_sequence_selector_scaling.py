"""Measure action-sequence selector sensitivity to the number of training cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from umm_reward_evaluator.benchmarks.common import load_jsonl, oracle_key
from umm_reward_evaluator.benchmarks.train_action_sequence_selector import ACTION_FEATURE_MODES, train_fold


def group_by_case(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), []).append(row)
    return grouped


def parse_train_counts(raw: str, *, max_cases: int) -> list[int]:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item == "all":
            out.append(max_cases)
        else:
            out.append(int(item))
    return sorted(set(min(max(count, 1), max_cases) for count in out))


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = group_by_case(scored)
    rank0_success = 0
    selector_success = 0
    oracle_success = 0
    selector_oracle_match = 0
    recovered_rank0_fail = 0
    rank0_failures = 0
    for case_rows in grouped.values():
        rank0 = min(case_rows, key=lambda row: row["candidate_rank_by_planner"])
        selector = max(
            case_rows,
            key=lambda row: (
                row["action_sequence_selector_score"],
                -int(row.get("candidate_rank_by_planner", 999)),
            ),
        )
        oracle = max(case_rows, key=oracle_key)
        rank0_success += int(bool(rank0["oracle_success"]))
        selector_success += int(bool(selector["oracle_success"]))
        oracle_success += int(bool(oracle["oracle_success"]))
        selector_oracle_match += int(selector["candidate_id"] == oracle["candidate_id"])
        if not rank0["oracle_success"]:
            rank0_failures += 1
            recovered_rank0_fail += int(bool(selector["oracle_success"]))
    cases = len(grouped)
    return {
        "cases": cases,
        "rank0_success": rank0_success,
        "selector_success": selector_success,
        "oracle_success": oracle_success,
        "selector_oracle_match": selector_oracle_match,
        "rank0_failures": rank0_failures,
        "recovered_rank0_fail": recovered_rank0_fail,
        "rank0_success_rate": rank0_success / cases if cases else 0.0,
        "selector_success_rate": selector_success / cases if cases else 0.0,
        "oracle_success_rate": oracle_success / cases if cases else 0.0,
    }


def evaluate_count(
    rows: list[dict[str, Any]],
    *,
    train_case_count: int,
    repeat: int,
    feature_mode: str,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cases = group_by_case(rows)
    case_ids = sorted(cases)
    scored: list[dict[str, Any]] = []
    for fold, case_id in enumerate(case_ids):
        available = [other_id for other_id in case_ids if other_id != case_id]
        rng = np.random.default_rng(seed + repeat * 100_003 + fold)
        if train_case_count < len(available):
            train_case_ids = sorted(rng.choice(available, size=train_case_count, replace=False).tolist())
        else:
            train_case_ids = available
        train_rows = [row for train_case_id in train_case_ids for row in cases[train_case_id]]
        scored.extend(
            train_fold(
                train_rows,
                cases[case_id],
                feature_mode=feature_mode,
                hidden=hidden,
                epochs=epochs,
                lr=lr,
                seed=seed + repeat * 10_007 + fold,
            )
        )
    summary = summarize(scored)
    summary["train_case_count"] = train_case_count
    summary["repeat"] = repeat
    return summary, scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-case-counts", default="4,8,16,32,all")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--feature-mode", default="raw_no_length", choices=list(ACTION_FEATURE_MODES))
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    max_train_cases = max(len(group_by_case(rows)) - 1, 1)
    train_counts = parse_train_counts(args.train_case_counts, max_cases=max_train_cases)
    summaries = []
    all_scored = []
    for repeat in range(args.repeats):
        for train_case_count in train_counts:
            summary, scored = evaluate_count(
                rows,
                train_case_count=train_case_count,
                repeat=repeat,
                feature_mode=args.feature_mode,
                hidden=args.hidden,
                epochs=args.epochs,
                lr=args.lr,
                seed=args.seed,
            )
            summaries.append(summary)
            for row in scored:
                payload = dict(row)
                payload["scaling_repeat"] = repeat
                payload["scaling_train_case_count"] = train_case_count
                all_scored.append(payload)
            print(json.dumps(summary, sort_keys=True))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "scored_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in all_scored:
            f.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
