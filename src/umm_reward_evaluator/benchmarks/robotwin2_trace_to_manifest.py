"""Convert RoboTwin 2.0 candidate traces into executable-future manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.common import annotate_oracle_best, summarize_headroom, write_jsonl


def _load_records(input_path: Path | None, input_dir: Path | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if input_path is not None:
        with input_path.open("r", encoding="utf-8") as f:
            if input_path.suffix == ".json":
                payload = json.load(f)
                if isinstance(payload, list):
                    records.extend(payload)
                else:
                    records.append(payload)
            else:
                records.extend(json.loads(line) for line in f if line.strip())
    if input_dir is not None:
        for path in sorted(input_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                records.extend(payload)
            else:
                records.append(payload)
        for path in sorted(input_dir.glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as f:
                records.extend(json.loads(line) for line in f if line.strip())
    return records


def _as_actions(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("actions must be a list")
    if not value:
        return []
    if all(isinstance(item, (int, float)) for item in value):
        return [[float(item) for item in value]]
    actions: list[list[float]] = []
    for step in value:
        if not isinstance(step, list):
            raise ValueError("each action step must be a list")
        actions.append([float(item) for item in step])
    return actions


def _success(record: dict[str, Any]) -> bool:
    for key in ("oracle_success", "success", "eval_success"):
        if key in record:
            return bool(record[key])
    raise ValueError("record missing success/oracle_success/eval_success")


def _case_id(record: dict[str, Any]) -> str:
    if "case_id" in record:
        return str(record["case_id"])
    if "seed" in record:
        instruction_id = record.get("instruction_id", record.get("instruction", ""))
        return f"seed={record['seed']}|instruction={instruction_id}"
    raise ValueError("record missing case_id or seed")


def _candidate_id(record: dict[str, Any], fallback_index: int) -> str:
    if "candidate_id" in record:
        return str(record["candidate_id"])
    parts = [
        str(record.get("policy_name", "policy")),
        str(record.get("ckpt_setting", record.get("checkpoint", "ckpt"))),
        str(record.get("candidate_seed", record.get("model_seed", fallback_index))),
    ]
    return ":".join(parts)


def filter_records_by_case_size(
    records: list[dict[str, Any]],
    *,
    required_candidates_per_case: int | None,
    drop_cases_with_candidate_error: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if required_candidates_per_case is None and not drop_cases_with_candidate_error:
        return records, []

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["task_name"]), _case_id(record))].append(record)

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for (task_name, case_id), case_records in sorted(grouped.items()):
        count = len(case_records)
        error_records = [
            record
            for record in case_records
            if record.get("candidate_error") or (record.get("metadata") or {}).get("candidate_error")
        ]
        wrong_case_size = required_candidates_per_case is not None and count != required_candidates_per_case
        has_candidate_error = drop_cases_with_candidate_error and bool(error_records)
        if not wrong_case_size and not has_candidate_error:
            kept.extend(case_records)
        else:
            drop_reasons = []
            if wrong_case_size:
                drop_reasons.append("candidate_count_mismatch")
            if has_candidate_error:
                drop_reasons.append("candidate_error")
            dropped_case = {
                "task_name": task_name,
                "case_id": case_id,
                "candidate_count": count,
                "required_candidates_per_case": required_candidates_per_case,
                "drop_reasons": drop_reasons,
            }
            if error_records:
                dropped_case["candidate_error_count"] = len(error_records)
                dropped_case["candidate_error_candidate_ids"] = [
                    _candidate_id(record, idx) for idx, record in enumerate(error_records)
                ]
            dropped.append(dropped_case)
    return kept, dropped


def convert_records(records: list[dict[str, Any]], *, default_suite: str) -> list[dict[str, Any]]:
    grouped_order: Counter[tuple[str, str]] = Counter()
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        task_name = str(record["task_name"])
        case_id = _case_id(record)
        case_key = (task_name, case_id)
        if "candidate_rank_by_planner" in record:
            rank = int(record["candidate_rank_by_planner"])
        else:
            rank = grouped_order[case_key]
        grouped_order[case_key] += 1

        metadata = dict(record.get("metadata") or {})
        metadata.setdefault("future_source", record.get("future_source", "policy_rollout"))
        metadata.setdefault("future_representation", record.get("future_representation", "actions"))
        metadata.setdefault("verification_target", record.get("verification_target", "task_success"))
        for key in (
            "policy_name",
            "ckpt_setting",
            "checkpoint",
            "seed",
            "candidate_seed",
            "model_seed",
            "action_type",
            "task_config",
            "domain_randomization",
            "trace_path",
            "state_trace",
        ):
            if key in record and key not in metadata:
                metadata[key] = record[key]

        rows.append(
            {
                "benchmark": "robotwin2",
                "suite": str(record.get("suite", record.get("task_config", default_suite))),
                "task_name": task_name,
                "case_id": case_id,
                "candidate_id": _candidate_id(record, idx),
                "candidate_rank_by_planner": rank,
                "rollout_video_path": str(record.get("rollout_video_path", record.get("video_path", ""))),
                "rollout_video_layout": str(record.get("rollout_video_layout", "head_camera_rollout")),
                "actions": _as_actions(record.get("actions")),
                "oracle_success": _success(record),
                "instruction": record.get("instruction"),
                "planner_score": record.get("planner_score"),
                "oracle_return": record.get("oracle_return"),
                "oracle_progress": record.get("oracle_progress"),
                "oracle_state_dist": record.get("oracle_state_dist"),
                "metadata": metadata,
            }
        )
    return annotate_oracle_best(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path)
    parser.add_argument("--default-suite", default="demo_randomized")
    parser.add_argument(
        "--require-candidates-per-case",
        type=int,
        help="Drop entire cases unless they contain exactly this many candidates.",
    )
    parser.add_argument(
        "--drop-cases-with-candidate-error",
        action="store_true",
        help="Drop entire cases containing system-error candidates, such as CUDA OOM rows.",
    )
    args = parser.parse_args()

    if args.input is None and args.input_dir is None:
        raise SystemExit("provide --input and/or --input-dir")

    records = _load_records(args.input, args.input_dir)
    filtered_records, dropped_cases = filter_records_by_case_size(
        records,
        required_candidates_per_case=args.require_candidates_per_case,
        drop_cases_with_candidate_error=args.drop_cases_with_candidate_error,
    )
    rows = convert_records(filtered_records, default_suite=args.default_suite)
    write_jsonl(args.output_manifest, rows)

    summary = summarize_headroom(rows)
    summary.update(
        {
            "benchmark": "robotwin2",
            "input_records": len(records),
            "filtered_input_records": len(filtered_records),
            "dropped_cases": dropped_cases,
            "drop_cases_with_candidate_error": args.drop_cases_with_candidate_error,
            "required_candidates_per_case": args.require_candidates_per_case,
            "output_manifest": str(args.output_manifest),
        }
    )
    summary_path = args.output_summary or args.output_manifest.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
