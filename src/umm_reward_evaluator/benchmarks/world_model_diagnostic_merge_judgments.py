"""Normalize public world-model benchmark judgment rows into EFV records."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.common import summarize_headroom, write_jsonl
from umm_reward_evaluator.benchmarks.world_model_diagnostic_to_manifest import convert_records


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        if path.suffix == ".json":
            payload = json.load(f)
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict) and isinstance(payload.get("records"), list):
                return payload["records"]
            if isinstance(payload, dict):
                return [payload]
            raise ValueError(f"unsupported JSON payload in {path}")
        return [json.loads(line) for line in f if line.strip()]


def _nested_get(payload: dict[str, Any], key: str) -> Any:
    value: Any = payload
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _first(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = _nested_get(row, key) if "." in key else row.get(key)
        if value is not None:
            return value
    return None


def _case_id(row: dict[str, Any], fallback_index: int) -> str:
    value = _first(row, ("case_id", "sample_id", "prompt_id", "episode_id", "initial_state_id", "id"))
    return str(value) if value is not None else f"case_{fallback_index:06d}"


def _candidate_id(row: dict[str, Any], fallback_index: int) -> str:
    value = _first(row, ("candidate_id", "generation_id", "video_id", "sample_video_id", "model_name", "video_model_name"))
    return str(value) if value is not None else f"candidate_{fallback_index:06d}"


def _request_index(requests: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(requests):
        indexed[_case_id(row, idx)] = row
    return indexed


def _metadata(request: dict[str, Any], judgment: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(request.get("metadata") or {})
    metadata.update(dict(judgment.get("metadata") or {}))
    for source in (request, judgment):
        for key in (
            "scenario",
            "category",
            "subcategory",
            "failure_category",
            "criterion",
            "dimension",
            "model_name",
            "video_model_name",
            "generation_seed",
            "decoding_seed",
            "visual_quality_score",
            "motion_consistency",
            "action_following_score",
            "physics_score",
            "constraint_score",
            "trust_score",
            "mllm_score",
            "human_score",
            "label",
            "judgment",
            "verdict",
            "file_name",
            "json_file",
            "image_source",
            "initial_image_path",
            "reference_video_path",
        ):
            if key in source and key not in metadata:
                metadata[key] = source[key]
    if "scenario" not in metadata and "category" in metadata:
        metadata["scenario"] = metadata["category"]
    if "failure_category" not in metadata and "subcategory" in metadata:
        metadata["failure_category"] = metadata["subcategory"]
    return metadata


def normalize_judgments(
    judgments: list[dict[str, Any]],
    *,
    requests: list[dict[str, Any]] | None = None,
    default_benchmark: str,
    default_suite: str,
    default_verification_target: str,
) -> list[dict[str, Any]]:
    request_by_case = _request_index(requests or [])
    rank_counter: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for idx, judgment in enumerate(judgments):
        case_id = _case_id(judgment, idx)
        request = request_by_case.get(case_id, {})
        rank = _first(judgment, ("candidate_rank_by_planner", "candidate_rank", "rank"))
        if rank is None:
            rank = rank_counter[case_id]
        rank_counter[case_id] += 1
        metadata = _metadata(request, judgment)
        metadata.setdefault("future_source", judgment.get("future_source", "world_model_video"))
        metadata.setdefault("future_representation", judgment.get("future_representation", "rgb_video"))
        metadata.setdefault(
            "verification_target",
            judgment.get("verification_target", request.get("verification_target", default_verification_target)),
        )

        task_name = _first(judgment, ("task_name", "task", "category", "scenario"))
        if task_name is None:
            task_name = _first(request, ("task_name", "scenario", "category", "task"))
        planner_score = _first(judgment, ("planner_score", "model_score", "visual_score", "visual_quality_score"))
        records.append(
            {
                "benchmark": str(judgment.get("benchmark", request.get("benchmark", default_benchmark))),
                "suite": str(judgment.get("suite", request.get("suite", default_suite))),
                "task_name": str(task_name or "unknown_task"),
                "case_id": case_id,
                "candidate_id": _candidate_id(judgment, idx),
                "rank": int(rank),
                "video_path": str(_first(judgment, ("video_path", "rollout_video_path", "generated_video_path")) or ""),
                "actions": judgment.get("actions", judgment.get("action_vector", [])),
                "instruction": _first(judgment, ("instruction", "prompt", "language_instruction"))
                or _first(request, ("instruction", "prompt", "language_instruction")),
                "planner_score": planner_score,
                "model_score": planner_score,
                "oracle_success": judgment.get("oracle_success", judgment.get("success")),
                "label": judgment.get("label"),
                "judgment": judgment.get("judgment"),
                "verdict": judgment.get("verdict"),
                "score": judgment.get("score"),
                "metadata": metadata,
            }
        )
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scenario_counts: Counter[str] = Counter()
    for row in records:
        grouped[str(row["case_id"])].append(row)
        metadata = row.get("metadata") or {}
        if metadata.get("scenario") is not None:
            scenario_counts[str(metadata["scenario"])] += 1
    return {
        "records": len(records),
        "cases": len(grouped),
        "candidate_count_histogram": dict(sorted(Counter(len(rows) for rows in grouped.values()).items())),
        "scenario_counts": dict(sorted(scenario_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--judgments", type=Path, action="append", required=True)
    parser.add_argument("--output-records", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--output-summary", type=Path)
    parser.add_argument("--default-benchmark", default="world_model_diagnostic")
    parser.add_argument("--default-suite", default="diagnostic")
    parser.add_argument("--default-verification-target", default="action_conditioned_reliability")
    parser.add_argument("--score-key")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()

    if args.output_records is None and args.output_manifest is None:
        raise SystemExit("provide --output-records and/or --output-manifest")
    if (args.score_key is None) != (args.threshold is None):
        raise SystemExit("--score-key and --threshold must be provided together")

    requests = _load_rows(args.requests) if args.requests else []
    judgments = [row for path in args.judgments for row in _load_rows(path)]
    records = normalize_judgments(
        judgments,
        requests=requests,
        default_benchmark=args.default_benchmark,
        default_suite=args.default_suite,
        default_verification_target=args.default_verification_target,
    )
    if args.output_records:
        write_jsonl(args.output_records, records)

    summary = summarize_records(records)
    if args.output_manifest:
        manifest = convert_records(
            records,
            default_benchmark=args.default_benchmark,
            default_suite=args.default_suite,
            default_verification_target=args.default_verification_target,
            score_key=args.score_key,
            threshold=args.threshold,
        )
        write_jsonl(args.output_manifest, manifest)
        summary.update({"manifest": str(args.output_manifest), **summarize_headroom(manifest)})
    if args.output_summary:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
