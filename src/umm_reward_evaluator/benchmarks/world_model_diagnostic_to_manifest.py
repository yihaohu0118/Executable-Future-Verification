"""Convert world-model diagnostic judgments into executable-future manifests.

This adapter is intentionally format-light. Benchmarks such as MiraBench and
RoboTrustBench may release human/MLLM judgments over generated videos rather
than simulator action traces. We still map them into the shared manifest so the
same selector/evaluation code can report rank0, oracle, and calibrated verifier
metrics over diagnostic targets such as action fidelity or trustworthiness.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.common import annotate_oracle_best, summarize_headroom, write_jsonl


def _load_records(input_path: Path | None, input_dir: Path | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if input_path is not None:
        records.extend(_load_one(input_path))
    if input_dir is not None:
        for path in sorted(input_dir.glob("*.json")) + sorted(input_dir.glob("*.jsonl")):
            records.extend(_load_one(path))
    return records


def _load_one(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        if path.suffix == ".json":
            payload = json.load(f)
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict):
                if isinstance(payload.get("records"), list):
                    return payload["records"]
                return [payload]
            raise ValueError(f"unsupported JSON payload in {path}")
        return [json.loads(line) for line in f if line.strip()]


def _nested_get(record: dict[str, Any], key: str) -> Any:
    value: Any = record
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _as_actions(record: dict[str, Any]) -> list[list[float]]:
    actions = record.get("actions", record.get("action_vector"))
    if actions is None:
        return []
    if not isinstance(actions, list):
        raise ValueError("actions/action_vector must be a list when present")
    if not actions:
        return []
    if all(isinstance(item, (int, float)) for item in actions):
        return [[float(item) for item in actions]]
    out: list[list[float]] = []
    for step in actions:
        if not isinstance(step, list):
            raise ValueError("each action step must be a list")
        out.append([float(item) for item in step])
    return out


def _success(record: dict[str, Any], *, score_key: str | None, threshold: float | None) -> bool:
    for key in ("oracle_success", "success", "pass", "is_valid", "is_reliable", "is_trustworthy"):
        if key in record and record[key] is not None:
            return bool(record[key])
    label = record.get("label", record.get("judgment", record.get("verdict")))
    if isinstance(label, str):
        return label.strip().lower() in {
            "pass",
            "success",
            "valid",
            "reliable",
            "trustworthy",
            "safe",
            "physically_plausible",
            "action_following",
        }
    if score_key is not None and threshold is not None:
        score = _nested_get(record, score_key)
        if score is None:
            raise ValueError(f"record missing score key {score_key!r}")
        return float(score) >= threshold
    raise ValueError("record missing success/pass label or score threshold fields")


def _case_id(record: dict[str, Any]) -> str:
    for key in ("case_id", "sample_id", "prompt_id", "episode_id", "initial_state_id"):
        if key in record:
            return str(record[key])
    task = record.get("task_name", record.get("task", "task"))
    instruction = record.get("instruction", record.get("prompt", ""))
    return f"{task}|{instruction}"


def _candidate_id(record: dict[str, Any], fallback_index: int) -> str:
    for key in ("candidate_id", "generation_id", "video_id", "sample_video_id"):
        if key in record:
            return str(record[key])
    model = record.get("model_name", record.get("video_model_name", "model"))
    return f"{model}:{fallback_index}"


def convert_records(
    records: list[dict[str, Any]],
    *,
    default_benchmark: str,
    default_suite: str,
    default_verification_target: str,
    score_key: str | None,
    threshold: float | None,
) -> list[dict[str, Any]]:
    grouped_order: Counter[tuple[str, str]] = Counter()
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        task_name = str(record.get("task_name", record.get("task", record.get("category", "unknown_task"))))
        case_id = _case_id(record)
        case_key = (task_name, case_id)
        if "candidate_rank_by_planner" in record:
            rank = int(record["candidate_rank_by_planner"])
        elif "rank" in record:
            rank = int(record["rank"])
        else:
            rank = grouped_order[case_key]
        grouped_order[case_key] += 1

        metadata = dict(record.get("metadata") or {})
        metadata.setdefault("future_source", record.get("future_source", "world_model_video"))
        metadata.setdefault("future_representation", record.get("future_representation", "rgb_video"))
        metadata.setdefault(
            "verification_target",
            record.get("verification_target", default_verification_target),
        )
        for key in (
            "model_name",
            "video_model_name",
            "action_text",
            "action_condition",
            "scenario",
            "category",
            "subcategory",
            "failure_category",
            "criterion",
            "dimension",
            "human_score",
            "mllm_score",
            "label",
            "judgment",
            "verdict",
            "initial_image_path",
            "reference_video_path",
            "file_name",
            "json_file",
            "image_source",
            "sample_id",
        ):
            if key in record and key not in metadata:
                metadata[key] = record[key]
        if "scenario" not in metadata and "category" in metadata:
            metadata["scenario"] = metadata["category"]
        if "failure_category" not in metadata and "subcategory" in metadata:
            metadata["failure_category"] = metadata["subcategory"]

        rows.append(
            {
                "benchmark": str(record.get("benchmark", default_benchmark)),
                "suite": str(record.get("suite", record.get("split", default_suite))),
                "task_name": task_name,
                "case_id": case_id,
                "candidate_id": _candidate_id(record, idx),
                "candidate_rank_by_planner": rank,
                "rollout_video_path": str(record.get("rollout_video_path", record.get("video_path", ""))),
                "rollout_video_layout": str(record.get("rollout_video_layout", "single_video")),
                "actions": _as_actions(record),
                "oracle_success": _success(record, score_key=score_key, threshold=threshold),
                "instruction": record.get("instruction", record.get("prompt", record.get("language_instruction"))),
                "planner_score": record.get("planner_score", record.get("model_score")),
                "oracle_return": record.get("oracle_return", record.get("score")),
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
    parser.add_argument("--default-benchmark", default="world_model_diagnostic")
    parser.add_argument("--default-suite", default="diagnostic")
    parser.add_argument("--default-verification-target", default="action_conditioned_reliability")
    parser.add_argument("--score-key", help="Nested score key, e.g. scores.action_following.")
    parser.add_argument("--threshold", type=float, help="Score threshold for oracle_success.")
    args = parser.parse_args()

    if args.input is None and args.input_dir is None:
        raise SystemExit("provide --input and/or --input-dir")
    if (args.score_key is None) != (args.threshold is None):
        raise SystemExit("--score-key and --threshold must be provided together")

    records = _load_records(args.input, args.input_dir)
    rows = convert_records(
        records,
        default_benchmark=args.default_benchmark,
        default_suite=args.default_suite,
        default_verification_target=args.default_verification_target,
        score_key=args.score_key,
        threshold=args.threshold,
    )
    write_jsonl(args.output_manifest, rows)

    summary = summarize_headroom(rows)
    summary.update(
        {
            "benchmark": args.default_benchmark,
            "input_records": len(records),
            "output_manifest": str(args.output_manifest),
            "diagnostic_layer": True,
        }
    )
    summary_path = args.output_summary or args.output_manifest.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
