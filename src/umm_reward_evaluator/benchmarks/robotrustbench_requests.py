"""Convert RoboTrustBench prompt metadata into generation/evaluation requests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXECUTE_SCENARIOS = {"normal", "constraint-sensitive", "constraint_sensitive"}
SUPPRESS_SCENARIOS = {"counterfactual", "adversarial"}


def _load_records(path: Path) -> list[dict[str, Any]]:
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


def _sample_id(record: dict[str, Any], fallback_index: int) -> str:
    for key in ("sample_id", "case_id", "prompt_id"):
        if key in record:
            return str(record[key])
    return f"robotrustbench_{fallback_index:06d}"


def _scenario(record: dict[str, Any]) -> str:
    return str(record.get("category", record.get("scenario", "unknown")))


def _expected_behavior(scenario: str) -> str:
    normalized = scenario.strip().lower().replace("_", "-")
    if normalized in EXECUTE_SCENARIOS:
        return "execute_instruction_reliably"
    if normalized in SUPPRESS_SCENARIOS:
        return "avoid_unsafe_or_infeasible_execution"
    return "judge_instruction_scene_consistency"


def convert_records(
    records: list[dict[str, Any]],
    *,
    default_suite: str = "robotrustbench_prompt_subset",
    image_root: str = "",
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        sample_id = _sample_id(record, idx)
        scenario = _scenario(record)
        file_name = str(record.get("file_name", record.get("image_source", "")))
        instruction = str(record.get("language_instruction", record.get("instruction", record.get("prompt", ""))))
        image_path = str(Path(image_root) / file_name) if image_root and file_name else file_name
        requests.append(
            {
                "benchmark": "robotrustbench",
                "suite": str(record.get("suite", default_suite)),
                "sample_id": sample_id,
                "case_id": sample_id,
                "task_name": scenario,
                "scenario": scenario,
                "subcategory": str(record.get("subcategory", record.get("failure_category", ""))),
                "instruction": instruction,
                "initial_image_path": image_path,
                "source_json_path": str(record.get("json_file", "")),
                "rank": int(record.get("rank", idx + 1)),
                "expected_behavior": _expected_behavior(scenario),
                "verification_target": "trustworthiness",
                "metadata": {
                    "category": scenario,
                    "scenario": scenario,
                    "subcategory": str(record.get("subcategory", "")),
                    "failure_category": str(record.get("subcategory", "")),
                    "file_name": file_name,
                    "json_file": str(record.get("json_file", "")),
                    "image_source": str(record.get("image_source", "")),
                },
            }
        )
    return requests


def summarize_requests(requests: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_counts = Counter(str(row["scenario"]) for row in requests)
    behavior_counts = Counter(str(row["expected_behavior"]) for row in requests)
    subcategory_counts = Counter(str(row["subcategory"]) for row in requests if row.get("subcategory"))
    return {
        "benchmark": "robotrustbench",
        "requests": len(requests),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "expected_behavior_counts": dict(sorted(behavior_counts.items())),
        "subcategory_counts": dict(sorted(subcategory_counts.items())),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-requests", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path)
    parser.add_argument("--default-suite", default="robotrustbench_prompt_subset")
    parser.add_argument("--image-root", default="")
    args = parser.parse_args()

    requests = convert_records(
        _load_records(args.metadata),
        default_suite=args.default_suite,
        image_root=args.image_root,
    )
    write_jsonl(args.output_requests, requests)
    summary = summarize_requests(requests)
    summary["output_requests"] = str(args.output_requests)
    summary_path = args.output_summary or args.output_requests.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
