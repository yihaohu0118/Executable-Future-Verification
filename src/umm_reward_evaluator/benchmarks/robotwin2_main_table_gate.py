"""Readiness gate for RoboTwin2 main-table candidate manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.common import load_jsonl, oracle_key
from umm_reward_evaluator.benchmarks.robotwin2_selector_baselines import feature_coverage
from umm_reward_evaluator.benchmarks.validate_future_verification_manifest import validate_rows


def _case_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["task_name"]), str(row["case_id"])


def _has_candidate_error(row: dict[str, Any]) -> bool:
    return bool(row.get("candidate_error") or (row.get("metadata") or {}).get("candidate_error"))


def _check(name: str, passed: bool, detail: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _case_summaries(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_case_key(row)].append(row)

    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    for key, case_rows in grouped.items():
        rank0 = min(case_rows, key=lambda row: int(row.get("candidate_rank_by_planner", 999999)))
        oracle = max(case_rows, key=oracle_key)
        summaries[key] = {
            "candidate_count": len(case_rows),
            "rank0_success": bool(rank0.get("oracle_success")),
            "oracle_success": bool(oracle.get("oracle_success")),
            "oracle_better": oracle_key(oracle) > oracle_key(rank0),
            "candidate_error_count": sum(1 for row in case_rows if _has_candidate_error(row)),
        }
    return summaries


def evaluate_gate(
    rows: list[dict[str, Any]],
    *,
    required_candidates_per_case: int | None = None,
    min_cases: int = 1,
    min_oracle_better_cases: int = 1,
    require_no_candidate_error: bool = True,
    require_future_metadata: bool = True,
    required_feature_modes: list[str] | None = None,
    min_feature_case_coverage: float = 1.0,
) -> dict[str, Any]:
    schema_summary, schema_errors = validate_rows(rows, require_metadata=require_future_metadata)
    case_summaries = _case_summaries(rows)
    candidate_counts = Counter(summary["candidate_count"] for summary in case_summaries.values())
    candidate_error_rows = [row for row in rows if _has_candidate_error(row)]
    candidate_error_cases = [
        {"task_name": task, "case_id": case_id, "candidate_error_count": summary["candidate_error_count"]}
        for (task, case_id), summary in sorted(case_summaries.items())
        if summary["candidate_error_count"]
    ]

    rank0_success = sum(int(summary["rank0_success"]) for summary in case_summaries.values())
    oracle_success = sum(int(summary["oracle_success"]) for summary in case_summaries.values())
    oracle_better = sum(int(summary["oracle_better"]) for summary in case_summaries.values())

    checks = [
        _check(
            "schema",
            not schema_errors,
            {"num_errors": len(schema_errors), "errors": schema_errors[:20]},
        ),
        _check(
            "min_cases",
            len(case_summaries) >= min_cases,
            {"cases": len(case_summaries), "min_cases": min_cases},
        ),
        _check(
            "oracle_headroom",
            oracle_better >= min_oracle_better_cases,
            {
                "oracle_better": oracle_better,
                "min_oracle_better_cases": min_oracle_better_cases,
                "rank0_success": rank0_success,
                "oracle_success": oracle_success,
            },
        ),
    ]
    if required_candidates_per_case is not None:
        bad_cases = [
            {"task_name": task, "case_id": case_id, "candidate_count": summary["candidate_count"]}
            for (task, case_id), summary in sorted(case_summaries.items())
            if summary["candidate_count"] != required_candidates_per_case
        ]
        checks.append(
            _check(
                "candidate_count",
                not bad_cases,
                {
                    "required_candidates_per_case": required_candidates_per_case,
                    "candidate_count_histogram": dict(sorted(candidate_counts.items())),
                    "bad_cases": bad_cases[:50],
                    "num_bad_cases": len(bad_cases),
                },
            )
        )
    if require_no_candidate_error:
        checks.append(
            _check(
                "candidate_error_free",
                not candidate_error_rows,
                {
                    "candidate_error_rows": len(candidate_error_rows),
                    "candidate_error_cases": candidate_error_cases[:50],
                    "num_candidate_error_cases": len(candidate_error_cases),
                },
            )
        )

    feature_coverages = []
    for feature_mode in required_feature_modes or []:
        coverage = feature_coverage(rows, feature_mode)
        feature_coverages.append(coverage)
        checks.append(
            _check(
                f"feature_coverage:{feature_mode}",
                float(coverage["case_coverage_rate"]) >= min_feature_case_coverage,
                {
                    **coverage,
                    "min_feature_case_coverage": min_feature_case_coverage,
                },
            )
        )

    passed = all(check["passed"] for check in checks)
    return {
        "passed": passed,
        "summary": {
            "rows": len(rows),
            "cases": len(case_summaries),
            "rank0_success": rank0_success,
            "oracle_success": oracle_success,
            "oracle_better": oracle_better,
            "candidate_count_histogram": dict(sorted(candidate_counts.items())),
            "schema_summary": schema_summary,
        },
        "checks": checks,
        "feature_coverages": feature_coverages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--required-candidates-per-case", type=int)
    parser.add_argument("--min-cases", type=int, default=1)
    parser.add_argument("--min-oracle-better-cases", type=int, default=1)
    parser.add_argument("--allow-candidate-error", action="store_true")
    parser.add_argument("--no-require-future-metadata", action="store_true")
    parser.add_argument("--require-feature", action="append", default=[])
    parser.add_argument("--min-feature-case-coverage", type=float, default=1.0)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    result = evaluate_gate(
        rows,
        required_candidates_per_case=args.required_candidates_per_case,
        min_cases=args.min_cases,
        min_oracle_better_cases=args.min_oracle_better_cases,
        require_no_candidate_error=not args.allow_candidate_error,
        require_future_metadata=not args.no_require_future_metadata,
        required_feature_modes=list(args.require_feature),
        min_feature_case_coverage=args.min_feature_case_coverage,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
