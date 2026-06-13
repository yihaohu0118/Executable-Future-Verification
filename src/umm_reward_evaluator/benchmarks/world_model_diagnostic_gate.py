"""Readiness gate for world-model diagnostic manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.common import case_group_key, load_jsonl, oracle_key
from umm_reward_evaluator.benchmarks.validate_future_verification_manifest import validate_rows


def _nested_get(payload: dict[str, Any], key: str) -> Any:
    value: Any = payload
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[case_group_key(row)].append(row)
    return grouped


def _planner_score(row: dict[str, Any]) -> float:
    value = row.get("planner_score")
    if value is None:
        value = row.get("model_score")
    if value is None:
        return float("-inf")
    return float(value)


def _selector_success_by_score(grouped: dict[tuple[str, str], list[dict[str, Any]]]) -> tuple[int, int]:
    covered = 0
    success = 0
    for case_rows in grouped.values():
        if not case_rows or any(row.get("planner_score") is None and row.get("model_score") is None for row in case_rows):
            continue
        covered += 1
        selected = max(case_rows, key=_planner_score)
        success += int(bool(selected.get("oracle_success")))
    return success, covered


def _check(name: str, passed: bool, detail: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def evaluate_diagnostic_gate(
    rows: list[dict[str, Any]],
    *,
    min_cases: int = 16,
    min_tasks: int = 1,
    min_oracle_better_cases: int = 1,
    min_candidates_per_case: int = 2,
    category_keys: list[str] | None = None,
    min_categories: int = 1,
    required_metadata_keys: list[str] | None = None,
    require_planner_score_baseline: bool = True,
    min_planner_score_oracle_gap: int = 1,
    min_planner_score_failures: int = 1,
) -> dict[str, Any]:
    schema_summary, schema_errors = validate_rows(rows, require_metadata=True)
    grouped = _group_rows(rows)
    task_names = sorted({str(row.get("task_name", "")) for row in rows})
    bad_candidate_cases = [
        {"task_name": task, "case_id": case_id, "candidate_count": len(case_rows)}
        for (task, case_id), case_rows in sorted(grouped.items())
        if len(case_rows) < min_candidates_per_case
    ]

    rank0_success = 0
    oracle_success = 0
    oracle_better = 0
    for case_rows in grouped.values():
        rank0 = min(case_rows, key=lambda row: int(row.get("candidate_rank_by_planner", 999999)))
        oracle = max(case_rows, key=oracle_key)
        rank0_success += int(bool(rank0.get("oracle_success")))
        oracle_success += int(bool(oracle.get("oracle_success")))
        oracle_better += int(oracle_key(oracle) > oracle_key(rank0))

    categories: dict[str, dict[str, int]] = {}
    for key in category_keys or []:
        counter: Counter[str] = Counter()
        for row in rows:
            value = _nested_get(row, key)
            if value is not None:
                counter[str(value)] += 1
        categories[key] = dict(sorted(counter.items()))
    total_category_values = sorted({value for counter in categories.values() for value in counter})

    missing_metadata_rows = []
    for idx, row in enumerate(rows):
        metadata = row.get("metadata") or {}
        for key in required_metadata_keys or []:
            if _nested_get({"metadata": metadata, **row}, key) is None:
                missing_metadata_rows.append({"row": idx, "missing_key": key})
                break

    score_success, score_covered = _selector_success_by_score(grouped)
    score_failures = score_covered - score_success
    score_oracle_gap = oracle_success - score_success
    checks = [
        _check("schema", not schema_errors, {"num_errors": len(schema_errors), "errors": schema_errors[:20]}),
        _check("min_cases", len(grouped) >= min_cases, {"cases": len(grouped), "minimum": min_cases}),
        _check("min_tasks", len(task_names) >= min_tasks, {"tasks": task_names, "minimum": min_tasks}),
        _check(
            "candidate_count",
            not bad_candidate_cases,
            {
                "min_candidates_per_case": min_candidates_per_case,
                "bad_cases": bad_candidate_cases[:50],
                "num_bad_cases": len(bad_candidate_cases),
            },
        ),
        _check(
            "oracle_headroom",
            oracle_better >= min_oracle_better_cases,
            {
                "oracle_better": oracle_better,
                "minimum": min_oracle_better_cases,
                "rank0_success": rank0_success,
                "oracle_success": oracle_success,
            },
        ),
        _check(
            "category_coverage",
            len(total_category_values) >= min_categories,
            {"categories": categories, "num_category_values": len(total_category_values), "minimum": min_categories},
        ),
        _check(
            "required_metadata",
            not missing_metadata_rows,
            {
                "required_metadata_keys": required_metadata_keys or [],
                "missing_rows": missing_metadata_rows[:50],
                "num_missing_rows": len(missing_metadata_rows),
            },
        ),
    ]
    if require_planner_score_baseline:
        checks.append(
            _check(
                "planner_score_baseline",
                score_covered == len(grouped) and score_covered > 0,
                {
                    "covered_cases": score_covered,
                    "cases": len(grouped),
                    "planner_score_success": score_success,
                },
            )
        )
        checks.append(
            _check(
                "planner_score_proxy_gap",
                score_covered == len(grouped)
                and score_oracle_gap >= min_planner_score_oracle_gap
                and score_failures >= min_planner_score_failures,
                {
                    "planner_score_success": score_success,
                    "planner_score_failures": score_failures,
                    "oracle_success": oracle_success,
                    "oracle_minus_planner_score": score_oracle_gap,
                    "min_oracle_gap": min_planner_score_oracle_gap,
                    "min_planner_score_failures": min_planner_score_failures,
                },
            )
        )

    return {
        "passed": all(check["passed"] for check in checks),
        "summary": {
            "rows": len(rows),
            "cases": len(grouped),
            "tasks": task_names,
            "rank0_success": rank0_success,
            "oracle_success": oracle_success,
            "oracle_better": oracle_better,
            "planner_score_success": score_success,
            "planner_score_covered_cases": score_covered,
            "planner_score_failures": score_failures,
            "planner_score_oracle_gap": score_oracle_gap,
            "schema_summary": schema_summary,
            "categories": categories,
        },
        "checks": checks,
    }


def render_markdown(result: dict[str, Any], *, title: str = "World-Model Diagnostic Gate") -> str:
    summary = result["summary"]
    lines = [
        f"# {title}",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in result["checks"]:
        detail = check["detail"]
        if check["name"] == "category_coverage":
            text = f"{detail['num_category_values']} / min {detail['minimum']}"
        elif "minimum" in detail:
            value = detail.get("cases", detail.get("tasks", detail.get("oracle_better", "-")))
            text = f"{value} / min {detail['minimum']}"
        elif check["name"] == "planner_score_baseline":
            text = f"{detail['covered_cases']}/{detail['cases']} cases covered"
        elif check["name"] == "planner_score_proxy_gap":
            text = f"gap {detail['oracle_minus_planner_score']} / min {detail['min_oracle_gap']}; failures {detail['planner_score_failures']} / min {detail['min_planner_score_failures']}"
        else:
            text = f"{detail.get('num_errors', detail.get('num_bad_cases', detail.get('num_missing_rows', '-')))}"
        lines.append(f"| `{check['name']}` | {'pass' if check['passed'] else 'fail'} | {text} |")

    lines.extend(
        [
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Rows | {summary['rows']} |",
            f"| Cases | {summary['cases']} |",
            f"| Rank0 success | {summary['rank0_success']} |",
            f"| Oracle success | {summary['oracle_success']} |",
            f"| Oracle better | {summary['oracle_better']} |",
            f"| Planner-score success | {summary['planner_score_success']} |",
            f"| Planner-score failures | {summary['planner_score_failures']} |",
            f"| Oracle minus planner-score | {summary['planner_score_oracle_gap']} |",
            "",
            "Interpretation:",
            "",
            "- This gate is for diagnostic world-model benchmarks such as MiraBench and RoboTrustBench.",
            "- Passing it means the manifest has enough labeled, multi-candidate, shortcut-controlled cases to update the ICLR evidence stack registry.",
            "- It does not by itself prove the EFV method is strong; selector results still need to beat the planner-score/visual proxy baseline.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--min-cases", type=int, default=16)
    parser.add_argument("--min-tasks", type=int, default=1)
    parser.add_argument("--min-oracle-better-cases", type=int, default=1)
    parser.add_argument("--min-candidates-per-case", type=int, default=2)
    parser.add_argument("--category-key", action="append", default=[])
    parser.add_argument("--min-categories", type=int, default=1)
    parser.add_argument("--require-metadata-key", action="append", default=[])
    parser.add_argument("--no-require-planner-score-baseline", action="store_true")
    parser.add_argument("--min-planner-score-oracle-gap", type=int, default=1)
    parser.add_argument("--min-planner-score-failures", type=int, default=1)
    args = parser.parse_args()

    result = evaluate_diagnostic_gate(
        load_jsonl(args.manifest),
        min_cases=args.min_cases,
        min_tasks=args.min_tasks,
        min_oracle_better_cases=args.min_oracle_better_cases,
        min_candidates_per_case=args.min_candidates_per_case,
        category_keys=list(args.category_key),
        min_categories=args.min_categories,
        required_metadata_keys=list(args.require_metadata_key),
        require_planner_score_baseline=not args.no_require_planner_score_baseline,
        min_planner_score_oracle_gap=args.min_planner_score_oracle_gap,
        min_planner_score_failures=args.min_planner_score_failures,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    markdown = render_markdown(result)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
