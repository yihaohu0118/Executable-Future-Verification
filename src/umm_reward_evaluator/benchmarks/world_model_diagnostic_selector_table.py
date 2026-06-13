"""Selector table for world-model diagnostic manifests."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from umm_reward_evaluator.benchmarks.common import case_group_key, load_jsonl, oracle_key


PLANNER_OR_MODEL_SCORE = "__planner_or_model__"


def _nested_get(payload: dict[str, Any], key: str) -> Any:
    value: Any = payload
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _score_value(row: dict[str, Any], score_key: str) -> float | None:
    if score_key == PLANNER_OR_MODEL_SCORE:
        value = row.get("planner_score")
        if value is None:
            value = row.get("model_score")
    else:
        value = _nested_get({"metadata": row.get("metadata") or {}, **row}, score_key)
    if value is None:
        return None
    return float(value)


def _group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[case_group_key(row)].append(row)
    return dict(grouped)


def _rank0(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(case_rows, key=lambda row: int(row.get("candidate_rank_by_planner", 999999)))


def _select_by_score(case_rows: list[dict[str, Any]], score_key: str) -> dict[str, Any] | None:
    scored = [(row, _score_value(row, score_key)) for row in case_rows]
    if any(score is None for _, score in scored):
        return None
    return max(
        scored,
        key=lambda item: (float(item[1]), -int(item[0].get("candidate_rank_by_planner", 999999))),
    )[0]


def _empty_selector(selector: str) -> dict[str, Any]:
    return {
        "selector": selector,
        "cases": 0,
        "covered_cases": 0,
        "selector_success": 0.0,
        "oracle_success": 0,
        "rank0_success": 0,
        "oracle_better_than_selector": 0,
        "selector_oracle_match": 0,
        "selector_success_rate": 0.0,
        "coverage_rate": 0.0,
    }


def _finalize_selector(summary: dict[str, Any]) -> dict[str, Any]:
    cases = int(summary["cases"])
    covered = int(summary["covered_cases"])
    summary["selector_success_rate"] = float(summary["selector_success"]) / covered if covered else 0.0
    summary["coverage_rate"] = covered / cases if cases else 0.0
    return summary


def evaluate_selectors(
    rows: list[dict[str, Any]],
    *,
    verifier_score_key: str | None = None,
    proxy_score_key: str = PLANNER_OR_MODEL_SCORE,
) -> dict[str, Any]:
    grouped = _group_rows(rows)
    selectors: dict[str, dict[str, Any]] = {
        "rank0": _empty_selector("rank0"),
        "random_expected": _empty_selector("random_expected"),
        "planner_or_model_score": _empty_selector("planner_or_model_score"),
        "oracle": _empty_selector("oracle"),
    }
    if verifier_score_key:
        selectors[f"verifier_score:{verifier_score_key}"] = _empty_selector(f"verifier_score:{verifier_score_key}")

    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    selections: list[dict[str, Any]] = []
    for (task, case_id), case_rows in sorted(grouped.items()):
        rank0 = _rank0(case_rows)
        oracle = max(case_rows, key=oracle_key)
        random_expected = float(np.mean([1.0 if row.get("oracle_success") else 0.0 for row in case_rows]))
        selected_by_name: dict[str, dict[str, Any] | None] = {
            "rank0": rank0,
            "random_expected": None,
            "planner_or_model_score": _select_by_score(case_rows, proxy_score_key),
            "oracle": oracle,
        }
        if verifier_score_key:
            selected_by_name[f"verifier_score:{verifier_score_key}"] = _select_by_score(case_rows, verifier_score_key)

        for name, summary in selectors.items():
            summary["cases"] += 1
            summary["oracle_success"] += int(bool(oracle.get("oracle_success")))
            summary["rank0_success"] += int(bool(rank0.get("oracle_success")))

            task_summary = by_task[task].setdefault(name, _empty_selector(name))
            task_summary["cases"] += 1
            task_summary["oracle_success"] += int(bool(oracle.get("oracle_success")))
            task_summary["rank0_success"] += int(bool(rank0.get("oracle_success")))

            if name == "random_expected":
                success_value = random_expected
                covered = True
                selected = None
            else:
                selected = selected_by_name[name]
                covered = selected is not None
                success_value = float(bool(selected.get("oracle_success"))) if selected is not None else 0.0

            if covered:
                summary["covered_cases"] += 1
                summary["selector_success"] += success_value
                summary["oracle_better_than_selector"] += int(selected is not None and oracle_key(oracle) > oracle_key(selected))
                summary["selector_oracle_match"] += int(selected is not None and selected.get("candidate_id") == oracle.get("candidate_id"))
                task_summary["covered_cases"] += 1
                task_summary["selector_success"] += success_value
                task_summary["oracle_better_than_selector"] += int(selected is not None and oracle_key(oracle) > oracle_key(selected))
                task_summary["selector_oracle_match"] += int(selected is not None and selected.get("candidate_id") == oracle.get("candidate_id"))

        selections.append(
            {
                "task_name": task,
                "case_id": case_id,
                "rank0_candidate_id": rank0.get("candidate_id"),
                "oracle_candidate_id": oracle.get("candidate_id"),
                "planner_or_model_score_candidate_id": None
                if selected_by_name["planner_or_model_score"] is None
                else selected_by_name["planner_or_model_score"].get("candidate_id"),
                "verifier_candidate_id": None
                if not verifier_score_key or selected_by_name[f"verifier_score:{verifier_score_key}"] is None
                else selected_by_name[f"verifier_score:{verifier_score_key}"].get("candidate_id"),
                "rank0_success": bool(rank0.get("oracle_success")),
                "oracle_success": bool(oracle.get("oracle_success")),
            }
        )

    selector_rows = [_finalize_selector(summary) for summary in selectors.values()]
    finalized_by_task = {
        task: {name: _finalize_selector(summary) for name, summary in selector_map.items()}
        for task, selector_map in sorted(by_task.items())
    }
    return {
        "summary": {
            "rows": len(rows),
            "cases": len(grouped),
            "tasks": sorted({task for task, _ in grouped}),
            "verifier_score_key": verifier_score_key,
            "proxy_score_key": proxy_score_key,
        },
        "selectors": selector_rows,
        "by_task": finalized_by_task,
        "selections": selections,
    }


def _fmt_success(value: float | int, cases: int) -> str:
    return f"{float(value):.1f}/{cases}"


def render_markdown(result: dict[str, Any], *, title: str = "World-Model Diagnostic Selector Table") -> str:
    summary = result["summary"]
    lines = [
        f"# {title}",
        "",
        f"- cases: {summary['cases']}",
        f"- tasks: {len(summary['tasks'])}",
        f"- proxy score key: `{summary['proxy_score_key']}`",
        f"- verifier score key: `{summary['verifier_score_key']}`",
        "",
        "| Selector | Covered | Success | Oracle | Rank0 | Oracle better | Oracle match |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["selectors"]:
        cases = int(row["cases"])
        lines.append(
            "| {selector} | {covered}/{cases} | {success} | {oracle} | {rank0} | {better} | {match} |".format(
                selector=row["selector"],
                covered=int(row["covered_cases"]),
                cases=cases,
                success=_fmt_success(row["selector_success"], cases),
                oracle=_fmt_success(row["oracle_success"], cases),
                rank0=_fmt_success(row["rank0_success"], cases),
                better=int(row["oracle_better_than_selector"]),
                match=int(row["selector_oracle_match"]),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `planner_or_model_score` is the visual/model-score proxy baseline.",
            "- `verifier_score:*` is the EFV-side score; it must beat the proxy before updating the ICLR evidence registry.",
            "- `oracle` is an upper bound from benchmark or human judgment labels, not a deployable selector.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--proxy-score-key", default=PLANNER_OR_MODEL_SCORE)
    parser.add_argument("--verifier-score-key")
    parser.add_argument("--title", default="World-Model Diagnostic Selector Table")
    args = parser.parse_args()

    result = evaluate_selectors(
        load_jsonl(args.manifest),
        verifier_score_key=args.verifier_score_key,
        proxy_score_key=args.proxy_score_key,
    )
    payload = {"manifest": str(args.manifest), **result}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(result, title=args.title)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
