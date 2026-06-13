"""Run RoboTwin2 no-exact-expert counterfactual selector analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.common import load_jsonl, summarize_headroom, write_jsonl
from umm_reward_evaluator.benchmarks.filter_candidate_manifest import filter_rows
from umm_reward_evaluator.benchmarks.robotwin2_rank_randomization_sweep import run_sweep


KEY_SELECTORS = (
    "rank0",
    "random_expected",
    "heuristic:energy_sum_max",
    "heuristic:length_max",
    "heuristic:smoothness_max",
    "linear_probe:action_distribution:same_task:ridge_l2_1",
    "prototype:action_distribution:same_task:nearest_positive",
    "prototype:contact_envelope:same_task:nearest_positive",
    "prototype:contact_envelope:same_task:nearest_pos_neg",
    "linear_probe:contact_envelope:same_task:ridge_l2_1",
    "trace_distance:dtw_action:same_task:nearest_positive",
    "trace_distance:dtw_contact_envelope:same_task:nearest_positive",
    "trace_distance:dtw_object_relation:same_task:nearest_positive",
)


def selector_by_name(sweep: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["selector"]: row
        for row in sweep.get("aggregate", {}).get("selectors", [])
        if isinstance(row, dict) and "selector" in row
    }


def selector_success(selectors: dict[str, dict[str, Any]], name: str) -> float:
    return float(selectors.get(name, {}).get("mean_success", 0.0))


def run_counterfactual(
    rows: list[dict[str, Any]],
    *,
    exclude_candidate_ids: set[str],
    num_seeds: int = 50,
    seed_start: int = 0,
    mode: str = "failure_rank0_shuffle_rest",
    remap_candidate_ids: bool = True,
) -> dict[str, Any]:
    filtered = filter_rows(
        rows,
        exclude_metadata_key=None,
        exclude_metadata_value=None,
        exclude_candidate_ids=exclude_candidate_ids,
        preserve_ranks=False,
    )
    seeds = list(range(seed_start, seed_start + num_seeds))
    sweep = run_sweep(
        filtered,
        seeds=seeds,
        mode=mode,
        remap_candidate_ids=remap_candidate_ids,
    )
    summary = summarize_headroom(filtered)
    selectors = selector_by_name(sweep)
    cases = float(summary.get("cases", 0))
    contact_best = max(
        selector_success(selectors, "prototype:contact_envelope:same_task:nearest_positive"),
        selector_success(selectors, "prototype:contact_envelope:same_task:nearest_pos_neg"),
        selector_success(selectors, "linear_probe:contact_envelope:same_task:ridge_l2_1"),
    )
    template_best = max(
        selector_success(selectors, "trace_distance:dtw_action:same_task:nearest_positive"),
        selector_success(selectors, "trace_distance:dtw_contact_envelope:same_task:nearest_positive"),
    )
    return {
        "excluded_candidate_ids": sorted(exclude_candidate_ids),
        "input_rows": len(rows),
        "filtered_rows": len(filtered),
        "headroom": summary,
        "mode": mode,
        "num_seeds": num_seeds,
        "seed_start": seed_start,
        "remap_candidate_ids": remap_candidate_ids,
        "key_selectors": {
            name: selectors[name]
            for name in KEY_SELECTORS
            if name in selectors
        },
        "diagnosis": {
            "exact_expert_removed_recoverable": bool(cases > 0 and float(summary.get("oracle_success", 0)) == cases),
            "rank0_still_fails": bool(float(summary.get("rank0_success", 0)) < float(summary.get("oracle_success", 0))),
            "contact_envelope_recovers_without_exact_expert": bool(cases > 0 and contact_best == cases),
            "dtw_template_still_recovers_without_exact_expert": bool(cases > 0 and template_best == cases),
            "contact_best_mean_success": contact_best,
            "dtw_template_best_mean_success": template_best,
        },
        "sweep": sweep,
    }


def render_markdown(report: dict[str, Any]) -> str:
    headroom = report["headroom"]
    diagnosis = report["diagnosis"]
    lines = [
        "# RoboTwin2 No-Exact-Expert Counterfactual",
        "",
        f"- excluded candidate ids: `{', '.join(report['excluded_candidate_ids'])}`",
        f"- input rows: `{report['input_rows']}`",
        f"- filtered rows: `{report['filtered_rows']}`",
        f"- cases: `{headroom['cases']}`",
        f"- rank0 success: `{headroom['rank0_success']}/{headroom['cases']}`",
        f"- oracle success: `{headroom['oracle_success']}/{headroom['cases']}`",
        f"- anonymous remap seeds: `{report['num_seeds']}`",
        "",
        "## Key Selectors",
        "",
        "| Selector | Mean success | Rate | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in KEY_SELECTORS:
        row = report["key_selectors"].get(name)
        if not row:
            continue
        lines.append(
            f"| `{name}` | {row['mean_success']:.3g} | {row['mean_success_rate']:.3g} | "
            f"{row['min_success']:.3g} | {row['max_success']:.3g} |"
        )
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            f"- exact expert removed but oracle remains recoverable: `{str(diagnosis['exact_expert_removed_recoverable']).lower()}`",
            f"- rank0 still fails below oracle: `{str(diagnosis['rank0_still_fails']).lower()}`",
            f"- contact envelope recovers without exact expert: `{str(diagnosis['contact_envelope_recovers_without_exact_expert']).lower()}`",
            f"- DTW template also recovers without exact expert: `{str(diagnosis['dtw_template_still_recovers_without_exact_expert']).lower()}`",
            "",
        ]
    )
    if diagnosis["contact_envelope_recovers_without_exact_expert"] and diagnosis["dtw_template_still_recovers_without_exact_expert"]:
        lines.extend(
            [
                "Interpretation: this rules out exact expert lookup, but not template-distance matching.",
                "",
            ]
        )
    elif diagnosis["contact_envelope_recovers_without_exact_expert"]:
        lines.extend(["Interpretation: this creates anti-template pressure in favor of contact-envelope verification.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-filtered-manifest", type=Path)
    parser.add_argument("--exclude-candidate-id", action="append", default=["full_gripper_aware"])
    parser.add_argument("--num-seeds", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--mode", default="failure_rank0_shuffle_rest")
    parser.add_argument("--no-remap-candidate-ids", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    exclude_ids = set(args.exclude_candidate_id)
    report = run_counterfactual(
        rows,
        exclude_candidate_ids=exclude_ids,
        num_seeds=args.num_seeds,
        seed_start=args.seed_start,
        mode=args.mode,
        remap_candidate_ids=not args.no_remap_candidate_ids,
    )
    if args.output_filtered_manifest:
        filtered = filter_rows(
            rows,
            exclude_metadata_key=None,
            exclude_metadata_value=None,
            exclude_candidate_ids=exclude_ids,
            preserve_ranks=False,
        )
        write_jsonl(args.output_filtered_manifest, filtered)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
