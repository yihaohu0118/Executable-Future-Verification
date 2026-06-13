"""Plan the next closure step for 2026 world-model diagnostic benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.iclr_evidence_stack_gate import DIAGNOSTIC_LAYERS, _load_json


def _diagnostic_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [entry for entry in entries if entry.get("layer") in DIAGNOSTIC_LAYERS]


def _classify(entry: dict[str, Any]) -> tuple[str, str, list[str]]:
    benchmark = str(entry.get("benchmark", ""))
    cases = int(entry.get("cases") or 0)
    oracle = float(entry.get("oracle_success") or 0)
    rank0 = float(entry.get("rank0_success") or 0)
    controls = set(entry.get("shortcut_controls") or [])
    missing: list[str] = []

    required_controls = {
        "rank0",
        "random",
        "energy_or_magnitude",
        "action_only",
        "candidate_id_or_rank_remap",
        "oracle_judgment_labels",
        "proxy_or_rank0_failure",
        "visual_or_model_score_proxy",
    }
    missing.extend(sorted(required_controls - controls))

    if cases <= 0:
        return (
            "blocked_public_artifacts",
            "Wait for public judgment/result artifacts, then convert them into a multi-candidate diagnostic manifest.",
            ["cases"] + missing,
        )
    if oracle <= rank0:
        missing.append("oracle_headroom")
    if "RoboTrustBench" in benchmark and cases < 100:
        return (
            "adapter_validation_only",
            "Use the public subset only to validate request generation and manifest plumbing; do not count it as a paper-level diagnostic.",
            missing,
        )
    if "RoboWM" in benchmark:
        return (
            "conditional_robustness_diagnostic",
            "Keep this as a robustness diagnostic until GT replay/evaluator stability is resolved under a pinned official patch.",
            missing,
        )
    return (
        "needs_selector_manifest",
        "Build a multi-candidate judgment manifest, run the diagnostic gate, then require EFV to beat the visual/model-score proxy.",
        missing,
    )


def build_closure_plan(entries: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for entry in _diagnostic_entries(entries):
        status, action, missing = _classify(entry)
        if entry.get("benchmark") == "MiraBench":
            priority = 1
        elif entry.get("benchmark") == "RoboTrustBench":
            priority = 2
        elif entry.get("benchmark") == "RoboWM-Bench":
            priority = 3
        else:
            priority = 9
        rows.append(
            {
                "benchmark": entry.get("benchmark"),
                "year": entry.get("year"),
                "layer": entry.get("layer"),
                "priority": priority,
                "closure_status": status,
                "cases": int(entry.get("cases") or 0),
                "tasks": int(entry.get("tasks") or 0),
                "missing": missing,
                "source_urls": entry.get("source_urls") or [],
                "next_action": action,
            }
        )
    rows.sort(key=lambda row: (int(row["priority"]), str(row["benchmark"])))
    return {
        "diagnostic_benchmarks": rows,
        "recommended_order": [row["benchmark"] for row in rows],
        "paper_rule": "Do not count a diagnostic benchmark until its gate passes and the EFV selector beats the visual/model-score proxy.",
    }


def render_markdown(plan: dict[str, Any], *, title: str = "World-Model Diagnostic Closure Plan") -> str:
    lines = [
        f"# {title}",
        "",
        f"- recommended order: {', '.join(str(item) for item in plan['recommended_order']) or '-'}",
        f"- paper rule: {plan['paper_rule']}",
        "",
        "| Priority | Benchmark | Status | Cases | Missing | Next action |",
        "| ---: | --- | --- | ---: | --- | --- |",
    ]
    for row in plan["diagnostic_benchmarks"]:
        lines.append(
            "| {priority} | {benchmark} | {status} | {cases} | {missing} | {action} |".format(
                priority=row["priority"],
                benchmark=row["benchmark"],
                status=row["closure_status"],
                cases=row["cases"],
                missing=", ".join(row["missing"]) or "-",
                action=row["next_action"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    payload = _load_json(args.evidence_json)
    entries = payload.get("benchmarks")
    if not isinstance(entries, list):
        raise SystemExit("--evidence-json must contain a 'benchmarks' list")
    plan = build_closure_plan(entries)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(plan)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
