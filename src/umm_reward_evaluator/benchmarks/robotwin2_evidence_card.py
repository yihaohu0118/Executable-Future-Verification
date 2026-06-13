"""Generate RoboTwin2 evidence-card proposals from finalized run outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.evidence_card_validator import validate_card, render_markdown as render_validation_markdown
from umm_reward_evaluator.benchmarks.iclr_registry_proposal import BASELINE_COLUMNS, ENVELOPE_COLUMNS


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _selector_totals(selector_table: dict[str, Any], columns: tuple[str, ...]) -> dict[str, float]:
    totals = {column: 0.0 for column in columns}
    for row in selector_table.get("tasks", []):
        for column in columns:
            if row.get(column) is not None:
                totals[column] += _as_float(row.get(column))
    return totals


def _best_name_and_value(selector_table: dict[str, Any], columns: tuple[str, ...]) -> tuple[str, float]:
    totals = _selector_totals(selector_table, columns)
    if not totals:
        return "-", 0.0
    name, value = max(totals.items(), key=lambda item: (item[1], item[0]))
    return name, float(value)


def build_card(
    *,
    registry_entry: dict[str, Any],
    selector_table: dict[str, Any],
    paper_gate: dict[str, Any],
    run_root: Path,
    evidence_docs: list[str] | None = None,
) -> dict[str, Any]:
    method_name, method_from_table = _best_name_and_value(selector_table, ENVELOPE_COLUMNS)
    baseline_name, baseline_from_table = _best_name_and_value(selector_table, BASELINE_COLUMNS)
    method_success = _as_float(registry_entry.get("method_success"), method_from_table)
    baseline_success = _as_float(registry_entry.get("best_non_oracle_baseline_success"), baseline_from_table)
    docs = evidence_docs or [
        str(run_root / "selectors" / "robotwin2_raw_integrity_report.md"),
        str(run_root / "selectors" / "robotwin2_readiness_report.md"),
        str(run_root / "selectors" / "robotwin2_selector_table.md"),
        str(run_root / "selectors" / "robotwin2_paper_readiness_gate.md"),
        str(run_root / "selectors" / "robotwin2_registry_entry_proposal.md"),
    ]
    paper_passed = bool(paper_gate.get("passed"))
    return {
        "benchmark": "RoboTwin2",
        "year": 2025,
        "layer": "executable_second",
        "status": str(registry_entry.get("status", "pending")),
        "cases": int(registry_entry.get("cases", 0) or 0),
        "tasks": int(registry_entry.get("tasks", 0) or 0),
        "rank0_success": _entry_float(registry_entry, "rank0_success"),
        "oracle_success": _entry_float(registry_entry, "oracle_success"),
        "method_name": method_name,
        "method_success": method_success,
        "best_non_oracle_baseline_name": baseline_name,
        "best_non_oracle_baseline_success": baseline_success,
        "shortcut_controls": list(registry_entry.get("shortcut_controls") or []),
        "mechanism_claim": (
            "Task/contact-conditioned execution-envelope selectors recover executable RoboTwin2 futures only after "
            "candidate ID, rank, action-only, energy, learned action-probe, and DTW template baselines are controlled."
        ),
        "counterintuitive_observation": (
            "The RoboTwin2 result is counted only if non-template successes and matched low-DTW failures survive; "
            "otherwise strong DTW nearest-positive performance downgrades it to a diagnostic."
        ),
        "claim_boundary": (
            "Counts as the second executable benchmark only when the RoboTwin2 paper-readiness gate passes. "
            f"Current generated card observes paper_readiness_passed={paper_passed}; no real-robot claim is implied."
        ),
        "evidence_docs": docs,
        "registry_evidence": str(registry_entry.get("evidence", "")),
    }


def _entry_float(registry_entry: dict[str, Any], key: str) -> float:
    return _as_float(registry_entry.get(key))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--registry-entry-json", type=Path)
    parser.add_argument("--selector-table-json", type=Path)
    parser.add_argument("--paper-gate-json", type=Path)
    parser.add_argument("--output-card", type=Path, required=True)
    parser.add_argument("--output-validation-json", type=Path)
    parser.add_argument("--output-validation-md", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--evidence-doc", action="append", default=[])
    args = parser.parse_args()

    selectors_dir = args.run_root / "selectors"
    registry_entry_path = args.registry_entry_json or selectors_dir / "robotwin2_registry_entry_proposal.json"
    selector_table_path = args.selector_table_json or selectors_dir / "robotwin2_selector_table.json"
    paper_gate_path = args.paper_gate_json or selectors_dir / "robotwin2_paper_readiness_gate.json"
    card = build_card(
        registry_entry=_load_json(registry_entry_path),
        selector_table=_load_json(selector_table_path),
        paper_gate=_load_json(paper_gate_path),
        run_root=args.run_root,
        evidence_docs=list(args.evidence_doc) or None,
    )
    args.output_card.parent.mkdir(parents=True, exist_ok=True)
    args.output_card.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validation = validate_card(card, base_dir=args.repo_root)
    if args.output_validation_json:
        args.output_validation_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_validation_json.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_validation_markdown(validation)
    if args.output_validation_md:
        args.output_validation_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_validation_md.write_text(markdown, encoding="utf-8")
    print(json.dumps({"card": str(args.output_card), "validation_passed": validation["passed"]}, indent=2, sort_keys=True))
    if not validation["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
