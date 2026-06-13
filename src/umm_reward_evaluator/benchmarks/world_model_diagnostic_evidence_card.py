"""Generate evidence-card proposals for world-model diagnostic benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.evidence_card_validator import validate_card, render_markdown as render_validation_markdown


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


def _selector_by_name(selector_table: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("selector")): row for row in selector_table.get("selectors", [])}


def build_card(
    *,
    registry_entry: dict[str, Any],
    diagnostic_gate: dict[str, Any],
    selector_table: dict[str, Any],
    readiness_gate: dict[str, Any],
    verifier_selector: str,
    run_root: Path,
    evidence_docs: list[str] | None = None,
) -> dict[str, Any]:
    selectors = _selector_by_name(selector_table)
    proxy = selectors.get("planner_or_model_score", {})
    verifier = selectors.get(verifier_selector, {})
    readiness_passed = bool(readiness_gate.get("passed"))
    diagnostic_passed = bool(diagnostic_gate.get("passed"))
    docs = evidence_docs or [
        str(run_root / "selectors" / "world_model_diagnostic_gate.md"),
        str(run_root / "selectors" / "world_model_diagnostic_selector_table.md"),
        str(run_root / "selectors" / "world_model_diagnostic_readiness_gate.md"),
        str(run_root / "selectors" / "world_model_diagnostic_registry_entry_proposal.md"),
    ]
    return {
        "benchmark": str(registry_entry.get("benchmark", "world_model_diagnostic")),
        "year": int(registry_entry.get("year", 0) or 0),
        "layer": str(registry_entry.get("layer", "world_model_diagnostic")),
        "status": str(registry_entry.get("status", "pending")),
        "cases": int(registry_entry.get("cases", 0) or 0),
        "tasks": int(registry_entry.get("tasks", 0) or 0),
        "rank0_success": _as_float(registry_entry.get("rank0_success")),
        "oracle_success": _as_float(registry_entry.get("oracle_success")),
        "method_name": verifier_selector,
        "method_success": _as_float(registry_entry.get("method_success"), _as_float(verifier.get("selector_success"))),
        "best_non_oracle_baseline_name": "max(rank0, random_expected, planner_or_model_score)",
        "best_non_oracle_baseline_success": _as_float(
            registry_entry.get("best_non_oracle_baseline_success"),
            _as_float(proxy.get("selector_success")),
        ),
        "shortcut_controls": list(registry_entry.get("shortcut_controls") or []),
        "mechanism_claim": (
            "Executable-future verification is evaluated as a selector over public world-model candidates: "
            "the EFV score must choose benchmark-judged reliable futures better than rank0, random, and the visual/model-score proxy."
        ),
        "counterintuitive_observation": (
            "The diagnostic counts only when oracle judgment exposes visual/model-score proxy failure; if that proxy already matches oracle, "
            "the benchmark is not evidence for future-selection brittleness."
        ),
        "claim_boundary": (
            "This is a diagnostic world-model layer, not simulator execution or real-robot evidence. "
            f"diagnostic_gate_passed={diagnostic_passed}; diagnostic_readiness_passed={readiness_passed}."
        ),
        "evidence_docs": docs,
        "registry_evidence": str(registry_entry.get("evidence", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--registry-entry-json", type=Path)
    parser.add_argument("--diagnostic-gate-json", type=Path)
    parser.add_argument("--selector-table-json", type=Path)
    parser.add_argument("--diagnostic-readiness-json", type=Path)
    parser.add_argument("--verifier-selector", required=True)
    parser.add_argument("--output-card", type=Path, required=True)
    parser.add_argument("--output-validation-json", type=Path)
    parser.add_argument("--output-validation-md", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--evidence-doc", action="append", default=[])
    args = parser.parse_args()

    selectors_dir = args.run_root / "selectors"
    registry_entry_path = args.registry_entry_json or selectors_dir / "world_model_diagnostic_registry_entry_proposal.json"
    diagnostic_gate_path = args.diagnostic_gate_json or selectors_dir / "world_model_diagnostic_gate.json"
    selector_table_path = args.selector_table_json or selectors_dir / "world_model_diagnostic_selector_table.json"
    readiness_path = args.diagnostic_readiness_json or selectors_dir / "world_model_diagnostic_readiness_gate.json"
    card = build_card(
        registry_entry=_load_json(registry_entry_path),
        diagnostic_gate=_load_json(diagnostic_gate_path),
        selector_table=_load_json(selector_table_path),
        readiness_gate=_load_json(readiness_path),
        verifier_selector=args.verifier_selector,
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
