"""Validate structured evidence cards used by the EFV evidence stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODERN_YEARS = {2025, 2026}
REQUIRED_CONTROLS = {
    "rank0",
    "random",
    "energy_or_magnitude",
    "action_only",
    "candidate_id_or_rank_remap",
}
REQUIRED_FIELDS = (
    "benchmark",
    "year",
    "layer",
    "status",
    "cases",
    "tasks",
    "rank0_success",
    "oracle_success",
    "method_success",
    "best_non_oracle_baseline_success",
    "shortcut_controls",
    "mechanism_claim",
    "counterintuitive_observation",
    "claim_boundary",
    "evidence_docs",
    "registry_evidence",
)


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


def validate_card(card: dict[str, Any], *, base_dir: Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in card:
            errors.append(f"missing required field: {field}")

    status = str(card.get("status", ""))
    if status == "passed":
        year = int(card.get("year", 0) or 0)
        if year not in MODERN_YEARS:
            errors.append(f"passed card must use a 2025-2026 benchmark, got year={year}")
        if int(card.get("cases", 0) or 0) <= 0:
            errors.append("passed card must have cases > 0")
        if int(card.get("tasks", 0) or 0) <= 0:
            errors.append("passed card must have tasks > 0")
        if _as_float(card.get("oracle_success")) <= _as_float(card.get("rank0_success")):
            errors.append("passed card must have oracle_success > rank0_success")
        if _as_float(card.get("method_success")) <= _as_float(card.get("best_non_oracle_baseline_success")):
            errors.append("passed card must have method_success > best_non_oracle_baseline_success")
        controls = set(card.get("shortcut_controls") or [])
        missing_controls = sorted(REQUIRED_CONTROLS - controls)
        if missing_controls:
            errors.append(f"passed card missing shortcut controls: {', '.join(missing_controls)}")

    for field in ("mechanism_claim", "counterintuitive_observation", "claim_boundary", "registry_evidence"):
        if not str(card.get(field, "")).strip():
            errors.append(f"{field} must be non-empty")

    evidence_docs = card.get("evidence_docs") or []
    if not isinstance(evidence_docs, list) or not evidence_docs:
        errors.append("evidence_docs must be a non-empty list")
    elif base_dir is not None:
        for doc in evidence_docs:
            path = base_dir / str(doc)
            if not path.exists():
                warnings.append(f"evidence doc does not exist: {doc}")

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "benchmark": card.get("benchmark"),
            "status": card.get("status"),
            "cases": card.get("cases"),
            "tasks": card.get("tasks"),
            "method_success": card.get("method_success"),
            "best_non_oracle_baseline_success": card.get("best_non_oracle_baseline_success"),
        },
    }


def render_markdown(result: dict[str, Any], *, title: str = "Evidence Card Validation") -> str:
    summary = result["summary"]
    lines = [
        f"# {title}",
        "",
        f"- passed: `{str(result['passed']).lower()}`",
        f"- benchmark: {summary.get('benchmark')}",
        f"- status: {summary.get('status')}",
        f"- cases: {summary.get('cases')}",
        f"- tasks: {summary.get('tasks')}",
        f"- method success: {summary.get('method_success')}",
        f"- best baseline success: {summary.get('best_non_oracle_baseline_success')}",
        "",
        "## Errors",
        "",
    ]
    lines.extend(f"- {error}" for error in result["errors"]) if result["errors"] else lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in result["warnings"]) if result["warnings"] else lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    result = validate_card(_load_json(args.card), base_dir=args.repo_root)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(result)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
