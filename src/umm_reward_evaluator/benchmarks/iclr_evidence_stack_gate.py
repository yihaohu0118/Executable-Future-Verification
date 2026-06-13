"""Cross-benchmark evidence gate for the EFV ICLR story."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODERN_YEARS = {2025, 2026}
EXECUTABLE_LAYERS = {"executable_primary", "executable_second"}
DIAGNOSTIC_LAYERS = {"world_model_diagnostic", "trust_diagnostic", "robustness_diagnostic"}
REQUIRED_CONTROLS = {
    "rank0",
    "random",
    "energy_or_magnitude",
    "action_only",
    "candidate_id_or_rank_remap",
}


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


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def _benchmark_passes(entry: dict[str, Any], *, min_cases: int, min_tasks: int, min_margin: float) -> bool:
    cases = _as_int(entry.get("cases"))
    tasks = _as_int(entry.get("tasks"), 1)
    method = _as_float(entry.get("method_success"))
    baseline = _as_float(entry.get("best_non_oracle_baseline_success"))
    oracle = _as_float(entry.get("oracle_success"))
    rank0 = _as_float(entry.get("rank0_success"))
    controls = set(entry.get("shortcut_controls") or [])
    return (
        entry.get("status") == "passed"
        and int(entry.get("year", 0)) in MODERN_YEARS
        and cases >= min_cases
        and tasks >= min_tasks
        and oracle > rank0
        and method >= baseline + min_margin
        and REQUIRED_CONTROLS.issubset(controls)
    )


def _check(name: str, passed: bool, detail: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def evaluate_evidence_stack(
    entries: list[dict[str, Any]],
    *,
    min_total_passed: int = 3,
    min_executable_passed: int = 2,
    min_diagnostic_passed: int = 1,
    min_cases_per_passed_benchmark: int = 16,
    min_tasks_per_passed_executable: int = 4,
    min_selector_margin: float = 1.0,
) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        layer = str(entry.get("layer", "unknown"))
        min_tasks = min_tasks_per_passed_executable if layer in EXECUTABLE_LAYERS else 1
        passed = _benchmark_passes(
            entry,
            min_cases=min_cases_per_passed_benchmark,
            min_tasks=min_tasks,
            min_margin=min_selector_margin,
        )
        controls = set(entry.get("shortcut_controls") or [])
        normalized.append(
            {
                **entry,
                "gate_passed": passed,
                "missing_controls": sorted(REQUIRED_CONTROLS - controls),
                "modern_year": int(entry.get("year", 0)) in MODERN_YEARS,
                "selector_margin": _as_float(entry.get("method_success")) - _as_float(entry.get("best_non_oracle_baseline_success")),
            }
        )

    passed_entries = [entry for entry in normalized if entry["gate_passed"]]
    executable_passed = [entry for entry in passed_entries if entry.get("layer") in EXECUTABLE_LAYERS]
    diagnostic_passed = [entry for entry in passed_entries if entry.get("layer") in DIAGNOSTIC_LAYERS]
    benchmark_names = [str(entry.get("benchmark")) for entry in passed_entries]

    checks = [
        _check(
            "modern_scope",
            all(entry["modern_year"] for entry in normalized),
            {
                "non_modern": [
                    {"benchmark": entry.get("benchmark"), "year": entry.get("year")}
                    for entry in normalized
                    if not entry["modern_year"]
                ]
            },
        ),
        _check(
            "total_passed_benchmarks",
            len(passed_entries) >= min_total_passed,
            {"benchmarks": benchmark_names, "minimum": min_total_passed},
        ),
        _check(
            "executable_layers",
            len(executable_passed) >= min_executable_passed,
            {"benchmarks": [entry.get("benchmark") for entry in executable_passed], "minimum": min_executable_passed},
        ),
        _check(
            "diagnostic_layers",
            len(diagnostic_passed) >= min_diagnostic_passed,
            {"benchmarks": [entry.get("benchmark") for entry in diagnostic_passed], "minimum": min_diagnostic_passed},
        ),
        _check(
            "primary_robocasa_present",
            any(entry.get("benchmark") == "RoboCasa365" and entry["gate_passed"] for entry in normalized),
            {"required": "RoboCasa365"},
        ),
        _check(
            "robotwin2_second_layer_present",
            any(entry.get("benchmark") == "RoboTwin2" and entry["gate_passed"] for entry in normalized),
            {"required": "RoboTwin2"},
        ),
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "benchmarks": normalized,
        "thresholds": {
            "min_total_passed": min_total_passed,
            "min_executable_passed": min_executable_passed,
            "min_diagnostic_passed": min_diagnostic_passed,
            "min_cases_per_passed_benchmark": min_cases_per_passed_benchmark,
            "min_tasks_per_passed_executable": min_tasks_per_passed_executable,
            "min_selector_margin": min_selector_margin,
        },
    }


def render_markdown(result: dict[str, Any], *, title: str = "ICLR Evidence Stack Gate") -> str:
    lines = [
        f"# {title}",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in result["checks"]:
        detail = check["detail"]
        benchmarks = detail.get("benchmarks")
        if benchmarks is None:
            benchmarks = detail.get("non_modern", detail.get("required", "-"))
        if isinstance(benchmarks, list):
            detail_text = ", ".join(str(item) for item in benchmarks) or "-"
        else:
            detail_text = str(benchmarks)
        minimum = detail.get("minimum")
        if minimum is not None:
            detail_text = f"{detail_text} / min {minimum}"
        lines.append(f"| `{check['name']}` | {'pass' if check['passed'] else 'fail'} | {detail_text} |")

    lines.extend(
        [
            "",
            "| Benchmark | Year | Layer | Status | Cases | Tasks | Rank0 | Oracle | Method | Best baseline | Margin | Missing controls |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for entry in result["benchmarks"]:
        lines.append(
            "| {benchmark} | {year} | {layer} | {status} | {cases} | {tasks} | {rank0:.1f} | {oracle:.1f} | {method:.1f} | {baseline:.1f} | {margin:.1f} | {missing} |".format(
                benchmark=entry.get("benchmark", "-"),
                year=int(entry.get("year", 0)),
                layer=entry.get("layer", "-"),
                status="pass" if entry.get("gate_passed") else "fail",
                cases=_as_int(entry.get("cases")),
                tasks=_as_int(entry.get("tasks"), 1),
                rank0=_as_float(entry.get("rank0_success")),
                oracle=_as_float(entry.get("oracle_success")),
                method=_as_float(entry.get("method_success")),
                baseline=_as_float(entry.get("best_non_oracle_baseline_success")),
                margin=_as_float(entry.get("selector_margin")),
                missing=", ".join(entry.get("missing_controls") or []) or "-",
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- This is a paper-level gate, not a selector metric.",
            "- A benchmark can be useful diagnostically while still failing this gate.",
            "- The ICLR claim should not be phrased as multi-benchmark evidence until this gate passes with current result files.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--min-total-passed", type=int, default=3)
    parser.add_argument("--min-executable-passed", type=int, default=2)
    parser.add_argument("--min-diagnostic-passed", type=int, default=1)
    parser.add_argument("--min-cases-per-passed-benchmark", type=int, default=16)
    parser.add_argument("--min-tasks-per-passed-executable", type=int, default=4)
    parser.add_argument("--min-selector-margin", type=float, default=1.0)
    args = parser.parse_args()

    payload = _load_json(args.evidence_json)
    entries = payload.get("benchmarks")
    if not isinstance(entries, list):
        raise SystemExit("--evidence-json must contain a 'benchmarks' list")
    result = evaluate_evidence_stack(
        entries,
        min_total_passed=args.min_total_passed,
        min_executable_passed=args.min_executable_passed,
        min_diagnostic_passed=args.min_diagnostic_passed,
        min_cases_per_passed_benchmark=args.min_cases_per_passed_benchmark,
        min_tasks_per_passed_executable=args.min_tasks_per_passed_executable,
        min_selector_margin=args.min_selector_margin,
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
