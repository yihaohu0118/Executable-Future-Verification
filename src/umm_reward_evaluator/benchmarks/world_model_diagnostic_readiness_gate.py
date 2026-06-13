"""Paper-level readiness gate for world-model diagnostic evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def _selector_by_name(selector_table: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("selector")): row for row in selector_table.get("selectors", [])}


def _check(name: str, passed: bool, detail: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def evaluate_readiness(
    *,
    diagnostic_gate: dict[str, Any],
    selector_table: dict[str, Any],
    verifier_selector: str,
    proxy_selector: str = "planner_or_model_score",
    min_verifier_proxy_margin: float = 1.0,
    require_oracle_headroom_over_verifier: bool = False,
) -> dict[str, Any]:
    selectors = _selector_by_name(selector_table)
    summary = selector_table.get("summary") or {}
    cases = _as_int(summary.get("cases"), _as_int(diagnostic_gate.get("summary", {}).get("cases")))
    tasks = list(summary.get("tasks") or diagnostic_gate.get("summary", {}).get("tasks") or [])
    verifier = selectors.get(verifier_selector)
    proxy = selectors.get(proxy_selector)
    oracle = selectors.get("oracle")
    rank0 = selectors.get("rank0")
    random_expected = selectors.get("random_expected")

    verifier_success = _as_float(verifier.get("selector_success") if verifier else None)
    proxy_success = _as_float(proxy.get("selector_success") if proxy else None)
    oracle_success = _as_float(
        oracle.get("selector_success") if oracle else diagnostic_gate.get("summary", {}).get("oracle_success")
    )
    rank0_success = _as_float(rank0.get("selector_success") if rank0 else diagnostic_gate.get("summary", {}).get("rank0_success"))
    random_success = _as_float(random_expected.get("selector_success") if random_expected else None)
    verifier_covered = _as_int(verifier.get("covered_cases") if verifier else 0)
    proxy_covered = _as_int(proxy.get("covered_cases") if proxy else 0)
    oracle_covered = _as_int(oracle.get("covered_cases") if oracle else 0)
    margin = verifier_success - proxy_success

    checks = [
        _check("diagnostic_gate", bool(diagnostic_gate.get("passed")), {"passed": bool(diagnostic_gate.get("passed"))}),
        _check("verifier_present", verifier is not None, {"verifier_selector": verifier_selector}),
        _check("proxy_present", proxy is not None, {"proxy_selector": proxy_selector}),
        _check("oracle_present", oracle is not None, {"present": oracle is not None}),
        _check("rank0_present", rank0 is not None, {"present": rank0 is not None}),
        _check("random_present", random_expected is not None, {"present": random_expected is not None}),
        _check(
            "verifier_full_coverage",
            cases > 0 and verifier_covered == cases,
            {"covered_cases": verifier_covered, "cases": cases},
        ),
        _check(
            "proxy_full_coverage",
            cases > 0 and proxy_covered == cases,
            {"covered_cases": proxy_covered, "cases": cases},
        ),
        _check(
            "oracle_full_coverage",
            cases > 0 and oracle_covered == cases,
            {"covered_cases": oracle_covered, "cases": cases},
        ),
        _check(
            "proxy_failure_gap",
            oracle_success > proxy_success,
            {"oracle_success": oracle_success, "proxy_success": proxy_success},
        ),
        _check(
            "verifier_beats_proxy",
            margin >= min_verifier_proxy_margin,
            {
                "verifier_success": verifier_success,
                "proxy_success": proxy_success,
                "margin": margin,
                "minimum": min_verifier_proxy_margin,
            },
        ),
        _check(
            "verifier_beats_rank0_and_random",
            verifier_success > max(rank0_success, random_success),
            {
                "verifier_success": verifier_success,
                "rank0_success": rank0_success,
                "random_expected_success": random_success,
            },
        ),
    ]
    if require_oracle_headroom_over_verifier:
        checks.append(
            _check(
                "oracle_headroom_over_verifier",
                oracle_success > verifier_success,
                {"oracle_success": oracle_success, "verifier_success": verifier_success},
            )
        )

    return {
        "passed": all(check["passed"] for check in checks),
        "summary": {
            "cases": cases,
            "tasks": tasks,
            "rank0_success": rank0_success,
            "random_expected_success": random_success,
            "proxy_success": proxy_success,
            "verifier_success": verifier_success,
            "oracle_success": oracle_success,
            "verifier_proxy_margin": margin,
            "verifier_selector": verifier_selector,
            "proxy_selector": proxy_selector,
        },
        "checks": checks,
    }


def render_markdown(result: dict[str, Any], *, title: str = "World-Model Diagnostic Readiness Gate") -> str:
    summary = result["summary"]
    lines = [
        f"# {title}",
        "",
        f"- passed: `{str(result['passed']).lower()}`",
        f"- cases: {summary['cases']}",
        f"- tasks: {len(summary['tasks'])}",
        f"- verifier selector: `{summary['verifier_selector']}`",
        f"- proxy selector: `{summary['proxy_selector']}`",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in result["checks"]:
        detail = check["detail"]
        if "minimum" in detail:
            text = f"{detail.get('margin', '-')} / min {detail['minimum']}"
        elif "covered_cases" in detail:
            text = f"{detail['covered_cases']}/{detail['cases']} cases"
        else:
            text = ", ".join(f"{key}={value}" for key, value in detail.items()) or "-"
        lines.append(f"| `{check['name']}` | {'pass' if check['passed'] else 'fail'} | {text} |")
    lines.extend(
        [
            "",
            "| Selector | Success |",
            "| --- | ---: |",
            f"| Rank0 | {summary['rank0_success']:.1f} |",
            f"| Random expected | {summary['random_expected_success']:.1f} |",
            f"| Proxy | {summary['proxy_success']:.1f} |",
            f"| Verifier | {summary['verifier_success']:.1f} |",
            f"| Oracle | {summary['oracle_success']:.1f} |",
            "",
            "Interpretation:",
            "",
            "- Passing this gate means a diagnostic benchmark can be proposed for the ICLR evidence registry.",
            "- The manifest gate alone is not enough; the verifier must beat the visual/model-score proxy with full coverage.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-gate-json", type=Path, required=True)
    parser.add_argument("--selector-table-json", type=Path, required=True)
    parser.add_argument("--verifier-selector", required=True)
    parser.add_argument("--proxy-selector", default="planner_or_model_score")
    parser.add_argument("--min-verifier-proxy-margin", type=float, default=1.0)
    parser.add_argument("--require-oracle-headroom-over-verifier", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    result = evaluate_readiness(
        diagnostic_gate=_load_json(args.diagnostic_gate_json),
        selector_table=_load_json(args.selector_table_json),
        verifier_selector=args.verifier_selector,
        proxy_selector=args.proxy_selector,
        min_verifier_proxy_margin=args.min_verifier_proxy_margin,
        require_oracle_headroom_over_verifier=args.require_oracle_headroom_over_verifier,
    )
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
