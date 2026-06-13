"""Few-shot verifier scoring for world-model diagnostic manifests."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from umm_reward_evaluator.benchmarks.common import case_group_key, load_jsonl, oracle_key, write_jsonl
from umm_reward_evaluator.benchmarks.robotwin2_selector_baselines import action_stats, linear_probe_scores, prototype_scores


FORBIDDEN_FEATURE_TOKENS = (
    "oracle",
    "success",
    "label",
    "judgment",
    "verdict",
    "human_score",
    "annotation",
)


def _nested_get(payload: dict[str, Any], key: str) -> Any:
    value: Any = payload
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _set_nested(payload: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    target = payload
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            child = {}
            target[part] = child
        target = child
    target[parts[-1]] = value


def _validate_feature_keys(feature_keys: list[str], *, allow_label_like_features: bool) -> None:
    if allow_label_like_features:
        return
    for key in feature_keys:
        lowered = key.lower()
        if any(token in lowered for token in FORBIDDEN_FEATURE_TOKENS):
            raise ValueError(
                f"feature key {key!r} looks label-derived; pass --allow-label-like-features only for debugging"
            )


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def _category_maps(rows: list[dict[str, Any]], categorical_keys: list[str]) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for key in categorical_keys:
        values = sorted({str(_nested_get({"metadata": row.get("metadata") or {}, **row}, key)) for row in rows})
        maps[key] = {value: index for index, value in enumerate(values)}
    return maps


def feature_vector(
    row: dict[str, Any],
    *,
    feature_keys: list[str],
    categorical_keys: list[str],
    categorical_maps: dict[str, dict[str, int]],
    include_action_stats: bool,
    include_planner_score: bool,
    include_rank: bool,
) -> np.ndarray:
    parts: list[float] = []
    if include_action_stats:
        stats = action_stats(row)
        parts.extend([stats["energy_mean"], stats["energy_sum"], stats["abs_mean"], stats["smoothness"], stats["length"]])
    if include_planner_score:
        parts.append(_as_float(row.get("planner_score", row.get("model_score"))))
    if include_rank:
        parts.append(_as_float(row.get("candidate_rank_by_planner")))
    payload = {"metadata": row.get("metadata") or {}, **row}
    for key in feature_keys:
        parts.append(_as_float(_nested_get(payload, key)))
    for key in categorical_keys:
        value = str(_nested_get(payload, key))
        mapping = categorical_maps[key]
        one_hot = [0.0] * len(mapping)
        if value in mapping:
            one_hot[mapping[value]] = 1.0
        parts.extend(one_hot)
    if not parts:
        parts.append(0.0)
    return np.asarray(parts, dtype=np.float32)


def _candidate_cases(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[case_group_key(row)].append(row)
    return dict(grouped)


def _score_rows(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    method: str,
    feature_keys: list[str],
    categorical_keys: list[str],
    categorical_maps: dict[str, dict[str, int]],
    include_action_stats: bool,
    include_planner_score: bool,
    include_rank: bool,
    l2: float,
) -> np.ndarray:
    if not train_rows:
        return np.zeros((len(test_rows),), dtype=np.float32)
    x_train = np.stack(
        [
            feature_vector(
                row,
                feature_keys=feature_keys,
                categorical_keys=categorical_keys,
                categorical_maps=categorical_maps,
                include_action_stats=include_action_stats,
                include_planner_score=include_planner_score,
                include_rank=include_rank,
            )
            for row in train_rows
        ]
    )
    x_test = np.stack(
        [
            feature_vector(
                row,
                feature_keys=feature_keys,
                categorical_keys=categorical_keys,
                categorical_maps=categorical_maps,
                include_action_stats=include_action_stats,
                include_planner_score=include_planner_score,
                include_rank=include_rank,
            )
            for row in test_rows
        ]
    )
    y_train = np.asarray([1.0 if row.get("oracle_success") else 0.0 for row in train_rows], dtype=np.float32)
    if len(np.unique(y_train > 0.5)) < 2:
        return np.zeros((len(test_rows),), dtype=np.float32)
    if method == "ridge":
        return linear_probe_scores(x_train, y_train, x_test, l2=l2)
    if method == "nearest_pos_neg":
        return prototype_scores(x_train, y_train, x_test, "nearest_pos_neg")
    if method == "pos_neg_centroid":
        return prototype_scores(x_train, y_train, x_test, "pos_neg_centroid")
    raise ValueError(f"unknown method: {method}")


def calibrate_scores(
    rows: list[dict[str, Any]],
    *,
    score_key: str,
    feature_keys: list[str],
    categorical_keys: list[str],
    method: str = "ridge",
    include_action_stats: bool = True,
    include_planner_score: bool = False,
    include_rank: bool = False,
    same_task_only: bool = False,
    l2: float = 1.0,
    allow_label_like_features: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_feature_keys(feature_keys, allow_label_like_features=allow_label_like_features)
    grouped = _candidate_cases(rows)
    categorical_maps = _category_maps(rows, categorical_keys)
    scored_rows: list[dict[str, Any]] = []
    selected_success = 0
    rank0_success = 0
    oracle_success = 0
    covered_cases = 0
    fallback_cases = 0
    by_case: list[dict[str, Any]] = []

    for case_key, case_rows in sorted(grouped.items()):
        task, case_id = case_key
        train_rows = [row for key, grouped_rows in grouped.items() if key != case_key for row in grouped_rows]
        if same_task_only:
            same_task_train = [row for row in train_rows if str(row.get("task_name")) == task]
            if same_task_train:
                train_rows = same_task_train
            else:
                fallback_cases += 1
        scores = _score_rows(
            train_rows,
            case_rows,
            method=method,
            feature_keys=feature_keys,
            categorical_keys=categorical_keys,
            categorical_maps=categorical_maps,
            include_action_stats=include_action_stats,
            include_planner_score=include_planner_score,
            include_rank=include_rank,
            l2=l2,
        )
        rank0 = min(case_rows, key=lambda row: int(row.get("candidate_rank_by_planner", 999999)))
        oracle = max(case_rows, key=oracle_key)
        selected = max(
            zip(case_rows, scores),
            key=lambda item: (float(item[1]), -int(item[0].get("candidate_rank_by_planner", 999999))),
        )[0]
        rank0_success += int(bool(rank0.get("oracle_success")))
        oracle_success += int(bool(oracle.get("oracle_success")))
        selected_success += int(bool(selected.get("oracle_success")))
        covered_cases += 1
        by_case.append(
            {
                "task_name": task,
                "case_id": case_id,
                "train_rows": len(train_rows),
                "rank0_candidate_id": rank0.get("candidate_id"),
                "selected_candidate_id": selected.get("candidate_id"),
                "oracle_candidate_id": oracle.get("candidate_id"),
                "rank0_success": bool(rank0.get("oracle_success")),
                "selected_success": bool(selected.get("oracle_success")),
                "oracle_success": bool(oracle.get("oracle_success")),
            }
        )
        for row, score in zip(case_rows, scores):
            updated = dict(row)
            metadata = dict(updated.get("metadata") or {})
            updated["metadata"] = metadata
            _set_nested(updated, score_key, float(score))
            scored_rows.append(updated)

    summary = {
        "rows": len(rows),
        "cases": len(grouped),
        "covered_cases": covered_cases,
        "rank0_success": rank0_success,
        "oracle_success": oracle_success,
        "verifier_success": selected_success,
        "fallback_cases": fallback_cases,
        "score_key": score_key,
        "method": method,
        "feature_keys": feature_keys,
        "categorical_keys": categorical_keys,
        "include_action_stats": include_action_stats,
        "include_planner_score": include_planner_score,
        "include_rank": include_rank,
        "same_task_only": same_task_only,
        "by_case": by_case,
    }
    return scored_rows, summary


def render_markdown(summary: dict[str, Any], *, title: str = "World-Model Diagnostic Calibrated Verifier") -> str:
    lines = [
        f"# {title}",
        "",
        f"- cases: {summary['cases']}",
        f"- method: `{summary['method']}`",
        f"- score key: `{summary['score_key']}`",
        f"- feature keys: {', '.join(summary['feature_keys']) or '-'}",
        f"- categorical keys: {', '.join(summary['categorical_keys']) or '-'}",
        "",
        "| Selector | Success |",
        "| --- | ---: |",
        f"| Rank0 | {summary['rank0_success']}/{summary['cases']} |",
        f"| Calibrated verifier | {summary['verifier_success']}/{summary['cases']} |",
        f"| Oracle | {summary['oracle_success']}/{summary['cases']} |",
        "",
        "| Task | Case | Train rows | Rank0 | Selected | Oracle | Selected success |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for row in summary["by_case"]:
        lines.append(
            "| {task} | {case} | {train} | {rank0} | {selected} | {oracle} | {success} |".format(
                task=row["task_name"],
                case=row["case_id"],
                train=row["train_rows"],
                rank0=row["rank0_candidate_id"],
                selected=row["selected_candidate_id"],
                oracle=row["oracle_candidate_id"],
                success="yes" if row["selected_success"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Scores are fit leave-one-case-out, so a candidate is not scored using labels from its own case.",
            "- This is a diagnostic verifier score generator; final evidence still requires the diagnostic gate and readiness gate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-summary-json", type=Path)
    parser.add_argument("--output-summary-md", type=Path)
    parser.add_argument("--score-key", default="metadata.efv_score")
    parser.add_argument("--feature-key", action="append", default=[])
    parser.add_argument("--categorical-key", action="append", default=[])
    parser.add_argument("--method", choices=["ridge", "nearest_pos_neg", "pos_neg_centroid"], default="ridge")
    parser.add_argument("--no-action-stats", action="store_true")
    parser.add_argument("--include-planner-score", action="store_true")
    parser.add_argument("--include-rank", action="store_true")
    parser.add_argument("--same-task-only", action="store_true")
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--allow-label-like-features", action="store_true")
    args = parser.parse_args()

    rows, summary = calibrate_scores(
        load_jsonl(args.manifest),
        score_key=args.score_key,
        feature_keys=list(args.feature_key),
        categorical_keys=list(args.categorical_key),
        method=args.method,
        include_action_stats=not args.no_action_stats,
        include_planner_score=args.include_planner_score,
        include_rank=args.include_rank,
        same_task_only=args.same_task_only,
        l2=args.l2,
        allow_label_like_features=args.allow_label_like_features,
    )
    write_jsonl(args.output_manifest, rows)
    if args.output_summary_json:
        args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(summary)
    if args.output_summary_md:
        args.output_summary_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
