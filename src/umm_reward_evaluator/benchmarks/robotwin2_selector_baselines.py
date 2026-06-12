"""Pure-numpy selector baselines for RoboTwin2 executable-future manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from umm_reward_evaluator.benchmarks.common import load_jsonl, oracle_key


HEURISTICS = (
    "energy_mean_max",
    "energy_mean_min",
    "energy_sum_max",
    "abs_mean_max",
    "smoothness_max",
    "smoothness_min",
    "length_max",
    "length_min",
)

PROTOTYPE_FEATURES = (
    "action_distribution",
    "state_distribution",
    "object_distribution",
    "gripper_distribution",
    "phase_gripper_distribution",
    "phase_object_distribution",
    "phase_joint_distribution",
    "phase_joint_gripper_distribution",
    "phase_object_joint_gripper_distribution",
)
PROTOTYPE_SCOPES = ("same_task", "all_tasks")
PROTOTYPE_MODES = ("nearest_positive", "nearest_pos_neg", "pos_neg_centroid", "pos_centroid")
TRACE_DISTANCE_FEATURES = (
    "dtw_action",
    "dtw_joint",
    "dtw_object",
    "dtw_gripper",
    "dtw_joint_gripper",
    "dtw_object_joint_gripper",
)
TRACE_DISTANCE_SCOPES = ("same_task", "all_tasks")


def case_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["task_name"]), str(row["case_id"])


def group_cases(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[case_key(row)].append(row)
    return dict(grouped)


def action_array(row: dict[str, Any]) -> np.ndarray:
    actions = np.asarray(row.get("actions") or [], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] == 0:
        return np.zeros((1, 1), dtype=np.float32)
    return np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)


def action_stats(row: dict[str, Any]) -> dict[str, float]:
    actions = action_array(row)
    diff = np.diff(actions, axis=0) if len(actions) > 1 else np.zeros_like(actions)
    return {
        "energy_mean": float(np.mean(np.square(actions))),
        "energy_sum": float(np.sum(np.square(actions))),
        "abs_mean": float(np.mean(np.abs(actions))),
        "smoothness": float(np.mean(np.square(diff))),
        "length": float(len(actions)),
    }


def heuristic_score(row: dict[str, Any], heuristic: str) -> float:
    stats = action_stats(row)
    if heuristic.endswith("_max"):
        return stats[heuristic[: -len("_max")]]
    if heuristic.endswith("_min"):
        return -stats[heuristic[: -len("_min")]]
    raise ValueError(f"unknown heuristic: {heuristic}")


def select_by_score(case_rows: list[dict[str, Any]], scores: list[float]) -> dict[str, Any]:
    return max(
        zip(case_rows, scores, strict=True),
        key=lambda item: (item[1], -int(item[0].get("candidate_rank_by_planner", 999999))),
    )[0]


def max_tie_expected_success(case_rows: list[dict[str, Any]], scores: list[float]) -> float:
    best = max(scores)
    tied = [row for row, score in zip(case_rows, scores, strict=True) if np.isclose(score, best, rtol=1e-6, atol=1e-8)]
    return float(np.mean([1.0 if row["oracle_success"] else 0.0 for row in tied])) if tied else 0.0


def summarize_selections(
    rows: list[dict[str, Any]],
    *,
    selector_name: str,
    selected_by_case: dict[tuple[str, str], dict[str, Any]],
    tie_expected_by_case: dict[tuple[str, str], float] | None = None,
) -> dict[str, Any]:
    grouped = group_cases(rows)
    by_task: dict[str, dict[str, Any]] = {}
    choice_counts: dict[str, Counter[str]] = defaultdict(Counter)
    selections: list[dict[str, Any]] = []
    for key, case_rows in sorted(grouped.items()):
        task, case_id = key
        rank0 = min(case_rows, key=lambda row: int(row.get("candidate_rank_by_planner", 999999)))
        oracle = max(case_rows, key=oracle_key)
        selected = selected_by_case[key]
        task_summary = by_task.setdefault(
            task,
            {
                "cases": 0,
                "rank0_success": 0,
                "selector_success": 0,
                "oracle_success": 0,
                "selector_oracle_match": 0,
                "tie_expected_success": 0.0,
            },
        )
        task_summary["cases"] += 1
        task_summary["rank0_success"] += int(bool(rank0["oracle_success"]))
        task_summary["selector_success"] += int(bool(selected["oracle_success"]))
        task_summary["oracle_success"] += int(bool(oracle["oracle_success"]))
        task_summary["selector_oracle_match"] += int(selected["candidate_id"] == oracle["candidate_id"])
        if tie_expected_by_case is not None:
            task_summary["tie_expected_success"] += float(tie_expected_by_case[key])
        choice_counts[task][str(selected["candidate_id"])] += 1
        selections.append(
            {
                "task_name": task,
                "case_id": case_id,
                "rank0_candidate_id": rank0["candidate_id"],
                "selector_candidate_id": selected["candidate_id"],
                "oracle_candidate_id": oracle["candidate_id"],
                "rank0_success": bool(rank0["oracle_success"]),
                "selector_success": bool(selected["oracle_success"]),
                "oracle_success": bool(oracle["oracle_success"]),
                "tie_expected_success": tie_expected_by_case.get(key) if tie_expected_by_case is not None else None,
            }
        )

    overall = {
        "cases": 0,
        "rank0_success": 0,
        "selector_success": 0,
        "oracle_success": 0,
        "selector_oracle_match": 0,
        "tie_expected_success": 0.0,
    }
    for task, summary in by_task.items():
        cases = int(summary["cases"])
        summary["selector_success_rate"] = summary["selector_success"] / cases if cases else 0.0
        summary["oracle_success_rate"] = summary["oracle_success"] / cases if cases else 0.0
        summary["tie_expected_success_rate"] = summary["tie_expected_success"] / cases if cases else None
        summary["choice_counts"] = dict(choice_counts[task])
        for key in overall:
            overall[key] += summary[key]
    cases = int(overall["cases"])
    overall["selector_success_rate"] = overall["selector_success"] / cases if cases else 0.0
    overall["oracle_success_rate"] = overall["oracle_success"] / cases if cases else 0.0
    overall["tie_expected_success_rate"] = overall["tie_expected_success"] / cases if cases else None
    return {"selector": selector_name, "overall": overall, "by_task": by_task, "selections": selections}


def evaluate_rank0(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = {
        key: min(case_rows, key=lambda row: int(row.get("candidate_rank_by_planner", 999999)))
        for key, case_rows in group_cases(rows).items()
    }
    return summarize_selections(rows, selector_name="rank0", selected_by_case=selected)


def evaluate_candidate_id(rows: list[dict[str, Any]], candidate_id: str) -> dict[str, Any]:
    selected = {}
    for key, case_rows in group_cases(rows).items():
        matches = [row for row in case_rows if row["candidate_id"] == candidate_id]
        selected[key] = matches[0] if matches else min(case_rows, key=lambda row: int(row["candidate_rank_by_planner"]))
    return summarize_selections(rows, selector_name=f"candidate_id:{candidate_id}", selected_by_case=selected)


def evaluate_random_expected(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = group_cases(rows)
    selected = {key: case_rows[0] for key, case_rows in grouped.items()}
    expected = {
        key: float(np.mean([1.0 if row["oracle_success"] else 0.0 for row in case_rows]))
        for key, case_rows in grouped.items()
    }
    return summarize_selections(rows, selector_name="random_expected", selected_by_case=selected, tie_expected_by_case=expected)


def evaluate_heuristic(rows: list[dict[str, Any]], heuristic: str) -> dict[str, Any]:
    selected = {}
    tie_expected = {}
    for key, case_rows in group_cases(rows).items():
        scores = [heuristic_score(row, heuristic) for row in case_rows]
        selected[key] = select_by_score(case_rows, scores)
        tie_expected[key] = max_tie_expected_success(case_rows, scores)
    return summarize_selections(rows, selector_name=f"heuristic:{heuristic}", selected_by_case=selected, tie_expected_by_case=tie_expected)


def state_trace(row: dict[str, Any]) -> list[dict[str, Any]]:
    trace = (row.get("metadata") or {}).get("state_trace", [])
    return trace if isinstance(trace, list) else []


def padded(arr: Any, dim: int) -> np.ndarray:
    value = np.asarray(arr if arr is not None else [], dtype=np.float32).reshape(-1)
    out = np.zeros((dim,), dtype=np.float32)
    keep = min(dim, int(value.size))
    if keep:
        out[:keep] = np.nan_to_num(value[:keep], nan=0.0, posinf=0.0, neginf=0.0)
    return out


def state_dims(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, int]:
    dims = {key: 1 for key in keys}
    for row in rows:
        for snapshot in state_trace(row):
            if not isinstance(snapshot, dict):
                continue
            for key in keys:
                dims[key] = max(dims[key], int(np.asarray(snapshot.get(key, [])).reshape(-1).size))
    return dims


def trace_distribution_features(trace: list[dict[str, Any]], keys: list[str], dims: dict[str, int]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for key in keys:
        dim = dims[key]
        if trace:
            x = np.stack([padded(snapshot.get(key), dim) if isinstance(snapshot, dict) else np.zeros(dim) for snapshot in trace])
        else:
            x = np.zeros((1, dim), dtype=np.float32)
        parts.extend([x.mean(axis=0), x.std(axis=0), x.min(axis=0), x.max(axis=0)])
    return np.concatenate(parts).astype(np.float32) if parts else np.zeros((1,), dtype=np.float32)


def phase_trace_distribution_features(
    trace: list[dict[str, Any]],
    keys: list[str],
    dims: dict[str, int],
    *,
    num_phases: int = 3,
) -> np.ndarray:
    snapshots = trace if trace else [{}]
    parts: list[np.ndarray] = []
    for indexes in np.array_split(np.arange(len(snapshots)), num_phases):
        phase = [snapshots[int(index)] for index in indexes] if len(indexes) else [snapshots[-1]]
        for key in keys:
            dim = dims[key]
            x = np.stack(
                [
                    padded(snapshot.get(key), dim) if isinstance(snapshot, dict) else np.zeros(dim, dtype=np.float32)
                    for snapshot in phase
                ]
            )
            parts.extend([x.mean(axis=0), x.std(axis=0), x.min(axis=0), x.max(axis=0)])
    for key in keys:
        dim = dims[key]
        first = padded(snapshots[0].get(key), dim) if isinstance(snapshots[0], dict) else np.zeros(dim, dtype=np.float32)
        last = padded(snapshots[-1].get(key), dim) if isinstance(snapshots[-1], dict) else np.zeros(dim, dtype=np.float32)
        parts.append(last - first)
    return np.concatenate(parts).astype(np.float32) if parts else np.zeros((1,), dtype=np.float32)


def feature_vector(row: dict[str, Any], feature_mode: str, *, dims: dict[str, int] | None = None) -> np.ndarray:
    if feature_mode == "action_distribution":
        actions = action_array(row)
        return np.concatenate([actions.mean(axis=0), actions.std(axis=0), actions.min(axis=0), actions.max(axis=0)]).astype(np.float32)
    if feature_mode == "state_distribution":
        keys = ["joint_action_vector", "left_gripper", "right_gripper"]
        return trace_distribution_features(state_trace(row), keys, dims or state_dims([row], keys))
    if feature_mode == "object_distribution":
        keys = ["actor_pose_vector", "actor_pairwise_distances"]
        return trace_distribution_features(state_trace(row), keys, dims or state_dims([row], keys))
    if feature_mode == "gripper_distribution":
        keys = ["left_gripper", "right_gripper"]
        return trace_distribution_features(state_trace(row), keys, dims or state_dims([row], keys))
    if feature_mode == "phase_gripper_distribution":
        keys = ["left_gripper", "right_gripper"]
        return phase_trace_distribution_features(state_trace(row), keys, dims or state_dims([row], keys))
    if feature_mode == "phase_object_distribution":
        keys = ["actor_pose_vector", "actor_pairwise_distances"]
        return phase_trace_distribution_features(state_trace(row), keys, dims or state_dims([row], keys))
    if feature_mode == "phase_joint_distribution":
        keys = ["joint_action_vector"]
        return phase_trace_distribution_features(state_trace(row), keys, dims or state_dims([row], keys))
    if feature_mode == "phase_joint_gripper_distribution":
        keys = ["joint_action_vector", "left_gripper", "right_gripper"]
        return phase_trace_distribution_features(state_trace(row), keys, dims or state_dims([row], keys))
    if feature_mode == "phase_object_joint_gripper_distribution":
        keys = ["actor_pose_vector", "actor_pairwise_distances", "joint_action_vector", "left_gripper", "right_gripper"]
        return phase_trace_distribution_features(state_trace(row), keys, dims or state_dims([row], keys))
    raise ValueError(f"unknown prototype feature mode: {feature_mode}")


def normalize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    return (x_train - mean) / std, (x_test - mean) / std


def prototype_scores(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, mode: str) -> np.ndarray:
    pos = x_train[y_train > 0.5]
    neg = x_train[y_train <= 0.5]
    if pos.size == 0:
        return np.zeros((x_test.shape[0],), dtype=np.float32)
    if mode == "nearest_positive":
        dist = np.linalg.norm(x_test[:, None, :] - pos[None, :, :], axis=2)
        return -dist.min(axis=1)
    if mode == "nearest_pos_neg":
        pos_dist = np.linalg.norm(x_test[:, None, :] - pos[None, :, :], axis=2).min(axis=1)
        if neg.size == 0:
            return -pos_dist
        neg_dist = np.linalg.norm(x_test[:, None, :] - neg[None, :, :], axis=2).min(axis=1)
        return neg_dist - pos_dist
    pos_centroid = pos.mean(axis=0, keepdims=True)
    pos_dist = np.linalg.norm(x_test - pos_centroid, axis=1)
    if mode == "pos_centroid":
        return -pos_dist
    if mode == "pos_neg_centroid":
        if neg.size == 0:
            return -pos_dist
        neg_centroid = neg.mean(axis=0, keepdims=True)
        neg_dist = np.linalg.norm(x_test - neg_centroid, axis=1)
        return neg_dist - pos_dist
    raise ValueError(f"unknown prototype mode: {mode}")


def trace_sequence(row: dict[str, Any], feature_mode: str, *, dims: dict[str, int] | None = None) -> np.ndarray:
    if feature_mode == "dtw_action":
        return action_array(row).astype(np.float32)

    if feature_mode == "dtw_joint":
        keys = ["joint_action_vector"]
    elif feature_mode == "dtw_object":
        keys = ["actor_pose_vector", "actor_pairwise_distances"]
    elif feature_mode == "dtw_gripper":
        keys = ["left_gripper", "right_gripper"]
    elif feature_mode == "dtw_joint_gripper":
        keys = ["joint_action_vector", "left_gripper", "right_gripper"]
    elif feature_mode == "dtw_object_joint_gripper":
        keys = ["actor_pose_vector", "actor_pairwise_distances", "joint_action_vector", "left_gripper", "right_gripper"]
    else:
        raise ValueError(f"unknown trace-distance feature mode: {feature_mode}")

    local_dims = dims or state_dims([row], keys)
    snapshots = state_trace(row) or [{}]
    frames = []
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            snapshot = {}
        frames.append(np.concatenate([padded(snapshot.get(key), local_dims[key]) for key in keys]))
    return np.stack(frames).astype(np.float32) if frames else np.zeros((1, 1), dtype=np.float32)


def normalize_sequences(
    train_sequences: list[np.ndarray],
    test_sequences: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    dim = max([seq.shape[1] for seq in train_sequences + test_sequences] or [1])

    def pad_dim(seq: np.ndarray) -> np.ndarray:
        if seq.shape[1] == dim:
            return seq
        out = np.zeros((seq.shape[0], dim), dtype=np.float32)
        out[:, : seq.shape[1]] = seq
        return out

    train_padded = [pad_dim(seq) for seq in train_sequences]
    test_padded = [pad_dim(seq) for seq in test_sequences]
    reference = np.concatenate(train_padded, axis=0) if train_padded else np.zeros((1, dim), dtype=np.float32)
    mean = reference.mean(axis=0, keepdims=True)
    std = reference.std(axis=0, keepdims=True) + 1e-6
    return [(seq - mean) / std for seq in train_padded], [(seq - mean) / std for seq in test_padded]


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    n, m = int(a.shape[0]), int(b.shape[0])
    prev = np.full((m + 1,), np.inf, dtype=np.float32)
    curr = np.full((m + 1,), np.inf, dtype=np.float32)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr[0] = np.inf
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(ai - b[j - 1]))
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return float(prev[m] / max(n + m, 1))


def trace_distance_scores(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    feature_mode: str,
    dims: dict[str, int] | None = None,
) -> np.ndarray:
    positive_rows = [row for row in train_rows if row["oracle_success"]]
    if not positive_rows:
        return np.zeros((len(test_rows),), dtype=np.float32)
    train_sequences = [trace_sequence(row, feature_mode, dims=dims) for row in train_rows]
    test_sequences = [trace_sequence(row, feature_mode, dims=dims) for row in test_rows]
    train_sequences, test_sequences = normalize_sequences(train_sequences, test_sequences)
    positive_indexes = [index for index, row in enumerate(train_rows) if row["oracle_success"]]
    positive_sequences = [train_sequences[index] for index in positive_indexes]
    scores = []
    for test_sequence in test_sequences:
        best = min(dtw_distance(test_sequence, positive_sequence) for positive_sequence in positive_sequences)
        scores.append(-best)
    return np.asarray(scores, dtype=np.float32)


def evaluate_prototype(rows: list[dict[str, Any]], *, feature_mode: str, scope: str, prototype_mode: str) -> dict[str, Any]:
    grouped = group_cases(rows)
    all_keys = [
        "joint_action_vector",
        "left_gripper",
        "right_gripper",
        "actor_pose_vector",
        "actor_pairwise_distances",
    ]
    dims = state_dims(rows, all_keys)
    selected = {}
    scored_rows: list[dict[str, Any]] = []
    for key, test_rows in sorted(grouped.items()):
        task, _case_id = key
        if scope == "same_task":
            train_rows = [row for other_key, case_rows in grouped.items() if other_key != key and other_key[0] == task for row in case_rows]
        elif scope == "all_tasks":
            train_rows = [row for other_key, case_rows in grouped.items() if other_key != key for row in case_rows]
        else:
            raise ValueError(f"unknown train scope: {scope}")
        x_train = np.stack([feature_vector(row, feature_mode, dims=dims) for row in train_rows]).astype(np.float32)
        x_test = np.stack([feature_vector(row, feature_mode, dims=dims) for row in test_rows]).astype(np.float32)
        x_train, x_test = normalize_train_test(x_train, x_test)
        y_train = np.asarray([1.0 if row["oracle_success"] else 0.0 for row in train_rows], dtype=np.float32)
        scores = prototype_scores(x_train, y_train, x_test, prototype_mode)
        selected[key] = select_by_score(test_rows, scores.tolist())
        for row, score in zip(test_rows, scores, strict=True):
            payload = dict(row)
            payload["robotwin2_prototype_score"] = float(score)
            scored_rows.append(payload)
    result = summarize_selections(
        rows,
        selector_name=f"prototype:{feature_mode}:{scope}:{prototype_mode}",
        selected_by_case=selected,
    )
    result["scored_rows"] = scored_rows
    return result


def evaluate_trace_distance(rows: list[dict[str, Any]], *, feature_mode: str, scope: str) -> dict[str, Any]:
    grouped = group_cases(rows)
    dims = state_dims(
        rows,
        [
            "joint_action_vector",
            "left_gripper",
            "right_gripper",
            "actor_pose_vector",
            "actor_pairwise_distances",
        ],
    )
    selected = {}
    scored_rows: list[dict[str, Any]] = []
    for key, test_rows in sorted(grouped.items()):
        task, _case_id = key
        if scope == "same_task":
            train_rows = [row for other_key, case_rows in grouped.items() if other_key != key and other_key[0] == task for row in case_rows]
        elif scope == "all_tasks":
            train_rows = [row for other_key, case_rows in grouped.items() if other_key != key for row in case_rows]
        else:
            raise ValueError(f"unknown train scope: {scope}")
        scores = trace_distance_scores(train_rows, test_rows, feature_mode=feature_mode, dims=dims)
        selected[key] = select_by_score(test_rows, scores.tolist())
        for row, score in zip(test_rows, scores, strict=True):
            payload = dict(row)
            payload["robotwin2_trace_distance_score"] = float(score)
            scored_rows.append(payload)
    result = summarize_selections(
        rows,
        selector_name=f"trace_distance:{feature_mode}:{scope}:nearest_positive",
        selected_by_case=selected,
    )
    result["scored_rows"] = scored_rows
    return result


def strip_large(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in {"selections", "scored_rows"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prototype-feature", action="append", choices=list(PROTOTYPE_FEATURES))
    parser.add_argument("--prototype-scope", action="append", choices=list(PROTOTYPE_SCOPES))
    parser.add_argument("--prototype-mode", default="nearest_positive", choices=list(PROTOTYPE_MODES))
    parser.add_argument("--trace-distance-feature", action="append", choices=list(TRACE_DISTANCE_FEATURES))
    parser.add_argument("--trace-distance-scope", action="append", choices=list(TRACE_DISTANCE_SCOPES))
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = [evaluate_rank0(rows), evaluate_random_expected(rows), evaluate_candidate_id(rows, "full_gripper_aware")]
    results.extend(evaluate_heuristic(rows, heuristic) for heuristic in HEURISTICS)

    prototype_features = args.prototype_feature or list(PROTOTYPE_FEATURES)
    prototype_scopes = args.prototype_scope or list(PROTOTYPE_SCOPES)
    for feature_mode in prototype_features:
        for scope in prototype_scopes:
            results.append(
                evaluate_prototype(rows, feature_mode=feature_mode, scope=scope, prototype_mode=args.prototype_mode)
            )
    distance_features = args.trace_distance_feature or list(TRACE_DISTANCE_FEATURES)
    distance_scopes = args.trace_distance_scope or list(TRACE_DISTANCE_SCOPES)
    for feature_mode in distance_features:
        for scope in distance_scopes:
            results.append(evaluate_trace_distance(rows, feature_mode=feature_mode, scope=scope))

    summary = {"manifest": str(args.manifest), "selectors": [strip_large(result) for result in results]}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for result in results:
        safe_name = result["selector"].replace(":", "__").replace("/", "_")
        (args.output_dir / f"{safe_name}_selections.json").write_text(
            json.dumps(result["selections"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if "scored_rows" in result:
            with (args.output_dir / f"{safe_name}_scored.jsonl").open("w", encoding="utf-8") as f:
                for row in result["scored_rows"]:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
