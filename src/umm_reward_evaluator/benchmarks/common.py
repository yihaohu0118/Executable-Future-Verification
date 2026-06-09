"""Common candidate-selection schema for external robot benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import json


@dataclass(frozen=True)
class CandidateRow:
    benchmark: str
    suite: str
    task_name: str
    case_id: str
    candidate_id: str
    candidate_rank_by_planner: int
    rollout_video_path: str
    rollout_video_layout: str
    actions: list[list[float]]
    oracle_success: bool
    instruction: str | None = None
    init_obs_path: str | None = None
    goal_obs_path: str | None = None
    planner_score: float | None = None
    oracle_return: float | None = None
    oracle_progress: float | None = None
    oracle_state_dist: float | None = None
    oracle_best_candidate_id: str | None = None
    metadata: dict[str, Any] | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def write_jsonl(path: str | Path, rows: Iterable[CandidateRow | dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            payload = asdict(row) if isinstance(row, CandidateRow) else row
            f.write(json.dumps(payload, sort_keys=True) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def oracle_key(row: dict[str, Any]) -> tuple[float, float, float]:
    success = 1.0 if row.get("oracle_success") else 0.0
    progress = row.get("oracle_progress")
    ret = row.get("oracle_return")
    state_dist = row.get("oracle_state_dist")
    if progress is None:
        progress = -float(state_dist) if state_dist is not None else 0.0
    if ret is None:
        ret = 0.0
    return success, float(progress), float(ret)


def annotate_oracle_best(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), []).append(dict(row))

    annotated: list[dict[str, Any]] = []
    for case_rows in grouped.values():
        best = max(case_rows, key=oracle_key)
        best_id = best["candidate_id"]
        for row in case_rows:
            row["oracle_best_candidate_id"] = best_id
            annotated.append(row)
    return annotated


def summarize_headroom(rows: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), []).append(row)

    cases = 0
    rank0_success = 0
    oracle_success = 0
    oracle_better = 0
    rank0_oracle_match = 0
    for case_rows in grouped.values():
        rank0 = min(case_rows, key=lambda row: row["candidate_rank_by_planner"])
        oracle = max(case_rows, key=oracle_key)
        cases += 1
        rank0_success += int(bool(rank0.get("oracle_success")))
        oracle_success += int(bool(oracle.get("oracle_success")))
        oracle_better += int(oracle_key(oracle) > oracle_key(rank0))
        rank0_oracle_match += int(rank0["candidate_id"] == oracle["candidate_id"])

    return {
        "cases": cases,
        "rank0_success": rank0_success,
        "oracle_success": oracle_success,
        "oracle_better": oracle_better,
        "rank0_oracle_match": rank0_oracle_match,
        "rank0_success_rate": rank0_success / cases if cases else 0.0,
        "oracle_success_rate": oracle_success / cases if cases else 0.0,
    }

