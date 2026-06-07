from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]


REQUIRED_FIELDS = {
    "case_id",
    "candidate_id",
    "task",
    "instruction",
}


def read_jsonl(path: str | Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[JsonDict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, rows: Iterable[JsonDict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_row(row: JsonDict) -> None:
    missing = sorted(REQUIRED_FIELDS - set(row))
    if missing:
        raise ValueError(f"manifest row missing required fields: {missing}")
    if not row.get("rollout_video") and not row.get("frame_paths") and not row.get("frames_dir"):
        raise ValueError(
            "manifest row must include one of rollout_video, frame_paths, or frames_dir"
        )


def validate_manifest(path: str | Path) -> list[JsonDict]:
    rows = read_jsonl(path)
    for row in rows:
        validate_row(row)
    return rows


def group_by_case(rows: Iterable[JsonDict]) -> dict[str, list[JsonDict]]:
    grouped: dict[str, list[JsonDict]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), []).append(row)
    return grouped
