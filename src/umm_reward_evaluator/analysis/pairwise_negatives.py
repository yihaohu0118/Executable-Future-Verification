from __future__ import annotations

import argparse

from umm_reward_evaluator.manifest import read_jsonl


def get_score(row: dict, key: str) -> float:
    if key in row:
        return float(row[key])
    return float((row.get("vlm") or {}).get(key, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure whether originals score above hard negatives.")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--score-key", default="overall_score")
    args = parser.parse_args()

    rows = read_jsonl(args.scores)
    originals = {
        (r.get("case_id"), r.get("candidate_id")): r
        for r in rows
        if not r.get("negative_type")
    }
    total = 0
    correct = 0
    by_type: dict[str, list[bool]] = {}
    for row in rows:
        neg_type = row.get("negative_type")
        if not neg_type:
            continue
        key = (row.get("case_id"), row.get("source_candidate_id"))
        original = originals.get(key)
        if original is None:
            continue
        ok = get_score(original, args.score_key) > get_score(row, args.score_key)
        total += 1
        correct += int(ok)
        by_type.setdefault(str(neg_type), []).append(ok)

    print(f"pairs={total}")
    print(f"pairwise_accuracy={correct / total:.4f}" if total else "pairwise_accuracy=nan")
    for neg_type, oks in sorted(by_type.items()):
        print(f"{neg_type}_accuracy={sum(oks) / len(oks):.4f} n={len(oks)}")


if __name__ == "__main__":
    main()
