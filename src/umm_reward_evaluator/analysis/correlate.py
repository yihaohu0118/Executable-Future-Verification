from __future__ import annotations

import argparse
import math
from statistics import mean
from typing import Any

from umm_reward_evaluator.manifest import read_jsonl


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (den_x * den_y) if den_x and den_y else float("nan")


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rank(xs), rank(ys))


def auroc(scores: list[float], labels: list[bool]) -> float:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def score_value(row: dict[str, Any], score_key: str) -> float | None:
    if score_key in row:
        return float(row[score_key])
    vlm = row.get("vlm") or {}
    if score_key in vlm:
        return float(vlm[score_key])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlate evaluator scores with oracle labels.")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--score-key", default="overall_score")
    args = parser.parse_args()

    rows = read_jsonl(args.scores)
    reward_scores: list[float] = []
    rewards: list[float] = []
    success_scores: list[float] = []
    labels: list[bool] = []
    for row in rows:
        score = score_value(row, args.score_key)
        if score is None:
            continue
        if row.get("oracle_reward") is not None:
            reward_scores.append(score)
            rewards.append(float(row["oracle_reward"]))
        if row.get("oracle_success") is not None:
            success_scores.append(score)
            labels.append(bool(row["oracle_success"]))

    print(f"rows_with_reward={len(rewards)}")
    if rewards:
        print(f"spearman_{args.score_key}_vs_oracle_reward={spearman(reward_scores, rewards):.4f}")
        print(f"pearson_{args.score_key}_vs_oracle_reward={pearson(reward_scores, rewards):.4f}")

    if labels:
        print(f"rows_with_success={len(labels)}")
        print(f"auroc_{args.score_key}_vs_oracle_success={auroc(success_scores, labels):.4f}")


if __name__ == "__main__":
    main()
