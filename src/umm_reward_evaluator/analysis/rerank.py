from __future__ import annotations

import argparse
from statistics import mean

from umm_reward_evaluator.manifest import group_by_case, read_jsonl


def get_score(row: dict, key: str) -> float:
    if key in row:
        return float(row[key])
    return float((row.get("vlm") or {}).get(key, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate candidate reranking by predicted score.")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--score-key", default="overall_score")
    args = parser.parse_args()

    rows = [r for r in read_jsonl(args.scores) if not r.get("is_hard_negative")]
    grouped = group_by_case(rows)
    selected_rewards = []
    oracle_best_rewards = []
    random_first_rewards = []
    selected_success = []
    oracle_best_success = []

    for case_rows in grouped.values():
        valid = [r for r in case_rows if r.get("oracle_reward") is not None]
        if not valid:
            continue
        chosen = max(valid, key=lambda r: get_score(r, args.score_key))
        oracle = max(valid, key=lambda r: float(r["oracle_reward"]))
        first = valid[0]
        selected_rewards.append(float(chosen["oracle_reward"]))
        oracle_best_rewards.append(float(oracle["oracle_reward"]))
        random_first_rewards.append(float(first["oracle_reward"]))
        if chosen.get("oracle_success") is not None:
            selected_success.append(bool(chosen["oracle_success"]))
            oracle_best_success.append(bool(oracle["oracle_success"]))

    print(f"cases={len(selected_rewards)}")
    if selected_rewards:
        print(f"mean_selected_reward={mean(selected_rewards):.4f}")
        print(f"mean_first_candidate_reward={mean(random_first_rewards):.4f}")
        print(f"mean_oracle_best_reward={mean(oracle_best_rewards):.4f}")
    if selected_success:
        print(f"selected_success_rate={mean(selected_success):.4f}")
        print(f"oracle_best_success_rate={mean(oracle_best_success):.4f}")


if __name__ == "__main__":
    main()
