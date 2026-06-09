from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from umm_reward_evaluator.analysis.compare_world_critic_selectors import load_selector, parse_selector, row_key
from umm_reward_evaluator.manifest import read_jsonl, write_jsonl
from umm_reward_evaluator.training.ggp_umm_pairwise_reward import make_folds
from umm_reward_evaluator.training.umm_latent_dynamics_reward import evaluate_scores, merge_dynamics_metrics


FEATURE_FIELDS = [
    "dino_ac_score",
    "dino_ac_goal",
    "dino_ac_consistency",
    "lance_sequence_world_score",
    "lance_raw_action_sequence_world_score",
    "lance_rollout_horizon_regularizer_score",
    "lance_rollout_horizon_margin",
    "lance_rollout_horizon_true_similarity",
    "lance_rollout_horizon_accuracy",
]


def cem_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if int(row.get("candidate_rank_by_planner", 0)) >= 0
        and not str(row.get("candidate_id", "")).endswith("expert")
    ]


def zstats(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std())
    return float(arr.mean()), std if std > 1e-8 else 1.0


def score_stats(rows: list[dict[str, Any]], scores: dict[tuple[str, str], float]) -> tuple[float, float, float]:
    values = [scores[row_key(row)] for row in rows]
    score_mean, score_std = zstats(values)
    order = sorted(values, reverse=True)
    margin = (order[0] - order[1]) / score_std if len(order) > 1 else 0.0
    return score_mean, score_std, float(margin)


def selected_features(
    selected: dict[str, Any],
    rows: list[dict[str, Any]],
    scores: dict[tuple[str, str], float],
) -> list[float]:
    score_mean, score_std, margin = score_stats(rows, scores)
    score_z = (scores[row_key(selected)] - score_mean) / score_std
    rank = float(selected.get("candidate_rank_by_planner", 0) or 0.0)
    planner_loss = float(selected.get("planner_loss", 0.0) or 0.0)
    rank_norm = rank / max(1.0, len(rows) - 1)
    dense = [float(selected.get(field, 0.0) or 0.0) for field in FEATURE_FIELDS]
    return [float(score_z), float(margin), float(rank_norm), planner_loss] + dense


def case_features(
    case_rows: list[dict[str, Any]],
    dino_scores: dict[tuple[str, str], float],
    umm_scores: dict[tuple[str, str], float],
) -> tuple[np.ndarray, dict[str, Any]]:
    rows = cem_rows(case_rows)
    dino_selected = max(rows, key=lambda row: dino_scores[row_key(row)])
    umm_selected = max(rows, key=lambda row: umm_scores[row_key(row)])
    rank0 = min(rows, key=lambda row: int(row.get("candidate_rank_by_planner", 999)))
    oracle = min(rows, key=lambda row: float(row["oracle_state_dist"]))
    dino_vec = selected_features(dino_selected, rows, dino_scores)
    umm_vec = selected_features(umm_selected, rows, umm_scores)
    same_candidate = float(str(dino_selected["candidate_id"]) == str(umm_selected["candidate_id"]))
    dino_dist = float(dino_selected["oracle_state_dist"])
    umm_dist = float(umm_selected["oracle_state_dist"])
    rank0_dist = float(rank0["oracle_state_dist"])
    oracle_id = str(oracle["candidate_id"])
    record = {
        "case_id": str(case_rows[0]["case_id"]),
        "dino_candidate_id": str(dino_selected["candidate_id"]),
        "umm_candidate_id": str(umm_selected["candidate_id"]),
        "oracle_candidate_id": oracle_id,
        "rank0_candidate_id": str(rank0["candidate_id"]),
        "dino_dist": dino_dist,
        "umm_dist": umm_dist,
        "rank0_dist": rank0_dist,
        "oracle_dist": float(oracle["oracle_state_dist"]),
        "dino_oracle": str(dino_selected["candidate_id"]) == oracle_id,
        "umm_oracle": str(umm_selected["candidate_id"]) == oracle_id,
        "same_candidate": bool(same_candidate),
        "umm_better": umm_dist < dino_dist,
        "dino_better": dino_dist < umm_dist,
        "umm_rescues_oracle": str(umm_selected["candidate_id"]) == oracle_id
        and str(dino_selected["candidate_id"]) != oracle_id,
        "dino_loses_oracle": str(dino_selected["candidate_id"]) == oracle_id
        and str(umm_selected["candidate_id"]) != oracle_id,
        "umm_improves_rank0": umm_dist < rank0_dist,
        "dino_improves_rank0": dino_dist < rank0_dist,
    }
    diff = [u - d for d, u in zip(dino_vec, umm_vec)]
    features = np.asarray(dino_vec + umm_vec + diff + [same_candidate], dtype=np.float32)
    return features, record


def label_from_record(record: dict[str, Any], objective: str) -> int:
    if objective == "lower_dist":
        return int(float(record["umm_dist"]) < float(record["dino_dist"]))
    if objective == "oracle_rescue":
        if record["umm_rescues_oracle"]:
            return 1
        if record["dino_loses_oracle"]:
            return 0
        return int(float(record["umm_dist"]) < float(record["dino_dist"]))
    raise ValueError(f"Unknown objective={objective!r}")


class BinaryGate(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            self.net = torch.nn.Linear(input_dim, 2)
        else:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_dim, 2),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_gate(
    features: list[np.ndarray],
    labels: list[int],
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> BinaryGate:
    torch.manual_seed(seed)
    model = BinaryGate(features[0].shape[0], hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    x = torch.from_numpy(np.stack(features)).float()
    y = torch.tensor(labels, dtype=torch.long)
    for _ in range(epochs):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


@torch.inference_mode()
def predict(model: BinaryGate, features: np.ndarray) -> tuple[int, list[float]]:
    logits = model(torch.from_numpy(features[None, :]).float())[0]
    return int(torch.argmax(logits).item()), [float(v) for v in logits.detach().cpu().tolist()]


def write_report(summary: dict[str, Any], output: str) -> None:
    cv = summary["crossval"]
    n_cases = len(cv.get("selected_cem", [])) or len(summary.get("case_predictions", []))
    lines = [
        "# DINO-Failure-Aware UMM World-Model Gate",
        "",
        "## Bottom Line",
        "",
        "This trains a fold-wise binary gate that decides whether a case should stay with the static DINO selector or be overridden by a UMM world-model selector. The gate uses only non-oracle confidence, planner, and action-conditioned rollout features at test time.",
        "",
        "## Result",
        "",
        "| Objective | Mean dist | Better | Oracle match | Pair acc | Expert acc | Gate acc | Overrides |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| {obj} | {mean:.4f} | {better}/{n_cases} | {oracle}/{n_cases} | {pair:.4f} | {expert:.4f} | {gate:.4f} | {overrides}/{n_cases} |".format(
            obj=summary["objective"],
            n_cases=n_cases,
            mean=cv["cem_selected_state_dist"]["mean"],
            better=cv["cem_better_than_rank0_count"],
            oracle=cv["cem_matches_oracle_best_count"],
            pair=cv["pair_label_accuracy"],
            expert=cv["oracle_expert_pair_accuracy"],
            gate=summary["gate_accuracy"],
            overrides=summary["override_count"],
        ),
        "",
        "## Case Decisions",
        "",
        "| Fold | Case | Pred | Target | DINO dist | UMM dist | Oracle dist | Rescue | Loss |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["case_predictions"]:
        lines.append(
            "| {fold} | {case} | {pred} | {target} | {dino:.4f} | {umm:.4f} | {oracle:.4f} | {rescue} | {loss} |".format(
                fold=row["fold_idx"],
                case=row["case_id"],
                pred="UMM" if row["predicted_override"] else "DINO",
                target="UMM" if row["target_override"] else "DINO",
                dino=row["dino_dist"],
                umm=row["umm_dist"],
                oracle=row["oracle_dist"],
                rescue=int(row["umm_rescues_oracle"]),
                loss=int(row["dino_loses_oracle"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This directly tests the UMM-as-world-model claim: UMM should intervene mainly when DINO's static visual choice is dynamically unreliable.",
            "- If this gate improves oracle matches but not mean distance, the proposal should frame UMM as a planning uncertainty and candidate-identification module.",
            "- If this gate is unstable, the next required step is richer generative rollout supervision rather than more static representation fusion.",
        ]
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DINO failure-aware UMM override gate.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--pairwise", required=True)
    parser.add_argument("--dino-selector", required=True, help="name=path:field")
    parser.add_argument("--umm-selector", required=True, help="name=path:field")
    parser.add_argument("--objective", choices=["lower_dist", "oracle_rescue"], default="lower_dist")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-folds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=181)
    parser.add_argument("--output-score-name", default="dino_failure_world_model_gate_score")
    parser.add_argument("--output-scores", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--report-output", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.base)
    pair_rows = read_jsonl(args.pairwise)
    reference_keys = [row_key(row) for row in rows]
    dino_name, dino_path, dino_field = parse_selector(args.dino_selector)
    umm_name, umm_path, umm_field = parse_selector(args.umm_selector)
    dino_scores = load_selector(dino_path, dino_field, reference_keys)
    umm_scores = load_selector(umm_path, umm_field, reference_keys)
    rows_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_case.setdefault(str(row["case_id"]), []).append(row)
    case_ids = sorted(rows_by_case)
    case_feature_map = {}
    case_record_map = {}
    for case in case_ids:
        features, record = case_features(rows_by_case[case], dino_scores, umm_scores)
        case_feature_map[case] = features
        case_record_map[case] = record

    folds = make_folds(case_ids, args.num_folds, args.seed)
    nested_scores = [0.0 for _ in rows]
    predictions = []
    fold_records = []
    for fold_idx, test_cases in enumerate(folds):
        train_cases = [case for case in case_ids if case not in set(test_cases)]
        train_features = [case_feature_map[case] for case in train_cases]
        train_labels = [label_from_record(case_record_map[case], args.objective) for case in train_cases]
        model = train_gate(
            features=train_features,
            labels=train_labels,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed + fold_idx,
        )
        for case in test_cases:
            pred, logits = predict(model, case_feature_map[case])
            target = label_from_record(case_record_map[case], args.objective)
            chosen_scores = umm_scores if pred else dino_scores
            for idx, row in enumerate(rows):
                if str(row["case_id"]) == str(case):
                    nested_scores[idx] = chosen_scores[row_key(row)]
            record = dict(case_record_map[case])
            record.update(
                {
                    "fold_idx": fold_idx,
                    "predicted_override": bool(pred),
                    "target_override": bool(target),
                    "gate_correct": pred == target,
                    "logits": {dino_name: logits[0], umm_name: logits[1]},
                }
            )
            predictions.append(record)
        fold_records.append({"fold_idx": fold_idx, "test_cases": list(test_cases), "num_train_cases": len(train_cases)})

    fold_metrics = [evaluate_scores(rows, nested_scores, set(test_cases), pair_rows) for test_cases in folds]
    out_rows = []
    for row, score in zip(rows, nested_scores):
        out = dict(row)
        out[args.output_score_name] = float(score)
        out_rows.append(out)
    write_jsonl(args.output_scores, out_rows)
    summary = {
        "method": "dino_failure_world_model_gate",
        "objective": args.objective,
        "dino_selector": {"name": dino_name, "path": dino_path, "field": dino_field},
        "umm_selector": {"name": umm_name, "path": umm_path, "field": umm_field},
        "hidden_dim": args.hidden_dim,
        "folds": fold_records,
        "gate_accuracy": float(mean([row["gate_correct"] for row in predictions])),
        "override_count": int(sum(row["predicted_override"] for row in predictions)),
        "target_override_count": int(sum(row["target_override"] for row in predictions)),
        "umm_oracle_rescue_opportunities": int(sum(row["umm_rescues_oracle"] for row in case_record_map.values())),
        "dino_oracle_loss_risks": int(sum(row["dino_loses_oracle"] for row in case_record_map.values())),
        "case_predictions": predictions,
        "crossval": merge_dynamics_metrics(fold_metrics),
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, args.report_output)
    cv = summary["crossval"]
    n_cases = len(cv.get("selected_cem", [])) or len(summary.get("case_predictions", []))
    print(f"[dino_failure_world_model_gate] objective={args.objective}")
    print(f"[dino_failure_world_model_gate] mean={cv['cem_selected_state_dist']['mean']:.4f}")
    print(f"[dino_failure_world_model_gate] better={cv['cem_better_than_rank0_count']}/{n_cases}")
    print(f"[dino_failure_world_model_gate] oracle={cv['cem_matches_oracle_best_count']}/{n_cases}")
    print(f"[dino_failure_world_model_gate] gate_acc={summary['gate_accuracy']:.4f}")
    print(f"[dino_failure_world_model_gate] overrides={summary['override_count']}/{n_cases}")


if __name__ == "__main__":
    main()
