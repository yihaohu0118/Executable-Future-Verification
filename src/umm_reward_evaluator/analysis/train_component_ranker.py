from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from umm_reward_evaluator.manifest import read_jsonl, write_jsonl
from umm_reward_evaluator.training.ggp_umm_pairwise_reward import make_folds
from umm_reward_evaluator.training.umm_latent_dynamics_reward import evaluate_scores, merge_dynamics_metrics


def case_zscore(values: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    by_case: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_case.setdefault(str(row["case_id"]), []).append(idx)
    for idxs in by_case.values():
        x = values[idxs]
        std = x.std()
        out[idxs] = (x - x.mean()) / (std if std > 1e-8 else 1.0)
    return out


def feature_matrix(rows: list[dict[str, Any]], components: list[str], include_products: bool) -> np.ndarray:
    comps = []
    for name in components:
        raw = np.asarray([row[name] for row in rows], dtype=np.float64)
        comps.append(case_zscore(raw, rows))
    x = np.stack(comps, axis=1)
    if include_products:
        extras = [x[:, i : i + 1] ** 2 for i in range(x.shape[1])]
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                extras.append((x[:, i] * x[:, j])[:, None])
        x = np.concatenate([x] + extras, axis=1)
    return x.astype(np.float32)


def oracle_pairs(rows: list[dict[str, Any]], train_cases: set[str], margin: float) -> list[tuple[int, int]]:
    by_case: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        if str(row["case_id"]) in train_cases:
            by_case.setdefault(str(row["case_id"]), []).append(idx)
    pairs = []
    for idxs in by_case.values():
        for a in idxs:
            for b in idxs:
                if float(rows[a]["oracle_state_dist"]) + margin < float(rows[b]["oracle_state_dist"]):
                    pairs.append((a, b))
    return pairs


def pairwise_label_pairs(rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]], train_cases: set[str]) -> list[tuple[int, int]]:
    row_to_idx = {(str(row["case_id"]), str(row["candidate_id"])): idx for idx, row in enumerate(rows)}
    pairs = []
    for row in pair_rows:
        case_id = str(row.get("case_id"))
        if case_id not in train_cases:
            continue
        pos = row_to_idx.get((case_id, str(row.get("preferred_candidate_id"))))
        neg = row_to_idx.get((case_id, str(row.get("rejected_candidate_id"))))
        if pos is not None and neg is not None:
            pairs.append((pos, neg))
    return pairs


class Ranker(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            self.net = torch.nn.Linear(input_dim, 1)
        else:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_ranker(
    x: np.ndarray,
    pairs: list[tuple[int, int]],
    aux_pairs: list[tuple[int, int]] | None,
    aux_weight: float,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: str,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    if not pairs:
        raise ValueError("No training pairs")
    torch.manual_seed(seed)
    model = Ranker(x.shape[1], hidden_dim).to(device)
    xt = torch.from_numpy(x).float().to(device)
    pos = torch.tensor([p for p, _ in pairs], dtype=torch.long, device=device)
    neg = torch.tensor([n for _, n in pairs], dtype=torch.long, device=device)
    aux_pos = aux_neg = None
    if aux_pairs:
        aux_pos = torch.tensor([p for p, _ in aux_pairs], dtype=torch.long, device=device)
        aux_neg = torch.tensor([n for _, n in aux_pairs], dtype=torch.long, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        scores = model(xt)
        diff = scores[pos] - scores[neg]
        loss = F.softplus(-diff).mean()
        if aux_pos is not None and aux_neg is not None and aux_weight > 0:
            loss = loss + aux_weight * F.softplus(-(scores[aux_pos] - scores[aux_neg])).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            record = {
                "epoch": float(epoch),
                "loss": float(loss.detach().cpu()),
                "pair_acc": float((diff > 0).float().mean().detach().cpu()),
            }
            if aux_pos is not None and aux_neg is not None:
                record["aux_pair_acc"] = float(((scores[aux_pos] - scores[aux_neg]) > 0).float().mean().detach().cpu())
            history.append(record)
    with torch.no_grad():
        scores = model(xt).detach().cpu().numpy().astype(np.float64)
    return scores, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fold-wise RankNet on arbitrary score components.")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--pairwise", required=True)
    parser.add_argument("--components", nargs="+", required=True)
    parser.add_argument("--output-score-name", default="component_ranknet_score")
    parser.add_argument("--output-scores", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--label-source", choices=["oracle", "pairwise", "mixed"], default="oracle")
    parser.add_argument("--pairwise-weight", type=float, default=0.25)
    parser.add_argument("--oracle-margin", type=float, default=0.0)
    parser.add_argument("--include-products", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-folds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=181)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows = read_jsonl(args.scores)
    pair_rows = read_jsonl(args.pairwise)
    x = feature_matrix(rows, args.components, include_products=args.include_products)
    case_ids = sorted({str(row["case_id"]) for row in rows})
    folds = make_folds(case_ids, args.num_folds, args.seed)
    nested_scores = np.zeros(len(rows), dtype=np.float64)
    fold_records = []
    for fold_idx, test_cases in enumerate(folds):
        train_cases = set(case_ids) - set(test_cases)
        aux_pairs = None
        aux_weight = 0.0
        if args.label_source == "oracle":
            pairs = oracle_pairs(rows, train_cases, args.oracle_margin)
        elif args.label_source == "pairwise":
            pairs = pairwise_label_pairs(rows, pair_rows, train_cases)
        else:
            pairs = oracle_pairs(rows, train_cases, args.oracle_margin)
            aux_pairs = pairwise_label_pairs(rows, pair_rows, train_cases)
            aux_weight = args.pairwise_weight
        scores, history = train_ranker(
            x=x,
            pairs=pairs,
            aux_pairs=aux_pairs,
            aux_weight=aux_weight,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed + fold_idx,
            device=args.device,
        )
        for idx, row in enumerate(rows):
            if str(row["case_id"]) in set(test_cases):
                nested_scores[idx] = scores[idx]
        fold_records.append({"test_cases": list(test_cases), "num_train_pairs": len(pairs), "last_train": history[-1]})

    fold_metrics = [evaluate_scores(rows, nested_scores, set(test_cases), pair_rows) for test_cases in folds]
    summary = {
        "method": "component_ranknet",
        "scores": args.scores,
        "components": args.components,
        "label_source": args.label_source,
        "pairwise_weight": args.pairwise_weight,
        "include_products": args.include_products,
        "hidden_dim": args.hidden_dim,
        "folds": fold_records,
        "crossval": merge_dynamics_metrics(fold_metrics),
    }
    out_rows = []
    for row, score in zip(rows, nested_scores):
        out = dict(row)
        out[args.output_score_name] = float(score)
        out_rows.append(out)
    write_jsonl(args.output_scores, out_rows)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cv = summary["crossval"]
    n_cases = len(cv.get("selected_cem", []))
    print(f"[component_ranknet] mean={cv['cem_selected_state_dist']['mean']:.4f}")
    print(f"[component_ranknet] median={cv['cem_selected_state_dist']['median']:.4f}")
    print(f"[component_ranknet] better={cv['cem_better_than_rank0_count']}/{n_cases}")
    print(f"[component_ranknet] oracle={cv['cem_matches_oracle_best_count']}/{n_cases}")
    print(f"[component_ranknet] pair_acc={cv['pair_label_accuracy']:.4f}")
    print(f"[component_ranknet] expert_pair_acc={cv['oracle_expert_pair_accuracy']:.4f}")


if __name__ == "__main__":
    main()
