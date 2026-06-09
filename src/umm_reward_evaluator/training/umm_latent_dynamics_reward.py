from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import torch

from umm_reward_evaluator.analysis.umm_dense_tokens import load_or_compute_row_tokens
from umm_reward_evaluator.manifest import read_jsonl, write_jsonl
from umm_reward_evaluator.training.ggp_umm_pairwise_reward import evaluate, is_expert, make_folds


def frame_vectors(tokens: np.ndarray) -> np.ndarray:
    if len(tokens) == 0:
        return np.zeros((0, 1), dtype=np.float32)
    mean_tokens = np.mean(tokens, axis=1)
    std_tokens = np.std(tokens, axis=1)
    out = np.concatenate([mean_tokens, std_tokens], axis=1)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.maximum(norm, 1e-6)).astype(np.float32)


def action_vectors(row: dict[str, Any], num_transitions: int) -> np.ndarray:
    actions = row.get("actions")
    if isinstance(actions, list) and actions and num_transitions > 0:
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        chunks = np.array_split(arr, num_transitions)
        feats = []
        for chunk in chunks:
            if len(chunk) == 0:
                feats.append(np.zeros(4, dtype=np.float32))
            else:
                feats.append(np.concatenate([np.mean(chunk, axis=0), np.std(chunk, axis=0)]).astype(np.float32))
        return np.asarray(feats, dtype=np.float32)

    model_actions = row.get("model_actions")
    if isinstance(model_actions, list) and model_actions:
        arr = np.asarray(model_actions, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if len(arr) >= num_transitions:
            return arr[:num_transitions].astype(np.float32)
        if len(arr) > 0:
            pad = np.repeat(arr[-1:], num_transitions - len(arr), axis=0)
            return np.concatenate([arr, pad], axis=0).astype(np.float32)
    return np.zeros((num_transitions, 10), dtype=np.float32)


def pad_actions(actions: list[np.ndarray]) -> list[np.ndarray]:
    max_dim = max((arr.shape[1] for arr in actions if arr.ndim == 2), default=1)
    out = []
    for arr in actions:
        if arr.shape[1] == max_dim:
            out.append(arr)
        else:
            pad = np.zeros((arr.shape[0], max_dim - arr.shape[1]), dtype=np.float32)
            out.append(np.concatenate([arr, pad], axis=1))
    return out


def build_sequences(
    rows: list[dict[str, Any]],
    row_tokens: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    pred_vecs = []
    goal_vecs = []
    action_vecs = []
    raw_actions = []
    for row, (pred_tokens, goal_tokens) in zip(rows, row_tokens):
        pred = frame_vectors(pred_tokens)
        goal = frame_vectors(goal_tokens)
        transitions = max(0, len(pred) - 1)
        pred_vecs.append(pred)
        goal_vecs.append(goal)
        raw_actions.append(action_vectors(row, transitions))
    action_vecs = pad_actions(raw_actions)
    return pred_vecs, goal_vecs, action_vecs


def transition_samples(
    rows: list[dict[str, Any]],
    pred_vecs: list[np.ndarray],
    action_vecs: list[np.ndarray],
    train_cases: set[str],
    train_rows: str,
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for row, z_seq, a_seq in zip(rows, pred_vecs, action_vecs):
        if str(row["case_id"]) not in train_cases:
            continue
        expert = is_expert(row)
        if train_rows == "expert" and not expert:
            continue
        if train_rows == "cem" and expert:
            continue
        for step in range(max(0, len(z_seq) - 1)):
            xs.append(np.concatenate([z_seq[step], a_seq[step]], axis=0))
            ys.append(z_seq[step + 1] - z_seq[step])
    if not xs:
        raise ValueError("no transition samples")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


class DynamicsMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_dynamics(
    x: np.ndarray,
    y: np.ndarray,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: str,
) -> tuple[DynamicsMLP, dict[str, np.ndarray], list[dict[str, float]]]:
    torch.manual_seed(seed)
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    y_mean = y.mean(axis=0, keepdims=True)
    y_std = y.std(axis=0, keepdims=True)
    y_std[y_std < 1e-6] = 1.0
    xt = torch.from_numpy((x - x_mean) / x_std).float().to(device)
    yt = torch.from_numpy((y - y_mean) / y_std).float().to(device)
    model = DynamicsMLP(input_dim=xt.shape[1], output_dim=yt.shape[1], hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(1, epochs + 1):
        pred = model(xt)
        loss = torch.mean((pred - yt) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            history.append({"epoch": float(epoch), "mse": float(loss.detach().cpu())})
    stats = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    return model.eval(), stats, history


@torch.inference_mode()
def rollout_score(
    model: DynamicsMLP,
    stats: dict[str, np.ndarray],
    z_seq: np.ndarray,
    goal_seq: np.ndarray,
    action_seq: np.ndarray,
) -> tuple[float, float]:
    if len(z_seq) == 0:
        return 0.0, 0.0
    z = z_seq[0].copy()
    transition_errors = []
    device = next(model.parameters()).device
    for step in range(max(0, len(z_seq) - 1)):
        x = np.concatenate([z, action_seq[step]], axis=0)[None, :]
        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        delta_norm = model(torch.from_numpy(x_norm).float().to(device)).cpu().numpy()
        delta = delta_norm * stats["y_std"] + stats["y_mean"]
        z_next = z + delta[0]
        transition_errors.append(float(np.mean((z_next - z_seq[step + 1]) ** 2)))
        z = z_next
        z = z / max(float(np.linalg.norm(z)), 1e-6)
    goal = goal_seq[-1] if len(goal_seq) else z_seq[-1]
    goal_score = float(np.dot(z, goal) / max(float(np.linalg.norm(z) * np.linalg.norm(goal)), 1e-6))
    consistency = -float(mean(transition_errors)) if transition_errors else 0.0
    return goal_score, consistency


def merge_dynamics_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    selected_cem = [item for fold in fold_metrics for item in fold["selected_cem"]]
    selected_all = [item for fold in fold_metrics for item in fold["selected_all"]]

    def avg(key: str) -> float | None:
        vals = [fold[key] for fold in fold_metrics if fold.get(key) is not None]
        return float(mean(vals)) if vals else None

    def dist_stats(values: list[float]) -> dict[str, float]:
        return {"mean": float(mean(values)), "median": float(median(values)), "min": float(min(values)), "max": float(max(values))}

    dists = [x["oracle_state_dist"] for x in selected_cem]
    rank0 = [x["rank0_state_dist"] for x in selected_cem]
    oracle = [x["oracle_best_cem_state_dist"] for x in selected_cem]
    return {
        "pair_label_accuracy": avg("pair_label_accuracy"),
        "oracle_expert_pair_accuracy": avg("oracle_expert_pair_accuracy"),
        "all_candidate_top_success": float(mean([x["oracle_success"] for x in selected_all])),
        "cem_selected_state_dist": dist_stats(dists),
        "rank0_state_dist": dist_stats(rank0),
        "oracle_best_cem_state_dist": dist_stats(oracle),
        "cem_better_than_rank0_count": int(sum(a < b for a, b in zip(dists, rank0))),
        "cem_matches_oracle_best_count": int(sum(x["candidate_id"] == x["oracle_best_cem_id"] for x in selected_cem)),
        "selected_all": selected_all,
        "selected_cem": selected_cem,
    }


def evaluate_scores(rows: list[dict[str, Any]], scores: np.ndarray, cases: set[str], pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = evaluate(rows, scores, pair_rows, cases)
    return metrics


def write_report(summary: dict[str, Any], args: argparse.Namespace) -> None:
    m = summary["crossval"]
    c = m["cem_selected_state_dist"]
    r = m["rank0_state_dist"]
    o = m["oracle_best_cem_state_dist"]
    n_cases = len(m.get("selected_cem", []))
    report = f"""# AC-UMM-LDM: Action-Conditioned UMM Latent Dynamics Model

## Setup

- Manifest: `{args.manifest}`
- Pairwise labels for evaluation: `{args.pairwise}`
- Patch-token cache: `{args.patch_cache}`
- Encoder backbone: `{args.encoder_backbone}`
- UMM/DINO/Lance model: `{args.model_name}`
- Hidden dim: {args.hidden_dim}
- Epochs: {args.epochs}
- Train rows: `{args.train_rows}`

This is a training-based world-model idea. It trains a dynamics model in dense UMM/DINO latent space:

```text
z_t = DenseUMMEncoder(o_t)
z_{{t+1}} = f_theta(z_t, a_t)
score(a_{{0:H}}) = cosine(rollout_theta(z_0, a_{{0:H}}), z_goal)
```

It is not a training-free metric and is closer to "UMM as a world model" than evaluator-only reranking.

## Cross-Validated Results

| Metric | Value |
| --- | ---: |
| Held-out Qwen pair-label accuracy | {m['pair_label_accuracy']:.4f} |
| Held-out expert-vs-CEM oracle pair accuracy | {m['oracle_expert_pair_accuracy']:.4f} |
| All-candidate top success | {m['all_candidate_top_success']:.4f} |
| CEM better than planner rank0 | {m['cem_better_than_rank0_count']}/{n_cases} |
| CEM matched oracle-best CEM | {m['cem_matches_oracle_best_count']}/{n_cases} |

## CEM-Only Reranking by State Distance

| Selector | Mean | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| NanoWM planner rank0 | {r['mean']:.4f} | {r['median']:.4f} | {r['min']:.4f} | {r['max']:.4f} |
| AC-UMM-LDM | {c['mean']:.4f} | {c['median']:.4f} | {c['min']:.4f} | {c['max']:.4f} |
| Oracle-best CEM | {o['mean']:.4f} | {o['median']:.4f} | {o['min']:.4f} | {o['max']:.4f} |
"""
    out = Path(args.report_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train action-conditioned UMM latent dynamics and score candidate rollouts by goal latent reachability.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pairwise", required=True)
    parser.add_argument("--patch-cache", required=True)
    parser.add_argument("--output-scores", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--model-name", default="facebook/dinov2-base")
    parser.add_argument("--encoder-backbone", choices=["dinov2", "lance_cache", "lance"], default="dinov2")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-frames", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-folds", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--score-mode", choices=["goal", "goal_plus_consistency"], default="goal")
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--train-rows", choices=["all", "expert", "cem"], default="all")
    parser.add_argument("--seed", type=int, default=151)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    pair_rows = read_jsonl(args.pairwise)
    row_tokens = load_or_compute_row_tokens(
        rows=rows,
        cache_path=args.patch_cache,
        encoder_backbone=args.encoder_backbone,
        model_name=args.model_name,
        image_size=args.image_size,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        device=args.device,
    )
    pred_vecs, goal_vecs, action_vecs = build_sequences(rows, row_tokens)
    case_ids = sorted({str(row["case_id"]) for row in rows})
    folds = make_folds(case_ids, args.num_folds, args.seed)
    fold_scores: dict[int, float] = {}
    fold_goal_scores: dict[int, float] = {}
    fold_consistency_scores: dict[int, float] = {}
    fold_metrics = []
    fold_models = []
    for fold_id, test_case_list in enumerate(folds):
        test_cases = set(test_case_list)
        train_cases = set(case_ids) - test_cases
        x_train, y_train = transition_samples(rows, pred_vecs, action_vecs, train_cases, args.train_rows)
        model, stats, history = train_dynamics(
            x=x_train,
            y=y_train,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed + fold_id,
            device=args.device,
        )
        scores = np.zeros(len(rows), dtype=np.float64)
        for idx, row in enumerate(rows):
            goal_score, consistency = rollout_score(model, stats, pred_vecs[idx], goal_vecs[idx], action_vecs[idx])
            score = goal_score
            if args.score_mode == "goal_plus_consistency":
                score = goal_score + args.consistency_weight * consistency
            scores[idx] = score
            if str(row["case_id"]) in test_cases:
                fold_scores[idx] = float(score)
                fold_goal_scores[idx] = float(goal_score)
                fold_consistency_scores[idx] = float(consistency)
        fold_metrics.append(evaluate_scores(rows, scores, test_cases, pair_rows))
        fold_models.append(
            {
                "fold": fold_id,
                "train_cases": sorted(train_cases),
                "test_cases": sorted(test_cases),
                "history": history,
                "x_mean": stats["x_mean"].tolist(),
                "x_std": stats["x_std"].tolist(),
                "y_mean": stats["y_mean"].tolist(),
                "y_std": stats["y_std"].tolist(),
            }
        )

    out_rows = []
    for idx, row in enumerate(rows):
        out = dict(row)
        out["ac_umm_ldm_score"] = fold_scores[idx]
        out["ac_umm_ldm_goal_score"] = fold_goal_scores[idx]
        out["ac_umm_ldm_consistency_score"] = fold_consistency_scores[idx]
        out_rows.append(out)
    write_jsonl(args.output_scores, out_rows)

    summary = {
        "method": "AC-UMM-LDM",
        "manifest": args.manifest,
        "pairwise": args.pairwise,
        "patch_cache": args.patch_cache,
        "model_name": args.model_name,
        "encoder_backbone": args.encoder_backbone,
        "num_cases": len(case_ids),
        "num_rows": len(rows),
        "num_folds": args.num_folds,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "score_mode": args.score_mode,
        "consistency_weight": args.consistency_weight,
        "train_rows": args.train_rows,
        "fold_metrics": fold_metrics,
        "crossval": merge_dynamics_metrics(fold_metrics),
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.model_output).write_text(json.dumps({"fold_models": fold_models}, indent=2), encoding="utf-8")
    write_report(summary, args)
    cv = summary["crossval"]
    print(f"[ac_umm_ldm] rows={len(rows)} folds={args.num_folds}")
    print(f"[ac_umm_ldm] qwen_pair_label_acc={cv['pair_label_accuracy']:.4f}")
    print(f"[ac_umm_ldm] expert_pair_acc={cv['oracle_expert_pair_accuracy']:.4f}")
    print(f"[ac_umm_ldm] cem_mean_dist={cv['cem_selected_state_dist']['mean']:.4f}")
    print(f"[ac_umm_ldm] report={args.report_output}")


if __name__ == "__main__":
    main()
