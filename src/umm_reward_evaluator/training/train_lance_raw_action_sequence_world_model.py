from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from umm_reward_evaluator.analysis.umm_dense_tokens import load_dense_token_cache
from umm_reward_evaluator.manifest import read_jsonl, write_jsonl
from umm_reward_evaluator.training.ggp_umm_pairwise_reward import is_expert, make_folds
from umm_reward_evaluator.training.train_lance_hidden_world_action_model import HiddenWorldActionModel, transform_state
from umm_reward_evaluator.training.train_lance_joint_bottleneck_dynamics import (
    apply_pca,
    fit_pca,
    standardize_apply,
    standardize_fit,
)
from umm_reward_evaluator.training.umm_latent_dynamics_reward import (
    evaluate_scores,
    frame_vectors,
    merge_dynamics_metrics,
)


def fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


class TemporalActionWorldModel(torch.nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_steps: int,
        action_step_dim: int,
        latent_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if action_steps * action_step_dim != action_dim:
            raise ValueError(f"action_steps * action_step_dim must equal action_dim, got {action_steps} * {action_step_dim} != {action_dim}")
        self.action_steps = action_steps
        self.action_step_dim = action_step_dim
        self.state_encoder = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )
        self.step_encoder = torch.nn.Sequential(
            torch.nn.Linear(action_step_dim, hidden_dim),
            torch.nn.GELU(),
        )
        self.action_gru = torch.nn.GRU(hidden_dim, latent_dim, batch_first=True)
        self.dynamics = torch.nn.Sequential(
            torch.nn.Linear(2 * latent_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )
        self.transition_encoder = torch.nn.Sequential(
            torch.nn.Linear(2 * latent_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )
        self.inverse_action = torch.nn.Sequential(
            torch.nn.Linear(2 * latent_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, action_dim),
        )
        self.state_decoder = torch.nn.Linear(latent_dim, state_dim)

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.state_encoder(state), dim=-1)

    def action_embed(self, action: torch.Tensor) -> torch.Tensor:
        steps = action.reshape(action.shape[0], self.action_steps, self.action_step_dim)
        encoded = self.step_encoder(steps)
        _out, hidden = self.action_gru(encoded)
        return F.normalize(hidden[-1], dim=-1)

    def predict_next(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        act = self.action_embed(action)
        delta = self.dynamics(torch.cat([z, act], dim=-1))
        return F.normalize(z + delta, dim=-1)

    def transition_embed(self, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.transition_encoder(torch.cat([z0, z1], dim=-1)), dim=-1)

    def predict_action(self, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        return self.inverse_action(torch.cat([z0, z1], dim=-1))


def build_candidate_states(
    rows: list[dict[str, Any]],
    candidate_cache: str,
    state_pca: dict[str, np.ndarray],
    state_mean: np.ndarray,
    state_std: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    row_tokens = load_dense_token_cache(candidate_cache, expected_rows=len(rows))
    pred_states = [transform_state(frame_vectors(pred), state_pca, state_mean, state_std) for pred, _ in row_tokens]
    goal_states = [transform_state(frame_vectors(goal), state_pca, state_mean, state_std) for _, goal in row_tokens]
    return pred_states, goal_states


def candidate_raw_action_chunks(row: dict[str, Any], num_transitions: int, chunk_dim: int) -> np.ndarray:
    model_actions = row.get("model_actions")
    if isinstance(model_actions, list) and model_actions:
        arr = np.asarray(model_actions, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < chunk_dim:
            pad = np.zeros((arr.shape[0], chunk_dim - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        arr = arr[:, :chunk_dim]
        if len(arr) >= num_transitions:
            return arr[:num_transitions].astype(np.float32)
        if len(arr) > 0:
            pad = np.repeat(arr[-1:], num_transitions - len(arr), axis=0)
            return np.concatenate([arr, pad], axis=0).astype(np.float32)

    actions = row.get("actions")
    if isinstance(actions, list) and actions and num_transitions > 0:
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)
        chunks = np.array_split(arr, num_transitions)
        feats = []
        for chunk in chunks:
            flat = chunk.reshape(-1).astype(np.float32)
            if len(flat) < chunk_dim:
                flat = np.concatenate([flat, np.zeros(chunk_dim - len(flat), dtype=np.float32)])
            feats.append(flat[:chunk_dim])
        return np.asarray(feats, dtype=np.float32)
    return np.zeros((num_transitions, chunk_dim), dtype=np.float32)


def build_raw_candidate_actions(
    rows: list[dict[str, Any]],
    pred_states: list[np.ndarray],
    chunk_dim: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    raw = [candidate_raw_action_chunks(row, max(0, len(seq) - 1), chunk_dim) for row, seq in zip(rows, pred_states)]
    stacked = np.concatenate([arr for arr in raw if len(arr)], axis=0)
    action_mean, action_std = standardize_fit(stacked)
    return [standardize_apply(arr, action_mean, action_std) for arr in raw], action_mean, action_std


def raw_action_interventions(actions: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> list[np.ndarray]:
    raw = (actions.astype(np.float32) * action_std.astype(np.float32)) + action_mean.astype(np.float32)
    negatives: list[np.ndarray] = []
    if len(raw) >= 2:
        negatives.append(np.roll(raw, shift=1, axis=0).astype(np.float32))
        negatives.append(raw[::-1].copy().astype(np.float32))
        half = max(1, len(raw) // 2)
        negatives.append(np.concatenate([raw[half:], raw[:half]], axis=0).astype(np.float32))
    negatives.append(np.zeros_like(raw, dtype=np.float32))
    negatives.append((-raw).astype(np.float32))
    negatives.append((0.5 * raw).astype(np.float32))
    negatives.append((2.0 * raw).astype(np.float32))
    return [standardize_apply(arr, action_mean, action_std) for arr in negatives]


def sequence_tensors(
    rows: list[dict[str, Any]],
    pred_states: list[np.ndarray],
    goal_states: list[np.ndarray],
    cand_actions: list[np.ndarray],
    pair_rows: list[dict[str, Any]],
    train_cases: set[str],
    train_rows: str,
    negative_source: str,
    max_negatives: int,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    preference_source: str,
    oracle_margin: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None] | None:
    selected: list[tuple[int, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]] = []
    for idx, (row, seq, goal, actions) in enumerate(zip(rows, pred_states, goal_states, cand_actions)):
        case_id = str(row["case_id"])
        if case_id not in train_cases or len(seq) < 2:
            continue
        expert = is_expert(row)
        if train_rows == "expert" and not expert:
            continue
        if train_rows == "cem" and expert:
            continue
        if train_rows == "all_non_oracle" and expert:
            continue
        goal_vec = goal[-1] if len(goal) else seq[-1]
        selected.append((idx, row, seq.astype(np.float32), goal_vec.astype(np.float32), actions[: len(seq) - 1].astype(np.float32)))
    if not selected:
        return None

    min_len = min(len(item[2]) for item in selected)
    min_transitions = min_len - 1
    state = np.stack([item[2][:min_len] for item in selected], axis=0)
    goal_state = np.stack([item[3] for item in selected], axis=0)
    action = np.stack([item[4][:min_transitions] for item in selected], axis=0)

    neg_action = None
    if negative_source != "none":
        neg_chunks = []
        for row_idx, row, _seq, _goal, actions in selected:
            candidates = []
            if negative_source in {"same_case", "mixed_raw"}:
                case_id = str(row["case_id"])
                candidates.extend(
                    [
                        other_actions[:min_transitions].astype(np.float32)
                        for other_idx, (other_row, other_actions) in enumerate(zip(rows, cand_actions))
                        if other_idx != row_idx and str(other_row["case_id"]) == case_id and len(other_actions) >= min_transitions
                    ]
                )
            if negative_source in {"raw_intervention", "mixed_raw"}:
                candidates.extend(raw_action_interventions(actions[:min_transitions], action_mean, action_std))
            if not candidates:
                candidates = [actions[:min_transitions].astype(np.float32)]
            candidates = candidates[: max(1, max_negatives)]
            while len(candidates) < max(1, max_negatives):
                candidates.append(candidates[-1])
            neg_chunks.append(np.stack(candidates, axis=1))
        neg_action = np.stack(neg_chunks, axis=0)

    row_to_seq = {(str(row["case_id"]), str(row["candidate_id"])): seq_idx for seq_idx, (_idx, row, _seq, _goal, _act) in enumerate(selected)}
    pref_pairs: list[tuple[int, int]] = []
    if preference_source == "pairwise":
        for pair in pair_rows:
            case_id = str(pair.get("case_id"))
            if case_id not in train_cases:
                continue
            pos = row_to_seq.get((case_id, str(pair.get("preferred_candidate_id"))))
            neg = row_to_seq.get((case_id, str(pair.get("rejected_candidate_id"))))
            if pos is not None and neg is not None:
                pref_pairs.append((pos, neg))
    elif preference_source == "oracle":
        by_case: dict[str, list[int]] = {}
        for seq_idx, (_idx, row, _seq, _goal, _act) in enumerate(selected):
            by_case.setdefault(str(row["case_id"]), []).append(seq_idx)
        for seq_idxs in by_case.values():
            for a in seq_idxs:
                for b in seq_idxs:
                    da = float(selected[a][1]["oracle_state_dist"])
                    db = float(selected[b][1]["oracle_state_dist"])
                    if da + oracle_margin < db:
                        pref_pairs.append((a, b))
    elif preference_source != "none":
        raise ValueError(f"preference_source={preference_source!r}")

    return (
        torch.from_numpy(state).float().to(device),
        torch.from_numpy(goal_state).float().to(device),
        torch.from_numpy(action).float().to(device),
        None if neg_action is None else torch.from_numpy(neg_action).float().to(device),
        None if not pref_pairs else torch.tensor(pref_pairs, dtype=torch.long, device=device),
    )


def train_fold_model(
    seq_pack: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None],
    state_dim: int,
    action_dim: int,
    action_encoder: str,
    action_steps: int,
    action_step_dim: int,
    latent_dim: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    sequence_weight: float,
    rollout_weight: float,
    counterfactual_weight: float,
    action_contrastive_weight: float,
    inverse_weight: float,
    recon_weight: float,
    preference_weight: float,
    temperature: float,
    seed: int,
    device: str,
) -> tuple[HiddenWorldActionModel, list[dict[str, float]]]:
    torch.manual_seed(seed)
    if action_encoder == "flat":
        model = HiddenWorldActionModel(state_dim, action_dim, latent_dim, hidden_dim).to(device)
    elif action_encoder == "temporal_gru":
        model = TemporalActionWorldModel(
            state_dim=state_dim,
            action_dim=action_dim,
            action_steps=action_steps,
            action_step_dim=action_step_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        ).to(device)
    else:
        raise ValueError(f"Unknown action_encoder={action_encoder!r}")
    seq_state, goal_state, seq_action, neg_action, pref_pairs = seq_pack
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        seq_z = model.encode(seq_state.reshape(-1, seq_state.shape[-1])).reshape(*seq_state.shape[:2], -1)
        goal_z = model.encode(goal_state)
        flat_z0 = seq_z[:, :-1].reshape(-1, seq_z.shape[-1])
        flat_z1 = seq_z[:, 1:].reshape(-1, seq_z.shape[-1])
        flat_actions = seq_action.reshape(-1, seq_action.shape[-1])
        pred = model.predict_next(flat_z0, flat_actions)
        pos_score = torch.sum(pred * flat_z1.detach(), dim=-1)
        seq_loss = torch.mean(1.0 - pos_score)

        trans = model.transition_embed(flat_z0, flat_z1.detach())
        act_emb = model.action_embed(flat_actions)
        labels = torch.arange(len(flat_z0), device=device)
        action_logits = trans @ act_emb.T / temperature
        action_nce = 0.5 * (F.cross_entropy(action_logits, labels) + F.cross_entropy(action_logits.T, labels))
        inverse = F.mse_loss(model.predict_action(flat_z0, flat_z1.detach()), flat_actions)
        recon = F.mse_loss(model.state_decoder(seq_z.reshape(-1, seq_z.shape[-1])), seq_state.reshape(-1, seq_state.shape[-1]))

        cf_loss = torch.zeros((), device=device)
        seq_cf_score = torch.zeros(seq_z.shape[0], device=device)
        if neg_action is not None:
            flat_neg = neg_action.reshape(-1, neg_action.shape[2], neg_action.shape[3])
            num_neg = flat_neg.shape[1]
            z_rep = flat_z0[:, None, :].expand(-1, num_neg, -1).reshape(-1, flat_z0.shape[-1])
            neg_pred = model.predict_next(z_rep, flat_neg.reshape(-1, flat_neg.shape[-1])).reshape(flat_z0.shape[0], num_neg, -1)
            neg_score = torch.max(torch.sum(neg_pred * flat_z1[:, None, :].detach(), dim=-1), dim=-1).values
            cf_margin = pos_score - neg_score
            seq_cf_score = cf_margin.reshape(seq_z.shape[0], seq_action.shape[1]).mean(dim=1)
            cf_loss = F.softplus(-cf_margin / temperature).mean()

        z_roll = seq_z[:, 0]
        rollout_terms = []
        for step in range(seq_action.shape[1]):
            z_roll = model.predict_next(z_roll, seq_action[:, step])
            rollout_terms.append(1.0 - torch.sum(z_roll * seq_z[:, step + 1].detach(), dim=-1))
        rollout_loss = torch.stack(rollout_terms, dim=0).mean()

        pref_loss = torch.zeros((), device=device)
        if pref_pairs is not None and preference_weight > 0:
            seq_goal_score = torch.sum(F.normalize(z_roll, dim=-1) * F.normalize(goal_z, dim=-1), dim=-1)
            seq_consistency = pos_score.reshape(seq_z.shape[0], seq_action.shape[1]).mean(dim=1)
            seq_score = seq_goal_score + seq_consistency + 0.5 * seq_cf_score
            pref_diff = seq_score[pref_pairs[:, 0]] - seq_score[pref_pairs[:, 1]]
            pref_loss = F.softplus(-pref_diff).mean()

        loss = (
            sequence_weight * seq_loss
            + rollout_weight * rollout_loss
            + counterfactual_weight * cf_loss
            + action_contrastive_weight * action_nce
            + inverse_weight * inverse
            + recon_weight * recon
            + preference_weight * pref_loss
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": float(loss.detach().cpu()),
                    "sequence_loss": float(seq_loss.detach().cpu()),
                    "rollout_loss": float(rollout_loss.detach().cpu()),
                    "counterfactual_loss": float(cf_loss.detach().cpu()),
                    "action_nce": float(action_nce.detach().cpu()),
                    "inverse": float(inverse.detach().cpu()),
                    "recon": float(recon.detach().cpu()),
                    "preference_loss": float(pref_loss.detach().cpu()),
                }
            )
    return model.eval(), history


@torch.inference_mode()
def score_sequence(
    model: HiddenWorldActionModel,
    state_seq: np.ndarray,
    goal_seq: np.ndarray,
    action_seq: np.ndarray,
    device: str,
    negative_actions: list[np.ndarray] | None,
) -> tuple[float, float, float, float, float]:
    if len(state_seq) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    state = torch.from_numpy(state_seq).float().to(device)
    action = torch.from_numpy(action_seq[: max(0, len(state_seq) - 1)]).float().to(device)
    z_obs = model.encode(state)
    goal_vec = goal_seq[-1] if len(goal_seq) else state_seq[-1]
    goal = model.encode(torch.from_numpy(goal_vec[None, :]).float().to(device))[0]
    z_roll = z_obs[0]
    consistency_terms = []
    cf_terms = []
    progress_terms = []
    prev_goal = torch.sum(F.normalize(z_roll, dim=0) * F.normalize(goal, dim=0))
    for step in range(len(action)):
        pred = model.predict_next(z_roll[None, :], action[step : step + 1])[0]
        true_next = z_obs[step + 1]
        pos_score = torch.sum(pred * true_next)
        consistency_terms.append(pos_score)
        if negative_actions:
            step_negs = [arr[step] for arr in negative_actions if len(arr) > step]
            if step_negs:
                neg = torch.from_numpy(np.stack(step_negs, axis=0)).float().to(device)
                z_batch = z_roll[None, :].repeat(len(step_negs), 1)
                neg_pred = model.predict_next(z_batch, neg)
                neg_score = torch.max(torch.sum(neg_pred * true_next[None, :], dim=-1))
            else:
                wrong_action = action[(step + 1) % len(action)] if len(action) > 1 else -action[step]
                neg_score = torch.sum(model.predict_next(z_roll[None, :], wrong_action[None, :])[0] * true_next)
        else:
            wrong_action = action[(step + 1) % len(action)] if len(action) > 1 else -action[step]
            neg_score = torch.sum(model.predict_next(z_roll[None, :], wrong_action[None, :])[0] * true_next)
        cf_terms.append(pos_score - neg_score)
        current_goal = torch.sum(F.normalize(pred, dim=0) * F.normalize(goal, dim=0))
        progress_terms.append(current_goal - prev_goal)
        prev_goal = current_goal
        z_roll = pred
    final_goal = torch.sum(F.normalize(z_roll, dim=0) * F.normalize(goal, dim=0))
    observed_goal = torch.sum(F.normalize(z_obs[-1], dim=0) * F.normalize(goal, dim=0))
    return (
        float(final_goal.detach().cpu()),
        float(observed_goal.detach().cpu()),
        float(torch.stack(consistency_terms).mean().detach().cpu()) if consistency_terms else 0.0,
        float(torch.stack(cf_terms).mean().detach().cpu()) if cf_terms else 0.0,
        float(torch.stack(progress_terms).mean().detach().cpu()) if progress_terms else 0.0,
    )


def write_report(summary: dict[str, Any], args: argparse.Namespace) -> None:
    cv = summary["crossval"]
    c = cv["cem_selected_state_dist"]
    r = cv["rank0_state_dist"]
    o = cv["oracle_best_cem_state_dist"]
    n_cases = len(cv.get("selected_cem", []))
    report = f"""# Lance Raw-Action Sequence World Model

## Idea

This trains a sequence world model on Lance hidden states with raw trajectory-level action chunks. Each transition receives a 10D action chunk corresponding to five raw 2D actions, instead of the previous 4D mean/std summary.

This is not training-free: each fold trains action-conditioned dynamics, rollout consistency, transition-action contrastive alignment, inverse action prediction, and same-case/raw-action counterfactual margins on training cases only.

## Setup

- Candidate cache: `{args.candidate_cache}`
- Real state cache: `{args.real_cache}`
- Action chunk dim: {args.action_chunk_dim}
- Action encoder: `{args.action_encoder}`
- Sequence train rows: `{args.sequence_train_rows}`
- Negative source: `{args.train_negative_source}`
- Preference source: `{args.sequence_preference_source}`
- State PCA dim: {args.state_pca_dim}
- Latent dim: {args.latent_dim}

## Results

| Metric | Value |
| --- | ---: |
| Held-out Qwen pair-label accuracy | {fmt(cv['pair_label_accuracy'])} |
| Held-out expert-vs-CEM oracle pair accuracy | {fmt(cv['oracle_expert_pair_accuracy'])} |
| All-candidate top success | {fmt(cv['all_candidate_top_success'])} |
| CEM better than planner rank0 | {cv['cem_better_than_rank0_count']}/{n_cases} |
| CEM matched oracle-best CEM | {cv['cem_matches_oracle_best_count']}/{n_cases} |

| Selector | Mean | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| NanoWM planner rank0 | {r['mean']:.4f} | {r['median']:.4f} | {r['min']:.4f} | {r['max']:.4f} |
| Raw-action sequence world model | {c['mean']:.4f} | {c['median']:.4f} | {c['min']:.4f} | {c['max']:.4f} |
| Oracle-best CEM | {o['mean']:.4f} | {o['median']:.4f} | {o['min']:.4f} | {o['max']:.4f} |
"""
    Path(args.report_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_output).write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fold-wise Lance raw-action sequence world model.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pairwise", required=True)
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--real-cache", required=True)
    parser.add_argument("--output-scores", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--model-output-dir", required=True)
    parser.add_argument("--state-pca-dim", type=int, default=256)
    parser.add_argument("--action-chunk-dim", type=int, default=10)
    parser.add_argument("--action-encoder", choices=["flat", "temporal_gru"], default="flat")
    parser.add_argument("--action-steps", type=int, default=5)
    parser.add_argument("--action-step-dim", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--sequence-weight", type=float, default=0.5)
    parser.add_argument("--rollout-weight", type=float, default=0.5)
    parser.add_argument("--counterfactual-weight", type=float, default=0.2)
    parser.add_argument("--action-contrastive-weight", type=float, default=0.1)
    parser.add_argument("--inverse-weight", type=float, default=0.02)
    parser.add_argument("--recon-weight", type=float, default=0.02)
    parser.add_argument("--preference-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--goal-weight", type=float, default=1.0)
    parser.add_argument("--observed-goal-weight", type=float, default=0.25)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--counterfactual-score-weight", type=float, default=0.5)
    parser.add_argument("--progress-weight", type=float, default=0.25)
    parser.add_argument("--train-negative-source", choices=["none", "same_case", "raw_intervention", "mixed_raw"], default="mixed_raw")
    parser.add_argument("--max-train-negatives", type=int, default=6)
    parser.add_argument("--score-negative-source", choices=["next_action", "same_case"], default="same_case")
    parser.add_argument("--sequence-preference-source", choices=["none", "oracle", "pairwise"], default="none")
    parser.add_argument("--sequence-preference-oracle-margin", type=float, default=0.0)
    parser.add_argument("--sequence-train-rows", choices=["expert", "cem", "all", "all_non_oracle"], default="expert")
    parser.add_argument("--num-folds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=181)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    rows = read_jsonl(args.manifest)
    pair_rows = read_jsonl(args.pairwise)
    real = np.load(args.real_cache, allow_pickle=False)
    real0_raw = real["z0"].astype(np.float32)
    real1_raw = real["z1"].astype(np.float32)
    state_pca = fit_pca(np.concatenate([real0_raw, real1_raw], axis=0), args.state_pca_dim)
    real_all = apply_pca(np.concatenate([real0_raw, real1_raw], axis=0), state_pca)
    state_mean, state_std = standardize_fit(real_all)
    pred_states, goal_states = build_candidate_states(rows, args.candidate_cache, state_pca, state_mean, state_std)
    cand_actions, action_mean, action_std = build_raw_candidate_actions(rows, pred_states, args.action_chunk_dim)

    case_ids = sorted({str(row["case_id"]) for row in rows})
    folds = make_folds(case_ids, args.num_folds, args.seed)
    all_scores = np.zeros(len(rows), dtype=np.float64)
    goal_scores = np.zeros(len(rows), dtype=np.float64)
    observed_goal_scores = np.zeros(len(rows), dtype=np.float64)
    consistency_scores = np.zeros(len(rows), dtype=np.float64)
    counterfactual_scores = np.zeros(len(rows), dtype=np.float64)
    progress_scores = np.zeros(len(rows), dtype=np.float64)
    fold_metrics = []
    fold_records = []
    Path(args.model_output_dir).mkdir(parents=True, exist_ok=True)

    for fold_idx, test_cases_list in enumerate(folds):
        test_cases = set(test_cases_list)
        train_cases = set(case_ids) - test_cases
        seq_pack = sequence_tensors(
            rows=rows,
            pred_states=pred_states,
            goal_states=goal_states,
            cand_actions=cand_actions,
            pair_rows=pair_rows,
            train_cases=train_cases,
            train_rows=args.sequence_train_rows,
            negative_source=args.train_negative_source,
            max_negatives=args.max_train_negatives,
            action_mean=action_mean,
            action_std=action_std,
            preference_source=args.sequence_preference_source,
            oracle_margin=args.sequence_preference_oracle_margin,
            device=args.device,
        )
        if seq_pack is None:
            raise ValueError(f"No sequence training data for fold {fold_idx}")
        model, history = train_fold_model(
            seq_pack=seq_pack,
            state_dim=args.state_pca_dim,
            action_dim=args.action_chunk_dim,
            action_encoder=args.action_encoder,
            action_steps=args.action_steps,
            action_step_dim=args.action_step_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            sequence_weight=args.sequence_weight,
            rollout_weight=args.rollout_weight,
            counterfactual_weight=args.counterfactual_weight,
            action_contrastive_weight=args.action_contrastive_weight,
            inverse_weight=args.inverse_weight,
            recon_weight=args.recon_weight,
            preference_weight=args.preference_weight,
            temperature=args.temperature,
            seed=args.seed + fold_idx,
            device=args.device,
        )
        for idx, row in enumerate(rows):
            if str(row["case_id"]) not in test_cases:
                continue
            hard_negative_actions = None
            if args.score_negative_source == "same_case":
                hard_negative_actions = [
                    other_actions
                    for other_idx, (other_row, other_actions) in enumerate(zip(rows, cand_actions))
                    if other_idx != idx and str(other_row["case_id"]) == str(row["case_id"])
                ]
            goal, observed_goal, consistency, counterfactual, progress = score_sequence(
                model=model,
                state_seq=pred_states[idx],
                goal_seq=goal_states[idx],
                action_seq=cand_actions[idx],
                device=args.device,
                negative_actions=hard_negative_actions,
            )
            goal_scores[idx] = goal
            observed_goal_scores[idx] = observed_goal
            consistency_scores[idx] = consistency
            counterfactual_scores[idx] = counterfactual
            progress_scores[idx] = progress
            all_scores[idx] = (
                args.goal_weight * goal
                + args.observed_goal_weight * observed_goal
                + args.consistency_weight * consistency
                + args.counterfactual_score_weight * counterfactual
                + args.progress_weight * progress
            )
        fold_metrics.append(evaluate_scores(rows, all_scores, test_cases, pair_rows))
        fold_records.append(
            {
                "fold_idx": fold_idx,
                "test_cases": sorted(test_cases),
                "sequence_train_count": int(seq_pack[0].shape[0]),
                "last_train": history[-1],
            }
        )
        torch.save(
            {
                "model_state": model.state_dict(),
                "fold": fold_records[-1],
                "action_mean": action_mean,
                "action_std": action_std,
            },
            str(Path(args.model_output_dir) / f"fold_{fold_idx:02d}.pt"),
        )
        print(f"[lance_raw_action_sequence_world] fold={fold_idx} seq={seq_pack[0].shape[0]} loss={history[-1]['loss']:.4f}", flush=True)

    summary = {
        "method": "Lance Raw-Action Sequence World Model",
        "manifest": args.manifest,
        "pairwise": args.pairwise,
        "candidate_cache": args.candidate_cache,
        "real_cache": args.real_cache,
        "state_pca_dim": args.state_pca_dim,
        "action_chunk_dim": args.action_chunk_dim,
        "action_encoder": args.action_encoder,
        "action_steps": args.action_steps,
        "action_step_dim": args.action_step_dim,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "sequence_train_rows": args.sequence_train_rows,
        "train_negative_source": args.train_negative_source,
        "max_train_negatives": args.max_train_negatives,
        "score_negative_source": args.score_negative_source,
        "sequence_preference_source": args.sequence_preference_source,
        "preference_weight": args.preference_weight,
        "score_weights": {
            "goal": args.goal_weight,
            "observed_goal": args.observed_goal_weight,
            "consistency": args.consistency_weight,
            "counterfactual": args.counterfactual_score_weight,
            "progress": args.progress_weight,
        },
        "folds": fold_records,
        "fold_metrics": fold_metrics,
        "crossval": merge_dynamics_metrics(fold_metrics),
    }
    out_rows = []
    for idx, row in enumerate(rows):
        out = dict(row)
        out["lance_sequence_world_score"] = float(all_scores[idx])
        out["lance_sequence_world_goal_score"] = float(goal_scores[idx])
        out["lance_sequence_world_observed_goal_score"] = float(observed_goal_scores[idx])
        out["lance_sequence_world_consistency_score"] = float(consistency_scores[idx])
        out["lance_sequence_world_counterfactual_score"] = float(counterfactual_scores[idx])
        out["lance_sequence_world_progress_score"] = float(progress_scores[idx])
        out["lance_raw_action_sequence_world_score"] = float(all_scores[idx])
        out_rows.append(out)
    write_jsonl(args.output_scores, out_rows)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, args)
    cv = summary["crossval"]
    n_cases = len(cv.get("selected_cem", []))
    print(f"[lance_raw_action_sequence_world] pair_acc={fmt(cv['pair_label_accuracy'])}")
    print(f"[lance_raw_action_sequence_world] expert_pair_acc={fmt(cv['oracle_expert_pair_accuracy'])}")
    print(f"[lance_raw_action_sequence_world] cem_mean_dist={cv['cem_selected_state_dist']['mean']:.4f}")
    print(f"[lance_raw_action_sequence_world] better={cv['cem_better_than_rank0_count']}/{n_cases}")
    print(f"[lance_raw_action_sequence_world] oracle={cv['cem_matches_oracle_best_count']}/{n_cases}")
    print(f"[lance_raw_action_sequence_world] report={args.report_output}")


if __name__ == "__main__":
    main()
