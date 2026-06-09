"""Generate ManiSkill candidate pools from a trained state-BC policy.

This is a bridge between privileged diagnostic candidate families and
policy-generated candidates. Scripted successful rollouts provide supervision,
but test-time candidates are produced by an MLP policy rollout, optionally with
action noise.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from umm_reward_evaluator.benchmarks.common import CandidateRow, annotate_oracle_best, summarize_headroom, write_jsonl
from umm_reward_evaluator.benchmarks.maniskill_candidate_pool import (
    CandidateSpec,
    _delta_action,
    _extra_pos,
    _final_distance,
    _render_frame,
    _run_stage,
    _scalar,
    _success,
    _write_video,
    run_candidate,
)


class BCPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 7),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def flatten_obs(obs: dict[str, Any], *, step: int = 0, max_steps: int = 200) -> np.ndarray:
    parts: list[np.ndarray] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key])
        else:
            arr = _to_numpy(value).astype(np.float32).reshape(-1)
            parts.append(arr)

    visit(obs)
    step_frac = float(step) / max(float(max_steps), 1.0)
    phase = np.array(
        [step_frac, np.sin(2.0 * np.pi * step_frac), np.cos(2.0 * np.pi * step_frac)],
        dtype=np.float32,
    )
    return np.concatenate([*parts, phase]).astype(np.float32)


def pick_demo_specs() -> tuple[CandidateSpec, ...]:
    return (
        CandidateSpec("demo_center", 0, "pick_bc_demo", gain=18.0, grasp_z=0.035),
        CandidateSpec("demo_low", 1, "pick_bc_demo", gain=18.0, grasp_z=0.026),
        CandidateSpec("demo_slow", 2, "pick_bc_demo", gain=10.0, grasp_z=0.035, speed_scale=1.15),
    )


def _pick_stages(obs: dict[str, Any], spec: CandidateSpec) -> list[tuple[np.ndarray, float, int]]:
    obj = _extra_pos(obs, "obj_pose")
    goal = _extra_pos(obs, "goal_pos")
    offset = np.array([spec.xy_offset[0], spec.xy_offset[1], 0.0], dtype=np.float32)
    above = obj + offset
    above[2] = 0.17
    grasp = above.copy()
    grasp[2] = spec.grasp_z
    lift = above.copy()
    lift[2] = 0.18
    return [
        (above, 1.0, int(25 * spec.speed_scale)),
        (grasp, 1.0, int(25 * spec.speed_scale)),
        (grasp, -1.0, int(18 * spec.speed_scale)),
        (lift, -1.0, int(25 * spec.speed_scale)),
        (goal.astype(np.float32), -1.0, int(70 * spec.speed_scale)),
    ]


def collect_demo_pairs(
    env: Any,
    seed: int,
    spec: CandidateSpec,
    *,
    max_steps: int,
) -> tuple[list[np.ndarray], list[np.ndarray], bool]:
    obs, _ = env.reset(seed=seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    info: dict[str, Any] = {}
    step = 0
    for target, gripper, steps in _pick_stages(obs, spec):
        for _ in range(max(1, steps)):
            action = _delta_action(obs, target, gripper, spec.gain)
            xs.append(flatten_obs(obs, step=step, max_steps=max_steps))
            ys.append(action.astype(np.float32))
            obs, _, terminated, truncated, info = env.step(action[None, :])
            step += 1
            if _success(info) or bool(_scalar(terminated)) or bool(_scalar(truncated)):
                break
        if _success(info):
            break
    return xs, ys, _success(info)


def collect_dataset(env: Any, seeds: list[int], *, max_steps: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    attempts = 0
    successful_demos = 0
    for seed in seeds:
        for spec in pick_demo_specs():
            attempts += 1
            demo_xs, demo_ys, ok = collect_demo_pairs(env, seed, spec, max_steps=max_steps)
            if ok:
                xs.extend(demo_xs)
                ys.extend(demo_ys)
                successful_demos += 1
    if not xs:
        raise RuntimeError("No successful demonstrations collected")
    return (
        np.stack(xs).astype(np.float32),
        np.stack(ys).astype(np.float32),
        {"demo_attempts": attempts, "successful_demos": successful_demos, "num_transitions": len(xs)},
    )


def train_bc_policy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[BCPolicy, np.ndarray, np.ndarray, dict[str, float]]:
    torch.manual_seed(seed)
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True) + 1e-6
    x_norm = (x - x_mean) / x_std

    model = BCPolicy(x.shape[1], hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x_t = torch.from_numpy(x_norm)
    y_t = torch.from_numpy(y)
    loss_fn = nn.MSELoss()
    final_loss = 0.0
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_t), y_t)
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().cpu().item())
    return model, x_mean.astype(np.float32), x_std.astype(np.float32), {"bc_train_mse": final_loss}


def policy_action(
    model: BCPolicy,
    obs: dict[str, Any],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    *,
    rng: np.random.Generator,
    noise_std: float,
    step: int,
    max_steps: int,
) -> np.ndarray:
    x = flatten_obs(obs, step=step, max_steps=max_steps)[None, :]
    x = (x - x_mean) / x_std
    with torch.no_grad():
        action = model(torch.from_numpy(x.astype(np.float32))).cpu().numpy()[0]
    if noise_std > 0:
        noise = rng.normal(0.0, noise_std, size=action.shape).astype(np.float32)
        noise[3:6] *= 0.25
        action = action + noise
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def run_policy_candidate(
    env: Any,
    seed: int,
    candidate_id: str,
    rank: int,
    model: BCPolicy,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    *,
    noise_std: float,
    action_seed: int,
    max_steps: int,
    render_video: bool,
    video_path: Path,
    video_fps: int,
) -> CandidateRow:
    env_id = "PickCube-v1"
    obs, _ = env.reset(seed=seed)
    initial_distance = _final_distance(env_id, obs)
    rng = np.random.default_rng(action_seed)
    actions: list[list[float]] = []
    total_reward = 0.0
    info: dict[str, Any] = {}
    frames: list[np.ndarray] = []
    if render_video:
        frames.append(_render_frame(env))
    for step in range(max_steps):
        action = policy_action(
            model,
            obs,
            x_mean,
            x_std,
            rng=rng,
            noise_std=noise_std,
            step=step,
            max_steps=max_steps,
        )
        obs, reward, terminated, truncated, info = env.step(action[None, :])
        actions.append(action.astype(float).tolist())
        total_reward += _scalar(reward)
        if render_video and step % 6 == 0 and len(frames) < 96:
            frames.append(_render_frame(env))
        if _success(info) or bool(_scalar(terminated)) or bool(_scalar(truncated)):
            break
    final_distance = _final_distance(env_id, obs)
    progress = None
    if initial_distance is not None and final_distance is not None:
        progress = initial_distance - final_distance
    if render_video:
        _write_video(video_path, frames, video_fps)
    return CandidateRow(
        benchmark="maniskill",
        suite=env_id,
        task_name=env_id,
        case_id=f"{env_id}:seed={seed}",
        candidate_id=candidate_id,
        candidate_rank_by_planner=rank,
        rollout_video_path=str(video_path) if render_video else "",
        rollout_video=str(video_path) if render_video else None,
        rollout_video_layout="prediction_only",
        actions=actions,
        oracle_success=_success(info),
        oracle_return=total_reward,
        oracle_progress=progress,
        oracle_state_dist=final_distance,
        metadata={
            "candidate_family": "pick_bc_policy",
            "controller": "state_bc_policy",
            "noise_std": noise_std,
            "action_seed": action_seed,
            "num_actions": len(actions),
        },
    )


def generate_policy_pool(
    *,
    demo_seeds: list[int],
    eval_seeds: list[int],
    output_dir: Path,
    noise_stds: list[float],
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
    max_steps: int,
    render_video: bool,
    video_fps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    env_id = "PickCube-v1"
    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array" if render_video else None,
        max_episode_steps=max_steps,
    )
    rows: list[dict[str, Any]] = []
    try:
        x, y, demo_summary = collect_dataset(env, demo_seeds, max_steps=max_steps)
        model, x_mean, x_std, train_summary = train_bc_policy(x, y, hidden=hidden, epochs=epochs, lr=lr, seed=seed)
        for eval_seed in eval_seeds:
            for rank, noise_std in enumerate(noise_stds):
                candidate_id = f"bc_{rank:02d}_noise{noise_std:.3f}"
                video_path = output_dir / "videos" / env_id / f"seed_{eval_seed:04d}_{candidate_id}.mp4"
                row = run_policy_candidate(
                    env,
                    eval_seed,
                    candidate_id,
                    rank,
                    model,
                    x_mean,
                    x_std,
                    noise_std=noise_std,
                    action_seed=seed * 1_000_003 + eval_seed * 101 + rank,
                    max_steps=max_steps,
                    render_video=render_video,
                    video_path=video_path,
                    video_fps=video_fps,
                )
                rows.append(json.loads(row.to_json()))
    finally:
        env.close()
    rows = annotate_oracle_best(rows)
    summary = summarize_headroom(rows)
    summary.update(demo_summary)
    summary.update(train_summary)
    summary["pool"] = "pick_bc_policy"
    summary["demo_cases"] = len(demo_seeds)
    summary["eval_cases"] = len(eval_seeds)
    summary["noise_stds"] = noise_stds
    return rows, summary


def parse_noise_stds(raw: str, *, num_candidates: int | None) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if num_candidates is not None:
        if len(values) == 1:
            values = [values[0]] * num_candidates
        elif len(values) < num_candidates:
            values = values + [values[-1]] * (num_candidates - len(values))
        else:
            values = values[:num_candidates]
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-cases", type=int, default=32)
    parser.add_argument("--demo-seed-offset", type=int, default=1000)
    parser.add_argument("--eval-cases", type=int, default=50)
    parser.add_argument("--eval-seed-offset", type=int, default=0)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--noise-stds", default="0.0,0.015,0.025,0.035,0.045,0.060,0.080,0.100")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maniskill_pickcube_bc_policy_pool"))
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    demo_seeds = list(range(args.demo_seed_offset, args.demo_seed_offset + args.demo_cases))
    eval_seeds = list(range(args.eval_seed_offset, args.eval_seed_offset + args.eval_cases))
    rows, summary = generate_policy_pool(
        demo_seeds=demo_seeds,
        eval_seeds=eval_seeds,
        output_dir=args.output_dir,
        noise_stds=parse_noise_stds(args.noise_stds, num_candidates=args.num_candidates),
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        max_steps=args.max_steps,
        render_video=args.render_video,
        video_fps=args.video_fps,
    )
    manifest_path = args.output_dir / "PickCube-v1_candidate_manifest.jsonl"
    write_jsonl(manifest_path, rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"PickCube-v1": summary}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"PickCube-v1": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
