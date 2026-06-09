"""Generate PickCube candidates from a learned grasp-parameter proposal."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from umm_reward_evaluator.benchmarks.common import annotate_oracle_best, load_jsonl, summarize_headroom, write_jsonl
from umm_reward_evaluator.benchmarks.maniskill_candidate_pool import CandidateSpec, run_candidate


PARAM_DIM = 4


def row_params(row: dict[str, Any]) -> np.ndarray | None:
    metadata = row.get("metadata") or {}
    if not {"xy_offset", "gain", "grasp_z"}.issubset(metadata):
        return None
    xy = metadata["xy_offset"]
    return np.array([float(xy[0]), float(xy[1]), float(metadata["gain"]), float(metadata["grasp_z"])], dtype=np.float32)


def fit_success_gaussian(rows: list[dict[str, Any]], *, min_std: float) -> dict[str, Any]:
    success_params = [row_params(row) for row in rows if bool(row.get("oracle_success"))]
    success_params = [param for param in success_params if param is not None]
    if len(success_params) < 2:
        raise RuntimeError("Need at least two successful parameter rows to fit proposal")
    x = np.stack(success_params).astype(np.float32)
    mean = x.mean(axis=0)
    cov = np.cov(x.T).astype(np.float32)
    cov = cov + np.eye(PARAM_DIM, dtype=np.float32) * (min_std**2)
    return {"mean": mean, "cov": cov, "num_success_params": len(x)}


def logpdf_diag(param: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    z = (param - mean) / std
    return float(-0.5 * np.sum(z * z + 2.0 * np.log(std) + np.log(2.0 * np.pi)))


def sample_specs(
    *,
    proposal: dict[str, Any],
    seed: int,
    num_candidates: int,
    temperature: float,
    rank_by_likelihood: bool,
) -> tuple[CandidateSpec, ...]:
    rng = np.random.default_rng(seed)
    mean = np.asarray(proposal["mean"], dtype=np.float32)
    cov = np.asarray(proposal["cov"], dtype=np.float32) * float(temperature)
    samples = rng.multivariate_normal(mean, cov, size=num_candidates).astype(np.float32)
    samples[:, 0] = np.clip(samples[:, 0], -0.035, 0.035)
    samples[:, 1] = np.clip(samples[:, 1], -0.035, 0.035)
    samples[:, 2] = np.clip(samples[:, 2], 8.0, 26.0)
    samples[:, 3] = np.clip(samples[:, 3], 0.020, 0.065)

    order = list(range(num_candidates))
    if rank_by_likelihood:
        std = np.sqrt(np.diag(cov) + 1e-6).astype(np.float32)
        order = sorted(order, key=lambda idx: logpdf_diag(samples[idx], mean, std), reverse=True)

    specs = []
    for rank, idx in enumerate(order):
        xy_x, xy_y, gain, grasp_z = samples[idx]
        specs.append(
            CandidateSpec(
                candidate_id=f"learned_{rank:02d}_z{grasp_z:.3f}_g{gain:.1f}",
                rank=rank,
                family="pick_learned_grasp_proposal",
                xy_offset=(float(xy_x), float(xy_y)),
                gain=float(gain),
                grasp_z=float(grasp_z),
            )
        )
    return tuple(specs)


def generate_pool(
    *,
    train_manifest: Path,
    eval_seeds: list[int],
    output_dir: Path,
    num_candidates: int,
    temperature: float,
    min_std: float,
    rank_by_likelihood: bool,
    seed: int,
    render_video: bool,
    video_fps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    env_id = "PickCube-v1"
    train_rows = load_jsonl(train_manifest)
    proposal = fit_success_gaussian(train_rows, min_std=min_std)
    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array" if render_video else None,
        max_episode_steps=200,
    )
    rows: list[dict[str, Any]] = []
    try:
        for eval_seed in eval_seeds:
            specs = sample_specs(
                proposal=proposal,
                seed=seed * 1_000_003 + eval_seed,
                num_candidates=num_candidates,
                temperature=temperature,
                rank_by_likelihood=rank_by_likelihood,
            )
            for spec in specs:
                video_path = output_dir / "videos" / env_id / f"seed_{eval_seed:04d}_{spec.candidate_id}.mp4"
                row = run_candidate(
                    env,
                    env_id,
                    eval_seed,
                    spec,
                    render_video=render_video,
                    video_path=video_path,
                    video_fps=video_fps,
                )
                payload = json.loads(row.to_json())
                payload["metadata"]["proposal_train_manifest"] = str(train_manifest)
                payload["metadata"]["proposal_temperature"] = temperature
                payload["metadata"]["proposal_rank_by_likelihood"] = rank_by_likelihood
                rows.append(payload)
    finally:
        env.close()
    rows = annotate_oracle_best(rows)
    summary = summarize_headroom(rows)
    summary["pool"] = "pick_learned_grasp_proposal"
    summary["num_candidates"] = num_candidates
    summary["temperature"] = temperature
    summary["min_std"] = min_std
    summary["rank_by_likelihood"] = rank_by_likelihood
    summary["num_success_params"] = int(proposal["num_success_params"])
    summary["proposal_mean"] = np.asarray(proposal["mean"]).astype(float).tolist()
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-cases", type=int, default=50)
    parser.add_argument("--eval-seed-offset", type=int, default=50)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--min-std", type=float, default=0.006)
    parser.add_argument("--rank-by-likelihood", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maniskill_pickcube_learned_grasp_proposal"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_seeds = list(range(args.eval_seed_offset, args.eval_seed_offset + args.eval_cases))
    rows, summary = generate_pool(
        train_manifest=args.train_manifest,
        eval_seeds=eval_seeds,
        output_dir=args.output_dir,
        num_candidates=args.num_candidates,
        temperature=args.temperature,
        min_std=args.min_std,
        rank_by_likelihood=args.rank_by_likelihood,
        seed=args.seed,
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
