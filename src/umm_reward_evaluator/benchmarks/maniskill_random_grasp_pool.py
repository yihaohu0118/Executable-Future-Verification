"""Generate randomized PickCube grasp candidate pools.

This is a stronger diagnostic than the fixed-family brittle_grasp pool. Each
case gets one brittle rank0 candidate plus continuously sampled counterfactual
grasp candidates, reducing the chance that a selector only learns a fixed
candidate-family label.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from umm_reward_evaluator.benchmarks.common import annotate_oracle_best, summarize_headroom, write_jsonl
from umm_reward_evaluator.benchmarks.maniskill_candidate_pool import CandidateSpec, run_candidate


def sampled_specs(seed: int, num_candidates: int) -> tuple[CandidateSpec, ...]:
    rng = np.random.default_rng(seed)
    specs: list[CandidateSpec] = []

    rank0_z = float(rng.uniform(0.048, 0.065))
    specs.append(
        CandidateSpec(
            candidate_id="cand_00_rank0_highz",
            rank=0,
            family="pick_random_grasp",
            xy_offset=(float(rng.normal(0.0, 0.006)), float(rng.normal(0.0, 0.006))),
            gain=float(rng.uniform(14.0, 22.0)),
            grasp_z=rank0_z,
        )
    )

    for rank in range(1, num_candidates):
        if rank == 1:
            # Ensure at least one likely recovery candidate.
            grasp_z = float(rng.uniform(0.024, 0.032))
            gain = float(rng.uniform(16.0, 22.0))
            xy = (float(rng.normal(0.0, 0.004)), float(rng.normal(0.0, 0.004)))
        else:
            grasp_z = float(rng.uniform(0.024, 0.052))
            gain = float(rng.uniform(9.0, 24.0))
            xy = (float(rng.normal(0.0, 0.012)), float(rng.normal(0.0, 0.012)))
        specs.append(
            CandidateSpec(
                candidate_id=f"cand_{rank:02d}_z{grasp_z:.3f}",
                rank=rank,
                family="pick_random_grasp",
                xy_offset=xy,
                gain=gain,
                grasp_z=grasp_z,
            )
        )
    return tuple(specs)


def generate_pool(
    seeds: list[int],
    output_dir: Path,
    *,
    num_candidates: int,
    render_video: bool,
    video_fps: int,
) -> tuple[list[dict], dict]:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    env_id = "PickCube-v1"
    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array" if render_video else None,
        max_episode_steps=200,
    )
    rows = []
    try:
        for seed in seeds:
            for spec in sampled_specs(seed, num_candidates):
                video_path = output_dir / "videos" / env_id / f"seed_{seed:04d}_{spec.candidate_id}.mp4"
                row = run_candidate(
                    env,
                    env_id,
                    seed,
                    replace(spec, family="pick_random_grasp"),
                    render_video=render_video,
                    video_path=video_path,
                    video_fps=video_fps,
                )
                payload = json.loads(row.to_json())
                payload["metadata"]["random_pool_seed"] = seed
                rows.append(payload)
    finally:
        env.close()
    rows = annotate_oracle_best(rows)
    summary = summarize_headroom(rows)
    summary["pool"] = "pick_random_grasp"
    summary["num_candidates"] = num_candidates
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cases", type=int, default=50)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maniskill_pickcube_random_grasp_pool"))
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_offset, args.seed_offset + args.num_cases))
    rows, summary = generate_pool(
        seeds,
        args.output_dir,
        num_candidates=args.num_candidates,
        render_video=args.render_video,
        video_fps=args.video_fps,
    )
    manifest_path = args.output_dir / "PickCube-v1_candidate_manifest.jsonl"
    write_jsonl(manifest_path, rows)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps({"PickCube-v1": summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"PickCube-v1": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

