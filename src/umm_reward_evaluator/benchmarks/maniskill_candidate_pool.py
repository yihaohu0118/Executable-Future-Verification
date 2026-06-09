"""Generate diagnostic ManiSkill candidate pools.

This adapter is intentionally a headroom probe. The built-in controllers use
privileged simulator state, so they are not paper-quality baselines. They are
useful for answering the first benchmark question: does an executable candidate
pool contain better alternatives than rank0?
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from umm_reward_evaluator.benchmarks.common import (
    CandidateRow,
    annotate_oracle_best,
    summarize_headroom,
    write_jsonl,
)


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    rank: int
    family: str
    xy_offset: tuple[float, float] = (0.0, 0.0)
    gain: float = 18.0
    grasp_z: float = 0.035
    push_depth: float = 0.12
    speed_scale: float = 1.0


PICK_SPECS = (
    CandidateSpec("rank0_center", 0, "pick_privileged", gain=18.0, grasp_z=0.035),
    CandidateSpec("low_grasp", 1, "pick_privileged", gain=18.0, grasp_z=0.026),
    CandidateSpec("high_grasp", 2, "pick_privileged", gain=18.0, grasp_z=0.046),
    CandidateSpec("x_offset", 3, "pick_privileged", xy_offset=(0.012, 0.0), gain=18.0),
    CandidateSpec("slow_center", 4, "pick_privileged", gain=10.0, grasp_z=0.035),
)

PUSH_SPECS = (
    CandidateSpec("rank0_center", 0, "push_privileged", gain=18.0, push_depth=0.12),
    CandidateSpec("shallow_push", 1, "push_privileged", gain=18.0, push_depth=0.08),
    CandidateSpec("deep_push", 2, "push_privileged", gain=18.0, push_depth=0.16),
    CandidateSpec("left_offset", 3, "push_privileged", xy_offset=(0.0, 0.012), gain=18.0),
    CandidateSpec("slow_center", 4, "push_privileged", gain=10.0, push_depth=0.12),
)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(_to_numpy(value).reshape(-1)[0])


def _success(info: dict[str, Any]) -> bool:
    return bool(_scalar(info.get("success"), 0.0))


def _extra_pos(obs: dict[str, Any], key: str) -> np.ndarray:
    return _to_numpy(obs["extra"][key]).reshape(-1)[:3].astype(np.float32)


def _delta_action(obs: dict[str, Any], target: np.ndarray, gripper: float, gain: float) -> np.ndarray:
    tcp = _extra_pos(obs, "tcp_pose")
    delta = np.clip((target.astype(np.float32) - tcp) * gain, -1.0, 1.0)
    return np.array([delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, gripper], dtype=np.float32)


def _run_stage(env: Any, obs: dict[str, Any], target: np.ndarray, gripper: float, steps: int, gain: float):
    actions: list[list[float]] = []
    rewards = 0.0
    info: dict[str, Any] = {}
    for _ in range(max(1, steps)):
        action = _delta_action(obs, target, gripper, gain)
        obs, reward, terminated, truncated, info = env.step(action[None, :])
        actions.append(action.astype(float).tolist())
        rewards += _scalar(reward)
        if _success(info) or bool(_scalar(terminated)) or bool(_scalar(truncated)):
            break
    return obs, actions, rewards, info


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


def _push_stages(obs: dict[str, Any], spec: CandidateSpec) -> list[tuple[np.ndarray, float, int]]:
    obj = _extra_pos(obs, "obj_pose")
    goal = _extra_pos(obs, "goal_pos")
    direction = goal[:2] - obj[:2]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.array([1.0, 0.0], dtype=np.float32)
    else:
        direction = direction / norm
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    lateral = normal * spec.xy_offset[1] + direction * spec.xy_offset[0]
    behind = obj.copy()
    behind[:2] = obj[:2] - direction * 0.07 + lateral
    behind[2] = 0.055
    contact = obj.copy()
    contact[:2] = obj[:2] - direction * 0.035 + lateral
    contact[2] = 0.045
    through = goal.copy()
    through[:2] = goal[:2] + direction * spec.push_depth + lateral
    through[2] = 0.045
    return [
        (behind.astype(np.float32), 1.0, int(35 * spec.speed_scale)),
        (contact.astype(np.float32), -1.0, int(20 * spec.speed_scale)),
        (through.astype(np.float32), -1.0, int(80 * spec.speed_scale)),
    ]


def _final_distance(env_id: str, obs: dict[str, Any]) -> float | None:
    extra = obs.get("extra", {})
    if env_id == "PickCube-v1" and "obj_to_goal_pos" in extra:
        return float(np.linalg.norm(_to_numpy(extra["obj_to_goal_pos"]).reshape(-1)[:3]))
    if env_id == "PushCube-v1" and "obj_pose" in extra and "goal_pos" in extra:
        obj = _extra_pos(obs, "obj_pose")
        goal = _extra_pos(obs, "goal_pos")
        return float(np.linalg.norm(obj[:2] - goal[:2]))
    return None


def _render_frame(env: Any) -> np.ndarray:
    frame = env.render()
    frame = _to_numpy(frame)
    if frame.ndim == 4:
        frame = frame[0]
    return frame.astype(np.uint8)


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    import imageio.v3 as iio

    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.stack(frames, axis=0), fps=fps)


def run_candidate(
    env: Any,
    env_id: str,
    seed: int,
    spec: CandidateSpec,
    *,
    render_video: bool,
    video_path: Path,
    video_fps: int,
) -> CandidateRow:
    obs, _ = env.reset(seed=seed)
    initial_distance = _final_distance(env_id, obs)
    stages = _pick_stages(obs, spec) if env_id == "PickCube-v1" else _push_stages(obs, spec)
    all_actions: list[list[float]] = []
    total_reward = 0.0
    info: dict[str, Any] = {}
    frames: list[np.ndarray] = []
    if render_video:
        frames.append(_render_frame(env))

    for target, gripper, steps in stages:
        obs, actions, reward, info = _run_stage(env, obs, target, gripper, steps, spec.gain)
        all_actions.extend(actions)
        total_reward += reward
        if render_video and (len(frames) == 0 or len(frames) < 96):
            frames.append(_render_frame(env))
        if _success(info):
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
        candidate_id=spec.candidate_id,
        candidate_rank_by_planner=spec.rank,
        rollout_video_path=str(video_path) if render_video else "",
        rollout_video_layout="prediction_only",
        actions=all_actions,
        oracle_success=_success(info),
        oracle_return=total_reward,
        oracle_progress=progress,
        oracle_state_dist=final_distance,
        metadata={
            "candidate_family": spec.family,
            "controller": "privileged_diagnostic",
            "xy_offset": list(spec.xy_offset),
            "gain": spec.gain,
            "grasp_z": spec.grasp_z,
            "push_depth": spec.push_depth,
            "num_actions": len(all_actions),
        },
    )


def generate_pool(
    env_id: str,
    seeds: list[int],
    output_dir: Path,
    *,
    render_video: bool,
    video_fps: int,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    specs = PICK_SPECS if env_id == "PickCube-v1" else PUSH_SPECS
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
        for seed in seeds:
            for spec in specs:
                video_path = output_dir / "videos" / env_id / f"seed_{seed:04d}_{spec.candidate_id}.mp4"
                row = run_candidate(
                    env,
                    env_id,
                    seed,
                    spec,
                    render_video=render_video,
                    video_path=video_path,
                    video_fps=video_fps,
                )
                rows.append(json.loads(row.to_json()))
    finally:
        env.close()
    rows = annotate_oracle_best(rows)
    return rows, summarize_headroom(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["PickCube-v1", "PushCube-v1"])
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/maniskill_candidate_pool"))
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: dict[str, dict[str, float | int]] = {}
    for env_id in args.tasks:
        if env_id not in {"PickCube-v1", "PushCube-v1"}:
            raise ValueError(f"Unsupported diagnostic controller for {env_id}")
        seeds = list(range(args.seed_offset, args.seed_offset + args.num_cases))
        rows, summary = generate_pool(
            env_id,
            seeds,
            args.output_dir,
            render_video=args.render_video,
            video_fps=args.video_fps,
        )
        manifest_path = args.output_dir / f"{env_id}_candidate_manifest.jsonl"
        write_jsonl(manifest_path, rows)
        all_summaries[env_id] = summary

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(all_summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(all_summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

