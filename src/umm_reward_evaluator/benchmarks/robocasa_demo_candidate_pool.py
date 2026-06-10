"""Generate RoboCasa365 candidate pools from official demonstration episodes.

This is a headroom diagnostic, not a deployable planner. It replays official
demonstration actions from the same initial simulator state, then evaluates
simple counterfactual action transforms such as under-actuation, noise, and
truncation. The resulting JSONL uses the shared candidate schema so the same
selectors used for ManiSkill can be evaluated on RoboCasa.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
class ActionProfile:
    candidate_id: str
    rank: int
    kind: str
    scale: float = 1.0
    noise_std: float = 0.0
    truncate_frac: float = 1.0
    seed_offset: int = 0
    prior_score: float = 0.0


DEFAULT_PROFILES = (
    ActionProfile("cand_00_underactuated_065", 0, "scale", scale=0.65),
    ActionProfile("cand_01_demo_original", 1, "original"),
    ActionProfile("cand_02_underactuated_085", 2, "scale", scale=0.85),
    ActionProfile("cand_03_noisy_003", 3, "noise", noise_std=0.03, seed_offset=3),
    ActionProfile("cand_04_truncated_080", 4, "truncate", truncate_frac=0.80),
)


def conservative_prior_score(actions: np.ndarray) -> float:
    """Non-oracle policy prior: prefer smaller, smoother action chunks."""
    if actions.size == 0:
        return 0.0
    action_energy = float(np.mean(np.square(actions)))
    smoothness = float(np.mean(np.square(np.diff(actions, axis=0)))) if len(actions) > 1 else 0.0
    return -(action_energy + 0.1 * smoothness)


def random_profiles(
    actions: np.ndarray,
    *,
    episode_index: int,
    num_candidates: int,
    seed: int,
    include_original: bool,
) -> tuple[ActionProfile, ...]:
    rng = np.random.default_rng(seed + 1009 * episode_index)
    profiles: list[ActionProfile] = []
    raw_profiles: list[ActionProfile] = []
    if include_original:
        raw_profiles.append(ActionProfile("raw_demo_original", 0, "original"))
    while len(raw_profiles) < num_candidates:
        scale = float(rng.uniform(0.55, 1.15))
        noise_std = float(rng.choice([0.0, 0.01, 0.02, 0.03, 0.05]))
        truncate_frac = float(rng.choice([1.0, 1.0, 0.95, 0.90, 0.80]))
        raw_profiles.append(
            ActionProfile(
                f"raw_scale{scale:.3f}_noise{noise_std:.3f}_trunc{truncate_frac:.2f}",
                0,
                "scale_noise_truncate",
                scale=scale,
                noise_std=noise_std,
                truncate_frac=truncate_frac,
                seed_offset=len(raw_profiles),
            )
        )

    scored = []
    for raw_profile in raw_profiles:
        transformed = transform_actions(actions, raw_profile, case_seed=episode_index)
        scored.append((conservative_prior_score(transformed), raw_profile))

    # Higher prior score means more likely under the conservative non-oracle prior.
    for rank, (score, raw_profile) in enumerate(sorted(scored, key=lambda item: item[0], reverse=True)):
        profiles.append(
            ActionProfile(
                candidate_id=f"cand_{rank:02d}",
                rank=rank,
                kind=raw_profile.kind,
                scale=raw_profile.scale,
                noise_std=raw_profile.noise_std,
                truncate_frac=raw_profile.truncate_frac,
                seed_offset=raw_profile.seed_offset,
                prior_score=float(score),
            )
        )
    return tuple(profiles)


def _import_robocasa_playback() -> tuple[Any, Any, Any]:
    import robosuite  # noqa: F401
    import robocasa  # noqa: F401
    import robocasa.utils.lerobot_utils as lerobot_utils
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

    return robosuite, lerobot_utils, reset_to


def _make_env(dataset: Path) -> Any:
    robosuite, lerobot_utils, _ = _import_robocasa_playback()
    env_meta = lerobot_utils.get_env_metadata(dataset)
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False
    return robosuite.make(**env_kwargs), env_meta


def _initial_state(dataset: Path, episode_index: int) -> dict[str, Any]:
    _, lerobot_utils, _ = _import_robocasa_playback()
    states = lerobot_utils.get_episode_states(dataset, episode_index)
    return {
        "states": states[0],
        "model": lerobot_utils.get_episode_model_xml(dataset, episode_index),
        "ep_meta": json.dumps(lerobot_utils.get_episode_meta(dataset, episode_index)),
    }


def _episode_actions(dataset: Path, episode_index: int) -> np.ndarray:
    _, lerobot_utils, _ = _import_robocasa_playback()
    return np.asarray(
        lerobot_utils.get_episode_actions(dataset, episode_index, abs_actions=False),
        dtype=np.float32,
    )


def _episode_instruction(dataset: Path, episode_index: int) -> str | None:
    _, lerobot_utils, _ = _import_robocasa_playback()
    meta = lerobot_utils.get_episode_meta(dataset, episode_index)
    try:
        from robocasa.scripts.dataset_scripts.playback_utils import (
            resolve_instruction_from_ep_meta,
        )

        return resolve_instruction_from_ep_meta(meta)
    except Exception:
        return meta.get("lang") or meta.get("task_description")


def transform_actions(
    actions: np.ndarray,
    profile: ActionProfile,
    *,
    case_seed: int,
) -> np.ndarray:
    transformed = actions.copy()
    if profile.kind == "original":
        return transformed
    if profile.kind == "scale":
        return np.clip(transformed * profile.scale, -1.0, 1.0)
    if profile.kind == "noise":
        rng = np.random.default_rng(case_seed + profile.seed_offset)
        noise = rng.normal(0.0, profile.noise_std, size=transformed.shape)
        return np.clip(transformed + noise, -1.0, 1.0)
    if profile.kind == "scale_noise_truncate":
        transformed = transformed * profile.scale
        if profile.noise_std > 0.0:
            rng = np.random.default_rng(case_seed + profile.seed_offset)
            transformed = transformed + rng.normal(0.0, profile.noise_std, size=transformed.shape)
        if profile.truncate_frac < 1.0:
            keep = max(1, int(round(len(transformed) * profile.truncate_frac)))
            transformed[keep:] = 0.0
        return np.clip(transformed, -1.0, 1.0)
    if profile.kind == "truncate":
        keep = max(1, int(round(len(transformed) * profile.truncate_frac)))
        transformed[keep:] = 0.0
        return transformed
    raise ValueError(f"unknown profile kind: {profile.kind}")


def _success(env: Any) -> bool:
    value = env._check_success()
    try:
        return bool(np.asarray(value).reshape(-1)[0])
    except Exception:
        return bool(value)


def run_actions(
    env: Any,
    initial_state: dict[str, Any],
    actions: np.ndarray,
    *,
    max_steps: int | None,
) -> tuple[bool, float, int]:
    _, _, reset_to = _import_robocasa_playback()
    reset_to(env, initial_state)

    total_reward = 0.0
    steps = len(actions) if max_steps is None else min(len(actions), max_steps)
    success = _success(env)
    for t in range(steps):
        _, reward, done, _ = env.step(actions[t])
        try:
            total_reward += float(reward)
        except Exception:
            pass
        success = success or _success(env)
        if done:
            break
    return success, total_reward, t + 1 if steps else 0


def action_snapshot(actions: np.ndarray, stride: int) -> list[list[float]]:
    if stride <= 1:
        sampled = actions
    else:
        sampled = actions[::stride]
        if len(sampled) == 0 or not np.array_equal(sampled[-1], actions[-1]):
            sampled = np.concatenate([sampled, actions[-1:]], axis=0)
    return np.round(sampled.astype(np.float32), 5).tolist()


def build_pool(
    dataset: Path,
    *,
    task_name: str,
    suite: str,
    start_episode: int,
    num_episodes: int,
    profiles: tuple[ActionProfile, ...],
    profile_mode: str,
    num_candidates: int,
    seed: int,
    include_original: bool,
    max_steps: int | None,
    action_stride: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, lerobot_utils, _ = _import_robocasa_playback()
    episodes = lerobot_utils.get_episodes(dataset)
    if start_episode < 0:
        raise ValueError("start_episode must be non-negative")
    episodes = episodes[start_episode:]
    if num_episodes > 0:
        episodes = episodes[:num_episodes]

    env, env_meta = _make_env(dataset)
    rows: list[dict[str, Any]] = []
    try:
        for local_episode_index, episode in enumerate(episodes):
            episode_index = start_episode + local_episode_index
            actions = _episode_actions(dataset, episode_index)
            initial_state = _initial_state(dataset, episode_index)
            instruction = _episode_instruction(dataset, episode_index)
            case_id = f"{task_name}:ep_{episode_index:04d}"
            case_profiles = (
                random_profiles(
                    actions,
                    episode_index=episode_index,
                    num_candidates=num_candidates,
                    seed=seed,
                    include_original=include_original,
                )
                if profile_mode == "random"
                else profiles
            )
            for profile in case_profiles:
                print(
                    f"[robocasa_demo_pool] episode={episode_index} candidate={profile.candidate_id}",
                    flush=True,
                )
                candidate_actions = transform_actions(
                    actions,
                    profile,
                    case_seed=episode_index,
                )
                success, total_reward, executed_steps = run_actions(
                    env,
                    initial_state,
                    candidate_actions,
                    max_steps=max_steps,
                )
                row = CandidateRow(
                    benchmark="RoboCasa365",
                    suite=suite,
                    task_name=task_name,
                    case_id=case_id,
                    candidate_id=profile.candidate_id,
                    candidate_rank_by_planner=profile.rank,
                    rollout_video_path="",
                    rollout_video_layout="none",
                    actions=action_snapshot(candidate_actions, action_stride),
                    oracle_success=success,
                    instruction=instruction,
                    oracle_return=total_reward,
                    oracle_progress=1.0 if success else 0.0,
                    metadata={
                        "dataset": str(dataset),
                        "episode": episode.stem,
                        "episode_index": episode_index,
                        "env_name": env_meta.get("env_name"),
                        "profile": profile.kind,
                        "scale": profile.scale,
                        "noise_std": profile.noise_std,
                        "truncate_frac": profile.truncate_frac,
                        "conservative_prior_score": profile.prior_score,
                        "original_action_shape": list(actions.shape),
                        "stored_action_stride": action_stride,
                        "executed_steps": executed_steps,
                    },
                )
                rows.append(json.loads(row.to_json()))
    finally:
        env.close()

    rows = annotate_oracle_best(rows)
    summary = summarize_headroom(rows)
    summary.update(
        {
            "benchmark": "RoboCasa365",
            "suite": suite,
            "task_name": task_name,
            "dataset": str(dataset),
            "profile_mode": profile_mode,
            "start_episode": start_episode,
            "num_profiles": num_candidates if profile_mode == "random" else len(profiles),
            "include_original": include_original,
            "ranking_prior": "conservative_action_energy",
        }
    )
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--suite", default="target-human")
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--profile-mode", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude-original", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--action-stride", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/robocasa365_demo_pool"))
    args = parser.parse_args()

    rows, summary = build_pool(
        args.dataset,
        task_name=args.task_name,
        suite=args.suite,
        start_episode=args.start_episode,
        num_episodes=args.num_episodes,
        profiles=DEFAULT_PROFILES,
        profile_mode=args.profile_mode,
        num_candidates=args.num_candidates,
        seed=args.seed,
        include_original=not args.exclude_original,
        max_steps=args.max_steps,
        action_stride=args.action_stride,
    )

    manifest_path = args.output_dir / f"{args.task_name}_candidate_manifest.jsonl"
    summary_path = args.output_dir / f"{args.task_name}_summary.json"
    write_jsonl(manifest_path, rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"manifest={manifest_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
