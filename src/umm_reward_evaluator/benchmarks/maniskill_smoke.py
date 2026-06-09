"""Smoke-test ManiSkill tasks for external benchmark integration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_TASKS = (
    "PickCube-v1",
    "PushCube-v1",
    "StackCube-v1",
    "PegInsertionSide-v1",
)


def _to_float(value: Any) -> float:
    try:
        return float(value.detach().cpu().reshape(-1)[0])
    except AttributeError:
        return float(value)


def _to_bool(value: Any) -> bool:
    try:
        return bool(value.detach().cpu().reshape(-1)[0])
    except AttributeError:
        return bool(value)


def smoke_task(
    env_id: str,
    *,
    seed: int,
    obs_mode: str,
    control_mode: str,
    render: bool,
    num_steps: int,
) -> dict[str, Any]:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    render_mode = "rgb_array" if render else None
    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode=render_mode,
    )
    try:
        obs, _ = env.reset(seed=seed)
        frame_shape = None
        if render:
            frame = env.render()
            frame_shape = list(frame.shape)
        reward = 0.0
        success = False
        for _ in range(num_steps):
            obs, reward_value, terminated, truncated, info = env.step(env.action_space.sample())
            reward = _to_float(reward_value)
            success = _to_bool(info.get("success", False))
            if _to_bool(terminated) or _to_bool(truncated):
                break
        obs_shape = list(obs.shape) if hasattr(obs, "shape") else type(obs).__name__
        return {
            "env_id": env_id,
            "ok": True,
            "obs_shape": obs_shape,
            "action_space": repr(env.action_space),
            "reward": reward,
            "success": success,
            "frame_shape": frame_shape,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--control-mode", default="pd_ee_delta_pose")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for task in args.tasks:
        try:
            rows.append(
                smoke_task(
                    task,
                    seed=args.seed,
                    obs_mode=args.obs_mode,
                    control_mode=args.control_mode,
                    render=args.render,
                    num_steps=args.num_steps,
                )
            )
        except Exception as exc:  # pragma: no cover - optional dependency path
            rows.append({"env_id": task, "ok": False, "error": f"{type(exc).__name__}: {exc}"})

    payload = {"tasks": rows}
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

