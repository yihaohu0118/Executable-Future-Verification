"""Smoke-test RoboCasa365 tasks for the 2025-2026 benchmark track.

RoboCasa is intentionally kept as an optional dependency. Run this script
inside an environment where the official robocasa and robosuite packages are
installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_TASKS = (
    "robocasa/PickPlaceCounterToCabinet",
    "robocasa/PickPlaceCounterToSink",
    "robocasa/CloseCabinet",
    "robocasa/TurnOnSinkFaucet",
)


def _import_robocasa_envs() -> None:
    import robocasa  # noqa: F401
    import robocasa.wrappers.gym_wrapper  # noqa: F401


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "shape"):
        return {"type": type(value).__name__, "shape": list(value.shape)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return type(value).__name__


def list_robocasa_envs(limit: int | None = None) -> list[str]:
    import gymnasium as gym

    _import_robocasa_envs()

    env_ids = sorted(
        env_id
        for env_id in gym.registry.keys()
        if str(env_id).startswith("robocasa/")
    )
    if limit is not None:
        return env_ids[:limit]
    return env_ids


def smoke_task(
    env_id: str,
    *,
    seed: int,
    split: str,
    num_steps: int,
) -> dict[str, Any]:
    import gymnasium as gym

    _import_robocasa_envs()

    env = gym.make(env_id, split=split, seed=seed)
    try:
        obs, reset_info = env.reset()
        rewards: list[float] = []
        terminated = False
        truncated = False
        last_info: dict[str, Any] = {}
        for _ in range(num_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            last_info = dict(info)
            if terminated or truncated:
                break
        return {
            "env_id": env_id,
            "split": split,
            "seed": seed,
            "ok": True,
            "obs": _jsonable(obs),
            "reset_info": _jsonable(reset_info),
            "action_space": repr(env.action_space),
            "num_steps": len(rewards),
            "rewards": rewards,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "step_info": _jsonable(last_info),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="target")
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--list-envs", action="store_true")
    parser.add_argument("--list-limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload: dict[str, Any] = {}
    if args.list_envs:
        payload["env_ids"] = list_robocasa_envs(args.list_limit)

    rows = []
    for task in args.tasks:
        try:
            rows.append(
                smoke_task(
                    task,
                    seed=args.seed,
                    split=args.split,
                    num_steps=args.num_steps,
                )
            )
        except Exception as exc:  # pragma: no cover - optional dependency path
            rows.append(
                {
                    "env_id": task,
                    "split": args.split,
                    "seed": args.seed,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    payload["tasks"] = rows

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
