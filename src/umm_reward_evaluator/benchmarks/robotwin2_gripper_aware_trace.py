"""Generate RoboTwin 2.0 gripper-aware candidate traces.

Run this script from the root of an official RoboTwin checkout. It monkey-patches
`Base_Task.take_dense_action` during one expert rollout to record compact qpos
targets:

    [left_arm_qpos(6), left_gripper, right_arm_qpos(6), right_gripper]

It then replays simple counterfactual candidate futures with `take_action`.
The output JSONL can be converted with `robotwin2_trace_to_manifest.py`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def class_decorator(task_name: str) -> Any:
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        return env_class()
    except Exception as exc:
        raise SystemExit(f"No such task: {task_name}") from exc


def get_embodiment_config(robot_file: str) -> dict[str, Any]:
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def read_seed(task_name: str, task_config: str, seed: int | None) -> int:
    if seed is not None:
        return seed
    seed_path = Path("data") / task_name / task_config / "seed.txt"
    if not seed_path.exists():
        raise SystemExit(f"missing seed file {seed_path}; pass --seed or run collect_data.py first")
    text = seed_path.read_text(encoding="utf-8").strip().split()
    if not text:
        raise SystemExit(f"empty seed file {seed_path}; pass --seed or run collect_data.py first")
    return int(text[0])


def build_args(task_name: str, task_config: str, *, eval_mode: bool, need_plan: bool) -> dict[str, Any]:
    from envs import CONFIGS_PATH

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args.update(
        {
            "task_name": task_name,
            "task_config": task_config,
            "eval_mode": eval_mode,
            "eval_video_log": False,
            "render_freq": 0,
            "need_plan": need_plan,
        }
    )

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment: str) -> str:
        robot_file = embodiment_types[embodiment]["file_path"]
        if robot_file is None:
            raise ValueError(f"missing embodiment file for {embodiment}")
        return robot_file

    if len(embodiment_type) != 1:
        raise ValueError("this smoke script currently expects a shared dual-arm embodiment")
    args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
    args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
    args["dual_arm_embodied"] = True
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def current_action(env: Any) -> np.ndarray:
    left = np.asarray(env.robot.get_left_arm_jointState(), dtype=np.float32)
    right = np.asarray(env.robot.get_right_arm_jointState(), dtype=np.float32)
    return np.concatenate(
        [
            left[:-1],
            [float(env.robot.get_left_gripper_val())],
            right[:-1],
            [float(env.robot.get_right_gripper_val())],
        ]
    ).astype(float)


def control_endpoint(env: Any, control_seq: dict[str, Any]) -> np.ndarray:
    action = current_action(env)
    if control_seq.get("left_arm") is not None:
        action[:6] = np.asarray(control_seq["left_arm"]["position"][-1], dtype=float)
    if control_seq.get("left_gripper") is not None:
        action[6] = float(control_seq["left_gripper"]["result"][-1])
    if control_seq.get("right_arm") is not None:
        action[7:13] = np.asarray(control_seq["right_arm"]["position"][-1], dtype=float)
    if control_seq.get("right_gripper") is not None:
        action[13] = float(control_seq["right_gripper"]["result"][-1])
    return action


def compact_state(env: Any, obs: dict[str, Any] | None) -> dict[str, Any]:
    state: dict[str, Any] = {}
    if isinstance(obs, dict):
        joint_vec = obs.get("joint_action", {}).get("vector")
        if joint_vec is not None:
            state["joint_action_vector"] = np.asarray(joint_vec).reshape(-1).astype(float).tolist()
    try:
        state["left_gripper"] = float(env.robot.get_left_gripper_val())
        state["right_gripper"] = float(env.robot.get_right_gripper_val())
    except Exception:
        pass
    return state


def restore_replay_state(env: Any, expert_metadata: dict[str, Any]) -> dict[str, Any]:
    """Restore task-local fields that official checks expect after play_once."""
    expert_info = expert_metadata.get("expert_info", {})
    bindings = expert_info.get("info", expert_info) if isinstance(expert_info, dict) else {}
    restored: dict[str, Any] = {}

    arm = bindings.get("{a}") if isinstance(bindings, dict) else None
    if arm is not None:
        arm_text = str(arm).lower()
        if "left" in arm_text:
            env.arm_tag = "left"
            restored["arm_tag"] = "left"
        elif "right" in arm_text:
            env.arm_tag = "right"
            restored["arm_tag"] = "right"
    return restored


def record_expert_actions(task_name: str, task_config: str, seed: int) -> tuple[list[list[float]], dict[str, Any]]:
    env = class_decorator(task_name)
    args = build_args(task_name, task_config, eval_mode=True, need_plan=True)
    env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
    actions: list[list[float]] = []
    original_take_dense_action = env.take_dense_action

    def traced_take_dense_action(control_seq: dict[str, Any], save_freq: int = -1):
        actions.append(control_endpoint(env, control_seq).tolist())
        return original_take_dense_action(control_seq, save_freq=save_freq)

    env.take_dense_action = traced_take_dense_action
    try:
        info = env.play_once()
        success = bool(env.plan_success and env.check_success())
        payload = {"expert_success": success, "expert_info": info, "num_expert_actions": len(actions)}
        return actions, payload
    finally:
        try:
            env.close_env()
        except Exception:
            traceback.print_exc()


def run_candidate(
    *,
    task_name: str,
    task_config: str,
    seed: int,
    instruction: str,
    candidate_id: str,
    rank: int,
    action_seq: list[list[float]],
    expert_metadata: dict[str, Any],
) -> dict[str, Any]:
    env = class_decorator(task_name)
    args = build_args(task_name, task_config, eval_mode=True, need_plan=True)
    env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
    restored_replay_state = restore_replay_state(env, expert_metadata)
    executed: list[list[float]] = []
    state_trace: list[dict[str, Any]] = []
    try:
        for action in action_seq:
            action_arr = np.asarray(action, dtype=float)
            env.take_action(action_arr, action_type="qpos")
            executed.append(action_arr.tolist())
            obs = env.get_obs()
            state_trace.append(compact_state(env, obs))
            if env.eval_success:
                break
        success = bool(env.eval_success or env.check_success())
        return {
            "task_name": task_name,
            "task_config": task_config,
            "seed": seed,
            "instruction": instruction,
            "policy_name": "gripper_aware_expert_trace",
            "ckpt_setting": "none",
            "candidate_id": candidate_id,
            "candidate_seed": rank,
            "candidate_rank_by_planner": rank,
            "video_path": "",
            "actions": executed,
            "success": success,
            "action_type": "qpos",
            "metadata": {
                "future_source": "gripper_aware_expert_trace",
                "future_representation": "actions_and_state_trace",
                "verification_target": "task_success",
                "state_trace": state_trace,
                "expert_metadata": expert_metadata,
                "restored_replay_state": restored_replay_state,
            },
        }
    finally:
        try:
            env.close_env()
        except Exception:
            traceback.print_exc()


def build_default_candidates(actions: list[list[float]]) -> list[tuple[str, int, list[list[float]]]]:
    if not actions:
        return [("noop", 0, [np.zeros(14, dtype=float).tolist()])]
    mid = max(1, len(actions) // 2)
    return [
        ("first_action_rank0", 0, actions[:1]),
        ("full_gripper_aware", 1, actions),
        ("first_half", 2, actions[:mid]),
        ("drop_last", 3, actions[:-1] if len(actions) > 1 else actions[:1]),
        ("reverse", 4, list(reversed(actions))),
        ("noop", 5, [np.zeros(len(actions[0]), dtype=float).tolist()]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", default="demo_clean_smoke")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--instruction")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seed = read_seed(args.task_name, args.task_config, args.seed)
    instruction = args.instruction or args.task_name.replace("_", " ")
    actions, expert_metadata = record_expert_actions(args.task_name, args.task_config, seed)
    if not expert_metadata["expert_success"]:
        raise SystemExit(f"expert rollout did not succeed for {args.task_name} seed {seed}")

    rows: list[dict[str, Any]] = []
    for candidate_id, rank, action_seq in build_default_candidates(actions):
        print(f"running {candidate_id} len={len(action_seq)}", flush=True)
        row = run_candidate(
            task_name=args.task_name,
            task_config=args.task_config,
            seed=seed,
            instruction=instruction,
            candidate_id=candidate_id,
            rank=rank,
            action_seq=action_seq,
            expert_metadata=expert_metadata,
        )
        print(
            f"{candidate_id} success={row['success']} executed={len(row['actions'])}",
            flush=True,
        )
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
