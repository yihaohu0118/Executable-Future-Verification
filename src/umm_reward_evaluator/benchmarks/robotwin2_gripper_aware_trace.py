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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    rank: int
    actions: list[list[float]]
    candidate_source: str


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


def read_seed_list(task_name: str, task_config: str) -> list[int]:
    seed_path = Path("data") / task_name / task_config / "seed.txt"
    if not seed_path.exists():
        raise SystemExit(f"missing seed file {seed_path}; run collect_data.py first")
    seeds = [int(item) for item in seed_path.read_text(encoding="utf-8").strip().split()]
    if not seeds:
        raise SystemExit(f"empty seed file {seed_path}; run collect_data.py first")
    return seeds


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
    candidate_source: str,
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
                "candidate_source": candidate_source,
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


def as_action_list(actions: np.ndarray) -> list[list[float]]:
    return actions.astype(float).tolist()


def action_matrix(actions: list[list[float]]) -> np.ndarray:
    if not actions:
        return np.zeros((1, 14), dtype=float)
    return np.asarray(actions, dtype=float)


def shifted_gripper(actions: list[list[float]], shift: int) -> list[list[float]]:
    arr = action_matrix(actions).copy()
    if arr.shape[1] < 14 or len(arr) <= 1:
        return as_action_list(arr)
    for gripper_col in (6, 13):
        arr[:, gripper_col] = np.roll(arr[:, gripper_col], shift)
        if shift > 0:
            arr[:shift, gripper_col] = arr[shift, gripper_col]
        elif shift < 0:
            arr[shift:, gripper_col] = arr[shift - 1, gripper_col]
    return as_action_list(arr)


def repeat_middle(actions: list[list[float]], repeats: int = 2) -> list[list[float]]:
    if not actions:
        return actions
    mid = len(actions) // 2
    return actions[:mid] + [actions[mid]] * max(1, repeats) + actions[mid:]


def time_subsample_then_hold(actions: list[list[float]], stride: int = 2) -> list[list[float]]:
    if len(actions) <= 2:
        return actions
    kept = actions[::stride]
    if kept[-1] != actions[-1]:
        kept.append(actions[-1])
    return kept


def perturb_contact_segment(actions: list[list[float]], *, scale: float = 0.08) -> list[list[float]]:
    arr = action_matrix(actions).copy()
    if len(arr) <= 2:
        return as_action_list(arr)
    start = max(0, int(len(arr) * 0.6))
    base = arr[start - 1].copy() if start > 0 else arr[0].copy()
    arr[start:, :6] = base[:6] + (arr[start:, :6] - base[:6]) * (1.0 + scale)
    arr[start:, 7:13] = base[7:13] + (arr[start:, 7:13] - base[7:13]) * (1.0 - scale)
    return as_action_list(arr)


def build_default_candidates(actions: list[list[float]]) -> list[CandidateSpec]:
    if not actions:
        return [CandidateSpec("noop", 0, [np.zeros(14, dtype=float).tolist()], "noop")]
    mid = max(1, len(actions) // 2)
    return [
        CandidateSpec("first_action_rank0", 0, actions[:1], "first_action"),
        CandidateSpec("full_gripper_aware", 1, actions, "full_expert_trace"),
        CandidateSpec("first_half", 2, actions[:mid], "prefix_truncation"),
        CandidateSpec("drop_last", 3, actions[:-1] if len(actions) > 1 else actions[:1], "suffix_truncation"),
        CandidateSpec("reverse", 4, list(reversed(actions)), "time_reverse"),
        CandidateSpec("noop", 5, [np.zeros(len(actions[0]), dtype=float).tolist()], "noop"),
    ]


def build_antitemplate_candidates(actions: list[list[float]]) -> list[CandidateSpec]:
    candidates = build_default_candidates(actions)
    if not actions:
        return candidates
    candidates.extend(
        [
            CandidateSpec("repeat_middle", 6, repeat_middle(actions, repeats=2), "time_warp_hard_positive_probe"),
            CandidateSpec("stride2_hold_endpoint", 7, time_subsample_then_hold(actions, stride=2), "time_warp_hard_positive_probe"),
            CandidateSpec("gripper_early_1", 8, shifted_gripper(actions, shift=-1), "matched_gripper_timing_negative_probe"),
            CandidateSpec("gripper_late_1", 9, shifted_gripper(actions, shift=1), "matched_gripper_timing_negative_probe"),
            CandidateSpec("contact_joint_perturb", 10, perturb_contact_segment(actions), "matched_contact_direction_negative_probe"),
        ]
    )
    return candidates


def build_candidates(actions: list[list[float]], preset: str) -> list[CandidateSpec]:
    if preset == "default":
        return build_default_candidates(actions)
    if preset == "anti_template":
        return build_antitemplate_candidates(actions)
    raise ValueError(f"unknown candidate preset: {preset}")


def run_one_seed(
    *,
    task_name: str,
    task_config: str,
    seed: int,
    instruction: str,
    output: Path,
    skip_existing: bool,
    candidate_preset: str,
) -> None:
    if skip_existing and output.exists() and output.stat().st_size > 0:
        print(f"skip existing {output}", flush=True)
        return

    actions, expert_metadata = record_expert_actions(task_name, task_config, seed)
    if not expert_metadata["expert_success"]:
        raise SystemExit(f"expert rollout did not succeed for {task_name} seed {seed}")

    rows: list[dict[str, Any]] = []
    for candidate in build_candidates(actions, candidate_preset):
        print(f"running seed={seed} {candidate.candidate_id} len={len(candidate.actions)}", flush=True)
        row = run_candidate(
            task_name=task_name,
            task_config=task_config,
            seed=seed,
            instruction=instruction,
            candidate_id=candidate.candidate_id,
            candidate_source=candidate.candidate_source,
            rank=candidate.rank,
            action_seq=candidate.actions,
            expert_metadata=expert_metadata,
        )
        print(
            f"seed={seed} {candidate.candidate_id} success={row['success']} executed={len(row['actions'])}",
            flush=True,
        )
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", default="demo_clean_smoke")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--instruction")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--candidate-preset", choices=["default", "anti_template"], default="default")
    args = parser.parse_args()

    instruction = args.instruction or args.task_name.replace("_", " ")

    if args.all_seeds:
        if args.output_dir is None:
            raise SystemExit("--all-seeds requires --output-dir")
        seeds = read_seed_list(args.task_name, args.task_config)
        if args.max_seeds is not None:
            seeds = seeds[: args.max_seeds]
        for seed in seeds:
            run_one_seed(
                task_name=args.task_name,
                task_config=args.task_config,
                seed=seed,
                instruction=instruction,
                output=args.output_dir / f"seed_{seed}.jsonl",
                skip_existing=args.skip_existing,
                candidate_preset=args.candidate_preset,
            )
        return

    if args.output is None:
        raise SystemExit("--output is required unless --all-seeds is used")
    seed = read_seed(args.task_name, args.task_config, args.seed)
    run_one_seed(
        task_name=args.task_name,
        task_config=args.task_config,
        seed=seed,
        instruction=instruction,
        output=args.output,
        skip_existing=args.skip_existing,
        candidate_preset=args.candidate_preset,
    )


if __name__ == "__main__":
    main()
