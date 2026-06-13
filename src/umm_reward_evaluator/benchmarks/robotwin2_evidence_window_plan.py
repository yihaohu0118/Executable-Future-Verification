"""Generate a reproducible RoboTwin2 evidence-window command plan."""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskTarget:
    task_name: str
    role: str
    minimum_cases: int
    keep_rule: str
    diagnostic_only: bool = False


DEFAULT_MAIN_TASKS = (
    TaskTarget("handover_block", "main anti-template/contact task", 4, "keep"),
    TaskTarget("place_object_basket", "spatial constraint task", 4, "keep"),
    TaskTarget("stack_bowls_two", "multistage gripper/contact task", 4, "keep if oracle headroom appears"),
    TaskTarget("stack_blocks_two", "multistage endpoint-vs-trace stress task", 4, "keep if gripper-aware trace works"),
)
DEFAULT_DIAGNOSTIC_TASKS = (
    TaskTarget("press_stapler", "permissive counterexample", 3, "keep as negative control", diagnostic_only=True),
)
READINESS_GATES = (
    "at least four base-ready tasks have rank0 failure and oracle success",
    "at least three tasks have matched negative cases",
    "at least two tasks have diverse non-template successes",
    "at least two tasks have matched low-DTW failures near the expert trace",
    "at least three tasks show a supported envelope selector beating the strongest simple/template baseline",
    "at least one task shows relation/contact-aware rescue over gripper-only or template-distance selectors",
    "no main-table method column is unsupported by held-out calibration data",
)
CANDIDATE_POOL_REQUIREMENTS = (
    "planner/rank0 failure when possible",
    "full gripper-aware expert candidate",
    "hard positives that are not full-template copies",
    "energy-matched negatives",
    "contact-direction or gripper-timing negatives",
    "low-DTW negatives near the expert trace",
    "reverse/shuffle/block-swap/action-axis controls",
    "candidate-ID and rank remapping in selector sweeps",
)


def _seed_range(num_cases: int, *, seed_start: int = 0) -> str:
    if num_cases <= 0:
        raise ValueError("num_cases must be positive")
    if num_cases == 1:
        return str(seed_start)
    return f"{seed_start}-{seed_start + num_cases - 1}"


def _shell_env(env: dict[str, str]) -> str:
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())


def _parse_gpu_ids(value: str | None) -> list[str] | None:
    if value is None:
        return None
    gpu_ids = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    if not gpu_ids:
        raise ValueError("gpu_ids must include at least one GPU id")
    return gpu_ids


def build_evidence_window_plan(
    *,
    run_root: str | None = None,
    task_names: list[str] | None = None,
    include_diagnostic: bool = False,
    seed_start: int = 0,
    cases_per_main_task: int = 4,
    cases_per_diagnostic_task: int = 3,
    execute: bool = False,
    gpu_id: str = "auto",
    gpu_ids: list[str] | None = None,
    task_config: str = "demo_clean_k5",
    candidate_preset: str = "targeted_energy_matched",
    launcher: str = "scripts/robotwin2_bounded_window_launcher.sh",
    finalizer: str = "scripts/robotwin2_finalize_run.sh",
    required_candidates_per_case: int = 24,
    num_sweep_seeds: int = 10,
) -> dict[str, Any]:
    resolved_run_root = run_root or f"/home/yihao_hyh/efv_runs/robotwin2_evidence_window_{date.today():%Y%m%d}"
    target_by_name = {target.task_name: target for target in DEFAULT_MAIN_TASKS + DEFAULT_DIAGNOSTIC_TASKS}
    if task_names:
        missing = [task for task in task_names if task not in target_by_name]
        if missing:
            raise ValueError(f"unknown RoboTwin2 evidence-window task(s): {', '.join(missing)}")
        targets = [target_by_name[task] for task in task_names]
    else:
        targets = list(DEFAULT_MAIN_TASKS)
        if include_diagnostic:
            targets.extend(DEFAULT_DIAGNOSTIC_TASKS)

    assigned_gpu_ids = gpu_ids or [gpu_id]
    commands = []
    finalize_tasks: list[str] = []
    for target_index, target in enumerate(targets):
        cases = cases_per_diagnostic_task if target.diagnostic_only else cases_per_main_task
        cases = max(cases, target.minimum_cases)
        seeds = _seed_range(cases, seed_start=seed_start)
        finalize_tasks.append(target.task_name)
        assigned_gpu_id = assigned_gpu_ids[target_index % len(assigned_gpu_ids)]
        env = {
            "EXECUTE": "1" if execute else "0",
            "GPU_ID": assigned_gpu_id,
            "TASK_CONFIG": task_config,
            "CANDIDATE_PRESET": candidate_preset,
            "TASKS": target.task_name,
            "SEEDS": seeds,
            "RUN_ANALYSIS_AFTER": "0",
            "RESUME_PARTIAL": "0",
            "REQUIRE_CANDIDATES_PER_CASE": str(required_candidates_per_case),
            "NUM_SWEEP_SEEDS": str(num_sweep_seeds),
        }
        commands.append(
            {
                "task_name": target.task_name,
                "role": target.role,
                "diagnostic_only": target.diagnostic_only,
                "minimum_cases": target.minimum_cases,
                "planned_cases": cases,
                "seeds": seeds,
                "gpu_id": assigned_gpu_id,
                "command": f"{_shell_env(env)} {shlex.quote(launcher)} {shlex.quote(resolved_run_root)}",
            }
        )

    finalize_env = {
        "PYTHONPATH": "src",
        "PYTHON_BIN": "python3",
        "REQUIRE_CANDIDATES_PER_CASE": str(required_candidates_per_case),
        "NUM_SWEEP_SEEDS": str(num_sweep_seeds),
    }
    finalize_command = (
        f"{_shell_env(finalize_env)} {shlex.quote(finalizer)} "
        f"{shlex.quote(resolved_run_root)} "
        + " ".join(shlex.quote(task) for task in finalize_tasks)
    )
    return {
        "run_root": resolved_run_root,
        "execute": execute,
        "gpu_id": gpu_id,
        "gpu_ids": assigned_gpu_ids,
        "task_config": task_config,
        "candidate_preset": candidate_preset,
        "required_candidates_per_case": required_candidates_per_case,
        "num_sweep_seeds": num_sweep_seeds,
        "seed_start": seed_start,
        "targets": [asdict(target) for target in targets],
        "commands": commands,
        "finalize_command": finalize_command,
        "candidate_pool_requirements": list(CANDIDATE_POOL_REQUIREMENTS),
        "readiness_gates": list(READINESS_GATES),
        "stop_rule": (
            "Downgrade to diagnostic/workshop scope if 4 tasks x 4 cases keep showing oracle "
            "headroom but no supported selector margin over DTW/action/heuristic baselines."
        ),
    }


def render_markdown(plan: dict[str, Any], *, title: str = "RoboTwin2 Evidence Window Plan") -> str:
    lines = [
        f"# {title}",
        "",
        f"- run root: `{plan['run_root']}`",
        f"- execute: `{str(plan['execute']).lower()}`",
        f"- gpu id: `{plan['gpu_id']}`",
        f"- gpu ids: `{', '.join(plan.get('gpu_ids', [plan['gpu_id']]))}`",
        f"- task config: `{plan['task_config']}`",
        f"- candidate preset: `{plan['candidate_preset']}`",
        f"- required candidates per case: `{plan['required_candidates_per_case']}`",
        f"- rank-randomization sweep seeds: `{plan['num_sweep_seeds']}`",
        "",
        "## Task Window",
        "",
        "| Task | GPU | Role | Planned cases | Seeds | Keep rule |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    target_by_name = {target["task_name"]: target for target in plan["targets"]}
    for command in plan["commands"]:
        target = target_by_name[command["task_name"]]
        diagnostic = "diagnostic only; " if command["diagnostic_only"] else ""
        lines.append(
            f"| `{command['task_name']}` | `{command['gpu_id']}` | {command['role']} | "
            f"{command['planned_cases']} | `{command['seeds']}` | {diagnostic}{target['keep_rule']} |"
        )

    lines.extend(
        [
            "",
            "## Trace Commands",
            "",
        ]
    )
    for command in plan["commands"]:
        lines.extend(
            [
                f"### {command['task_name']}",
                "",
                "```bash",
                command["command"],
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Finalize Command",
            "",
            "```bash",
            plan["finalize_command"],
            "```",
            "",
            "## Candidate Pool Requirements",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in plan["candidate_pool_requirements"])
    lines.extend(["", "## Readiness Gates", ""])
    lines.extend(f"{index}. {item}" for index, item in enumerate(plan["readiness_gates"], start=1))
    lines.extend(["", "## Stop Rule", "", plan["stop_rule"], ""])
    return "\n".join(lines)


def render_shell_script(plan: dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Auto-generated by robotwin2_evidence_window_plan.py.",
        "# Commands are dry-run unless this plan was generated with --execute.",
        f"# Run root: {plan['run_root']}",
        "",
    ]
    for command in plan["commands"]:
        lines.extend(
            [
                f"echo '=== {command['task_name']} seeds {command['seeds']} ==='",
                command["command"],
                "",
            ]
        )
    if plan["execute"]:
        lines.extend(
            [
                "echo '=== finalize RoboTwin2 evidence window ==='",
                plan["finalize_command"],
                "",
            ]
        )
    else:
        lines.extend(
            [
                "echo '=== dry-run finalize command ==='",
                f"printf '%s\\n' {shlex.quote(plan['finalize_command'])}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root")
    parser.add_argument("--task", action="append", dest="tasks", help="Task to include. Repeat to override defaults.")
    parser.add_argument("--include-diagnostic", action="store_true", help="Include press_stapler as a diagnostic counterexample.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--cases-per-main-task", type=int, default=4)
    parser.add_argument("--cases-per-diagnostic-task", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--gpu-ids", help="Comma- or space-separated GPU ids assigned round-robin by task.")
    parser.add_argument("--task-config", default="demo_clean_k5")
    parser.add_argument("--candidate-preset", default="targeted_energy_matched")
    parser.add_argument("--launcher", default="scripts/robotwin2_bounded_window_launcher.sh")
    parser.add_argument("--finalizer", default="scripts/robotwin2_finalize_run.sh")
    parser.add_argument("--required-candidates-per-case", type=int, default=24)
    parser.add_argument("--num-sweep-seeds", type=int, default=10)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-sh", type=Path)
    args = parser.parse_args()

    plan = build_evidence_window_plan(
        run_root=args.run_root,
        task_names=args.tasks,
        include_diagnostic=args.include_diagnostic,
        seed_start=args.seed_start,
        cases_per_main_task=args.cases_per_main_task,
        cases_per_diagnostic_task=args.cases_per_diagnostic_task,
        execute=args.execute,
        gpu_id=args.gpu_id,
        gpu_ids=_parse_gpu_ids(args.gpu_ids),
        task_config=args.task_config,
        candidate_preset=args.candidate_preset,
        launcher=args.launcher,
        finalizer=args.finalizer,
        required_candidates_per_case=args.required_candidates_per_case,
        num_sweep_seeds=args.num_sweep_seeds,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(plan)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    if args.output_sh:
        args.output_sh.parent.mkdir(parents=True, exist_ok=True)
        args.output_sh.write_text(render_shell_script(plan), encoding="utf-8")
        args.output_sh.chmod(0o755)
    print(markdown)


if __name__ == "__main__":
    main()
