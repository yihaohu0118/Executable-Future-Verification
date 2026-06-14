"""Generate a focused RoboTwin2 anti-template closure plan."""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FreshTarget:
    task_name: str
    role: str
    planned_cases: int
    seed_start: int
    keep_rule: str
    candidate_preset: str = "targeted_energy_matched"
    diagnostic_only: bool = False


@dataclass(frozen=True)
class ResumeTarget:
    run_root: str
    task_name: str
    seeds: str
    role: str
    keep_rule: str
    candidate_preset: str = "targeted_energy_matched"
    diagnostic_only: bool = False


DEFAULT_FRESH_TARGETS = (
    FreshTarget(
        "stack_blocks_two",
        "known clean mechanism target; extend beyond the 2 complete seeds",
        6,
        2,
        "keep only if EFV-family selector continues to beat DTW/template baselines",
    ),
    FreshTarget(
        "stack_bowls_two",
        "new multistage contact task for a second DTW-breaking result",
        6,
        0,
        "keep if rank0 fails, oracle succeeds, and DTW/template is not near oracle",
    ),
    FreshTarget(
        "handover_block",
        "relation-ready bimanual transfer candidate",
        4,
        2,
        "keep if relation/contact traces rescue cases beyond gripper-only",
    ),
    FreshTarget(
        "place_object_basket",
        "spatial constraint candidate with object-state coverage",
        4,
        2,
        "keep if relation/contact traces beat smoothness and DTW baselines",
    ),
    FreshTarget(
        "press_stapler",
        "permissive negative-control task",
        3,
        2,
        "diagnostic only; do not count if energy/smoothness solves it",
        diagnostic_only=True,
    ),
)

DEFAULT_RESUME_TARGETS = (
    ResumeTarget(
        "/home/yihao_hyh/efv_runs/robotwin2_iclr_clean_20260613_0905",
        "handover_block",
        "0",
        "complete high-value partial with mixed success/failure and object-state rows",
        "keep if relation gate passes after completing all 24 candidates",
    ),
    ResumeTarget(
        "/home/yihao_hyh/efv_runs/robotwin2_iclr_clean_20260613_0905",
        "place_object_basket",
        "1",
        "complete high-value partial with mixed success/failure and object-state rows",
        "keep if relation gate passes and simple heuristics do not solve it",
    ),
    ResumeTarget(
        "/home/yihao_hyh/efv_runs/robotwin2_iclr_clean_20260613_0905",
        "press_stapler",
        "1",
        "complete permissive partial as a negative control",
        "diagnostic only; use to show where EFV is not needed",
        diagnostic_only=True,
    ),
)

READINESS_REQUIREMENTS = (
    "at least four base-ready RoboTwin2 tasks with rank0 below oracle",
    "at least two pressured tasks where EFV-family selectors beat the best DTW/template baseline",
    "at least three strong-envelope tasks after candidate-ID/rank remap and simple heuristic controls",
    "at least one relation/contact rescue task with object-relation trace coverage",
    "no headline task where energy, smoothness, length, or DTW reaches oracle",
)

KILL_RULES = (
    "downgrade RoboTwin2 to diagnostic if DTW/template stays within one success of EFV on all new pressured tasks",
    "do not count tasks with fewer than two complete cases in the main table",
    "do not count relation selectors unless the relation gate reports nonzero coverage for every claimed case",
    "do not use GPU0/1 unless the owner explicitly releases active training jobs",
)


def _seed_range(seed_start: int, cases: int) -> str:
    if cases <= 0:
        raise ValueError("planned cases must be positive")
    if cases == 1:
        return str(seed_start)
    return f"{seed_start}-{seed_start + cases - 1}"


def _shell_env(env: dict[str, str]) -> str:
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())


def _parse_gpu_ids(value: str) -> list[str]:
    gpu_ids = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    if not gpu_ids:
        raise ValueError("auto_gpu_ids must contain at least one GPU id")
    return gpu_ids


def _fresh_command(
    *,
    run_root: str,
    target: FreshTarget,
    execute: bool,
    gpu_id: str,
    auto_gpu_ids: str,
    task_config: str,
    launcher: str,
    required_candidates_per_case: int,
    num_sweep_seeds: int,
) -> dict[str, Any]:
    seeds = _seed_range(target.seed_start, target.planned_cases)
    env = {
        "EXECUTE": "1" if execute else "0",
        "GPU_ID": gpu_id,
        "AUTO_GPU_IDS": auto_gpu_ids,
        "WAIT_FOR_GPU": "1",
        "TASK_CONFIG": task_config,
        "CANDIDATE_PRESET": target.candidate_preset,
        "TASKS": target.task_name,
        "SEEDS": seeds,
        "RUN_ANALYSIS_AFTER": "0",
        "RESUME_PARTIAL": "1",
        "REQUIRE_CANDIDATES_PER_CASE": str(required_candidates_per_case),
        "NUM_SWEEP_SEEDS": str(num_sweep_seeds),
    }
    return {
        "phase": "fresh_pressure",
        "task_name": target.task_name,
        "role": target.role,
        "diagnostic_only": target.diagnostic_only,
        "planned_cases": target.planned_cases,
        "seeds": seeds,
        "candidate_preset": target.candidate_preset,
        "keep_rule": target.keep_rule,
        "command": f"{_shell_env(env)} {shlex.quote(launcher)} {shlex.quote(run_root)}",
    }


def _resume_command(
    *,
    target: ResumeTarget,
    execute: bool,
    gpu_id: str,
    auto_gpu_ids: str,
    task_config: str,
    launcher: str,
    required_candidates_per_case: int,
    num_sweep_seeds: int,
) -> dict[str, Any]:
    env = {
        "EXECUTE": "1" if execute else "0",
        "GPU_ID": gpu_id,
        "AUTO_GPU_IDS": auto_gpu_ids,
        "WAIT_FOR_GPU": "1",
        "TASK_CONFIG": task_config,
        "CANDIDATE_PRESET": target.candidate_preset,
        "TASKS": target.task_name,
        "SEEDS": target.seeds,
        "RUN_ANALYSIS_AFTER": "0",
        "RESUME_PARTIAL": "1",
        "REQUIRE_CANDIDATES_PER_CASE": str(required_candidates_per_case),
        "NUM_SWEEP_SEEDS": str(num_sweep_seeds),
    }
    return {
        "phase": "resume_relation_partial",
        "run_root": target.run_root,
        "task_name": target.task_name,
        "role": target.role,
        "diagnostic_only": target.diagnostic_only,
        "seeds": target.seeds,
        "candidate_preset": target.candidate_preset,
        "keep_rule": target.keep_rule,
        "command": f"{_shell_env(env)} {shlex.quote(launcher)} {shlex.quote(target.run_root)}",
    }


def build_pressure_closure_plan(
    *,
    fresh_run_root: str | None = None,
    execute: bool = False,
    gpu_id: str = "auto",
    auto_gpu_ids: str = "2 3 4 5 6 7",
    task_config: str = "demo_clean_k5",
    launcher: str = "scripts/robotwin2_bounded_window_launcher.sh",
    finalizer: str = "scripts/robotwin2_finalize_run.sh",
    required_candidates_per_case: int = 24,
    num_sweep_seeds: int = 10,
    include_resume_partials: bool = True,
) -> dict[str, Any]:
    resolved_fresh_root = fresh_run_root or f"/home/yihao_hyh/efv_runs/robotwin2_pressure_closure_{date.today():%Y%m%d}"
    _parse_gpu_ids(auto_gpu_ids)
    fresh_commands = [
        _fresh_command(
            run_root=resolved_fresh_root,
            target=target,
            execute=execute,
            gpu_id=gpu_id,
            auto_gpu_ids=auto_gpu_ids,
            task_config=task_config,
            launcher=launcher,
            required_candidates_per_case=required_candidates_per_case,
            num_sweep_seeds=num_sweep_seeds,
        )
        for target in DEFAULT_FRESH_TARGETS
    ]
    resume_commands = (
        [
            _resume_command(
                target=target,
                execute=execute,
                gpu_id=gpu_id,
                auto_gpu_ids=auto_gpu_ids,
                task_config=task_config,
                launcher=launcher,
                required_candidates_per_case=required_candidates_per_case,
                num_sweep_seeds=num_sweep_seeds,
            )
            for target in DEFAULT_RESUME_TARGETS
        ]
        if include_resume_partials
        else []
    )

    finalize_env = {
        "PYTHONPATH": "src",
        "PYTHON_BIN": "python3",
        "REQUIRE_CANDIDATES_PER_CASE": str(required_candidates_per_case),
        "NUM_SWEEP_SEEDS": str(num_sweep_seeds),
    }
    fresh_tasks = [command["task_name"] for command in fresh_commands]
    fresh_finalize_command = (
        f"{_shell_env(finalize_env)} {shlex.quote(finalizer)} {shlex.quote(resolved_fresh_root)} "
        + " ".join(shlex.quote(task) for task in fresh_tasks)
    )
    resume_finalize_commands: list[str] = []
    resume_tasks_by_root: dict[str, list[str]] = {}
    for command in resume_commands:
        resume_tasks_by_root.setdefault(str(command["run_root"]), []).append(str(command["task_name"]))
    for run_root, tasks in sorted(resume_tasks_by_root.items()):
        unique_tasks = sorted(set(tasks))
        resume_finalize_commands.append(
            f"{_shell_env(finalize_env)} {shlex.quote(finalizer)} {shlex.quote(run_root)} "
            + " ".join(shlex.quote(task) for task in unique_tasks)
        )

    return {
        "fresh_run_root": resolved_fresh_root,
        "execute": execute,
        "gpu_id": gpu_id,
        "auto_gpu_ids": auto_gpu_ids,
        "task_config": task_config,
        "required_candidates_per_case": required_candidates_per_case,
        "num_sweep_seeds": num_sweep_seeds,
        "fresh_targets": [asdict(target) for target in DEFAULT_FRESH_TARGETS],
        "resume_targets": [asdict(target) for target in DEFAULT_RESUME_TARGETS] if include_resume_partials else [],
        "commands": resume_commands + fresh_commands,
        "fresh_finalize_command": fresh_finalize_command,
        "resume_finalize_commands": resume_finalize_commands,
        "readiness_requirements": list(READINESS_REQUIREMENTS),
        "kill_rules": list(KILL_RULES),
        "decision": (
            "This plan is designed to close the expert-template objection, not to maximize raw success. "
            "Promote RoboTwin2 only if the generated gates show at least two DTW-breaking pressured tasks."
        ),
    }


def render_markdown(plan: dict[str, Any], *, title: str = "RoboTwin2 Pressure Closure Plan") -> str:
    lines = [
        f"# {title}",
        "",
        f"- fresh run root: `{plan['fresh_run_root']}`",
        f"- execute: `{str(plan['execute']).lower()}`",
        f"- gpu id: `{plan['gpu_id']}`",
        f"- auto gpu ids: `{plan['auto_gpu_ids']}`",
        f"- task config: `{plan['task_config']}`",
        f"- required candidates per case: `{plan['required_candidates_per_case']}`",
        f"- rank-randomization sweep seeds: `{plan['num_sweep_seeds']}`",
        "",
        "## Why This Plan Exists",
        "",
        plan["decision"],
        "",
        "## Commands",
        "",
        "| Phase | Task | Seeds | Diagnostic | Keep rule |",
        "| --- | --- | --- | --- | --- |",
    ]
    for command in plan["commands"]:
        lines.append(
            f"| `{command['phase']}` | `{command['task_name']}` | `{command['seeds']}` | "
            f"{str(command['diagnostic_only']).lower()} | {command['keep_rule']} |"
        )
    lines.append("")
    for command in plan["commands"]:
        lines.extend(
            [
                f"### {command['phase']}: {command['task_name']}",
                "",
                f"Role: {command['role']}",
                "",
                "```bash",
                command["command"],
                "```",
                "",
            ]
        )
    lines.extend(["## Finalize Commands", "", "### Fresh pressure run", "", "```bash", plan["fresh_finalize_command"], "```", ""])
    for index, command in enumerate(plan["resume_finalize_commands"], start=1):
        lines.extend([f"### Resume root {index}", "", "```bash", command, "```", ""])
    lines.extend(["## Readiness Requirements", ""])
    lines.extend(f"{index}. {item}" for index, item in enumerate(plan["readiness_requirements"], start=1))
    lines.extend(["", "## Kill Rules", ""])
    lines.extend(f"- {item}" for item in plan["kill_rules"])
    lines.append("")
    return "\n".join(lines)


def render_shell_script(plan: dict[str, Any]) -> str:
    mode = "EXECUTE=1" if plan["execute"] else "EXECUTE=0"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Auto-generated by robotwin2_pressure_closure_plan.py.",
        f"# Mode: {mode}. Review disk/GPU ownership before executing.",
        f"# AUTO_GPU_IDS={plan['auto_gpu_ids']}",
        "",
    ]
    for command in plan["commands"]:
        lines.extend(
            [
                f"echo '=== {command['phase']} {command['task_name']} seeds {command['seeds']} ==='",
                command["command"],
                "",
            ]
        )
    if plan["execute"]:
        lines.extend(
            [
                "echo '=== finalize fresh pressure run ==='",
                plan["fresh_finalize_command"],
                "",
            ]
        )
        for index, command in enumerate(plan["resume_finalize_commands"], start=1):
            lines.extend([f"echo '=== finalize resumed root {index} ==='", command, ""])
    else:
        lines.extend(
            [
                "echo '=== dry-run finalize fresh pressure run ==='",
                f"printf '%s\\n' {shlex.quote(plan['fresh_finalize_command'])}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-run-root")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--auto-gpu-ids", default="2 3 4 5 6 7")
    parser.add_argument("--task-config", default="demo_clean_k5")
    parser.add_argument("--launcher", default="scripts/robotwin2_bounded_window_launcher.sh")
    parser.add_argument("--finalizer", default="scripts/robotwin2_finalize_run.sh")
    parser.add_argument("--required-candidates-per-case", type=int, default=24)
    parser.add_argument("--num-sweep-seeds", type=int, default=10)
    parser.add_argument("--skip-resume-partials", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-sh", type=Path)
    args = parser.parse_args()

    plan = build_pressure_closure_plan(
        fresh_run_root=args.fresh_run_root,
        execute=args.execute,
        gpu_id=args.gpu_id,
        auto_gpu_ids=args.auto_gpu_ids,
        task_config=args.task_config,
        launcher=args.launcher,
        finalizer=args.finalizer,
        required_candidates_per_case=args.required_candidates_per_case,
        num_sweep_seeds=args.num_sweep_seeds,
        include_resume_partials=not args.skip_resume_partials,
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
