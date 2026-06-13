"""Generate safe resume commands for high-value partial RoboTwin2 traces."""

from __future__ import annotations

import argparse
import json
import shlex
from collections import defaultdict
from pathlib import Path
from typing import Any

from umm_reward_evaluator.benchmarks.robotwin2_partial_raw_rescue_plan import build_rescue_plan


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def infer_run_root(plan: dict[str, Any], explicit_run_root: str | None = None) -> str:
    if explicit_run_root:
        return explicit_run_root
    raw_root = Path(str(plan.get("raw_root", "")))
    if raw_root.name == "raw":
        return str(raw_root.parent)
    return str(raw_root)


def select_resume_items(
    plan: dict[str, Any],
    *,
    max_priority: int = 5,
    require_mixed: bool = True,
    require_object_state: bool = False,
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in plan.get("rescue_files", []):
        if int(item.get("rescue_priority", 99)) > max_priority:
            continue
        if require_mixed and (int(item.get("success_rows", 0)) <= 0 or int(item.get("failure_rows", 0)) <= 0):
            continue
        if require_object_state and int(item.get("object_state_rows", 0)) <= 0:
            continue
        selected.append(item)
    selected.sort(
        key=lambda item: (
            int(item.get("rescue_priority", 99)),
            -int(item.get("rows", 0)),
            str(item.get("task_name", "")),
            str(item.get("seed", "")),
        )
    )
    if max_items is not None:
        return selected[:max_items]
    return selected


def _seed_sort_key(seed: str) -> tuple[int, str]:
    try:
        return int(seed), seed
    except ValueError:
        return 10**9, seed


def _command_for_group(
    *,
    run_root: str,
    task_name: str,
    seeds: list[str],
    execute: bool,
    gpu_id: str,
    task_config: str,
    candidate_preset: str,
    launcher: str,
) -> str:
    env = {
        "RESUME_PARTIAL": "1",
        "EXECUTE": "1" if execute else "0",
        "GPU_ID": gpu_id,
        "TASK_CONFIG": task_config,
        "CANDIDATE_PRESET": candidate_preset,
        "TASKS": task_name,
        "SEEDS": ",".join(sorted(seeds, key=_seed_sort_key)),
    }
    env_text = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    return f"{env_text} {shlex.quote(launcher)} {shlex.quote(run_root)}"


def build_command_plan(
    rescue_plan: dict[str, Any],
    *,
    run_root: str | None = None,
    execute: bool = False,
    gpu_id: str = "auto",
    task_config: str = "demo_clean_k5",
    candidate_preset: str = "targeted_energy_matched",
    launcher: str = "scripts/robotwin2_bounded_window_launcher.sh",
    max_priority: int = 5,
    require_mixed: bool = True,
    require_object_state: bool = False,
    max_items: int | None = None,
) -> dict[str, Any]:
    selected = select_resume_items(
        rescue_plan,
        max_priority=max_priority,
        require_mixed=require_mixed,
        require_object_state=require_object_state,
        max_items=max_items,
    )
    selected_ids = {str(item.get("path")) for item in selected}
    skipped = [
        item
        for item in rescue_plan.get("rescue_files", [])
        if str(item.get("path")) not in selected_ids
    ]
    resolved_run_root = infer_run_root(rescue_plan, run_root)
    grouped: dict[str, list[str]] = defaultdict(list)
    for item in selected:
        grouped[str(item["task_name"])].append(str(item["seed"]))

    commands = [
        {
            "task_name": task_name,
            "seeds": sorted(seeds, key=_seed_sort_key),
            "command": _command_for_group(
                run_root=resolved_run_root,
                task_name=task_name,
                seeds=seeds,
                execute=execute,
                gpu_id=gpu_id,
                task_config=task_config,
                candidate_preset=candidate_preset,
                launcher=launcher,
            ),
        }
        for task_name, seeds in sorted(grouped.items())
    ]
    return {
        "run_root": resolved_run_root,
        "raw_root": rescue_plan.get("raw_root"),
        "execute": execute,
        "gpu_id": gpu_id,
        "task_config": task_config,
        "candidate_preset": candidate_preset,
        "launcher": launcher,
        "selection_filters": {
            "max_priority": max_priority,
            "require_mixed": require_mixed,
            "require_object_state": require_object_state,
            "max_items": max_items,
        },
        "selected_files": selected,
        "skipped_files": skipped,
        "commands": commands,
    }


def render_markdown(plan: dict[str, Any], *, title: str = "RoboTwin2 Resume Command Plan") -> str:
    filters = plan["selection_filters"]
    lines = [
        f"# {title}",
        "",
        f"- run root: `{plan['run_root']}`",
        f"- raw root: `{plan['raw_root']}`",
        f"- execute: `{str(plan['execute']).lower()}`",
        f"- gpu id: `{plan['gpu_id']}`",
        f"- task config: `{plan['task_config']}`",
        f"- candidate preset: `{plan['candidate_preset']}`",
        f"- filters: max_priority={filters['max_priority']}, require_mixed={filters['require_mixed']}, require_object_state={filters['require_object_state']}, max_items={filters['max_items']}",
        "",
        "## Commands",
        "",
    ]
    if not plan["commands"]:
        lines.append("No resume commands selected by the current filters.")
    for command in plan["commands"]:
        lines.extend(
            [
                f"### {command['task_name']} seeds {','.join(command['seeds'])}",
                "",
                "```bash",
                command["command"],
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Selected Files",
            "",
            "| File | Rows | Missing | Success | Failure | Object rows | Reason |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in plan["selected_files"]:
        lines.append(
            "| `{path}` | {rows} | {missing} | {success} | {failure} | {object_rows} | {reason} |".format(
                path=item.get("path", "-"),
                rows=item.get("rows", 0),
                missing=item.get("missing_candidates", 0),
                success=item.get("success_rows", 0),
                failure=item.get("failure_rows", 0),
                object_rows=item.get("object_state_rows", 0),
                reason=item.get("rescue_reason", "-"),
            )
        )
    lines.extend(
        [
            "",
            "## Skipped Partials",
            "",
            "| File | Rows | Missing | Success | Failure | Object rows | Reason |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in plan["skipped_files"]:
        lines.append(
            "| `{path}` | {rows} | {missing} | {success} | {failure} | {object_rows} | {reason} |".format(
                path=item.get("path", "-"),
                rows=item.get("rows", 0),
                missing=item.get("missing_candidates", 0),
                success=item.get("success_rows", 0),
                failure=item.get("failure_rows", 0),
                object_rows=item.get("object_state_rows", 0),
                reason=item.get("rescue_reason", "-"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_shell_script(plan: dict[str, Any]) -> str:
    mode = "EXECUTE=1" if plan["execute"] else "EXECUTE=0"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Auto-generated by robotwin2_resume_command_plan.py.",
        f"# Mode: {mode}. Regenerate with --execute only after checking GPU/train safety.",
        f"# Run root: {plan['run_root']}",
        "",
    ]
    if not plan["commands"]:
        lines.extend(
            [
                "echo 'No RoboTwin2 resume commands selected by the current filters.'",
                "",
            ]
        )
        return "\n".join(lines)
    for command in plan["commands"]:
        lines.extend(
            [
                f"echo '=== {command['task_name']} seeds {','.join(command['seeds'])} ==='",
                command["command"],
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rescue-json", type=Path)
    source.add_argument("--raw-root", type=Path)
    parser.add_argument("--run-root")
    parser.add_argument("--required-candidates-per-case", type=int, default=24)
    parser.add_argument("--execute", action="store_true", help="Emit EXECUTE=1 commands. Default emits dry-run commands.")
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--task-config", default="demo_clean_k5")
    parser.add_argument("--candidate-preset", default="targeted_energy_matched")
    parser.add_argument("--launcher", default="scripts/robotwin2_bounded_window_launcher.sh")
    parser.add_argument("--max-priority", type=int, default=5)
    parser.add_argument("--allow-unmixed", action="store_true")
    parser.add_argument("--require-object-state", action="store_true")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-sh", type=Path)
    args = parser.parse_args()

    rescue_plan = (
        _load_json(args.rescue_json)
        if args.rescue_json
        else build_rescue_plan(args.raw_root, required_candidates_per_case=args.required_candidates_per_case)
    )
    command_plan = build_command_plan(
        rescue_plan,
        run_root=args.run_root,
        execute=args.execute,
        gpu_id=args.gpu_id,
        task_config=args.task_config,
        candidate_preset=args.candidate_preset,
        launcher=args.launcher,
        max_priority=args.max_priority,
        require_mixed=not args.allow_unmixed,
        require_object_state=args.require_object_state,
        max_items=args.max_items,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(command_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = render_markdown(command_plan)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    if args.output_sh:
        args.output_sh.parent.mkdir(parents=True, exist_ok=True)
        args.output_sh.write_text(render_shell_script(command_plan), encoding="utf-8")
        args.output_sh.chmod(0o755)
    print(markdown)


if __name__ == "__main__":
    main()
