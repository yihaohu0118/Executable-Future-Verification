from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from umm_reward_evaluator.manifest import append_jsonl, read_jsonl, validate_row
from umm_reward_evaluator.media import discover_frame_paths, image_to_data_url, sample_frames, summarize_actions
from umm_reward_evaluator.evaluator.prompts import ROLL_OUT_REWARD_PROMPT, SYSTEM_PROMPT


def parse_json_response(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip().strip("`")
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in response: {text[:300]}")
    return json.loads(match.group(0))


def wait_ready(api_base: str, timeout: int) -> bool:
    import requests

    deadline = time.time() + timeout
    url = f"{api_base.rstrip('/')}/models"
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def build_messages(row: dict[str, Any], num_frames: int) -> list[dict[str, Any]]:
    validate_row(row)
    frames = sample_frames(discover_frame_paths(row), num_frames)
    if not frames:
        raise ValueError(
            f"{row.get('case_id')}/{row.get('candidate_id')} has no frame_paths or frames_dir"
        )
    prompt = ROLL_OUT_REWARD_PROMPT.format(
        instruction=row.get("instruction", ""),
        task=row.get("task", ""),
        candidate_id=row.get("candidate_id", ""),
        action_summary=summarize_actions(row),
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in frames:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def score_row(
    row: dict[str, Any],
    api_base: str,
    model: str,
    num_frames: int,
    timeout: int,
    temperature: float,
    max_tokens: int,
    dry_run: bool,
) -> dict[str, Any]:
    if dry_run:
        vlm = {
            "task_success_score": 0,
            "task_progress_score": 0,
            "action_outcome_consistency_score": 0,
            "temporal_consistency_score": 0,
            "physical_plausibility_score": 0,
            "identity_layout_consistency_score": 0,
            "overall_score": 0,
            "verdict": "partial",
            "reason": "dry-run: evaluator was not called",
        }
    else:
        import requests

        payload = {
            "model": model,
            "messages": build_messages(row, num_frames),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = requests.post(
            f"{api_base.rstrip('/')}/chat/completions",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        vlm = parse_json_response(raw)
    return {
        "case_id": row.get("case_id"),
        "candidate_id": row.get("candidate_id"),
        "source_candidate_id": row.get("source_candidate_id"),
        "task": row.get("task"),
        "negative_type": row.get("negative_type"),
        "is_hard_negative": row.get("is_hard_negative", False),
        "oracle_reward": row.get("oracle_reward"),
        "oracle_success": row.get("oracle_success"),
        "model": model,
        "vlm": vlm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score rollout manifests with an OpenAI-compatible multimodal API.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-base", default=os.environ.get("UMM_API_BASE", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("UMM_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct"))
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--startup-wait-sec", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.startup_wait_sec and not args.dry_run:
        if not wait_ready(args.api_base, args.startup_wait_sec):
            raise SystemExit(f"evaluator server not ready at {args.api_base}")

    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[: args.limit]

    scored = []
    for row in rows:
        result = score_row(
            row=row,
            api_base=args.api_base,
            model=args.model,
            num_frames=args.num_frames,
            timeout=args.timeout,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            dry_run=args.dry_run,
        )
        scored.append(result)
        print(f"[score] {result['case_id']}/{result['candidate_id']} -> {result['vlm'].get('overall_score')}")

    append_jsonl(args.output, scored)
    print(f"[score] wrote {len(scored)} rows -> {args.output}")


if __name__ == "__main__":
    main()
