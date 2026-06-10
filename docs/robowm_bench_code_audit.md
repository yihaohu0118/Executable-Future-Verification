# RoboWM-Bench Code Audit And Adapter Plan

## Why This Becomes The Main Second-Layer Benchmark

The user-provided repository `https://github.com/fffstrong/RoboWM-Bench` is now the best second-layer benchmark for the paper story. Unlike RoboTwin 2.0, it is explicitly a 2026 world-model benchmark and its code matches our claim boundary:

> generated manipulation futures should be judged by embodied executability, not only visual realism.

The repo is not just a paper shell. It includes world-model inputs, GT action trajectories, Isaac Lab Franka tasks, IDM reference data, and evaluation scripts.

## Code-Grounded Findings

Repository audit from the current public code:

- `README.md` describes the intended pipeline: initial RGB/prompt in `wm_inputs`, world-model video generation, IDM or Phantom extraction, action JSON conversion, then Isaac Lab execution.
- `scripts/robot/eval_franka.py` is the robot execution evaluator. It replays JSON action trajectories, clips Franka joint actions, calls `env.step(action)`, and records `env._get_success().item()`.
- `eval_franka.py` can optionally report per-stage `part_scores` through `env.get_part_scores()` for some tasks.
- `GT/*/episode_*.json` files are newline-delimited action sequences. Each line is a 9D Franka joint/gripper action.
- `GT/*/pose.jsonl` fixes object or drawer initialization per episode.
- `wm_inputs/*/episode_*.png` and `.txt` provide initial images and prompts for robot world-model generation.
- Public robot tasks include `pick`, `put_on_plate`, `discard_trash`, `put_in_drawer`, `press_button`, `close_drawer`, and `pull_and_push` style tasks.
- The README references `tools/parquet_actions_to_json.py`, but the current checkout only contains `tools/convertv21.py`; this is a code/doc mismatch to track before relying on IDM parquet conversion.

## Manifest Mapping

Each RoboWM-Bench action JSON is one executable future candidate:

| Manifest field | RoboWM-Bench source |
| --- | --- |
| `benchmark` | `robowm_bench` |
| `suite` | Isaac Lab task id, e.g. `Franka-pick` |
| `task_name` | task folder, e.g. `pick` |
| `case_id` | `{task_name}:episode_000000` |
| `candidate_id` | `{model_or_method}:episode_000000` |
| `candidate_rank_by_planner` | world-model/IDM/default rank, or input-order rank |
| `actions` | lines from `episode_*.json` |
| `oracle_success` | `Replay result: episode=... success=...` from `eval_franka.py` |
| `oracle_progress` / `oracle_return` | optional `part_scores` and `scores` |
| `instruction` | `wm_inputs/{task}/episode_*.txt` |
| `rollout_video_path` | generated video path when available |

The converter is:

```bash
python -m umm_reward_evaluator.benchmarks.robowm_bench_actions_to_manifest \
  --task-name pick \
  --suite Franka-pick \
  --instruction-root /path/to/RoboWM-Bench/wm_inputs \
  --candidate id=veo,rank=0,path=/path/to/veo_actions,log=/path/to/veo_eval.log \
  --candidate id=cosmos,rank=1,path=/path/to/cosmos_actions,log=/path/to/cosmos_eval.log \
  --episode-index 0 \
  --episode-index 1 \
  --output-manifest /path/to/robowm_pick_manifest.jsonl

python -m umm_reward_evaluator.benchmarks.validate_future_verification_manifest \
  --manifest /path/to/robowm_pick_manifest.jsonl \
  --require-future-metadata
```

## Immediate Experimental Design

First paper-relevant run:

1. Use robot tasks only first, because they directly produce Franka action JSON and simulator success.
2. Use 3-4 tasks spanning different failure modes:
   - `pick`: spatial grasping;
   - `put_on_plate`: placement/contact;
   - `press_button`: precise contact;
   - `put_in_drawer` or `close_drawer`: articulated-object interaction.
3. For each episode, produce several candidates:
   - GT trajectory as sanity upper bound;
   - IDM trajectories from generated videos if available;
   - simple corruptions of GT/action JSON as controlled hard negatives;
   - later, outputs from multiple video world models.
4. Evaluate every candidate with `scripts/robot/eval_franka.py`.
5. Convert action roots and eval logs to the shared executable-future manifest.
6. Run the existing action-only, action-envelope, and execution-envelope selectors.

## Main Counterintuitive Claim To Test

RoboCasa365 suggests that action magnitude is a shortcut and compact robot execution-envelope statistics are unexpectedly strong after controlling that shortcut. RoboWM-Bench lets us test a sharper world-model version:

> More visually plausible generated futures are not necessarily more executable. A compact verifier over recovered action/execution envelopes can select physically executable futures better than visual/model-likelihood ranking, especially after action-energy and rank-ID shortcuts are controlled.

## Environment Risk

RoboWM-Bench requires IsaacSim 5.1 and IsaacLab v2.3.0. This is heavier than RoboCasa365 and should be installed in a clean environment on dev2. The repo itself is about 1.1GB and includes assets/GT/wm inputs; the likely bottleneck is IsaacSim/IsaacLab installation, not repository data.

Current dev2 status:

- `/tmp/RoboWM-Bench` is cloned on Linux and is about 1.2GB.
- The existing `umm-reward-evaluator` conda environment does not contain `isaacsim`, `isaaclab`, `isaaclab_tasks`, or `lerobot`.
- The RoboWM manifest converter was smoke-tested on dev2 with `/tmp/RoboWM-Bench/GT/pick`, two synthetic candidate eval logs, and episodes 0-1. The validator passed with 4 rows, 2 cases, 2 candidates per case, rank0 success 0/2, oracle success 1/2, and no metadata errors.

Before full runs, the minimal smoke should be:

```bash
python -m pip install -e source/lehome
python scripts/robot/eval_franka.py \
  --task Franka-pick \
  --json_root GT/pick \
  --episode_index 0 \
  --device cpu
```

If this GT replay succeeds, the benchmark path is viable. If it fails because IsaacSim/IsaacLab is missing, install only the required Isaac stack in a dedicated `RWMBench` conda env.
