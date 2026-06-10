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
- Host Python environments on dev2 cannot install IsaacSim 5.0/5.1 from pip because the host is Debian 11 / glibc 2.31, while the IsaacSim wheels require `manylinux_2_35`.
- The working path is a dedicated Docker container, `robowmbench_env`, based on an Ubuntu 22.04 / glibc 2.35 image with GPU passthrough.
- The container has `isaacsim==5.1.0`, IsaacLab v2.3.0 core packages, `isaaclab_tasks`, `isaaclab_assets`, `lehome`, and `lerobot` installed. The optional IsaacLab mimic extra failed on `egl_probe` because its CMake file is incompatible with the installed CMake policy defaults, but core RoboWM replay does not require that extra.
- The container was committed locally as `robowmbench:isaacsim51-isaaclab23` to avoid reinstalling the large IsaacSim/IsaacLab stack.
- PyTorch CUDA works in the container on H100 (`torch 2.7.0+cu128`, CUDA available), but IsaacSim graphics/Vulkan does not expose a usable graphics device on the H100 machine. Official camera-enabled replay fails with `vkCreateInstance failed` / `GPU Foundation is not initialized`.
- `libGLU.so.1` was missing and was fixed by installing `libglu1-mesa`; this removes one loader error but does not solve the H100 Vulkan graphics-device limitation.
- Headless import of `pynput.keyboard` fails unless `PYNPUT_BACKEND=dummy` is set, because `lehome.devices` imports keyboard backends even during offline replay.
- The RoboWM manifest converter was smoke-tested on dev2 with `/tmp/RoboWM-Bench/GT/pick`, two synthetic candidate eval logs, and episodes 0-1. The validator passed with 4 rows, 2 cases, 2 candidates per case, rank0 success 0/2, oracle success 1/2, and no metadata errors.
- Disk cleanup on dev2 removed explicitly approved HuggingFace cache entries only: SenseNova-U1-8B-MoT, Qwen2.5-14B-Instruct, Qwen3-VL-8B-Instruct, ToolACE-2-Llama-3.1-8B, and Qwen2.5-7B-Instruct. The HF hub cache dropped from about 138GB to 32GB and filesystem free space rose to about 249-257GB. VideoZeroBench data/cache was not deleted.

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

## 2026-06-10 Execution Smoke Findings

The first real dev2 smoke exposed several code-level assumptions that are easy to miss from the paper or README:

- Camera dependency is unconditional in task code. `Task00_Pick/pick.py` constructs `TiledCamera` objects inside `_setup_scene` and registers them in `self.scene.sensors` even when `eval_franka.py` is not saving a dataset. On H100, camera-enabled replay hangs because IsaacSim requires a Vulkan graphics device.
- For success-only replay, the camera is not semantically required. `eval_franka.py` only calls `_get_observations()` when `--save_dataset` is enabled. The Pick success function uses `success_checker_pick(self.object_A, self.ori_z)`, which depends on simulated object state, not rendered RGB.
- A temporary `LEHOME_PHYSICS_ONLY=1` patch to Pick skipped camera construction/registration and made `_get_observations()` return action/state only. This allowed headless H100 replay to run without changing action replay or success logic.
- The official eval script calls `env.reset(pose_name=..., pose_xyz=..., pose_quat_wxyz=...)`, but the current Pick task implements `_reset_idx(...)` and does not implement a compatible `reset(...)`. A small compatibility shim mapping those pose arguments to `_reset_idx` was required for replay to proceed. This is a benchmark code/API mismatch, not a method issue.
- `--part_scores` is not uniformly supported. `eval_franka.py` assumes `env.get_part_scores()` exists, but `PickEnv` has no such method. Pick must be evaluated by success rate only unless the benchmark code is hardened.
- With the temporary physics-only Pick shim, the official GT action replay completed on dev2. A single GT episode (`episode_000000`) succeeded.
- A deterministic 10-episode GT smoke on `GT/pick` produced 7/10 success. Repeating the same 10 episodes produced the same result: episodes `000003`, `000005`, and `000007` failed in both runs.

The last point is an important research signal: even the provided GT action JSON is not a perfect oracle under the current public evaluation stack, at least for this pinned IsaacSim/IsaacLab/H100 physics-only replay. Before reporting RoboWM-Bench numbers, the paper must treat GT replay success as an empirical upper bound, not an assumed 100% oracle.

## Research Consequences

These findings strengthen the experimental idea rather than weakening it:

- The benchmark has hidden executable-evaluation fragility: camera availability, reset API compatibility, and optional part-score interfaces all affect whether "world-model futures" can even be judged.
- A realistic paper contribution can include benchmark hardening and diagnostics: separate visual-generation evaluation from action-executability evaluation, report GT replay ceiling, and expose failure classes where GT actions are not robust under the official simulator.
- The no-real-robot story remains valid if framed as executable-future verification on current simulator benchmarks. We should not claim real-robot validation, but we can claim that current world-model benchmarks need explicit executability checks and robust verifier protocols.

Next implementation step:

1. Add a repo-side reproducibility note or helper patch for RoboWM-Bench physics-only replay on non-graphics accelerators.
2. Extend the physics-only shim beyond Pick only if the same success-only semantics hold for the target tasks.
3. Run a larger GT-ceiling table per task before evaluating generated or corrupted candidates.
4. Convert successful/failed RoboWM replay logs into the shared manifest and use them as the second benchmark layer for execution-envelope verification.
