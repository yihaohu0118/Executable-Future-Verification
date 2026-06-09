# ManiSkill3 First Benchmark Plan

## Why This Benchmark

ManiSkill3 is the most practical high-visibility next target. It is broad enough to include pick, push, stack, insertion, articulated-object, mobile manipulation, and dexterous tasks, while still exposing a Gymnasium API and per-task success flags.

Official documentation points that matter for our method:

- Built-in rigid-body tasks are grouped by task category and expose success/fail conditions.
- State observations include privileged ground-truth data, while visual observations remove data that would not be available in the real world.
- `PickCube-v1`, `PushCube-v1`, and `PegInsertionSide-v1` are documented quickstart examples.
- `num_envs > 1` enables GPU-parallelized simulation.

Sources:

- Tasks: https://maniskill.readthedocs.io/en/latest/tasks/index.html
- Installation/system support: https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html
- Quickstart/API: https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html

## Remote Smoke Results

Environment:

- Python 3.11.15
- `torch==2.8.0`
- `mani_skill==3.0.1`
- `gymnasium==1.3.0`
- `sapien==3.0.3`

State-mode smoke:

| Task | Status | Observation shape | Action space |
| --- | --- | --- | --- |
| `PickCube-v1` | OK | `[1, 42]` | `Box(-1.0, 1.0, (7,), float32)` |
| `PushCube-v1` | OK | `[1, 35]` | `Box(-1.0, 1.0, (7,), float32)` |
| `StackCube-v1` | OK | `[1, 48]` | `Box(-1.0, 1.0, (7,), float32)` |
| `PegInsertionSide-v1` | OK | `[1, 43]` | `Box(-1.0, 1.0, (7,), float32)` |

Visual smoke:

- `PickCube-v1`, `obs_mode=rgbd`, `render_mode=rgb_array` returns a `[1, 512, 512, 3]` RGB tensor.
- SAPIEN emits a Vulkan ICD warning, but rendering still works.

Motion-planning baseline:

- ManiSkill ships official Panda motion-planning solutions for the target tasks.
- On this remote setup, the solver path crashes inside `solvePickCube`, and the official runner exits unsuccessfully.
- Treat official motion planning as a valuable but currently unstable candidate source.

## Candidate Pool Strategy

Use three stages, from easiest to most reviewer-proof:

| Stage | Candidate source | Use | Risk |
| --- | --- | --- | --- |
| A | Random/action-perturbation rollout | Adapter and oracle-headroom smoke | Weak final evidence |
| B | Official motion-planning trajectories with perturbations | Strong diagnostic once solver crash is fixed | May look privileged |
| C | Public policy or trained BC/diffusion policy top-k samples | Paper-quality benchmark result | More training and compute |

The final benchmark table should use Stage C or a clearly non-oracle planner. Stages A/B are for mechanism discovery and debugging.

## Required First Experiment

For `PickCube-v1` and `PushCube-v1`:

1. Generate 50-100 cases.
2. Produce 5 candidates per case.
3. Record rollout video, action sequence, `oracle_success`, dense return, and final progress.
4. Compute headroom:
   - rank0 success
   - oracle-best success
   - oracle-better cases
5. Train/evaluate:
   - static visual endpoint
   - static visual progress
   - action-world global
   - progress-anchor failure gate

If oracle-best success is not meaningfully higher than rank0 success, change candidate source before training any reranker.
