# Benchmark Expansion Roadmap

## Current Claim

The strongest PushT-100 result is not "a learned world score is globally better." The more defensible claim is:

> Temporal visual progress is a strong anchor, and a trained action-world critic helps when it is used as a failure override rather than as a global replacement.

Current PushT-100 numbers:

| Setting | Mean state dist | Oracle match |
| --- | ---: | ---: |
| NanoWM rank0 | 190.9580 | 21/100 |
| Static DINO mean | 179.5808 | 33/100 |
| Static DINO progress | 181.1198 | 41/100 |
| ActionWorld e200 | 178.8655 | 39/100 |
| Static DINO progress + ActionWorld failure gate | 177.3960 | 43/100 |
| Oracle-best CEM | 157.4419 | 100/100 |

## Updated External-Benchmark Plan

### 1. ManiSkill3 First

ManiSkill3 is now the best near-term external benchmark target. Official docs describe built-in rigid-body tasks, task cards with success/fail conditions, sparse/dense rewards, demonstrations, and GPU-parallel simulation. It also exposes a Gymnasium-style API, which maps cleanly to our candidate executor.

Official sources:

- ManiSkill tasks: https://maniskill.readthedocs.io/en/latest/tasks/index.html
- ManiSkill installation/system support: https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html
- ManiSkill quickstart/API: https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html

Local finding on the remote machine:

- `mani_skill==3.0.1`, `gymnasium==1.3.0`, and `sapien==3.0.3` install successfully in the existing Python 3.11 environment.
- `PickCube-v1`, `PushCube-v1`, `StackCube-v1`, and `PegInsertionSide-v1` all run one state-mode step and expose `info["success"]`.
- `PickCube-v1` also renders RGB frames with `obs_mode=rgbd` and `render_mode=rgb_array`.
- Official motion-planning solvers currently crash inside the solver path on this machine, so they should be treated as a high-risk candidate source until isolated.

First ManiSkill milestone:

1. Build a candidate manifest for `PickCube-v1` and `PushCube-v1`.
2. Verify oracle headroom: rank0 success vs oracle-best success.
3. Train/evaluate the same selectors used on PushT:
   - static endpoint
   - static progress
   - action-world global
   - component ranker
   - progress-anchor failure gate
4. Report hard-case recovery where rank0 fails.

### 2. LIBERO Next

LIBERO has higher VLA visibility. The LeRobot documentation lists five suites covering 130 tasks: Spatial, Object, Goal, LIBERO-90, and Long. The standard evaluation uses task success over suites, with observations containing proprioception and two camera views and actions as 7D end-effector control.

Official source:

- LeRobot LIBERO benchmark docs: https://huggingface.co/docs/lerobot/libero

This should be the second target because it needs more infrastructure:

- Linux/MuJoCo rendering setup.
- LeRobot/LIBERO policy checkpoint.
- Candidate generation for K action chunks per decision point.
- Language-conditioned manifest fields.

First LIBERO milestone:

1. Start with `libero_spatial` or `libero_object`.
2. Use a public policy checkpoint or a small BC policy to produce K candidates.
3. Execute candidates in simulator to build oracle-best upper bound.
4. Evaluate whether the failure gate improves success where rank0 fails.

### 3. PointMaze As A Cross-Task Sanity Check

NanoWM has a PointMaze config/checkpoint path, but the current environment is missing `d4rl`, MuJoCo 2.10, and PointMaze data. This is still useful because it shares the NanoWM planning stack with PushT, but it should not block the higher-impact ManiSkill/LIBERO work.

## Reviewer-Facing Minimum Bar

The paper-quality result should not be framed as "we improve PushT." The target claim should be:

1. PushT-100 shows the mechanism under NanoWM/CEM planning.
2. ManiSkill3 shows it improves executable manipulation candidate selection under standard simulator success metrics.
3. LIBERO shows the same failure-gated mechanism transfers to language-conditioned manipulation.

The key ablation is whether the trained action-world critic helps only when used as a gated override. If global ActionWorld is worse than static progress but the gate is better, that is a stronger and more counterintuitive story.
