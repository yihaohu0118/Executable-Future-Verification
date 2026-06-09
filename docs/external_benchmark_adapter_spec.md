# External Benchmark Adapter Spec

The PushT result should be treated as a candidate-selection result, not as a standalone reward-model score. Every larger benchmark should therefore be converted into the same problem:

1. A baseline policy, planner, or controller proposes K executable action candidates from the same case.
2. Each candidate is executed or predicted under a fixed protocol.
3. The benchmark oracle records task success, dense return, progress, and optional final-state distance.
4. Selectors are evaluated fold-wise on held-out cases.

## Manifest

Each candidate row is JSONL:

| Field | Required | Meaning |
| --- | --- | --- |
| `benchmark` | yes | `maniskill`, `libero`, `robomimic`, etc. |
| `suite` | yes | Suite or environment id, such as `PickCube-v1` |
| `task_name` | yes | Human-readable task id |
| `case_id` | yes | Initial-state / instruction / goal id |
| `candidate_id` | yes | Candidate id within the case |
| `candidate_rank_by_planner` | yes | 0 is the baseline choice |
| `rollout_video_path` | yes | Video used by visual selectors |
| `rollout_video_layout` | yes | Usually `prediction_only` or `init_goal_rollout` |
| `actions` | yes | Raw environment action sequence |
| `oracle_success` | yes | Simulator success flag |
| `oracle_return` | optional | Dense return |
| `oracle_progress` | optional | Task progress, higher is better |
| `oracle_state_dist` | optional | Distance metric, lower is better |
| `instruction` | optional | Required for language-conditioned suites |
| `planner_score` | optional | Baseline policy/planner confidence |

The helper module `umm_reward_evaluator.benchmarks.common` provides row serialization, oracle-best annotation, and headroom summaries.

## Benchmark Priorities

| Priority | Benchmark | Why | Main risk |
| ---: | --- | --- | --- |
| 1 | ManiSkill3 | Recognized manipulation suite, Gymnasium API, built-in success flags, GPU-parallel execution | Need candidate source that is not just oracle perturbation |
| 2 | LIBERO | High VLA visibility, language-conditioned manipulation, standard published protocol | Heavy MuJoCo/robosuite/LeRobot stack and policy checkpoint dependency |
| 3 | NanoWM PointMaze | Same planning stack as PushT, fast cross-task sanity | Current environment lacks D4RL/MuJoCo 2.10/data |
| 4 | RoboMimic/RoboSuite | Standard imitation learning benchmark | Less directly aligned with world-model candidate rollouts |

## Required Evidence

Per benchmark, report:

| Selector | Primary metric | Hard-case metric | Oracle match | Pair acc | Compute cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| rank0 | | | | | |
| static endpoint | | | | | |
| static progress | | | | | |
| action-world global | | | | | |
| component ranker | | | | | |
| failure gate | | | | | |
| oracle-best candidate | | | | | |

The most important subset is cases where rank0 fails or is not oracle-best. If the oracle-best candidate is not better than rank0, reranking cannot prove anything.

