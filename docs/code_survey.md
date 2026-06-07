# Code Survey: NanoWM and Echo-Memory

## Nano World Model

Local path:

`third_party/nano-world-model`

Relevant files:

- `src/planning/cem_planner.py`
- `src/planning/diffusion_world_model.py`
- `src/planning/objective.py`
- `src/configs/planning/base.yaml`
- `docs/applications/planning.md`

Current planning flow:

1. Encode current observation and goal observation into latents.
2. Sample candidate action sequences with CEM.
3. Roll out each action sequence through `DiffusionWorldModel.rollout`.
4. Score predicted latent trajectory against goal latent using `objective_fn`.
5. Use top-k action sequences to update the CEM sampling distribution.

Important interface:

```text
world_model.rollout(obs_0, act) -> z_obses, aux
objective_fn(z_obses, z_obs_g) -> loss[B]
```

The built-in objective is latent-space MSE or cosine distance. It does not score
semantic task success, action-outcome consistency, physics, or identity/layout
drift.

Recommended integration:

- First build an offline evaluator around exported rollout frames/videos.
- Later add a top-k reranking stage after CEM produces candidate rollouts.
- Avoid putting slow UMM calls inside every CEM iteration.

## Echo-Memory

Local path:

`third_party/Echo-Memory`

Relevant files:

- `eval/v2/revisit_suite/qwen_vlm_score.py`
- `eval/v2/revisit_suite/prepare_eval_manifest.py`
- `eval/v2/revisit_suite/run_one_click_revisit_eval.sh`
- `eval/metrics/identity_preservation.py`
- `eval/metrics/long_horizon_consistency.py`
- `eval/metrics/semantic_consistency.py`
- `eval/metrics/temporal_coherence.py`
- `eval/metrics/loop_closure.py`

Useful design:

Echo-Memory already uses an OpenAI-compatible multimodal API to score first
frames plus revisit-tail frames. The evaluator returns structured JSON scores
and short evidence. This is the right template for our rollout reward evaluator.

The key benchmark idea to borrow is controlled corruption/revisit:

```text
first frame fixes world state
-> generated trajectory leaves current view
-> revisit tail should return to the same object/layout/viewpoint
```

For this project, the same idea becomes hard-negative evaluation for NanoWM:

- temporal reverse or shuffle;
- action shuffle;
- goal swap;
- no-op mismatch;
- identity/layout drift when generated data is available.

## First Implementation Choice

The first code path is intentionally external to NanoWM:

```text
rollout_manifest.jsonl
-> hard negative manifest
-> OpenAI-compatible UMM/VLM scoring
-> correlation, pairwise, and reranking analysis
```

This keeps the research loop fast and makes UMM, pure VLM, and pixel/latent
metrics easy to compare.
