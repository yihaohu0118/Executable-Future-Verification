# AC-UMM-LDM: Action-Conditioned UMM Latent Dynamics Model

## Setup

- Manifest: `outputs/nanowm_pusht_dset_100_expert/planning_results/candidate_rollouts.jsonl`
- Pairwise labels for evaluation: `outputs/nanowm_pusht_dset_100_expert/oracle_state_dist_pairwise.jsonl`
- Patch-token cache: `outputs/nanowm_pusht_dset_100_expert/dinov2_base_patch_tokens_224_6f.npz`
- Encoder backbone: `dinov2`
- UMM/DINO/Lance model: `facebook/dinov2-base`
- Hidden dim: 128
- Epochs: 200
- Train rows: `all`

This is a training-based world-model idea. It trains a dynamics model in dense UMM/DINO latent space:

```text
z_t = DenseUMMEncoder(o_t)
z_{t+1} = f_theta(z_t, a_t)
score(a_{0:H}) = cosine(rollout_theta(z_0, a_{0:H}), z_goal)
```

It is not a training-free metric and is closer to "UMM as a world model" than evaluator-only reranking.

## Cross-Validated Results

| Metric | Value |
| --- | ---: |
| Held-out Qwen pair-label accuracy | 0.5090 |
| Held-out expert-vs-CEM oracle pair accuracy | 0.5400 |
| All-candidate top success | 0.2800 |
| CEM better than planner rank0 | 48/100 |
| CEM matched oracle-best CEM | 26/100 |

## CEM-Only Reranking by State Distance

| Selector | Mean | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| NanoWM planner rank0 | 190.9580 | 177.6954 | 70.1992 | 402.7642 |
| AC-UMM-LDM | 185.1602 | 180.1527 | 81.0742 | 389.9575 |
| Oracle-best CEM | 157.4419 | 153.8775 | 61.0930 | 291.4912 |
