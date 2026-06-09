# DINOv2 Patch-Token Similarity Baselines

## Setup

- Manifest: `outputs/nanowm_pusht_dset_100_expert/planning_results/candidate_rollouts.jsonl`
- Encoder: `facebook/dinov2-small`
- Image size: 224
- Patch-token cache: `outputs/nanowm_pusht_dset_100_expert/dinov2_small_patch_tokens_224_6f.npz`

These baselines compare spatially aligned DINOv2 patch tokens between prediction and goal frames, matching NanoWM's WebDINO-style dense encoder interface more closely than global pooled embeddings.

| Selector | Mean state dist | Median state dist | Better than rank0 | Matched oracle-best CEM |
| --- | ---: | ---: | ---: | ---: |
| patch_cos_final (min) | 182.5589 | 172.2013 | 46/21 | 38/21 |
| patch_cos_mean (min) | 179.5808 | 175.2536 | 47/21 | 33/21 |
| patch_cos_min (min) | 187.8064 | 179.2543 | 43/21 | 36/21 |
| patch_cos_progress (max) | 181.1198 | 173.3717 | 49/21 | 41/21 |
| patch_cos_best_progress (max) | 184.1420 | 168.3961 | 34/21 | 36/21 |
| patch_cos_slope (min) | 185.1348 | 176.2981 | 43/21 | 32/21 |
| patch_l2_final (min) | 183.7170 | 172.2013 | 45/21 | 36/21 |
| patch_l2_mean (min) | 180.2063 | 175.2536 | 47/21 | 33/21 |
