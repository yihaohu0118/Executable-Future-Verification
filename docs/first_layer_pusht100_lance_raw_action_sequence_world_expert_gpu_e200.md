# Lance Raw-Action Sequence World Model

## Idea

This trains a sequence world model on Lance hidden states with raw trajectory-level action chunks. Each transition receives a 10D action chunk corresponding to five raw 2D actions, instead of the previous 4D mean/std summary.

This is not training-free: each fold trains action-conditioned dynamics, rollout consistency, transition-action contrastive alignment, inverse action prediction, and same-case/raw-action counterfactual margins on training cases only.

## Setup

- Candidate cache: `outputs/nanowm_pusht_dset_100_expert/lance_llm_hidden_hybrid_tokens_224_6f.npz`
- Real state cache: `outputs/nanowm_pusht_dset_100_expert/real_pusht_lance_llm_hidden_hybrid_transitions_32x1.npz`
- Action chunk dim: 10
- Action encoder: `flat`
- Sequence train rows: `expert`
- Negative source: `mixed_raw`
- Preference source: `none`
- State PCA dim: 128
- Latent dim: 64

## Results

| Metric | Value |
| --- | ---: |
| Held-out Qwen pair-label accuracy | 0.6717 |
| Held-out expert-vs-CEM oracle pair accuracy | 0.8085 |
| All-candidate top success | 0.6400 |
| CEM better than planner rank0 | 50/21 |
| CEM matched oracle-best CEM | 39/21 |

| Selector | Mean | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| NanoWM planner rank0 | 190.9580 | 177.6954 | 70.1992 | 402.7642 |
| Raw-action sequence world model | 178.8655 | 169.6787 | 61.0930 | 341.2427 |
| Oracle-best CEM | 157.4419 | 153.8775 | 61.0930 | 291.4912 |
