# First-Layer PushT-100 Code-Driven Audit

## Scope

This is the first benchmark layer for the failure-gated world-action idea. It is not a final full benchmark. The goal here is to expose mechanisms: where reranking headroom exists, how much static visual similarity explains, and which code paths can create hidden advantages or failures.

Artifacts:

- Candidate manifest: `outputs/nanowm_pusht_dset_100_expert/planning_results/candidate_rollouts.jsonl`
- Oracle pairwise labels: `outputs/nanowm_pusht_dset_100_expert/oracle_state_dist_pairwise.jsonl`
- Rank0/oracle summary: `outputs/nanowm_pusht_dset_100_expert/first_layer_rank0_oracle_summary.json`
- Static DINO summary: `outputs/nanowm_pusht_dset_100_expert/dinov2_patch_similarity_cem_baselines.json`
- Machine summary: `outputs/nanowm_pusht_dset_100_expert/first_layer_pusht100_summary.json`

## First-Layer Result

| Selector | Mean state dist | Median | Better than rank0 | Oracle-best CEM match |
| --- | ---: | ---: | ---: | ---: |
| NanoWM planner rank0 | 190.9580 | 177.6954 | n/a | 21/100 |
| Oracle-best CEM upper bound | 157.4419 | 153.8775 | 79/100 | 100/100 |
| Expert replay sanity check | 1.6878 | 1.1225 | n/a | n/a |

Static DINOv2-small patch-token selectors:

| Selector | Mean state dist | Median | Better than rank0 | Oracle-best CEM match |
| --- | ---: | ---: | ---: | ---: |
| patch_cos_final (min) | 182.5589 | 172.2013 | 46/100 | 38/100 |
| patch_cos_mean (min) | 179.5808 | 175.2536 | 47/100 | 33/100 |
| patch_cos_min (min) | 187.8064 | 179.2543 | 43/100 | 36/100 |
| patch_cos_progress (max) | 181.1198 | 173.3717 | 49/100 | 41/100 |
| patch_cos_best_progress (max) | 184.1420 | 168.3961 | 34/100 | 36/100 |
| patch_cos_slope (min) | 185.1348 | 176.2981 | 43/100 | 32/100 |
| patch_l2_final (min) | 183.7170 | 172.2013 | 45/100 | 36/100 |
| patch_l2_mean (min) | 180.2063 | 175.2536 | 47/100 | 33/100 |

Immediate read:

- There is real reranking headroom: oracle-best CEM improves mean state distance from 190.9580 to 157.4419, and beats rank0 in 79/100 cases.
- Static visual similarity helps but does not close the gap. Best mean static DINO is `patch_cos_mean` at 179.5808; best oracle-match static DINO is `patch_cos_progress` at 41/100.
- Expert replay has mean distance 1.6878, confirming these dset goals are reachable and the oracle metric is not saturated by impossible goals.

## What The Code Actually Does

1. Goals are reachable dataset goals, not arbitrary random target states. In `third_party/nano-world-model/src/experiments/planning_experiment.py:296`, `_sample_dset_goals` replays ground-truth validation actions for `goal_H * frame_interval`; lines 379-429 store `expert_actions`. This makes the benchmark a world-model/planning accuracy test, not a goal-feasibility test.
2. Candidate export is not a raw action-sample benchmark. In `third_party/nano-world-model/src/planning/cem_planner.py:203`, candidates are recorded only on the final CEM iteration, after CEM has already filtered by NanoWM latent objective.
3. The first candidate in every CEM sample batch is the current mean action sequence. `cem_planner.py:179` sets `action_samples[0] = mu[b]`. This can bias exported candidates toward previous CEM mean behavior.
4. CEM variance has a floor. `cem_planner.py:215-217` clamps sigma with `sigma_min`; this may keep candidate diversity alive but can also produce late-iteration artifacts.
5. The earlier DINO anchor name is misleading. `src/umm_reward_evaluator/training/umm_latent_dynamics_reward.py:238-246` describes a trained action-conditioned latent dynamics model; it is not static DINO similarity.
6. DINO-LDM currently trains the MLP on CPU tensors. `umm_latent_dynamics_reward.py:143-146` creates `xt`, `yt`, and `model` without moving them to `args.device`; `rollout_score` also constructs CPU tensors at line 175. This made 600/200/50 epoch PushT-100 attempts too slow for first-layer turnaround despite the DINOv2-base cache being generated.
7. DINO-LDM report strings hard-code `/21` in `umm_latent_dynamics_reward.py:255-256`. Static DINO report code has the same issue. JSON counts are correct; markdown/stdout denominators can be misleading on 100-case runs.
8. The oracle pairwise file used here is diagnostic: labels are derived from `oracle_state_dist`, not from a VLM judge. This is deliberate for first-layer mechanism testing.

## High-Information Modules To Stress

| Module | Why it matters |
| --- | --- |
| dset goal replay | Determines whether expert is an upper-bound sanity check and whether failures are model/planner failures. |
| final-iteration CEM candidate export | Rerankers only see candidates already filtered by NanoWM, not raw action proposals. |
| static DINO progress metrics | `patch_cos_progress` gives more oracle matches than final/mean similarity, suggesting temporal change matters more than endpoint appearance. |
| DINO-LDM CPU training path | An implementation detail blocks full evaluation and may bias future iteration toward easier baselines. |
| case-wise ranker normalization | Existing ranker learns within-case relative ordering, not global calibrated rewards; this can hide calibration failures. |

## Ten Testable Observations

| # | Code basis | Counterintuitive hypothesis | Minimal experiment | Expected signal if true |
| ---: | --- | --- | --- | --- |
| 1 | `cem_planner.py:179` mean action is always sampled | Planner rank0 may be strong partly because the export contains a deterministic mean-sample shortcut | Export with `action_samples[0]=mu` disabled | Rank0/oracle match and static DINO behavior shift, especially early cases |
| 2 | `cem_planner.py:203` records only final CEM top-k | Reranker performance may depend on NanoWM prefiltering, not evaluator quality | Export candidates from every CEM iteration | Oracle gap increases; static DINO may fail more on raw samples |
| 3 | `planning_experiment.py:296-429` dset goals are expert reachable | Success/failure is mostly action-model quality, not visual goal ambiguity | Run random-state goals with same evaluator | Expert sanity no longer near zero; evaluator gaps become harder to interpret |
| 4 | Static DINO progress has 41/100 oracle matches, better than final/mean in oracle match | Matching visual progress may be more useful than endpoint similarity | Compare final-only vs progress-only on hard cases where rank0 fails | Progress selector wins mostly on non-rank0 oracle cases |
| 5 | DINO-LDM is CPU-trained at `umm_latent_dynamics_reward.py:143-146` | Reported method complexity may be dominated by an avoidable implementation bottleneck | Move MLP/tensors to CUDA and rerun e50/e200/full | Same numbers with much shorter wall time; enables full ablation |
| 6 | DINO-LDM uses mean+std token vectors at lines 18-25 | Dense tokens may be collapsed too aggressively; static patch baseline may retain more useful spatial info | Replace mean/std with PCA pooled patch tokens | Better oracle match without changing model class |
| 7 | Action vectors fallback/padding at lines 28-53 | Some candidate action features may be padded or summarized in a way that discards dynamics | Use raw 5-step flattened action chunks consistently | DINO-LDM changes more than expected, revealing action encoding bottleneck |
| 8 | Existing component ranker z-scores per case | A model can look good without globally calibrated reward | Run ranker with and without case-wise z-score | No-zscore drops even if zscore works, showing selector not reward-calibrated |
| 9 | Expert rows are included in manifest but CEM-only metrics exclude them | All-candidate success can be inflated by expert rows if not separated | Report CEM-only and all-candidate separately everywhere | All-candidate top success is not comparable to CEM reranking |
| 10 | DINO report hard-codes `/21` | Some earlier conclusions can be numerically misread | Fix denominator and regenerate reports | Same JSON numbers, clearer 100-case interpretation |

## First Ablation Set

| Type | Experiment | Purpose |
| --- | --- | --- |
| remove | Disable CEM mean-sample insertion | Test whether planner/rank0 shortcut shapes candidate pool |
| remove | Exclude expert rows from all reports | Prevent upper-bound rows from leaking into selector comparisons |
| replace | Static DINO progress vs final/mean only | Test temporal-progress claim without training |
| replace | DINO-LDM CPU path -> CUDA path | Separate method validity from implementation bottleneck |
| reverse | Score visual stagnation instead of visual progress | Test whether progress itself or endpoint closeness matters |
| stress | Evaluate cases where rank0 is not oracle-best | Focus on actual reranking failures instead of easy cases |
| stress | Export all CEM iterations | Test if evaluator survives less prefiltered candidates |
| bug-fix | Fix `/21` denominator in reports | Prevent misleading benchmark summaries |
| oracle | Oracle-best CEM and expert replay | Bound reranker headroom and goal feasibility |
| negative | Train DINO-LDM with shuffled actions | If performance stays similar, the action-conditioned claim is weak |

## Candidate Research Ideas From This Layer

| Priority | Idea | Counterintuitive claim | Code observation | Minimal next test |
| ---: | --- | --- | --- | --- |
| 1 | Failure-Only World-Action Override | A world model should not replace visual similarity globally; it should intervene only on visual failure cases | Static DINO helps but leaves large gap; previous gate worked on 21-case | Run UMM/action-world selector + DINO-failure gate on this 100-case set |
| 2 | Progress Beats Endpoint | Temporal visual progress can identify better actions than final-state similarity, even without a trained evaluator | `patch_cos_progress` has best oracle match among static selectors | Hard-case analysis: rank0-fail subset and progress-vs-final disagreement cases |
| 3 | Candidate-Pool Causality Benchmark | Many evaluator gains may be artifacts of CEM prefiltering, not general action understanding | Candidates are final CEM top-k only | Export raw/all-iteration candidates and rerun all selectors |
| 4 | Calibration-Free Reward Is Not Reward | Per-case z-scored rankers may be selectors, not globally meaningful reward models | Component ranker normalizes inside each case | Compare zscore/no-zscore ranker and cross-goal score calibration |
| 5 | Action-Shuffle Diagnostic For World Models | If shuffled actions do not hurt, the model is visual-only despite action-conditioned naming | DINO-LDM action encoding is summarized/padded | Train DINO-LDM or UMM critic with shuffled action chunks |

## Dynamic DINO-LDM After CUDA Fix

After fixing `train_dynamics` to move the MLP and tensors to `--device`, the DINO-LDM branch completed on PushT-100. This produced a useful negative result: the trained action-conditioned DINO latent dynamics model is still weaker than the best static DINO patch selectors.

| Run | Pair acc | Expert pair acc | Default max mean | Median | Better than rank0 | Oracle match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpu_e50_h64 | 0.4710 | 0.4975 | 192.8293 | 192.2227 | 38/100 | 22/100 |
| gpu_e200_h128 | 0.5090 | 0.5400 | 185.1602 | 180.1527 | 48/100 | 26/100 |
| gpu_e600_h128 | 0.4870 | 0.4900 | 194.7505 | 192.5255 | 39/100 | 25/100 |

Score-direction diagnostic:

| Run | Field | Select | Mean | Median | Better than rank0 | Oracle match |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| gpu_e50_h64 | ac_umm_ldm_score | max | 192.8293 | 192.2227 | 38/100 | 22/100 |
| gpu_e50_h64 | ac_umm_ldm_score | min | 188.5029 | 181.3805 | 42/100 | 28/100 |
| gpu_e50_h64 | ac_umm_ldm_consistency_score | max | 187.8707 | 184.5560 | 50/100 | 29/100 |
| gpu_e50_h64 | ac_umm_ldm_consistency_score | min | 189.3999 | 188.2273 | 34/100 | 22/100 |
| gpu_e200_h128 | ac_umm_ldm_score | max | 185.1602 | 180.1527 | 48/100 | 26/100 |
| gpu_e200_h128 | ac_umm_ldm_score | min | 188.5620 | 188.8546 | 42/100 | 29/100 |
| gpu_e200_h128 | ac_umm_ldm_consistency_score | max | 189.9707 | 184.9535 | 44/100 | 23/100 |
| gpu_e200_h128 | ac_umm_ldm_consistency_score | min | 190.8612 | 185.0451 | 35/100 | 21/100 |
| gpu_e600_h128 | ac_umm_ldm_score | max | 194.7505 | 192.5255 | 39/100 | 25/100 |
| gpu_e600_h128 | ac_umm_ldm_score | min | 187.8487 | 186.2915 | 42/100 | 25/100 |
| gpu_e600_h128 | ac_umm_ldm_consistency_score | max | 191.9407 | 187.0348 | 42/100 | 24/100 |
| gpu_e600_h128 | ac_umm_ldm_consistency_score | min | 189.8098 | 182.0772 | 39/100 | 26/100 |

Interpretation:

- The CPU training bottleneck was real, but fixing it did not make DINO-LDM the strongest selector.
- `gpu_e200_h128` is the best dynamic DINO run by mean distance at 185.1602, still worse than static `patch_cos_mean` at 179.5808.
- In e50/e600, selecting the minimum DINO-LDM score is better than the default maximum by mean distance, which suggests the learned latent rollout score is not a stable monotonic reward.
- `ac_umm_ldm_score` and `ac_umm_ldm_goal_score` select the same candidates in these runs; the consistency term is not the main failure mode.
- This strengthens the next research hypothesis: do not globally replace static DINO with a trained latent dynamics score; use a failure-gated override or action-counterfactual critic only where static visual progress is unreliable.

## Action-World Critic And Failure Gate

After generating the 100-case Lance `llm_hidden_hybrid` cache, I trained the raw-action sequence world critic. The first e50 run had useful pairwise signal but weak selector performance; e200 converted that signal into a stronger selector.

| Selector | Pair acc | Expert pair acc | Mean | Median | Better than rank0 | Oracle match | Gate acc | Overrides |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ActionWorld e50 | 0.6187 | 0.6969 | 183.9390 | 177.0651 | 51/100 | 33/100 | n/a | n/a |
| ActionWorld e200 | 0.6717 | 0.8085 | 178.8655 | 169.6787 | 50/100 | 39/100 | n/a | n/a |
| Gate mean-anchor e200 | 0.6067 | 0.6791 | 182.0821 | 173.0237 | 45/100 | 32/100 | 0.5900 | 37/100 |
| Gate progress-anchor e200 | 0.6031 | 0.6301 | 177.3960 | 171.0635 | 50/100 | 43/100 | 0.7500 | 28/100 |
| Gate progress-anchor oracle-rescue e200 | 0.6031 | 0.6301 | 177.3960 | 171.0635 | 50/100 | 43/100 | 0.7500 | 28/100 |
| Ranker h0 e200 | 0.7162 | 0.9337 | 179.0400 | 172.3713 | 46/100 | 36/100 | n/a | n/a |

Key observations:

- `ActionWorld e200` reaches mean 178.8655, slightly better than static `patch_cos_mean` at 179.5808 and better than static `patch_cos_progress` at 181.1198.
- The best first-layer selector is not global action-world and not a generic component ranker; it is `Static DINO progress + ActionWorld e200 failure gate`, with mean 177.3960 and 43/100 oracle matches.
- The mean-anchor gate fails, but the progress-anchor gate succeeds. This supports the more specific claim that temporal visual progress is the right visual anchor for failure-gated action-world override.
- Linear component ranker has very high pair accuracy and expert-pair accuracy but worse selection mean than the failure gate. Pairwise accuracy alone is therefore not a sufficient success metric for this benchmark.
- The e50/e200 contrast shows training strength matters: e50 action-world was not enough, while e200 became competitive. This is also a useful minimum training-cost estimate for the method.
- The Lance hybrid cache itself is expensive: it required VAE, ViT, and native projection passes over 6000 candidate images plus 192 real-transition images. The method should report cache cost explicitly.

## Recommended Next Moves

1. Fix DINO-LDM training device and report denominator bugs, then rerun `e50`, `e200`, and full only if e50/e200 show directionally useful gains.
2. Run first UMM/action-world branch on the 100-case manifest, but report it as an action-counterfactual critic because the code uses same-case and raw-action intervention negatives.
3. Prioritize the failure-gate experiment over global fusion. The first-layer result already suggests static DINO is useful but insufficient, which is exactly the regime where a failure-only override can be meaningful.
