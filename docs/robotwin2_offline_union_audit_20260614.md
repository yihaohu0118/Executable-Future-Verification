# RoboTwin2 Offline Union Audit, 2026-06-14

This note records the offline union built from completed dev2 RoboTwin2 trace
runs while dev2 was disk-full. It is a recovery/audit artifact, not a new
paper-countable registry result.

## Source

- Local copied archive: `/private/tmp/dev2_efv_runs_20260614.tgz`
- Extracted dev2 run copy: `/private/tmp/dev2_efv_runs_20260614/efv_runs`
- Offline union root: `/private/tmp/robotwin2_offline_union_20260614`
- Finalize command:

```bash
PYTHONPATH=src PYTHON_BIN=python3 REQUIRE_CANDIDATES_PER_CASE=24 NUM_SWEEP_SEEDS=10 \
  scripts/robotwin2_finalize_run.sh /private/tmp/robotwin2_offline_union_20260614 \
  handover_block place_object_basket press_stapler stack_blocks_two open_laptop stamp_seal
```

The union only includes seed files with exactly 24 candidates.

## Included Complete Cases

| Task | Complete seeds | Source run |
| --- | --- | --- |
| `handover_block` | 0, 1 | `robotwin2_evidence_window_20260613_k4` |
| `place_object_basket` | 1 | `robotwin2_iclr_clean_20260613_0905` |
| `press_stapler` | 1 | `robotwin2_iclr_clean_20260613_0905` |
| `stack_blocks_two` | 0, 1 | `robotwin2_targeted_energy_matched_multitask_interim_complete_20260612_1539` |
| `open_laptop` | 0, 1 | `robotwin2_targeted_energy_matched_multitask_interim_complete_20260612_1539` |
| `stamp_seal` | 0, 3, 4, 6, 9 | `robotwin2_targeted_energy_matched_k5_official_20260612` |

Total: 6 tasks, 13 complete cases, 312 candidates.

## Readiness Snapshot

| Task | Cases | Rank0 | Oracle | Base gate | Relation gate | Relation coverage |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| `handover_block` | 2 | 0/2 | 2/2 | pass | pass | 1.00 |
| `open_laptop` | 2 | 0/2 | 2/2 | pass | fail | 0.00 |
| `place_object_basket` | 1 | 0/1 | 1/1 | pass | pass | 1.00 |
| `press_stapler` | 1 | 0/1 | 1/1 | pass | pass | 1.00 |
| `stack_blocks_two` | 2 | 0/2 | 2/2 | pass | fail | 0.00 |
| `stamp_seal` | 5 | 0/5 | 5/5 | pass | fail | 0.00 |

The strongest positive signal is broad headroom: rank0 gets 0/13 while the
oracle can recover 13/13. This supports the future-selection framing.

The main limitation is relation coverage. `open_laptop`, `stack_blocks_two`,
and `stamp_seal` have usable gripper/action traces but no object-relation trace
coverage, so they cannot support relation-rescue claims.

## Selector Summary

- `handover_block`: gripper/contact envelope and DTW baselines reach 2/2;
  rank0, energy, and action-only shortcuts are 0/2.
- `open_laptop`: gripper/contact envelope reach 2/2, but smoothness also reaches
  2/2, so this task is not shortcut-clean enough for a main claim.
- `place_object_basket`: only 1 complete case; useful as a smoke diagnostic,
  not as a stable task-level result.
- `press_stapler`: only 1 complete case; simple heuristics can solve it, so it
  is diagnostic rather than paper-grade evidence.
- `stack_blocks_two`: the cleanest task. Contact-envelope contrast reaches 2/2
  while DTW joint+gripper and simple energy/smoothness are 0/2.
- `stamp_seal`: gripper/contact selectors reach 5/5, but DTW gripper also
  reaches 5/5, so it is exposed to the expert-template objection.

The registry proposal is still pending:

| Metric | Value |
| --- | ---: |
| Cases | 13 |
| Tasks | 6 |
| Rank0 success | 0.0 |
| Oracle success | 13.0 |
| Method success | 11.0 |
| Best non-oracle baseline success | 11.04 |

The best non-oracle baseline slightly exceeds the proposed method, so this
union must not be promoted to a passed evidence card.

## Gate Outcome

Passed:

- `base_ready_tasks`: 6 tasks, above the minimum of 4.
- `relation_ready_tasks`: 3 tasks, above the minimum of 1.
- `non_template_success_tasks`: all 6 tasks.
- `matched_negative_tasks`: 5 tasks.
- `diverse_antitemplate_success_tasks`: all 6 tasks.
- `matched_low_dtw_negative_tasks`: 5 tasks.

Failed:

- `strong_envelope_tasks`: 0 tasks, below the minimum of 3.
- `relation_rescue_tasks`: 0 tasks, below the minimum of 1.
- `method_beats_template_tasks`: only `stack_blocks_two`, below the minimum of 2.
- `no_template_oracle_risk`: fails because `handover_block`, `open_laptop`, and
  `stamp_seal` have DTW/template baselines near oracle.

## Interpretation

This union is useful because it shows that the second-benchmark story is not
dead: RoboTwin2 has broad candidate-selection headroom and several shortcut
controls are already present. However, it also exposes the current reviewer
risk cleanly: too many successes can still be explained by expert-template or
simple smoothness/gripper matching.

The current defendable claim is narrow:

> RoboTwin2 contains executable future-selection headroom, and at least one
> multi-stage task (`stack_blocks_two`) shows contact-envelope verification
> beating template-style DTW baselines under anti-template pressure.

The current non-defendable claim is:

> EFV is already a paper-ready general RoboTwin2 verifier.

## Next Experiments

1. Add at least one more pressured task like `stack_blocks_two`, where the best
   envelope/contact verifier beats DTW-template baselines.
2. Regenerate `open_laptop`, `stack_blocks_two`, and `stamp_seal` with
   object-relation trace fields if we want a relation-rescue claim.
3. Build candidate pools with matched low-DTW failures: similar endpoint/path
   and gripper timing, but wrong contact direction, wrong closure timing, or
   unmet object constraint.
4. Add hard positives that are successful but not expert-template-like: alternate
   approach route, different grasp timing, or different intermediate pose.
5. Keep GPU launch policy explicit on dev2: `GPU_ID=auto` should use
   `AUTO_GPU_IDS="2 3 4 5 6 7"` unless GPU0/1 are explicitly released.

