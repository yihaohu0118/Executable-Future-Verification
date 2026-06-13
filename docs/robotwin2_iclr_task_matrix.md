# RoboTwin2 ICLR Task Matrix

Date: 2026-06-13

This matrix turns the RoboTwin2 evidence protocol into a bounded run list. It
is based on the official task code inspected on dev2 under
`/home/yihao_hyh/robotwin_probe/envs`.

## Primary Window

These tasks should be run first with `demo_clean_k5` and
`targeted_energy_matched`. The goal is not for every task to pass; the goal is
to cover distinct failure modes.

| Priority | Task | Role | Official success signal | Desired EFV phenomenon |
| ---: | --- | --- | --- | --- |
| 1 | `stack_blocks_two` | Multi-stage spatial relation | block2 is above block1 within xyz tolerance and both grippers are open | Gripper-only and DTW-gripper should fail on some cases; object/contact relation should recover. |
| 2 | `stamp_seal` | Contact placement positive control | seal xy matches target and both grippers are open | Gripper/contact envelope should beat energy, smoothness, rank, and action distribution. |
| 3 | `open_microwave` | Articulated object progress | microwave joint qpos exceeds a target fraction | A pure gripper template may be insufficient; progress/contact-phase features should matter. |
| 4 | `place_object_basket` | Contact/object containment | object is lifted, close to basket, off table, and in contact with basket | Object-contact relation should beat action-only and gripper-only shortcuts. |
| 5 | `stack_bowls_two` | Relation variant | bowls align in xy and target heights with open grippers | Tests whether the stack relation result transfers beyond blocks. |
| 6 | `press_stapler` | Local contact timing | gripper contact near stapler contact point | Helps separate contact-timing tasks from spatial-relation tasks. |

Because the latest dev2 window was interrupted by Ray training reclaiming GPUs,
the next rerun should be the bounded sequential four-task window, not the full
six-task window:

```bash
cd /home/yihao_hyh/Executable-Future-Verification
EXECUTE=1 RUN_ANALYSIS_AFTER=1 GPU_ID=auto SEEDS=0-7 \
  scripts/robotwin2_bounded_window_launcher.sh \
  /home/yihao_hyh/efv_runs/robotwin2_bounded_window_YYYYMMDD
```

The bounded default is `stack_blocks_two stamp_seal place_object_basket
stack_bowls_two`. This keeps one relation task, one contact positive control,
one object-containment relation task, and one relation transfer task while
minimizing GPU contention. If this four-task window produces complete candidate
pools and at least three base-ready tasks, expand to the six-task primary
window. The paper-level gate still requires at least four base-ready tasks and
at least one relation-rescue task before RoboTwin2 can be used as main evidence.

## Diagnostic-Only Tasks

These are useful, but they should not be allowed to carry the main claim unless
the gate shows clean headroom and hard negatives.

| Task | Why diagnostic |
| --- | --- |
| `open_laptop` | Earlier smoke looked permissive: smoothness/gripper baselines can succeed. Useful as a boundary case. |
| `click_bell` | Too order-insensitive for a main table; useful only as a contact sanity check. |
| `handover_block` | Interesting bimanual handoff, but more expensive and potentially brittle; use after the primary window. |
| `stack_blocks_three` / `stack_bowls_three` | Stronger multi-stage variants; use if two-object variants pass and we need scaling evidence. |

## Minimum Evidence Needed Per Primary Task

Each primary task should produce:

- complete K=5 or K=8 cases with no `candidate_error`;
- rank0 below oracle-best;
- at least one successful non-template future if possible;
- matched or energy-controlled negative candidates;
- rank/candidate-ID remap analysis;
- main-table gate output;
- relation gate output when object/contact claims are made;
- selector table and failure analysis.

## Decision After The Primary Window

Continue toward the ICLR story if:

- at least four primary tasks pass the base main-table gate;
- at least one spatial task shows relation rescue over gripper/DTW-gripper;
- at least three tasks show envelope selectors beating simple baselines;
- hard positives show the verifier is not only choosing full expert traces;
- matched negatives show it is not only using action energy, length, or gripper
  timing.

Downgrade if the primary window fails because oracle headroom is rare,
successful candidates are full-template dominated, or relation features do not
rescue any gripper failure.
