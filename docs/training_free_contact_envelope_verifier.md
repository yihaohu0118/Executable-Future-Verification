# Training-Free Contact-Envelope Verification

Date: 2026-06-14

## Positioning

This project should be framed as a verifier layer for robot world models:

> World models generate candidate futures; EFV selects which future is
> executable and task-completing.

The method is not a new world model and not a learned reward model. The primary
method is a training-free, non-parametric selector over compact execution
envelopes. Few-shot task calibration can be added by storing successful and
failed reference envelopes, but no neural verifier needs to be trained.

## Core Claim

The current evidence supports this narrower claim:

> Before training another reward model, a compact contact-phase execution
> envelope can recover much of the executable-future selection signal that
> rank0, action energy, smoothness, visual plausibility, and full-trajectory
> template similarity miss.

Avoid claiming universal zero-shot physical understanding. The safer boundary is
training-free but task-calibrated future verification.

## Contact-Envelope Feature

The `contact_envelope` feature is a fixed, hand-designed summary extracted from
each candidate future:

- gripper open/close trajectory;
- inferred contact-window start/end/duration;
- gripper switching rate;
- joint/action norm and change around the contact window;
- pre-contact, contact, and post-contact window statistics.

It is deliberately compact and interpretable. It is meant to test whether the
success signal lives in contact timing and local execution phase, rather than in
global action magnitude, endpoint state, or full expert-template similarity.

Implemented selectors:

- `prototype:contact_envelope:same_task:nearest_positive`;
- `prototype:contact_envelope:same_task:nearest_pos_neg`;
- `trace_distance:dtw_contact_envelope:same_task:nearest_positive`;
- `linear_probe:contact_envelope:same_task:ridge_l2_1` as a learned diagnostic,
  not the main method.

## Why This Is World-Model-Relevant

Robot world-model benchmarks increasingly expose a gap between plausible future
generation and executable future selection. EFV turns that gap into a concrete
test-time selection problem:

1. A generator produces multiple candidate futures.
2. Futures may come from video prediction, policy sampling, planner rollouts,
   inverse-dynamics reconstruction, or expert perturbations.
3. A training-free contact-envelope verifier ranks those futures.
4. The selected future is evaluated by executable task success.

This keeps the method generator-agnostic while still addressing a world-model
failure mode: plausible futures can violate contact timing, gripper timing, or
local execution feasibility.

## Current Mechanism Signal

RoboCasa365 already shows a large oracle-selection gap and strong robot-envelope
signals. A RoboTwin2 `handover_block` quick analysis on two complete
24-candidate seeds gives a sharper mechanism example:

- `rank0` is `0/2` while oracle is `2/2`;
- random selection is only `0.4/2` in expectation;
- action energy, length, smoothness, and action-only learned baselines are
  `0/2`;
- `prototype:contact_envelope:same_task:nearest_positive`,
  `prototype:contact_envelope:same_task:nearest_pos_neg`,
  `trace_distance:dtw_contact_envelope:same_task:nearest_positive`, and
  `linear_probe:contact_envelope:same_task:ridge_l2_1` are all `2/2`;
- object-relation-only selectors are not stable on this task
  (`object_relation_distribution` is `0/2`, DTW object relation is about
  `1.28/2` over anonymous remaps);
- the anti-template diagnostics find non-full-template successes and matched
  negatives in both cases.

At the candidate level:

- `first_action_rank0` fails;
- `full_gripper_aware` succeeds;
- non-template candidates such as `repeat_middle`, `contact_joint_perturb`,
  `repeat_precontact`, and `repeat_contact_long` also succeed;
- gripper timing/contact negatives such as `gripper_early_1`,
  `gripper_late_1`, `gripper_contact_pulse`, `long_gripper_contact_pulse`,
  `long_contact_joint_perturb_strong`, and `long_reverse_contact` fail.

This is the useful phenomenon: success is not simply "closest to full expert."
Some non-template futures are executable, while many globally similar or
energy-matched futures fail when the contact phase is wrong.

The current limitation is equally important: DTW template baselines also reach
`2/2` on this small handover window. This means the handover quick analysis can
support the contact-envelope-vs-shortcut claim, but it cannot yet support the
stronger claim that EFV beats expert-template matching. The RoboTwin2 paper
readiness gate therefore still fails until more tasks show selector margins
under anti-template pressure.

A stricter no-exact-expert counterfactual removes `full_gripper_aware` from the
same two cases and recomputes oracle labels. The pool remains recoverable:
`rank0` is still `0/2`, oracle is still `2/2`, random is `0.35/2`, and energy,
length, smoothness, and action-only baselines remain `0/2`. Contact-envelope
selectors remain `2/2` by selecting non-template successes such as
`repeat_precontact` and `repeat_middle`.

This counterfactual rules out the weakest explanation that the result only
comes from selecting the exact expert trace. It still does not rule out the
stronger template-distance explanation, because DTW action/contact baselines
also remain `2/2` by selecting non-template successful variants such as
`contact_joint_perturb` and `repeat_contact_long`.

## Key Reviewer Risks

Expert-template matching:

- Keep DTW-to-success as a baseline.
- Show low-DTW or energy-matched failures.
- Show non-template successes.

Shortcut detection:

- Keep rank0, random, action energy, smoothness, length, and action-only
  selectors.
- Report permissive tasks such as `press_stapler` as negative controls.

World-model relevance:

- Use RoboCasa365 and RoboTwin2 for executable ground truth.
- Add a world-model-facing diagnostic layer where candidate futures come from
  video/world-model outputs or video-to-action reconstructions.

No real robot:

- Do not claim deployment.
- Claim executable simulation evidence and a verifier interface that can be
  attached to world-model futures.

## Next Evidence Needed

The contact-envelope selector becomes paper-relevant only if the next RoboTwin2
window shows:

1. multiple tasks with rank0 failure and oracle success;
2. non-template successes;
3. contact/gripper hard negatives;
4. contact-envelope selector margins over action heuristics and full-trajectory
   DTW baselines;
5. no unsupported same-task selectors in the final table.

If this fails, the result should be downgraded to a diagnostic study rather than
sold as a main ICLR result.
