# RoboTwin2 Stack Blocks Two Seed 2 Interim

This is an operational/interim note, not paper evidence. The case is still
incomplete: `stack_blocks_two` seed 2 currently has 22 of the required 24
`targeted_energy_matched` candidates.

Run root:

```text
/home/yihao_hyh/efv_runs/robotwin2_pressure_closure_20260614
```

Active resume launcher:

```text
/home/yihao_hyh/efv_runs/robotwin2_pressure_closure_20260614_stack_only_resume_20260614_055011_stable180.log
```

## Current Candidate Outcomes

| Candidate | Success |
| --- | ---: |
| `first_action_rank0` | false |
| `full_gripper_aware` | true |
| `first_half` | false |
| `drop_last` | true |
| `reverse` | false |
| `noop` | false |
| `repeat_middle` | true |
| `stride2_hold_endpoint` | false |
| `gripper_early_1` | false |
| `gripper_late_1` | true |
| `contact_joint_perturb` | true |
| `repeat_precontact` | true |
| `repeat_contact_long` | true |
| `repeat_middle_drop_final` | true |
| `delete_contact_step` | false |
| `contact_joint_perturb_strong` | true |
| `contact_joint_offset_small` | true |
| `gripper_contact_pulse` | false |
| `gripper_contact_pulse_wide` | false |
| `long_gripper_contact_pulse` | false |
| `long_gripper_contact_pulse_wide` | false |
| `long_contact_joint_perturb_strong` | true |

Likely missing candidates:

- `long_gripper_late_1`
- `long_reverse_contact`

## Interim Interpretation

The partial already has the structure needed for an anti-template pressure
case:

- rank0 fails while the full executable future succeeds;
- several non-full hard positives succeed, including `drop_last`,
  `repeat_middle`, `contact_joint_perturb_strong`, and
  `long_contact_joint_perturb_strong`;
- several matched hard negatives fail, including `delete_contact_step` and the
  gripper-contact pulse variants;
- no candidate has a `candidate_error` row in the current 22-line partial;
- metadata contains `state_trace`, so relation/contact selectors are possible.

Do not promote this to the evidence registry until the case reaches all 24
candidates and passes the selector, anti-template, and main-table gates.
