#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:?usage: robotwin2_multitask_analysis.sh RUN_ROOT [task ...]}"
shift || true

TASKS=("$@")
if [ "${#TASKS[@]}" -eq 0 ]; then
  TASKS=(stack_blocks_two open_laptop handover_block press_stapler)
fi

REQUIRE_CANDIDATES_PER_CASE="${REQUIRE_CANDIDATES_PER_CASE:-24}"
NUM_SWEEP_SEEDS="${NUM_SWEEP_SEEDS:-10}"
DEFAULT_SUITE="${DEFAULT_SUITE:-demo_clean_k5}"
MIN_MAIN_TABLE_CASES="${MIN_MAIN_TABLE_CASES:-1}"
MIN_ORACLE_BETTER_CASES="${MIN_ORACLE_BETTER_CASES:-1}"

mkdir -p "$RUN_ROOT/manifests" "$RUN_ROOT/selectors"
readiness_json="$RUN_ROOT/selectors/robotwin2_readiness_report.json"
readiness_md="$RUN_ROOT/selectors/robotwin2_readiness_report.md"

for task in "${TASKS[@]}"; do
  input_dir="$RUN_ROOT/raw/$task"
  if [ ! -d "$input_dir" ]; then
    echo "skip $task: missing $input_dir"
    continue
  fi

  manifest="$RUN_ROOT/manifests/${task}_targeted_energy_matched_manifest.jsonl"
  summary="$RUN_ROOT/manifests/${task}_targeted_energy_matched_summary.json"
  diagnostics_json="$RUN_ROOT/selectors/${task}_targeted_energy_matched_diagnostics.json"
  diagnostics_md="$RUN_ROOT/selectors/${task}_targeted_energy_matched_diagnostics.md"
  trace_audit_json="$RUN_ROOT/selectors/${task}_targeted_energy_matched_trace_field_audit.json"
  trace_audit_md="$RUN_ROOT/selectors/${task}_targeted_energy_matched_trace_field_audit.md"
  base_gate_json="$RUN_ROOT/selectors/${task}_targeted_energy_matched_main_table_gate.json"
  relation_gate_json="$RUN_ROOT/selectors/${task}_targeted_energy_matched_relation_gate.json"
  sweep_json="$RUN_ROOT/selectors/${task}_targeted_energy_matched_rankrand_sweep.json"
  failures_json="$RUN_ROOT/selectors/${task}_targeted_energy_matched_failure_analysis.json"
  failures_md="$RUN_ROOT/selectors/${task}_targeted_energy_matched_failure_analysis.md"

  python -m umm_reward_evaluator.benchmarks.robotwin2_trace_to_manifest \
    --input-dir "$input_dir" \
    --output-manifest "$manifest" \
    --output-summary "$summary" \
    --default-suite "$DEFAULT_SUITE" \
    --require-candidates-per-case "$REQUIRE_CANDIDATES_PER_CASE" \
    --drop-cases-with-candidate-error

  cases="$(python - "$summary" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(json.load(f).get("cases", 0))
PY
)"
  if [ "$cases" = "0" ]; then
    echo "skip selector analysis for $task: no complete cases"
    continue
  fi

  if python -m umm_reward_evaluator.benchmarks.robotwin2_main_table_gate \
    --manifest "$manifest" \
    --output "$base_gate_json" \
    --required-candidates-per-case "$REQUIRE_CANDIDATES_PER_CASE" \
    --min-cases "$MIN_MAIN_TABLE_CASES" \
    --min-oracle-better-cases "$MIN_ORACLE_BETTER_CASES"; then
    echo "base main-table gate passed for $task: $base_gate_json"
  else
    echo "base main-table gate failed for $task: $base_gate_json"
  fi

  if python -m umm_reward_evaluator.benchmarks.robotwin2_main_table_gate \
    --manifest "$manifest" \
    --output "$relation_gate_json" \
    --required-candidates-per-case "$REQUIRE_CANDIDATES_PER_CASE" \
    --min-cases "$MIN_MAIN_TABLE_CASES" \
    --min-oracle-better-cases "$MIN_ORACLE_BETTER_CASES" \
    --require-feature object_relation_distribution \
    --require-feature phase_object_relation_joint_gripper_distribution; then
    echo "relation main-table gate passed for $task: $relation_gate_json"
  else
    echo "relation main-table gate failed for $task: $relation_gate_json"
  fi

  python -m umm_reward_evaluator.benchmarks.robotwin2_trace_field_audit \
    --manifest "$manifest" \
    --output-json "$trace_audit_json" \
    --output-md "$trace_audit_md"

  python -m umm_reward_evaluator.benchmarks.robotwin2_antitemplate_diagnostics \
    --manifest "$manifest" \
    --output-json "$diagnostics_json" \
    --output-md "$diagnostics_md" \
    --feature-mode dtw_joint_gripper

  python -m umm_reward_evaluator.benchmarks.robotwin2_rank_randomization_sweep \
    --manifest "$manifest" \
    --output "$sweep_json" \
    --num-seeds "$NUM_SWEEP_SEEDS" \
    --mode failure_rank0_shuffle_rest \
    --remap-candidate-ids

  python -m umm_reward_evaluator.benchmarks.robotwin2_selector_failure_analysis \
    --manifest "$manifest" \
    --output-json "$failures_json" \
    --output-md "$failures_md" \
    --num-seeds "$NUM_SWEEP_SEEDS" \
    --mode failure_rank0_shuffle_rest \
    --remap-candidate-ids
done

python -m umm_reward_evaluator.benchmarks.robotwin2_readiness_report \
  --selectors-dir "$RUN_ROOT/selectors" \
  --output-json "$readiness_json" \
  --output-md "$readiness_md"
