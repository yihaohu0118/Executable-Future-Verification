#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:?usage: world_model_diagnostic_finalize_run.sh RUN_ROOT MANIFEST BENCHMARK YEAR LAYER VERIFIER_SCORE_KEY [CATEGORY_KEY ...]}"
MANIFEST="${2:?usage: world_model_diagnostic_finalize_run.sh RUN_ROOT MANIFEST BENCHMARK YEAR LAYER VERIFIER_SCORE_KEY [CATEGORY_KEY ...]}"
BENCHMARK="${3:?usage: world_model_diagnostic_finalize_run.sh RUN_ROOT MANIFEST BENCHMARK YEAR LAYER VERIFIER_SCORE_KEY [CATEGORY_KEY ...]}"
YEAR="${4:?usage: world_model_diagnostic_finalize_run.sh RUN_ROOT MANIFEST BENCHMARK YEAR LAYER VERIFIER_SCORE_KEY [CATEGORY_KEY ...]}"
LAYER="${5:?usage: world_model_diagnostic_finalize_run.sh RUN_ROOT MANIFEST BENCHMARK YEAR LAYER VERIFIER_SCORE_KEY [CATEGORY_KEY ...]}"
VERIFIER_SCORE_KEY="${6:?usage: world_model_diagnostic_finalize_run.sh RUN_ROOT MANIFEST BENCHMARK YEAR LAYER VERIFIER_SCORE_KEY [CATEGORY_KEY ...]}"
shift 6

CATEGORY_KEYS=("$@")
if [ "${#CATEGORY_KEYS[@]}" -eq 0 ]; then
  CATEGORY_KEYS=(metadata.scenario metadata.failure_category)
fi

PYTHON_BIN_CMD="${PYTHON_BIN:-python3}"
MIN_CASES="${MIN_CASES:-16}"
MIN_TASKS="${MIN_TASKS:-1}"
MIN_ORACLE_BETTER_CASES="${MIN_ORACLE_BETTER_CASES:-1}"
MIN_CANDIDATES_PER_CASE="${MIN_CANDIDATES_PER_CASE:-2}"
MIN_CATEGORIES="${MIN_CATEGORIES:-1}"
MIN_PLANNER_SCORE_ORACLE_GAP="${MIN_PLANNER_SCORE_ORACLE_GAP:-1}"
MIN_PLANNER_SCORE_FAILURES="${MIN_PLANNER_SCORE_FAILURES:-1}"
MIN_VERIFIER_PROXY_MARGIN="${MIN_VERIFIER_PROXY_MARGIN:-1.0}"
PROXY_SCORE_KEY="${PROXY_SCORE_KEY:-__planner_or_model__}"

mkdir -p "$RUN_ROOT/selectors"

diagnostic_gate_json="$RUN_ROOT/selectors/world_model_diagnostic_gate.json"
diagnostic_gate_md="$RUN_ROOT/selectors/world_model_diagnostic_gate.md"
selector_table_json="$RUN_ROOT/selectors/world_model_diagnostic_selector_table.json"
selector_table_md="$RUN_ROOT/selectors/world_model_diagnostic_selector_table.md"
readiness_json="$RUN_ROOT/selectors/world_model_diagnostic_readiness_gate.json"
readiness_md="$RUN_ROOT/selectors/world_model_diagnostic_readiness_gate.md"
registry_json="$RUN_ROOT/selectors/world_model_diagnostic_registry_entry_proposal.json"
registry_md="$RUN_ROOT/selectors/world_model_diagnostic_registry_entry_proposal.md"
card_json="$RUN_ROOT/selectors/world_model_diagnostic_evidence_card_proposal.json"
card_validation_json="$RUN_ROOT/selectors/world_model_diagnostic_evidence_card_validation.json"
card_validation_md="$RUN_ROOT/selectors/world_model_diagnostic_evidence_card_validation.md"

category_args=()
for key in "${CATEGORY_KEYS[@]}"; do
  category_args+=(--category-key "$key")
done

echo "RUN_ROOT=$RUN_ROOT"
echo "MANIFEST=$MANIFEST"
echo "BENCHMARK=$BENCHMARK"
echo "YEAR=$YEAR"
echo "LAYER=$LAYER"
echo "VERIFIER_SCORE_KEY=$VERIFIER_SCORE_KEY"
echo "CATEGORY_KEYS=${CATEGORY_KEYS[*]}"

echo
echo "=== world-model diagnostic gate ==="
"$PYTHON_BIN_CMD" -m umm_reward_evaluator.benchmarks.world_model_diagnostic_gate \
  --manifest "$MANIFEST" \
  --min-cases "$MIN_CASES" \
  --min-tasks "$MIN_TASKS" \
  --min-oracle-better-cases "$MIN_ORACLE_BETTER_CASES" \
  --min-candidates-per-case "$MIN_CANDIDATES_PER_CASE" \
  --min-categories "$MIN_CATEGORIES" \
  --min-planner-score-oracle-gap "$MIN_PLANNER_SCORE_ORACLE_GAP" \
  --min-planner-score-failures "$MIN_PLANNER_SCORE_FAILURES" \
  "${category_args[@]}" \
  --require-metadata-key metadata.verification_target \
  --output-json "$diagnostic_gate_json" \
  --output-md "$diagnostic_gate_md"

echo
echo "=== world-model diagnostic selector table ==="
"$PYTHON_BIN_CMD" -m umm_reward_evaluator.benchmarks.world_model_diagnostic_selector_table \
  --manifest "$MANIFEST" \
  --proxy-score-key "$PROXY_SCORE_KEY" \
  --verifier-score-key "$VERIFIER_SCORE_KEY" \
  --output-json "$selector_table_json" \
  --output-md "$selector_table_md"

echo
echo "=== world-model diagnostic readiness gate ==="
"$PYTHON_BIN_CMD" -m umm_reward_evaluator.benchmarks.world_model_diagnostic_readiness_gate \
  --diagnostic-gate-json "$diagnostic_gate_json" \
  --selector-table-json "$selector_table_json" \
  --verifier-selector "verifier_score:$VERIFIER_SCORE_KEY" \
  --proxy-selector planner_or_model_score \
  --min-verifier-proxy-margin "$MIN_VERIFIER_PROXY_MARGIN" \
  --output-json "$readiness_json" \
  --output-md "$readiness_md"

echo
echo "=== diagnostic registry proposal ==="
"$PYTHON_BIN_CMD" -m umm_reward_evaluator.benchmarks.iclr_registry_proposal diagnostic \
  --benchmark "$BENCHMARK" \
  --year "$YEAR" \
  --layer "$LAYER" \
  --diagnostic-gate-json "$diagnostic_gate_json" \
  --selector-table-json "$selector_table_json" \
  --diagnostic-readiness-json "$readiness_json" \
  --verifier-selector "verifier_score:$VERIFIER_SCORE_KEY" \
  --shortcut-control energy_or_magnitude \
  --shortcut-control action_only \
  --shortcut-control candidate_id_or_rank_remap \
  --evidence "Generated from world-model diagnostic finalized run root: $RUN_ROOT" \
  --output-json "$registry_json" \
  --output-md "$registry_md"

echo
echo "=== diagnostic evidence card proposal ==="
"$PYTHON_BIN_CMD" -m umm_reward_evaluator.benchmarks.world_model_diagnostic_evidence_card \
  --run-root "$RUN_ROOT" \
  --verifier-selector "verifier_score:$VERIFIER_SCORE_KEY" \
  --output-card "$card_json" \
  --output-validation-json "$card_validation_json" \
  --output-validation-md "$card_validation_md"

echo
echo "Diagnostic proposal written to:"
echo "  $registry_json"
echo "Evidence card proposal written to:"
echo "  $card_json"
echo "Inspect these before updating docs/iclr_evidence_stack_registry.json."
