#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:?usage: robotwin2_finalize_run.sh RUN_ROOT [task ...]}"
shift || true

TASKS=("$@")
if [ "${#TASKS[@]}" -eq 0 ]; then
  TASKS=(stack_blocks_two stamp_seal place_object_basket stack_bowls_two)
fi

PYTHON_BIN_CMD="${PYTHON_BIN:-python3}"
REQUIRE_CANDIDATES_PER_CASE="${REQUIRE_CANDIDATES_PER_CASE:-24}"
NUM_SWEEP_SEEDS="${NUM_SWEEP_SEEDS:-10}"
DEFAULT_SUITE="${DEFAULT_SUITE:-demo_clean_k5}"
REFRESH_ICLR_REPORTS="${REFRESH_ICLR_REPORTS:-0}"

mkdir -p "$RUN_ROOT/selectors"

echo "RUN_ROOT=$RUN_ROOT"
echo "TASKS=${TASKS[*]}"
echo "PYTHON_BIN=$PYTHON_BIN_CMD"
echo "REQUIRE_CANDIDATES_PER_CASE=$REQUIRE_CANDIDATES_PER_CASE"
echo "NUM_SWEEP_SEEDS=$NUM_SWEEP_SEEDS"
echo "DEFAULT_SUITE=$DEFAULT_SUITE"
echo "REFRESH_ICLR_REPORTS=$REFRESH_ICLR_REPORTS"

echo
echo "=== raw integrity audit ==="
PYTHONPATH=src "$PYTHON_BIN_CMD" -m umm_reward_evaluator.benchmarks.robotwin2_raw_integrity_report \
  --raw-root "$RUN_ROOT/raw" \
  --required-candidates-per-case "$REQUIRE_CANDIDATES_PER_CASE" \
  --output-json "$RUN_ROOT/selectors/robotwin2_raw_integrity_report.json" \
  --output-md "$RUN_ROOT/selectors/robotwin2_raw_integrity_report.md"

echo
echo "=== multitask analysis ==="
PYTHONPATH=src \
PYTHON_BIN="$PYTHON_BIN_CMD" \
REQUIRE_CANDIDATES_PER_CASE="$REQUIRE_CANDIDATES_PER_CASE" \
NUM_SWEEP_SEEDS="$NUM_SWEEP_SEEDS" \
DEFAULT_SUITE="$DEFAULT_SUITE" \
  scripts/robotwin2_multitask_analysis.sh "$RUN_ROOT" "${TASKS[@]}"

echo
echo "=== registry proposal ==="
PYTHONPATH=src "$PYTHON_BIN_CMD" -m umm_reward_evaluator.benchmarks.iclr_registry_proposal robotwin2 \
  --readiness-json "$RUN_ROOT/selectors/robotwin2_readiness_report.json" \
  --selector-table-json "$RUN_ROOT/selectors/robotwin2_selector_table.json" \
  --paper-gate-json "$RUN_ROOT/selectors/robotwin2_paper_readiness_gate.json" \
  --evidence "Generated from RoboTwin2 finalized run root: $RUN_ROOT" \
  --output-json "$RUN_ROOT/selectors/robotwin2_registry_entry_proposal.json" \
  --output-md "$RUN_ROOT/selectors/robotwin2_registry_entry_proposal.md"

echo
echo "Registry proposal written to:"
echo "  $RUN_ROOT/selectors/robotwin2_registry_entry_proposal.json"
echo "Inspect status before updating docs/iclr_evidence_stack_registry.json."

echo
echo "=== evidence card proposal ==="
PYTHONPATH=src "$PYTHON_BIN_CMD" -m umm_reward_evaluator.benchmarks.robotwin2_evidence_card \
  --run-root "$RUN_ROOT" \
  --output-card "$RUN_ROOT/selectors/robotwin2_evidence_card_proposal.json" \
  --output-validation-json "$RUN_ROOT/selectors/robotwin2_evidence_card_validation.json" \
  --output-validation-md "$RUN_ROOT/selectors/robotwin2_evidence_card_validation.md"

echo
echo "Evidence card proposal written to:"
echo "  $RUN_ROOT/selectors/robotwin2_evidence_card_proposal.json"

if [ "$REFRESH_ICLR_REPORTS" = "1" ]; then
  echo
  echo "=== refresh ICLR reports ==="
  scripts/iclr_refresh_reports.sh
fi
