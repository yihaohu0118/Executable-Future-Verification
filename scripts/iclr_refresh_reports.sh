#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
EVIDENCE_JSON="${EVIDENCE_JSON:-docs/iclr_evidence_stack_registry.json}"
DOCS_DIR="${DOCS_DIR:-docs}"
REQUIRE_EVIDENCE_CARDS="${REQUIRE_EVIDENCE_CARDS:-1}"

if [ ! -f "$EVIDENCE_JSON" ]; then
  echo "missing evidence registry: $EVIDENCE_JSON" >&2
  exit 2
fi

CARD_ARGS=()
if [ "$REQUIRE_EVIDENCE_CARDS" = "1" ]; then
  CARD_ARGS=(--require-evidence-cards --evidence-card-root .)
fi

echo "EVIDENCE_JSON=$EVIDENCE_JSON"
echo "DOCS_DIR=$DOCS_DIR"
echo "REQUIRE_EVIDENCE_CARDS=$REQUIRE_EVIDENCE_CARDS"

echo
echo "=== evidence stack gate ==="
set +e
PYTHONPATH=src "$PYTHON_BIN" -m umm_reward_evaluator.benchmarks.iclr_evidence_stack_gate \
  --evidence-json "$EVIDENCE_JSON" \
  "${CARD_ARGS[@]}" \
  --output-json "$DOCS_DIR/iclr_evidence_stack_gate_result.json" \
  --output-md "$DOCS_DIR/iclr_evidence_stack_gate_result.md"
GATE_STATUS=$?
set -e
echo "evidence_stack_gate_exit=$GATE_STATUS"

echo
echo "=== claim report ==="
PYTHONPATH=src "$PYTHON_BIN" -m umm_reward_evaluator.benchmarks.iclr_claim_report \
  --evidence-json "$EVIDENCE_JSON" \
  --output-json "$DOCS_DIR/iclr_claim_report_result.json" \
  --output-md "$DOCS_DIR/iclr_claim_report_result.md"

echo
echo "=== status report ==="
PYTHONPATH=src "$PYTHON_BIN" -m umm_reward_evaluator.benchmarks.iclr_status_report \
  --evidence-json "$EVIDENCE_JSON" \
  "${CARD_ARGS[@]}" \
  --output-json "$DOCS_DIR/iclr_status_report_current.json" \
  --output-md "$DOCS_DIR/iclr_status_report_current.md"

echo
echo "=== gap report ==="
PYTHONPATH=src "$PYTHON_BIN" -m umm_reward_evaluator.benchmarks.iclr_gap_report \
  --evidence-json "$EVIDENCE_JSON" \
  "${CARD_ARGS[@]}" \
  --output-json "$DOCS_DIR/iclr_gap_report_current.json" \
  --output-md "$DOCS_DIR/iclr_gap_report_current.md"

echo
echo "=== boss dashboard ==="
PYTHONPATH=src "$PYTHON_BIN" -m umm_reward_evaluator.benchmarks.iclr_boss_dashboard \
  --evidence-json "$EVIDENCE_JSON" \
  "${CARD_ARGS[@]}" \
  --output-json "$DOCS_DIR/iclr_boss_dashboard_current.json" \
  --output-md "$DOCS_DIR/iclr_boss_dashboard_current.md"

echo
echo "=== diagnostic closure plan ==="
PYTHONPATH=src "$PYTHON_BIN" -m umm_reward_evaluator.benchmarks.world_model_diagnostic_closure_plan \
  --evidence-json "$EVIDENCE_JSON" \
  --output-json "$DOCS_DIR/world_model_diagnostic_closure_plan_current.json" \
  --output-md "$DOCS_DIR/world_model_diagnostic_closure_plan_current.md"

echo
echo "=== world-model artifact audit ==="
PYTHONPATH=src "$PYTHON_BIN" -m umm_reward_evaluator.benchmarks.world_model_artifact_audit \
  --evidence-json "$EVIDENCE_JSON" \
  --output-json "$DOCS_DIR/world_model_artifact_audit_current.json" \
  --output-md "$DOCS_DIR/world_model_artifact_audit_current.md"

echo
echo "refreshed ICLR evidence reports under $DOCS_DIR"
echo "gate_exit=$GATE_STATUS"
