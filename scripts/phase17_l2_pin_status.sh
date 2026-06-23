#!/usr/bin/env bash
# Phase 17 / L2 pin status — read-only report for operator merge decisions.
#
# WHY: Criterion 7 (coordinate llama.cpp pin with borrowings L2) stays open until
# measured fork wins justify vendor merge. This script prints pins + doc pointers without GPU.
#
# Usage:
#   ./scripts/phase17_l2_pin_status.sh
#   P17_L2_OUT=/tmp/phase17-l2-pin-status.json ./scripts/phase17_l2_pin_status.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P17_L2_OUT="${P17_L2_OUT:-}"

STOCK_PIN="$(grep -E '^FETCH_HEAD=' "${ROOT}/Makefile.sync" | cut -d= -f2)"
VERSION="$(cat "${ROOT}/LLAMA_CPP_VERSION" 2>/dev/null || true)"
ELIZA_REF="96dd1a8466c84bdd419faf3866425260623fb6b0"
ELIZA_DIR="${ELIZA_LLAMA_CPP_ROOT:-${ROOT}/../eliza-llama.cpp}"

echo "== Phase 17 / L2 pin status =="
echo "zerollama stock pin (Makefile.sync): ${STOCK_PIN}"
echo "LLAMA_CPP_VERSION file:              ${VERSION}"
echo "eliza fork eval ref (gpu-profiles-l2): ${ELIZA_REF}"
echo "eliza sibling:                       ${ELIZA_DIR}"

if [[ -d "${ELIZA_DIR}/.git" ]]; then
  ELIZA_HEAD="$(git -C "${ELIZA_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "eliza sibling HEAD:                  ${ELIZA_HEAD}"
else
  echo "eliza sibling HEAD:                  (not cloned — optional for L2 gates)"
fi

echo ""
echo "L2 verdict (Jun 2026, documented): FAIL merge @ 8k CUDA — stock decode faster; vendor merge blocked."
echo "Gates: ./scripts/l2_cuda_full_gate.sh (CUDA) · ./scripts/l2_full_gate.sh (Metal)"
echo "Doc: docs/gpu-profiles-l2.md · docs/phase17-llama-server.md"

if [[ -n "${P17_L2_OUT}" ]]; then
  ELIZA_HEAD="${ELIZA_HEAD:-}"
  STOCK_PIN="${STOCK_PIN}" VERSION="${VERSION}" ELIZA_REF="${ELIZA_REF}" ELIZA_HEAD="${ELIZA_HEAD:-}" \
    P17_L2_OUT="${P17_L2_OUT}" python3 <<'PY'
import json, os, pathlib
out = pathlib.Path(os.environ["P17_L2_OUT"])
report = {
    "status": "documented_fail_merge",
    "stock_pin": os.environ.get("STOCK_PIN", ""),
    "llama_cpp_version": os.environ.get("VERSION", ""),
    "eliza_fork_ref": os.environ.get("ELIZA_REF", ""),
    "eliza_sibling_head": os.environ.get("ELIZA_HEAD") or None,
    "merge_blocked_reason": "5080 Jun 2026: stock wins decode @ 8k; long-ctx fork legs blocked",
    "docs": ["docs/gpu-profiles-l2.md", "docs/phase17-llama-server.md"],
    "gates": ["scripts/l2_cuda_full_gate.sh", "scripts/l2_full_gate.sh"],
}
out.write_text(json.dumps(report, indent=2) + "\n")
print(f"report: {out}")
PY
fi

echo ""
echo "PASS: phase17_l2_pin_status"
