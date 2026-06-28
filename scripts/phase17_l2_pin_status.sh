#!/usr/bin/env bash
# Phase 17 / L2 pin status — read-only report for operator merge decisions.
#
# WHY: Criterion 7 (coordinate llama.cpp pin) tracks unified runtime binary vs
# in-process vendor until measured fork-profile wins justify vendor rebase.
#
# Usage:
#   ./scripts/phase17_l2_pin_status.sh
#   P17_L2_OUT=/tmp/phase17-l2-pin-status.json ./scripts/phase17_l2_pin_status.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P17_L2_OUT="${P17_L2_OUT:-}"

VENDOR_PIN="$(grep -E '^FETCH_HEAD=' "${ROOT}/Makefile.sync" | cut -d= -f2)"
VERSION="$(cat "${ROOT}/LLAMA_CPP_VERSION" 2>/dev/null || true)"
UNIFIED_REF="$(cat "${ROOT}/LLAMA_CPP_COMMIT" 2>/dev/null | tr -d '[:space:]' || true)"
UNIFIED_REF="${UNIFIED_REF:-c84b30200c8d512c00c9d61c96bed078f1c0024d}"
UNIFIED_DIR="${LLAMA_CPP_ROOT:-${ROOT}/../llama.cpp}"

echo "== Phase 17 / L2 pin status =="
echo "in-process vendor pin (Makefile.sync): ${VENDOR_PIN}"
echo "LLAMA_CPP_VERSION file:                ${VERSION}"
echo "unified runtime ref (LLAMA_CPP_COMMIT): ${UNIFIED_REF}"
echo "unified sibling:                       ${UNIFIED_DIR}"

if [[ -d "${UNIFIED_DIR}/.git" ]]; then
  UNIFIED_HEAD="$(git -C "${UNIFIED_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "unified sibling HEAD:                  ${UNIFIED_HEAD}"
else
  echo "unified sibling HEAD:                  (not cloned — run ./scripts/build_llama_server.sh)"
fi

echo ""
echo "L2 verdict (Jun 2026, documented): FAIL fork-profile merge @ 8k CUDA — L1 q8_0 faster; fork profiles stay opt-in."
echo "Gates: ./scripts/l2_cuda_full_gate.sh (CUDA) · ./scripts/l2_full_gate.sh (Metal)"
echo "Doc: docs/gpu-profiles-l2.md · docs/phase17-llama-server.md"

if [[ -n "${P17_L2_OUT}" ]]; then
  UNIFIED_HEAD="${UNIFIED_HEAD:-}"
  VENDOR_PIN="${VENDOR_PIN}" VERSION="${VERSION}" UNIFIED_REF="${UNIFIED_REF}" UNIFIED_HEAD="${UNIFIED_HEAD:-}" \
    P17_L2_OUT="${P17_L2_OUT}" python3 <<'PY'
import json, os, pathlib
out = pathlib.Path(os.environ["P17_L2_OUT"])
report = {
    "status": "documented_fail_fork_profile_merge",
    "vendor_pin": os.environ.get("VENDOR_PIN", ""),
    "llama_cpp_version": os.environ.get("VERSION", ""),
    "unified_runtime_ref": os.environ.get("UNIFIED_REF", ""),
    "unified_sibling_head": os.environ.get("UNIFIED_HEAD") or None,
    "merge_blocked_reason": "5080 Jun 2026: L1 q8_0 wins decode @ 8k; fork KV profiles opt-in until gate passes",
    "docs": ["docs/gpu-profiles-l2.md", "docs/phase17-llama-server.md"],
    "gates": ["scripts/l2_cuda_full_gate.sh", "scripts/l2_full_gate.sh"],
}
out.write_text(json.dumps(report, indent=2) + "\n")
print(f"report: {out}")
PY
fi

echo ""
echo "PASS: phase17_l2_pin_status"
