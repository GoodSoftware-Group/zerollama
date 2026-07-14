#!/usr/bin/env bash
# Phase 17 / L2 pin status — read-only report for operator merge decisions.
#
# WHY: Criterion 7 tracks whether fork KV *profiles* (QJL/Polar/TBQ) should become
# defaults. Kernels are already extracted onto ggml-org via patches 0026–0030;
# CUDA L2 completeness is 0067–0070 (fattn TBQ, CLI KV types, SET_ROWS, CPU traits).
# Vendor rebase of the full elizaOS tree is no longer the merge question.
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
UNIFIED_REF="${UNIFIED_REF:-8f114a9b573b69035299f9b924047f53c1e22c7e}"
UNIFIED_DIR="${LLAMA_CPP_ROOT:-${ROOT}/../llama.cpp}"
VENDOR_DIR="${ROOT}/vendor/llama-cpp-${VENDOR_PIN}"

# Built vendor fallback when pin dir not materialised yet.
BUILT_VENDOR=""
if [[ -x "${VENDOR_DIR}/build/bin/llama-server" ]]; then
  BUILT_VENDOR="${VENDOR_DIR}"
else
  _cand="$(ls -1dt "${ROOT}"/vendor/llama-cpp-*/build/bin/llama-server 2>/dev/null | head -1 || true)"
  if [[ -n "${_cand}" ]]; then
    BUILT_VENDOR="$(cd "$(dirname "${_cand}")/../.." && pwd)"
  fi
fi

echo "== Phase 17 / L2 pin status =="
echo "in-process vendor pin (Makefile.sync): ${VENDOR_PIN}"
echo "LLAMA_CPP_VERSION file:                ${VERSION}"
echo "unified runtime ref (LLAMA_CPP_COMMIT): ${UNIFIED_REF}"
echo "vendor tree (pin path):                ${VENDOR_DIR}"
if [[ -d "${VENDOR_DIR}" ]]; then
  echo "vendor tree present:                   yes"
else
  echo "vendor tree present:                   NO — run: make -f Makefile.sync clean apply-patches"
fi
echo "built llama-server vendor:             ${BUILT_VENDOR:-"(none)"}"
echo "unified sibling:                       ${UNIFIED_DIR}"

if [[ -d "${UNIFIED_DIR}/.git" ]]; then
  UNIFIED_HEAD="$(git -C "${UNIFIED_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "unified sibling HEAD:                  ${UNIFIED_HEAD}"
else
  echo "unified sibling HEAD:                  (not cloned — prefer vendor build)"
fi

echo ""
echo "L2 model (Jul 2026): QJL/Polar/TBQ extracted as patches 0026–0030 on ggml-org;"
echo "  CUDA L2 follow-ups 0067–0070 (fattn TBQ helpers, CLI types, SET_ROWS, CPU type_traits)."
echo "  fork *profiles* stay opt-in (ZEROLLAMA_LLAMA_FORK) until gate passes."
  echo "  VRAM opt-in: ZEROLLAMA_LLAMA_FORK=1 (defaults to TBQ / FORK_PROFILE=vram),"
  echo "  or runtime/configs/dual_4090_vram.yaml — long-ctx −27…−35% VRAM (4090;"
  echo "  direct llama-server −5…−6% decode; 8f sidecar re-gate −20…−21%)."
  echo "  Speed profile (qjl/polar): FORK_PROFILE=speed — runs on 8f but FAIL tok/s"
  echo "  (−48…−85% @ 8k/27k) — keep experimental."
echo "Ship-of-record (5080 eliza-1, Jun 2026): FAIL default profiles — stock q8_0 faster @ 8k/27k."
echo "Gates: ./scripts/l2_cuda_full_gate.sh (CUDA) · ./scripts/l2_full_gate.sh (Metal)"
echo "Doc: docs/gpu-profiles-l2.md · docs/phase17-llama-server.md · runtime/LLAMA_CPP_PIN.md"

if [[ -n "${P17_L2_OUT}" ]]; then
  UNIFIED_HEAD="${UNIFIED_HEAD:-}"
  VENDOR_PIN="${VENDOR_PIN}" VERSION="${VERSION}" UNIFIED_REF="${UNIFIED_REF}" \
    UNIFIED_HEAD="${UNIFIED_HEAD:-}" VENDOR_DIR="${VENDOR_DIR}" BUILT_VENDOR="${BUILT_VENDOR}" \
    P17_L2_OUT="${P17_L2_OUT}" python3 <<'PY'
import json, os, pathlib
out = pathlib.Path(os.environ["P17_L2_OUT"])
vendor_dir = os.environ.get("VENDOR_DIR", "")
report = {
    "status": "fork_profiles_opt_in_kernels_extracted",
    "vendor_pin": os.environ.get("VENDOR_PIN", ""),
    "llama_cpp_version": os.environ.get("VERSION", ""),
    "unified_runtime_ref": os.environ.get("UNIFIED_REF", ""),
    "unified_sibling_head": os.environ.get("UNIFIED_HEAD") or None,
    "vendor_tree_present": pathlib.Path(vendor_dir).is_dir() if vendor_dir else False,
    "built_vendor_llama_server": os.environ.get("BUILT_VENDOR") or None,
    "merge_blocked_reason": (
        "5080 Jun 2026 ship gate: L1 q8_0 wins decode @ 8k/27k on eliza-1; "
        "kernels already in patches 0026-0030 — flip defaults only when gate passes"
    ),
    "docs": [
        "docs/gpu-profiles-l2.md",
        "docs/phase17-llama-server.md",
        "runtime/LLAMA_CPP_PIN.md",
    ],
    "gates": ["scripts/l2_cuda_full_gate.sh", "scripts/l2_full_gate.sh"],
}
out.write_text(json.dumps(report, indent=2) + "\n")
print(f"report: {out}")
PY
fi

echo ""
echo "PASS: phase17_l2_pin_status"
