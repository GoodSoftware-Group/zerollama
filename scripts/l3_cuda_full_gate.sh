#!/usr/bin/env bash
# L3 CUDA production gate — 8k cache smoke + 27k production gate + merged verdict.
#
# WHY: L3 exit needs wiring (8k) and agent-scale prefix win (27k). Strict turn2/turn1
# ratio at 27k may stay >0.75 on 9B; cached-vs-no-cache PASS is the ship bar.
#
# Usage:
#   CUDA_LLAMA_MODEL=/root/eliza-1-9b-256k.gguf ./scripts/l3_cuda_full_gate.sh
#   CUDA_LLAMA_MODEL=/path/to/supernova-fp16.gguf ./scripts/l3_cuda_full_gate.sh  # optional re-validate
#
# Env:
#   CUDA_LLAMA_MODEL / LLAMA_MODEL  — production GGUF (required; 7B–9B+ class)
#   L3_GATE_DIR                     — artifact root (default /tmp/l3-cuda-full-gate)
#   L3_GATE_OUT                     — merged JSON (default ${L3_GATE_DIR}/gate.json)
#   L3_SKIP_SMOKE=1                 — 27k production-only re-run
#   L3_SKIP_PRODUCTION=1            — 8k smoke-only
#   L3_RUN_SPEC_CACHE=1             — also run l3_spec_cache_smoke (policy leg; ngram default)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "warn: l3_cuda_full_gate targets Linux CUDA; continuing anyway" >&2
fi

LLAMA_MODEL="${CUDA_LLAMA_MODEL:-${LLAMA_MODEL:-}}"
if [[ -z "${LLAMA_MODEL:-}" ]]; then
  echo "Set CUDA_LLAMA_MODEL to a production GGUF (9B+ recommended; eliza-1 9B is ship proxy)" >&2
  exit 1
fi
export CUDA_LLAMA_MODEL="${LLAMA_MODEL}"

L3_GATE_DIR="${L3_GATE_DIR:-/tmp/l3-cuda-full-gate}"
L3_GATE_OUT="${L3_GATE_OUT:-${L3_GATE_DIR}/gate.json}"
mkdir -p "${L3_GATE_DIR}"

L3_RAN_SMOKE=1
L3_RAN_PRODUCTION=1
[[ "${L3_SKIP_SMOKE:-0}" == "1" ]] && L3_RAN_SMOKE=0
[[ "${L3_SKIP_PRODUCTION:-0}" == "1" ]] && L3_RAN_PRODUCTION=0

echo "== L3 CUDA full gate =="
echo "model: ${LLAMA_MODEL}"
echo "artifacts: ${L3_GATE_DIR}/"
echo "report: ${L3_GATE_OUT}"
echo ""

SMOKE_JSON="${L3_GATE_DIR}/smoke-8k.json"
PROD_JSON="${L3_GATE_DIR}/production-27k.json"

if [[ "${L3_RAN_SMOKE}" == "1" ]]; then
  CUDA_LLAMA_MODEL="${LLAMA_MODEL}" \
  L3_NUM_CTX="${L3_SMOKE_NUM_CTX:-8192}" \
  L3_PREFIX_REPEAT="${L3_SMOKE_PREFIX_REPEAT:-150}" \
  L3_COMPARE_NO_CACHE="${L3_COMPARE_NO_CACHE:-1}" \
  L3_OUT="${SMOKE_JSON}" \
    "${ROOT}/scripts/l3_cache_smoke.sh"
fi

if [[ "${L3_RAN_PRODUCTION}" == "1" ]]; then
  CUDA_LLAMA_MODEL="${LLAMA_MODEL}" \
  L3_GATE_OUT="${PROD_JSON}" \
  L3_PREFIX_REPEAT="${L3_PROD_PREFIX_REPEAT:-150}" \
  L3_COMPARE_NO_CACHE="${L3_COMPARE_NO_CACHE:-1}" \
    "${ROOT}/scripts/l3_production_gate.sh"
fi

SPEC_JSON="${L3_GATE_DIR}/spec-cache-policy.json"
if [[ "${L3_RUN_SPEC_CACHE:-0}" == "1" ]]; then
  CUDA_LLAMA_MODEL="${LLAMA_MODEL}" \
  L3_SPEC_METHOD="${L3_SPEC_METHOD:-ngram}" \
  L3_OUT="${SPEC_JSON}" \
    "${ROOT}/scripts/l3_spec_cache_smoke.sh"
fi

python3 <<PY
import json
from pathlib import Path

gate_dir = Path("${L3_GATE_DIR}")
smoke_path = Path("${SMOKE_JSON}")
prod_path = Path("${PROD_JSON}")
spec_path = Path("${SPEC_JSON}")

def load(path):
    if path.is_file():
        return json.loads(path.read_text())
    return None

report = {
    "model": "${LLAMA_MODEL}",
    "legs": {
        "smoke_8k": ${L3_RAN_SMOKE} == 1,
        "production_27k": ${L3_RAN_PRODUCTION} == 1,
        "spec_cache_policy": ${L3_RUN_SPEC_CACHE:-0} == 1,
    },
    "smoke_8k": load(smoke_path),
    "production_27k": load(prod_path),
    "spec_cache_policy": load(spec_path),
}
Path("${L3_GATE_OUT}").write_text(json.dumps(report, indent=2) + "\n")
print(f"wrote ${L3_GATE_OUT}")
PY

"${ROOT}/scripts/l3_gate_report.sh" "${L3_GATE_OUT}"
