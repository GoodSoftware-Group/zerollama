#!/usr/bin/env bash
# Phase 13 Mac VRAM estimate smoke — metal-unified probe + suggest/clamp policy.
#
# WHY: Phase 13 is estimate/calibration (distinct from Phase 11 admission throttling).
# Mac uses apple_silicon.yaml vram block + metal-unified free bytes, not NVML.
#
# Offline:
#   vram suggest / autotune / yaml defaults pytest
#
# Live (runtime up + GGUF path):
#   POST /internal/vram-estimate + gpu_phase13_snapshot JSON
#
# Usage:
#   ./scripts/phase/phase13_metal_vram_smoke.sh
#   M3_LLAMA_MODEL=/path/to/small.gguf ./scripts/phase/phase13_metal_vram_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/runtime/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime/runtime_smoke_lib.sh"
export ZEROLLAMA_RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
export ZEROLLAMA_REPO_ROOT="${ZEROLLAMA_REPO_ROOT:-$ROOT}"
P13_OUT="${P13_OUT:-/tmp/phase13-metal-vram-smoke.json}"

echo "== Phase 13 VRAM pytest (offline) =="
# shellcheck source=scripts/runtime/runtime_uv_venv.sh
source "${ROOT}/scripts/runtime/runtime_uv_venv.sh"
runtime_uv_venv
(
  cd "${ROOT}/runtime"
  PYTHONPATH=. "${RUNTIME_UV_PYTHON}" -m pytest \
    tests/test_vram_suggest.py \
    tests/test_gpu_vram.py \
    tests/test_internal_vram_estimate.py \
    tests/test_vram_yaml_defaults.py \
    tests/test_gpu_snapshot.py::test_recommend_apple_silicon_autoconfig \
    tests/test_engine_health_vram.py \
    -q
)

if ! curl -sf -m 5 "${ZEROLLAMA_RUNTIME_URL%/}/health" >/dev/null 2>&1; then
  echo "skip: live vram-estimate (runtime /health not on ${ZEROLLAMA_RUNTIME_URL})"
  echo "PASS: phase13_metal_vram_smoke (offline only)"
  exit 0
fi

if [[ -n "${CUDA_LLAMA_MODEL:-}" ]]; then
  export M3_LLAMA_MODEL="${CUDA_LLAMA_MODEL}"
fi
if [[ -z "${M3_LLAMA_MODEL:-${LLAMA_MODEL:-}}" ]]; then
  smoke_m3_resolve_signoff_model 2>/dev/null || true
fi
GGUF="${M3_LLAMA_MODEL:-${LLAMA_MODEL:-}}"
if [[ -z "${GGUF}" || ! -f "${GGUF}" ]]; then
  echo "skip: live snapshot (set M3_LLAMA_MODEL to a local GGUF blob)"
  echo "PASS: phase13_metal_vram_smoke (offline only)"
  exit 0
fi

echo ""
echo "== Phase 13 live vram-estimate + snapshot =="
estimate_json=$(curl -sf -m 120 -X POST "${ZEROLLAMA_RUNTIME_URL%/}/internal/vram-estimate" \
  -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"gguf":sys.argv[1],"num_ctx":4096}))' "$GGUF")")
export ESTIMATE_JSON="$estimate_json"
python3 <<'PY'
import json, os
est = json.loads(os.environ["ESTIMATE_JSON"])
bud = est.get("vram_budget") or est.get("budget") or {}
if "suggested_max_num_ctx" not in bud and "suggested_max_num_ctx" not in est:
    raise SystemExit(f"missing suggested_max_num_ctx in estimate: {est!r}")
suggest = bud.get("suggested_max_num_ctx", est.get("suggested_max_num_ctx"))
print(f"suggested_max_num_ctx: {suggest}")
policy = est.get("vram_num_ctx_policy") or {}
if policy.get("clamp_enabled"):
    print("warn: clamp enabled on serve (yaml default is off)")
else:
    print("vram_num_ctx clamp: off (expected default)")
print("ok: /internal/vram-estimate")
PY

export GPU_PHASE13_SNAPSHOT_OUT="${P13_OUT}"
export LLAMA_MODEL="${GGUF}"
"${ROOT}/scripts/gpu/gpu_phase13_snapshot.sh" --gguf "${GGUF}" --num-ctx 4096 >/dev/null
echo "wrote ${P13_OUT}"
echo "PASS: phase13_metal_vram_smoke"
