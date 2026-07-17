#!/usr/bin/env bash
# Phase 11 Mac admission smoke — unified-memory policy + coordination (no CUDA).
#
# WHY: gpu_5080_session validates Phase 11 on discrete VRAM; Mac needs apple_silicon.yaml
# defaults + metal-unified probe + inference-first gates without NVML.
#
# Offline (always):
#   admission / inference_policy / defer / ggml backlog pytest
#
# Live (when :8081 /health responds):
#   e2e_coordination_smoke + apple_silicon vram/admission fields
#
# Usage:
#   ./scripts/phase/phase11_metal_admission_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ZEROLLAMA_RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:8080}"

echo "== Phase 11 admission pytest (offline) =="
# shellcheck source=scripts/runtime/runtime_uv_venv.sh
source "${ROOT}/scripts/runtime/runtime_uv_venv.sh"
runtime_uv_venv
(
  cd "${ROOT}/runtime"
  PYTHONPATH=. "${RUNTIME_UV_PYTHON}" -m pytest \
    tests/test_admission.py \
    tests/test_inference_policy.py \
    tests/test_defer_admission.py \
    tests/test_ggml_backlog_admission.py \
    tests/test_ggml_paused_admission.py \
    tests/test_priority_admission.py \
    tests/test_scheduler_low_dequeue.py \
    tests/test_admit_vram_precheck.py \
    tests/test_vram_yaml_defaults.py::test_apply_apple_silicon_repo_defaults \
    tests/test_metal_unified_probe.py \
    -q
)

if curl -sf -m 5 "${ZEROLLAMA_RUNTIME_URL%/}/health" >/dev/null 2>&1; then
  echo ""
  echo "== Phase 11 coordination + /health admission (live) =="
  "${ROOT}/scripts/e2e/e2e_coordination_smoke.sh"
  health_json=$(curl -sf -m 30 "${ZEROLLAMA_RUNTIME_URL%/}/health")
  export HEALTH_JSON="$health_json"
  python3 <<'PY'
import json, os
h = json.loads(os.environ["HEALTH_JSON"])
ad = h.get("admission") or {}
ac = h.get("autoconfig") or {}
if ac.get("pick") not in ("apple_silicon", "custom"):
    raise SystemExit(f"expected autoconfig apple_silicon on Mac, got {ac!r}")
min_free = ad.get("vram_min_free_configured")
reserve = ad.get("vram_training_reserve_configured")
if min_free is None or reserve is None:
    raise SystemExit(f"missing admission vram config on /health: {ad!r}")
# apple_silicon.yaml defaults (512MiB / 1GiB) unless env override
if min_free > 600 * 1024**2:
    print(f"warn: vram_min_free_configured={min_free} (env may override 512MiB yaml default)")
else:
    print(f"admission.vram_min_free_configured: {min_free}")
print(f"admission.vram_training_reserve_configured: {reserve}")
if ad.get("inference_policy") is False:
    raise SystemExit("inference_policy should be on by default")
print(f"vram_probe_effective: {h.get('vram_probe_effective')}")
print("ok: Phase 11 live admission fields")
PY
else
  echo "skip: live coordination (runtime /health not on ${ZEROLLAMA_RUNTIME_URL})"
fi

echo "PASS: phase11_metal_admission_smoke"
