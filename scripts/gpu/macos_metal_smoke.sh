#!/usr/bin/env bash
# macOS Metal / Apple Silicon smoke — no NVIDIA required.
#
# WHY: CUDA-centric gpu_5080_session does not run on Mac; this validates autoconfig,
# metal-unified VRAM probe, and coordination fields after serve is up.
#
#   zerollama serve   # separate terminal
#   ./scripts/gpu/macos_metal_smoke.sh
#
# Optional runtime infer (Metal llama-server):
#   export LLAMA_MODEL=/path/to/small.gguf LLAMA_SERVER_BIN=$(which llama-server)
#   RUN_E2E_GPU=1 ./scripts/gpu/macos_metal_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ZEROLLAMA_RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:8080}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "warn: macos_metal_smoke targets Darwin; continuing anyway" >&2
fi

echo "== Phase 12 preflight (no GPU) =="
"${ROOT}/scripts/phase/phase12_golden_ci.sh" go
# shellcheck source=scripts/runtime/runtime_uv_venv.sh
source "${ROOT}/scripts/runtime/runtime_uv_venv.sh"
runtime_uv_venv
(cd "${ROOT}/runtime" && PYTHONPATH=. "${RUNTIME_UV_PYTHON}" -m pytest \
  tests/test_host_memory_darwin.py \
  tests/test_metal_unified_probe.py \
  tests/test_m3_model_picker.py \
  tests/test_autoconfig.py::test_apple_silicon_yaml_inprocess_backend \
  tests/test_vram_yaml_defaults.py::test_apply_apple_silicon_repo_defaults \
  tests/test_gpu_profiles.py \
  tests/test_engine_inprocess_fallback.py -q)

echo "== coordination smoke =="
"${ROOT}/scripts/e2e/e2e_coordination_smoke.sh"

health=$(curl -sf -m 30 "${ZEROLLAMA_RUNTIME_URL%/}/health") || {
  echo "runtime /health failed — start zerollama serve with embedded or sidecar runtime" >&2
  exit 1
}

echo "== Metal autoconfig + probe =="
python3 <<'PY' "$health"
import json, sys
h = json.loads(sys.argv[1])
ac = h.get("autoconfig") or {}
probe = h.get("vram_probe_effective")
pick = ac.get("pick")
if pick not in ("apple_silicon", "custom"):
    print(f"warn: autoconfig.pick={pick!r} (expected apple_silicon on default Mac autoconfig)", file=sys.stderr)
else:
    print(f"autoconfig.pick: {pick}")
print(f"vram_probe_effective: {probe}")
if probe not in ("metal-unified", "skipped", None):
    print(f"note: probe={probe} (nvidia path on Mac is unusual)", file=sys.stderr)
gp = h.get("gpu_profile") or {}
if gp:
    print(f"gpu_profile.id: {gp.get('id')}")
    print(f"gpu_profile.bucket_label: {gp.get('bucket_label')}")
    print(f"gpu_profile.n_parallel: {gp.get('n_parallel')}")
    if gp.get("unified_memory_gb") is not None:
        print(f"gpu_profile.unified_memory_gb: {gp.get('unified_memory_gb')}")
else:
    print("warn: gpu_profile missing from /health (ZEROLLAMA_GPU_PROFILE=0?)", file=sys.stderr)
PY

if [[ "${RUN_E2E_GPU:-0}" == "1" ]]; then
  if [[ -z "${LLAMA_MODEL:-}" || -z "${LLAMA_SERVER_BIN:-}" ]]; then
    echo "RUN_E2E_GPU=1 needs LLAMA_MODEL and LLAMA_SERVER_BIN (Metal llama-server)" >&2
    exit 1
  fi
  export RUN_E2E_GGUF="${RUN_E2E_GGUF:-$LLAMA_MODEL}"
  echo "== runtime GPU smoke (Metal llama-server) =="
  "${ROOT}/scripts/e2e/e2e_runtime_smoke.sh"
fi

if [[ "${RUN_E2E_FLASH_MOE_TIER0:-0}" == "1" ]]; then
  echo "== Flash-MoE tier 0 (toolchain only) =="
  FLASH_MOE_SKIP_GO_TEST="${FLASH_MOE_SKIP_GO_TEST:-0}" "${ROOT}/scripts/phase/flash_moe_smoke.sh"
fi

echo "PASS: macos_metal_smoke"
