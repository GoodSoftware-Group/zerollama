#!/usr/bin/env bash
# Phase 14 GPU smoke against an already-running zerollama serve.
#
# Why not start serve here: backend is read at Python/Go startup; this script validates
# the binary you already built and the env on the running process.
#
# Start serve with the backend under test, then:
#
#   source ./scripts/phase14_serve_env.sh
#   export LLAMA_MODEL=/path/to/small.q8_0.gguf
#   export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess   # or llama-cpp-python
#   # or uncomment llama_backend: inprocess in runtime/configs/single_gpu.yaml
#   # subprocess (default) still needs LLAMA_SERVER_BIN on serve
#
#   ./scripts/phase14_backend_smoke.sh
#
# Optional:
#   RUN_E2E_GGUF=... RUN_E2E_TOOLS=0
#   RUN_E2E_PROXY_MODEL=pulled-tag  — required for Go /internal/render-chat (truncate_mode=tokenize)
#   RUN_E2E_INPROCESS=1  — fail if /health llama_backend != inprocess
#   RUN_E2E_LLAMA_CPP_PYTHON=1 — fail if backend != llama-cpp-python
#   RUN_E2E_LLAMA_CPP_PYTHON_GPU=1 — after generate, assert /health llama_cpp.gpu_mode=gpu
#   RUN_E2E_LLAMA_BACKEND_SOURCE=config|env|default — assert /health provenance (YAML vs env)
#   When source=config, RUN_E2E_INPROCESS / RUN_E2E_LLAMA_CPP_PYTHON are inferred from /health
#   unless already set (rejects subprocess — YAML key must be present).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime_smoke_lib.sh"

export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:8080}"
# Curl-only URL; do not export ZEROLLAMA_RUNTIME_URL (embed serve must not see it).
# Why RUNTIME_URL local: phase14_serve_env unsets URL for the serve child; smokes still
# need to curl :8081 without re-exporting URL into the environment.
RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"

if [[ -z "${LLAMA_MODEL:-}" && -z "${RUN_E2E_GGUF:-}" ]]; then
  echo "Set LLAMA_MODEL or RUN_E2E_GGUF (and configure the same on serve)" >&2
  exit 1
fi
if [[ -z "${RUN_E2E_PROXY_MODEL:-}" ]]; then
  echo "warn: RUN_E2E_PROXY_MODEL unset — render-chat tokenize check skipped (set a pulled local tag)" >&2
fi

echo "== Phase 14 preflight (current runtime binary) =="
smoke_runtime_require_listening "$RUNTIME_URL"
_health=$(runtime_fetch_health "$RUNTIME_URL")
_backend=$(smoke_runtime_llama_backend "$_health" strict)
_backend_source=$(smoke_runtime_llama_backend_source "$_health" strict)
echo "llama_backend=${_backend}"
echo "llama_backend_source=${_backend_source}"
smoke_runtime_assert_llama_backend_source "$_health" "${RUN_E2E_LLAMA_BACKEND_SOURCE:-}"
if [[ "${RUN_E2E_LLAMA_BACKEND_SOURCE:-}" == "config" ]]; then
  smoke_runtime_apply_backend_flags_from_health "$_health"
fi
if [[ "${RUN_E2E_LLAMA_BACKEND_SOURCE:-}" == "default" && "$_backend" != "subprocess" ]]; then
  echo "RUN_E2E_LLAMA_BACKEND_SOURCE=default but /health llama_backend=${_backend} (expected subprocess)" >&2
  exit 1
fi
if [[ "${RUN_E2E_LLAMA_CPP_PYTHON_GPU:-0}" == "1" ]]; then
  _wheel_ngl=$(python3 -c "import json,sys; print((json.loads(sys.argv[1]).get('llama_cpp') or {}).get('env_n_gpu_layers') or '')" "$_health")
  if [[ -z "$_wheel_ngl" ]]; then
    echo "RUN_E2E_LLAMA_CPP_PYTHON_GPU=1 but /health llama_cpp.env_n_gpu_layers unset" >&2
    echo "  Set ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS on serve and restart." >&2
    exit 1
  fi
  echo "llama_cpp.env_n_gpu_layers=${_wheel_ngl}"
fi
_gguf_pf="${RUN_E2E_GGUF:-${LLAMA_MODEL:-}}"
if [[ -z "$_gguf_pf" ]]; then
  _gguf_pf=$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('llama_model') or '')" "$_health")
fi
echo "== Phase 14 backend smoke (runtime GPU + optional proxy/render-chat) =="
runtime_resume_if_needed "$_health"
smoke_runtime_require_phase14_endpoints "$RUNTIME_URL" "$_gguf_pf"
smoke_prepare_vram_for_runtime
# WHY unset RUN_E2E_GGUF: broker already loaded this GGUF into the sidecar;
# repeating options.gguf on subsequent calls is redundant and forces the
# engine to validate path equality on every request (harmless but wasteful).
unset RUN_E2E_GGUF

e2e_env=(
  RUN_E2E_GPU=1
  RUN_E2E_PHASE14=1
  RUN_E2E_PROXY=0
)
if [[ -n "${RUN_E2E_PROXY_MODEL:-}" ]]; then
  e2e_env+=(RUN_E2E_PROXY=1)
else
  e2e_env+=(RUN_E2E_PROXY=0)
fi
[[ "${RUN_E2E_INPROCESS:-0}" == "1" ]] && e2e_env+=(RUN_E2E_INPROCESS=1)
[[ "${RUN_E2E_LLAMA_CPP_PYTHON:-0}" == "1" ]] && e2e_env+=(RUN_E2E_LLAMA_CPP_PYTHON=1)
[[ "${RUN_E2E_LLAMA_CPP_PYTHON_GPU:-0}" == "1" ]] && e2e_env+=(RUN_E2E_LLAMA_CPP_PYTHON_GPU=1)
[[ -n "${RUN_E2E_LLAMA_BACKEND_SOURCE:-}" ]] && e2e_env+=(RUN_E2E_LLAMA_BACKEND_SOURCE="${RUN_E2E_LLAMA_BACKEND_SOURCE}")
# shellcheck disable=SC2086
env "${e2e_env[@]}" "${ROOT}/scripts/e2e_runtime_smoke.sh"

echo "PASS: phase14_backend_smoke"
