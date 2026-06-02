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
#   # subprocess (default) still needs LLAMA_SERVER_BIN on serve
#
#   ./scripts/phase14_backend_smoke.sh
#
# Optional:
#   RUN_E2E_GGUF=... RUN_E2E_TOOLS=0
#   RUN_E2E_PROXY_MODEL=pulled-tag  — required for Go /internal/render-chat (truncate_mode=tokenize)
#   RUN_E2E_INPROCESS=1  — fail if /health llama_backend != inprocess
#   RUN_E2E_LLAMA_CPP_PYTHON=1 — fail if backend != llama-cpp-python
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
echo "llama_backend=${_backend}"
_gguf_pf="${RUN_E2E_GGUF:-${LLAMA_MODEL:-}}"
if [[ -z "$_gguf_pf" ]]; then
  _gguf_pf=$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('llama_model') or '')" "$_health")
fi
echo "== Phase 14 backend smoke (runtime GPU + optional proxy/render-chat) =="
runtime_resume_if_needed "$_health"
smoke_runtime_require_phase14_endpoints "$RUNTIME_URL" "$_gguf_pf"
smoke_prepare_vram_for_runtime

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
# shellcheck disable=SC2086
env "${e2e_env[@]}" "${ROOT}/scripts/e2e_runtime_smoke.sh"

echo "PASS: phase14_backend_smoke"
