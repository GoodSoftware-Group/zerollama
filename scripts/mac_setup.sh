#!/usr/bin/env bash
# One-shot Apple Silicon setup: Metal llama.cpp + uv runtime venv + optional sign-off.
#
#   ./scripts/mac_setup.sh
#
# Env:
#   LLAMA_CPP_ROOT=../llama.cpp
#   MAC_SETUP_SIGNOFF=0   — skip ./scripts/metal_signoff.sh
#   MAC_SETUP_BUILD=0     — skip llama.cpp build (lib already present)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runtime_uv_venv.sh
source "${ROOT}/scripts/runtime_uv_venv.sh"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "warn: mac_setup targets Darwin; continuing anyway" >&2
fi

LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-${ROOT}/../llama.cpp}"
export LLAMA_CPP_ROOT
export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_CPP_ROOT}/build/bin/libllama.dylib}"

echo "== Mac setup: uv runtime venv =="
runtime_uv_venv

if [[ "${MAC_SETUP_BUILD:-1}" == "1" ]]; then
  echo ""
  echo "== Mac setup: build Metal llama.cpp =="
  LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT}" "${ROOT}/scripts/build_llama_server.sh"
else
  if [[ ! -f "${LLAMA_CPP_LIB}" ]]; then
    echo "Missing ${LLAMA_CPP_LIB}; run with MAC_SETUP_BUILD=1 or build manually" >&2
    exit 1
  fi
  echo "skip build (MAC_SETUP_BUILD=0, ${LLAMA_CPP_LIB} present)"
fi

echo ""
echo "== Mac setup: doctor =="
export ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-$ROOT}"
if [[ -x "${ROOT}/zerollama" ]]; then
  "${ROOT}/zerollama" doctor || true
else
  (cd "${ROOT}" && go run . doctor) || true
fi

if [[ "${MAC_SETUP_SIGNOFF:-1}" == "1" ]]; then
  echo ""
  echo "== Mac setup: metal sign-off =="
  OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:8080}" "${ROOT}/scripts/metal_signoff.sh"
fi

echo ""
echo "PASS: mac_setup"
echo "  daily serve:  ./scripts/serve_mac_runtime.sh"
echo "  default chat: zerollama serve  (ggml Metal on :11434)"
