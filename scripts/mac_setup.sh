#!/usr/bin/env bash
# One-shot macOS dev setup: build zerollama, uv venvs, Metal llama.cpp, doctor, optional sign-off.
#
# Prerequisites (once per machine):
#   - Xcode Command Line Tools:  xcode-select --install
#   - Go 1.22+ from https://go.dev/dl/
#   - uv:  curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Usage:
#   ./scripts/mac_setup.sh
#   MAC_SETUP_TRAINING=1 ./scripts/mac_setup.sh   # include .venv-training for /api/train
#   MAC_SETUP_SIGNOFF=0 ./scripts/mac_setup.sh    # skip metal sign-off (faster)
#
# Env:
#   LLAMA_CPP_ROOT=../llama.cpp
#   MAC_SETUP_GO=1          — build ./zerollama via build_zerollama_mac.sh (default)
#   MAC_SETUP_BUILD=1       — build Metal llama.cpp (default)
#   MAC_SETUP_TRAINING=1     — also create .venv-training (uv) for MPS LoRA
#   MAC_SETUP_SIGNOFF=1     — run ./scripts/metal_signoff.sh after setup
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runtime_uv_venv.sh
source "${ROOT}/scripts/runtime_uv_venv.sh"
# shellcheck source=scripts/mac_cgo_env.sh
source "${ROOT}/scripts/mac_cgo_env.sh"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "warn: mac_setup targets Darwin; continuing anyway" >&2
fi

export ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-$ROOT}"

if [[ "${MAC_SETUP_GO:-1}" == "1" ]]; then
  echo "== Mac setup: CGO build env =="
  mac_cgo_env_warn_path
  mac_cgo_env
  echo "  CC=${CC}"
  echo "  python3-embed=$(pkg-config --modversion python3-embed)"
  echo ""
  echo "== Mac setup: go build zerollama =="
  "${ROOT}/scripts/build_zerollama_mac.sh"
fi

LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-${ROOT}/../llama.cpp}"
export LLAMA_CPP_ROOT
export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_CPP_ROOT}/build/bin/libllama.dylib}"

echo ""
echo "== Mac setup: uv runtime venv =="
runtime_uv_venv

if [[ "${MAC_SETUP_TRAINING:-0}" == "1" ]]; then
  echo ""
  echo "== Mac setup: uv training venv (.venv-training) =="
  # shellcheck source=scripts/training_uv_venv.sh
  source "${ROOT}/scripts/training_uv_venv.sh"
  training_uv_venv
  training_uv_verify
fi

if [[ "${MAC_SETUP_BUILD:-1}" == "1" ]]; then
  echo ""
  echo "== Mac setup: build Metal llama.cpp =="
  LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT}" "${ROOT}/scripts/build_llama_server.sh"
else
  if [[ ! -f "${LLAMA_CPP_LIB}" ]]; then
    echo "Missing ${LLAMA_CPP_LIB}; run with MAC_SETUP_BUILD=1 or build manually" >&2
    exit 1
  fi
  echo "skip llama build (MAC_SETUP_BUILD=0, ${LLAMA_CPP_LIB} present)"
fi

echo ""
echo "== Mac setup: doctor =="
if [[ -x "${ROOT}/zerollama" ]]; then
  "${ROOT}/zerollama" doctor || true
else
  mac_cgo_env
  (cd "${ROOT}" && go run . doctor) || true
fi

if [[ "${MAC_SETUP_SIGNOFF:-1}" == "1" ]]; then
  echo ""
  echo "== Mac setup: metal sign-off =="
  OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:8080}" "${ROOT}/scripts/metal_signoff.sh"
fi

echo ""
echo "PASS: mac_setup"
echo "  daily:  cd ${ROOT} && ./zerollama serve"
echo "  doctor: ./zerollama doctor"
echo "  shell:  eval \"\$(./scripts/mac_cgo_env.sh --export)\"   # if plain go build fails"
