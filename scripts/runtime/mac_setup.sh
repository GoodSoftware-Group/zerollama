#!/usr/bin/env bash
# One-shot macOS dev setup: build zerollama, uv venvs, Metal llama.cpp, doctor, optional sign-off.
#
# Prerequisites (once per machine):
#   - Xcode Command Line Tools:  xcode-select --install
#   - Go 1.22+ from https://go.dev/dl/
#   - uv:  curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Usage:
#   ./scripts/runtime/dev_bootstrap.sh          # recommended for fresh clones (sign-off off)
#   ./scripts/runtime/mac_setup.sh
#   MAC_SETUP_SIGNOFF=1 ./scripts/runtime/mac_setup.sh   # after: zerollama pull llama3.2:3b
#   MAC_SETUP_TRAINING=1 ./scripts/runtime/mac_setup.sh # include .venv-training for /api/train
#
# Env:
#   LLAMA_CPP_ROOT=../llama.cpp
#   MAC_SETUP_GO=1              — build ./zerollama (default)
#   MAC_SETUP_BUILD=1           — build Metal llama.cpp when sibling exists (default)
#   MAC_SETUP_LLAMA_CLONE=1     — clone ../llama.cpp if missing (default)
#   MAC_SETUP_LLAMA_OPTIONAL=0  — when 1, skip llama build failure (ggml-only dev)
#   MAC_SETUP_TRAINING=0        — create .venv-training when 1
#   MAC_SETUP_SIGNOFF=0         — metal sign-off (default off; needs pulled text GGUF)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/runtime/runtime_uv_venv.sh
source "${ROOT}/scripts/runtime/runtime_uv_venv.sh"
# shellcheck source=scripts/runtime/mac_cgo_env.sh
source "${ROOT}/scripts/runtime/mac_cgo_env.sh"
# shellcheck source=scripts/vendor/ensure_llama_cpp_sibling.sh
source "${ROOT}/scripts/vendor/ensure_llama_cpp_sibling.sh"
# shellcheck source=scripts/runtime/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime/runtime_smoke_lib.sh"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "warn: mac_setup targets Darwin; continuing anyway" >&2
fi

export ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-$ROOT}"

# Sign-off smokes need a local text GGUF (M3_LLAMA_MODEL or smallest ~/.ollama blob).
# Why checked before metal_signoff: default mac_setup used to fail on fresh clones with no pulls.
mac_setup_has_signoff_model() {
  if [[ -n "${M3_LLAMA_MODEL:-}" && -f "${M3_LLAMA_MODEL}" ]]; then
    return 0
  fi
  local pick model
  pick="$(smoke_m3_pick_text_gguf 2>/dev/null || true)"
  model="$(echo "$pick" | sed -n '1p')"
  [[ -n "$model" && -f "$model" ]]
}

if [[ "${MAC_SETUP_GO:-1}" == "1" ]]; then
  echo "== Mac setup: CGO build env =="
  mac_cgo_env_warn_path
  mac_cgo_env
  echo "  CC=${CC}"
  echo "  python3-embed=$(pkg-config --modversion python3-embed)"
  echo ""
  echo "== Mac setup: go build zerollama =="
  "${ROOT}/scripts/build/build_zerollama_mac.sh"
fi

export LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-${ROOT}/../llama.cpp}"
export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_CPP_ROOT}/build/bin/libllama.dylib}"

echo ""
echo "== Mac setup: uv runtime venv =="
runtime_uv_venv

if [[ "${MAC_SETUP_TRAINING:-0}" == "1" ]]; then
  echo ""
  echo "== Mac setup: uv training venv (.venv-training) =="
  # shellcheck source=scripts/training/training_uv_venv.sh
  source "${ROOT}/scripts/training/training_uv_venv.sh"
  training_uv_venv
  training_uv_verify
fi

_llama_build_ok=0
if [[ "${MAC_SETUP_BUILD:-1}" == "1" ]]; then
  echo ""
  echo "== Mac setup: llama.cpp sibling =="
  if ensure_llama_cpp_sibling; then
    echo ""
    echo "== Mac setup: build Metal llama.cpp =="
    if LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT}" "${ROOT}/scripts/build/build_llama_server.sh"; then
      _llama_build_ok=1
    elif [[ "${MAC_SETUP_LLAMA_OPTIONAL:-0}" == "1" ]]; then
      echo "warn: llama.cpp build failed; continuing (MAC_SETUP_LLAMA_OPTIONAL=1)" >&2
      echo "  ggml-only: ./zerollama serve still works; runtime inprocess needs libllama" >&2
    else
      echo "error: llama.cpp build failed" >&2
      echo "  fix build, or: MAC_SETUP_LLAMA_OPTIONAL=1 MAC_SETUP_BUILD=0 ./scripts/runtime/mac_setup.sh" >&2
      exit 1
    fi
  elif [[ "${MAC_SETUP_LLAMA_OPTIONAL:-0}" == "1" ]]; then
    echo "warn: no llama.cpp sibling; skipping build (MAC_SETUP_LLAMA_OPTIONAL=1)" >&2
  else
    exit 1
  fi
else
  if [[ -f "${LLAMA_CPP_LIB}" ]]; then
    _llama_build_ok=1
    echo "skip llama build (MAC_SETUP_BUILD=0, ${LLAMA_CPP_LIB} present)"
  else
    echo "warn: MAC_SETUP_BUILD=0 and ${LLAMA_CPP_LIB} missing — runtime inprocess may fail" >&2
  fi
fi

echo ""
echo "== Mac setup: doctor =="
if [[ -x "${ROOT}/zerollama" ]]; then
  "${ROOT}/zerollama" doctor || true
else
  mac_cgo_env
  (cd "${ROOT}" && GOFLAGS=-mod=mod go run . doctor) || true
fi

if [[ "${MAC_SETUP_SIGNOFF:-0}" == "1" ]]; then
  echo ""
  if mac_setup_has_signoff_model; then
    echo "== Mac setup: metal sign-off (CI ports :8080 + :8081) =="
    OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:8080}" "${ROOT}/scripts/gpu/metal_signoff.sh"
  else
    echo "== Mac setup: skip metal sign-off (no local text GGUF) ==" >&2
    echo "  pull a model, then re-run:" >&2
    echo "    ./zerollama serve && ./zerollama pull llama3.2:3b" >&2
    echo "    MAC_SETUP_SIGNOFF=1 MAC_SETUP_GO=0 MAC_SETUP_BUILD=0 ./scripts/runtime/mac_setup.sh" >&2
    echo "  or: M3_LLAMA_MODEL=/path/to/model.gguf MAC_SETUP_SIGNOFF=1 ..." >&2
  fi
fi

echo ""
echo "PASS: mac_setup (tier 0 — build + serve ready)"
echo "  daily:    cd ${ROOT} && ./zerollama serve     # Go :11434, sidecar :8081"
echo "  doctor:   ./zerollama doctor"
echo "  pull:     ./zerollama pull llama3.2:3b"
if [[ "${MAC_SETUP_SIGNOFF:-0}" != "1" ]]; then
  echo "  sign-off: MAC_SETUP_SIGNOFF=1 MAC_SETUP_GO=0 MAC_SETUP_BUILD=0 ./scripts/runtime/mac_setup.sh"
fi
if [[ "${_llama_build_ok}" != "1" ]]; then
  echo "  llama:    ensure ${LLAMA_CPP_ROOT} then MAC_SETUP_GO=0 ./scripts/runtime/mac_setup.sh"
fi
echo "  shell:    eval \"\$(./scripts/runtime/mac_cgo_env.sh --export)\"   # if plain go build fails"
