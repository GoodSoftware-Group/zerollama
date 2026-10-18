#!/usr/bin/env bash
# One-shot macOS dev setup: host deps, build zerollama, install, uv venvs,
# Metal llama.cpp (best-effort), capability summary, optional sign-off.
#
# Prerequisites: Xcode.app (or Homebrew python@3.12) + CLI tools.
# Go / cmake / pkg-config / uv are installed user-local by mac_host_deps.sh
# when MAC_SETUP_HOST_DEPS=1 (dev_bootstrap default).
#
# See docs/mac-dev-setup.md for the full script map (post-reorg paths).
#
# Usage:
#   ./scripts/runtime/dev_bootstrap.sh          # recommended for fresh clones
#   ./scripts/runtime/mac_setup.sh
#   MAC_SETUP_SIGNOFF=1 ./scripts/runtime/mac_setup.sh   # after: zerollama pull llama3.2:3b
#   MAC_SETUP_TRAINING=1 ./scripts/runtime/mac_setup.sh # include .venv-training for /api/train
#
# Env:
#   LLAMA_CPP_ROOT=../llama.cpp
#   MAC_SETUP_HOST_DEPS=1       — install Go/cmake/pkg-config/uv user-local (dev_bootstrap default)
#   MAC_SETUP_GO=1              — build ./zerollama (default)
#   MAC_SETUP_INSTALL=1         — copy binary to $PREFIX/bin (dev_bootstrap default; PREFIX=~/.local)
#   MAC_SETUP_BUILD=1           — build Metal llama.cpp when sibling exists (default)
#   MAC_SETUP_LLAMA_CLONE=1     — clone ../llama.cpp if missing (default)
#   MAC_SETUP_LLAMA_OPTIONAL=1  — skip llama build failure (dev_bootstrap default; ggml-only OK)
#   MAC_SETUP_TRAINING=0        — create .venv-training when 1
#   MAC_SETUP_SIGNOFF=0         — metal sign-off (default off; needs pulled text GGUF)
#   MAC_SETUP_SMOKE=1           — curl :11434/api/version only if already listening (never starts serve)
#   BUILD_ANE_RESTORE=0         — skip ANE intree restore during build (dev_bootstrap default)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/runtime/mac_host_deps.sh
source "${ROOT}/scripts/runtime/mac_host_deps.sh"
# shellcheck source=scripts/runtime/runtime_uv_venv.sh
source "${ROOT}/scripts/runtime/runtime_uv_venv.sh"
# shellcheck source=scripts/runtime/mac_cgo_env.sh
source "${ROOT}/scripts/runtime/mac_cgo_env.sh"
# shellcheck source=scripts/vendor/ensure_llama_cpp_sibling.sh
source "${ROOT}/scripts/vendor/ensure_llama_cpp_sibling.sh"
# shellcheck source=scripts/vendor/ensure_llama_cgo_vendor.sh
source "${ROOT}/scripts/vendor/ensure_llama_cgo_vendor.sh"
# shellcheck source=scripts/runtime/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime/runtime_smoke_lib.sh"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "warn: mac_setup targets Darwin; continuing anyway" >&2
fi

export ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-$ROOT}"
export LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-${ROOT}/../llama.cpp}"
export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_CPP_ROOT}/build/bin/libllama.dylib}"

# --- progress ---------------------------------------------------------------
_MAC_SETUP_STEP=0
_MAC_SETUP_TOTAL=0
_mac_setup_count_steps() {
  local n=0
  [[ "${MAC_SETUP_HOST_DEPS:-0}" == "1" ]] && n=$((n + 1))
  [[ "${MAC_SETUP_GO:-1}" == "1" ]] && n=$((n + 2)) # vendor + build
  [[ "${MAC_SETUP_INSTALL:-0}" == "1" ]] && n=$((n + 1))
  n=$((n + 1)) # runtime venv always
  [[ "${MAC_SETUP_TRAINING:-0}" == "1" ]] && n=$((n + 1))
  [[ "${MAC_SETUP_BUILD:-1}" == "1" ]] && n=$((n + 1))
  n=$((n + 1)) # summary / smoke
  [[ "${MAC_SETUP_SIGNOFF:-0}" == "1" ]] && n=$((n + 1))
  _MAC_SETUP_TOTAL="${n}"
}
_mac_setup_step() {
  _MAC_SETUP_STEP=$((_MAC_SETUP_STEP + 1))
  echo ""
  echo "[${_MAC_SETUP_STEP}/${_MAC_SETUP_TOTAL}] $*"
}

mac_setup_has_signoff_model() {
  if [[ -n "${M3_LLAMA_MODEL:-}" && -f "${M3_LLAMA_MODEL}" ]]; then
    return 0
  fi
  local pick model
  pick="$(smoke_m3_pick_text_gguf 2>/dev/null || true)"
  model="$(echo "$pick" | sed -n '1p')"
  [[ -n "$model" && -f "$model" ]]
}

# Read-only: never bind or kill production :11434.
mac_setup_smoke_if_up() {
  local url="http://127.0.0.1:11434/api/version"
  local body ver
  if ! body="$(curl -fsS --connect-timeout 1 --max-time 2 "${url}" 2>/dev/null)"; then
    echo "  smoke:    :11434 not listening (start later: zerollama serve)"
    return 0
  fi
  ver="$(printf '%s' "${body}" | sed -n 's/.*"version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)"
  if [[ -n "${ver}" ]]; then
    echo "  smoke:    :11434 up (version ${ver})"
  else
    echo "  smoke:    :11434 up"
  fi
}

mac_setup_capability_summary() {
  local ggml="ok" libllama="missing" mlx="skipped" bin=""
  if [[ -x "${ROOT}/zerollama" ]] || command -v zerollama >/dev/null 2>&1; then
    ggml="ok"
  else
    ggml="missing"
  fi
  if [[ -f "${LLAMA_CPP_LIB}" ]]; then
    libllama="ok (${LLAMA_CPP_LIB})"
  elif [[ "${_llama_build_ok:-0}" == "1" ]]; then
    libllama="ok"
  else
    libllama="missing — runtime inprocess needs MAC_SETUP_LLAMA_OPTIONAL=0 rebuild"
  fi
  if [[ -f "${ROOT}/build/metal-v3/lib/ollama/mlx_metal_v3/libmlxc.dylib" ]] \
    || [[ -f "${ROOT}/build/metal-v4/lib/ollama/mlx_metal_v4/libmlxc.dylib" ]]; then
    mlx="ok"
  elif [[ -d "${ROOT}/../mlx/.git" ]]; then
    mlx="sibling present — BUILD_MLX=1 ./scripts/build/build_zerollama_mac.sh"
  else
    mlx="skipped (no ../mlx)"
  fi
  if command -v zerollama >/dev/null 2>&1; then
    bin="$(command -v zerollama)"
  elif [[ -x "${ROOT}/zerollama" ]]; then
    bin="${ROOT}/zerollama"
  else
    bin="(not found)"
  fi

  echo ""
  echo "PASS: mac_setup (tier 0)"
  echo "  ggml:     ${ggml}  (${bin})"
  echo "  libllama: ${libllama}"
  echo "  mlx:      ${mlx}"
  echo "  daily:    zerollama serve     # Go :11434, sidecar :8081"
  echo "  pull:     zerollama pull llama3.2:3b"
  if [[ "${MAC_SETUP_SIGNOFF:-0}" != "1" ]]; then
    echo "  sign-off: MAC_SETUP_SIGNOFF=1 MAC_SETUP_GO=0 MAC_SETUP_BUILD=0 ./scripts/runtime/mac_setup.sh"
  fi
  if [[ "${MAC_SETUP_SMOKE:-1}" == "1" ]]; then
    mac_setup_smoke_if_up
  fi
}

_mac_setup_count_steps
echo "Mac setup: ${_MAC_SETUP_TOTAL} steps (ggml-first; optional stacks soft-fail)"

_llama_build_ok=0

if [[ "${MAC_SETUP_HOST_DEPS:-0}" == "1" ]]; then
  _mac_setup_step "Host deps (Go / cmake / pkg-config / uv)"
  mac_host_deps_ensure
  mac_host_deps_path_prepend
fi

if [[ "${MAC_SETUP_GO:-1}" == "1" ]]; then
  _mac_setup_step "CGO vendor headers"
  ensure_llama_cgo_vendor
  _mac_setup_step "Build zerollama (Metal ggml)"
  mac_cgo_env_warn_path
  mac_cgo_env
  echo "  CC=${CC}"
  echo "  python3-embed=$(pkg-config --modversion python3-embed)"
  # Avoid dirtying llama/llama.cpp/common during tier-0 bootstrap unless asked.
  export BUILD_ANE_RESTORE="${BUILD_ANE_RESTORE:-0}"
  "${ROOT}/scripts/build/build_zerollama_mac.sh"
fi

if [[ "${MAC_SETUP_INSTALL:-0}" == "1" ]]; then
  _mac_setup_step "Install binary → ${PREFIX:-${HOME}/.local}/bin"
  PREFIX="${PREFIX:-${HOME}/.local}" "${ROOT}/scripts/install_mac.sh"
fi

_mac_setup_step "Runtime uv venv"
runtime_uv_venv

if [[ "${MAC_SETUP_TRAINING:-0}" == "1" ]]; then
  _mac_setup_step "Training uv venv (.venv-training)"
  # shellcheck source=scripts/training/training_uv_venv.sh
  source "${ROOT}/scripts/training/training_uv_venv.sh"
  training_uv_venv
  training_uv_verify
fi

if [[ "${MAC_SETUP_BUILD:-1}" == "1" ]]; then
  _mac_setup_step "Sibling Metal llama.cpp (best-effort)"
  if ensure_llama_cpp_sibling; then
    if LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT}" "${ROOT}/scripts/build/build_llama_server.sh"; then
      _llama_build_ok=1
    elif [[ "${MAC_SETUP_LLAMA_OPTIONAL:-0}" == "1" ]]; then
      echo "warn: llama.cpp build failed; continuing (MAC_SETUP_LLAMA_OPTIONAL=1)" >&2
      echo "  ggml-only: zerollama serve still works; runtime inprocess needs libllama" >&2
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

if [[ "${MAC_SETUP_SIGNOFF:-0}" == "1" ]]; then
  _mac_setup_step "Metal sign-off"
  if mac_setup_has_signoff_model; then
    OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:8080}" "${ROOT}/scripts/gpu/metal_signoff.sh"
  else
    echo "skip metal sign-off (no local text GGUF)" >&2
    echo "  pull a model, then re-run:" >&2
    echo "    zerollama serve && zerollama pull llama3.2:3b" >&2
    echo "    MAC_SETUP_SIGNOFF=1 MAC_SETUP_GO=0 MAC_SETUP_BUILD=0 ./scripts/runtime/mac_setup.sh" >&2
  fi
fi

_mac_setup_step "Capability summary"
mac_setup_capability_summary
