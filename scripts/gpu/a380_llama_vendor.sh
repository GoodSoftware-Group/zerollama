#!/usr/bin/env bash
# Resolve zerollama vendor llama-server paths for Intel Arc A380 (Vulkan).
#
# WHY: go build alone leaves inference on upstream /usr/lib/ollama (no fork KV types).
# Full deploy installs patched eliza vendor build to /usr/lib/ollama-zerollama.
#
# Source from serve_a380_example.sh, a380_env.sh, or install scripts.
#   source ./scripts/gpu/a380_llama_vendor.sh
#   a380_export_llama_vendor_env
set -euo pipefail

a380_llama_vendor_root() {
  if [[ -n "${ZEROLLAMA_REPO:-}" && -f "${ZEROLLAMA_REPO}/Makefile.sync" ]]; then
    echo "${ZEROLLAMA_REPO}"
    return 0
  fi
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  echo "$(cd "${script_dir}/.." && pwd)"
}

a380_llama_vendor_pin() {
  grep '^FETCH_HEAD=' "$(a380_llama_vendor_root)/Makefile.sync" | cut -d= -f2
}

a380_llama_vendor_build_dir() {
  echo "$(a380_llama_vendor_root)/vendor/llama-cpp-$(a380_llama_vendor_pin)/build/bin"
}

a380_llama_install_dir() {
  echo "${A380_LLAMA_INSTALL_DIR:-/usr/lib/ollama-zerollama}"
}

# Sets LLAMA_SERVER_BIN, LLAMA_CPP_LIB, LD_LIBRARY_PATH, OLLAMA_LIBRARY_PATH when found.
a380_export_llama_vendor_env() {
  local install build bin lib vulkan_fallback
  install="$(a380_llama_install_dir)"
  build="$(a380_llama_vendor_build_dir)"
  vulkan_fallback="/usr/lib/ollama/vulkan"

  if [[ -x "${install}/llama-server" ]]; then
    bin="${install}/llama-server"
    lib="${install}/libllama.so"
  elif [[ -x "${build}/llama-server" ]]; then
    bin="${build}/llama-server"
    lib="${build}/libllama.so"
  else
    return 1
  fi

  export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${bin}}"
  if [[ -f "${lib}" ]]; then
    export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${lib}}"
  fi
  local libdir
  libdir="$(dirname "${bin}")"
  export LD_LIBRARY_PATH="${libdir}:${vulkan_fallback}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export OLLAMA_LIBRARY_PATH="${libdir}:${vulkan_fallback}${OLLAMA_LIBRARY_PATH:+:${OLLAMA_LIBRARY_PATH}}"
  return 0
}

a380_verify_fork_llama_server() {
  local bin="${1:-${LLAMA_SERVER_BIN:-}}"
  if [[ -z "${bin}" || ! -x "${bin}" ]]; then
    echo "a380: fork llama-server missing (run ./scripts/build/build_zerollama_a380.sh --llama)" >&2
    return 1
  fi
  local help libdir
  libdir="$(dirname "${bin}")"
  help="$(LD_LIBRARY_PATH="${libdir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" "${bin}" --help 2>&1 || true)"
  if echo "${help}" | grep -qE 'qjl1_256|tbq3_0|tbq4_0'; then
    echo "a380: fork llama-server OK (${bin})"
    return 0
  fi
  echo "a380: ${bin} lacks fork KV types in --help (stock upstream?)" >&2
  return 1
}
