#!/usr/bin/env bash
# Shared CUDA env for all NVIDIA lanes (5080, dual 4090, generic single-GPU).
#
#   source ./scripts/cuda_common_env.sh
#
# Lane-specific files (5080_env.sh, dual_4090_env.sh) source this first, then override
# topology YAML, serve layout, and sign-off fixtures.
#
# Doc: docs/cuda-lanes.md
set -euo pipefail

_CUDA_COMMON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")/.." && pwd)"
export ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-${_CUDA_COMMON_ROOT}}"
export CUDA_COMMON_ENV_LOADED=1

# --- Build (Go CGO + ggml CPU flags) ---
export CGO_ENABLED="${CGO_ENABLED:-1}"
export CGO_CFLAGS_ALLOW="${CGO_CFLAGS_ALLOW:--mfma|-mavx2|-O3}"

# --- CUDA libs (upstream ollama bundle layout) ---
export OLLAMA_LIBRARY_PATH="${OLLAMA_LIBRARY_PATH:-/usr/local/lib/ollama:/usr/lib/ollama:/usr/lib/ollama/cuda_v12:/usr/local/lib/ollama/cuda_v12}"
export LD_LIBRARY_PATH="${OLLAMA_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
export OLLAMA_LLM_LIBRARY="${OLLAMA_LLM_LIBRARY:-cuda_v12}"

# --- Runtime autoconfig (lane YAML still wins via ZEROLLAMA_RUNTIME_CONFIG) ---
export ZEROLLAMA_AUTO_CONFIG="${ZEROLLAMA_AUTO_CONFIG:-1}"
export ZEROLLAMA_GPU_PROFILE="${ZEROLLAMA_GPU_PROFILE:-1}"

# --- Discover llama-server when unset ---
cuda_resolve_llama_server_bin() {
  if [[ -n "${LLAMA_SERVER_BIN:-}" && -x "${LLAMA_SERVER_BIN}" ]]; then
    return 0
  fi
  local path vend
  for path in \
    "${LLAMA_CPP_BIN:-}/llama-server" \
    "${ZEROLLAMA_REPO}/../llama.cpp/build/bin/llama-server" \
    /usr/local/lib/ollama/llama-server \
    /usr/lib/ollama/llama-server; do
    if [[ -x "${path}" ]]; then
      export LLAMA_SERVER_BIN="${path}"
      export LLAMA_CPP_BIN="$(dirname "${path}")"
      export LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-$(dirname "${LLAMA_CPP_BIN}")}"
      export LD_LIBRARY_PATH="${LLAMA_CPP_BIN}:${LD_LIBRARY_PATH}"
      return 0
    fi
  done
  for vend in "${ZEROLLAMA_REPO}"/vendor/llama-cpp-*/build/bin/llama-server; do
    if [[ -x "${vend}" ]]; then
      export LLAMA_SERVER_BIN="${vend}"
      export LLAMA_CPP_BIN="$(dirname "${vend}")"
      export LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-$(dirname "${LLAMA_CPP_BIN}")}"
      export LD_LIBRARY_PATH="${LLAMA_CPP_BIN}:${LD_LIBRARY_PATH}"
      return 0
    fi
  done
  return 1
}
cuda_resolve_llama_server_bin || true

# --- Topology probe (for lane dispatch + docs) ---
cuda_visible_gpu_count() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return 0
  fi
  nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || echo 0
}

# Heuristic lane id — override with CUDA_LANE= when autodetect is wrong.
cuda_lane_detect() {
  if [[ -n "${CUDA_LANE:-}" ]]; then
    echo "${CUDA_LANE}"
    return 0
  fi
  local count vram_mb
  count="$(cuda_visible_gpu_count)"
  if [[ "${count}" -ge 2 ]]; then
    echo "dual_4090"
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    vram_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
    if [[ -n "${vram_mb}" && "${vram_mb}" -le 18000 ]]; then
      echo "rtx_5080"
      return 0
    fi
    if [[ -n "${vram_mb}" && "${vram_mb}" -ge 20000 ]]; then
      echo "single_4090"
      return 0
    fi
  fi
  echo "single_cuda"
}

cuda_print_lane_summary() {
  local lane count
  lane="$(cuda_lane_detect)"
  count="$(cuda_visible_gpu_count)"
  echo "cuda lane: ${lane} (visible_gpus=${count})"
  echo "  repo: ${ZEROLLAMA_REPO}"
  echo "  llama-server: ${LLAMA_SERVER_BIN:-unset}"
  echo "  runtime config: ${ZEROLLAMA_RUNTIME_CONFIG:-autoconfig}"
}
