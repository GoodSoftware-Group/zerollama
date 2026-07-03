#!/usr/bin/env bash
# Intel Arc A380 (6 GB GDDR6, Vulkan/Mesa ANV) session env — source once per shell.
#
#   cd ~/zerollama && source ./scripts/a380_env.sh
#   ./scripts/a380_signoff.sh
#   ./scripts/gpu_a380_session.sh
#
# WHY one file: A380 production path is Vulkan ggml, not CUDA runtime. Research lane
# asm_lab/lanes/arc-a380 documents measured pitfalls (load_ms tax, partial num_gpu cliff).
#
# Doc: docs/a380-runbook.md
# Research: ~/bmtl/asm_lab/lanes/arc-a380 (override ZA380_RESEARCH_LANE)

_ZA380_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")/.." && pwd)"
export ZA380_ROOT="${ZA380_ROOT:-$_ZA380_ROOT}"
export PYTHONPATH="${ZA380_ROOT}/runtime${PYTHONPATH:+:${PYTHONPATH}}"

# --- Research lane (asm_lab measurements, Jun 2026) ---
export ZA380_RESEARCH_LANE="${ZA380_RESEARCH_LANE:-${HOME}/bmtl/asm_lab/lanes/arc-a380}"
export ZA380_RESEARCH_SYNTHESIS="${ZA380_RESEARCH_SYNTHESIS:-${ZA380_RESEARCH_LANE}/runs/research_synthesis.json}"

# --- Host layout ---
# Private LAN bind — agents use http://192.168.255.105:11434 directly (no SSH port-forward).
export ZA380_PRIVATE_HOST="${ZA380_PRIVATE_HOST:-192.168.255.105:11434}"
export OLLAMA_HOST="${OLLAMA_HOST:-${ZA380_PRIVATE_HOST}}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-/usr/share/ollama/.ollama/models}"
_host="${OLLAMA_HOST#http://}"
_host="${_host#https://}"
export ZEROLLAMA_GO_URL="${ZEROLLAMA_GO_URL:-http://${_host}}"

# --- Vulkan production stack (WHY: Mesa ANV on A380; integer dot unstable here) ---
export OLLAMA_VULKAN="${OLLAMA_VULKAN:-1}"
export GGML_VK_DISABLE_INTEGER_DOT_PRODUCT="${GGML_VK_DISABLE_INTEGER_DOT_PRODUCT:-1}"
export OLLAMA_LLM_LIBRARY="${OLLAMA_LLM_LIBRARY:-vulkan}"
# shellcheck source=scripts/a380_llama_vendor.sh disable=SC1091
source "${ZA380_ROOT}/scripts/a380_llama_vendor.sh"
if ! a380_export_llama_vendor_env; then
  export OLLAMA_LIBRARY_PATH="${OLLAMA_LIBRARY_PATH:-/usr/lib/ollama:/usr/lib/ollama/vulkan}"
  export LD_LIBRARY_PATH="${OLLAMA_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
fi

# --- Sign-off fixture (0.5B Q8_0 — fits 6 GB; see research lane) ---
export A380_SIGNOFF_GGUF="${A380_SIGNOFF_GGUF:-/root/models/tiny-agent/Tiny-Agent-a-0.5B.Q8_0.gguf}"
export A380_SIGNOFF_MODEL="${A380_SIGNOFF_MODEL:-tiny-agent:q8}"
export A380_SIGNOFF_PROMPT="${A380_SIGNOFF_PROMPT:-hello}"
export A380_SIGNOFF_TOKENS="${A380_SIGNOFF_TOKENS:-8}"

# --- Runtime / training (Vulkan hosts: ggml inference only today) ---
export OLLAMA_TRAINING="${OLLAMA_TRAINING:-false}"
export ZEROLLAMA_RUNTIME_EMBED="${ZEROLLAMA_RUNTIME_EMBED:-off}"
export ZEROLLAMA_GPU_PROFILE_ID="${ZEROLLAMA_GPU_PROFILE_ID:-arc-a380}"
export ZEROLLAMA_RUNTIME_CONFIG="${ZEROLLAMA_RUNTIME_CONFIG:-${ZA380_ROOT}/runtime/configs/arc_a380.yaml}"

# --- Gate thresholds (from research_synthesis.json, Jun 2026 EPYC+A380) ---
# WHY total_duration not eval_tok_s: ~580ms load_ms every API call even with keep_alive.
# Zerollama on A380: higher per-request load_ms than upstream ollama (asm_lab baseline).
export A380_MIN_EVAL_TOK_S="${A380_MIN_EVAL_TOK_S:-35}"
export A380_MIN_TOTAL_TOK_S_8="${A380_MIN_TOTAL_TOK_S_8:-4}"
export A380_MAX_LOAD_MS="${A380_MAX_LOAD_MS:-1600}"

# --- Build ---
export CGO_ENABLED="${CGO_ENABLED:-1}"

a380_check_vulkan() {
  if ! command -v vulkaninfo >/dev/null 2>&1; then
    echo "a380: install vulkan-tools (vulkaninfo)" >&2
    return 1
  fi
  if ! vulkaninfo --summary 2>/dev/null | grep -qi 'Arc.*A380\|DG2'; then
    echo "a380: warn — Arc A380 not found in vulkaninfo --summary" >&2
  fi
  if [[ ! -r /dev/dri/card0 ]] && [[ ! -r /dev/dri/renderD128 ]]; then
    echo "a380: /dev/dri not readable — add user to render/video group" >&2
    return 1
  fi
}

a380_stop_serve() {
  local pid
  for pid in $(pgrep -x zerollama 2>/dev/null || true); do
    kill "$pid" 2>/dev/null || true
  done
  sleep 1
  for pid in $(pgrep -x zerollama 2>/dev/null || true); do
    kill -9 "$pid" 2>/dev/null || true
  done
  fuser -k 11434/tcp 8080/tcp 8081/tcp 2>/dev/null || true
  sleep 1
}

a380_wait_api() {
  local host="${OLLAMA_HOST#http://}"
  host="${host#https://}"
  local i
  for i in $(seq 1 30); do
    curl -sf -m 5 "http://${host}/api/tags" >/dev/null && return 0
    sleep 1
  done
  echo "a380: API timeout on ${OLLAMA_HOST}" >&2
  return 1
}

a380_start_serve() {
  a380_stop_serve
  a380_cd_repo
  # shellcheck source=scripts/serve_a380_example.sh disable=SC1091
  bash "${ZA380_ROOT}/scripts/serve_a380_example.sh" >> /tmp/zerollama-a380-serve.log 2>&1 &
  echo "a380: serve pid=$! log=/tmp/zerollama-a380-serve.log"
  a380_wait_api
}

a380_cd_repo() {
  cd "${ZA380_ROOT}"
}

a380_build_zerollama() {
  a380_cd_repo
  bash "${ZA380_ROOT}/scripts/build_zerollama_a380.sh"
}

a380_build_llama_vendor() {
  a380_cd_repo
  bash "${ZA380_ROOT}/scripts/build_zerollama_a380.sh" --llama-only
}

a380_ensure_signoff_model() {
  if curl -sf "http://${OLLAMA_HOST#http://}/api/tags" 2>/dev/null | grep -q "${A380_SIGNOFF_MODEL%%:*}"; then
    return 0
  fi
  if [[ ! -f "${A380_SIGNOFF_GGUF}" ]]; then
    echo "a380: missing sign-off GGUF ${A380_SIGNOFF_GGUF}" >&2
    return 1
  fi
  local import_dir="/var/lib/ollama/imports"
  if [[ ! -f "${import_dir}/$(basename "${A380_SIGNOFF_GGUF}")" ]]; then
    echo "a380: copy GGUF to ${import_dir} and ollama create ${A380_SIGNOFF_MODEL}" >&2
    return 1
  fi
  zerollama pull "${A380_SIGNOFF_MODEL}" || true
}

a380_print_env() {
  cat <<EOF
a380 env (source: scripts/a380_env.sh)
  ZA380_ROOT=$ZA380_ROOT
  ZA380_RESEARCH_LANE=$ZA380_RESEARCH_LANE
  OLLAMA_HOST=$OLLAMA_HOST
  OLLAMA_VULKAN=$OLLAMA_VULKAN
  GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=$GGML_VK_DISABLE_INTEGER_DOT_PRODUCT
  A380_SIGNOFF_MODEL=$A380_SIGNOFF_MODEL
  A380_SIGNOFF_GGUF=$A380_SIGNOFF_GGUF
  ZEROLLAMA_GPU_PROFILE_ID=$ZEROLLAMA_GPU_PROFILE_ID
EOF
}

export ZA380_ENV_LOADED=1

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  a380_print_env
fi
