#!/usr/bin/env bash
# Example zerollama serve for Intel Arc A380 (Vulkan / Mesa ANV).
#
# WHY separate from serve_gpu_example.sh: CUDA env (OLLAMA_LLM_LIBRARY=cuda_v12,
# PYTHONPATH runtime/training) is wrong on Arc. Production path is ggml Vulkan only.
#
# In-repo: bash scripts/serve/serve_a380_example.sh
# Doc: docs/a380-runbook.md
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${_SCRIPT_DIR}/../runtime/runtime/server/app.py" ]]; then
  _ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
elif [[ -n "${ZEROLLAMA_REPO:-}" && -f "${ZEROLLAMA_REPO}/runtime/runtime/server/app.py" ]]; then
  _ROOT="${ZEROLLAMA_REPO}"
elif [[ -f "${HOME}/zerollama/runtime/runtime/server/app.py" ]]; then
  _ROOT="${HOME}/zerollama"
else
  echo "serve_a380_example: cannot find zerollama repo; set ZEROLLAMA_REPO" >&2
  exit 1
fi
export ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-${_ROOT}}"
# shellcheck source=scripts/runtime/sched_watchdog_env.sh
source "${_ROOT}/scripts/runtime/sched_watchdog_env.sh"
# shellcheck source=scripts/gpu/a380_llama_vendor.sh
source "${_ROOT}/scripts/gpu/a380_llama_vendor.sh"

_host="${OLLAMA_HOST:-192.168.255.105:11434}"
_host="${_host#http://}"
_host="${_host#https://}"
export OLLAMA_HOST="${OLLAMA_HOST:-${_host}}"
export ZEROLLAMA_GO_URL="${ZEROLLAMA_GO_URL:-http://${OLLAMA_HOST#http://}}"

# Vulkan stack (Mesa ANV on A380; integer dot unstable here)
export OLLAMA_VULKAN="${OLLAMA_VULKAN:-1}"
export GGML_VK_DISABLE_INTEGER_DOT_PRODUCT="${GGML_VK_DISABLE_INTEGER_DOT_PRODUCT:-1}"
export OLLAMA_LLM_LIBRARY="${OLLAMA_LLM_LIBRARY:-vulkan}"
if a380_export_llama_vendor_env; then
  : # LLAMA_SERVER_BIN + LD_LIBRARY_PATH set
else
  export OLLAMA_LIBRARY_PATH="${OLLAMA_LIBRARY_PATH:-/usr/lib/ollama:/usr/lib/ollama/vulkan}"
  export LD_LIBRARY_PATH="${OLLAMA_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
  echo "serve_a380: warn — using stock /usr/lib/ollama (run ./scripts/build/build_zerollama_a380.sh)" >&2
fi

# Inference-only: no CUDA training embed on Arc
export OLLAMA_TRAINING="${OLLAMA_TRAINING:-false}"
export ZEROLLAMA_RUNTIME_EMBED="${ZEROLLAMA_RUNTIME_EMBED:-off}"

# L1 profile when runtime is enabled later (force id — no nvidia-smi on Intel)
export ZEROLLAMA_GPU_PROFILE_ID="${ZEROLLAMA_GPU_PROFILE_ID:-arc-a380}"
export ZEROLLAMA_RUNTIME_CONFIG="${ZEROLLAMA_RUNTIME_CONFIG:-${_ROOT}/runtime/configs/arc_a380.yaml}"

# Conservative defaults for 6 GB VRAM
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-4096}"

# Stable Diffusion 1.5 via stable-diffusion.cpp (Vulkan); see docs/sd-vulkan-a380.md
export OLLAMA_EXTERNAL_IMAGE_BIN="${OLLAMA_EXTERNAL_IMAGE_BIN:-${_ROOT}/scripts/image/sd_external_image.sh}"

exec "${ZEROLLAMA_REPO}/zerollama" serve
