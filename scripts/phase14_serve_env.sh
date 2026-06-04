#!/usr/bin/env bash
# Source before `zerollama serve` for Phase 14 in-process / wheel smoke.
#
# Why this file exists: the #1 smoke failure is exporting ZEROLLAMA_RUNTIME_URL in the
# shell while expecting embed on :8081. Go then runs "external sidecar" mode and never
# starts embedded Python (see envconfig.RuntimeEmbedEnabled).
#
# Usage:
#   source ./scripts/phase14_serve_env.sh
#   export LLAMA_MODEL=/path/to/model.q8_0.gguf
#   export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess   # or llama-cpp-python
#   # or uncomment llama_backend: inprocess in runtime/configs/single_gpu.yaml (omit env)
#   ./zerollama serve
#
# Why unset ZEROLLAMA_RUNTIME_URL: if set, Go expects an external :8081 sidecar and
# does not embed the Python runtime (see envconfig.RuntimeEmbedEnabled).
unset ZEROLLAMA_RUNTIME_URL
export ZEROLLAMA_RUNTIME_EMBED="${ZEROLLAMA_RUNTIME_EMBED:-on}"
# Default-on for eligible text models (Phase 12); does NOT proxy every pulled tag.
# Why document OLLAMA_RUNTIME_ALL separately: ZEROLLAMA_RUNTIME=1 enables default-on
# routing for eligible models only; ALL forces every local name to the runtime.
#   export OLLAMA_RUNTIME_ALL=1
export ZEROLLAMA_RUNTIME="${ZEROLLAMA_RUNTIME:-1}"
export OLLAMA_NO_CLOUD="${OLLAMA_NO_CLOUD:-true}"
export ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Warn when embed port is already taken (stale zerollama serve or sidecar).
_embed_port="${ZEROLLAMA_RUNTIME_EMBED_PORT:-8081}"
if command -v ss >/dev/null 2>&1 && ss -tln 2>/dev/null | grep -q ":${_embed_port} "; then
  echo "WARN: 127.0.0.1:${_embed_port} already in use — stop stale 'zerollama serve' or zerollama-runtime sidecar before embed" >&2
fi

# Optional lib path when using ctypes inprocess backend:
# export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-$HOME/llama.cpp/build/bin/libllama.so}"
