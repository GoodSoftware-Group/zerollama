#!/usr/bin/env bash
# Launch SGLang with Dual Chunk Attention for Qwen 1M / long-ctx HF models.
#
# WHY SGLang (lab oracle only): dense DualChunk reference for n≥1 logit compare.
# Product path is native ggml CUDA DCA (patches 0094–0098) on stamped GGUF.
# Stock vLLM removed the V0 DCA backend; SGLang still has dual_chunk_flash_attn.
# Zerollama can still proxy /v1/chat/completions when modality_backends.inference=sglang.
#
# Doc: docs/dca-dual-chunk-attention.md
set -euo pipefail

SGLANG_ROOT="${SGLANG_ROOT:-/var/lib/vz/private/1564/root/sglang}"
MODEL_PATH="${DCA_MODEL_PATH:-${1:-}}"
HOST="${DCA_HOST:-127.0.0.1}"
PORT="${DCA_PORT:-30000}"
TP="${DCA_TP:-1}"
CTX="${DCA_CTX:-131072}"
MEM_FRAC="${DCA_MEM_FRAC:-0.85}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "usage: DCA_MODEL_PATH=/path/to/Qwen2.5-*-Instruct-1M $0" >&2
  echo "  or:  $0 /path/to/HF-model" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "sglang_dca: missing ${MODEL_PATH}/config.json" >&2
  exit 1
fi

if ! python3 -c 'import sglang' 2>/dev/null; then
  if [[ -d "${SGLANG_ROOT}/python" ]]; then
    export PYTHONPATH="${SGLANG_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
  fi
fi
if ! python3 -c 'import sglang' 2>/dev/null; then
  echo "sglang_dca: install sglang (pip) or set SGLANG_ROOT=${SGLANG_ROOT}" >&2
  exit 1
fi

# Auto-selects dual_chunk_flash_attn when HF config has dual_chunk_attention_config.
# Explicit flag keeps the intent visible in process lists.
echo "sglang_dca: model=${MODEL_PATH} listen=${HOST}:${PORT} ctx=${CTX}" >&2
echo "sglang_dca: point zerollama at this with:" >&2
echo "  export OLLAMA_SGLANG_URL=http://${HOST}:${PORT}" >&2
echo "  # Modelfile / config.json: modality_backends.inference=sglang" >&2

# shellcheck disable=SC2086
exec python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp "${TP}" \
  --context-length "${CTX}" \
  --mem-fraction-static "${MEM_FRAC}" \
  --attention-backend dual_chunk_flash_attn \
  --disable-radix-cache \
  ${DCA_EXTRA_ARGS:-}
