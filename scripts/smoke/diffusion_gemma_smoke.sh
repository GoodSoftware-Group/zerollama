#!/usr/bin/env bash
# Lab smoke for DiffusionGemma CLI (DG0/DG1). Never binds production ports.
#
# WHY -ngl 25 default: Q4_K_M full offload OOMs on 5080 16GB (Finding 1).
# WHY free competing GPU jobs first: peak ~15.3 GiB leaves no room for CLM emb.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${DIFFUSION_GGUF:-/root/models/diffusiongemma/diffusiongemma-26B-A4B-it-Q4_K_M.gguf}"
NGL="${DIFFUSION_NGL:-25}"
N="${DIFFUSION_N:-64}"
CTX="${DIFFUSION_CTX:-2048}"

export LD_LIBRARY_PATH="/root/nvidia-host:${LD_LIBRARY_PATH:-}"

if [[ -z "${ZEROLLAMA_DIFFUSION_BIN:-}" ]]; then
  CAND="${ROOT}/vendor/llama-cpp-diffusion-dd0cf0445/build/bin/llama-diffusion-gemma-cli"
  if [[ -x "${CAND}" ]]; then
    ZEROLLAMA_DIFFUSION_BIN="${CAND}"
  elif [[ -x /root/llama.cpp-diffusion-spike/build/bin/llama-diffusion-gemma-cli ]]; then
    ZEROLLAMA_DIFFUSION_BIN=/root/llama.cpp-diffusion-spike/build/bin/llama-diffusion-gemma-cli
  else
    echo "error: set ZEROLLAMA_DIFFUSION_BIN or run ./scripts/build/build_llama_diffusion.sh" >&2
    exit 1
  fi
fi

if [[ ! -f "${MODEL}" ]]; then
  echo "error: missing ${MODEL}" >&2
  exit 1
fi

echo ">>> smoke ${ZEROLLAMA_DIFFUSION_BIN} -ngl ${NGL} -n ${N} -c ${CTX}" >&2
exec "${ZEROLLAMA_DIFFUSION_BIN}" \
  -m "${MODEL}" \
  -ngl "${NGL}" -n "${N}" -c "${CTX}" \
  -p "${DIFFUSION_PROMPT:-Say hello in one short sentence.}"
