#!/usr/bin/env bash
# Wrapper for zerollama external-image (stable-diffusion.cpp).
# Invoked via OLLAMA_EXTERNAL_IMAGE_BIN; reads OLLAMA_IMAGE_* and OLLAMA_SD_* env vars.
set -euo pipefail

SD_CLI="${OLLAMA_SD_CLI:-}"
SD_MODEL="${OLLAMA_SD_MODEL:-}"
OUTPUT="${OLLAMA_IMAGE_OUTPUT:?OLLAMA_IMAGE_OUTPUT required}"
PROMPT="${OLLAMA_IMAGE_PROMPT:-}"
W="${OLLAMA_IMAGE_WIDTH:-${OLLAMA_SD_DEFAULT_WIDTH:-512}}"
H="${OLLAMA_IMAGE_HEIGHT:-${OLLAMA_SD_DEFAULT_HEIGHT:-512}}"
SEED="${OLLAMA_IMAGE_SEED:--1}"
STEPS="${OLLAMA_SD_STEPS:-20}"
CFG="${OLLAMA_SD_CFG_SCALE:-7.0}"
SAMPLER="${OLLAMA_SD_SAMPLER:-euler_a}"
DIFFUSION_FA="${OLLAMA_SD_DIFFUSION_FA:-1}"
VAE_ON_CPU="${OLLAMA_SD_VAE_ON_CPU:-1}"
VAE_TILING="${OLLAMA_SD_VAE_TILING:-0}"

if [[ -z "$SD_CLI" || ! -x "$SD_CLI" ]]; then
  echo "OLLAMA_SD_CLI must point to sd-cli (set backend_paths.sd_cli or install via scripts/install_stable_diffusion.sh)" >&2
  exit 1
fi
if [[ -z "$SD_MODEL" || ! -f "$SD_MODEL" ]]; then
  echo "OLLAMA_SD_MODEL must point to a GGUF weights file (backend_paths.sd_model)" >&2
  exit 1
fi
if [[ -z "$PROMPT" ]]; then
  echo "OLLAMA_IMAGE_PROMPT is empty" >&2
  exit 1
fi

export LD_LIBRARY_PATH="$(dirname "$SD_CLI")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

args=(
  -m "$SD_MODEL"
  -p "$PROMPT"
  -o "$OUTPUT"
  -W "$W"
  -H "$H"
  --sampling-method "$SAMPLER"
  --steps "$STEPS"
  --cfg-scale "$CFG"
)

if [[ "$SEED" != "-1" && -n "$SEED" ]]; then
  args+=(-s "$SEED")
fi
if [[ "$DIFFUSION_FA" == "1" || "$DIFFUSION_FA" == "true" ]]; then
  args+=(--diffusion-fa)
fi
if [[ "$VAE_ON_CPU" == "1" || "$VAE_ON_CPU" == "true" ]]; then
  args+=(--vae-on-cpu)
fi
if [[ "$VAE_TILING" == "1" || "$VAE_TILING" == "true" ]]; then
  args+=(--vae-tiling)
fi

exec "$SD_CLI" "${args[@]}"
