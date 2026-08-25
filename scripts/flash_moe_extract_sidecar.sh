#!/usr/bin/env bash
# Extract Flash-MoE sidecar from a MoE GGUF (wrapper around anemll flashmoe_sidecar.py).
#
# WHY: sidecar extract is a one-time operator step before ZEROLLAMA_FLASH_MOE_SIDECAR can
# be set. This script pins repo paths and prints the env vars for serve/smoke.
#
# Usage:
#   ./scripts/flash_moe_extract_sidecar.sh --model ~/Models/qwen35.gguf --out-dir ~/Models/flash/qwen35
#   ./scripts/flash_moe_extract_sidecar.sh --model ... --out-dir ... --verify
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLASH_MOE_REPO="${FLASH_MOE_REPO:-${HOME}/Sites/inference/anemll-flash-llama.cpp}"
EXTRACT="${FLASH_MOE_REPO}/tools/flashmoe-sidecar/flashmoe_sidecar.py"

if [[ ! -f "${EXTRACT}" ]]; then
  echo "flash_moe_extract_sidecar: missing ${EXTRACT}" >&2
  echo "  git clone --branch Server-Flash-Moe --depth 1 https://github.com/Anemll/anemll-flash-llama.cpp.git ${FLASH_MOE_REPO}" >&2
  exit 1
fi

MODEL=""
OUT=""
FORCE=0
VERIFY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --out-dir) OUT="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --verify) VERIFY=1; shift ;;
    -h|--help)
      echo "usage: $0 --model PATH.gguf --out-dir DIR [--force] [--verify]"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${MODEL}" || -z "${OUT}" ]]; then
  echo "usage: $0 --model PATH.gguf --out-dir DIR [--force] [--verify]" >&2
  exit 1
fi
if [[ ! -f "${MODEL}" ]]; then
  echo "model not found: ${MODEL}" >&2
  exit 1
fi

args=(extract --model "${MODEL}" --out-dir "${OUT}")
[[ "${FORCE}" == "1" ]] && args+=(--force)

echo ">>> flashmoe_sidecar extract → ${OUT}" >&2
python3 "${EXTRACT}" "${args[@]}"

if [[ "${VERIFY}" == "1" ]]; then
  echo ">>> verify sidecar" >&2
  python3 "${EXTRACT}" verify --model "${MODEL}" --sidecar "${OUT}"
fi

cat >&2 <<EOF

Next:
  export ZEROLLAMA_FLASH_MOE=1
  export ZEROLLAMA_FLASH_MOE_SIDECAR=${OUT}
  export ZEROLLAMA_LLAMA_SERVER=1
  ${ROOT}/zerollama serve --llama-server-backend

Smoke (startup):
  RUN_E2E_FLASH_MOE_STARTUP=1 FLASH_MOE_GGUF=${MODEL} FLASH_MOE_SIDECAR=${OUT} ./scripts/flash_moe_smoke.sh
EOF
