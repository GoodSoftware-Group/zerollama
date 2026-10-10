#!/usr/bin/env bash
# GF1 — native glm-flash-lite NVMe serve (astra). No Docker. Lab port default :30000.
# Checkout: GLM_FLASH_LITE_ROOT (default /mnt/ssd2/src/glm-flash-lite).
# Logs: GLM53_LOG (default /mnt/ssd2/glm53-serve.log) when run via nohup from smoke docs.
set -euo pipefail
ROOT="${GLM_FLASH_LITE_ROOT:-/mnt/ssd2/src/glm-flash-lite}"
VENV="${VENV:-/mnt/ssd2/venv-glm53}"
MODEL="${GLM53_MODEL_DIR:-/mnt/ssd2/models/glm53}"
STORE="${GLM53_NV_STORE:-/mnt/nvme/glm53/glm53_flash_exl3_3.05bpw_experts.bin}"
PORT="${PORT:-30000}"

if [[ ! -d "$ROOT/glm53" ]]; then
  echo "missing glm-flash-lite checkout at $ROOT (set GLM_FLASH_LITE_ROOT)" >&2
  exit 1
fi

export PATH="$VENV/bin:/usr/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr}"
export TORCH_DONT_CHECK_COMPILER_ABI=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export GLM53_MODE=nvme
export GLM53_NV=2
export GLM53_NV_CPU="${GLM53_NV_CPU:-1}"
# Exclusive RAM tier size (GiB). Upstream Docker uses --memory 55g.
export GLM53_NV_RAM_GB="${GLM53_NV_RAM_GB:-55}"
export GLM53_NV_STORE="$STORE"
export GLM53_NV_VRING="${GLM53_NV_VRING:-24}"
export GLM53_NV_PREFETCH="${GLM53_NV_PREFETCH:-1}"
export GLM53_NV_CPU_KERN="${GLM53_NV_CPU_KERN:-1}"
export GLM53_LA="${GLM53_LA:-1}"
export GLM53_NV_PUBFAST="${GLM53_NV_PUBFAST:-1}"
export GLM53_NV_PFSIDE="${GLM53_NV_PFSIDE:-1}"
export GLM53_LA_BTTRIM="${GLM53_LA_BTTRIM:-1}"
export GLM53_MAX_RQ_TOKENS="${GLM53_MAX_RQ_TOKENS:-4096}"
export GLM53_K_HCFUSE="${GLM53_K_HCFUSE:-1}"
export GLM53_K_FTSPLIT="${GLM53_K_FTSPLIT:-0}"
export GLM53_K_OVL="${GLM53_K_OVL:-0}"
export GLM53_ZC_VRAM="${GLM53_ZC_VRAM-}"
export GLM53_EC="${GLM53_EC:-1}"
export GLM53_EC_RESERVE_GB="${GLM53_EC_RESERVE_GB:-1.5}"
export GLM53_EC_STAGE_GB="${GLM53_EC_STAGE_GB:-2.6}"
export GLM53_EC_ELASTIC_GB="${GLM53_EC_ELASTIC_GB:-10}"
export GLM53_EC_STAGE_MIN="${GLM53_EC_STAGE_MIN:-512}"
export GLM53_EC_MAX_SLOTS="${GLM53_EC_MAX_SLOTS:-1376}"
export GLM53_EC_WARM="${GLM53_EC_WARM:-$ROOT/data/stats_own_dec.json}"
export GLM53_ZC_STATS="${GLM53_ZC_STATS:-$ROOT/data/stats_own_dec.json}"
export GLM53_TRITON_PIN="${GLM53_TRITON_PIN:-$ROOT/data/triton_pin_exact.json}"
export EXLLAMAV3_TUNE_CACHE="${EXLLAMAV3_TUNE_CACHE:-$ROOT/data/coop_autotune_1gpu.bin}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/triton_cache}"
export GLM53_CPU_TIER=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/glm53${PYTHONPATH:+:$PYTHONPATH}"

ulimit -l unlimited 2>/dev/null || true
test -r "$STORE" && test -r "${STORE%.bin}.json"
test -d "$MODEL"

cd "$ROOT"
exec "$VENV/bin/python" glm53/serve.py \
  -m "$MODEL" \
  -cs 131072 \
  --max-batch-size 8 \
  -chunk_size 8192 \
  -ambs 4 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-name glm-5.3-flash \
  "$@"
