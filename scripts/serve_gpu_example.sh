#!/usr/bin/env bash
# Example zerollama serve for a GPU host (e.g. cudallama with RTX 5080).
# Copy to ~/bin/serve.sh and adjust paths. Why each block is documented inline.
set -euo pipefail

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/sched_watchdog_env.sh
source "${_ROOT}/scripts/sched_watchdog_env.sh"

# Repo root: training.py + optional runtime/ checkout.
# Auto-detect now checks $HOME/zerollama; set explicitly if your layout differs.
export OLLAMA_TRAINING_PYTHONPATH="${OLLAMA_TRAINING_PYTHONPATH:-$HOME/zerollama}"
export ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-$OLLAMA_TRAINING_PYTHONPATH}"

export OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:8080}"
# Runtime → Go /internal/* (cross-queue-seq, render-chat) must use loopback, not the bind address.
export ZEROLLAMA_GO_URL="${ZEROLLAMA_GO_URL:-http://127.0.0.1:8080}"
export OLLAMA_LLM_LIBRARY="${OLLAMA_LLM_LIBRARY:-cuda_v12}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_N_CTX="${OLLAMA_N_CTX:-12288}"

# Runtime YAML + VRAM policy (single-GPU 16GB — tune after measurement).
export ZEROLLAMA_RUNTIME_CONFIG="${ZEROLLAMA_RUNTIME_CONFIG:-$ZEROLLAMA_REPO/runtime/configs/single_gpu.yaml}"
export ZEROLLAMA_RUNTIME_VRAM_MIN_FREE="${ZEROLLAMA_RUNTIME_VRAM_MIN_FREE:-1GiB}"
export ZEROLLAMA_RUNTIME_TRAINING_VRAM_RESERVE="${ZEROLLAMA_RUNTIME_TRAINING_VRAM_RESERVE:-2GiB}"
# Optional: lower num_ctx to /health suggestion when GPU checks on (default off).
# export ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX=auto
export ZEROLLAMA_RUNTIME_VRAM_PROBE_CALIBRATE="${ZEROLLAMA_RUNTIME_VRAM_PROBE_CALIBRATE:-auto}"
export ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE="${ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE:-auto}"

# Training: Go listens on :9500 and /api/train/* (embedded CPython, not a python sidecar).
export OLLAMA_TRAINING="${OLLAMA_TRAINING:-true}"
export OLLAMA_TRAINING_TCP="${OLLAMA_TRAINING_TCP:-:9500}"

# Stability on new GPU architectures (optional).
export GGML_CUDA_USE_GRAPHS="${GGML_CUDA_USE_GRAPHS:-0}"
export GGML_CUDA_FORCE_CUBLAS="${GGML_CUDA_FORCE_CUBLAS:-1}"

# CUDA libs for ggml cuda_v12 — adjust for your install.
# /usr/hostlibs may expose libcudnn older than PyTorch in the Wan venv; keep hostlibs for
# ggml. Wan run_script subprocesses sanitize LD (prepend venv torch/lib, drop hostlibs).
# See docs/wan-t2v.md troubleshooting: "cuDNN version incompatibility".
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/hostlibs:/usr/local/cuda-12.6/targets/x86_64-linux/lib}"
# ggml CUDA backend (chat/completion runners) — without /usr/lib/ollama + cuda_v12, ggml falls back to CPU-only.
export OLLAMA_LIBRARY_PATH="${OLLAMA_LIBRARY_PATH:-/usr/lib/ollama:/usr/lib/ollama/cuda_v12:/usr/lib/ollama/mlx_cuda_v12}"
export LD_LIBRARY_PATH="/usr/lib/ollama:/usr/lib/ollama/cuda_v12:/usr/lib/ollama/mlx_cuda_v12:${LD_LIBRARY_PATH}"

# MLX imagegen (x/z-image-turbo, etc.) — requires libmlxc.so from a one-time cmake MLX build.
# WHY separate dir: imagegen uses MLX-C CUDA backend, not ggml cuda_v12 runners.
# Build once: cmake -B build-mlx --preset "MLX CUDA 12" -DMLX_CUDA_ARCHITECTURES=120-real
#   && cmake --build build-mlx --target mlx --target mlxc --parallel
#   && cmake --install build-mlx --component MLX --strip
#   && sudo cp -a dist/lib/ollama/mlx_cuda_v12/* /usr/lib/ollama/mlx_cuda_v12/

# Wan T2V on 16g / SM120 (5080 class); unset to use manifest-only defaults.
export ZEROLLAMA_WAN_FORCE_SDPA="${ZEROLLAMA_WAN_FORCE_SDPA:-1}"
export ZEROLLAMA_WAN_VAE_CPU="${ZEROLLAMA_WAN_VAE_CPU:-1}"
export ZEROLLAMA_WAN_UNLOAD_T5="${ZEROLLAMA_WAN_UNLOAD_T5:-1}"

ZEROLLAMA_BIN="${ZEROLLAMA_BIN:-/usr/bin/zerollama}"
if [[ ! -x "$ZEROLLAMA_BIN" ]]; then
  ZEROLLAMA_BIN="$(command -v zerollama || true)"
fi
if [[ -z "$ZEROLLAMA_BIN" ]]; then
  echo "zerollama binary not found; set ZEROLLAMA_BIN" >&2
  exit 1
fi

# Embedded runtime binds 127.0.0.1:8081 (or ZEROLLAMA_RUNTIME_EMBED_PORT). Stop stale listeners.
EMBED_PORT="${ZEROLLAMA_RUNTIME_EMBED_PORT:-8081}"
if command -v ss >/dev/null 2>&1 && ss -tln 2>/dev/null | grep -q ":${EMBED_PORT} "; then
  echo "WARN: :${EMBED_PORT} in use — stopping stale zerollama serve (not systemd ollama)" >&2
  pkill -f 'zerollama serve' 2>/dev/null || true
  sleep 1
fi
unset ZEROLLAMA_RUNTIME_URL
export ZEROLLAMA_RUNTIME_EMBED="${ZEROLLAMA_RUNTIME_EMBED:-on}"

# After serve + one load: ./scripts/gpu_health_report.sh
# Preflight (no GPU): ./scripts/phase12_golden_ci.sh
# Calibration JSON: GPU_PHASE13_SNAPSHOT_OUT=5080.json ./scripts/gpu_phase13_snapshot.sh --gguf "$LLAMA_MODEL"
# Full check: ./scripts/gpu_smoke_all.sh  (needs LLAMA_MODEL, LLAMA_SERVER_BIN in env)
# Phase 14: source ./scripts/phase14_serve_env.sh; ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess|llama-cpp-python
#   ./scripts/phase14_backend_smoke.sh  OR  ./scripts/phase14_both_backends.sh (restarts serve per backend)
#   RUN_E2E_PREFLIGHT=1 RUN_E2E_TOOLS=1 RUN_E2E_PROXY_MODEL=tag RUN_E2E_LEGACY=1 RUN_E2E_LEGACY_MODEL=tag
# Optional: RUN_E2E_TOOLS=1 ./scripts/gpu_smoke_all.sh
# Optional clamp smoke: export ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX=auto
#   RUN_E2E_VRAM_CLAMP=1 RUN_E2E_GPU=1 ./scripts/e2e_runtime_smoke.sh
#
# Logging (pick one):
#   exec "$ZEROLLAMA_BIN" serve                                    # stdout to screen/tmux
#   exec "$ZEROLLAMA_BIN" serve >> /tmp/zerollama-serve.log 2>&1 # quiet screen; tail -f log
#   exec "$ZEROLLAMA_BIN" serve 2>&1 | tee -a /tmp/zerollama-serve.log  # both
# WHY log redirect on CT 1564: GIN + runner spam fills screen; operators use tail -f on one file.
SERVE_LOG="${SERVE_LOG:-}"
if [[ -n "$SERVE_LOG" ]]; then
  exec "$ZEROLLAMA_BIN" serve >> "$SERVE_LOG" 2>&1
fi
exec "$ZEROLLAMA_BIN" serve
