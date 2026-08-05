#!/usr/bin/env bash
# Example zerollama serve for a GPU host (e.g. cudallama with RTX 5080).
#
# WHY this file stays in scripts/: it sources repo helpers via _ROOT. Do NOT copy
# verbatim to ~/bin/serve.sh — dirname/.. becomes $HOME and breaks PYTHONPATH.
# Install: cp scripts/serve/serve_production_wrapper.sh ~/bin/serve.sh
#
# In-repo debug: SERVE_LOG=/tmp/zerollama-serve.log bash scripts/serve/serve_gpu_example.sh
# Doc: docs/5080-runbook.md#production-serve-binserve-sh
set -euo pipefail

# Resolve repo root: this file normally lives in scripts/; ~/bin/serve.sh wraps it.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${_SCRIPT_DIR}/../runtime/runtime/server/app.py" ]]; then
  _ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
elif [[ -n "${ZEROLLAMA_REPO:-}" && -f "${ZEROLLAMA_REPO}/runtime/runtime/server/app.py" ]]; then
  _ROOT="${ZEROLLAMA_REPO}"
elif [[ -f "${HOME}/zerollama/runtime/runtime/server/app.py" ]]; then
  _ROOT="${HOME}/zerollama"
else
  echo "serve_gpu_example: cannot find zerollama repo; set ZEROLLAMA_REPO" >&2
  exit 1
fi
export ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-${_ROOT}}"
# shellcheck source=scripts/runtime/sched_watchdog_env.sh
source "${_ROOT}/scripts/runtime/sched_watchdog_env.sh"

# Repo root: training.py + optional runtime/ checkout.
# Auto-detect now checks $HOME/zerollama; set explicitly if your layout differs.
export OLLAMA_TRAINING_PYTHONPATH="${OLLAMA_TRAINING_PYTHONPATH:-${ZEROLLAMA_REPO}}"
export ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-${OLLAMA_TRAINING_PYTHONPATH}}"

export OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:8080}"
# Runtime → Go /internal/* (cross-queue-seq, render-chat) must use loopback, not the bind address.
export ZEROLLAMA_GO_URL="${ZEROLLAMA_GO_URL:-http://127.0.0.1:8080}"
export OLLAMA_LLM_LIBRARY="${OLLAMA_LLM_LIBRARY:-cuda_v13}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-8}"
export ZEROLLAMA_GGML_AUTO_PARALLEL="${ZEROLLAMA_GGML_AUTO_PARALLEL:-auto}"
export OLLAMA_N_CTX="${OLLAMA_N_CTX:-12288}"

# Runtime YAML + VRAM policy (single-GPU 16GB — tune after measurement).
export ZEROLLAMA_RUNTIME_CONFIG="${ZEROLLAMA_RUNTIME_CONFIG:-$ZEROLLAMA_REPO/runtime/configs/single_gpu.yaml}"
export ZEROLLAMA_RUNTIME_VRAM_MIN_FREE="${ZEROLLAMA_RUNTIME_VRAM_MIN_FREE:-1GiB}"
export ZEROLLAMA_RUNTIME_TRAINING_VRAM_RESERVE="${ZEROLLAMA_RUNTIME_TRAINING_VRAM_RESERVE:-2GiB}"
# Optional: lower num_ctx to /health suggestion when GPU checks on (default off).
# export ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX=auto
export ZEROLLAMA_RUNTIME_VRAM_PROBE_CALIBRATE="${ZEROLLAMA_RUNTIME_VRAM_PROBE_CALIBRATE:-auto}"
export ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE="${ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE:-auto}"

# L1 GPU profile (5080 → runtime/configs/gpu/rtx-5080.json): stock q8_0 KV + np=2.
# WHY explicit: empty env already defaults ON in gpu_profiles_enabled(), but production
# must not silently inherit ZEROLLAMA_LLAMA_FORK=1 (QJL/polar) from a prior lab shell.
# Jul 2026 llama-bench: Llama-3.1-8B q8_0 beats f16 tg on this card — keep stock path.
export ZEROLLAMA_GPU_PROFILE="${ZEROLLAMA_GPU_PROFILE:-1}"
export ZEROLLAMA_LLAMA_FORK="${ZEROLLAMA_LLAMA_FORK:-0}"
# Optional force: export ZEROLLAMA_GPU_PROFILE_ID=rtx-5080

# Training: Go listens on :9500 and /api/train/* (embedded CPython, not a python sidecar).
export OLLAMA_TRAINING="${OLLAMA_TRAINING:-true}"
export OLLAMA_TRAINING_TCP="${OLLAMA_TRAINING_TCP:-:9500}"

# WHY embed PYTHONPATH: training + runtime share one CPython; uvicorn in runtime/.venv,
# torch in .venv-training (same order as scripts/gpu/5080_env.sh). See docs/gpu-training.md.
# shellcheck source=scripts/training/training_uv_venv.sh disable=SC1091
source "${_ROOT}/scripts/training/training_uv_venv.sh"
_EMBED_PY="$(embedded_training_python_ver)"
_RT_SITE="${ZEROLLAMA_REPO}/runtime/.venv/lib/python${_EMBED_PY}/site-packages"
if [[ ! -d "${_RT_SITE}" ]]; then
  echo "runtime: missing ${_RT_SITE} — run RUNTIME_UV_SYNC=1 ${_ROOT}/scripts/runtime/runtime_uv_venv.sh" >&2
  RUNTIME_UV_SYNC=1 "${_ROOT}/scripts/runtime/runtime_uv_venv.sh" >&2 || true
fi
_PYPP=""
if [[ -d "${_RT_SITE}" ]]; then
  _PYPP="${_RT_SITE}"
fi
if [[ "${OLLAMA_TRAINING}" == "true" ]]; then
  _TRAIN_VENV="${ZEROLLAMA_REPO}/.venv-training"
  _TRAIN_SITE="${_TRAIN_VENV}/lib/python${_EMBED_PY}/site-packages"
  if [[ ! -d "${_TRAIN_SITE}" ]]; then
    echo "training: missing ${_TRAIN_SITE} — run:" >&2
    echo "  TRAINING_UV_VENV=${_TRAIN_VENV} TRAINING_UV_PYTHON_VER=${_EMBED_PY} ${_ROOT}/scripts/training/training_uv_venv.sh --verify" >&2
    TRAINING_UV_VENV="${_TRAIN_VENV}" TRAINING_UV_PYTHON_VER="${_EMBED_PY}" \
      "${_ROOT}/scripts/training/training_uv_venv.sh" --verify >&2 || true
  fi
  if [[ -d "${_TRAIN_SITE}" ]]; then
    export TRAINING_UV_VENV="${_TRAIN_VENV}"
    export TRAINING_UV_SITE_PACKAGES="${_TRAIN_SITE}"
    _PYPP="${_PYPP}${_PYPP:+:}${_TRAIN_SITE}"
  fi
fi
if [[ -n "${_PYPP}" ]]; then
  export PYTHONPATH="${_PYPP}${PYTHONPATH:+:${PYTHONPATH}}"
fi

# Stability on new GPU architectures (optional).
export GGML_CUDA_USE_GRAPHS="${GGML_CUDA_USE_GRAPHS:-0}"
# Prefer 1 on 5080/CUDA 13: skip MMQ for IQ* stability (upstream: #21371 #24399).
# Go llama-server clears this to 0 for native MXFP4 (type 39) MoE — FORCE_CUBLAS
# breaks gpt-oss MUL_MAT_ID (llama.cpp#19659 → "?" token loops). Override:
# ZEROLLAMA_MXFP4_ALLOW_FORCE_CUBLAS=1
export GGML_CUDA_FORCE_CUBLAS="${GGML_CUDA_FORCE_CUBLAS:-1}"
# Clamp absurd client num_ctx before load (5080 / single-GPU hosts).
export ZEROLLAMA_GGML_CLAMP_NUM_CTX="${ZEROLLAMA_GGML_CLAMP_NUM_CTX:-1}"

# CUDA libs for ggml cuda_v13 (5080 / Blackwell). Override to cuda_v12 paths on older GPUs.
# /usr/hostlibs may expose libcudnn older than PyTorch in the Wan venv; keep hostlibs for
# ggml. Wan run_script subprocesses sanitize LD (prepend venv torch/lib, drop hostlibs).
# See docs/wan-t2v.md troubleshooting: "cuDNN version incompatibility".
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/root/nvidia-host:/usr/hostlibs:/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda-13.3/targets/x86_64-linux/lib}"
# ggml CUDA backend (chat/completion runners) — without /usr/lib/ollama + cuda_v13, ggml falls back to CPU-only on 5080.
export OLLAMA_LIBRARY_PATH="${OLLAMA_LIBRARY_PATH:-/usr/lib/ollama:/usr/lib/ollama/cuda_v13:/usr/lib/ollama/mlx_cuda_v12}"
export LD_LIBRARY_PATH="/root/nvidia-host:/usr/lib/ollama:/usr/lib/ollama/cuda_v13:/usr/lib/ollama/mlx_cuda_v12:${LD_LIBRARY_PATH}"

# Model store on data volume (CT 1564 rootfs is only ~30G).
export OLLAMA_MODELS="${OLLAMA_MODELS:-${ZEROLLAMA_REPO}/../.ollama/models}"
if [[ ! -d "$OLLAMA_MODELS" ]]; then
  # Fallback when ZEROLLAMA_REPO is ~/zerollama symlink into private/
  export OLLAMA_MODELS="${OLLAMA_MODELS_FALLBACK:-/var/lib/vz/private/1564/root/.ollama/models}"
fi

# Prefer current vendor pin llama-server (Makefile.sync FETCH_HEAD).
# WHY: run/llama-server-b10159 may lag the unified pin; TTS copy lacks Q2/Kokoro unify.
# Retire TTS unless ZEROLLAMA_KEEP_LLAMA_SERVER_BIN=1 (explicit operator pin).
_VENDOR_PIN="${Z5080_VENDOR_PIN:-$(grep '^FETCH_HEAD=' "${ZEROLLAMA_REPO}/Makefile.sync" 2>/dev/null | cut -d= -f2 || echo 5f55650a)}"
_VENDOR_ROOT="${ZEROLLAMA_VENDOR_ROOT:-${ZEROLLAMA_REPO}/vendor/llama-cpp-${_VENDOR_PIN}}"
_B10159_BIN="${ZEROLLAMA_REPO}/run/llama-server-b10159/llama-server"
_KEEP_TTS="${ZEROLLAMA_KEEP_LLAMA_SERVER_BIN:-0}"
if [[ "${_KEEP_TTS}" != "1" ]]; then
  case "${LLAMA_SERVER_BIN:-}" in
    */llama-server-tts/*) unset LLAMA_SERVER_BIN ;;
  esac
fi
if [[ -z "${LLAMA_SERVER_BIN:-}" && -x "${_VENDOR_ROOT}/build/bin/llama-server" ]]; then
  export LLAMA_CPP_ROOT="${_VENDOR_ROOT}"
  export LLAMA_CPP_BIN="${_VENDOR_ROOT}/build/bin"
  export LLAMA_SERVER_BIN="${LLAMA_CPP_BIN}/llama-server"
  export LLAMA_CPP_LIB="${LLAMA_CPP_BIN}/libllama.so"
  export LD_LIBRARY_PATH="${LLAMA_CPP_BIN}:${LD_LIBRARY_PATH:-}"
elif [[ -z "${LLAMA_SERVER_BIN:-}" && -x "${_B10159_BIN}" ]]; then
  export LLAMA_CPP_BIN="${ZEROLLAMA_REPO}/run/llama-server-b10159"
  export LLAMA_SERVER_BIN="${_B10159_BIN}"
  export LLAMA_CPP_LIB="${LLAMA_CPP_BIN}/libllama.so"
  export LD_LIBRARY_PATH="${LLAMA_CPP_BIN}:${LD_LIBRARY_PATH:-}"
fi

# MLX imagegen (x/z-image-turbo, etc.) — requires libmlxc.so from a one-time cmake MLX build.
# WHY separate dir: imagegen uses MLX-C CUDA backend, not ggml cuda_v12 runners.
# Build once: cmake -B build-mlx --preset "MLX CUDA 12" -DMLX_CUDA_ARCHITECTURES=120-real
#   && cmake --build build-mlx --target mlx --target mlxc --parallel
#   && cmake --install build-mlx --component MLX --strip
#   && sudo cp -a dist/lib/ollama/mlx_cuda_v12/* /usr/lib/ollama/mlx_cuda_v12/

# CUDA 12.8 for MLX imagegen NVRTC (include/cuda/std/tuple).
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"

# Wan T2V on 16g / SM120 (5080 class); unset to use manifest-only defaults.
# VAE_CPU default 0 — CPU VAE spikes host RAM and OOMs ~24G CTs (see docs/wan-t2v.md).
export ZEROLLAMA_WAN_FORCE_SDPA="${ZEROLLAMA_WAN_FORCE_SDPA:-1}"
export ZEROLLAMA_WAN_VAE_CPU="${ZEROLLAMA_WAN_VAE_CPU:-0}"
export ZEROLLAMA_WAN_UNLOAD_T5="${ZEROLLAMA_WAN_UNLOAD_T5:-1}"
export ZEROLLAMA_WAN_MIN_HOST_RAM_GIB="${ZEROLLAMA_WAN_MIN_HOST_RAM_GIB:-14}"

# Prefer repo run/ binary when present (CT 1564 install layout).
if [[ -z "${ZEROLLAMA_BIN:-}" && -x "${ZEROLLAMA_REPO}/run/zerollama" ]]; then
  ZEROLLAMA_BIN="${ZEROLLAMA_REPO}/run/zerollama"
fi
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

# After serve + one load: ./scripts/gpu/gpu_health_report.sh
# Preflight (no GPU): ./scripts/phase/phase12_golden_ci.sh
# Calibration JSON: GPU_PHASE13_SNAPSHOT_OUT=5080.json ./scripts/gpu/gpu_phase13_snapshot.sh --gguf "$LLAMA_MODEL"
# Full check: ./scripts/gpu/gpu_smoke_all.sh  (needs LLAMA_MODEL, LLAMA_SERVER_BIN in env)
# Phase 14: source ./scripts/phase/phase14_serve_env.sh; ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess|llama-cpp-python
#   ./scripts/phase/phase14_backend_smoke.sh  OR  ./scripts/phase/phase14_both_backends.sh (restarts serve per backend)
#   RUN_E2E_PREFLIGHT=1 RUN_E2E_TOOLS=1 RUN_E2E_PROXY_MODEL=tag RUN_E2E_LEGACY=1 RUN_E2E_LEGACY_MODEL=tag
# Optional: RUN_E2E_TOOLS=1 ./scripts/gpu/gpu_smoke_all.sh
# Optional clamp smoke: export ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX=auto
#   RUN_E2E_VRAM_CLAMP=1 RUN_E2E_GPU=1 ./scripts/e2e/e2e_runtime_smoke.sh
#
# Logging (pick one):
#   exec "$ZEROLLAMA_BIN" serve                                    # stdout to screen/tmux
#   exec "$ZEROLLAMA_BIN" serve >> /tmp/zerollama-serve.log 2>&1 # quiet screen; tail -f log
#   "$ZEROLLAMA_BIN" serve 2>&1 | (trap '' INT; tee -a /tmp/log)  # both; trap so tee ignores Ctrl+C
# WHY log redirect on CT 1564: GIN + runner spam fills screen; operators use tail -f on one file.
# WHY trap INT on tee: otherwise pipeline SIGINT kills tee first and races force-quit.
SERVE_LOG="${SERVE_LOG:-}"
if [[ -n "$SERVE_LOG" ]]; then
  exec "$ZEROLLAMA_BIN" serve >> "$SERVE_LOG" 2>&1
fi
exec "$ZEROLLAMA_BIN" serve
