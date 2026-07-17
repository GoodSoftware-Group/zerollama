#!/usr/bin/env bash
# RTX 5080 / CT 1564 session env — source once per shell, then run gates.
#
#   cd ~/zerollama && source ./scripts/5080_env.sh
#   ./scripts/5080_resignoff.sh              # full re-sign-off
#   ./scripts/gpu_5080_session.sh            # tier 1 only
#
# WHY one file: operators were copying env from gpu-5080-operator-guide, 5080-runbook,
# gpu-profiles-l3, and serve_gpu_example — missing PYTHONPATH or RUN_E2E_PREFLIGHT caused
# false FAILs. Override any variable after sourcing.
#
# Doc: docs/5080-runbook.md

_Z5080_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")/.." && pwd)"
export Z5080_ROOT="${Z5080_ROOT:-$_Z5080_ROOT}"
export ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-${Z5080_ROOT}}"
export CUDA_LANE="${CUDA_LANE:-rtx_5080}"

# Shared CUDA vars (libs, CGO, llama-server discovery) — docs/cuda-lanes.md
# shellcheck source=scripts/cuda_common_env.sh disable=SC1091
source "${Z5080_ROOT}/scripts/cuda_common_env.sh"

# --- CT layout (override on non-1564 hosts) ---
export Z5080_CT="${Z5080_CT:-1564}"
export Z5080_REPO="${Z5080_REPO:-${HOME}/zerollama}"
export Z5080_LLAMA_CPP="${Z5080_LLAMA_CPP:-${HOME}/llama.cpp}"
export Z5080_VENDOR_PIN="${Z5080_VENDOR_PIN:-c84b3020}"
export Z5080_VENDOR_ROOT="${Z5080_VENDOR_ROOT:-${Z5080_REPO}/vendor/llama-cpp-${Z5080_VENDOR_PIN}}"

# --- GGUF paths ---
export LLAMA_MODEL="${LLAMA_MODEL:-/root/Llama-OuteTTS-1.0-1B-Q8_0.gguf}"
export CUDA_LLAMA_MODEL="${CUDA_LLAMA_MODEL:-/root/eliza-1-9b-256k.gguf}"
export RUN_E2E_GGUF="${RUN_E2E_GGUF:-$LLAMA_MODEL}"
export RUN_E2E_PROXY_MODEL="${RUN_E2E_PROXY_MODEL:-llama3.2:3b}"

# --- Binaries: prefer patched vendor (fork QJL + /kv/seq-copy) when built ---
if [[ -z "${LLAMA_CPP_ROOT:-}" && -x "${Z5080_VENDOR_ROOT}/build/bin/llama-server" ]]; then
  export LLAMA_CPP_ROOT="${Z5080_VENDOR_ROOT}"
fi
export LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-${Z5080_LLAMA_CPP}}"
export LLAMA_CPP_BIN="${LLAMA_CPP_BIN:-${LLAMA_CPP_ROOT}/build/bin}"
export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_CPP_BIN}/libllama.so}"
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${LLAMA_CPP_BIN}/llama-server}"
export LD_LIBRARY_PATH="${LLAMA_CPP_BIN}:${LD_LIBRARY_PATH:-}"

# --- Serve / embed ---
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:8080}"
export ZEROLLAMA_GO_URL="${ZEROLLAMA_GO_URL:-http://127.0.0.1:8080}"
unset ZEROLLAMA_RUNTIME_URL
export ZEROLLAMA_RUNTIME_EMBED="${ZEROLLAMA_RUNTIME_EMBED:-on}"
export ZEROLLAMA_RUNTIME_CONFIG="${ZEROLLAMA_RUNTIME_CONFIG:-${Z5080_REPO}/runtime/configs/single_gpu.yaml}"
export ZEROLLAMA_RUNTIME_VRAM_PROBE_CALIBRATE="${ZEROLLAMA_RUNTIME_VRAM_PROBE_CALIBRATE:-auto}"
export ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE="${ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE:-auto}"

# --- Python embed (WHY: ModuleNotFoundError: uvicorn without runtime site-packages) ---
# Training site-packages ABI must match libpython linked into zerollama — see training_uv_venv.sh.
# shellcheck source=scripts/training_uv_venv.sh disable=SC1091
source "${Z5080_REPO}/scripts/training_uv_venv.sh"
_Z5080_EMBED_PY="$(embedded_training_python_ver)"
export RT_SITE="${RT_SITE:-${Z5080_REPO}/runtime/.venv/lib/python3.11/site-packages}"
export TRAIN_SITE="${TRAIN_SITE:-${Z5080_REPO}/.venv-training/lib/python${_Z5080_EMBED_PY}/site-packages}"
export PYTHONPATH="${RT_SITE}:${TRAIN_SITE}${PYTHONPATH:+:${PYTHONPATH}}"

# --- Gate defaults (CT minimal tree) ---
export RUN_E2E_PREFLIGHT="${RUN_E2E_PREFLIGHT:-0}"
export GPU_PHASE13_SNAPSHOT_OUT="${GPU_PHASE13_SNAPSHOT_OUT:-/tmp/5080-session.json}"
export L3_RADIX_OUT="${L3_RADIX_OUT:-/tmp/l3-radix-prefix-smoke-live.json}"

# --- Build (CGO) ---
export CGO_ENABLED="${CGO_ENABLED:-1}"
export CGO_CFLAGS_ALLOW="${CGO_CFLAGS_ALLOW:--mfma|-mavx2|-O3}"

# --- CUDA toolkit (prefer 13.3 after driver 610; fall back to 12.8) ---
if [[ -x /usr/local/cuda/bin/nvcc ]]; then
  export PATH="/usr/local/cuda/bin:${PATH}"
  export CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
  export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-/usr/local/cuda}"
elif [[ -x /usr/local/cuda-13.3/bin/nvcc ]]; then
  export PATH="/usr/local/cuda-13.3/bin:${PATH}"
  export CUDACXX="${CUDACXX:-/usr/local/cuda-13.3/bin/nvcc}"
  export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-/usr/local/cuda-13.3}"
elif [[ -d /usr/local/cuda-12.8/bin ]]; then
  export PATH="/usr/local/cuda-12.8/bin:${PATH}"
  export CUDACXX="${CUDACXX:-/usr/local/cuda-12.8/bin/nvcc}"
  export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-/usr/local/cuda-12.8}"
fi

# --- Phase 17 defaults ---
export P17_MODEL="${P17_MODEL:-${RUN_E2E_PROXY_MODEL}}"

5080_setup_cuda() {
  if [[ ! -x "${CUDACXX:-/usr/local/cuda/bin/nvcc}" ]]; then
    echo "5080: install CUDA toolkit (prefer /usr/local/cuda 13.3 for driver 610 / sm_120)" >&2
    return 1
  fi
  export PATH="$(dirname "${CUDACXX}"):${PATH}"
}

5080_setup_venvs() {
  5080_cd_repo
  if [[ ! -d runtime/.venv ]]; then
    RUNTIME_UV_SYNC=1 "${Z5080_REPO}/scripts/runtime_uv_venv.sh"
  fi
  if [[ "${OLLAMA_TRAINING:-false}" == "true" ]]; then
    local embed_py site
    embed_py="$(embedded_training_python_ver)"
    site="${Z5080_REPO}/.venv-training/lib/python${embed_py}/site-packages"
    if [[ ! -d "$site" ]]; then
      TRAINING_UV_VENV="${Z5080_REPO}/.venv-training" TRAINING_UV_PYTHON_VER="${embed_py}" \
        "${Z5080_REPO}/scripts/training_uv_venv.sh" --verify
    fi
  fi
  export RT_SITE="${Z5080_REPO}/runtime/.venv/lib/python3.11/site-packages"
  embed_py="$(embedded_training_python_ver)"
  export TRAIN_SITE="${Z5080_REPO}/.venv-training/lib/python${embed_py}/site-packages"
  export PYTHONPATH="${RT_SITE}:${TRAIN_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
}

5080_build_patched_libllama() {
  # Phase 15 in-process KV — kv-ext symbols in libllama.so (patch 0014).
  # WHY helper here: runbook had 25 lines of copy-paste; one function for tier 3 prep.
  5080_setup_cuda || return 1
  local L="${Z5080_LLAMA_CPP}"
  5080_cd_repo
  git -C "${L}" checkout -f b9781
  cp "${Z5080_REPO}/llama/llama.cpp/include/llama-kv-ext.h" "${L}/include/"
  cp "${Z5080_REPO}/llama/llama.cpp/src/llama-memory-kv-ext.cpp" "${L}/src/"
  cp "${Z5080_REPO}/llama/llama.cpp/src/llama-kv-cache.h" "${L}/src/"
  cp "${Z5080_REPO}/llama/llama.cpp/src/llama-kv-cache.cpp" "${L}/src/"
  grep -q llama-memory-kv-ext.cpp "${L}/src/CMakeLists.txt" || \
    sed -i '/llama-memory-recurrent.cpp/a\            llama-memory-kv-ext.cpp' "${L}/src/CMakeLists.txt"
  cmake -C "${L}/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120-real -DBUILD_SHARED_LIBS=ON 2>/dev/null || \
    cmake -B "${L}/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120-real -DBUILD_SHARED_LIBS=ON
  cmake --build "${L}/build" -j
  nm -D "${L}/build/bin/libllama.so" | grep llama_memory_kv_
  export LLAMA_CPP_LIB="${L}/build/bin/libllama.so"
  export LLAMA_CPP_ROOT="${L}"
  export LLAMA_CPP_BIN="${L}/build/bin"
  export LLAMA_SERVER_BIN="${LLAMA_CPP_BIN}/llama-server"
  export LD_LIBRARY_PATH="${LLAMA_CPP_BIN}:${LD_LIBRARY_PATH}"
}

5080_pull_proxy_model() {
  if ! curl -sf "${OLLAMA_HOST}/api/tags" | grep -q "${RUN_E2E_PROXY_MODEL%%:*}"; then
    echo "5080: pulling ${RUN_E2E_PROXY_MODEL} for Phase 17 smokes"
    zerollama pull "${RUN_E2E_PROXY_MODEL}" || true
  fi
}

5080_cd_repo() {
  cd "${Z5080_REPO}"
}

5080_stop_serve() {
  local pid
  for pid in $(pgrep -x zerollama 2>/dev/null || true); do
    kill "$pid" 2>/dev/null || true
  done
  sleep 1
  for pid in $(pgrep -x zerollama 2>/dev/null || true); do
    kill -9 "$pid" 2>/dev/null || true
  done
  fuser -k 8080/tcp 8081/tcp 8082/tcp 2>/dev/null || true
  sleep 1
}

5080_wait_health() {
  local i
  for i in $(seq 1 30); do
    curl -sf -m 15 http://127.0.0.1:8081/health >/dev/null && return 0
    sleep 1
  done
  echo "5080: /health timeout — tail /tmp/zerollama-serve.log" >&2
  return 1
}

5080_start_serve() {
  5080_stop_serve
  5080_cd_repo
  zerollama serve >> /tmp/zerollama-serve.log 2>&1 &
  echo "5080: serve pid=$! log=/tmp/zerollama-serve.log"
  5080_wait_health
  curl -sf http://127.0.0.1:8080/api/tags >/dev/null
}

5080_build_zerollama() {
  5080_cd_repo
  if [[ ! -d llama/llama.cpp/vendor/cpp-httplib ]] && [[ -d "${Z5080_LLAMA_CPP}/vendor/cpp-httplib" ]]; then
    echo "5080: rsync cpp-httplib from sibling llama.cpp"
    rsync -a "${Z5080_LLAMA_CPP}/vendor/cpp-httplib/" llama/llama.cpp/vendor/cpp-httplib/
  fi
  # WHY 3.11 embed: runtime/.venv is 3.11; training .venv-training should match (see training_embed_build_env.sh).
  if [[ -f "${Z5080_REPO}/scripts/training_embed_build_env.sh" ]] && pkg-config --exists python-3.11-embed 2>/dev/null; then
    # shellcheck source=scripts/training_embed_build_env.sh disable=SC1091
    source "${Z5080_REPO}/scripts/training_embed_build_env.sh" 3.11
  fi
  go build -o zerollama .
  if [[ -w /usr/bin ]] || [[ "$(id -u)" -eq 0 ]]; then
    cp zerollama /usr/bin/zerollama 2>/dev/null || true
  fi
}

5080_build_sibling_llama_server() {
  # Subprocess + Phase 17 default path (kv-ext on sibling — see runbook Phase 15 section).
  if [[ ! -x "${LLAMA_SERVER_BIN}" ]]; then
    LLAMA_CPP_ROOT="${Z5080_LLAMA_CPP}" "${Z5080_REPO}/scripts/build_llama_server.sh"
  fi
}

5080_build_vendor_llama_server() {
  # Radix cross-slot live requires POST /kv/seq-copy (patch 0017 on vendor pin).
  # WHY vendor: bare sibling ../llama.cpp often lacks seq-copy even when in-tree is patched.
  5080_cd_repo
  if [[ ! -d "${Z5080_VENDOR_ROOT}" ]]; then
    echo "5080: vendor missing — run: make -f Makefile.sync vendor" >&2
    return 1
  fi
  LLAMA_CPP_ROOT="${Z5080_VENDOR_ROOT}" "${Z5080_REPO}/scripts/build_llama_server.sh"
  export LLAMA_SERVER_BIN="${Z5080_VENDOR_ROOT}/build/bin/llama-server"
  export LLAMA_CPP_BIN="$(dirname "${LLAMA_SERVER_BIN}")"
  export LLAMA_CPP_ROOT="${Z5080_VENDOR_ROOT}"
  export LD_LIBRARY_PATH="${LLAMA_CPP_BIN}:${LD_LIBRARY_PATH}"
  "${Z5080_REPO}/scripts/llama_patch_doctor.sh"
}

5080_patch_doctor() {
  LLAMA_PATCH_PROBE_URL="${LLAMA_PATCH_PROBE_URL:-http://127.0.0.1:8082}" \
    "${Z5080_REPO}/scripts/llama_patch_doctor.sh"
}

5080_print_env() {
  cat <<EOF
5080 env (source: scripts/5080_env.sh)
  Z5080_REPO=$Z5080_REPO
  LLAMA_MODEL=$LLAMA_MODEL
  CUDA_LLAMA_MODEL=$CUDA_LLAMA_MODEL
  LLAMA_SERVER_BIN=$LLAMA_SERVER_BIN
  LLAMA_CPP_LIB=${LLAMA_CPP_LIB:-<unset — tier 3 needs 5080_build_patched_libllama>}
  RUN_E2E_PREFLIGHT=$RUN_E2E_PREFLIGHT
  PYTHONPATH=$PYTHONPATH
  CUDACXX=${CUDACXX:-<unset>}
EOF
}

export Z5080_ENV_LOADED=1

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  5080_print_env
fi
