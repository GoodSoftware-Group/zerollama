#!/usr/bin/env bash
# One-shot 5080-class GPU session: preflight goldens + full smoke + Phase 13 snapshot.
#
# WHY this script: CI proves parsers without a GPU; a 16GB host needs one repeatable gate
# (Phase 10–13) with a JSON artifact + gpu_snapshot hints — not ad-hoc e2e flags.
# Does NOT require gpt-oss harmony on host (needs ~40+ GiB RAM); see docs/gpu-5080-operator-guide.md
#
#   export LLAMA_MODEL LLAMA_SERVER_BIN RUN_E2E_GGUF
#   export RUN_E2E_PROXY_MODEL=llama3.2:3B   # optional tools + proxy manifest
#   ./scripts/gpu_5080_session.sh
#
# Optional:
#   RUN_E2E_TOOLS=1 RUN_E2E_LEGACY=1 RUN_E2E_VRAM_CLAMP=1  # forwarded to gpu_smoke_all
#   RUN_E2E_TRAINING_OPS=1 RUN_E2E_TRAINING_TCP=1           # embedded training surfaces (serve needs OLLAMA_TRAINING=true)
#   RUN_E2E_PHASE14=1 RUN_E2E_INPROCESS=1                   # phase14_inprocess_smoke (serve must use inprocess)
#   RUN_E2E_PHASE14_SIGNOFF=1                               # phase14_5080_signoff (needs LLAMA_CPP_LIB; self-contained restarts)
#   RUN_E2E_PHASE15=1                                       # phase15_inprocess_signoff (needs LLAMA_CPP_LIB)
#   RUN_E2E_L1=1                                            # l1_cuda_full_gate (needs CUDA_LLAMA_MODEL or LLAMA_MODEL 7B–9B)
#   RUN_E2E_L3=1                                            # l3_cuda_full_gate (needs CUDA_LLAMA_MODEL or LLAMA_MODEL 9B+)
#   RUN_E2E_L3_SPEC=1                                       # also L3_RUN_SPEC_CACHE=1 on l3_cuda_full_gate (ngram policy leg)
#   RUN_E2E_L3_RADIX=1                                      # l3_radix_prefix_smoke live (vendor /kv/seq-copy; needs 9B+ GGUF)
#   RUN_E2E_LLAMA_BACKEND_SOURCE=config                      # with phase14_yaml_config_smoke prerequisites
#   RUN_E2E_LLAMA_CPP_PYTHON_GPU=1                           # wheel GPU (with RUN_E2E_LLAMA_CPP_PYTHON=1)
#   RUN_E2E_PREFLIGHT=0                                      # skip Go golden in CT/minimal trees (see below)
#   RUN_E2E_P17=1                                            # phase17_llama_server_smoke (needs LLAMA_SERVER_BIN + pulled tag)
#   RUN_E2E_P17_LINUX_AUTO=1                                 # phase17_linux_auto_smoke (plain serve, Linux only)
#   RUN_E2E_EDGE=1                                           # phase16_edge_smoke (serve --edge, runtime off)
#   RUN_E2E_UPSTREAM_GGUF=1                                  # bundle P17 + P17_LINUX_AUTO + EDGE (upstream GGUF path)
#   RUN_E2E_P17_VISION=1                                     # phase17_llama_server_vision_smoke (heavy; needs projector tag)
#   GPU_PHASE13_SNAPSHOT_OUT=/tmp/5080-session.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/sched_watchdog_env.sh
source "${ROOT}/scripts/sched_watchdog_env.sh"
# CT 1564: source scripts/5080_env.sh once per shell (paths, PYTHONPATH, RUN_E2E_PREFLIGHT=0).
# When resignoff or the operator already sourced it, honor those exports.
if [[ -n "${Z5080_ENV_LOADED:-}" ]]; then
  :
elif [[ -f "${ROOT}/scripts/5080_env.sh" && "${Z5080_AUTO_ENV:-0}" == "1" ]]; then
  # shellcheck source=scripts/5080_env.sh
  source "${ROOT}/scripts/5080_env.sh"
fi
export ZEROLLAMA_REPO_ROOT="${ZEROLLAMA_REPO_ROOT:-$ROOT}"
# WHY default-on but overridable: full hosts should run phase12_golden_ci before GPU smokes.
# Proxmox CT 1564 (and other minimal checkouts) often lack vendored cpp-httplib for CGO
# (`fatal error: cpp-httplib/httplib.h`) — operators set RUN_E2E_PREFLIGHT=0 for GPU-only gate.
# CI still runs golden separately; this only gates the 5080 session wrapper.
export RUN_E2E_PREFLIGHT="${RUN_E2E_PREFLIGHT:-1}"
export RUN_E2E_PHASE13_SNAPSHOT=1

if [[ "${RUN_E2E_UPSTREAM_GGUF:-0}" == "1" ]]; then
  export RUN_E2E_P17=1
  export RUN_E2E_P17_LINUX_AUTO=1
  export RUN_E2E_EDGE=1
fi
export GPU_PHASE13_SNAPSHOT_OUT="${GPU_PHASE13_SNAPSHOT_OUT:-/tmp/5080-session.json}"

if [[ -z "${LLAMA_MODEL:-}" && -z "${RUN_E2E_GGUF:-}" ]]; then
  echo "Set LLAMA_MODEL or RUN_E2E_GGUF (small GGUF for 16GB, e.g. 1B Q8)" >&2
  exit 1
fi
if [[ -z "${LLAMA_SERVER_BIN:-}" ]]; then
  echo "Set LLAMA_SERVER_BIN (path to llama-server)" >&2
  exit 1
fi

if [[ "${RUN_E2E_PREFLIGHT}" == "1" ]]; then
  echo "== Phase 12 preflight + GPU smokes + snapshot =="
else
  echo "== GPU smokes + snapshot (RUN_E2E_PREFLIGHT=0 — skip Go golden) =="
fi
# Phase 14/15 smokes run after snapshot in this script; suppress during gpu_smoke_all
# so RUN_E2E_PHASE14*=1 does not execute twice (~15–20 min sign-off).
_saved_phase14_signoff="${RUN_E2E_PHASE14_SIGNOFF:-0}"
_saved_phase15="${RUN_E2E_PHASE15:-0}"
_saved_phase14="${RUN_E2E_PHASE14:-0}"
export RUN_E2E_PHASE14_SIGNOFF=0 RUN_E2E_PHASE15=0 RUN_E2E_PHASE14=0
"${ROOT}/scripts/gpu_smoke_all.sh"
export RUN_E2E_PHASE14_SIGNOFF="${_saved_phase14_signoff}"
export RUN_E2E_PHASE15="${_saved_phase15}"
export RUN_E2E_PHASE14="${_saved_phase14}"

if [[ -f "${GPU_PHASE13_SNAPSHOT_OUT}" ]]; then
  echo ""
  (cd "${ROOT}/runtime" && PYTHONPATH=. python3 -m runtime.gpu_snapshot "${GPU_PHASE13_SNAPSHOT_OUT}") || true
fi

if [[ "${RUN_E2E_PHASE14_SIGNOFF:-0}" == "1" ]]; then
  if [[ -z "${LLAMA_CPP_LIB:-}" ]]; then
    echo "RUN_E2E_PHASE14_SIGNOFF=1 requires LLAMA_CPP_LIB (ctypes libllama.so)" >&2
    exit 1
  fi
  echo ""
  echo "== Phase 14/15 5080 sign-off (self-contained) =="
  "${ROOT}/scripts/phase14_5080_signoff.sh"
elif [[ "${RUN_E2E_PHASE15:-0}" == "1" ]]; then
  if [[ -z "${LLAMA_CPP_LIB:-}" ]]; then
    echo "RUN_E2E_PHASE15=1 requires LLAMA_CPP_LIB (ctypes libllama.so)" >&2
    exit 1
  fi
  echo ""
  echo "== Phase 15 in-process GPU sign-off =="
  "${ROOT}/scripts/phase15_inprocess_signoff.sh"
elif [[ "${RUN_E2E_PHASE14:-0}" == "1" ]]; then
  echo ""
  echo "== Phase 14 backend smoke =="
  if [[ "${RUN_E2E_INPROCESS:-0}" == "1" && -z "${RUN_E2E_LLAMA_BACKEND_SOURCE:-}" ]]; then
    "${ROOT}/scripts/phase14_inprocess_smoke.sh"
  elif [[ "${RUN_E2E_LLAMA_CPP_PYTHON:-0}" == "1" && "${RUN_E2E_LLAMA_CPP_PYTHON_GPU:-0}" != "1" && -z "${RUN_E2E_LLAMA_BACKEND_SOURCE:-}" ]]; then
    "${ROOT}/scripts/phase14_wheel_cpu_smoke.sh"
  else
    phase14_env=()
    [[ "${RUN_E2E_INPROCESS:-0}" == "1" ]] && phase14_env+=(RUN_E2E_INPROCESS=1)
    [[ "${RUN_E2E_LLAMA_CPP_PYTHON:-0}" == "1" ]] && phase14_env+=(RUN_E2E_LLAMA_CPP_PYTHON=1)
    [[ "${RUN_E2E_LLAMA_CPP_PYTHON_GPU:-0}" == "1" ]] && phase14_env+=(RUN_E2E_LLAMA_CPP_PYTHON_GPU=1)
    [[ -n "${RUN_E2E_LLAMA_BACKEND_SOURCE:-}" ]] && phase14_env+=(RUN_E2E_LLAMA_BACKEND_SOURCE="${RUN_E2E_LLAMA_BACKEND_SOURCE}")
    # shellcheck disable=SC2086
    env "${phase14_env[@]}" "${ROOT}/scripts/phase14_backend_smoke.sh"
  fi
fi

if [[ "${RUN_E2E_L1:-0}" == "1" ]]; then
  export CUDA_LLAMA_MODEL="${CUDA_LLAMA_MODEL:-${LLAMA_MODEL:-}}"
  if [[ -z "${CUDA_LLAMA_MODEL:-}" ]]; then
    echo "RUN_E2E_L1=1 requires CUDA_LLAMA_MODEL or LLAMA_MODEL (7B–9B production GGUF)" >&2
    exit 1
  fi
  echo ""
  echo "== L1 CUDA full gate =="
  CUDA_LLAMA_MODEL="${CUDA_LLAMA_MODEL}" "${ROOT}/scripts/l1_cuda_full_gate.sh"
fi

if [[ "${RUN_E2E_L3:-0}" == "1" ]]; then
  export CUDA_LLAMA_MODEL="${CUDA_LLAMA_MODEL:-${LLAMA_MODEL:-}}"
  if [[ -z "${CUDA_LLAMA_MODEL:-}" ]]; then
    echo "RUN_E2E_L3=1 requires CUDA_LLAMA_MODEL or LLAMA_MODEL (9B+ production GGUF)" >&2
    exit 1
  fi
  echo ""
  echo "== L3 CUDA full gate =="
  L3_RUN_SPEC_CACHE="${RUN_E2E_L3_SPEC:-0}" \
  L3_RUN_RADIX="${RUN_E2E_L3_RADIX:-0}" \
  CUDA_LLAMA_MODEL="${CUDA_LLAMA_MODEL}" "${ROOT}/scripts/l3_cuda_full_gate.sh"
fi

if [[ "${RUN_E2E_L3_RADIX:-0}" == "1" && "${RUN_E2E_L3:-0}" != "1" ]]; then
  export CUDA_LLAMA_MODEL="${CUDA_LLAMA_MODEL:-${LLAMA_MODEL:-}}"
  if [[ -z "${CUDA_LLAMA_MODEL:-}" ]]; then
    echo "RUN_E2E_L3_RADIX=1 requires CUDA_LLAMA_MODEL or LLAMA_MODEL (9B+ production GGUF)" >&2
    exit 1
  fi
  echo ""
  echo "== L3 Radix cross-slot live gate =="
  # WHY vendor binary: bare sibling ../llama.cpp lacks POST /kv/seq-copy (patch 0017).
  L3_RADIX_LIVE=1 \
  L3_RADIX_OUT="${L3_RADIX_OUT:-/tmp/l3-radix-prefix-smoke-live.json}" \
  CUDA_LLAMA_MODEL="${CUDA_LLAMA_MODEL}" \
    "${ROOT}/scripts/l3_radix_prefix_smoke.sh"
fi

if [[ "${RUN_E2E_P17:-0}" == "1" ]]; then
  if [[ -z "${LLAMA_SERVER_BIN:-}" || ! -x "${LLAMA_SERVER_BIN}" ]]; then
    echo "RUN_E2E_P17=1 requires LLAMA_SERVER_BIN (built llama-server)" >&2
    exit 1
  fi
  export P17_MODEL="${P17_MODEL:-${RUN_E2E_PROXY_MODEL:-}}"
  if [[ -z "${P17_MODEL}" ]]; then
    echo "RUN_E2E_P17=1 requires RUN_E2E_PROXY_MODEL or P17_MODEL (pulled local tag)" >&2
    exit 1
  fi
  echo ""
  echo "== Phase 17 llama-server smoke =="
  LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN}" P17_MODEL="${P17_MODEL}" \
    "${ROOT}/scripts/phase17_llama_server_smoke.sh"
fi

if [[ "${RUN_E2E_P17_LINUX_AUTO:-0}" == "1" ]]; then
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "RUN_E2E_P17_LINUX_AUTO=1 is Linux-only; skipping on $(uname -s)" >&2
  else
    if [[ -z "${LLAMA_SERVER_BIN:-}" || ! -x "${LLAMA_SERVER_BIN}" ]]; then
      echo "RUN_E2E_P17_LINUX_AUTO=1 requires LLAMA_SERVER_BIN (built llama-server)" >&2
      exit 1
    fi
    export P17_MODEL="${P17_MODEL:-${RUN_E2E_PROXY_MODEL:-}}"
    if [[ -z "${P17_MODEL}" ]]; then
      echo "RUN_E2E_P17_LINUX_AUTO=1 requires RUN_E2E_PROXY_MODEL or P17_MODEL (pulled local tag)" >&2
      exit 1
    fi
    echo ""
    echo "== Phase 17 Linux auto smoke =="
    LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN}" P17_MODEL="${P17_MODEL}" \
      "${ROOT}/scripts/phase17_linux_auto_smoke.sh"
  fi
fi

if [[ "${RUN_E2E_EDGE:-0}" == "1" ]]; then
  if [[ -z "${LLAMA_SERVER_BIN:-}" || ! -x "${LLAMA_SERVER_BIN}" ]]; then
    echo "RUN_E2E_EDGE=1 requires LLAMA_SERVER_BIN (built llama-server)" >&2
    exit 1
  fi
  export P16_MODEL="${P16_MODEL:-${RUN_E2E_PROXY_MODEL:-${P17_MODEL:-}}}"
  if [[ -z "${P16_MODEL:-}" ]]; then
    echo "RUN_E2E_EDGE=1 requires RUN_E2E_PROXY_MODEL or P16_MODEL (pulled local tag)" >&2
    exit 1
  fi
  echo ""
  echo "== Phase 16 edge smoke =="
  LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN}" P16_MODEL="${P16_MODEL}" \
    "${ROOT}/scripts/phase16_edge_smoke.sh"
fi

if [[ "${RUN_E2E_P17_VISION:-0}" == "1" ]]; then
  if [[ -z "${LLAMA_SERVER_BIN:-}" || ! -x "${LLAMA_SERVER_BIN}" ]]; then
    echo "RUN_E2E_P17_VISION=1 requires LLAMA_SERVER_BIN (built llama-server)" >&2
    exit 1
  fi
  echo ""
  echo "== Phase 17 llama-server vision smoke =="
  RUN_E2E_P17_VISION=1 LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN}" \
    "${ROOT}/scripts/phase17_llama_server_vision_smoke.sh"
fi

echo "PASS: gpu_5080_session (snapshot: ${GPU_PHASE13_SNAPSHOT_OUT})"
