#!/usr/bin/env bash
# L2 fork evaluation smoke — probe unified llama-server + runtime profile flags (no full bench).
#
# WHY: L2 gate compares L1 (q8_0) vs fork (QJL/Polar) profiles on one binary.
#
# Prerequisite:
#   ./scripts/build/build_llama_server.sh
#
# Usage:
#   export LLAMA_SERVER_BIN=../llama.cpp/build/bin/llama-server
#   ./scripts/phase/l2_fork_eval.sh
#
# Optional self-build:
#   L2_BUILD=1 ./scripts/phase/l2_fork_eval.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/runtime/runtime_uv_venv.sh
source "${ROOT}/scripts/runtime/runtime_uv_venv.sh"
runtime_uv_venv

ZEROLLAMA_PARENT="$(cd "${ROOT}/.." && pwd)"
UNIFIED_ROOT="${LLAMA_CPP_ROOT:-${ZEROLLAMA_PARENT}/llama.cpp}"
BIN="${LLAMA_SERVER_BIN:-}"
if [[ "${L2_BUILD:-0}" == "1" || "${L2_BUILD_FORK:-0}" == "1" ]]; then
  LLAMA_CPP_ROOT="${UNIFIED_ROOT}" "${ROOT}/scripts/build/build_llama_server.sh"
  BIN="${LLAMA_SERVER_BIN:-${UNIFIED_ROOT}/build/bin/llama-server}"
fi

if [[ -z "${BIN}" || ! -x "${BIN}" ]]; then
  BIN="${UNIFIED_ROOT}/build/bin/llama-server"
fi
if [[ ! -x "${BIN}" ]]; then
  echo "Set LLAMA_SERVER_BIN or run L2_BUILD=1 ./scripts/phase/l2_fork_eval.sh" >&2
  exit 1
fi

echo "== fork binary probe =="
"${BIN}" --version 2>/dev/null || true
help_text="$("${BIN}" --help 2>&1 || true)"
if echo "${help_text}" | grep -qE 'ctx-checkpoints|qjl1_256'; then
  echo "probe: unified llama-server (QJL/Polar/TBQ or ctx-checkpoints)"
else
  echo "probe: binary missing fork markers in --help (wrong ref?)" >&2
fi

echo ""
echo "== runtime fork detection + profile argv =="
export LLAMA_SERVER_BIN="${BIN}"
(cd "${ROOT}/runtime" && PYTHONPATH=. "${RUNTIME_UV_PYTHON}" <<'PY'
import os
from pathlib import Path

from runtime.llama_fork import fork_health, llama_fork_enabled
from runtime.config import RuntimeConfig

bin_p = Path(os.environ["LLAMA_SERVER_BIN"])
print("fork_health:", fork_health(llama_server_bin=bin_p))
print("llama_fork_enabled:", llama_fork_enabled(llama_server_bin=bin_p))

os.environ["ZEROLLAMA_GPU_PROFILE"] = "1"
# Auto-probe path: omit ZEROLLAMA_LLAMA_FORK when binary advertises fork markers.

# Pick platform-appropriate YAML
cfg_path = Path("configs/apple_silicon.yaml")
if __import__("sys").platform != "darwin":
    cfg_path = Path("configs/single_gpu.yaml")

cfg = RuntimeConfig.from_file(cfg_path)
gp = cfg.gpu_profile or {}
print("gpu_profile.id:", gp.get("id"))
print("cache_types_fallback:", gp.get("cache_types_fallback"))
print("llama_fork:", gp.get("llama_fork"))
args = cfg.llama_server_args()
print("llama_server_args:", " ".join(args))
if llama_fork_enabled(llama_server_bin=bin_p):
    for flag in ("--cache-type-k", "--ctx-checkpoints"):
        if flag in args:
            print(f"  fork flag present: {flag}")
PY
)

echo ""
echo "== pytest (fork module) =="
(cd "${ROOT}/runtime" && PYTHONPATH=. "${RUNTIME_UV_PYTHON}" -m pytest tests/test_llama_fork.py tests/test_gpu_profiles.py -q)

echo ""
echo "PASS: l2_fork_eval (binary probe + profile argv)"
echo "Manual gate: compare decode tok/s + VRAM @ same model with ZEROLLAMA_LLAMA_FORK=0 vs 1"
echo "  Mac: ./scripts/phase/l2_metal_bench.sh   # automated A/B JSON"
echo "  Mac: ./scripts/phase/m3_metal_signoff.sh"
echo "  CUDA: ./scripts/gpu/gpu_5080_session.sh"
echo "Doc: docs/gpu-profiles-l2.md"
