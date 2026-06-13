#!/usr/bin/env bash
# L2 fork evaluation smoke — probe binary + runtime profile flags (no full bench).
#
# WHY: L2 gate needs stock vs fork comparison on 5080 + M-series before vendor
# replacement. This script validates build + profile emission; tok/s bench is manual.
#
# Prerequisite (fork build):
#   ./scripts/build_eliza_llama_server.sh
#
# Usage:
#   export LLAMA_SERVER_BIN=../eliza-llama.cpp/build/bin/llama-server
#   ./scripts/l2_fork_eval.sh
#
# Optional self-build:
#   L2_BUILD_FORK=1 ./scripts/l2_fork_eval.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runtime_uv_venv.sh
source "${ROOT}/scripts/runtime_uv_venv.sh"
runtime_uv_venv

BIN="${LLAMA_SERVER_BIN:-}"
if [[ "${L2_BUILD_FORK:-0}" == "1" ]]; then
  "${ROOT}/scripts/build_eliza_llama_server.sh"
  BIN="${LLAMA_SERVER_BIN:-${ROOT}/../eliza-llama.cpp/build/bin/llama-server}"
fi

if [[ -z "${BIN}" || ! -x "${BIN}" ]]; then
  echo "Set LLAMA_SERVER_BIN to elizaOS/llama.cpp llama-server or L2_BUILD_FORK=1" >&2
  exit 1
fi

echo "== fork binary probe =="
"${BIN}" --version 2>/dev/null || true
help_text="$("${BIN}" --help 2>&1 || true)"
if echo "${help_text}" | grep -qE 'ctx-checkpoints|qjl1_256'; then
  echo "probe: fork llama-server (QJL/Polar/TBQ or ctx-checkpoints)"
else
  echo "probe: stock llama-server (no fork markers in --help)" >&2
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
echo "  Mac: ./scripts/l2_metal_bench.sh   # automated A/B JSON"
echo "  Mac: ./scripts/m3_metal_signoff.sh"
echo "  CUDA: ./scripts/gpu_5080_session.sh"
echo "Doc: docs/gpu-profiles-l2.md"
