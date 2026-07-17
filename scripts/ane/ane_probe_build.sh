#!/usr/bin/env bash
# Build tools/ane-probe against maderix/ane (libane_bridge.dylib).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ANE_REPO="${ANE_REPO:-${HOME}/Sites/inference/ane}"
PROBE_DIR="${ROOT}/tools/ane-probe"
DRAFT_DIR="${ROOT}/tools/ane-draft"
IOSURFACE_DIR="${ROOT}/tools/ane-iosurface"
METAL_DIR="${ROOT}/tools/ane-metal"
GGML_MAP_DIR="${ROOT}/tools/ane-ggml-map"
INPROCESS_DIR="${ROOT}/tools/ane-inprocess"
PREFILL_DIR="${ROOT}/tools/ane-prefill"
OUT_DIR="${ROOT}/build/ane-probe-darwin/bin"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ane_probe_build: Darwin + Apple Silicon only" >&2
  exit 1
fi

if [[ ! -f "${ANE_REPO}/bridge/ane_bridge.h" ]]; then
  echo "ane_probe_build: missing ${ANE_REPO}/bridge — clone https://github.com/maderix/ane" >&2
  echo "  export ANE_REPO=/path/to/ane" >&2
  exit 1
fi

echo "== ane_probe_build: bridge @ ${ANE_REPO} =="
"${ROOT}/scripts/ane/ane_bridge_patch.sh"
make -C "${PROBE_DIR}" ANE_REPO="${ANE_REPO}"
make -C "${DRAFT_DIR}" ANE_REPO="${ANE_REPO}"
make -C "${IOSURFACE_DIR}" ANE_REPO="${ANE_REPO}"
make -C "${METAL_DIR}" ANE_REPO="${ANE_REPO}"
make -C "${INPROCESS_DIR}" ANE_REPO="${ANE_REPO}"
make -C "${GGML_MAP_DIR}"
make -C "${PREFILL_DIR}" ANE_REPO="${ANE_REPO}"

mkdir -p "${OUT_DIR}"
install -m 755 "${PROBE_DIR}/ane-probe" "${OUT_DIR}/ane-probe"
install -m 755 "${PROBE_DIR}/ane-matmul-bench" "${OUT_DIR}/ane-matmul-bench"
install -m 755 "${DRAFT_DIR}/ane-draft-bench" "${OUT_DIR}/ane-draft-bench"
install -m 755 "${IOSURFACE_DIR}/ane-iosurface-smoke" "${OUT_DIR}/ane-iosurface-smoke"
install -m 755 "${METAL_DIR}/ane-metal-handoff-smoke" "${OUT_DIR}/ane-metal-handoff-smoke"
install -m 755 "${METAL_DIR}/ane-draft-daemon" "${OUT_DIR}/ane-draft-daemon"
install -m 755 "${INPROCESS_DIR}/ane-inprocess-smoke" "${OUT_DIR}/ane-inprocess-smoke"
install -m 755 "${GGML_MAP_DIR}/ane-ggml-map-smoke" "${OUT_DIR}/ane-ggml-map-smoke"
install -m 755 "${PREFILL_DIR}/ane-prefill-bench" "${OUT_DIR}/ane-prefill-bench"
install -m 755 "${PREFILL_DIR}/metal-prefill-bench" "${OUT_DIR}/metal-prefill-bench"
install -m 755 "${PREFILL_DIR}/metal-prefill-mps-bench" "${OUT_DIR}/metal-prefill-mps-bench"
install -m 755 "${PREFILL_DIR}/ane-prefill-handoff-smoke" "${OUT_DIR}/ane-prefill-handoff-smoke"
install -m 755 "${ANE_REPO}/bridge/libane_bridge.dylib" "${OUT_DIR}/libane_bridge.dylib"
echo "== installed ${OUT_DIR}/{...,ane-prefill-handoff-smoke} + libane_bridge.dylib =="

if [[ "${RUN_SMOKE:-0}" == "1" ]]; then
  echo "== smoke: ane-probe =="
  "${OUT_DIR}/ane-probe"
  echo "== smoke: ane-matmul-bench --quick =="
  "${OUT_DIR}/ane-matmul-bench" --quick
  echo "== smoke: ane-draft-bench --quick =="
  "${OUT_DIR}/ane-draft-bench" --quick
  echo "== smoke: ane-iosurface-smoke --quick =="
  "${OUT_DIR}/ane-iosurface-smoke" --quick
  echo "== smoke: ane-metal-handoff-smoke --quick =="
  "${OUT_DIR}/ane-metal-handoff-smoke" --quick
  echo "== smoke: ane-prefill-bench --quick =="
  "${OUT_DIR}/ane-prefill-bench" --quick
  echo "== smoke: metal-prefill-mps-bench --quick =="
  "${OUT_DIR}/metal-prefill-mps-bench" --quick
  echo "== smoke: ane-prefill-handoff-smoke --quick =="
  "${OUT_DIR}/ane-prefill-handoff-smoke" --quick
fi
