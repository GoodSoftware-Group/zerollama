#!/usr/bin/env bash
# GL0: ensure vendor + deps; optional build if cargo+network available.
# Always exits 0 on skip (missing cargo); fails only on hard script errors.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
bash -n "${ROOT}/scripts/vendor/ensure_gliner_cpp.sh"
bash -n "${ROOT}/scripts/vendor/ensure_onnxruntime.sh"
bash -n "${ROOT}/scripts/vendor/ensure_gliner_server_deps.sh"
bash -n "${ROOT}/scripts/build/build_gliner_server.sh"

echo ">>> ensure third_party headers" >&2
bash "${ROOT}/scripts/vendor/ensure_gliner_server_deps.sh" >/dev/null

if ! command -v cargo >/dev/null 2>&1; then
  echo "skip: cargo not installed — vendor scripts OK; build later" >&2
  echo "GLINER_CPP_SMOKE_SKIP_NO_CARGO"
  exit 0
fi

echo ">>> ensure GLiNER.cpp + ORT (may download)" >&2
GLINER_ROOT="$(bash "${ROOT}/scripts/vendor/ensure_gliner_cpp.sh" | tail -1)"
ORT_ROOT="$(bash "${ROOT}/scripts/vendor/ensure_onnxruntime.sh" | tail -1)"
echo "GLiNER=${GLINER_ROOT}"
echo "ORT=${ORT_ROOT}"
[[ -f "${GLINER_ROOT}/CMakeLists.txt" ]]
[[ -d "${ORT_ROOT}/include" || -d "${ORT_ROOT}/include/onnxruntime" ]]
echo "GLINER_CPP_SMOKE_OK"
