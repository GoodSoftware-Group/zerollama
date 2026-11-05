#!/usr/bin/env bash
# Download ONNX Runtime for GLiNER.cpp builds (GL0 CPU / GL4 CUDA).
# WHY not system packages: version pin + portable include/lib layout.
#
# Usage:
#   ./scripts/vendor/ensure_onnxruntime.sh              # CPU (default)
#   ONNXRUNTIME_VARIANT=gpu ./scripts/vendor/ensure_onnxruntime.sh
#
# Env:
#   ONNXRUNTIME_ROOTDIR  — install root (default vendor/onnxruntime-<arch>-<ver>)
#   ONNXRUNTIME_VERSION  — default 1.19.2
#   ONNXRUNTIME_VARIANT  — cpu (default) | gpu
#   ONNXRUNTIME_ARCH     — default linux-x64 (CPU) or linux-x64-gpu (when VARIANT=gpu)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VER="${ONNXRUNTIME_VERSION:-1.19.2}"
VARIANT="${ONNXRUNTIME_VARIANT:-cpu}"
case "${VARIANT}" in
  cpu|CPU)
    ARCH="${ONNXRUNTIME_ARCH:-linux-x64}"
    ;;
  gpu|GPU|cuda|CUDA)
    ARCH="${ONNXRUNTIME_ARCH:-linux-x64-gpu}"
    VARIANT=gpu
    ;;
  *)
    echo "error: ONNXRUNTIME_VARIANT must be cpu|gpu (got ${VARIANT})" >&2
    exit 1
    ;;
esac
NAME="onnxruntime-${ARCH}-${VER}"
DEST="${ONNXRUNTIME_ROOTDIR:-${ROOT}/vendor/${NAME}}"
URL="${ONNXRUNTIME_URL:-https://github.com/microsoft/onnxruntime/releases/download/v${VER}/${NAME}.tgz}"

if [[ -f "${DEST}/lib/libonnxruntime.so" || -f "${DEST}/lib/libonnxruntime.dylib" ]]; then
  echo ">>> ORT (${VARIANT}) already at ${DEST}" >&2
  echo "${DEST}"
  exit 0
fi

mkdir -p "$(dirname "${DEST}")"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

echo ">>> downloading ${URL}" >&2
curl -fsSL "${URL}" -o "${TMP}/ort.tgz"
tar -xzf "${TMP}/ort.tgz" -C "${TMP}"
EXTRACTED="$(find "${TMP}" -maxdepth 1 -type d -name 'onnxruntime-*' | head -1)"
[[ -n "${EXTRACTED}" ]] || { echo "error: unpack failed" >&2; exit 1; }
rm -rf "${DEST}"
mv "${EXTRACTED}" "${DEST}"

echo ">>> OK ${DEST} (variant=${VARIANT})" >&2
echo "${DEST}"
