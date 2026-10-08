#!/usr/bin/env bash
# Write CGO version headers that CMake normally emits into the build tree.
#
# WHY: ggml.c / llama.cpp #include *-version.h; Go CGO compiles from the source
# tree. Without these files, `go test ./llm` fails with "No such file".
#
# Idempotent. Does not start serve.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIN="$(tr -d '[:space:]' < "${ROOT}/LLAMA_CPP_VERSION" 2>/dev/null || echo none)"
VENDOR_HEAD="$(tr -d '[:space:]' < "${ROOT}/LLAMA_CPP_VENDOR_HEAD" 2>/dev/null || true)"
COMMIT="${VENDOR_HEAD:0:9}"
[[ -n "${COMMIT}" ]] || COMMIT="unknown"

ensure_header() {
  local out="$1"
  local in_file="$2"
  shift 2
  local cand
  for cand in "$@"; do
    if [[ -f "${cand}" ]]; then
      cp -f "${cand}" "${out}"
      echo "OK: ${out} ← ${cand}"
      return 0
    fi
  done
  if [[ ! -f "${in_file}" ]]; then
    echo "error: missing ${in_file} and no vendor build header" >&2
    return 1
  fi
  sed -e "s/@GGML_VERSION@/${PIN}/g" \
      -e "s/@GGML_BUILD_COMMIT@/${COMMIT}/g" \
      -e "s/@LLAMA_VERSION@/${PIN}/g" \
      -e "s/@LLAMA_BUILD_COMMIT@/${COMMIT}/g" \
      "${in_file}" > "${out}.tmp"
  mv "${out}.tmp" "${out}"
  echo "OK: wrote ${out} from ${in_file}"
}

ensure_header \
  "${ROOT}/ml/backend/ggml/ggml/src/ggml-version.h" \
  "${ROOT}/ml/backend/ggml/ggml/src/ggml-version.h.in" \
  "${ROOT}/vendor/llama-cpp-${PIN}/build/ggml/src/ggml-version.h" \
  "${ROOT}/vendor/llama-cpp-b11351/build/ggml/src/ggml-version.h"

ensure_header \
  "${ROOT}/llama/llama.cpp/src/llama-version.h" \
  "${ROOT}/llama/llama.cpp/src/llama-version.h.in" \
  "${ROOT}/vendor/llama-cpp-${PIN}/build/src/llama-version.h" \
  "${ROOT}/vendor/llama-cpp-b11351/build/src/llama-version.h"
