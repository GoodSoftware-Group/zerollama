#!/usr/bin/env bash
# Commit eliza CUDA fork wiring on vendor and export llama/patches/0020-*.patch.
#
# Prerequisite: ./scripts/vendor/apply_llama_vendor_patches.sh && sibling ../llama.cpp
#
# Usage:
#   ./scripts/vendor/create_cuda_fork_patch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FETCH_HEAD="$(grep '^FETCH_HEAD=' "${ROOT}/Makefile.sync" 2>/dev/null | cut -d= -f2 || echo c84b3020)"
FETCH_REF="$(grep '^FETCH_REF=' "${ROOT}/Makefile.sync" 2>/dev/null | cut -d= -f2)"
VENDOR="${ROOT}/vendor/llama-cpp-${FETCH_HEAD}"
PATCH_DIR="${ROOT}/llama/patches"
PATCH_OUT="${PATCH_DIR}/0020-ollama-cuda-fork-kv-fused-attn.patch"
SUBJECT="ollama: CUDA fork SET_ROWS + fused QJL attn (eliza c84b302+)"

GIT=(git -C "${VENDOR}" -c user.name=zerollama -c user.email=zerollama@local)

"${ROOT}/scripts/vendor/sync_eliza_cuda_fork_to_vendor.sh" "${VENDOR}"

if "${GIT[@]}" log --format=%s "${FETCH_REF}..HEAD" 2>/dev/null | grep -Fq "${SUBJECT}"; then
  echo ">>> CUDA fork patch already committed on vendor"
else
  "${GIT[@]}" add \
    ggml/CMakeLists.txt \
    ggml/src/ggml.c \
    ggml/src/ggml-cuda/set-rows.cu \
    ggml/src/ggml-cuda/qjl.cu \
    ggml/src/ggml-cuda/qjl.cuh \
    ggml/src/ggml-cuda/ggml-cuda.cu \
    ggml/src/ggml-cuda/fused-attn-qjl-tbq.cu \
    ggml/src/ggml-cuda/fused-attn-qjl-polar.cu \
    ggml/src/ggml-cuda/fused-attn.cu \
    ggml/src/ggml-cuda/fused-attn.cuh \
    ggml/src/ggml-cuda/polar-set-rows.cuh \
    ggml/src/ggml-cuda/qjl-set-rows.cuh \
    src/llama-graph.cpp
  if "${GIT[@]}" diff --cached --quiet; then
    echo ">>> no CUDA fork diff vs vendor HEAD (already synced?)"
  else
    "${GIT[@]}" commit -m "${SUBJECT}"
    echo ">>> committed CUDA fork on vendor"
  fi
fi

head="$("${GIT[@]}" rev-parse HEAD)"
"${GIT[@]}" format-patch -1 HEAD \
  --stdout \
  --no-signature \
  --zero-commit \
  > "${PATCH_OUT}.tmp"
mv "${PATCH_OUT}.tmp" "${PATCH_OUT}"
touch "${PATCH_DIR}/.0020-ollama-cuda-fork-kv-fused-attn.patched"

echo ">>> wrote ${PATCH_OUT}"
wc -l "${PATCH_OUT}"
