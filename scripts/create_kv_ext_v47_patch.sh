#!/usr/bin/env bash
# Commit in-tree llama-kv-ext (Phase 15 v33–v47) on vendor and export 0019 patch.
#
# Base: vendor HEAD when 0019 subject missing, else parent of 0019 commit.
# Source of truth: llama/llama.cpp/{include,src}/llama-*-kv-ext*
#
# Usage:
#   ./scripts/create_kv_ext_v47_patch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FETCH_HEAD="$(grep '^FETCH_HEAD=' "${ROOT}/Makefile.sync" 2>/dev/null | cut -d= -f2 || echo c84b3020)"
FETCH_REF="$(grep '^FETCH_REF=' "${ROOT}/Makefile.sync" 2>/dev/null | cut -d= -f2)"
if [[ -f "${ROOT}/LLAMA_CPP_COMMIT" ]]; then
  _c="$(tr -d '[:space:]' < "${ROOT}/LLAMA_CPP_COMMIT")"
  [[ -n "${_c}" ]] && FETCH_REF="${_c}"
fi
VENDOR="${ROOT}/vendor/llama-cpp-${FETCH_HEAD}"
PATCH_DIR="${ROOT}/llama/patches"
PATCH_OUT="${PATCH_DIR}/0019-ollama-llama-kv-ext-external-buffer-alias-v47.patch"
SUBJECT="ollama: llama-kv-ext external buffer alias probe v47"
INTREE="${ROOT}/llama/llama.cpp"
WT="${VENDOR}/.zerollama-0019-worktree"

GIT=(git -C "${VENDOR}" -c user.name=zerollama -c user.email=zerollama@local)

if [[ ! -d "${INTREE}/include" ]]; then
  echo "error: in-tree kv-ext missing at ${INTREE}" >&2
  exit 1
fi

if "${GIT[@]}" log --format=%s "${FETCH_REF}..HEAD" 2>/dev/null | grep -Fq "${SUBJECT}"; then
  base="$("${GIT[@]}" rev-parse HEAD~1^{commit} 2>/dev/null || "${GIT[@]}" rev-parse HEAD)"
  while "${GIT[@]}" log --format=%s -1 "${base}" 2>/dev/null | grep -Fq "${SUBJECT}"; do
    base="$("${GIT[@]}" rev-parse "${base}^")"
  done
  echo ">>> 0019 already on vendor; exporting from $(git -C "${VENDOR}" rev-parse --short "${base}")..HEAD"
  "${GIT[@]}" format-patch -1 HEAD --stdout --no-signature --zero-commit > "${PATCH_OUT}.tmp"
  mv "${PATCH_OUT}.tmp" "${PATCH_OUT}"
  rm -f "${PATCH_DIR}/.0019-ollama-llama-kv-ext-external-buffer-alias-v47.patched"
  echo ">>> wrote ${PATCH_OUT}"
  wc -l "${PATCH_OUT}"
  exit 0
fi

# Parent of CUDA fork (0020) when present; else current HEAD.
base="$("${GIT[@]}" rev-parse HEAD)"
if "${GIT[@]}" log --format=%s -1 HEAD 2>/dev/null | grep -Fq 'CUDA fork SET_ROWS'; then
  base="$("${GIT[@]}" rev-parse HEAD~1)"
fi

rm -rf "${WT}"
"${GIT[@]}" worktree add -f "${WT}" "${base}" >/dev/null

_sync_cmake_kv_ext() {
  local cmake="${WT}/src/CMakeLists.txt"
  if ! grep -q 'LLAMA_KV_EXT_WRITABLE_PAGE_MAP' "${cmake}"; then
    cat >> "${cmake}" <<'EOF'

# zerollama Phase 15 v33: staging writable KV page-map (llama-kv-ext.h)
target_compile_definitions(llama PRIVATE LLAMA_KV_EXT_WRITABLE_PAGE_MAP=1)
EOF
  fi
  if ! grep -q 'LLAMA_KV_EXT_EXTERNAL_ALIAS' "${cmake}"; then
    cat >> "${cmake}" <<'EOF'
# zerollama Phase 15 v47: external buffer alias probe + validate (no tensor mutation)
target_compile_definitions(llama PRIVATE LLAMA_KV_EXT_EXTERNAL_ALIAS=1)
EOF
  fi
}

install -m 644 "${INTREE}/include/llama-kv-ext.h" "${WT}/include/llama-kv-ext.h"
install -m 644 "${INTREE}/src/llama-memory-kv-ext.cpp" "${WT}/src/llama-memory-kv-ext.cpp"
install -m 644 "${INTREE}/src/llama-kv-cache.h" "${WT}/src/llama-kv-cache.h"
_sync_cmake_kv_ext

WT_GIT=(git -C "${WT}" -c user.name=zerollama -c user.email=zerollama@local)
if "${WT_GIT[@]}" diff --quiet; then
  echo ">>> no kv-ext diff vs vendor base ${base:0:12}" >&2
  "${GIT[@]}" worktree remove -f "${WT}" 2>/dev/null || rm -rf "${WT}"
  exit 0
fi

"${WT_GIT[@]}" add \
  include/llama-kv-ext.h \
  src/llama-memory-kv-ext.cpp \
  src/llama-kv-cache.h \
  src/CMakeLists.txt
"${WT_GIT[@]}" commit -m "${SUBJECT}"
commit="$("${WT_GIT[@]}" rev-parse HEAD)"

"${GIT[@]}" format-patch -1 "${commit}" --stdout --no-signature --zero-commit > "${PATCH_OUT}.tmp"
mv "${PATCH_OUT}.tmp" "${PATCH_OUT}"
rm -f "${PATCH_DIR}/.0019-ollama-llama-kv-ext-external-buffer-alias-v47.patched"

"${GIT[@]}" worktree remove -f "${WT}" 2>/dev/null || rm -rf "${WT}"

echo ">>> wrote ${PATCH_OUT} (base ${base:0:12}..${commit:0:12})"
wc -l "${PATCH_OUT}"
