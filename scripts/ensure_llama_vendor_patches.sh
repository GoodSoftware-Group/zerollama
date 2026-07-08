#!/usr/bin/env bash
# Ensure unified vendor/llama-cpp-* has Ollama patch commits AND compat loader hooks.
#
# WHY: stage_llama_compat_for_vendor.sh only links compat .cpp into CMake — patch
# 0015 wires translate_metadata() into llama-model-loader.cpp. Stale
# llama/patches/.*.patched markers can skip `make apply-patches` while vendor sits
# on bare FETCH_REF, producing a libllama with dead compat code (qwen35moe 500).
#
# Usage:
#   ./scripts/ensure_llama_vendor_patches.sh
#   ./scripts/ensure_llama_vendor_patches.sh /path/to/vendor/llama-cpp-c84b3020
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FETCH_HEAD="$(grep '^FETCH_HEAD=' "${ROOT}/Makefile.sync" | cut -d= -f2)"
DEFAULT_VENDOR="${ROOT}/vendor/llama-cpp-${FETCH_HEAD}"
VENDOR="${1:-${DEFAULT_VENDOR}}"

vendor_is_unified() {
  [[ "${VENDOR}" == "${ROOT}/vendor/llama-cpp-"* ]]
}

vendor_has_compat_hooks() {
  [[ -f "${VENDOR}/src/llama-model-loader.cpp" ]] \
    && grep -q 'llama_ollama_compat::translate_metadata' "${VENDOR}/src/llama-model-loader.cpp" \
    && [[ -f "${VENDOR}/tools/mtmd/clip.cpp" ]] \
    && grep -q 'llama_ollama_compat::translate_clip_metadata' "${VENDOR}/tools/mtmd/clip.cpp"
}

vendor_patch_commit_count() {
  local fetch_ref
  fetch_ref="$(grep '^FETCH_REF=' "${ROOT}/Makefile.sync" | cut -d= -f2)"
  if [[ -f "${ROOT}/LLAMA_CPP_COMMIT" ]]; then
    local _c
    _c="$(tr -d '[:space:]' < "${ROOT}/LLAMA_CPP_COMMIT")"
    [[ -n "${_c}" ]] && fetch_ref="${_c}"
  fi
  git -C "${VENDOR}" rev-list --count "${fetch_ref}..HEAD" 2>/dev/null || echo 0
}

# Untracked copies from sync_ane_hook_to_llama_cpp.sh (or manual staging) block git am
# for patch 0018. stage_llama_compat symlinks block a clean re-apply of patch 0017.
scrub_untracked_patch_conflicts() {
  local rel
  for rel in \
    common/ane_draft_hook.cpp \
    common/ane_draft_hook.h \
    common/ane_draft_session.h \
    common/ane_draft_session.mm \
    common/ane_draft_session_stub.cpp \
    common/ane_iosurface_map.h \
    src/llama-ext.h \
    src/ollama-compat; do
    if [[ -e "${VENDOR}/${rel}" ]] \
      && ! git -C "${VENDOR}" ls-files --error-unmatch "${rel}" &>/dev/null; then
      echo ">>> removing untracked ${rel} (conflicts with llama/patches git am)" >&2
      rm -rf "${VENDOR}/${rel}"
    fi
  done
}

apply_vendor_patches() {
  echo ">>> resetting vendor to pin and applying llama/patches/*.patch" >&2
  if [[ -x "${ROOT}/scripts/apply_llama_vendor_patches.sh" ]]; then
    "${ROOT}/scripts/apply_llama_vendor_patches.sh" "${VENDOR}"
    return 0
  fi
  git -C "${VENDOR}" am --abort &>/dev/null || true
  scrub_untracked_patch_conflicts
  rm -f "${ROOT}"/llama/patches/.*.patched
  make -C "${ROOT}" -f Makefile.sync clean apply-patches
  "${ROOT}/scripts/stage_llama_compat_for_vendor.sh" "${VENDOR}"
  "${ROOT}/scripts/stage_llama_ext_b8_for_vendor.sh" "${VENDOR}"
  "${ROOT}/scripts/stage_llama_kv_ext_for_vendor.sh" "${VENDOR}"
  if [[ -f "${VENDOR}/common/ane_draft_hook.cpp" ]]; then
    "${ROOT}/scripts/stage_ane_hook_to_tree.sh" "${VENDOR}"
  fi
}

if [[ ! -f "${VENDOR}/CMakeLists.txt" ]]; then
  echo "error: vendor missing at ${VENDOR}" >&2
  echo "  run: ./scripts/rebase_vendor_unified.sh --apply --sync" >&2
  exit 1
fi

if vendor_has_compat_hooks; then
  count="$(vendor_patch_commit_count)"
  echo ">>> Ollama compat loader hooks present (${count} patch commits on vendor)" >&2
  # Keep compat sources staged even when patches already applied.
  if vendor_is_unified; then
    "${ROOT}/scripts/stage_llama_compat_for_vendor.sh" "${VENDOR}"
    "${ROOT}/scripts/stage_llama_ext_b8_for_vendor.sh" "${VENDOR}"
    "${ROOT}/scripts/stage_llama_kv_ext_for_vendor.sh" "${VENDOR}"
    if [[ -f "${VENDOR}/common/ane_draft_hook.cpp" ]]; then
      "${ROOT}/scripts/stage_ane_hook_to_tree.sh" "${VENDOR}"
    fi
  fi
  exit 0
fi

if ! vendor_is_unified; then
  echo "error: Ollama compat loader hooks missing in ${VENDOR}" >&2
  echo "  unified vendor required for auto-apply; use vendor/llama-cpp-${FETCH_HEAD}" >&2
  exit 1
fi

echo ">>> Ollama compat loader hooks missing in ${VENDOR}" >&2
apply_vendor_patches

if ! vendor_has_compat_hooks; then
  echo "error: patch apply finished but translate_metadata still missing" >&2
  echo "  resolve conflicts: git -C ${VENDOR} am --continue" >&2
  exit 1
fi

count="$(vendor_patch_commit_count)"
echo ">>> Ollama patches applied (${count} commits); compat hooks verified" >&2
"${ROOT}/scripts/stage_llama_kv_ext_for_vendor.sh" "${VENDOR}"
