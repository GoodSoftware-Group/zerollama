#!/usr/bin/env bash
# Apply llama/patches/*.patch to vendor/llama-cpp-<pin> (Linux-safe).
#
# Skips Darwin-only 0018-ollama-ane-draft-hook.patch on Linux.
# Uses git -c user.* so bare-root hosts without global git identity still work.
#
# Usage:
#   ./scripts/apply_llama_vendor_patches.sh
#   ./scripts/apply_llama_vendor_patches.sh /path/to/vendor/llama-cpp-c84b3020
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FETCH_HEAD="$(grep '^FETCH_HEAD=' "${ROOT}/Makefile.sync" | cut -d= -f2)"
FETCH_REF="$(grep '^FETCH_REF=' "${ROOT}/Makefile.sync" | cut -d= -f2)"
if [[ -f "${ROOT}/LLAMA_CPP_COMMIT" ]]; then
  _c="$(tr -d '[:space:]' < "${ROOT}/LLAMA_CPP_COMMIT")"
  [[ -n "${_c}" ]] && FETCH_REF="${_c}"
fi
VENDOR="${1:-${ROOT}/vendor/llama-cpp-${FETCH_HEAD}}"
PATCH_DIR="${ROOT}/llama/patches"

GIT_AM=(git -C "${VENDOR}" -c user.name=zerollama -c user.email=zerollama@local)

if [[ ! -d "${VENDOR}/.git" ]]; then
  echo "error: vendor git tree missing at ${VENDOR}" >&2
  exit 1
fi

_is_linux_skip_patch() {
  local base="$1"
  if [[ "$(uname -s)" != Darwin && "${base}" == 0018-ollama-ane-draft-hook ]]; then
    return 0
  fi
  return 1
}

_vendor_has_patch_subject() {
  local subj="$1"
  git -C "${VENDOR}" log --format=%s "${FETCH_REF}..HEAD" 2>/dev/null | grep -Fq "${subj}"
}

_patch_subject() {
  sed -n 's/^Subject: \[PATCH\] //p; s/^Subject: \[PATCH [0-9]*\/[0-9]*\] //p' "$1" | head -1
}

_vendor_tree_has() {
  local needle="$1"
  local file="$2"
  if [[ -f "${VENDOR}/${file}" ]] && grep -q "${needle}" "${VENDOR}/${file}" 2>/dev/null; then
    return 0
  fi
  "${GIT_AM[@]}" show "HEAD:${file}" 2>/dev/null | grep -q "${needle}"
}

_vendor_is_patched() {
  local count
  count="$("${GIT_AM[@]}" rev-list --count "${FETCH_REF}..HEAD" 2>/dev/null || echo 0)"
  [[ "${count}" -gt 0 ]] \
    && _vendor_tree_has 'llama_ollama_compat::translate_metadata' 'src/llama-model-loader.cpp' \
    && _vendor_tree_has 'kv/seq-copy' 'tools/server/server.cpp'
}

echo ">>> vendor checkout ${FETCH_REF:0:12}"
"${GIT_AM[@]}" fetch origin "${FETCH_REF}" --depth 1 2>/dev/null || true

if _vendor_is_patched; then
  existing_count="$("${GIT_AM[@]}" rev-list --count "${FETCH_REF}..HEAD")"
  echo ">>> vendor already patched (+${existing_count} commits); applying any missing patches only"
else
  "${GIT_AM[@]}" checkout --force "${FETCH_REF}"
  "${GIT_AM[@]}" am --abort 2>/dev/null || true
  rm -rf "${VENDOR}/.git/rebase-apply"
  "${GIT_AM[@]}" clean -fd
  existing_count=0
  rm -f "${PATCH_DIR}"/.*.patched
fi

if [[ "${existing_count}" -eq 0 ]]; then
  "${GIT_AM[@]}" am --abort 2>/dev/null || true
  rm -rf "${VENDOR}/.git/rebase-apply"
  for rel in \
    common/ane_draft_hook.cpp common/ane_draft_hook.h \
    common/ane_draft_session.h common/ane_draft_session.mm \
    common/ane_draft_session_stub.cpp common/ane_iosurface_map.h \
    src/llama-ext.h src/ollama-compat; do
    if [[ -e "${VENDOR}/${rel}" ]] \
      && ! git -C "${VENDOR}" ls-files --error-unmatch "${rel}" &>/dev/null; then
      rm -rf "${VENDOR}/${rel}"
    fi
  done
  rm -f "${PATCH_DIR}"/.*.patched
fi

applied=0
skipped=0
for patch in "${PATCH_DIR}"/*.patch; do
  [[ -f "${patch}" ]] || continue
  base="$(basename "${patch}" .patch)"
  subj="$(_patch_subject "${patch}")"

  if _is_linux_skip_patch "${base}"; then
    echo ">>> skip (Linux): ${base}"
    touch "${PATCH_DIR}/.${base}.patched"
    skipped=$((skipped + 1))
    continue
  fi

  if [[ -n "${subj}" ]] && _vendor_has_patch_subject "${subj}"; then
    echo ">>> already applied: ${base}"
    touch "${PATCH_DIR}/.${base}.patched"
    continue
  fi

  if [[ -f "${PATCH_DIR}/.${base}.patched" ]]; then
    echo ">>> skip (marked patched): ${base}"
    skipped=$((skipped + 1))
    continue
  fi

  # Optional/broken patch — never retry once skipped.
  if [[ "${base}" == 0019-ollama-llama-kv-ext-external-buffer-alias-v47 ]]; then
    echo ">>> skip optional: ${base}"
    touch "${PATCH_DIR}/.${base}.patched"
    skipped=$((skipped + 1))
    continue
  fi

  # Duplicate seq-copy patch (0018 Radix) — same subject as 0017 after 0017 applies.
  if [[ "${base}" == 0018-ollama-kv-seq-copy-endpoint-Radix-L3-v1 ]]; then
    if grep -q 'kv/seq-copy' "${VENDOR}/tools/server/server.cpp" 2>/dev/null; then
      echo ">>> skip duplicate: ${base} (/kv/seq-copy already present)"
      touch "${PATCH_DIR}/.${base}.patched"
      skipped=$((skipped + 1))
      continue
    fi
  fi

  echo ">>> am: ${base}"
  if "${GIT_AM[@]}" am -3 "${patch}"; then
    touch "${PATCH_DIR}/.${base}.patched"
    applied=$((applied + 1))
  elif [[ "${base}" == 0019-ollama-llama-kv-ext-external-buffer-alias-v47 ]]; then
    echo ">>> warn: optional patch failed (skipping): ${base}" >&2
    "${GIT_AM[@]}" am --abort 2>/dev/null || true
    rm -rf "${VENDOR}/.git/rebase-apply"
    touch "${PATCH_DIR}/.${base}.patched"
    skipped=$((skipped + 1))
  else
    echo "error: git am failed on ${patch}" >&2
    echo "  resolve in ${VENDOR} then: git am --continue" >&2
    exit 1
  fi
done

count="$("${GIT_AM[@]}" rev-list --count "${FETCH_REF}..HEAD" 2>/dev/null || echo 0)"
echo ">>> OK: vendor +${count} commits (applied ${applied}, skipped ${skipped})"

if ! grep -q 'llama_ollama_compat::translate_metadata' "${VENDOR}/src/llama-model-loader.cpp"; then
  echo "error: vendor missing Ollama compat hooks after patch apply" >&2
  exit 1
fi
if ! grep -q 'kv/seq-copy' "${VENDOR}/tools/server/server.cpp"; then
  echo "error: vendor missing /kv/seq-copy route" >&2
  exit 1
fi

"${ROOT}/scripts/stage_llama_compat_for_vendor.sh" "${VENDOR}" 2>/dev/null || true
"${ROOT}/scripts/stage_llama_ext_b8_for_vendor.sh" "${VENDOR}" 2>/dev/null || true
"${ROOT}/scripts/stage_llama_kv_ext_for_vendor.sh" "${VENDOR}" 2>/dev/null || true
