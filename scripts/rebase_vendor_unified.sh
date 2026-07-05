#!/usr/bin/env bash
# Rebase in-process vendor onto unified elizaOS/llama.cpp @ LLAMA_CPP_COMMIT + Ollama patches.
#
# WHY: runtime llama-server and Go ggml should share one source tree (eliza base +
# llama/patches/). This script materializes vendor/llama-cpp-<short> and applies the
# patch series. On pin bumps, resolve conflicts per docs/ggml-b9509-migration.md.
#
# Usage:
#   ./scripts/rebase_vendor_unified.sh          # clone/checkout only (patches pre-applied)
#   ./scripts/rebase_vendor_unified.sh --sync   # rsync vendor → in-tree ggml + llama.cpp
#   ./scripts/rebase_vendor_unified.sh --apply  # git am all patches (fresh vendor reset)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FETCH_HEAD="$(grep '^FETCH_HEAD=' "${ROOT}/Makefile.sync" | cut -d= -f2)"
FETCH_REF="$(grep '^FETCH_REF=' "${ROOT}/Makefile.sync" | cut -d= -f2)"
if [[ -f "${ROOT}/LLAMA_CPP_COMMIT" ]]; then
  _c="$(tr -d '[:space:]' < "${ROOT}/LLAMA_CPP_COMMIT")"
  [[ -n "${_c}" ]] && FETCH_REF="${_c}"
fi
UPSTREAM="$(grep '^UPSTREAM=' "${ROOT}/Makefile.sync" | cut -d= -f2)"
VENDOR="${ROOT}/vendor/llama-cpp-${FETCH_HEAD}"

DO_SYNC=0
DO_APPLY=0
for arg in "$@"; do
  case "${arg}" in
    --sync) DO_SYNC=1 ;;
    --apply) DO_APPLY=1 ;;
  esac
done

if [[ ! -d "${VENDOR}/.git" ]]; then
  echo ">>> cloning ${UPSTREAM} → ${VENDOR}" >&2
  git clone "${UPSTREAM}" "${VENDOR}"
fi

echo ">>> checkout ${FETCH_REF}" >&2
git -C "${VENDOR}" fetch origin "${FETCH_REF}" --depth 1 2>/dev/null || \
  git -C "${VENDOR}" fetch origin --tags --force 2>/dev/null || true
git -C "${VENDOR}" checkout --force "${FETCH_REF}"

if [[ "${DO_APPLY}" == "1" ]]; then
  echo ">>> applying llama/patches/*.patch" >&2
  rm -f "${ROOT}"/llama/patches/.*.patched
  make -C "${ROOT}" -f Makefile.sync apply-patches
else
  "${ROOT}/scripts/ensure_llama_vendor_patches.sh" "${VENDOR}"
fi

PATCH_COUNT="$(git -C "${VENDOR}" rev-list --count "${FETCH_REF}..HEAD" 2>/dev/null || echo 0)"
echo ">>> vendor @ $(git -C "${VENDOR}" rev-parse --short HEAD) (+${PATCH_COUNT} Ollama patches on ${FETCH_REF:0:12})" >&2

if [[ "${PATCH_COUNT}" -eq 0 ]]; then
  echo "error: no patch commits on vendor — run with --apply or ./scripts/ensure_llama_vendor_patches.sh" >&2
  exit 1
fi

if [[ "${DO_SYNC}" == "1" || "${DO_APPLY}" == "1" ]]; then
  "${ROOT}/scripts/sync_vendor_llama.sh"
fi

echo "PASS: rebase_vendor_unified (${VENDOR})"
