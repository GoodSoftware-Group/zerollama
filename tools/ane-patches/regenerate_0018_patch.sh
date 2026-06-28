#!/usr/bin/env bash
# Regenerate llama/patches/0018-ollama-ane-draft-hook.patch from vendor + sync_ane_hook.
#
# Usage (after vendor is at FETCH_REF + Ollama patches 0001–0017):
#   ./tools/ane-patches/regenerate_0018_patch.sh
#
# Commits staged ANE hook on vendor, then format-patch into llama/patches/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FETCH_HEAD="$(grep '^FETCH_HEAD=' "${ROOT}/Makefile.sync" | cut -d= -f2)"
VENDOR="${ROOT}/vendor/llama-cpp-${FETCH_HEAD}"
OUT="${ROOT}/llama/patches/0018-ollama-ane-draft-hook.patch"

if [[ ! -d "${VENDOR}/.git" ]]; then
  echo "regenerate_0018: missing ${VENDOR}; run ./scripts/rebase_vendor_unified.sh first" >&2
  exit 1
fi

echo "== regenerate_0018: stage ANE hook on ${VENDOR} =="
LLAMA_CPP_ROOT="${VENDOR}" "${ROOT}/scripts/sync_ane_hook_to_llama_cpp.sh"

# IOSurface ggml changes are part of 0018 (same as sync applies to sibling).
python3 "${ROOT}/tools/ane-patches/apply_iosurface_sibling.py" "${VENDOR}"

if git -C "${VENDOR}" diff --quiet && git -C "${VENDOR}" diff --cached --quiet; then
  echo "regenerate_0018: no changes — patch already applied on vendor?" >&2
  exit 0
fi

git -C "${VENDOR}" add \
  common/ane_draft_hook.h common/ane_draft_hook.cpp \
  common/ane_draft_session.h common/ane_draft_session.mm \
  common/ane_draft_session_stub.cpp common/ane_iosurface_map.h \
  common/speculative.cpp common/CMakeLists.txt \
  ggml/include/ggml-metal.h \
  ggml/src/ggml-metal/CMakeLists.txt \
  ggml/src/ggml-metal/ggml-metal-device.h \
  ggml/src/ggml-metal/ggml-metal-device.m \
  ggml/src/ggml-metal/ggml-metal.cpp 2>/dev/null || true

git -C "${VENDOR}" commit -m "$(cat <<'EOF'
ollama: ANE in-process dflash draft hook (B1–B7 lab)

In-process ANE session, ggml IOSurface handoff, sidecar conv proxy,
optional B7 tied-embed drive (shadow/force). Lab port only.

Co-authored-by: Cursor <cursoragent@cursor.com>
EOF
)"

make -C "${ROOT}" -f Makefile.sync format-patches
shopt -s nullglob
for p in "${ROOT}"/llama/patches/0018-*.patch; do
  mv -f "${p}" "${OUT}"
  break
done

echo "== regenerate_0018: wrote ${OUT} =="
echo "Next: make -f Makefile.sync clean apply-patches && ./scripts/sync_vendor_llama.sh"
