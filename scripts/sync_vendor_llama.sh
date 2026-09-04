#!/usr/bin/env bash
# Sync vendor/llama-cpp-<pin> (llama.cpp + Ollama patches) into in-tree vendored trees.
#
# WHY: vendor/ is a gitignored workspace for `git am` / format-patch — not a second
# source of truth. In-tree ml/backend/ggml/ggml and llama/llama.cpp are what CGO builds.
# rsync --delete keeps them aligned with vendor, but excludes Ollama-only files that
# upstream does not contain (NVML/ROCm mem helpers, CGO wrappers, build-info).
#
# Usage:
#   ./scripts/sync_vendor_llama.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FETCH_HEAD="${FETCH_HEAD:-$(grep '^FETCH_HEAD=' "${ROOT}/Makefile.sync" | cut -d= -f2)}"
FETCH_REF="${FETCH_REF:-$(grep '^FETCH_REF=' "${ROOT}/Makefile.sync" | cut -d= -f2)}"
if [[ -f "${ROOT}/LLAMA_CPP_COMMIT" ]]; then
  _commit="$(tr -d '[:space:]' < "${ROOT}/LLAMA_CPP_COMMIT")"
  [[ -n "${_commit}" ]] && FETCH_REF="${_commit}"
fi
BUILD_NUMBER="${BUILD_NUMBER:-$(grep '^BUILD_NUMBER=' "${ROOT}/Makefile.sync" | cut -d= -f2)}"
VENDOR="${ROOT}/vendor/llama-cpp-${FETCH_HEAD}"

if [[ ! -d "${VENDOR}/.git" ]]; then
  echo "error: missing ${VENDOR}; clone and apply patches first:" >&2
  echo "  ./scripts/rebase_vendor_unified.sh" >&2
  echo "  or: make -f Makefile.sync clean apply-patches" >&2
  exit 1
fi

# WHY: bare FETCH_REF in vendor means Ollama patches were never applied (or were
# wiped by `make checkout` / `make clean`). Rsyncing then ships upstream-only ggml
# while build-info.cpp still reports the pin — CGO misses dev_reset, no-alloc sched, kv-ext.
PATCH_COUNT="$(git -C "${VENDOR}" rev-list --count "${FETCH_REF}..HEAD" 2>/dev/null || echo 0)"
if [[ "${PATCH_COUNT}" -eq 0 ]]; then
  echo "error: ${VENDOR} is at bare ref ${FETCH_REF} with no Ollama patch commits" >&2
  echo "  ./scripts/rebase_vendor_unified.sh" >&2
  exit 1
fi
echo ">>> vendor HEAD +${PATCH_COUNT} commits on ${FETCH_REF}" >&2

echo ">>> sync ggml → ml/backend/ggml/ggml" >&2
# Preserve Ollama-only mem_nvml.cpp / mem_hip.cpp (accurate CUDA/ROCm VRAM).
rsync -a --delete \
  --exclude '.rsync-filter' \
  --exclude '*.go' \
  --exclude '*-embed.*' \
  --exclude 'ollama-*' \
  --exclude 'src/mem_nvml.cpp' \
  --exclude 'src/mem_hip.cpp' \
  "${VENDOR}/ggml/" "${ROOT}/ml/backend/ggml/ggml/"

echo ">>> sync llama.cpp subset → llama/llama.cpp" >&2
for dir in common include src tools vendor; do
  if [[ -d "${VENDOR}/${dir}" ]]; then
    # Preserve CGO-only files: build-info, jinja/httplib wrappers; skip mtmd CLI mains.
    rsync -a --delete \
      --exclude '.rsync-filter' \
      --exclude '*.go' \
      --exclude 'build-info.cpp' \
      --exclude 'license.cpp' \
      --exclude 'jinja_wrap.cpp' \
      --exclude 'httplib_wrap.cpp' \
      --exclude 'tools/mtmd/mtmd-cli.cpp' \
      --exclude 'tools/mtmd/deprecation-warning.cpp' \
      --exclude 'ane_draft_hook.*' \
      --exclude 'ane_draft_session.*' \
      --exclude 'ane_iosurface_map.h' \
      "${VENDOR}/${dir}/" "${ROOT}/llama/llama.cpp/${dir}/"
  fi
done
[[ -f "${VENDOR}/LICENSE" ]] && cp "${VENDOR}/LICENSE" "${ROOT}/llama/llama.cpp/LICENSE"

# CGO compiles all *.cpp under tools/mtmd/ — drop CLI mains not used by Go.
rm -f \
  "${ROOT}/llama/llama.cpp/tools/mtmd/mtmd-cli.cpp" \
  "${ROOT}/llama/llama.cpp/tools/mtmd/deprecation-warning.cpp" \
  "${ROOT}/llama/llama.cpp/tools/mtmd/debug/mtmd-debug.cpp"

echo ">>> regenerate build-info.cpp" >&2
BUILD_NUMBER="${BUILD_NUMBER:-${FETCH_HEAD#b}}"
sed -e "s|@FETCH_HEAD@|${FETCH_HEAD}|" \
    -e "s|@LLAMA_BUILD_NUMBER@|${BUILD_NUMBER}|" \
    -e 's|@BUILD_COMPILER@||' \
    -e 's|@BUILD_TARGET@||' \
    "${ROOT}/llama/build-info.cpp.in" >"${ROOT}/llama/build-info.cpp"

echo ">>> regenerate metal embed" >&2
cd "${ROOT}"
bash "${ROOT}/scripts/build/gen_ggml_metal_embed.sh"

echo ">>> restore ANE hook in-tree (vendor lacks 0018 until git am)" >&2
"${ROOT}/scripts/restore_ane_hook_intree.sh"

echo ">>> OK: vendored trees synced from ${VENDOR}" >&2
