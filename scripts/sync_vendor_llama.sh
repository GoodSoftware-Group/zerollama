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
VENDOR="${ROOT}/vendor/llama-cpp-${FETCH_HEAD}"

if [[ ! -d "${VENDOR}/.git" ]]; then
  echo "error: missing ${VENDOR}; clone and apply patches first:" >&2
  echo "  git clone https://github.com/ggml-org/llama.cpp.git ${VENDOR}" >&2
  echo "  make -f Makefile.sync clean apply-patches" >&2
  exit 1
fi

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
      --exclude 'jinja_wrap.cpp' \
      --exclude 'httplib_wrap.cpp' \
      --exclude 'tools/mtmd/mtmd-cli.cpp' \
      --exclude 'tools/mtmd/deprecation-warning.cpp' \
      "${VENDOR}/${dir}/" "${ROOT}/llama/llama.cpp/${dir}/"
  fi
done
[[ -f "${VENDOR}/LICENSE" ]] && cp "${VENDOR}/LICENSE" "${ROOT}/llama/llama.cpp/LICENSE"

# CGO compiles all *.cpp under tools/mtmd/ — drop CLI mains not used by Go.
rm -f \
  "${ROOT}/llama/llama.cpp/tools/mtmd/mtmd-cli.cpp" \
  "${ROOT}/llama/llama.cpp/tools/mtmd/deprecation-warning.cpp" \
  "${ROOT}/llama/llama.cpp/tools/mtmd/debug/mtmd-debug.cpp"

echo ">>> regenerate metal embed" >&2
cd "${ROOT}"
GOFLAGS=-mod=mod go generate ./ml/backend/ggml/ggml/src

echo ">>> OK: vendored trees synced from ${VENDOR}" >&2
