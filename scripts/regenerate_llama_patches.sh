#!/usr/bin/env bash
# Regenerate llama/patches/*.patch from current vendored trees on a fresh b9509 clone.
#
# Reads FETCH_HEAD from Makefile.sync (default: b9509).
# Backs up existing patches to llama/patches.pre-<date>/.
#
# Usage:
#   ./scripts/regenerate_llama_patches.sh
#   FETCH_HEAD=b9509 ./scripts/regenerate_llama_patches.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FETCH_HEAD="${FETCH_HEAD:-$(grep '^FETCH_HEAD=' "${ROOT}/Makefile.sync" | cut -d= -f2)}"
VENDOR="${ROOT}/vendor/llama-cpp-${FETCH_HEAD}"
GGML_V="${ROOT}/ml/backend/ggml/ggml"
LLAMA_V="${ROOT}/llama/llama.cpp"
PATCH_DIR="${ROOT}/llama/patches"
ARCHIVE="${PATCH_DIR}.pre-$(date +%Y%m%d-%H%M%S)"

if [[ ! -d "${VENDOR}/.git" ]]; then
  echo ">>> cloning ${VENDOR}" >&2
  mkdir -p "${ROOT}/vendor"
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git "${VENDOR}"
fi

git -C "${VENDOR}" fetch --tags origin
git -C "${VENDOR}" checkout -f "${FETCH_HEAD}"
git -C "${VENDOR}" clean -fdx

echo ">>> overlay ggml from ${GGML_V}" >&2
rsync -a \
  --exclude '.rsync-filter' \
  --exclude '*.go' \
  --exclude '*-embed.*' \
  --exclude 'ollama-*' \
  "${GGML_V}/" "${VENDOR}/ggml/"

echo ">>> overlay llama.cpp subset from ${LLAMA_V}" >&2
for dir in common include src tools vendor; do
  if [[ -d "${LLAMA_V}/${dir}" ]]; then
    rsync -a \
      --exclude '.rsync-filter' \
      --exclude '*.go' \
      "${LLAMA_V}/${dir}/" "${VENDOR}/${dir}/"
  fi
done
[[ -f "${LLAMA_V}/LICENSE" ]] && cp "${LLAMA_V}/LICENSE" "${VENDOR}/LICENSE"

git -C "${VENDOR}" add ggml/
git -C "${VENDOR}" commit -m "ollama: ggml vendor overlay for in-process runner (${FETCH_HEAD} base)"

git -C "${VENDOR}" add common/ include/ src/ tools/ vendor/ LICENSE
git -C "${VENDOR}" commit -m "ollama: llama.cpp vendor overlay for CGO runner (${FETCH_HEAD} base)"

cp -a "${PATCH_DIR}" "${ARCHIVE}"
echo ">>> backed up patches to ${ARCHIVE}" >&2

rm -f "${PATCH_DIR}"/*.patch "${PATCH_DIR}"/.*.patched 2>/dev/null || true
git -C "${VENDOR}" format-patch \
  --no-signature \
  --no-numbered \
  --zero-commit \
  -o "${PATCH_DIR}" \
  "${FETCH_HEAD}"

echo ">>> verify apply" >&2
git -C "${VENDOR}" checkout -f "${FETCH_HEAD}"
git -C "${VENDOR}" am --abort 2>/dev/null || true
git -C "${VENDOR}" am -3 "${PATCH_DIR}"/*.patch

echo ">>> OK: $(ls "${PATCH_DIR}"/*.patch | wc -l | tr -d ' ') patch(es) for ${FETCH_HEAD}" >&2
ls -lh "${PATCH_DIR}"/*.patch
