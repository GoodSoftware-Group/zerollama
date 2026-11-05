#!/usr/bin/env bash
# Ensure sibling Knowledgator/GLiNER.cpp under vendor/.
# Does NOT touch llama.cpp / prod llama-server.
#
# WHY sibling: ORT + tokenizers must not risk prod chat/Laya/CLM (Finding 2).
#
# Usage:
#   ./scripts/vendor/ensure_gliner_cpp.sh
#
# Env:
#   ZEROLLAMA_GLINER_CPP_ROOT — default ${REPO}/vendor/gliner-cpp-<shortsha>
#   ZEROLLAMA_GLINER_CPP_SHA  — full commit (default: GLINER_CPP_COMMIT)
#   GLINER_CPP_REPO           — default https://github.com/Knowledgator/GLiNER.cpp.git
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-${ROOT}}"

if [[ -f "${ROOT}/GLINER_CPP_COMMIT" ]]; then
  SHA="${ZEROLLAMA_GLINER_CPP_SHA:-$(tr -d '[:space:]' < "${ROOT}/GLINER_CPP_COMMIT")}"
else
  SHA="${ZEROLLAMA_GLINER_CPP_SHA:-af99d2b1fddbfde644e07b159730a76f876ae517}"
fi
SHORT="${SHA:0:9}"
DEST="${ZEROLLAMA_GLINER_CPP_ROOT:-${ROOT}/vendor/gliner-cpp-${SHORT}}"
REPO_URL="${GLINER_CPP_REPO:-https://github.com/Knowledgator/GLiNER.cpp.git}"

mkdir -p "$(dirname "${DEST}")"

if [[ -f "${DEST}/CMakeLists.txt" && -d "${DEST}/.git" ]]; then
  head="$(git -C "${DEST}" rev-parse HEAD)"
  if [[ "${head}" != "${SHA}" ]]; then
    echo ">>> checking out ${SHA} in ${DEST}..." >&2
    git -C "${DEST}" fetch origin "${SHA}" --depth 1 2>/dev/null \
      || git -C "${DEST}" fetch origin --depth 1
    git -C "${DEST}" checkout --force "${SHA}"
  else
    echo ">>> already at ${SHA}" >&2
  fi
else
  echo ">>> cloning ${REPO_URL} @ ${SHA} → ${DEST}" >&2
  rm -rf "${DEST}"
  git clone --filter=blob:none --depth 1 "${REPO_URL}" "${DEST}"
  git -C "${DEST}" fetch origin "${SHA}" --depth 1 2>/dev/null || true
  git -C "${DEST}" checkout --force "${SHA}" 2>/dev/null \
    || git -C "${DEST}" checkout --force FETCH_HEAD
  git -C "${DEST}" submodule update --init --recursive
fi

# Submodules (tokenizers) — required for build.
if [[ -f "${DEST}/.gitmodules" ]]; then
  git -C "${DEST}" submodule update --init --recursive
fi

# zerollama patches (re-apply after clean checkout). Marker: Finding 10 logits layout.
PATCH_DIR="${ROOT}/gliner/patches"
if [[ -d "${PATCH_DIR}" ]]; then
  shopt -s nullglob
  for p in "${PATCH_DIR}"/*.patch; do
    # Already applied → skip; otherwise apply from vendor root.
    if git -C "${DEST}" apply --check "${p}" >/dev/null 2>&1; then
      echo ">>> applying $(basename "${p}")" >&2
      git -C "${DEST}" apply "${p}"
    elif git -C "${DEST}" apply --reverse --check "${p}" >/dev/null 2>&1; then
      echo ">>> already applied $(basename "${p}")" >&2
    else
      echo ">>> warning: could not apply $(basename "${p}") (tree may already differ)" >&2
    fi
  done
  shopt -u nullglob
fi

echo "${DEST}"
