#!/usr/bin/env bash
# Ensure sibling DiffusionGemma llama.cpp tree (PR #24427 tip) under vendor/.
# Does NOT touch vendor/llama-cpp-b10615 or prod llama-server.
#
# WHY sibling (not b10615): draft block-diffusion must not risk prod chat/Laya/CLM emb.
# WHY apply llama/patches/diffusion/: prompt-only /v1/systemone + DG2b tokenize/gather
#   must be reproducible from git — not a host-only spike symlink edit (Finding 4).
# WHY refuse spike symlink by default: durable clones; DIFFUSION_ALLOW_SPIKE_LINK=1 only.
#
# Usage:
#   ./scripts/vendor/ensure_diffusion_llama.sh
#
# Env:
#   ZEROLLAMA_DIFFUSION_ROOT  — default ${REPO}/vendor/llama-cpp-diffusion-<shortsha>
#   ZEROLLAMA_DIFFUSION_SHA   — full commit (default: DIFFUSION_LLAMA_COMMIT or tip below)
#   DIFFUSION_LLAMA_REPO      — default https://github.com/ggml-org/llama.cpp.git
#   DIFFUSION_ALLOW_SPIKE_LINK=1 — lab-only: symlink /root/llama.cpp-diffusion-spike (discouraged)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-${ROOT}}"

if [[ -f "${ROOT}/DIFFUSION_LLAMA_COMMIT" ]]; then
  DIFFUSION_SHA="${ZEROLLAMA_DIFFUSION_SHA:-$(tr -d '[:space:]' < "${ROOT}/DIFFUSION_LLAMA_COMMIT")}"
else
  DIFFUSION_SHA="${ZEROLLAMA_DIFFUSION_SHA:-dd0cf04459b0c4f43aa6667dbc0879ac0cd50323}"
fi
SHORT="${DIFFUSION_SHA:0:9}"
DEST="${ZEROLLAMA_DIFFUSION_ROOT:-${ROOT}/vendor/llama-cpp-diffusion-${SHORT}}"
REPO_URL="${DIFFUSION_LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
PATCH_DIR="${ROOT}/llama/patches/diffusion"
SPIKE="${DIFFUSION_SPIKE_ROOT:-/root/llama.cpp-diffusion-spike}"

apply_diffusion_patches() {
  local tree="$1"
  [[ -d "${PATCH_DIR}" ]] || return 0
  shopt -s nullglob
  local patches=("${PATCH_DIR}"/*.patch)
  shopt -u nullglob
  ((${#patches[@]})) || return 0
  for p in "${patches[@]}"; do
    local base
    base="$(basename "${p}")"
    if git -C "${tree}" apply --check "${p}" >/dev/null 2>&1; then
      echo ">>> apply ${base}" >&2
      git -C "${tree}" apply "${p}"
    elif [[ "${base}" == 0001-* ]] && grep -qE "prompt-only denoise|/v1/systemone" \
      "${tree}/examples/diffusion-gemma/diffusion-gemma-server.cpp" 2>/dev/null; then
      echo ">>> skip ${base} (already applied)" >&2
    elif [[ "${base}" == 0002-* ]] && grep -qE "score_gathers|final_canvas_logit_gather|Post\(\"/tokenize\"" \
      "${tree}/examples/diffusion-gemma/diffusion-gemma-server.cpp" 2>/dev/null; then
      echo ">>> skip ${base} (already applied)" >&2
    else
      echo "error: cannot apply ${p} to ${tree}" >&2
      return 1
    fi
  done
}

# Drop discouraged spike symlink so rebuilds are durable.
if [[ -L "${DEST}" ]]; then
  if [[ "${DIFFUSION_ALLOW_SPIKE_LINK:-0}" == "1" ]]; then
    echo ">>> keeping spike symlink ${DEST} (DIFFUSION_ALLOW_SPIKE_LINK=1)" >&2
    apply_diffusion_patches "${DEST}"
    echo "${DEST}"
    exit 0
  fi
  echo ">>> removing spike symlink ${DEST} (use a real checkout)" >&2
  rm -f "${DEST}"
fi

mkdir -p "$(dirname "${DEST}")"

if [[ -f "${DEST}/CMakeLists.txt" && -d "${DEST}/.git" ]]; then
  head="$(git -C "${DEST}" rev-parse HEAD)"
  if [[ "${head}" != "${DIFFUSION_SHA}" ]]; then
    echo ">>> checking out ${DIFFUSION_SHA} in ${DEST}..." >&2
    git -C "${DEST}" fetch origin "${DIFFUSION_SHA}" --depth 1 2>/dev/null \
      || git -C "${DEST}" fetch origin pull/24427/head --depth 1
    git -C "${DEST}" checkout --force "${DIFFUSION_SHA}"
  fi
  apply_diffusion_patches "${DEST}"
  echo ">>> diffusion llama.cpp ready @ ${DEST} (${SHORT})" >&2
  echo "${DEST}"
  exit 0
fi

# Optional lab shortcut: materialize from spike as a real copy (not symlink).
if [[ -d "${SPIKE}/.git" ]] && [[ "$(git -C "${SPIKE}" rev-parse HEAD)" == "${DIFFUSION_SHA}" ]]; then
  echo ">>> copying spike ${SPIKE} → ${DEST} (real tree, not symlink)" >&2
  rm -rf "${DEST}"
  # Prefer local clone from spike to preserve objects without full network refetch.
  git clone --dissociate "${SPIKE}" "${DEST}" 2>/dev/null \
    || { mkdir -p "${DEST}" && cp -a "${SPIKE}/." "${DEST}/"; }
  git -C "${DEST}" checkout --force "${DIFFUSION_SHA}"
  # Reset server.cpp then re-apply repo patch so tree matches durable patch (spike may be dirty).
  git -C "${DEST}" checkout -- examples/diffusion-gemma/diffusion-gemma-server.cpp 2>/dev/null || true
  apply_diffusion_patches "${DEST}"
  echo ">>> diffusion llama.cpp ready @ ${DEST}" >&2
  echo "${DEST}"
  exit 0
fi

echo ">>> cloning ${REPO_URL} PR #24427 tip → ${DEST}" >&2
rm -rf "${DEST}"
git clone --filter=blob:none "${REPO_URL}" "${DEST}"
git -C "${DEST}" fetch origin pull/24427/head:pr-24427
git -C "${DEST}" checkout --force "${DIFFUSION_SHA}"
apply_diffusion_patches "${DEST}"
echo ">>> diffusion llama.cpp ready @ ${DEST}" >&2
echo "${DEST}"
