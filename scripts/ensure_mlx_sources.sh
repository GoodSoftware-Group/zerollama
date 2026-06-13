#!/usr/bin/env bash
# Ensure local MLX / MLX-C checkouts contain zerollama's pinned commits (and HEAD).
# Shallow clones (e.g. git clone --depth 1) omit history; this fetches missing SHAs.
#
# WHY this exists: MLX_VERSION / MLX_C_VERSION pin safetensors inference separately from
# llama.cpp (GGUF). After bumping those files, checkout the SHAs here then rebuild dylibs
# (GOFLAGS=-mod=mod ./scripts/build_zerollama_mac.sh with BUILD_MLX=1) — or
# build_mlx_dylibs_mac.sh alone after pins change.
#
# Usage:
#   ./scripts/ensure_mlx_sources.sh
#   ./scripts/ensure_mlx_sources.sh --clone    # clone missing sibling repos (full history)
#   source ./scripts/ensure_mlx_sources.sh && ensure_mlx_sources
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MLX_PIN="$(tr -d '[:space:]' < "${ROOT}/MLX_VERSION")"
MLX_C_PIN="$(tr -d '[:space:]' < "${ROOT}/MLX_C_VERSION")"

OLLAMA_MLX_SOURCE="${OLLAMA_MLX_SOURCE:-${ROOT}/../mlx}"
OLLAMA_MLX_C_SOURCE="${OLLAMA_MLX_C_SOURCE:-${ROOT}/../mlx-c}"

DO_CLONE=0
for arg in "$@"; do
  case "$arg" in
    --clone) DO_CLONE=1 ;;
    -h | --help)
      echo "usage: $0 [--clone]" >&2
      exit 0
      ;;
    *)
      echo "unknown arg: $arg" >&2
      exit 1
      ;;
  esac
done

_ensure_git_repo() {
  local label=$1 path=$2 url=$3
  if [[ -d "${path}/.git" ]]; then
    return 0
  fi
  if [[ "${DO_CLONE}" -ne 1 ]]; then
    echo "error: ${label} checkout missing at ${path}" >&2
    echo "  clone: git clone ${url} ${path}" >&2
    echo "  or re-run: $0 --clone" >&2
    return 1
  fi
  echo ">>> cloning ${label} -> ${path}" >&2
  git clone "${url}" "${path}"
}

_ensure_commit() {
  local label=$1 repo=$2 commit=$3
  if git -C "${repo}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    echo "ok ${label}: has ${commit:0:12}" >&2
    return 0
  fi

  echo ">>> ${label}: fetching missing commit ${commit:0:12} in ${repo}" >&2
  if git -C "${repo}" fetch origin "${commit}" --depth=1 2>/dev/null \
    && git -C "${repo}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    echo "ok ${label}: fetched ${commit:0:12}" >&2
    return 0
  fi

  if [[ "$(git -C "${repo}" rev-parse --is-shallow-repository)" == "true" ]]; then
    echo ">>> ${label}: deepening shallow clone" >&2
    git -C "${repo}" fetch --unshallow origin 2>/dev/null \
      || git -C "${repo}" fetch origin --deepen=1000
  else
    git -C "${repo}" fetch origin
  fi

  if git -C "${repo}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    echo "ok ${label}: has ${commit:0:12} after fetch" >&2
    return 0
  fi

  echo "error: ${label}: commit ${commit} still missing in ${repo}" >&2
  return 1
}

ensure_mlx_sources() {
  _ensure_git_repo "MLX" "${OLLAMA_MLX_SOURCE}" "https://github.com/ml-explore/mlx.git"
  _ensure_git_repo "MLX-C" "${OLLAMA_MLX_C_SOURCE}" "https://github.com/ml-explore/mlx-c.git"

  _ensure_commit "MLX (zerollama pin)" "${OLLAMA_MLX_SOURCE}" "${MLX_PIN}"
  _ensure_commit "MLX-C (zerollama pin)" "${OLLAMA_MLX_C_SOURCE}" "${MLX_C_PIN}"

  local head
  head="$(git -C "${OLLAMA_MLX_SOURCE}" rev-parse HEAD)"
  _ensure_commit "MLX (HEAD)" "${OLLAMA_MLX_SOURCE}" "${head}"

  head="$(git -C "${OLLAMA_MLX_C_SOURCE}" rev-parse HEAD)"
  _ensure_commit "MLX-C (HEAD)" "${OLLAMA_MLX_C_SOURCE}" "${head}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  ensure_mlx_sources
fi
