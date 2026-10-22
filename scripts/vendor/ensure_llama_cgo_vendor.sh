#!/usr/bin/env bash
# Restore llama/llama.cpp/vendor extras required by CGO (gitignored vendor/).
#
# Copies nlohmann / miniaudio / stb / hash / cpp-httplib from a sibling or
# shallow pin checkout when missing. Does not start serve.
#
# Usage:
#   ./scripts/vendor/ensure_llama_cgo_vendor.sh
#   source ./scripts/vendor/ensure_llama_cgo_vendor.sh && ensure_llama_cgo_vendor
#
# Env:
#   LLAMA_CPP_ROOT — sibling tree (default ../llama.cpp)
#   MAC_SETUP_LLAMA_CLONE=1 — allow shallow pin fetch when sibling missing (default 1)
set -euo pipefail

_CGO_VENDOR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-${_CGO_VENDOR_ROOT}}"

ensure_llama_cgo_vendor() {
  local dest="${ZEROLLAMA_REPO}/llama/llama.cpp/vendor"
  local need=0
  local name
  for name in nlohmann miniaudio stb hash cpp-httplib; do
    case "${name}" in
      nlohmann)
        [[ -f "${dest}/nlohmann/json.hpp" ]] || need=1
        ;;
      miniaudio)
        [[ -f "${dest}/miniaudio/miniaudio.h" ]] || need=1
        ;;
      stb)
        [[ -f "${dest}/stb/stb_image.h" ]] || need=1
        ;;
      hash)
        [[ -f "${dest}/hash/sha256/sha256.c" ]] || need=1
        ;;
      cpp-httplib)
        [[ -f "${dest}/cpp-httplib/httplib.h" ]] || need=1
        ;;
    esac
  done
  if [[ "${need}" -eq 0 ]]; then
    echo ">>> CGO llama vendor headers OK at ${dest}" >&2
    return 0
  fi

  local src="" d pin dest_real
  pin="$(tr -d '[:space:]' <"${ZEROLLAMA_REPO}/LLAMA_CPP_COMMIT" 2>/dev/null || true)"
  dest_real="$(cd "${dest}" 2>/dev/null && pwd -P || echo "${dest}")"
  # Prefer a *complete* vendor tree (incl. hash/). Skip dest itself when
  # LLAMA_CPP_ROOT points at the in-tree llama.cpp (common on CT builds).
  _cgo_vendor_complete() {
    [[ -f "$1/nlohmann/json.hpp" && -f "$1/cpp-httplib/httplib.h" && -f "$1/hash/sha256/sha256.c" ]]
  }
  for d in \
    "${LLAMA_CPP_ROOT:-${ZEROLLAMA_REPO}/../llama.cpp}/vendor" \
    "${ZEROLLAMA_REPO}/../llama.cpp/vendor" \
    "${ZEROLLAMA_REPO}/vendor/llama-cpp-${pin}/vendor" \
    "${ZEROLLAMA_REPO}/vendor/llama-cpp-b11351/vendor" \
    "${ZEROLLAMA_REPO}/vendor/llama-cpp-b10615/vendor" \
    "${ZEROLLAMA_REPO}/vendor/llama-cpp-b10488/vendor" \
    "${ZEROLLAMA_REPO}"/vendor/llama-cpp-*/vendor
  do
    [[ -d "${d}" ]] || continue
    local d_real
    d_real="$(cd "${d}" 2>/dev/null && pwd -P || echo "${d}")"
    [[ "${d_real}" == "${dest_real}" ]] && continue
    if _cgo_vendor_complete "${d}"; then
      src="${d}"
      break
    fi
    # Incomplete but usable fallback if we never find hash.
    if [[ -z "${src}" && -f "${d}/nlohmann/json.hpp" ]]; then
      src="${d}"
    fi
  done

  if { [[ -z "${src}" ]] || ! _cgo_vendor_complete "${src}"; } && [[ "${MAC_SETUP_LLAMA_CLONE:-1}" == "1" ]]; then
    # Headers-only: shallow ggml-org pin into a cache under the repo (not full elizaOS).
    local cache="${ZEROLLAMA_REPO}/.cache/llama-cpp-vendor-pin"
    local ref="${pin:-631109b34da437a3c4a5ebd75091d677671392e3}"
    echo ">>> fetching llama.cpp vendor headers @ ${ref:0:12} (shallow)" >&2
    mkdir -p "$(dirname "${cache}")"
    if [[ ! -d "${cache}/.git" ]]; then
      rm -rf "${cache}"
      if ! git clone --filter=blob:none --sparse --depth 1 \
        https://github.com/ggml-org/llama.cpp.git "${cache}"; then
        echo "error: shallow clone for CGO vendor failed" >&2
        return 1
      fi
    fi
    git -C "${cache}" fetch --depth 1 origin "${ref}" 2>/dev/null || true
    git -C "${cache}" checkout --force "${ref}" 2>/dev/null \
      || git -C "${cache}" checkout --force FETCH_HEAD
    git -C "${cache}" sparse-checkout set vendor 2>/dev/null || true
    if _cgo_vendor_complete "${cache}/vendor"; then
      src="${cache}/vendor"
    elif [[ -z "${src}" && -f "${cache}/vendor/nlohmann/json.hpp" ]]; then
      src="${cache}/vendor"
    fi
  fi

  if [[ -z "${src}" ]]; then
    echo "error: missing CGO vendor headers under ${dest}" >&2
    echo "  fix: ./scripts/vendor/ensure_llama_cpp_sibling.sh" >&2
    echo "  or:  MAC_SETUP_LLAMA_CLONE=1 ./scripts/vendor/ensure_llama_cgo_vendor.sh" >&2
    return 1
  fi

  echo ">>> restoring CGO llama vendor headers from ${src}" >&2
  mkdir -p "${dest}"
  for name in nlohmann miniaudio stb hash cpp-httplib; do
    if [[ -d "${src}/${name}" ]]; then
      rm -rf "${dest:?}/${name}"
      cp -R "${src}/${name}" "${dest}/${name}"
    fi
  done
  if [[ ! -f "${dest}/nlohmann/json.hpp" || ! -f "${dest}/cpp-httplib/httplib.h" ]]; then
    echo "error: restore incomplete under ${dest}" >&2
    return 1
  fi
  if [[ ! -f "${dest}/hash/sha256/sha256.c" ]]; then
    echo "error: restore missing vendor/hash (mtmd CGO). Source was ${src}" >&2
    return 1
  fi
  echo ">>> CGO llama vendor headers ready" >&2
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  ensure_llama_cgo_vendor
fi
