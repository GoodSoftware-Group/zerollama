#!/usr/bin/env bash
# Sync zerollama in-tree llama-kv-ext (Phase 15 v20/v33) into vendor or sibling llama.cpp.
#
# WHY: authoritative kv-ext sources live under llama/llama.cpp/ (patch 0014 + v33
# page_map). Vendor rebuilds must copy them before libllama links, or runtime _kv_native
# and subprocess llama-server miss writable page-map symbols.
#
# Usage:
#   ./scripts/vendor/stage_llama_kv_ext_for_vendor.sh /path/to/vendor/llama-cpp-<pin>
#   ./scripts/vendor/stage_llama_kv_ext_for_vendor.sh /path/to/vendor/llama-cpp-<pin> --check
#
# Exit 0 always; prints "unchanged" or "synced". With --check: exit 1 when staging
# would change files or CMake define is missing (caller should rebuild libllama).
set -euo pipefail

VENDOR_ROOT="${1:-}"
CHECK_ONLY="${2:-}"
if [[ -z "${VENDOR_ROOT}" || ! -f "${VENDOR_ROOT}/CMakeLists.txt" ]]; then
  echo "usage: $0 /path/to/vendor/llama-cpp-<pin> [--check]" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INTREE="${ROOT}/llama/llama.cpp"
VENDOR_SRC="${VENDOR_ROOT}/src"
VENDOR_INC="${VENDOR_ROOT}/include"
CMAKE="${VENDOR_SRC}/CMakeLists.txt"

for f in \
  "${INTREE}/include/llama-kv-ext.h" \
  "${INTREE}/src/llama-memory-kv-ext.cpp" \
  "${INTREE}/src/llama-kv-cache.h" \
  "${INTREE}/src/llama-kv-cache.cpp"; do
  if [[ ! -f "${f}" ]]; then
    echo "error: missing in-tree kv-ext source ${f}" >&2
    exit 1
  fi
done

_changed=0
_sync_one() {
  local src="$1" dst="$2"
  if [[ ! -f "${dst}" ]] || ! cmp -s "${src}" "${dst}"; then
    _changed=1
    if [[ "${CHECK_ONLY}" == "--check" ]]; then
      return 0
    fi
    install -m 644 "${src}" "${dst}"
  fi
}

_sync_one "${INTREE}/include/llama-kv-ext.h" "${VENDOR_INC}/llama-kv-ext.h"
_sync_one "${INTREE}/src/llama-memory-kv-ext.cpp" "${VENDOR_SRC}/llama-memory-kv-ext.cpp"
# WHY: vendor dig (MTP / dual-cache) can be ahead of in-tree llama-kv-cache.h.
# Blind overwrite breaks llama-server (ctor mismatch vs llama-kv-cache.cpp).
if grep -q 'llama_memory_t[[:space:]]*mem_other' "${VENDOR_SRC}/llama-kv-cache.h" 2>/dev/null \
  && ! grep -q 'llama_memory_t[[:space:]]*mem_other' "${INTREE}/src/llama-kv-cache.h" 2>/dev/null; then
  echo ">>> skip llama-kv-cache.h stage (vendor ctor ahead of in-tree)" >&2
else
  _sync_one "${INTREE}/src/llama-kv-cache.h" "${VENDOR_SRC}/llama-kv-cache.h"
fi
# WHY sync llama-kv-cache.cpp (v48): the donor-buffer overlay bind hook lives in
# the ctor's per-buft allocation loop, not just the staging header/impl pair.
# Same "vendor ahead of in-tree" guard as the .h sync above — a vendor dig with a
# newer ctor (e.g. MTP/dual-cache) must not be blind-overwritten.
if grep -q 'llama_memory_t[[:space:]]*mem_other' "${VENDOR_SRC}/llama-kv-cache.cpp" 2>/dev/null \
  && ! grep -q 'llama_memory_t[[:space:]]*mem_other' "${INTREE}/src/llama-kv-cache.cpp" 2>/dev/null; then
  echo ">>> skip llama-kv-cache.cpp stage (vendor ctor ahead of in-tree)" >&2
else
  _sync_one "${INTREE}/src/llama-kv-cache.cpp" "${VENDOR_SRC}/llama-kv-cache.cpp"
fi

if ! grep -q 'llama-memory-kv-ext.cpp' "${CMAKE}" 2>/dev/null; then
  _changed=1
  if [[ "${CHECK_ONLY}" != "--check" ]]; then
    sed -i '' '/llama-memory-recurrent.cpp/a\
            llama-memory-kv-ext.cpp
' "${CMAKE}" 2>/dev/null || \
    sed -i '/llama-memory-recurrent.cpp/a\            llama-memory-kv-ext.cpp' "${CMAKE}"
  fi
fi

if ! grep -q 'LLAMA_KV_EXT_WRITABLE_PAGE_MAP' "${CMAKE}" 2>/dev/null; then
  _changed=1
  if [[ "${CHECK_ONLY}" != "--check" ]]; then
    if grep -q 'target_compile_definitions(llama PRIVATE LLAMA_BUILD)' "${CMAKE}"; then
      sed -i '' '/target_compile_definitions(llama PRIVATE LLAMA_BUILD)/a\
target_compile_definitions(llama PRIVATE LLAMA_KV_EXT_WRITABLE_PAGE_MAP=1)
' "${CMAKE}" 2>/dev/null || \
      sed -i '/target_compile_definitions(llama PRIVATE LLAMA_BUILD)/a target_compile_definitions(llama PRIVATE LLAMA_KV_EXT_WRITABLE_PAGE_MAP=1)' "${CMAKE}"
    else
      printf '\n# zerollama Phase 15 v33\n' >> "${CMAKE}"
      printf 'target_compile_definitions(llama PRIVATE LLAMA_KV_EXT_WRITABLE_PAGE_MAP=1)\n' >> "${CMAKE}"
    fi
  fi
fi

if ! grep -q 'LLAMA_KV_EXT_EXTERNAL_ALIAS' "${CMAKE}" 2>/dev/null; then
  _changed=1
  if [[ "${CHECK_ONLY}" != "--check" ]]; then
    printf '# zerollama Phase 15 v47: external buffer alias probe + validate (no tensor mutation)\n' >> "${CMAKE}"
    printf 'target_compile_definitions(llama PRIVATE LLAMA_KV_EXT_EXTERNAL_ALIAS=1)\n' >> "${CMAKE}"
  fi
fi

if ! grep -q 'LLAMA_KV_EXT_DONOR_BUFFER' "${CMAKE}" 2>/dev/null; then
  _changed=1
  if [[ "${CHECK_ONLY}" != "--check" ]]; then
    printf '# zerollama Phase 15 v48: CPU-only donor-buffer overlay bind (opt-in, zero-copy KV cache alloc)\n' >> "${CMAKE}"
    printf 'target_compile_definitions(llama PRIVATE LLAMA_KV_EXT_DONOR_BUFFER=1)\n' >> "${CMAKE}"
  fi
fi

if [[ "${CHECK_ONLY}" == "--check" ]]; then
  if [[ "${_changed}" -eq 1 ]]; then
    exit 1
  fi
  exit 0
fi

if [[ "${_changed}" -eq 1 ]]; then
  echo ">>> staged llama-kv-ext (+ writable page-map) → ${VENDOR_ROOT}" >&2
else
  echo ">>> llama-kv-ext already synced at ${VENDOR_ROOT}" >&2
fi
