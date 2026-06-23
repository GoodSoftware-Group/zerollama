#!/usr/bin/env bash
# Verify llama-kv-ext staging API is present in-tree and matches the llama.cpp pin.
#
# WHY: vendor sync (sync_vendor_llama.sh) rsync --delete can wipe untracked fork files.
# Patch 0014 + this check keep Phase 15 tensor bind from silently regressing on pin bumps.
#
# Usage:
#   ./scripts/phase15_llama_kv_ext_pin_check.sh
#   LLAMA_CPP_ROOT=../llama.cpp ./scripts/phase15_llama_kv_ext_pin_check.sh
#   P15_PIN_JSON=/tmp/phase15-kv-ext-pin-check.json  — optional JSON report
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN="$(grep -E '^FETCH_HEAD=' "${ROOT}/Makefile.sync" | cut -d= -f2)"
VERSION="$(cat "${ROOT}/LLAMA_CPP_VERSION" 2>/dev/null || true)"
P15_PIN_JSON="${P15_PIN_JSON:-}"
UPSTREAM_WATCH_FOUND=()

echo "== Phase 15 llama-kv-ext pin check (pin=${PIN}, LLAMA_CPP_VERSION=${VERSION}) =="

IN_TREE=(
  "${ROOT}/llama/llama.cpp/include/llama-kv-ext.h"
  "${ROOT}/llama/llama.cpp/src/llama-memory-kv-ext.cpp"
  "${ROOT}/llama/patches/0014-ollama-llama-kv-ext-Phase-15-tensor-page-bind-b9611.patch"
)

for f in "${IN_TREE[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "FAIL: missing ${f#${ROOT}/}" >&2
    exit 1
  fi
done

grep -q 'llama-memory-kv-ext.cpp' "${ROOT}/llama/llama.cpp/src/CMakeLists.txt"
grep -q 'cell_index_for' "${ROOT}/llama/llama.cpp/src/llama-kv-cache.h"
grep -q 'llama_memory_kv_ext_classify' "${ROOT}/llama/llama.cpp/include/llama-kv-ext.h"

# Staging API symbols we export from libllama when linked.
REQUIRED_API=(
  llama_memory_kv_cell_for_pos
  llama_memory_kv_cell_map_range
  llama_memory_kv_tensor_info
  llama_memory_kv_ext_classify
  llama_memory_kv_ext_writable_bind_probe
)
for sym in "${REQUIRED_API[@]}"; do
  grep -q "${sym}" "${ROOT}/llama/llama.cpp/include/llama-kv-ext.h"
done

# Upstream stable memory API that llama-kv-ext builds on (must exist at pin).
UPSTREAM_DEPS=(
  llama_get_memory
  llama_memory_can_shift
  llama_memory_seq_pos_min
  llama_memory_seq_pos_max
)
LLAMA_H="${ROOT}/llama/llama.cpp/include/llama.h"

# Upstream writable page-handle symbols — when any appear in llama.h, refresh v32b tracker.
UPSTREAM_WRITABLE_WATCH=(
  llama_memory_kv_page_map
  llama_memory_kv_page_write
  llama_kv_cache_get_block
)
for sym in "${UPSTREAM_WRITABLE_WATCH[@]}"; do
  if grep -q "${sym}" "${LLAMA_H}"; then
    echo "NOTICE: upstream writable KV API '${sym}' found in llama.h — refresh Phase 15 writable bind tracker (v32b)"
    UPSTREAM_WATCH_FOUND+=("${sym}")
  fi
done

for sym in "${UPSTREAM_DEPS[@]}"; do
  if ! grep -q "${sym}" "${LLAMA_H}"; then
    echo "FAIL: upstream dep ${sym} missing from in-tree llama.h — pin bump may need ext refresh" >&2
    exit 1
  fi
done

# Optional: sibling llama.cpp used for runtime libllama build.
LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-${ROOT}/../llama.cpp}"
if [[ -f "${LLAMA_CPP_ROOT}/include/llama.h" ]]; then
  for sym in "${UPSTREAM_DEPS[@]}"; do
    if ! grep -q "${sym}" "${LLAMA_CPP_ROOT}/include/llama.h"; then
      echo "WARN: sibling ${LLAMA_CPP_ROOT} missing ${sym} — rebuild libllama after pin bump" >&2
    fi
  done
  if [[ ! -f "${LLAMA_CPP_ROOT}/include/llama-kv-ext.h" ]]; then
    echo "WARN: sibling llama.cpp lacks llama-kv-ext.h — runtime GPU build needs zerollama fork or patch 0014 applied" >&2
  fi
  LIB=""
  for candidate in \
    "${LLAMA_CPP_ROOT}/build/bin/libllama.dylib" \
    "${LLAMA_CPP_ROOT}/build/bin/libllama.so" \
    "${LLAMA_CPP_ROOT}/build/libllama.dylib" \
    "${LLAMA_CPP_ROOT}/build/libllama.so"; do
    if [[ -f "$candidate" ]]; then
      LIB="$candidate"
      break
    fi
  done
  if [[ -n "$LIB" ]] && command -v nm >/dev/null 2>&1; then
    for sym in "${REQUIRED_API[@]}"; do
      if ! nm -gU "$LIB" 2>/dev/null | grep -qE " _${sym}$| T _${sym}$| T ${sym}$"; then
        echo "WARN: ${LIB} missing exported symbol ${sym} — rebuild libllama from patched tree" >&2
      fi
    done
  fi
fi

if [[ -n "${P15_PIN_JSON}" ]]; then
  WATCH_JSON="$(printf '%s\n' "${UPSTREAM_WATCH_FOUND[@]:-}" | python3 -c 'import json,sys; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))')"
  PIN="${PIN}" VERSION="${VERSION}" P15_PIN_JSON="${P15_PIN_JSON}" WATCH_JSON="${WATCH_JSON}" python3 <<'PY'
import json, os, pathlib
out = pathlib.Path(os.environ["P15_PIN_JSON"])
report = {
    "status": "pass",
    "pin": os.environ.get("PIN", ""),
    "llama_cpp_version": os.environ.get("VERSION", ""),
    "patch": "0014-ollama-llama-kv-ext-Phase-15-tensor-page-bind-b9611.patch",
    "staging_api": [
        "llama_memory_kv_cell_for_pos",
        "llama_memory_kv_cell_map_range",
        "llama_memory_kv_tensor_info",
        "llama_memory_kv_ext_classify",
        "llama_memory_kv_ext_writable_bind_probe",
    ],
    "upstream_writable_watch_found": json.loads(os.environ.get("WATCH_JSON", "[]")),
    "upstream_writable_watch": [
        "llama_memory_kv_page_map",
        "llama_memory_kv_page_write",
        "llama_kv_cache_get_block",
    ],
}
out.write_text(json.dumps(report, indent=2) + "\n")
print(f"report: {out}")
PY
fi

echo "PASS: llama-kv-ext in-tree + patch 0014 present; upstream memory deps OK at pin ${PIN}"
