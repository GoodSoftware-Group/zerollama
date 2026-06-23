#!/usr/bin/env bash
# Phase 15 upstream writable KV watch — scan llama.h for page-handle APIs.
#
# WHY: Writable cross-allocator bind (criterion #5) unblocks when upstream ships stable
# page-map symbols. Complements phase15_llama_kv_ext_pin_check.sh (in-tree staging).
#
# Usage:
#   ./scripts/phase15_upstream_kv_watch.sh
#   OLLAMA_UPSTREAM_DIR=../ollama-upstream ./scripts/phase15_upstream_kv_watch.sh
#   P15_UPSTREAM_JSON=/tmp/phase15-upstream-kv-watch.json ./scripts/phase15_upstream_kv_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${OLLAMA_UPSTREAM_DIR:-${ROOT}/../ollama-upstream}"
P15_UPSTREAM_JSON="${P15_UPSTREAM_JSON:-}"
PIN="$(grep -E '^FETCH_HEAD=' "${ROOT}/Makefile.sync" | cut -d= -f2)"

WATCH=(
  llama_memory_kv_page_map
  llama_memory_kv_page_write
  llama_kv_cache_get_block
)

FOUND=()

scan_llama_h() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "skip: ${label} (${path} missing)"
    return 0
  fi
  local hit=0
  for sym in "${WATCH[@]}"; do
    if grep -q "${sym}" "${path}"; then
      echo "NOTICE: ${label} llama.h contains ${sym} — refresh Phase 15 writable bind tracker"
      FOUND+=("${label}:${sym}")
      hit=1
    fi
  done
  if [[ "${hit}" == "0" ]]; then
    echo "ok: ${label} — no upstream writable watch symbols yet"
  fi
}

echo "== Phase 15 upstream writable KV watch (pin=${PIN}) =="
scan_llama_h "${ROOT}/llama/llama.cpp/include/llama.h" "in-tree"
scan_llama_h "${UPSTREAM}/llama/llama.cpp/include/llama.h" "ollama-upstream"

if [[ -n "${P15_UPSTREAM_JSON}" ]]; then
  FOUND_JSON="$(printf '%s\n' "${FOUND[@]:-}" | python3 -c 'import json,sys; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))')"
  PIN="${PIN}" P15_UPSTREAM_JSON="${P15_UPSTREAM_JSON}" FOUND_JSON="${FOUND_JSON}" python3 <<'PY'
import json, os, pathlib
out = pathlib.Path(os.environ["P15_UPSTREAM_JSON"])
report = {
    "status": "watch",
    "pin": os.environ.get("PIN", ""),
    "symbols_found": json.loads(os.environ.get("FOUND_JSON", "[]")),
    "watch_list": [
        "llama_memory_kv_page_map",
        "llama_memory_kv_page_write",
        "llama_kv_cache_get_block",
    ],
    "blocked_until": "upstream ships writable page-handle API",
}
out.write_text(json.dumps(report, indent=2) + "\n")
print(f"report: {out}")
PY
fi

echo "PASS: phase15_upstream_kv_watch"
echo "Doc: docs/phase15-llama-kv-ext-upstream.md"
