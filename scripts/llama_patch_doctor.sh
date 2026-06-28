#!/usr/bin/env bash
# Offline llama.cpp patch / vendor doctor — catches "lost patch" footguns before live smokes.
#
# WHY: zerollama patches live in llama/patches/ → vendor git am → rsync in-tree → rebuild
# llama-server. Stale ../llama.cpp binaries lack POST /kv/seq-copy even when in-tree looks fine.
#
# Usage:
#   ./scripts/llama_patch_doctor.sh
#   LLAMA_PATCH_JSON=/tmp/llama-patch-doctor.json ./scripts/llama_patch_doctor.sh
#   LLAMA_PATCH_PROBE_URL=http://127.0.0.1:8082 ./scripts/llama_patch_doctor.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_PATCH_JSON="${LLAMA_PATCH_JSON:-}"
LLAMA_PATCH_PROBE_URL="${LLAMA_PATCH_PROBE_URL:-}"

cd "${ROOT}/runtime"
export LLAMA_PATCH_PROBE_URL

report="$(
  PYTHONPATH=. python3 <<'PY'
import json
import os
import sys

from runtime.llama_patch_health import llama_patch_health

probe = os.environ.get("LLAMA_PATCH_PROBE_URL", "").strip() or None
report = llama_patch_health(probe_http_base=probe)
print(json.dumps(report, indent=2))
if report.get("status") != "pass":
    sys.exit(1)
PY
)"

echo "${report}"

if [[ -n "${LLAMA_PATCH_JSON}" ]]; then
  printf '%s\n' "${report}" >"${LLAMA_PATCH_JSON}"
  echo "report: ${LLAMA_PATCH_JSON}" >&2
fi

echo "PASS: llama_patch_doctor" >&2
