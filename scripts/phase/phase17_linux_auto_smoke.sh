#!/usr/bin/env bash
# Phase 17 Linux auto smoke — plain zerollama serve (ZEROLLAMA_LLAMA_SERVER=auto).
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "phase17_linux_auto_smoke is Linux-only (Darwin uses ggml default)" >&2
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export P17_OUT="${P17_OUT:-/tmp/phase17-linux-auto-smoke.json}"
export P17_LINUX_AUTO=1
export ZEROLLAMA_RUNTIME=0
unset ZEROLLAMA_RUNTIME_URL

echo "== Phase 17 Linux auto smoke (plain zerollama serve) =="
"${ROOT}/scripts/phase/phase17_llama_server_smoke.sh"

echo ""
echo "PASS: phase17_linux_auto_smoke (${P17_OUT})"
echo "Doc: docs/phase17-llama-server.md"
