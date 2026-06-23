#!/usr/bin/env bash
# Phase 16 edge smoke — upstream-shaped serve without Python runtime chat.
#
# WHY: Phase 16 v0 must prove --edge routes GGUF through llama-server and does not
# require the Python runtime sidecar for generate/chat.
#
# Usage:
#   ./scripts/phase16_edge_smoke.sh
#   P16_MODEL=llama3.2:3b ./scripts/phase16_edge_smoke.sh
#
# Env: same as phase17_llama_server_smoke.sh plus:
#   P16_OUT — JSON report (default /tmp/phase16-edge-smoke.json)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

P16_OUT="${P16_OUT:-/tmp/phase16-edge-smoke.json}"
export P17_OUT="${P16_OUT}"
export P17_MODEL="${P16_MODEL:-${P17_MODEL:-}}"
export P17_SERVE_EXTRA="--edge"
export P17_MODE=edge
export P17_ASSERT_RUNTIME_OFF=1

echo "== Phase 16 edge smoke (serve --edge) =="
"${ROOT}/scripts/phase17_llama_server_smoke.sh"

echo ""
echo "PASS: phase16_edge_smoke (${P16_OUT})"
echo "Doc: docs/phase16-thin-edge.md"
