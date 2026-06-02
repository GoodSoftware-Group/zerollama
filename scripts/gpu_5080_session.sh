#!/usr/bin/env bash
# One-shot 5080-class GPU session: preflight goldens + full smoke + Phase 13 snapshot.
#
# WHY this script: CI proves parsers without a GPU; a 16GB host needs one repeatable gate
# (Phase 10–13) with a JSON artifact + gpu_snapshot hints — not ad-hoc e2e flags.
# Does NOT require gpt-oss harmony on host (needs ~40+ GiB RAM); see docs/gpu-5080-operator-guide.md
#
#   export LLAMA_MODEL LLAMA_SERVER_BIN RUN_E2E_GGUF
#   export RUN_E2E_PROXY_MODEL=llama3.2:3B   # optional tools + proxy manifest
#   ./scripts/gpu_5080_session.sh
#
# Optional:
#   RUN_E2E_TOOLS=1 RUN_E2E_LEGACY=1 RUN_E2E_VRAM_CLAMP=1  # forwarded to gpu_smoke_all
#   RUN_E2E_TRAINING_OPS=1 RUN_E2E_TRAINING_TCP=1           # embedded training surfaces (serve needs OLLAMA_TRAINING=true)
#   GPU_PHASE13_SNAPSHOT_OUT=/tmp/5080-session.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ZEROLLAMA_REPO_ROOT="${ZEROLLAMA_REPO_ROOT:-$ROOT}"
export RUN_E2E_PREFLIGHT=1
export RUN_E2E_PHASE13_SNAPSHOT=1
export GPU_PHASE13_SNAPSHOT_OUT="${GPU_PHASE13_SNAPSHOT_OUT:-/tmp/5080-session.json}"

if [[ -z "${LLAMA_MODEL:-}" && -z "${RUN_E2E_GGUF:-}" ]]; then
  echo "Set LLAMA_MODEL or RUN_E2E_GGUF (small GGUF for 16GB, e.g. 1B Q8)" >&2
  exit 1
fi
if [[ -z "${LLAMA_SERVER_BIN:-}" ]]; then
  echo "Set LLAMA_SERVER_BIN (path to llama-server)" >&2
  exit 1
fi

echo "== Phase 12 preflight + GPU smokes + snapshot =="
"${ROOT}/scripts/gpu_smoke_all.sh"

if [[ -f "${GPU_PHASE13_SNAPSHOT_OUT}" ]]; then
  echo ""
  (cd "${ROOT}/runtime" && PYTHONPATH=. python3 -m runtime.gpu_snapshot "${GPU_PHASE13_SNAPSHOT_OUT}") || true
fi

echo "PASS: gpu_5080_session (snapshot: ${GPU_PHASE13_SNAPSHOT_OUT})"
