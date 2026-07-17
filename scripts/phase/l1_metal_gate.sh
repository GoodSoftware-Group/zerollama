#!/usr/bin/env bash
# L1 Metal gate — RAM-tier profile unit tests + optional live /health check.
#
# WHY: Apple L1 is tier-based (hw.memsize), not per-model CUDA calibrate. Ship gate
# is pytest coverage + gpu_profile fields on a running sidecar when available.
#
# Usage:
#   ./scripts/phase/l1_metal_gate.sh
#   ZEROLLAMA_RUNTIME_URL=http://127.0.0.1:8081 ./scripts/phase/l1_metal_gate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "== L1 Metal gate =="

if [[ -x "${ROOT}/scripts/runtime/runtime_uv_venv.sh" ]]; then
  "${ROOT}/scripts/runtime/runtime_uv_venv.sh" >/dev/null
fi

(cd "${ROOT}/runtime" && PYTHONPATH=. python3 -m pytest tests/test_gpu_profiles.py -q)

RT_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
if curl -sf --max-time 3 "${RT_URL}/health" >/dev/null 2>&1; then
  echo ""
  echo "== L1 Metal live /health =="
  curl -sf "${RT_URL}/health" | python3 -c "
import json, sys
h = json.load(sys.stdin)
gp = h.get('gpu_profile') or {}
if not gp.get('id'):
    print('VERDICT: L1 Metal FAIL — gpu_profile.id missing (ZEROLLAMA_GPU_PROFILE=0?)', file=sys.stderr)
    sys.exit(1)
print(f\"gpu_profile.id: {gp.get('id')}\")
print(f\"gpu_profile.n_parallel: {gp.get('n_parallel')}\")
print(f\"gpu_profile.bucket_label: {gp.get('bucket_label')}\")
if gp.get('unified_memory_gb') is not None:
    print(f\"gpu_profile.unified_memory_gb: {gp.get('unified_memory_gb')}\")
"
else
  echo "skip: no runtime at ${RT_URL} (unit tests only)"
fi

echo ""
echo "PASS: L1 Metal gate (RAM-tier profiles)"
