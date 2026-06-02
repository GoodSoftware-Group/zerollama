#!/usr/bin/env bash
# Print Phase 11–13 /health fields for GPU tuning (5080-class hosts).
#
#   ./scripts/gpu_health_report.sh
#   ZEROLLAMA_RUNTIME_URL=http://127.0.0.1:8081 ./scripts/gpu_health_report.sh
#
# Why: after one probed load, operators tune VRAM_MARGIN, estimate factor, min-free, and clamp.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
TIMEOUT="${GPU_HEALTH_TIMEOUT:-10}"

health_json=$(curl -sf -m "$TIMEOUT" "${RUNTIME_URL}/health")
export HEALTH_JSON="$health_json"
cd "${ROOT}/runtime"
PYTHONPATH=. python3 -c "
import json, os
from runtime.gpu_health_report import format_gpu_health_tuning_report
print(format_gpu_health_tuning_report(json.loads(os.environ['HEALTH_JSON'])))
"
