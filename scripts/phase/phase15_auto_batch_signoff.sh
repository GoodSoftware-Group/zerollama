#!/usr/bin/env bash
# Phase 15 v45: combined non-stream + stream auto-batch GPU sign-off.
#
# WHY: v41/v44 added separate smokes; operators running both gates need one entry
# point and sidecar env wired before multiseq restart (see phase15_runtime_auto_batch_env_apply).
#
# Prerequisites (multiseq sidecar with both env vars — metal signoff sets these when
# RUN_P15_AUTO_BATCH_ALL=1 before sidecar restart):
#   ZEROLLAMA_KV_AUTO_BATCH=1
#   ZEROLLAMA_KV_AUTO_BATCH_STREAM=1
#
# Usage:
#   RUN_P15_AUTO_BATCH_ALL=1 ./scripts/phase/phase15_metal_signoff.sh
#   # or standalone when sidecar already up with both env set:
#   ./scripts/phase/phase15_auto_batch_signoff.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "== Phase 15 auto-batch sign-off (non-stream + stream) =="

"${ROOT}/scripts/phase/phase15_auto_batch_smoke.sh"
"${ROOT}/scripts/phase/phase15_stream_auto_batch_smoke.sh"

echo "PASS: phase15_auto_batch_signoff"
