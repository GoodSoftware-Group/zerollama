#!/usr/bin/env bash
# Phase 15 v43: post-decode migration summary sign-off (v42 fields).
#
# WHY: v42 added page_migration_summary on /health.kv_page_bind and migration_summary
# on kv_page_migration snapshot branches via last-probe fallback. GPU sign-off must
# prove those fields appear after a real in-process decode when kv-ext is linked.
#
# Prerequisites (sidecar must already be running):
#   - In-process multiseq sidecar with linked llama-kv-ext (writable page-map or batch decode)
#
# Usage:
#   ./scripts/phase15_migration_summary_smoke.sh
#   MIGRATION_SMOKE_SKIP_GEN=1 ./scripts/phase15_migration_summary_smoke.sh  # after prior generate
#
# Env:
#   RUNTIME_URL — default http://127.0.0.1:8081 (read-only)
#   MIGRATION_SMOKE_SKIP_GEN — skip POST /api/generate when caller already ran one
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime_smoke_lib.sh"

RUNTIME_URL="${RUNTIME_URL:-http://127.0.0.1:8081}"

echo "== Phase 15 migration summary smoke =="

if [[ "${MIGRATION_SMOKE_SKIP_GEN:-0}" != "1" ]]; then
  runtime_resume_if_needed "$(curl -sf "${RUNTIME_URL}/health")"
  gen_payload='{"model":"smoke","prompt":"Say: migration ok","stream":false,"options":{"num_predict":4,"num_ctx":4096,"temperature":0}}'
  gen_code=$(curl -s -o /tmp/phase15-mig-sum-gen.json -w '%{http_code}' \
    -X POST "${RUNTIME_URL}/api/generate" \
    -H 'Content-Type: application/json' \
    -d "$gen_payload")
  if [[ "$gen_code" != "200" ]]; then
    echo "HTTP ${gen_code} /api/generate:" >&2
    head -c 400 /tmp/phase15-mig-sum-gen.json >&2
    echo >&2
    exit 1
  fi
  python3 -c "
import json
d = json.load(open('/tmp/phase15-mig-sum-gen.json'))
assert d.get('done') and d.get('response'), d
print('generate ok, kv_decode_steps=', d.get('kv_decode_steps'))
"
fi

smoke_runtime_assert_kv_snapshot "$RUNTIME_URL"
smoke_runtime_assert_migration_summary "$RUNTIME_URL" 1

echo "PASS: phase15_migration_summary_smoke"
