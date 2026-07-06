#!/usr/bin/env bash
# Phase 15 v44: concurrent non-stream /api/generate with auto-batch (v32).
#
# WHY separate from phase15_batch_decode_smoke.sh: batch decode smoke uses
# /internal/generate-batch; v32 auto-batch coalesces concurrent public
# /api/generate stream=false requests when ZEROLLAMA_KV_AUTO_BATCH=1.
#
# Prerequisites (sidecar must already be running):
#   - kv_inprocess_n_seq_max >= 2
#   - Linked native batch decode (batch_decode_in_c)
#   - Sidecar started with ZEROLLAMA_KV_AUTO_BATCH=1
#
# Usage (after multiseq sidecar boot with auto-batch env):
#   RUN_P15_AUTO_BATCH=1 ZEROLLAMA_KV_AUTO_BATCH=1 ./scripts/phase15_metal_signoff.sh
#   # or, sidecar already up with env set:
#   ./scripts/phase15_auto_batch_smoke.sh
#
# Env:
#   RUNTIME_URL — default http://127.0.0.1:8081 (read-only; do not start servers here)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime_smoke_lib.sh"

RUNTIME_URL="${RUNTIME_URL:-http://127.0.0.1:8081}"

echo "== Phase 15 non-stream auto-batch smoke =="

pre_health="$(curl -sf "${RUNTIME_URL}/health")"
python3 -c "
import json, sys
h = json.loads(sys.argv[1])
nseq = int(h.get('kv_inprocess_n_seq_max') or 0)
assert nseq >= 2, f'need multiseq sidecar, got kv_inprocess_n_seq_max={nseq}'
dl = h.get('kv_decode_loop') or {}
assert dl.get('available') and dl.get('batch_decode_in_c'), dl
ab = h.get('kv_auto_batch') or {}
ns = ab.get('non_stream') or {}
assert ns.get('enabled') is True, (
    'ZEROLLAMA_KV_AUTO_BATCH=1 required on sidecar; got non-stream auto-batch disabled'
)
print(
    'pre-smoke ok:',
    f'n_seq_max={nseq}',
    f'non_stream_enabled={ns.get(\"enabled\")}',
    f'flush_count={ns.get(\"flush_count\")}',
)
" "$pre_health"

read -r pre_flush pre_batched < <(
  python3 -c "
import json, sys
h = json.loads(sys.argv[1])
s = (h.get('kv_auto_batch') or {}).get('non_stream') or {}
print(int(s.get('flush_count') or 0), int(s.get('batched_requests') or 0))
" "$pre_health"
)

payload_a='{"model":"smoke","prompt":"Say: batch alpha","stream":false,"options":{"num_predict":6,"num_ctx":4096,"temperature":0}}'
payload_b='{"model":"smoke","prompt":"Say: batch beta","stream":false,"options":{"num_predict":6,"num_ctx":4096,"temperature":0}}'

curl -s -m 120 -X POST "${RUNTIME_URL}/api/generate" \
  -H 'Content-Type: application/json' \
  -d "$payload_a" > /tmp/phase15-ab-a.json &
pid_a=$!
curl -s -m 120 -X POST "${RUNTIME_URL}/api/generate" \
  -H 'Content-Type: application/json' \
  -d "$payload_b" > /tmp/phase15-ab-b.json &
pid_b=$!

wait "$pid_a" "$pid_b"

python3 -c "
import json

def load_done(path):
    d = json.load(open(path))
    assert d.get('done') and (d.get('response') or '').strip(), d
    return d['response'].strip()

a = load_done('/tmp/phase15-ab-a.json')
b = load_done('/tmp/phase15-ab-b.json')
print('concurrent non-stream ok:', 'A=', a[:24], 'B=', b[:24])
"

post_health="$(curl -sf "${RUNTIME_URL}/health")"
python3 -c "
import json, sys
h = json.loads(sys.argv[1])
pre_flush, pre_batched = int(sys.argv[2]), int(sys.argv[3])
s = (h.get('kv_auto_batch') or {}).get('non_stream') or {}
flush = int(s.get('flush_count') or 0)
batched = int(s.get('batched_requests') or 0)
assert flush > pre_flush or batched >= pre_batched + 2, (
    f'expected non-stream auto-batch activity: flush {pre_flush}->{flush}, '
    f'batched {pre_batched}->{batched}'
)
print('post-smoke non-stream auto-batch:', f'flush_count={flush}', f'batched_requests={batched}')
" "$post_health" "$pre_flush" "$pre_batched"

smoke_runtime_assert_kv_snapshot "$RUNTIME_URL"

echo "PASS: phase15_auto_batch_smoke"
