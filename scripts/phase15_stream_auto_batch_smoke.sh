#!/usr/bin/env bash
# Phase 15 v41: concurrent streaming /api/generate with stream auto-batch (v37).
#
# WHY separate from phase15_batch_decode_smoke.sh: batch decode smoke uses
# /internal/generate-batch; stream auto-batch coalesces concurrent public
# /api/generate stream=true requests when ZEROLLAMA_KV_AUTO_BATCH_STREAM=1.
#
# Prerequisites (sidecar must already be running):
#   - kv_inprocess_n_seq_max >= 2
#   - Linked native batch decode (batch_decode_in_c)
#   - Sidecar started with ZEROLLAMA_KV_AUTO_BATCH_STREAM=1
#
# Usage (after multiseq sidecar boot with stream auto-batch env):
#   ZEROLLAMA_KV_AUTO_BATCH_STREAM=1 ./scripts/phase15_metal_signoff.sh
#   # or, sidecar already up with env set:
#   ./scripts/phase15_stream_auto_batch_smoke.sh
#
# Env:
#   RUNTIME_URL — default http://127.0.0.1:8081 (read-only; do not start servers here)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime_smoke_lib.sh"

RUNTIME_URL="${RUNTIME_URL:-http://127.0.0.1:8081}"

echo "== Phase 15 stream auto-batch smoke =="

pre_health="$(curl -sf "${RUNTIME_URL}/health")"
python3 -c "
import json, sys
h = json.loads(sys.argv[1])
nseq = int(h.get('kv_inprocess_n_seq_max') or 0)
assert nseq >= 2, f'need multiseq sidecar, got kv_inprocess_n_seq_max={nseq}'
dl = h.get('kv_decode_loop') or {}
assert dl.get('available') and dl.get('batch_decode_in_c'), dl
ab = h.get('kv_auto_batch') or {}
stream_ab = ab.get('stream') or {}
assert stream_ab.get('enabled') is True, (
    'ZEROLLAMA_KV_AUTO_BATCH_STREAM=1 required on sidecar; got stream auto-batch disabled'
)
print(
    'pre-smoke ok:',
    f'n_seq_max={nseq}',
    f'stream_enabled={stream_ab.get(\"enabled\")}',
    f'flush_count={stream_ab.get(\"flush_count\")}',
)
" "$pre_health"

read -r pre_flush pre_batched < <(
  python3 -c "
import json, sys
h = json.loads(sys.argv[1])
s = (h.get('kv_auto_batch') or {}).get('stream') or {}
print(int(s.get('flush_count') or 0), int(s.get('batched_requests') or 0))
" "$pre_health"
)

payload_a='{"model":"smoke","prompt":"Say: stream alpha","stream":true,"options":{"num_predict":6,"num_ctx":4096,"temperature":0}}'
payload_b='{"model":"smoke","prompt":"Say: stream beta","stream":true,"options":{"num_predict":6,"num_ctx":4096,"temperature":0}}'

curl -N -s -m 120 -X POST "${RUNTIME_URL}/api/generate" \
  -H 'Content-Type: application/json' \
  -d "$payload_a" > /tmp/phase15-sab-a.ndjson &
pid_a=$!
curl -N -s -m 120 -X POST "${RUNTIME_URL}/api/generate" \
  -H 'Content-Type: application/json' \
  -d "$payload_b" > /tmp/phase15-sab-b.ndjson &
pid_b=$!

wait "$pid_a" "$pid_b"

python3 -c "
import json, sys

def load_done(path):
    done = False
    text = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o.get('response'):
            text.append(o['response'])
        if o.get('done'):
            done = True
    assert done, path
    return ''.join(text).strip()

a = load_done('/tmp/phase15-sab-a.ndjson')
b = load_done('/tmp/phase15-sab-b.ndjson')
assert a, 'empty stream A'
assert b, 'empty stream B'
print('concurrent streams ok:', 'A=', a[:24], 'B=', b[:24])
"

post_health="$(curl -sf "${RUNTIME_URL}/health")"
python3 -c "
import json, sys
h = json.loads(sys.argv[1])
pre_flush, pre_batched = int(sys.argv[2]), int(sys.argv[3])
s = (h.get('kv_auto_batch') or {}).get('stream') or {}
flush = int(s.get('flush_count') or 0)
batched = int(s.get('batched_requests') or 0)
assert flush > pre_flush or batched >= pre_batched + 2, (
    f'expected stream auto-batch activity: flush {pre_flush}->{flush}, '
    f'batched {pre_batched}->{batched}'
)
print('post-smoke stream auto-batch:', f'flush_count={flush}', f'batched_requests={batched}')
" "$post_health" "$pre_flush" "$pre_batched"

smoke_runtime_assert_kv_snapshot "$RUNTIME_URL"

echo "PASS: phase15_stream_auto_batch_smoke"
