#!/usr/bin/env bash
# Phase 15 v27–v30: continuous batch decode on GPU (multiseq sidecar must be up).
#
# WHY this smoke exists: pytest covers batch layout and engine wiring with mocks;
# Metal/Linux sign-off must prove linked ext + multiseq sidecar + real
# generate_batch/stream_generate_batch over HTTP (POST /internal/generate-batch).
#
# Requires:
#   - Running runtime at RUNTIME_URL (default :8081)
#   - kv_inprocess_n_seq_max >= 2
#   - Linked decode loop with batch_decode_in_c (phase15_runtime_kv_env.sh)
#
# Usage (after multiseq sidecar boot):
#   ./scripts/phase15_batch_decode_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime_smoke_lib.sh"

RUNTIME_URL="${RUNTIME_URL:-http://127.0.0.1:8081}"

echo "== Phase 15 batch decode smoke =="

health="$(curl -sf "${RUNTIME_URL}/health")"
python3 -c "
import json, sys
h = json.loads(sys.argv[1])
nseq = int(h.get('kv_inprocess_n_seq_max') or 0)
assert nseq >= 2, f'need multiseq sidecar, got kv_inprocess_n_seq_max={nseq}'
dl = h.get('kv_decode_loop') or {}
assert dl.get('available'), dl
assert dl.get('batch_decode_in_c'), dl
cb = h.get('kv_continuous_batch') or {}
print(
    'pre-batch health ok:',
    f'n_seq_max={nseq}',
    f'batch_decode_in_c={dl.get(\"batch_decode_in_c\")}',
    f'would_batch={cb.get(\"would_batch\")}',
)
" "$health"

batch_payload='{"prompts":["Say: alpha","Say: beta"],"n_predict":6,"max_admit":2,"stream":false,"options":{"num_ctx":4096,"temperature":0}}'
batch_code=$(curl -s -o /tmp/phase15-batch-gen.json -w '%{http_code}' \
  -X POST "${RUNTIME_URL}/internal/generate-batch" \
  -H 'Content-Type: application/json' \
  -d "$batch_payload")
if [[ "$batch_code" != "200" ]]; then
  echo "HTTP ${batch_code} /internal/generate-batch:" >&2
  head -c 600 /tmp/phase15-batch-gen.json >&2
  echo >&2
  exit 1
fi
python3 -c "
import json, sys
d = json.loads(open('/tmp/phase15-batch-gen.json').read())
results = d.get('results') or []
assert len(results) == 2, d
for r in results:
    assert (r.get('content') or '').strip(), r
print('batch generate ok:', [r.get('content', '')[:24] for r in results])
" 

stream_payload='{"prompts":["Count: one","Count: two"],"n_predict":4,"max_admit":2,"stream":true,"options":{"num_ctx":4096,"temperature":0}}'
stream_code=$(curl -s -o /tmp/phase15-batch-stream.ndjson -w '%{http_code}' \
  -X POST "${RUNTIME_URL}/internal/generate-batch" \
  -H 'Content-Type: application/json' \
  -d "$stream_payload")
if [[ "$stream_code" != "200" ]]; then
  echo "HTTP ${stream_code} streaming /internal/generate-batch:" >&2
  head -c 600 /tmp/phase15-batch-stream.ndjson >&2
  echo >&2
  exit 1
fi
python3 -c "
import json
chunks = []
for line in open('/tmp/phase15-batch-stream.ndjson'):
    line = line.strip()
    if not line:
        continue
    chunks.append(json.loads(line))
assert chunks, 'empty stream'
by_idx = {0: [], 1: []}
done = {0: False, 1: False}
for c in chunks:
    idx = int(c.get('seq_idx', 0))
    if c.get('done'):
        done[idx] = True
    else:
        by_idx.setdefault(idx, []).append(c.get('response') or '')
assert done.get(0) and done.get(1), (done, chunks[-3:])
assert any(''.join(by_idx[i]).strip() for i in (0, 1)), by_idx
print('batch stream ok: seq0=', ''.join(by_idx[0])[:20], 'seq1=', ''.join(by_idx[1])[:20])
"

post_health="$(curl -sf "${RUNTIME_URL}/health")"
python3 -c "
import json, sys
h = json.loads(sys.argv[1])
kd = h.get('kv_decode_steps') or {}
assert kd.get('active') is True, kd
steps = int(kd.get('value') or 0)
assert steps > 0, kd
print('post-batch kv_decode_steps=', steps)
" "$post_health"

smoke_runtime_assert_kv_snapshot "$RUNTIME_URL"

echo "PASS: phase15_batch_decode_smoke"
