#!/usr/bin/env bash
# Phase 15 v2: in-process multi-seq shared context (llama_parallel_slots > 1).
#
# Uses temp YAML with llama_backend: inprocess and llama_parallel_slots: 2, then
# asserts /health kv_inprocess_n_seq_max and a successful generate (num_ctx capped
# for PA block pool — default autoconfig budget can exceed num_blocks*block_size).
#
#   export LLAMA_MODEL=/path/to/small.q8_0.gguf
#   export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so
#   ./scripts/phase15_inprocess_multiseq_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${LLAMA_MODEL:-}" ]]; then
  echo "Set LLAMA_MODEL to a small GGUF on this host" >&2
  exit 1
fi
if [[ -z "${LLAMA_CPP_LIB:-}" ]]; then
  echo "Set LLAMA_CPP_LIB for inprocess (ctypes libllama.so)" >&2
  exit 1
fi

pkill -f "${ROOT}/zerollama serve" 2>/dev/null || pkill -f './zerollama serve' 2>/dev/null || true
sleep 2

# shellcheck source=scripts/phase14_serve_env.sh
source "${ROOT}/scripts/phase14_serve_env.sh"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime_smoke_lib.sh"

RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
TMPYAML="$(mktemp /tmp/zerollama-phase15-multiseq-XXXX.yaml)"
cleanup() {
  pkill -f "${ROOT}/zerollama serve" 2>/dev/null || pkill -f './zerollama serve' 2>/dev/null || true
  rm -f "$TMPYAML"
}
trap cleanup EXIT

{
  sed -e 's/^# llama_backend: inprocess/llama_backend: inprocess/' \
      -e 's/^llama_parallel_slots: 1/llama_parallel_slots: 2/' \
    "${ROOT}/runtime/configs/single_gpu.yaml"
} >"$TMPYAML"

echo "== Phase 15 in-process multi-seq smoke =="

: > /tmp/zerollama-phase15-multiseq-serve.log
(
  cd "${ROOT}"
  env -u ZEROLLAMA_RUNTIME_URL -u ZEROLLAMA_RUNTIME_LLAMA_BACKEND \
    ZEROLLAMA_RUNTIME_CONFIG="$TMPYAML" \
    ZEROLLAMA_RUNTIME_EMBED="${ZEROLLAMA_RUNTIME_EMBED:-on}" \
    ZEROLLAMA_RUNTIME="${ZEROLLAMA_RUNTIME:-1}" \
    OLLAMA_NO_CLOUD="${OLLAMA_NO_CLOUD:-true}" \
    OLLAMA_TRAINING="${OLLAMA_TRAINING:-false}" \
    LLAMA_MODEL="${LLAMA_MODEL}" \
    LLAMA_CPP_LIB="${LLAMA_CPP_LIB}" \
    ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-$ROOT}" \
    ./zerollama serve >> /tmp/zerollama-phase15-multiseq-serve.log 2>&1
) &

backend=""
nseq=""
slots=""
for _ in $(seq 1 60); do
  if curl -sf -m 3 "${RUNTIME_URL}/health" -o /tmp/phase15-ms-health.json 2>/dev/null; then
    read -r backend nseq slots < <(
      python3 -c "
import json
h = json.load(open('/tmp/phase15-ms-health.json'))
ks = h.get('kv_scheduler') or {}
print(
    h.get('llama_backend') or '',
    h.get('kv_inprocess_n_seq_max') or '',
    ks.get('llama_parallel_slots') or '',
)
"
    )
    if [[ "$backend" == "inprocess" && "$nseq" == "2" ]]; then
      echo "serve ready: kv_inprocess_n_seq_max=${nseq} llama_parallel_slots=${slots}"
      break
    fi
  fi
  sleep 2
done

if [[ "${nseq:-}" != "2" ]]; then
  echo "expected kv_inprocess_n_seq_max=2; health:" >&2
  cat /tmp/phase15-ms-health.json >&2 || true
  tail -40 /tmp/zerollama-phase15-multiseq-serve.log >&2
  exit 1
fi

runtime_resume_if_needed "$(cat /tmp/phase15-ms-health.json)"

gen_payload='{"model":"smoke","prompt":"Say: ok","stream":false,"options":{"num_predict":4,"num_ctx":4096,"temperature":0.7}}'
gen_code=$(curl -s -o /tmp/phase15-ms-gen.json -w '%{http_code}' -X POST "${RUNTIME_URL}/api/generate" \
  -H 'Content-Type: application/json' -d "$gen_payload")
if [[ "$gen_code" != "200" ]]; then
  echo "HTTP ${gen_code} /api/generate:" >&2
  head -c 400 /tmp/phase15-ms-gen.json >&2
  echo >&2
  exit 1
fi
gen_json=$(cat /tmp/phase15-ms-gen.json)
python3 -c "
import json, sys
d = json.loads(sys.argv[1])
assert d.get('done') and d.get('response'), d
steps = d.get('kv_decode_steps')
assert steps is not None and int(steps) > 0, d
print('generate ok, kv_decode_steps=', steps)
" "$gen_json"

post_health=$(curl -sf "${RUNTIME_URL}/health")
python3 -c "
import json, sys
h = json.loads(sys.argv[1])
assert h.get('kv_inprocess_n_seq_max') == 2, h.get('kv_inprocess_n_seq_max')
kd = h.get('kv_decode_steps') or {}
assert kd.get('active') is True, kd
print('post-generate kv_decode_steps:', kd.get('value'))
" "$post_health"

echo "PASS: phase15_inprocess_multiseq_smoke"
