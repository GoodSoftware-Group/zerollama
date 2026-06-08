#!/usr/bin/env bash
# Phase 15 Apple Silicon sign-off (uv sidecar + inprocess Metal).
#
# Mirrors phase15_inprocess_signoff.sh without embed serve (Mac system Python is often 3.9).
#
# Prerequisite:
#   LLAMA_CPP_ROOT=../llama.cpp ./scripts/build_llama_server.sh
#
# Usage:
#   ./scripts/phase15_metal_signoff.sh
#
# Env:
#   M3_LLAMA_MODEL       — GGUF blob (default: auto-pick smallest text GGUF)
#   PHASE15_SKIP_BOOT=1  — skip step-1 sidecar+go start (M3 chain); step-2 multiseq still reloads
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/macos_runtime_serve_lib.sh
source "${ROOT}/scripts/macos_runtime_serve_lib.sh"

LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-${ROOT}/../llama.cpp}"
export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_CPP_ROOT}/build/bin/libllama.dylib}"
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${LLAMA_CPP_ROOT}/build/bin/llama-server}"

smoke_m3_resolve_signoff_model

if [[ ! -f "${LLAMA_CPP_LIB}" ]]; then
  echo "Missing ${LLAMA_CPP_LIB}; run ./scripts/build_llama_server.sh" >&2
  exit 1
fi

macos_runtime_urls
RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL}"

TMPYAML=""
_phase15_cleanup() {
  rm -f "${TMPYAML:-}"
  macos_runtime_sidecar_cleanup
}
trap _phase15_cleanup EXIT INT TERM

echo "== Phase 15 Metal in-process sign-off =="

echo ""
echo "== [1/2] KV decode hook (single-seq, apple_silicon.yaml) =="
if [[ "${PHASE15_SKIP_BOOT:-0}" != "1" ]]; then
  macos_runtime_start_sidecar "$LLAMA_MODEL" "" 1
  macos_runtime_start_go
fi
"${ROOT}/scripts/phase14_yaml_config_smoke.sh"
smoke_runtime_assert_kv_snapshot "$RUNTIME_URL"
echo "PASS: phase15 metal kv hook"

echo ""
echo "== [2/2] multi-seq shared context (llama_parallel_slots=2) =="
TMPYAML="$(mktemp /tmp/zerollama-phase15-metal-multiseq-XXXX.yaml)"
sed -e 's/^llama_parallel_slots: 1/llama_parallel_slots: 2/' \
  "${ROOT}/runtime/configs/apple_silicon.yaml" >"$TMPYAML"

macos_runtime_start_sidecar "$LLAMA_MODEL" "$TMPYAML" 0

nseq=""
slots=""
for _ in $(seq 1 30); do
  if curl -sf -m 3 "${RUNTIME_URL}/health" -o /tmp/phase15-metal-ms-health.json 2>/dev/null; then
    read -r nseq slots < <(
      python3 -c "
import json
h = json.load(open('/tmp/phase15-metal-ms-health.json'))
ks = h.get('kv_scheduler') or {}
print(h.get('kv_inprocess_n_seq_max') or '', ks.get('llama_parallel_slots') or '')
"
    )
    if [[ "$nseq" == "2" ]]; then
      echo "sidecar ready: kv_inprocess_n_seq_max=${nseq} llama_parallel_slots=${slots}"
      break
    fi
  fi
  sleep 1
done

if [[ "${nseq:-}" != "2" ]]; then
  echo "expected kv_inprocess_n_seq_max=2; health:" >&2
  cat /tmp/phase15-metal-ms-health.json >&2 || true
  exit 1
fi

runtime_resume_if_needed "$(cat /tmp/phase15-metal-ms-health.json)"
gen_payload='{"model":"smoke","prompt":"Say: ok","stream":false,"options":{"num_predict":4,"num_ctx":4096,"temperature":0.7}}'
gen_code=$(curl -s -o /tmp/phase15-metal-ms-gen.json -w '%{http_code}' -X POST "${RUNTIME_URL}/api/generate" \
  -H 'Content-Type: application/json' -d "$gen_payload")
if [[ "$gen_code" != "200" ]]; then
  echo "HTTP ${gen_code} /api/generate:" >&2
  head -c 400 /tmp/phase15-metal-ms-gen.json >&2
  echo >&2
  exit 1
fi
python3 -c "
import json, sys
d = json.loads(sys.argv[1])
assert d.get('done') and d.get('response'), d
steps = d.get('kv_decode_steps')
assert steps is not None and int(steps) > 0, d
print('generate ok, kv_decode_steps=', steps)
" "$(cat /tmp/phase15-metal-ms-gen.json)"

post_health=$(curl -sf "${RUNTIME_URL}/health")
python3 -c "
import json, sys
h = json.loads(sys.argv[1])
assert h.get('kv_inprocess_n_seq_max') == 2, h.get('kv_inprocess_n_seq_max')
kd = h.get('kv_decode_steps') or {}
assert kd.get('active') is True, kd
print('post-generate kv_decode_steps:', kd.get('value'), 'n_seq_max=2')
" "$post_health"

smoke_runtime_assert_kv_snapshot "$RUNTIME_URL"
echo "PASS: phase15 metal multiseq"

echo ""
echo "PASS: phase15_metal_signoff"
