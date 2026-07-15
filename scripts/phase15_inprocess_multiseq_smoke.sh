#!/usr/bin/env bash
# Phase 15 v2: in-process multi-seq shared context (llama_parallel_slots > 1).
#
# Uses temp YAML with llama_backend: inprocess and llama_parallel_slots: 2, then
# asserts /health kv_inprocess_n_seq_max and a successful generate (num_ctx capped
# for PA block pool — default autoconfig budget can exceed num_blocks*block_size).
#
# WHY ZEROLLAMA_GPU_PROFILE=0: L1 profiles override YAML llama_parallel_slots.
# Edge binaries cannot embed — auto-switch to Linux uv sidecar (PHASE15_USE_SIDECAR=1).
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

# shellcheck source=scripts/phase14_serve_env.sh
source "${ROOT}/scripts/phase14_serve_env.sh"
# shellcheck source=scripts/phase15_runtime_kv_env.sh
source "${ROOT}/scripts/phase15_runtime_kv_env.sh"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime_smoke_lib.sh"
# shellcheck source=scripts/linux_runtime_serve_lib.sh
source "${ROOT}/scripts/linux_runtime_serve_lib.sh"

phase15_runtime_kv_env_apply
phase15_runtime_auto_batch_env_apply

ZEROLLAMA_BIN="${ZEROLLAMA_BIN:-${ROOT}/zerollama}"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:18083}"
ZEROLLAMA_RUNTIME_EMBED_PORT="${ZEROLLAMA_RUNTIME_EMBED_PORT:-18081}"
export OLLAMA_HOST ZEROLLAMA_RUNTIME_EMBED_PORT ZEROLLAMA_BIN
RUNTIME_URL="http://127.0.0.1:${ZEROLLAMA_RUNTIME_EMBED_PORT}"
TMPYAML="$(mktemp /tmp/zerollama-phase15-multiseq-XXXX.yaml)"

_use_sidecar=0
_ver_out="$("${ZEROLLAMA_BIN}" -v 2>&1 || true)"
if [[ "${PHASE15_USE_SIDECAR:-}" == "1" ]]; then
  _use_sidecar=1
elif [[ "${_ver_out}" == *"edge build: true"* ]]; then
  _use_sidecar=1
fi

_GO_PID=""
cleanup() {
  [[ -n "${_GO_PID}" ]] && kill "${_GO_PID}" 2>/dev/null || true
  if [[ "${_use_sidecar}" -eq 1 ]]; then
    linux_runtime_sidecar_cleanup
    linux_runtime_stop_sidecar_port
  else
    pkill -f "${ZEROLLAMA_BIN} serve" 2>/dev/null || pkill -f "${ROOT}/zerollama serve" 2>/dev/null || true
  fi
  rm -f "$TMPYAML"
}
trap cleanup EXIT

{
  sed -e 's/^# llama_backend: inprocess/llama_backend: inprocess/' \
      -e 's/^llama_parallel_slots: 1/llama_parallel_slots: 2/' \
    "${ROOT}/runtime/configs/single_gpu.yaml"
} >"$TMPYAML"

echo "== Phase 15 in-process multi-seq smoke =="
echo "OLLAMA_HOST=${OLLAMA_HOST} runtime=${RUNTIME_URL} mode=$([ "${_use_sidecar}" -eq 1 ] && echo sidecar || echo embed)"

if [[ "${_use_sidecar}" -eq 1 ]]; then
  export ZEROLLAMA_RUNTIME_URL="${RUNTIME_URL}"
  export ZEROLLAMA_RUNTIME_EMBED=0
  # YAML sets inprocess; do not force env override over slots YAML.
  unset ZEROLLAMA_RUNTIME_LLAMA_BACKEND || true
  export ZEROLLAMA_GPU_PROFILE=0
  export LINUX_RT_LOG="${LINUX_RT_LOG:-/tmp/zerollama-phase15-multiseq-runtime.log}"
  linux_runtime_urls
  linux_runtime_stop_sidecar_port
  linux_runtime_start_sidecar "${LLAMA_MODEL}" "$TMPYAML"

  : > /tmp/zerollama-phase15-multiseq-serve.log
  (
    cd "${ROOT}"
    env \
      ZEROLLAMA_RUNTIME_URL="${RUNTIME_URL}" \
      ZEROLLAMA_RUNTIME_EMBED=0 \
      ZEROLLAMA_RUNTIME="${ZEROLLAMA_RUNTIME:-1}" \
      ZEROLLAMA_GPU_PROFILE=0 \
      OLLAMA_HOST="${OLLAMA_HOST}" \
      OLLAMA_NO_CLOUD="${OLLAMA_NO_CLOUD:-true}" \
      OLLAMA_TRAINING="${OLLAMA_TRAINING:-false}" \
      ZEROLLAMA_EDGE="${ZEROLLAMA_EDGE:-0}" \
      LLAMA_MODEL="${LLAMA_MODEL}" \
      LLAMA_CPP_LIB="${LLAMA_CPP_LIB}" \
      ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-$ROOT}" \
      "${ZEROLLAMA_BIN}" serve >> /tmp/zerollama-phase15-multiseq-serve.log 2>&1
  ) &
  _GO_PID=$!
  for _ in $(seq 1 30); do
    if curl -sf -m 3 "${OLLAMA_HOST%/}/api/tags" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
else
  pkill -f "${ROOT}/zerollama serve" 2>/dev/null || pkill -f './zerollama serve' 2>/dev/null || true
  sleep 2
  : > /tmp/zerollama-phase15-multiseq-serve.log
  (
    cd "${ROOT}"
    env -u ZEROLLAMA_RUNTIME_URL -u ZEROLLAMA_RUNTIME_LLAMA_BACKEND \
      ZEROLLAMA_GPU_PROFILE=0 \
      ZEROLLAMA_RUNTIME_KV_NATIVE="${ZEROLLAMA_RUNTIME_KV_NATIVE:-1}" \
      ZEROLLAMA_KV_NATIVE_DECODE="${ZEROLLAMA_KV_NATIVE_DECODE:-1}" \
      ZEROLLAMA_KV_NATIVE_SAMPLE="${ZEROLLAMA_KV_NATIVE_SAMPLE:-1}" \
      ZEROLLAMA_KV_AUTO_BATCH="${ZEROLLAMA_KV_AUTO_BATCH:-0}" \
      ZEROLLAMA_KV_AUTO_BATCH_STREAM="${ZEROLLAMA_KV_AUTO_BATCH_STREAM:-0}" \
      ZEROLLAMA_RUNTIME_CONFIG="$TMPYAML" \
      ZEROLLAMA_RUNTIME_EMBED="${ZEROLLAMA_RUNTIME_EMBED:-on}" \
      ZEROLLAMA_RUNTIME_EMBED_PORT="${ZEROLLAMA_RUNTIME_EMBED_PORT}" \
      ZEROLLAMA_RUNTIME="${ZEROLLAMA_RUNTIME:-1}" \
      OLLAMA_HOST="${OLLAMA_HOST}" \
      OLLAMA_NO_CLOUD="${OLLAMA_NO_CLOUD:-true}" \
      OLLAMA_TRAINING="${OLLAMA_TRAINING:-false}" \
      ZEROLLAMA_EDGE="${ZEROLLAMA_EDGE:-0}" \
      LLAMA_MODEL="${LLAMA_MODEL}" \
      LLAMA_CPP_LIB="${LLAMA_CPP_LIB}" \
      ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-$ROOT}" \
      "${ZEROLLAMA_BIN}" serve >> /tmp/zerollama-phase15-multiseq-serve.log 2>&1
  ) &
  _GO_PID=$!
fi

backend=""
nseq=""
slots=""
for _ in $(seq 1 90); do
  if curl -sf -m 15 "${RUNTIME_URL}/health" -o /tmp/phase15-ms-health.json 2>/dev/null; then
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
  [[ -f "${LINUX_RT_LOG:-}" ]] && tail -40 "${LINUX_RT_LOG}" >&2
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
print('post-generate kv_decode_steps:', kd.get('value'), 'n_seq_max=2')
" "$post_health"

smoke_runtime_assert_kv_snapshot "$RUNTIME_URL"
smoke_runtime_assert_migration_summary "$RUNTIME_URL" 1

echo ""
echo "== [3/3] continuous batch decode (generate_batch + stream) =="
export ZEROLLAMA_RUNTIME_URL="${RUNTIME_URL}"
export RUNTIME_URL="${RUNTIME_URL}"
"${ROOT}/scripts/phase15_batch_decode_smoke.sh"

echo "PASS: phase15_inprocess_multiseq_smoke"
