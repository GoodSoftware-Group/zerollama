#!/usr/bin/env bash
# Phase 15 Apple Silicon sign-off (uv sidecar + inprocess Metal).
#
# Why separate from phase15_inprocess_signoff.sh: Mac daily path uses uv sidecar, not
# embedded CPython (system Python is often 3.9 without torch).
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
#   PHASE15_SKIP_PHASE14=1 — skip embedded phase14_yaml_config_smoke (lab stack already validated)
#
# Multiseq step: temp YAML llama_parallel_slots=2 + ZEROLLAMA_GPU_PROFILE=0 — why: L1 128g
# profile sets n_parallel=8 and breaks kv_inprocess_n_seq_max=2 assertions.
# KV snapshot (step 1): smoke_runtime_assert_kv_snapshot accepts bound+tensor when vendor kv-ext
# is linked — why: linked _kv_native builds report status=bound after decode, not partial-only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/macos_runtime_serve_lib.sh
source "${ROOT}/scripts/macos_runtime_serve_lib.sh"
# shellcheck source=scripts/phase15_runtime_kv_env.sh
source "${ROOT}/scripts/phase15_runtime_kv_env.sh"

macos_export_llama_cpp_paths
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${LLAMA_CPP_ROOT}/build/bin/llama-server}"
phase15_runtime_kv_env_apply

if [[ "${PHASE15_BUILD_KV_EXT:-1}" == "1" ]]; then
  phase15_runtime_kv_ext_build
fi

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
echo "== [1/4] KV decode hook (single-seq, apple_silicon.yaml) =="
if [[ "${PHASE15_SKIP_BOOT:-0}" != "1" ]]; then
  macos_runtime_start_sidecar "$LLAMA_MODEL" "" 1
  macos_runtime_start_go
fi
if [[ "${PHASE15_SKIP_PHASE14:-0}" != "1" ]]; then
  "${ROOT}/scripts/phase14_yaml_config_smoke.sh"
else
  echo "skip phase14_yaml_config_smoke (PHASE15_SKIP_PHASE14=1)"
fi
smoke_runtime_assert_kv_snapshot "$RUNTIME_URL"
echo "PASS: phase15 metal kv hook"

echo ""
echo "== [2/5] multi-seq shared context (llama_parallel_slots=2) =="
TMPYAML="$(mktemp /tmp/zerollama-phase15-metal-multiseq-XXXX.yaml)"
sed -e 's/^llama_parallel_slots: 1/llama_parallel_slots: 2/' \
  "${ROOT}/runtime/configs/apple_silicon.yaml" >"$TMPYAML"

# Why disable L1 GPU profile here: apple-silicon-128g sets n_parallel=8, overriding yaml:2
# and breaking kv_inprocess_n_seq_max assertions for this multiseq gate.
phase15_runtime_auto_batch_env_apply
ZEROLLAMA_GPU_PROFILE=0 macos_runtime_start_sidecar "$LLAMA_MODEL" "$TMPYAML" 0

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
pb = h.get('kv_page_bind') or {}
assert pb.get('available') is True, pb
assert 'bind_level' in pb, pb
assert 'slots' in pb, pb
# WHY not assert tensor_pages_bound here: complete() unregisters page bind
# after _tensor_probe_after_decode; post-generate health has no running request.
print('post-generate kv_decode_steps:', kd.get('value'), 'kv_page_bind=', pb.get('bind_level'), 'n_seq_max=2')
" "$post_health"

smoke_runtime_assert_kv_snapshot "$RUNTIME_URL"
echo "PASS: phase15 metal multiseq"

echo ""
echo "== [2b/5] migration summary post-decode (v42/v43) =="
MIGRATION_SMOKE_SKIP_GEN=1 "${ROOT}/scripts/phase15_migration_summary_smoke.sh"

echo ""
echo "== [3/5] continuous batch decode (generate_batch + stream) =="
"${ROOT}/scripts/phase15_batch_decode_smoke.sh"

if [[ "${RUN_P15_AUTO_BATCH_ALL:-0}" == "1" ]]; then
  echo ""
  echo "== [3b/5] auto-batch sign-off (non-stream + stream) =="
  phase15_runtime_auto_batch_env_apply
  if [[ "${ZEROLLAMA_KV_AUTO_BATCH:-0}" != "1" || "${ZEROLLAMA_KV_AUTO_BATCH_STREAM:-0}" != "1" ]]; then
    echo "WARN: restart sidecar with ZEROLLAMA_KV_AUTO_BATCH=1 and ZEROLLAMA_KV_AUTO_BATCH_STREAM=1" >&2
    echo "      (metal signoff exports these when RUN_P15_AUTO_BATCH_ALL=1 before multiseq boot)" >&2
    exit 1
  fi
  "${ROOT}/scripts/phase15_auto_batch_signoff.sh"
elif [[ "${RUN_P15_STREAM_AUTO_BATCH:-0}" == "1" ]]; then
  echo ""
  echo "== [3b/5] stream auto-batch (concurrent /api/generate stream=true) =="
  if [[ "${ZEROLLAMA_KV_AUTO_BATCH_STREAM:-0}" != "1" ]]; then
    echo "WARN: ZEROLLAMA_KV_AUTO_BATCH_STREAM not set on sidecar — restart sidecar with env=1" >&2
    echo "      or export RUN_P15_STREAM_AUTO_BATCH=0 to skip this gate" >&2
    exit 1
  fi
  "${ROOT}/scripts/phase15_stream_auto_batch_smoke.sh"
fi

if [[ "${RUN_P15_AUTO_BATCH:-0}" == "1" ]]; then
  echo ""
  echo "== [3c/5] non-stream auto-batch (concurrent /api/generate stream=false) =="
  if [[ "${ZEROLLAMA_KV_AUTO_BATCH:-0}" != "1" ]]; then
    echo "WARN: ZEROLLAMA_KV_AUTO_BATCH not set on sidecar — restart sidecar with env=1" >&2
    echo "      or export RUN_P15_AUTO_BATCH=0 to skip this gate" >&2
    exit 1
  fi
  "${ROOT}/scripts/phase15_auto_batch_smoke.sh"
fi

echo ""
echo "== [4/5] L3 prompt_cache_key two-turn (in-process resume wiring) =="
l3_key="phase15-metal-$(date +%s)"
l3_turn1='{"model":"smoke","prompt":"System: helpful.\nUser: hi","stream":false,"options":{"num_predict":4,"num_ctx":4096,"temperature":0,"prompt_cache_key":"'"${l3_key}"'"}}'
l3_code=$(curl -s -o /tmp/phase15-metal-l3-t1.json -w '%{http_code}' -X POST "${RUNTIME_URL}/api/generate" \
  -H 'Content-Type: application/json' -d "$l3_turn1")
if [[ "$l3_code" != "200" ]]; then
  echo "HTTP ${l3_code} L3 turn 1:" >&2
  head -c 400 /tmp/phase15-metal-l3-t1.json >&2
  echo >&2
  exit 1
fi
l3_turn2='{"model":"smoke","prompt":"System: helpful.\nUser: hi\nAssistant: ok\nUser: follow up","stream":false,"options":{"num_predict":4,"num_ctx":4096,"temperature":0,"prompt_cache_key":"'"${l3_key}"'"}}'
l3_code2=$(curl -s -o /tmp/phase15-metal-l3-t2.json -w '%{http_code}' -X POST "${RUNTIME_URL}/api/generate" \
  -H 'Content-Type: application/json' -d "$l3_turn2")
if [[ "$l3_code2" != "200" ]]; then
  echo "HTTP ${l3_code2} L3 turn 2:" >&2
  head -c 400 /tmp/phase15-metal-l3-t2.json >&2
  echo >&2
  exit 1
fi
post_l3_health=$(curl -sf "${RUNTIME_URL}/health")
python3 -c "
import json, sys
h = json.loads(sys.argv[1])
kr = h.get('kv_resume') or {}
assert kr.get('active') is True, kr
owners = kr.get('owners_by_slot') or {}
assert len(owners) >= 1, kr
print('kv_resume active, owners_by_slot=', owners)
" "$post_l3_health"
echo "PASS: phase15 metal L3 two-turn"

echo ""
echo "== [5/5] tensor bind scaffold (page_bind_table + linked ext probe) =="
"${ROOT}/scripts/phase15_tensor_bind_probe.sh"

echo ""
echo "PASS: phase15_metal_signoff"
