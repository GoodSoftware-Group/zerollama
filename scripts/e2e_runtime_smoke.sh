#!/usr/bin/env bash
# Smoke-test zerollama Python runtime (see docs/testing-smoke.md for WHYs).
#
#   ./scripts/e2e_runtime_smoke.sh
#   RUN_E2E_GPU=1 LLAMA_MODEL=... LLAMA_SERVER_BIN=... ./scripts/e2e_runtime_smoke.sh
#   RUN_E2E_PROXY=1 OLLAMA_HOST=http://127.0.0.1:8080 ./scripts/e2e_runtime_smoke.sh
#
# Why LLAMA_* on the serve process: embedded runtime reads env at startup, not this shell.
# Why auto /internal/inference/resume: training-handoff leaves inference unloaded (503).
# Why X-Zerollama-Runtime on proxy: default "smoke" is not a pulled name; set RUN_E2E_PROXY_MODEL to a local tag for Phase 9 manifest gguf.
# RUN_E2E_VRAM_CLAMP=1: assert /health vram_num_ctx_policy.clamp_enabled after GPU steps (serve must set VRAM_CLAMP_NUM_CTX).
# RUN_E2E_TOOLS=1: POST /api/chat with tools (runtime path; HTTP must be 200 — 501 legacy is a failure).
# RUN_E2E_LEGACY=1: ggml /api/generate (needs RUN_E2E_LEGACY_MODEL). Use with RUN_E2E_LEGACY_ONLY=1
#   for legacy-only, or RUN_E2E_GPU/PROXY=0. gpu_smoke_all runs legacy after runtime via LEGACY_ONLY.
# RUN_E2E_LEGACY_ONLY=1: health + legacy only (gpu_smoke_all after runtime/proxy steps).
# RUN_E2E_PHASE14=1: /internal/tokenize + sampling; render-chat (needs RUN_E2E_PROXY_MODEL);
#   also X-Zerollama-Runtime on Go proxy (smoke-only; ZEROLLAMA_RUNTIME != OLLAMA_RUNTIME_ALL).
#   Why header on proxy: sign-off must exercise runtime + tokenize render, not accidental
#   ggml for pulled model names that are not runtime-default eligible.
# RUN_E2E_INPROCESS=1 / RUN_E2E_LLAMA_CPP_PYTHON=1: assert /health llama_backend matches (serve env).
# RUN_E2E_LLAMA_CPP_PYTHON_GPU=1: after GPU generate, assert /health llama_cpp.gpu_mode=gpu (wheel only).
# RUN_E2E_INPROCESS=1: assert /health llama_backend=inprocess; after generate assert kv_decode_steps active.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime_smoke_lib.sh"

RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
OLLAMA_URL="${OLLAMA_HOST:-http://127.0.0.1:8080}"
# Stream smokes: cap wait so a wedged SSE proxy cannot block CI for 20+ minutes.
SMOKE_STREAM_MAX="${RUN_E2E_STREAM_MAX:-120}"

# Optional: override GGUF for generate when daemon LLAMA_MODEL is too large for free VRAM.
# Example: RUN_E2E_GGUF=/path/to/small.q8_0.gguf RUN_E2E_GPU=1 ./scripts/e2e_runtime_smoke.sh
smoke_generate_options() {
  python3 -c 'import json, os
o = {"num_predict": 8, "num_ctx": int(os.environ.get("RUN_E2E_NUM_CTX", "4096"))}
g = os.environ.get("RUN_E2E_GGUF", "").strip()
if g:
    o["gguf"] = g
print(json.dumps(o))'
}

curl_runtime() {
  local method="$1"
  local path="$2"
  local data="${3:-}"
  local tmp
  tmp=$(mktemp)
  local code
  if [[ -n "$data" ]]; then
    code=$(curl -sS -o "$tmp" -w "%{http_code}" -X "$method" \
      -H 'Content-Type: application/json' \
      -d "$data" "${RUNTIME_URL}${path}")
  else
    code=$(curl -sS -o "$tmp" -w "%{http_code}" -X "$method" "${RUNTIME_URL}${path}")
  fi
  if [[ "$code" != "200" ]]; then
    echo "HTTP ${code} ${path}:" >&2
    cat "$tmp" >&2
    echo >&2
    rm -f "$tmp"
    exit 1
  fi
  cat "$tmp"
  rm -f "$tmp"
}

run_legacy_smoke() {
  runtime_resume_if_needed "$(runtime_fetch_health "$RUNTIME_URL")" "$RUNTIME_URL"
  local legacy_model="${RUN_E2E_LEGACY_MODEL:-}"
  if [[ -z "$legacy_model" ]]; then
    echo "RUN_E2E_LEGACY=1 requires RUN_E2E_LEGACY_MODEL (pulled local tag)" >&2
    return 1
  fi
  if [[ "$(uname -s)" == "Darwin" && "${RUN_E2E_LEGACY_FORCE:-0}" != "1" ]]; then
    local rt_llama
    rt_llama=$(runtime_fetch_health "$RUNTIME_URL" | python3 -c 'import json,sys; print("1" if json.load(sys.stdin).get("llama_server") else "")' 2>/dev/null || true)
    if [[ "$rt_llama" == "1" ]]; then
      echo "skip legacy ggml on darwin while runtime holds Metal (llama_server=true); set RUN_E2E_LEGACY_FORCE=1 to override"
      return 0
    fi
  fi
  echo "== zerollama legacy /api/generate (ggml) =="
  local legacy_opts
  legacy_opts=$(smoke_generate_options)
  local legacy_payload
  legacy_payload=$(python3 -c "import json,sys; print(json.dumps({
      'model': sys.argv[1],
      'prompt': 'Say: ok',
      'stream': False,
      'options': json.loads(sys.argv[2]),
  }))" "$legacy_model" "$legacy_opts")
  local code
  code=$(curl -sS -o /tmp/ollama_legacy_gen.json -w "%{http_code}" -X POST \
    -H 'Content-Type: application/json' \
    -d "$legacy_payload" \
    "${OLLAMA_URL}/api/generate")
  if [[ "$code" != "200" ]]; then
    echo "HTTP ${code} ${OLLAMA_URL}/api/generate (legacy ${legacy_model}):" >&2
    cat /tmp/ollama_legacy_gen.json >&2
    return 1
  fi
  python3 -c "import json; d=json.load(open('/tmp/ollama_legacy_gen.json')); assert d.get('done'), d"
  echo "legacy generate: ok (${legacy_model})"
}

echo "== runtime health =="
health_json=$(curl_runtime GET /health)
python3 -c "import sys,json; d=json.loads(sys.argv[1]); assert d.get('status')=='ok', d" "$health_json"
_phase14_strict=""
[[ "${RUN_E2E_PHASE14:-0}" == "1" ]] && _phase14_strict=strict
llama_backend=$(smoke_runtime_llama_backend "$health_json" "$_phase14_strict")
echo "llama_backend=${llama_backend}"
if [[ "${RUN_E2E_PHASE14:-0}" == "1" ]]; then
  llama_backend_source=$(smoke_runtime_llama_backend_source "$health_json" "$_phase14_strict")
  echo "llama_backend_source=${llama_backend_source}"
  smoke_runtime_assert_llama_backend_source "$health_json" "${RUN_E2E_LLAMA_BACKEND_SOURCE:-}"
fi

if [[ "${RUN_E2E_INPROCESS:-0}" == "1" ]]; then
  if [[ "$llama_backend" != "inprocess" ]]; then
    echo "RUN_E2E_INPROCESS=1 but /health llama_backend=${llama_backend} (set ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess or llama_backend: inprocess in runtime YAML on serve)" >&2
    exit 1
  fi
fi
if [[ "${RUN_E2E_LLAMA_CPP_PYTHON:-0}" == "1" ]]; then
  if [[ "$llama_backend" != "llama-cpp-python" ]]; then
    echo "RUN_E2E_LLAMA_CPP_PYTHON=1 but /health llama_backend=${llama_backend}" >&2
    exit 1
  fi
fi

if [[ "${RUN_E2E_GPU:-0}" == "1" ]] || [[ "${RUN_E2E_PROXY:-0}" == "1" ]]; then
  runtime_resume_if_needed "$health_json"
fi

if [[ "${RUN_E2E_PHASE14:-0}" == "1" ]]; then
  _gguf_preflight="${RUN_E2E_GGUF:-${LLAMA_MODEL:-}}"
  if [[ -z "$_gguf_preflight" ]]; then
    _gguf_preflight=$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('llama_model') or '')" "$health_json")
  fi
  smoke_runtime_require_phase14_endpoints "$RUNTIME_URL" "$_gguf_preflight"
fi

if [[ "${RUN_E2E_LEGACY_ONLY:-0}" == "1" ]]; then
  if [[ "${RUN_E2E_LEGACY:-0}" != "1" ]]; then
    echo "RUN_E2E_LEGACY_ONLY=1 requires RUN_E2E_LEGACY=1" >&2
    exit 1
  fi
  run_legacy_smoke
  echo "OK"
  exit 0
fi

# Legacy-only shortcut (no runtime/proxy steps in this invocation).
if [[ "${RUN_E2E_LEGACY:-0}" == "1" && "${RUN_E2E_GPU:-0}" != "1" && "${RUN_E2E_PROXY:-0}" != "1" ]]; then
  run_legacy_smoke
  echo "OK"
  exit 0
fi

if [[ "${RUN_E2E_LEGACY:-0}" == "1" ]]; then
  echo "RUN_E2E_LEGACY=1 with RUN_E2E_GPU or RUN_E2E_PROXY: use gpu_smoke_all (legacy after runtime)" >&2
  echo "  or set RUN_E2E_LEGACY=0 for runtime-only e2e" >&2
  exit 1
fi

if [[ "${RUN_E2E_GPU:-0}" == "1" ]]; then
  if [[ -z "${LLAMA_MODEL:-}" ]] && [[ -z "$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('llama_model') or '')" "$health_json")" ]]; then
    echo "RUN_E2E_GPU=1 requires LLAMA_MODEL on serve or in env" >&2
    exit 1
  fi
  if smoke_runtime_needs_server_bin "$llama_backend" && [[ -z "${LLAMA_SERVER_BIN:-}" ]]; then
    if [[ "${RUN_E2E_PHASE14:-0}" == "1" ]]; then
      echo "warn: LLAMA_SERVER_BIN unset in smoke shell (subprocess uses serve config; set if you want local parity)" >&2
    else
      echo "RUN_E2E_GPU=1 requires LLAMA_SERVER_BIN for subprocess backend (llama_backend=${llama_backend})" >&2
      exit 1
    fi
  fi
  llama_model_hint=$(smoke_llama_model_config_hint "$llama_backend")
  python3 -c "
import json, sys
h = json.loads(sys.argv[1])
if not h.get('llama_model'):
    print(
        'runtime has no LLAMA_MODEL configured (health llama_model is null).\\n'
        + sys.argv[2],
        file=sys.stderr,
    )
    sys.exit(1)
" "$health_json" "$llama_model_hint"

  gen_opts=$(smoke_generate_options)
  if [[ "${RUN_E2E_PHASE14:-0}" == "1" ]]; then
    gen_opts=$(python3 -c "import json,sys; o=json.loads(sys.argv[1]); o['temperature']=0.7; print(json.dumps(o))" "$gen_opts")
    echo "Phase 14: generate with options.temperature=0.7"
  fi
  if [[ -n "${RUN_E2E_GGUF:-}" ]]; then
    echo "generate options.gguf=${RUN_E2E_GGUF}"
  fi

  if [[ "${RUN_E2E_PHASE14:-0}" == "1" ]]; then
    gguf_tok="${RUN_E2E_GGUF:-${LLAMA_MODEL:-}}"
    if [[ -z "$gguf_tok" ]]; then
      gguf_tok=$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('llama_model') or '')" "$health_json")
    fi
    if [[ -n "$gguf_tok" && -f "$gguf_tok" ]]; then
      echo "== Phase 14 /internal/tokenize =="
      tok_payload=$(python3 -c "import json,sys; print(json.dumps({'gguf': sys.argv[1], 'text': 'Hello'}))" "$gguf_tok")
      tok_json=$(curl_runtime POST /internal/tokenize "$tok_payload")
      python3 -c "import json,sys; d=json.loads(sys.argv[1]); assert d.get('count',0) > 0 and d.get('tokens'), d" "$tok_json"
    else
      echo "skip /internal/tokenize (no GGUF path)"
    fi
  fi

  gguf_for_est="${RUN_E2E_GGUF:-${LLAMA_MODEL:-}}"
  if [[ -n "$gguf_for_est" && -f "$gguf_for_est" ]]; then
    echo "== runtime /internal/vram-estimate =="
    est_payload=$(python3 -c "import json,sys; print(json.dumps({'gguf': sys.argv[1], 'options': json.loads(sys.argv[2])}))" \
      "$gguf_for_est" "$gen_opts")
    est_json=$(curl_runtime POST /internal/vram-estimate "$est_payload")
    python3 -c "
import json, sys
b = json.loads(sys.argv[1])
bud = b.get('vram_budget') or {}
est = b.get('vram_estimate') or {}
assert est.get('gguf'), b
print('vram_estimate.required_per_gpu_bytes:', est.get('required_per_gpu_bytes'))
if bud.get('suggested_max_num_ctx') is not None:
    print('vram_budget.suggested_max_num_ctx:', bud.get('suggested_max_num_ctx'))
if bud.get('fits_with_margin') is not None:
    print('vram_budget.fits_with_margin:', bud.get('fits_with_margin'))
" "$est_json"
  fi

  echo "== runtime /api/generate (non-stream) =="
  gen_payload=$(python3 -c "import json,sys; print(json.dumps({'model':'smoke','prompt':'Say exactly: pong','stream':False,'options':json.loads(sys.argv[1])}))" "$gen_opts")
  gen_json=$(curl_runtime POST /api/generate "$gen_payload")
  python3 -c "import sys,json; d=json.loads(sys.argv[1]); assert d.get('done') and d.get('response'), d" "$gen_json"
  if [[ "${RUN_E2E_INPROCESS:-0}" == "1" ]]; then
    python3 -c "
import json, os, sys
d = json.loads(sys.argv[1])
steps = d.get('kv_decode_steps')
assert steps is not None, f'inprocess generate missing kv_decode_steps: {d!r}'
assert int(steps) > 0, f'expected kv_decode_steps > 0, got {steps!r}'
print('kv_decode_steps (generate):', steps)
" "$gen_json"
  fi

  echo "== runtime /api/chat (non-stream, no tools) =="
  chat_payload=$(python3 -c "import json,sys; print(json.dumps({
    'model': 'smoke',
    'messages': [{'role': 'user', 'content': 'Say: hi'}],
    'stream': False,
    'options': json.loads(sys.argv[1]),
  }))" "$gen_opts")
  chat_json=$(curl_runtime POST /api/chat "$chat_payload")
  python3 -c "
import sys, json
d = json.loads(sys.argv[1])
assert d.get('done'), d
msg = d.get('message') or {}
assert msg.get('content'), d
" "$chat_json"

  echo "== runtime /v1/chat/completions (non-stream) =="
  v1_payload=$(python3 -c "import json,sys; print(json.dumps({
    'model': 'smoke',
    'messages': [{'role': 'user', 'content': 'Say: hi'}],
    'stream': False,
    'max_tokens': 8,
    'options': json.loads(sys.argv[1]),
  }))" "$gen_opts")
  v1_json=$(curl_runtime POST /v1/chat/completions "$v1_payload")
  python3 -c "
import sys, json
d = json.loads(sys.argv[1])
assert d.get('object') == 'chat.completion', d
choices = d.get('choices') or []
assert choices, d
msg = (choices[0].get('message') or {})
assert msg.get('content'), d
" "$v1_json"

  echo "== runtime /api/generate (stream) =="
  stream_payload=$(python3 -c "import json,sys; print(json.dumps({'model':'smoke','prompt':'Say: hi','stream':True,'options':json.loads(sys.argv[1])}))" "$gen_opts")
  stream_tmp=$(mktemp)
  code=$(curl -sS -m "${SMOKE_STREAM_MAX}" -o "$stream_tmp" -w "%{http_code}" -X POST \
    -H 'Content-Type: application/json' \
    -d "$stream_payload" \
    "${RUNTIME_URL}/api/generate")
  if [[ "$code" != "200" ]]; then
    echo "HTTP ${code} /api/generate (stream):" >&2
    cat "$stream_tmp" >&2
    rm -f "$stream_tmp"
    exit 1
  fi
  lines=$(wc -l <"$stream_tmp")
  rm -f "$stream_tmp"
  if [[ "${lines}" -lt 1 ]]; then
    echo "expected stream lines" >&2
    exit 1
  fi

  echo "== runtime /api/chat (stream) =="
  chat_stream_payload=$(python3 -c "import json,sys; print(json.dumps({
    'model': 'smoke',
    'messages': [{'role': 'user', 'content': 'Say: hi'}],
    'stream': True,
    'options': json.loads(sys.argv[1]),
  }))" "$gen_opts")
  chat_stream_tmp=$(mktemp)
  code=$(curl -sS -m "${SMOKE_STREAM_MAX}" -o "$chat_stream_tmp" -w "%{http_code}" -X POST \
    -H 'Content-Type: application/json' \
    -d "$chat_stream_payload" \
    "${RUNTIME_URL}/api/chat")
  if [[ "$code" != "200" ]]; then
    echo "HTTP ${code} /api/chat (stream):" >&2
    cat "$chat_stream_tmp" >&2
    rm -f "$chat_stream_tmp"
    exit 1
  fi
  python3 -c "
import json, sys
found = False
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)
    if d.get('message') or d.get('done'):
        found = True
        break
if not found:
    raise SystemExit('no chat stream chunks')
" "$chat_stream_tmp"
  rm -f "$chat_stream_tmp"

  echo "== runtime /v1/chat/completions (stream) =="
  v1_stream_payload=$(python3 -c "import json,sys; print(json.dumps({
    'model': 'smoke',
    'messages': [{'role': 'user', 'content': 'Say: hi'}],
    'stream': True,
    'max_tokens': 8,
    'options': json.loads(sys.argv[1]),
  }))" "$gen_opts")
  v1_stream_tmp=$(mktemp)
  code=$(curl -sS -m "${SMOKE_STREAM_MAX}" -o "$v1_stream_tmp" -w "%{http_code}" -X POST \
    -H 'Content-Type: application/json' \
    -d "$v1_stream_payload" \
    "${RUNTIME_URL}/v1/chat/completions")
  if [[ "$code" != "200" ]]; then
    echo "HTTP ${code} /v1/chat/completions (stream):" >&2
    cat "$v1_stream_tmp" >&2
    rm -f "$v1_stream_tmp"
    exit 1
  fi
  python3 -c "
import json, sys
found = False
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith('data:'):
        continue
    payload = line[5:].strip()
    if payload == '[DONE]':
        continue
    d = json.loads(payload)
    if d.get('object') == 'chat.completion.chunk':
        found = True
        break
if not found:
    raise SystemExit('no v1 SSE chunks')
" "$v1_stream_tmp"
  rm -f "$v1_stream_tmp"

  if [[ "${RUN_E2E_TOOLS:-0}" == "1" ]]; then
    runtime_resume_if_needed "$(runtime_fetch_health "$RUNTIME_URL")" "$RUNTIME_URL"
    echo "== runtime /api/chat (tools, non-stream) =="
    tools_payload=$(python3 -c "import json,sys; print(json.dumps({
        'model': 'smoke',
        'messages': [{'role': 'user', 'content': 'What is 2+2?'}],
        'stream': False,
        'tools': [{
            'type': 'function',
            'function': {
                'name': 'calculator',
                'description': 'math',
                'parameters': {'type': 'object', 'properties': {}},
            },
        }],
        'options': json.loads(sys.argv[1]),
    }))" "$gen_opts")
    tools_json=$(curl_runtime POST /api/chat "$tools_payload")
    python3 -c "
import json, sys
d = json.loads(sys.argv[1])
assert d.get('done'), d
blob = json.dumps(d).lower()
assert 'legacy runner' not in blob, d
msg = d.get('message') or {}
assert msg.get('content') or msg.get('tool_calls'), d
print('tools chat: ok (runtime path, not legacy 501)')
" "$tools_json"

    echo "== runtime /v1/chat/completions (tools, non-stream) =="
    v1_tools_payload=$(python3 -c "import json,sys; print(json.dumps({
        'model': 'smoke',
        'messages': [{'role': 'user', 'content': 'What is 2+2?'}],
        'stream': False,
        'max_tokens': 8,
        'tools': [{
            'type': 'function',
            'function': {
                'name': 'calculator',
                'description': 'math',
                'parameters': {'type': 'object', 'properties': {}},
            },
        }],
        'options': json.loads(sys.argv[1]),
    }))" "$gen_opts")
    v1_tools_json=$(curl_runtime POST /v1/chat/completions "$v1_tools_payload")
    python3 -c "
import json, sys
d = json.loads(sys.argv[1])
assert d.get('object') == 'chat.completion', d
blob = json.dumps(d).lower()
assert 'legacy runner' not in blob, d
choices = d.get('choices') or []
msg = (choices[0].get('message') if choices else {}) or {}
assert msg.get('content') or msg.get('tool_calls'), d
print('v1 tools chat: ok (runtime path, not legacy 501)')
" "$v1_tools_json"
  fi

  echo "== runtime /health (post-generate calibration + vram policy) =="
  post_health=$(curl_runtime GET /health)
  python3 -c "
import json, os, sys
h = json.loads(sys.argv[1])
cal = h.get('vram_calibration')
if cal:
    if cal.get('suggested_estimate_factor') is not None:
        print('vram_calibration.suggested_estimate_factor:', cal.get('suggested_estimate_factor'))
        print('  (set ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR or rely on autotune persist)')
    if cal.get('observed_bytes') is not None:
        print('vram_calibration.observed_bytes:', cal.get('observed_bytes'))
at = h.get('vram_autotune') or {}
if at.get('enabled') and not at.get('pending_first_calibration'):
    print('vram_autotune: enabled')
elif at.get('pending_first_calibration'):
    print('vram_autotune: pending first probed load')
policy = h.get('vram_num_ctx_policy') or {}
if policy:
    print('vram_num_ctx_policy.clamp_enabled:', policy.get('clamp_enabled'))
if os.environ.get('RUN_E2E_VRAM_CLAMP', '').strip() == '1':
    assert policy.get('clamp_enabled') is True, (
        'RUN_E2E_VRAM_CLAMP=1 requires ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX=auto or 1 on serve'
    )
if os.environ.get('RUN_E2E_LLAMA_CPP_PYTHON_GPU', '').strip() == '1':
    lcp = h.get('llama_cpp') or {}
    mode = lcp.get('gpu_mode')
    assert mode == 'gpu', f'RUN_E2E_LLAMA_CPP_PYTHON_GPU=1 expected gpu_mode=gpu, got {mode!r} llama_cpp={lcp!r}'
    assert lcp.get('loaded') is True, f'wheel GPU smoke expected loaded model, llama_cpp={lcp!r}'
    assert int(lcp.get('n_gpu_layers') or 0) > 0, lcp
    print('llama_cpp.gpu_mode: gpu (n_gpu_layers=%s)' % lcp.get('n_gpu_layers'))
elif os.environ.get('RUN_E2E_LLAMA_CPP_PYTHON', '').strip() == '1':
    lcp = h.get('llama_cpp') or {}
    if lcp:
        mode = lcp.get('gpu_mode')
        assert mode == 'cpu', f'wheel CPU smoke expected gpu_mode=cpu, got {mode!r} llama_cpp={lcp!r}'
        assert lcp.get('loaded') is True, f'wheel CPU smoke expected loaded model, llama_cpp={lcp!r}'
        print('llama_cpp.gpu_mode: cpu')
if os.environ.get('RUN_E2E_INPROCESS', '').strip() == '1':
    kd = h.get('kv_decode_steps') or {}
    assert kd.get('active') is True, f'inprocess /health kv_decode_steps inactive: {kd!r}'
    assert int(kd.get('value') or 0) > 0, kd
    print('kv_decode_steps (health):', kd.get('value'), 'source=', kd.get('source'))
" "$post_health"
  if [[ "${RUN_E2E_LLAMA_CPP_PYTHON_GPU:-0}" == "1" ]]; then
    smoke_runtime_assert_llama_cpp_gpu "$post_health" gpu
  elif [[ "${RUN_E2E_LLAMA_CPP_PYTHON:-0}" == "1" ]]; then
    smoke_runtime_assert_llama_cpp_gpu "$post_health" cpu
  fi

  if [[ "${RUN_E2E_VRAM_CLAMP:-0}" == "1" ]]; then
    echo "== runtime clamp behavior (high num_ctx request) =="
    clamp_payload=$(python3 -c "import json,sys; print(json.dumps({
        'model': 'smoke',
        'prompt': 'Say: ok',
        'stream': False,
        'options': {**json.loads(sys.argv[1]), 'num_ctx': 131072},
    }))" "$gen_opts")
    clamp_tmp=$(mktemp)
    clamp_code=$(curl -sS -o "$clamp_tmp" -w "%{http_code}" -X POST \
      -H 'Content-Type: application/json' \
      -d "$clamp_payload" \
      "${RUNTIME_URL}/api/generate")
    python3 -c "
import json, sys
code = int(sys.argv[1])
raw = open(sys.argv[2]).read()
if code == 200:
    d = json.loads(raw)
    meta = d.get('vram_num_ctx') or {}
    if meta.get('num_ctx_clamped'):
        print('vram_num_ctx: clamped', meta.get('num_ctx_clamped_from'), '->', meta.get('num_ctx'))
    else:
        print('warn: clamp enabled but response had no vram_num_ctx.num_ctx_clamped (estimate may already fit)')
elif code in (502, 503):
    print('clamp/budget rejected load (acceptable):', code, raw[:200])
else:
    print('unexpected HTTP', code, raw[:300], file=sys.stderr)
    sys.exit(1)
" "$clamp_code" "$clamp_tmp"
    rm -f "$clamp_tmp"
  fi
fi

if [[ "${RUN_E2E_PROXY:-0}" == "1" ]]; then
  runtime_resume_if_needed "$(runtime_fetch_health "$RUNTIME_URL")" "$RUNTIME_URL"
  echo "== zerollama proxy /api/generate =="
  proxy_model="${RUN_E2E_PROXY_MODEL:-smoke}"
  proxy_headers=(-H 'Content-Type: application/json')
  if [[ "$proxy_model" == "smoke" ]] || [[ "${RUN_E2E_PHASE14:-0}" == "1" ]] || [[ "${RUN_E2E_GPU:-0}" == "1" ]]; then
    # Ad-hoc name, Phase 14, or GPU smokes: force runtime proxy (ZEROLLAMA_RUNTIME=1 is
    # per-model default-on; pulled tags may still hit ggml without header on darwin).
    proxy_headers+=(-H 'X-Zerollama-Runtime: 1')
  fi
  proxy_opts=$(smoke_generate_options)
  if [[ -n "${RUN_E2E_GGUF:-}" ]]; then
    echo "proxy options.gguf=${RUN_E2E_GGUF}"
  fi
  proxy_payload=$(python3 -c "import json,sys; print(json.dumps({'model':sys.argv[1],'prompt':'hi','stream':False,'options':json.loads(sys.argv[2])}))" "$proxy_model" "$proxy_opts")
  code=$(curl -sS -o /tmp/ollama_gen.json -w "%{http_code}" -X POST \
    "${proxy_headers[@]}" \
    -d "$proxy_payload" \
    "${OLLAMA_URL}/api/generate")
  if [[ "$code" != "200" ]]; then
    echo "HTTP ${code} ${OLLAMA_URL}/api/generate:" >&2
    cat /tmp/ollama_gen.json >&2
    echo "hint: zerollama serve must be up with ZEROLLAMA_RUNTIME_EMBED or ZEROLLAMA_RUNTIME_URL set" >&2
    exit 1
  fi
  python3 -c "import json; d=json.load(open('/tmp/ollama_gen.json')); assert d.get('done'), d"

  echo "== zerollama proxy /api/chat =="
  chat_proxy_payload=$(python3 -c "import json,sys; print(json.dumps({
    'model': sys.argv[1],
    'messages': [{'role': 'user', 'content': 'Say: hi'}],
    'stream': False,
    'options': json.loads(sys.argv[2]),
  }))" "$proxy_model" "$proxy_opts")
  code=$(curl -sS -o /tmp/ollama_chat.json -w "%{http_code}" -X POST \
    "${proxy_headers[@]}" \
    -d "$chat_proxy_payload" \
    "${OLLAMA_URL}/api/chat")
  if [[ "$code" != "200" ]]; then
    echo "HTTP ${code} ${OLLAMA_URL}/api/chat:" >&2
    cat /tmp/ollama_chat.json >&2
    exit 1
  fi
  python3 -c "
import json
d = json.load(open('/tmp/ollama_chat.json'))
assert d.get('done'), d
msg = d.get('message') or {}
assert msg.get('content'), d
"

  if [[ "${RUN_E2E_PHASE14:-0}" == "1" && "$proxy_model" != "smoke" ]]; then
    echo "== Phase 14 Go /internal/render-chat =="
    render_payload=$(RUN_E2E_NUM_CTX="${RUN_E2E_NUM_CTX:-4096}" python3 -c "import json,os,sys; print(json.dumps({
        'model': sys.argv[1],
        'messages': [{'role': 'user', 'content': 'hello'}],
        'num_ctx': int(os.environ.get('RUN_E2E_NUM_CTX', '4096')),
        'num_predict': 32,
        'truncate': True,
    }))" "$proxy_model")
    render_code=$(curl -sS -o /tmp/ollama_render_chat.json -w "%{http_code}" -X POST \
      -H 'Content-Type: application/json' \
      -d "$render_payload" \
      "${OLLAMA_URL}/internal/render-chat")
    if [[ "$render_code" != "200" ]]; then
      echo "HTTP ${render_code} /internal/render-chat:" >&2
      cat /tmp/ollama_render_chat.json >&2
      exit 1
    fi
    python3 -c "
import json, sys
d = json.loads(open(sys.argv[1]).read())
mode = d.get('truncate_mode') or ''
assert d.get('prompt'), d
if sys.argv[2] == '1':
    assert mode == 'tokenize', f'Phase 14 expected truncate_mode=tokenize, got {mode!r}: {d}'
else:
    assert mode in ('tokenize', 'heuristic', 'none'), d
print('render-chat truncate_mode:', mode)
" /tmp/ollama_render_chat.json "${RUN_E2E_PHASE14:-0}"
  fi

  echo "== zerollama proxy /v1/chat/completions =="
  v1_proxy_payload=$(python3 -c "import json,sys; print(json.dumps({
    'model': sys.argv[1],
    'messages': [{'role': 'user', 'content': 'Say: hi'}],
    'stream': False,
    'max_tokens': 8,
    'options': json.loads(sys.argv[2]),
  }))" "$proxy_model" "$proxy_opts")
  code=$(curl -sS -o /tmp/ollama_v1.json -w "%{http_code}" -X POST \
    "${proxy_headers[@]}" \
    -d "$v1_proxy_payload" \
    "${OLLAMA_URL}/v1/chat/completions")
  if [[ "$code" != "200" ]]; then
    echo "HTTP ${code} ${OLLAMA_URL}/v1/chat/completions:" >&2
    cat /tmp/ollama_v1.json >&2
    exit 1
  fi
  python3 -c "
import json
d = json.load(open('/tmp/ollama_v1.json'))
assert d.get('object') == 'chat.completion', d
choices = d.get('choices') or []
assert choices and (choices[0].get('message') or {}).get('content'), d
"

  echo "== zerollama proxy /api/chat (stream) =="
  chat_stream_proxy=$(python3 -c "import json,sys; print(json.dumps({
    'model': sys.argv[1],
    'messages': [{'role': 'user', 'content': 'Say: hi'}],
    'stream': True,
    'options': json.loads(sys.argv[2]),
  }))" "$proxy_model" "$proxy_opts")
  chat_stream_tmp=$(mktemp)
  code=$(curl -sS -m "${SMOKE_STREAM_MAX}" -o "$chat_stream_tmp" -w "%{http_code}" -X POST \
    "${proxy_headers[@]}" \
    -d "$chat_stream_proxy" \
    "${OLLAMA_URL}/api/chat")
  if [[ "$code" != "200" ]]; then
    echo "HTTP ${code} ${OLLAMA_URL}/api/chat (stream):" >&2
    cat "$chat_stream_tmp" >&2
    rm -f "$chat_stream_tmp"
    exit 1
  fi
  python3 -c "
import json, sys
for line in open(sys.argv[1]):
    line = line.strip()
    if line and json.loads(line).get('message'):
        break
else:
    raise SystemExit('no proxy chat stream chunks')
" "$chat_stream_tmp"
  rm -f "$chat_stream_tmp"

  echo "== zerollama proxy /v1/chat/completions (stream) =="
  v1_stream_proxy=$(python3 -c "import json,sys; print(json.dumps({
    'model': sys.argv[1],
    'messages': [{'role': 'user', 'content': 'Say: hi'}],
    'stream': True,
    'max_tokens': 8,
    'options': json.loads(sys.argv[2]),
  }))" "$proxy_model" "$proxy_opts")
  v1_stream_tmp=$(mktemp)
  code=$(curl -sS -m "${SMOKE_STREAM_MAX}" -o "$v1_stream_tmp" -w "%{http_code}" -X POST \
    "${proxy_headers[@]}" \
    -d "$v1_stream_proxy" \
    "${OLLAMA_URL}/v1/chat/completions")
  if [[ "$code" != "200" ]]; then
    echo "HTTP ${code} ${OLLAMA_URL}/v1/chat/completions (stream):" >&2
    cat "$v1_stream_tmp" >&2
    rm -f "$v1_stream_tmp"
    exit 1
  fi
  python3 -c "
import json, sys
for line in open(sys.argv[1]):
    line = line.strip()
    if line.startswith('data:') and line[5:].strip() not in ('[DONE]', ''):
        d = json.loads(line[5:].strip())
        if d.get('object') == 'chat.completion.chunk':
            break
else:
    raise SystemExit('no proxy v1 SSE chunks')
" "$v1_stream_tmp"
  rm -f "$v1_stream_tmp"

  if [[ "${RUN_E2E_GPU:-0}" == "1" && "${RUN_E2E_TOOLS:-0}" == "1" ]]; then
    echo "== zerollama proxy /api/chat (tools, non-stream) =="
    tools_proxy_payload=$(python3 -c "import json,sys; print(json.dumps({
        'model': sys.argv[1],
        'messages': [{'role': 'user', 'content': 'What is 2+2?'}],
        'stream': False,
        'tools': [{
            'type': 'function',
            'function': {
                'name': 'calculator',
                'description': 'math',
                'parameters': {'type': 'object', 'properties': {}},
            },
        }],
        'options': json.loads(sys.argv[2]),
    }))" "$proxy_model" "$proxy_opts")
    code=$(curl -sS -o /tmp/ollama_tools.json -w "%{http_code}" -X POST \
      "${proxy_headers[@]}" \
      -d "$tools_proxy_payload" \
      "${OLLAMA_URL}/api/chat")
    if [[ "$code" != "200" ]]; then
      echo "HTTP ${code} ${OLLAMA_URL}/api/chat (tools):" >&2
      cat /tmp/ollama_tools.json >&2
      exit 1
    fi
    python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
assert d.get('done'), d
blob = json.dumps(d).lower()
assert 'legacy runner' not in blob, d
msg = d.get('message') or {}
assert msg.get('content') or msg.get('tool_calls'), d
print('proxy tools chat: ok (runtime path, not legacy 501)')
" /tmp/ollama_tools.json

    echo "== zerollama proxy /v1/chat/completions (tools, non-stream) =="
    v1_tools_proxy_payload=$(python3 -c "import json,sys; print(json.dumps({
        'model': sys.argv[1],
        'messages': [{'role': 'user', 'content': 'What is 2+2?'}],
        'stream': False,
        'max_tokens': 8,
        'tools': [{
            'type': 'function',
            'function': {
                'name': 'calculator',
                'description': 'math',
                'parameters': {'type': 'object', 'properties': {}},
            },
        }],
        'options': json.loads(sys.argv[2]),
    }))" "$proxy_model" "$proxy_opts")
    code=$(curl -sS -o /tmp/ollama_v1_tools.json -w "%{http_code}" -X POST \
      "${proxy_headers[@]}" \
      -d "$v1_tools_proxy_payload" \
      "${OLLAMA_URL}/v1/chat/completions")
    if [[ "$code" != "200" ]]; then
      echo "HTTP ${code} ${OLLAMA_URL}/v1/chat/completions (tools):" >&2
      cat /tmp/ollama_v1_tools.json >&2
      exit 1
    fi
    python3 -c "
import json
d = json.load(open('/tmp/ollama_v1_tools.json'))
assert d.get('object') == 'chat.completion', d
blob = json.dumps(d).lower()
assert 'legacy runner' not in blob, d
choices = d.get('choices') or []
msg = (choices[0].get('message') if choices else {}) or {}
assert msg.get('content') or msg.get('tool_calls'), d
print('proxy v1 tools chat: ok (runtime path, not legacy 501)')
"
  fi
fi

echo "OK"
