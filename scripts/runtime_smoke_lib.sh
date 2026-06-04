# Shared helpers for GPU/runtime e2e scripts (source, do not execute).
#
# WHY this library exists: Go Phase-8 VRAM broker runs on runtime-proxy /api/generate, but
# training/admission can return 503 *before* the broker while a stale ggml runner still holds
# VRAM. We evict via the public unload API (keep_alive:0) instead of pkill so smokes match
# operator docs and avoid pgrep false positives on shell command lines.
# shellcheck shell=bash

# Match the ggml child process only (not shell snippets containing "zerollama runner").
# WHY: pgrep -f "zerollama runner" matched our own bash -lc test commands and looked like
# unload failed when /api/ps was already empty.
smoke_ggml_runner_running() {
  pgrep -f '/zerollama runner --' >/dev/null 2>&1
}

runtime_fetch_health() {
  local url="${1:-${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}}"
  local timeout="${2:-30}"
  curl -sf -m "$timeout" "${url%/}/health"
}

# Phase 14: read llama_backend from /health JSON (subprocess | inprocess | llama-cpp-python).
# Second arg "strict" fails when llama_backend is missing (stale serve binary).
smoke_runtime_llama_backend() {
  local strict="${2:-}"
  python3 -c "
import json, sys
strict = sys.argv[2] == 'strict'
b = (json.loads(sys.argv[1]).get('llama_backend') or '').strip()
if not b:
    msg = (
        '/health missing llama_backend — rebuild zerollama and restart serve '
        '(current tree includes Phase 14 runtime)'
    )
    if strict:
        print(msg, file=sys.stderr)
        sys.exit(1)
    print('subprocess', file=sys.stderr)
    print('warn: ' + msg, file=sys.stderr)
    b = 'subprocess'
print(b)
" "$1" "$strict"
}

# Phase 14: env vs YAML/config provenance from /health (strict fails when field missing).
smoke_runtime_llama_backend_source() {
  local strict="${2:-}"
  python3 -c "
import json, sys
strict = sys.argv[2] == 'strict'
src = (json.loads(sys.argv[1]).get('llama_backend_source') or '').strip()
if not src:
    msg = (
        '/health missing llama_backend_source — rebuild zerollama and restart serve '
        '(current tree includes Phase 14 runtime)'
    )
    if strict:
        print(msg, file=sys.stderr)
        sys.exit(1)
    print('warn: ' + msg, file=sys.stderr)
    src = 'unknown'
print(src)
" "$1" "$strict"
}

# Optional: assert /health llama_backend_source matches (config | env | default).
smoke_runtime_assert_llama_backend_source() {
  local health="$1"
  local want="${2:-}"
  [[ -n "$want" ]] || return 0
  local got
  got=$(smoke_runtime_llama_backend_source "$health" strict)
  if [[ "$got" != "$want" ]]; then
    echo "RUN_E2E_LLAMA_BACKEND_SOURCE=${want} but /health llama_backend_source=${got}" >&2
    return 1
  fi
  return 0
}

# When llama_backend comes from YAML (source=config), infer RUN_E2E_* backend flags from /health
# unless the caller already set RUN_E2E_INPROCESS or RUN_E2E_LLAMA_CPP_PYTHON.
smoke_runtime_apply_backend_flags_from_health() {
  local health="$1"
  local backend
  backend=$(smoke_runtime_llama_backend "$health" strict)
  if [[ "${RUN_E2E_INPROCESS:-0}" == "1" || "${RUN_E2E_LLAMA_CPP_PYTHON:-0}" == "1" ]]; then
    return 0
  fi
  case "$backend" in
    inprocess)
      export RUN_E2E_INPROCESS=1
      echo "yaml config smoke: llama_backend=inprocess (from /health)"
      ;;
    llama-cpp-python)
      export RUN_E2E_LLAMA_CPP_PYTHON=1
      echo "yaml config smoke: llama_backend=llama-cpp-python (from /health)"
      ;;
    subprocess)
      echo "llama_backend_source=config but /health llama_backend=subprocess" >&2
      echo "  Uncomment llama_backend: inprocess (or llama-cpp-python) in runtime YAML and restart serve without ZEROLLAMA_RUNTIME_LLAMA_BACKEND." >&2
      return 1
      ;;
    *)
      echo "yaml config smoke: unsupported llama_backend=${backend}" >&2
      return 1
      ;;
  esac
}

# Parse /health llama_cpp.gpu_mode (cpu | gpu); empty when absent.
smoke_runtime_llama_cpp_gpu_mode() {
  local strict="${2:-}"
  python3 -c "
import json, sys
strict = sys.argv[2] == 'strict'
h = json.loads(sys.argv[1])
lcp = h.get('llama_cpp') or {}
mode = (lcp.get('gpu_mode') or '').strip()
if not mode:
    if h.get('llama_backend') == 'llama-cpp-python':
        msg = '/health missing llama_cpp.gpu_mode for llama-cpp-python backend — rebuild serve'
        if strict:
            print(msg, file=sys.stderr)
            sys.exit(1)
        print('warn: ' + msg, file=sys.stderr)
        mode = 'unknown'
print(mode)
" "$1" "$strict"
}

# Assert wheel GPU offload (RUN_E2E_LLAMA_CPP_PYTHON_GPU=1); use post-generate /health.
smoke_runtime_assert_llama_cpp_gpu() {
  local health="$1"
  local want="${2:-gpu}"
  local mode ngl loaded
  read -r mode ngl loaded < <(
    python3 -c "
import json, sys
lcp = json.loads(sys.argv[1]).get('llama_cpp') or {}
print(
    (lcp.get('gpu_mode') or ''),
    lcp.get('n_gpu_layers', ''),
    '1' if lcp.get('loaded') else '0',
)
" "$health"
  )
  if [[ "$mode" != "$want" ]]; then
    echo "expected llama_cpp.gpu_mode=${want} but got ${mode:-<missing>} (n_gpu_layers=${ngl}, loaded=${loaded})" >&2
    if [[ "$want" == "gpu" && -z "${ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS:-}" ]]; then
      echo "  Set ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS on serve before wheel GPU smoke." >&2
    fi
    return 1
  fi
  if [[ "$want" == "gpu" ]]; then
    if [[ "${ngl:-0}" -le 0 ]]; then
      echo "llama_cpp.gpu_mode=gpu but n_gpu_layers=${ngl}" >&2
      return 1
    fi
    if [[ "$loaded" != "1" ]]; then
      echo "llama_cpp.gpu_mode=gpu but loaded=false (refetch /health after generate)" >&2
      return 1
    fi
  elif [[ "$want" == "cpu" && "$loaded" != "1" ]]; then
    echo "llama_cpp.gpu_mode=cpu but loaded=false (refetch /health after generate)" >&2
    return 1
  fi
  return 0
}

# Ensure :8081 is reachable (embedded or sidecar) before Phase 14 smokes.
smoke_runtime_require_listening() {
  local runtime_url="${1:-${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}}"
  runtime_url="${runtime_url%/}"
  if runtime_fetch_health "$runtime_url" >/dev/null 2>&1; then
    return 0
  fi
  echo "Python runtime is not listening on ${runtime_url}." >&2
  if [[ -n "${ZEROLLAMA_RUNTIME_URL:-}" ]]; then
    echo "  Start a sidecar on :8081, or unset ZEROLLAMA_RUNTIME_URL and use embed:" >&2
    echo "    source ./scripts/phase14_serve_env.sh && ./zerollama serve" >&2
  else
    echo "  Start serve with embed enabled:" >&2
    echo "    source ./scripts/phase14_serve_env.sh && export LLAMA_MODEL=... && ./zerollama serve" >&2
  fi
  return 1
}

# Fail fast when the running :8081 process predates Phase 14 /internal/tokenize.
smoke_runtime_require_phase14_endpoints() {
  local runtime_url="${1:-${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}}"
  local gguf="${2:-}"
  runtime_url="${runtime_url%/}"
  if [[ -z "$gguf" || ! -f "$gguf" ]]; then
    echo "phase14 preflight: skip /internal/tokenize probe (no GGUF file)" >&2
    return 0
  fi
  local tmp code
  tmp=$(mktemp)
  code=$(curl -sS -o "$tmp" -w "%{http_code}" -X POST \
    -H 'Content-Type: application/json' \
    -d "{\"gguf\":\"${gguf}\",\"text\":\"hi\"}" \
    "${runtime_url}/internal/tokenize" 2>/dev/null || echo "000")
  if [[ "$code" == "404" ]]; then
    echo "Phase 14 /internal/tokenize returned HTTP 404." >&2
    echo "Rebuild and restart zerollama serve from the current repo, then retry." >&2
    rm -f "$tmp"
    return 1
  fi
  if [[ "$code" != "200" ]]; then
    echo "warn: /internal/tokenize HTTP ${code} (expected 200 after resume/load): $(head -c 200 "$tmp")" >&2
  fi
  rm -f "$tmp"
  return 0
}

# Phase 15 v7/v8: assert KV export keys on /health and GET /internal/kv-snapshot.
smoke_runtime_assert_kv_snapshot() {
  local runtime_url="${1:-${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}}"
  runtime_url="${runtime_url%/}"
  local health_json
  health_json=$(curl -sf "${runtime_url}/health")
  python3 -c "
import json, sys
h = json.loads(sys.argv[1])
pb = h.get('kv_page_bind') or {}
assert pb.get('status') == 'not_implemented' and pb.get('available') is False, pb
bind = h.get('kv_bind') or {}
assert bind.get('physical_pages_bound') is False, bind
assert isinstance(h.get('kv_forward_plans'), list)
kd = h.get('kv_decode_steps') or {}
if kd.get('active') is True:
    assert int(kd.get('value') or 0) >= 0
print('kv /health ok: page_bind=', pb.get('status'))
" "$health_json"
  curl -sf "${runtime_url}/internal/kv-snapshot" -o /tmp/zerollama-kv-snapshot.json
  python3 -c "
import json
b = json.load(open('/tmp/zerollama-kv-snapshot.json'))
for key in ('kv_forward_plans', 'kv_page_bind', 'kv_decode_steps', 'kv_bind'):
    assert key in b, sorted(b.keys())
assert b['kv_page_bind']['status'] == 'not_implemented'
print('kv-snapshot ok')
"
}

# In-process backends do not need LLAMA_SERVER_BIN on the serve process.
smoke_runtime_needs_server_bin() {
  local backend="$1"
  [[ "$backend" != "inprocess" && "$backend" != "llama-cpp-python" ]]
}

# Human-readable hint when health.llama_model is missing.
smoke_llama_model_config_hint() {
  local backend="$1"
  local hint="Set LLAMA_MODEL on zerollama serve and restart."
  if smoke_runtime_needs_server_bin "$backend"; then
    hint="${hint} Subprocess backend also needs LLAMA_SERVER_BIN."
  else
    hint="${hint} In-process backends use ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess|llama-cpp-python or llama_backend: inprocess in runtime YAML (no llama-server binary)."
  fi
  printf '%s' "$hint"
}

# POST :8081/internal/inference/resume when handoff or pause blocks smokes.
# Why not :8080 — resume is implemented on the Python runtime only.
runtime_resume_if_needed() {
  local health="${1:-}"
  local runtime_url="${2:-${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}}"
  if [[ -z "$health" ]]; then
    health=$(runtime_fetch_health "$runtime_url") || {
      echo "runtime /health failed (is serve up on ${runtime_url}?)" >&2
      return 1
    }
  fi
  python3 -c "
import json, sys, urllib.request
h = json.loads(sys.argv[1])
ad = h.get('admission') or {}
need = (
    h.get('inference_state') != 'running'
    or ad.get('training_handoff_active')
    or ad.get('accepts_new_loads') is False
)
if not need:
    sys.exit(0)
url = sys.argv[2].rstrip('/') + '/internal/inference/resume'
req = urllib.request.Request(url, method='POST', data=b'')
with urllib.request.urlopen(req, timeout=60) as r:
    body = json.loads(r.read().decode())
if body.get('inference_state') != 'running':
    print('resume failed:', body, file=sys.stderr)
    sys.exit(1)
print('resumed inference:', body.get('inference_state'))
" "$health" "$runtime_url"
}

# Unload ggml runners via public API (empty prompt, keep_alive=0). No Phase-8 broker required.
# WHY mapfile: model tags must not be word-split on spaces; one name per line from Python.
# WHY SMOKE_UNLOAD_MAX_WAIT: large models can take >15s to exit after HTTP 200 unload.
smoke_unload_ggml_runners() {
  local ollama_url="${OLLAMA_HOST:-http://127.0.0.1:8080}"
  if ! smoke_ggml_runner_running; then
    return 0
  fi
  local models
  mapfile -t _unload_models < <(
    curl -sf -m 10 "${ollama_url}/api/ps" | python3 -c "
import json, sys
data = json.load(sys.stdin)
names = [m.get('name') or m.get('model') for m in (data.get('models') or [])]
names = [n for n in names if n]
if not names:
    sys.exit(1)
for n in names:
    print(n)
" 2>/dev/null
  ) || true
  if [[ ${#_unload_models[@]} -eq 0 ]]; then
    if [[ -n "${RUN_E2E_UNLOAD_MODEL:-${RUN_E2E_LEGACY_MODEL:-}}" ]]; then
      _unload_models=("${RUN_E2E_UNLOAD_MODEL:-${RUN_E2E_LEGACY_MODEL}}")
    else
      echo "warn: ggml runner up but /api/ps empty; set RUN_E2E_UNLOAD_MODEL" >&2
      return 1
    fi
  fi
  for m in "${_unload_models[@]}"; do
    code=$(curl -sS -o /dev/null -w "%{http_code}" -X POST \
      -H 'Content-Type: application/json' \
      -d "$(python3 -c "import json,sys; print(json.dumps({'model':sys.argv[1],'prompt':'','keep_alive':0}))" "$m")" \
      "${ollama_url}/api/generate" 2>/dev/null) || code=000
    echo "smoke_unload_ggml: ${m} (http ${code})"
  done
  local i=0
  local max_wait="${SMOKE_UNLOAD_MAX_WAIT:-30}"
  while smoke_ggml_runner_running && [[ "$i" -lt "$max_wait" ]]; do
    sleep 1
    i=$((i + 1))
  done
  if smoke_ggml_runner_running; then
    echo "warn: ggml runner still loaded after unload API" >&2
    return 1
  fi
  echo "smoke_unload_ggml: runners gone"
  return 0
}

# Phase 8 broker: one runtime-proxy generate evicts stale ggml runners before :8081 loads.
# WHY unload first: broker may never run on 503; WHY retry block: second chance after API unload.
smoke_prepare_vram_for_runtime() {
  local ollama_url="${OLLAMA_HOST:-http://127.0.0.1:8080}"
  if smoke_ggml_runner_running; then
    smoke_unload_ggml_runners || true
  fi
  local code
  code=$(python3 -c "
import json, os, sys, urllib.request
ollama = sys.argv[1].rstrip('/')
opts = {'num_predict': 1, 'num_ctx': int(os.environ.get('RUN_E2E_NUM_CTX', '4096'))}
g = os.environ.get('RUN_E2E_GGUF', '').strip() or os.environ.get('LLAMA_MODEL', '').strip()
if g:
    opts['gguf'] = g
payload = json.dumps({'model': 'smoke', 'prompt': 'x', 'stream': False, 'options': opts}).encode()
req = urllib.request.Request(
    ollama + '/api/generate',
    data=payload,
    method='POST',
    headers={'Content-Type': 'application/json', 'X-Zerollama-Runtime': '1'},
)
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        print(r.status)
except urllib.error.HTTPError as e:
    print(e.code)
" "$ollama_url") || code=000
  echo "smoke_prepare_vram: broker via runtime proxy (http ${code})"
  if [[ "$code" == "503" ]] && smoke_ggml_runner_running; then
    echo "smoke_prepare_vram: 503 with ggml loaded; retry after unload API" >&2
    if smoke_unload_ggml_runners; then
      runtime_resume_if_needed
      code=$(python3 -c "
import json, os, sys, urllib.request
ollama = sys.argv[1].rstrip('/')
opts = {'num_predict': 1, 'num_ctx': int(os.environ.get('RUN_E2E_NUM_CTX', '4096'))}
g = os.environ.get('RUN_E2E_GGUF', '').strip() or os.environ.get('LLAMA_MODEL', '').strip()
if g:
    opts['gguf'] = g
payload = json.dumps({'model': 'smoke', 'prompt': 'x', 'stream': False, 'options': opts}).encode()
req = urllib.request.Request(
    ollama + '/api/generate', data=payload, method='POST',
    headers={'Content-Type': 'application/json', 'X-Zerollama-Runtime': '1'},
)
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        print(r.status)
except urllib.error.HTTPError as e:
    print(e.code)
" "$ollama_url") || code=000
      echo "smoke_prepare_vram: broker retry (http ${code})"
    fi
  fi
  if [[ "$code" == "503" ]]; then
    if smoke_ggml_runner_running; then
      echo "error: Go 503 and ggml runner still loaded after unload retry" >&2
      return 1
    fi
    echo "warn: Go 503 before broker; no ggml runner process seen" >&2
  elif [[ "$code" != "200" && "$code" != "000" ]]; then
    echo "warn: broker probe non-200; runtime smokes may fail with 502/503" >&2
  fi
  runtime_resume_if_needed
  local runtime_url="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
  if ! runtime_fetch_health "$runtime_url" >/dev/null 2>&1; then
    echo "error: runtime not listening after broker probe (${runtime_url})" >&2
    if [[ "${RUN_E2E_LLAMA_CPP_PYTHON_GPU:-0}" == "1" ]]; then
      echo "  Wheel GPU offload may abort embedded Python (known: free(): invalid pointer on some cu124 wheels)." >&2
      echo "  Production GPU on 5080: use ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess + LLAMA_CPP_LIB." >&2
    fi
    return 1
  fi
}
