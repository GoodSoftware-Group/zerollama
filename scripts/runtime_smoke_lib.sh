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
    hint="${hint} In-process backends use ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess|llama-cpp-python (no llama-server binary)."
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
}
