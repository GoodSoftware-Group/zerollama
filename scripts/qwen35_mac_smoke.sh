#!/usr/bin/env bash
# Opt-in Qwen 3.5/3.6 ggml smoke on macOS (M10).
#
# Why separate from runtime smokes: qwen35* loads via Go ollama-engine (OllamaEngineRequired).
# Runtime Metal must be released before ggml can load on the same device.
# Why accept thinking OR response: qwen3.6 thinking models may return text only in `thinking`.
#
# Daily serve uses OLLAMA_HOST=:11434; this script defaults :8080 (CI smoke layout).
# Override: OLLAMA_HOST=http://127.0.0.1:11434 ./scripts/qwen35_mac_smoke.sh
#
# Usage:
#   RUN_E2E_QWEN35_MODEL=qwen3.6:latest ./scripts/qwen35_mac_smoke.sh
#   RUN_E2E_QWEN35=1 RUN_E2E_QWEN35_MODEL=... ./scripts/m3_metal_signoff.sh
#
# Prerequisite: fresh ./scripts/build_zerollama_mac.sh; model pulled locally.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime_smoke_lib.sh"

RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
OLLAMA_URL="${OLLAMA_HOST:-http://127.0.0.1:8080}"
QWEN_MODEL="${RUN_E2E_QWEN35_MODEL:-${QWEN35_SMOKE_MODEL:-}}"
QWEN_NUM_CTX="${RUN_E2E_QWEN35_NUM_CTX:-2048}"
QWEN_NUM_PREDICT="${RUN_E2E_QWEN35_NUM_PREDICT:-16}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "skip qwen35 mac smoke: darwin only" >&2
  exit 0
fi

if [[ -z "$QWEN_MODEL" ]]; then
  echo "Set RUN_E2E_QWEN35_MODEL (local pulled tag, e.g. qwen3.6:latest)" >&2
  exit 1
fi

runtime_llama_loaded() {
  runtime_fetch_health "$RUNTIME_URL" 2>/dev/null \
    | python3 -c 'import json,sys; print("1" if json.load(sys.stdin).get("llama_server") else "")' \
    2>/dev/null || true
}

runtime_handoff_for_ggml() {
  if ! curl -sf "${RUNTIME_URL%/}/health" >/dev/null 2>&1; then
    return 0
  fi
  if [[ "$(runtime_llama_loaded)" != "1" ]]; then
    return 0
  fi
  echo "== runtime training-handoff (free Metal for legacy qwen35 ggml) =="
  curl -sS -X POST "${RUNTIME_URL%/}/internal/training-handoff" >/dev/null || true
  local i
  for i in $(seq 1 30); do
    if [[ "$(runtime_llama_loaded)" != "1" ]]; then
      echo "runtime llama_server idle"
      return 0
    fi
    sleep 2
  done
  echo "runtime still holds llama_server after handoff; set ZEROLLAMA_LEGACY_RUNNER=1 to force (dual Metal risk)" >&2
  return 1
}

runtime_handoff_for_ggml

payload=$(QWEN_MODEL="$QWEN_MODEL" QWEN_NUM_CTX="$QWEN_NUM_CTX" QWEN_NUM_PREDICT="$QWEN_NUM_PREDICT" python3 -c "import json, os; print(json.dumps({
    'model': os.environ['QWEN_MODEL'],
    'prompt': 'Say hi in one word.',
    'stream': False,
    'options': {
        'num_ctx': int(os.environ.get('QWEN_NUM_CTX', '2048')),
        'num_predict': int(os.environ.get('QWEN_NUM_PREDICT', '16')),
    },
}))")

echo "== qwen35 ggml /api/generate (${QWEN_MODEL}, num_ctx=${QWEN_NUM_CTX}) =="
tmp=$(mktemp)
code=$(curl -sS -m 600 -o "$tmp" -w "%{http_code}" -X POST \
  -H 'Content-Type: application/json' \
  -d "$payload" \
  "${OLLAMA_URL}/api/generate")
if [[ "$code" != "200" ]]; then
  echo "HTTP ${code} ${OLLAMA_URL}/api/generate:" >&2
  cat "$tmp" >&2
  rm -f "$tmp"
  exit 1
fi
python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
assert d.get('done'), d
resp = (d.get('response') or '').strip()
think = (d.get('thinking') or '').strip()
text = resp or think
assert text, f'empty response/thinking: {d!r}'
blob = json.dumps(d).lower()
for bad in ('kernel_unary', 'dimension_sections', 'unknown architecture'):
    assert bad not in blob, d
print('qwen35 generate: ok', repr(text[:80]))
" "$tmp"
rm -f "$tmp"

# Best-effort unload so later runtime smokes can resume.
RUN_E2E_UNLOAD_MODEL="$QWEN_MODEL" smoke_unload_ggml_runners || true
runtime_resume_if_needed "$(runtime_fetch_health "$RUNTIME_URL")" "$RUNTIME_URL"

echo "PASS: qwen35 mac smoke (${QWEN_MODEL})"
