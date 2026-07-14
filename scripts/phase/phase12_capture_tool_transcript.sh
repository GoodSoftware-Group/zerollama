#!/usr/bin/env bash
# Capture a real tools-chat assistant transcript on GPU for Phase 12 golden updates.
#
#   export OLLAMA_HOST=http://127.0.0.1:8080
#   ./scripts/phase/phase12_capture_tool_transcript.sh gpt-oss:20b
#   ./scripts/phase/phase12_capture_tool_transcript.sh llama3.2:3B --out /tmp/transcript.txt
#   CAPTURE_RUNTIME_URL=http://127.0.0.1:8081/api/chat RUN_E2E_GGUF=... ./scripts/phase/phase12_capture_tool_transcript.sh smoke
#
# Uses runtime path on :8080 (model must route to zerollama-runtime or pass X-Zerollama-Runtime).
# Prints assistant text + tool_calls JSON; optional Go one-shot parse validation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/runtime/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime/runtime_smoke_lib.sh"

MODEL="${1:-}"
shift || true
OUT=""
VALIDATE_PARSE=0
HARMONY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --validate-parse) VALIDATE_PARSE=1; shift ;;
    --harmony) HARMONY=1; shift ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "usage: $0 <model> [--out file] [--validate-parse]" >&2
  exit 1
fi

export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:8080}"
export ZEROLLAMA_RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
OLLAMA_URL="$OLLAMA_HOST"
GO_URL="${ZEROLLAMA_GO_URL:-http://127.0.0.1:8080}"

runtime_resume_if_needed
smoke_prepare_vram_for_runtime
runtime_resume_if_needed

capture_url="${CAPTURE_RUNTIME_URL:-}"
if [[ -z "$capture_url" ]]; then
  capture_url="${OLLAMA_URL}/api/chat"
  use_runtime_header=1
else
  use_runtime_header=0
fi

export CAPTURE_HARMONY="$HARMONY"
payload=$(python3 -c "
import json, os, sys
model = sys.argv[1]
opts = {'num_ctx': int(os.environ.get('RUN_E2E_NUM_CTX', '4096'))}
# Only inject gguf for smoke path or explicit RUN_E2E_GGUF — not serve LLAMA_MODEL (wrong weights).
g = os.environ.get('RUN_E2E_GGUF', '').strip()
if not g and model == 'smoke':
    g = os.environ.get('LLAMA_MODEL', '').strip()
if g:
    opts['gguf'] = g
harmony = os.environ.get('CAPTURE_HARMONY', '').strip() in ('1', 'true', 'yes')
if harmony:
    tools = [{
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get weather for a city',
            'parameters': {
                'type': 'object',
                'properties': {'location': {'type': 'string'}},
                'required': ['location'],
            },
        },
    }]
    prompt = 'What is the weather in San Francisco? Use the get_weather tool.'
else:
    tools = [{
        'type': 'function',
        'function': {
            'name': 'calculator',
            'description': 'Evaluate a math expression',
            'parameters': {
                'type': 'object',
                'properties': {'expression': {'type': 'string'}},
                'required': ['expression'],
            },
        },
    }]
    prompt = 'What is 19+23? Use the calculator tool.'
body = {
    'model': model,
    'messages': [{'role': 'user', 'content': prompt}],
    'stream': False,
    'tools': tools,
}
if opts:
    body['options'] = opts
print(json.dumps(body))
" "$MODEL")

tmp=$(mktemp)
headers=(-H 'Content-Type: application/json')
if [[ "$use_runtime_header" == "1" ]]; then
  headers+=(-H 'X-Zerollama-Runtime: 1')
fi
code=$(curl -sS -o "$tmp" -w "%{http_code}" -X POST \
  "${headers[@]}" \
  -d "$payload" \
  "$capture_url")
if [[ "$code" != "200" ]]; then
  echo "HTTP ${code} ${capture_url}:" >&2
  cat "$tmp" >&2
  exit 1
fi

python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
msg = d.get('message') or {}
out = {
    'model': sys.argv[2],
    'done': d.get('done'),
    'content': msg.get('content'),
    'tool_calls': msg.get('tool_calls'),
    'meta': {k: d.get(k) for k in ('truncated', 'truncate_mode') if k in d},
}
text = json.dumps(out, indent=2, ensure_ascii=False)
if sys.argv[3]:
    open(sys.argv[3], 'w', encoding='utf-8').write(text + '\n')
    print('wrote', sys.argv[3])
else:
    print(text)
# raw assistant text for paste into Go golden
content = msg.get('content') or ''
if content.strip():
    print('--- assistant content (for parse golden) ---', file=sys.stderr)
    print(content, file=sys.stderr)
" "$tmp" "$MODEL" "$OUT"

if [[ "$VALIDATE_PARSE" == "1" ]]; then
  content=$(python3 -c "import json; d=json.load(open('$tmp')); m=d.get('message') or {}; print(m.get('content') or '')")
  tool_calls=$(python3 -c "import json; d=json.load(open('$tmp')); tc=(d.get('message') or {}).get('tool_calls'); print(json.dumps(tc) if tc else '')")
  if [[ -n "$tool_calls" && "$tool_calls" != "null" ]]; then
    echo "parse: tool_calls present in chat response (runtime parsed)" >&2
    python3 -m json.tool <<<"$tool_calls" >&2
  fi
  if [[ -n "$content" ]]; then
    parse_tmp=$(mktemp)
    parse_code=$(curl -sS -o "$parse_tmp" -w "%{http_code}" -X POST \
      -H 'Content-Type: application/json' \
      -d "$(python3 -c "import json,sys; print(json.dumps({'model':sys.argv[1],'content':sys.argv[2],'done':True}))" "$MODEL" "$content")" \
      "${GO_URL}/internal/parse-tool-output")
    if [[ "$parse_code" == "200" ]]; then
      echo "parse-tool-output: ok" >&2
      python3 -m json.tool "$parse_tmp" >&2
    else
      echo "parse-tool-output HTTP ${parse_code}:" >&2
      cat "$parse_tmp" >&2
      exit 1
    fi
    rm -f "$parse_tmp"
  fi
fi
rm -f "$tmp"
