#!/usr/bin/env bash
# Lightweight training ops smoke (HTTP + optional TCP). Does not submit real train jobs.
#
#   OLLAMA_HOST=http://127.0.0.1:8080 ./scripts/e2e/e2e_training_ops_smoke.sh
#   OLLAMA_HOST=http://127.0.0.1:19180 RUN_E2E_TRAINING_TCP=1 OLLAMA_TRAINING_TCP=:19650 ./scripts/e2e/e2e_training_ops_smoke.sh
#
# Use alternate ports for repro/dev — do not kill production :8080 without asking.
set -euo pipefail

OLLAMA_URL="${OLLAMA_HOST:-http://127.0.0.1:8080}"
TRAIN_TCP="${OLLAMA_TRAINING_TCP:-:9500}"
CURL_TIMEOUT="${E2E_TRAINING_CURL_TIMEOUT:-5}"

curl_ollama() {
  local path="$1"
  local tmp
  tmp=$(mktemp)
  local code
  code=$(curl -sS -o "$tmp" -w "%{http_code}" -m "$CURL_TIMEOUT" \
    "${OLLAMA_URL}${path}")
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

echo "== training ops smoke: ${OLLAMA_URL} =="

status_json=$(curl_ollama "/api/train/status")
python3 -c "
import json, sys
d = json.loads(sys.argv[1])
if not isinstance(d, dict):
    raise SystemExit('train/status: expected JSON object')
print('train/status keys:', sorted(d.keys())[:12], '...')
" "$status_json"

jobs_json=$(curl_ollama "/api/train/jobs")
python3 -c "
import json, sys
d = json.loads(sys.argv[1])
if not isinstance(d, (list, dict)):
    raise SystemExit('train/jobs: expected list or object')
print('train/jobs: ok')
" "$jobs_json"

if [[ "${RUN_E2E_TRAINING_TCP:-0}" == "1" ]]; then
  train_tcp="${TRAIN_TCP:-:9500}"
  train_tcp="${train_tcp#tcp://}"
  if [[ "$train_tcp" == :* ]]; then
    host=127.0.0.1
    port="${train_tcp#:}"
  elif [[ "$train_tcp" == *:* ]]; then
    host="${train_tcp%%:*}"
    port="${train_tcp##*:}"
    [[ -z "$host" ]] && host=127.0.0.1
  else
    host=127.0.0.1
    port="$train_tcp"
  fi
  python3 -c "
import json, socket, sys
host, port = sys.argv[1], int(sys.argv[2])
msg = json.dumps({'cmd': 'ping'}) + '\n'
s = socket.create_connection((host, port), timeout=5)
s.sendall(msg.encode())
s.shutdown(socket.SHUT_WR)
buf = b''
while True:
    chunk = s.recv(4096)
    if not chunk:
        break
    buf += chunk
s.close()
if not buf.strip():
    raise SystemExit('training TCP: empty response')
print('training TCP ping: ok')
" "$host" "$port"
fi

echo "PASS: training ops smoke"
