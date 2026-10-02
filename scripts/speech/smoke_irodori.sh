#!/usr/bin/env bash
# Deploy irodori-tts unit + rebuild zerollama (optional) + register + speech smoke.
# Assumes ./scripts/speech/install_irodori_tts.sh already ran.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
HOST="${OLLAMA_HOST:-http://127.0.0.1:2083}"
HOST="${HOST#http://}"
HOST="${HOST#https://}"
BASE="http://${HOST%%/*}"
if [[ "${BASE}" != http://* ]]; then BASE="http://${HOST}"; fi
# Normalize: OLLAMA_HOST may be 0.0.0.0:2083 or http://127.0.0.1:2083
if [[ "${ZEROLLAMA_SMOKE_URL:-}" != "" ]]; then
  BASE="${ZEROLLAMA_SMOKE_URL}"
elif [[ "${OLLAMA_HOST:-}" == http* ]]; then
  BASE="${OLLAMA_HOST}"
else
  BASE="http://127.0.0.1:2083"
fi

if [[ "${SMOKE_BUILD:-1}" == "1" ]]; then
  echo ">>> build zerollama"
  CGO_ENABLED=1 go build -o /tmp/zerollama.new .
  install -m 0755 /tmp/zerollama.new /usr/local/bin/zerollama
fi

echo ">>> systemd irodori-tts"
cp -af "${REPO_ROOT}/scripts/systemd/irodori-tts.service" /etc/systemd/system/irodori-tts.service
mkdir -p /mnt/ollama_img/speech/irodori-hf
chown -R ollama:ollama /mnt/ollama_img/speech/irodori-tts-server /mnt/ollama_img/speech/irodori /mnt/ollama_img/speech/irodori-hf 2>/dev/null || true
systemctl daemon-reload
systemctl enable --now irodori-tts
systemctl restart zerollama

echo ">>> wait for :8088/health"
ok=0
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8088/health >/tmp/irodori-health.json 2>/dev/null; then
    cat /tmp/irodori-health.json; echo
    ok=1
    break
  fi
  sleep 5
done
if [[ "$ok" != "1" ]]; then
  journalctl -u irodori-tts -n 50 --no-pager || true
  echo "ERROR: irodori-tts health failed" >&2
  exit 1
fi

echo ">>> register speech tags"
OLLAMA_MODELS="${OLLAMA_MODELS:-/mnt/ollama_img/models}" ./scripts/register_speech_models.sh

echo ">>> smoke ${BASE}/v1/audio/speech"
code=$(curl -sS -o /tmp/irodori-smoke.wav -w '%{http_code}' \
  "${BASE}/v1/audio/speech" \
  -H 'content-type: application/json' \
  -d '{"model":"irodori","input":"こんにちは。","voice":"none","emotion":"落ち着いた自然な声"}')
echo "http=${code}"
ls -lh /tmp/irodori-smoke.wav
file /tmp/irodori-smoke.wav || true
if [[ "$code" != "200" ]]; then
  head -c 400 /tmp/irodori-smoke.wav; echo
  exit 1
fi
echo "OK: irodori smoke wav at /tmp/irodori-smoke.wav"
