#!/usr/bin/env bash
# Re-test MLX lmstudio models after relative-path import fix.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZEROLLAMA="${ZEROLLAMA_BIN:-${ROOT}/zerollama}"
HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
BASE="http://${HOST}"
LOG="${LMSTUDIO_MLX_TEST_LOG:-/tmp/zerollama-lmstudio-mlx-test.log}"

MODELS=(
  "lmstudio-community/glm-4.7-flash:latest|1200"
  "lmstudio-community/hermes-4-70b:latest|1800"
  "lmstudio-community/qwen3-coder-next:latest|1800"
)

: >"${LOG}"
log() { echo "$*" | tee -a "${LOG}"; }

curl -sf "${BASE}/api/tags" >/dev/null

for entry in "${MODELS[@]}"; do
  model="${entry%%|*}"
  timeout_sec="${entry##*|}"
  log "--- ${model} ---"
  if ! "${ZEROLLAMA}" show "${model}" >/dev/null 2>&1; then
    log "pull ${model}"
    "${ZEROLLAMA}" pull "${model}" >>"${LOG}" 2>&1 || { log "FAIL pull"; continue; }
  fi
  payload=$(printf '{"model":"%s","prompt":"Reply with exactly: ok","stream":false,"keep_alive":0,"options":{"num_ctx":512,"num_predict":8,"temperature":0}}' "${model}")
  resp="$(mktemp)"
  if curl -sf --max-time "${timeout_sec}" "${BASE}/api/generate" -H 'Content-Type: application/json' -d "${payload}" >"${resp}"; then
    if python3 -c "import json,sys; d=json.load(open('${resp}')); sys.exit(0 if d.get('done') and not d.get('error') else 1)"; then
      log "PASS ${model}"
    else
      log "FAIL generate body"; head -c 800 "${resp}" | tee -a "${LOG}"
    fi
  else
    log "FAIL generate timeout/error"
  fi
  rm -f "${resp}"
  sleep 10
done
