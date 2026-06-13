#!/usr/bin/env bash
# Smoke-test all lmstudio-community catalog models: pull/import if needed, one generate each.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZEROLLAMA="${ZEROLLAMA_BIN:-${ROOT}/zerollama}"
HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
BASE="http://${HOST}"
LOG="${LMSTUDIO_TEST_LOG:-/tmp/zerollama-lmstudio-test.log}"

MODELS=(
  "lmstudio-community/qwen3.6-27b:q8_0|600"
  "lmstudio-community/gemma-4-31b-it:q8_0|900"
  "lmstudio-community/glm-4.7-flash:latest|900"
  "lmstudio-community/gpt-oss-120b:mxfp4|1200"
  "lmstudio-community/hermes-4-70b:latest|1200"
  "lmstudio-community/qwen3-coder-next:latest|1200"
)

: >"${LOG}"

log() { echo "$*" | tee -a "${LOG}"; }

if ! curl -sf "${BASE}/api/tags" >/dev/null; then
  log "ERROR: serve not reachable at ${BASE} — start: ${ZEROLLAMA} serve"
  exit 1
fi

log "=== lmstudio-community smoke test $(date -Iseconds) ==="
log "host=${BASE} binary=${ZEROLLAMA}"

pass=0
fail=0
skip=0

for entry in "${MODELS[@]}"; do
  model="${entry%%|*}"
  timeout_sec="${entry##*|}"
  log ""
  log "--- ${model} (timeout ${timeout_sec}s) ---"

  if ! "${ZEROLLAMA}" show "${model}" >/dev/null 2>&1; then
    log "pull/import ${model} ..."
    if ! "${ZEROLLAMA}" pull "${model}" >>"${LOG}" 2>&1; then
      log "FAIL pull: ${model}"
      fail=$((fail + 1))
      continue
    fi
  else
    log "already registered locally"
  fi

  payload=$(cat <<EOF
{"model":"${model}","prompt":"Reply with exactly: ok","stream":false,"keep_alive":0,"options":{"num_ctx":512,"num_predict":8,"temperature":0}}
EOF
)

  resp_file="$(mktemp)"
  set +e
  curl -sf --max-time "${timeout_sec}" "${BASE}/api/generate" \
    -H 'Content-Type: application/json' \
    -d "${payload}" >"${resp_file}" 2>>"${LOG}"
  curl_ec=$?
  set -e

  if [[ ${curl_ec} -ne 0 ]]; then
    log "FAIL generate (curl exit ${curl_ec}): ${model}"
    tail -5 "${LOG}" | tee -a "${LOG}" || true
    fail=$((fail + 1))
    rm -f "${resp_file}"
    sleep 5
    continue
  fi

  if python3 -c "
import json,sys
d=json.load(open('${resp_file}'))
if d.get('error'):
    print('error:', d['error']); sys.exit(1)
r=(d.get('response') or d.get('thinking') or '').strip()
if not d.get('done'):
    print('not done'); sys.exit(1)
print('response_len', len(r), 'eval', d.get('eval_count',0))
" >>"${LOG}" 2>&1; then
    log "PASS ${model}"
    pass=$((pass + 1))
  else
    log "FAIL generate body: ${model}"
    head -c 500 "${resp_file}" | tee -a "${LOG}" || true
    fail=$((fail + 1))
  fi
  rm -f "${resp_file}"
  sleep 10
done

log ""
log "=== summary: pass=${pass} fail=${fail} skip=${skip} ==="
log "full log: ${LOG}"
[[ ${fail} -eq 0 ]]
