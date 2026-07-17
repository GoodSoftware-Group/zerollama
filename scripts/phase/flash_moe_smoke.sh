#!/usr/bin/env bash
# Opt-in Flash-MoE smoke — anemll slot-bank via zerollama → llama-server.
#
# WHY tiered: sidecar extract + MoE GGUF are operator-local (100GB+). CI and fresh
# clones must validate wiring without requiring Qwen3.5 on disk. Full E2E is opt-in.
#
# Tier 0 (default): go tests + flash-moe binary --moe-* flags (+ optional build)
# Tier 1 (RUN_E2E_FLASH_MOE_STARTUP=1): direct llama-server startup with sidecar
# Tier 2 (RUN_E2E_FLASH_MOE=1): zerollama serve + /api/generate through Flash-MoE path
#
# Usage:
#   ./scripts/phase/flash_moe_smoke.sh
#   FLASH_MOE_BUILD=1 ./scripts/phase/flash_moe_smoke.sh
#
# Tier 1 — sidecar + GGUF required (validates slot-bank load, no generation):
#   RUN_E2E_FLASH_MOE_STARTUP=1 \
#     FLASH_MOE_GGUF=~/Models/qwen35.gguf \
#     FLASH_MOE_SIDECAR=~/Models/flash/qwen35 \
#     ./scripts/phase/flash_moe_smoke.sh
#
# Tier 2 — full generate (needs pulled tag or FLASH_MOE_MODEL):
#   RUN_E2E_FLASH_MOE=1 \
#     FLASH_MOE_GGUF=~/Models/qwen35.gguf \
#     FLASH_MOE_SIDECAR=~/Models/flash/qwen35 \
#     FLASH_MOE_MODEL=qwen35-flash:latest \
#     ./scripts/phase/flash_moe_smoke.sh
#
# Auto-extract sidecar when GGUF set but sidecar missing:
#   FLASH_MOE_EXTRACT=1 RUN_E2E_FLASH_MOE_STARTUP=1 FLASH_MOE_GGUF=... ./scripts/phase/flash_moe_smoke.sh
#
# Env:
#   FLASH_MOE_REPO          anemll-flash-llama.cpp checkout
#   FLASH_MOE_BIN           override llama-server path
#   FLASH_MOE_SIDECAR       sidecar dir (manifest.json)
#   FLASH_MOE_GGUF          MoE GGUF path (also LLAMA_MODEL)
#   FLASH_MOE_MODEL         pulled tag for tier 2 (else auto from blob)
#   FLASH_MOE_SLOT_BANK     default 16 for startup/e2e
#   FLASH_MOE_OUT           JSON report (default /tmp/flash-moe-smoke.json)
#   FLASH_MOE_HOST          serve host:port (default 127.0.0.1:11447)
#   FLASH_MOE_NUM_PREDICT   default 8
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/runtime/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime/runtime_smoke_lib.sh"

FLASH_MOE_REPO="${FLASH_MOE_REPO:-${HOME}/Sites/inference/anemll-flash-llama.cpp}"
FLASH_MOE_BIN="${FLASH_MOE_BIN:-${ROOT}/build/flash-moe-llama-server-darwin/bin/llama-server}"
FLASH_MOE_SIDECAR="${FLASH_MOE_SIDECAR:-${ZEROLLAMA_FLASH_MOE_SIDECAR:-}}"
FLASH_MOE_GGUF="${FLASH_MOE_GGUF:-${LLAMA_MODEL:-}}"
FLASH_MOE_OUT="${FLASH_MOE_OUT:-/tmp/flash-moe-smoke.json}"
FLASH_MOE_SLOT_BANK="${FLASH_MOE_SLOT_BANK:-16}"
FLASH_MOE_NUM_PREDICT="${FLASH_MOE_NUM_PREDICT:-8}"
_raw_host="${FLASH_MOE_HOST:-127.0.0.1:11447}"
_raw_host="${_raw_host#http://}"
_raw_host="${_raw_host#https://}"
FLASH_MOE_HOST="${_raw_host}"
FLASH_MOE_PORT="${FLASH_MOE_HOST##*:}"

tier0_pass() {
  echo "flash_moe_smoke: tier0 PASS"
}

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "skip: flash_moe_smoke requires Darwin (Flash-MoE build script is darwin-only)" >&2
  tier0_pass
  exit 0
fi

echo "== Flash-MoE smoke tier 0: unit tests =="
if [[ "${FLASH_MOE_SKIP_GO_TEST:-0}" != "1" ]]; then
  (cd "${ROOT}" && go test -count=1 ./envconfig/... -run 'TestFlashMoERepo' >/dev/null)
  (cd "${ROOT}" && go test -count=1 ./llm/... -run 'TestAppendFlashMoE|TestSetLlamaServer' >/dev/null)
  echo ">>> go tests ok"
else
  echo ">>> skip go tests (FLASH_MOE_SKIP_GO_TEST=1)"
fi

if [[ ! -x "${FLASH_MOE_BIN}" ]]; then
  if [[ "${FLASH_MOE_BUILD:-0}" == "1" ]]; then
    echo ">>> building flash-moe llama-server" >&2
    FLASH_MOE_REPO="${FLASH_MOE_REPO}" "${ROOT}/scripts/build/build_flash_moe_llama_server.sh"
  else
    echo "flash_moe_smoke: missing ${FLASH_MOE_BIN} (set FLASH_MOE_BUILD=1 or run ./scripts/build/build_flash_moe_llama_server.sh)" >&2
    exit 1
  fi
fi

_help="$("${FLASH_MOE_BIN}" --help 2>&1 || true)"
if ! grep -q 'moe-sidecar' <<< "${_help}"; then
  echo "flash_moe_smoke: ${FLASH_MOE_BIN} lacks --moe-sidecar (wrong binary?)" >&2
  exit 1
fi
echo ">>> flash-moe binary ok: ${FLASH_MOE_BIN}"

if [[ "${RUN_E2E_FLASH_MOE:-0}" != "1" && "${RUN_E2E_FLASH_MOE_STARTUP:-0}" != "1" ]]; then
  tier0_pass
  exit 0
fi

if [[ -z "${FLASH_MOE_GGUF:-}" || (-z "${FLASH_MOE_SIDECAR:-}" && -z "${FLASH_MOE_EXTRACT:-}") ]]; then
  echo ">>> resolving MoE model from zerollama store" >&2
  if pick="$(smoke_flash_moe_autoresolve "${ROOT}" 2>/dev/null)"; then
    auto_gguf="$(echo "${pick}" | sed -n '1p')"
    auto_sidecar="$(echo "${pick}" | sed -n '2p')"
    auto_tag="$(echo "${pick}" | sed -n '3p')"
    [[ -z "${FLASH_MOE_GGUF:-}" && -n "${auto_gguf}" ]] && FLASH_MOE_GGUF="${auto_gguf}"
    [[ -z "${FLASH_MOE_SIDECAR:-}" && -n "${auto_sidecar}" ]] && FLASH_MOE_SIDECAR="${auto_sidecar}"
    [[ -z "${FLASH_MOE_MODEL:-}" && -n "${auto_tag}" ]] && FLASH_MOE_MODEL="${auto_tag}"
    echo ">>> picked tag=${FLASH_MOE_MODEL:-?} gguf=${FLASH_MOE_GGUF:-?} sidecar=${FLASH_MOE_SIDECAR:-?}" >&2
  fi
fi

if [[ -z "${FLASH_MOE_GGUF:-}" ]]; then
  echo "tier 1/2: no MoE GGUF — pull a MoE model (zerollama pull …) or set FLASH_MOE_GGUF" >&2
  echo "hint: ./zerollama flash-moe-resolve --list" >&2
  exit 1
fi
if [[ ! -f "${FLASH_MOE_GGUF}" ]]; then
  echo "flash_moe_smoke: GGUF not found: ${FLASH_MOE_GGUF}" >&2
  exit 1
fi

if [[ -z "${FLASH_MOE_SIDECAR}" ]]; then
  if [[ "${FLASH_MOE_EXTRACT:-0}" == "1" ]]; then
    FLASH_MOE_SIDECAR="${FLASH_MOE_SIDECAR:-${HOME}/Models/flash/flash-moe-smoke-$(basename "${FLASH_MOE_GGUF}" .gguf)}"
    echo ">>> extracting sidecar to ${FLASH_MOE_SIDECAR}" >&2
    python3 "${FLASH_MOE_REPO}/tools/flashmoe-sidecar/flashmoe_sidecar.py" extract \
      --model "${FLASH_MOE_GGUF}" \
      --out-dir "${FLASH_MOE_SIDECAR}" \
      --force
  else
    echo "tier 1/2: sidecar missing at ${FLASH_MOE_SIDECAR:-<unset>}" >&2
    echo "hint: FLASH_MOE_EXTRACT=1 or ./scripts/gpu/flash_moe_extract_sidecar.sh --model ${FLASH_MOE_GGUF} --out-dir ~/Models/flash/$(basename "${FLASH_MOE_GGUF}" .gguf)" >&2
    exit 1
  fi
fi

if [[ ! -f "${FLASH_MOE_SIDECAR}/manifest.json" && ! -f "${FLASH_MOE_SIDECAR}" ]]; then
  echo "flash_moe_smoke: sidecar missing manifest at ${FLASH_MOE_SIDECAR}" >&2
  exit 1
fi
if [[ -f "${FLASH_MOE_SIDECAR}" && "${FLASH_MOE_SIDECAR##*/}" == "manifest.json" ]]; then
  FLASH_MOE_SIDECAR="$(dirname "${FLASH_MOE_SIDECAR}")"
fi

flash_moe_direct_startup() {
  local log="${1}"
  local host="${FLASH_MOE_HOST%%:*}"
  [[ -z "${host}" ]] && host="127.0.0.1"

  "${FLASH_MOE_BIN}" \
    -m "${FLASH_MOE_GGUF}" \
    --host "${host}" \
    --port "${FLASH_MOE_PORT}" \
    --moe-mode slot-bank \
    --moe-sidecar "${FLASH_MOE_SIDECAR}" \
    --moe-slot-bank "${FLASH_MOE_SLOT_BANK}" \
    -fit on \
    -ub 1 \
    -ngl "${FLASH_MOE_NGL:-99}" \
    >"${log}" 2>&1 &
  local pid=$!
  local deadline=$((SECONDS + 600))
  while (( SECONDS < deadline )); do
    if grep -Fq "server is listening" "${log}" 2>/dev/null; then
      echo "${pid}"
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "flash-moe llama-server exited before ready" >&2
      tail -40 "${log}" >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "timeout waiting for flash-moe llama-server" >&2
  tail -40 "${log}" >&2 || true
  kill "${pid}" 2>/dev/null || true
  return 1
}

if [[ "${RUN_E2E_FLASH_MOE_STARTUP:-0}" == "1" && "${RUN_E2E_FLASH_MOE:-0}" != "1" ]]; then
  echo "== Flash-MoE smoke tier 1: direct llama-server startup =="
  LOG="${FLASH_MOE_OUT%.json}-startup.log"
  pid="$(flash_moe_direct_startup "${LOG}")"
  sleep 3
  kill -INT "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
  python3 - <<PY
import json, time
print(json.dumps({
    "ok": True,
    "tier": "startup",
    "bin": "${FLASH_MOE_BIN}",
    "gguf": "${FLASH_MOE_GGUF}",
    "sidecar": "${FLASH_MOE_SIDECAR}",
    "slot_bank": int("${FLASH_MOE_SLOT_BANK}"),
    "ts": time.time(),
}, indent=2))
PY
  > "${FLASH_MOE_OUT}"
  echo "flash_moe_smoke: tier1 PASS (log ${LOG})"
  exit 0
fi

echo "== Flash-MoE smoke tier 2: zerollama serve E2E =="

export FLASH_MOE_MODEL="${FLASH_MOE_MODEL:-${P17_MODEL:-}}"
if [[ -z "${FLASH_MOE_MODEL}" ]]; then
  export FLASH_MOE_MODEL
  FLASH_MOE_MODEL="$(smoke_m3_proxy_tag_for_gguf "${FLASH_MOE_GGUF}")"
  if [[ -n "${FLASH_MOE_MODEL}" ]]; then
    FLASH_MOE_MODEL="${FLASH_MOE_MODEL}:latest"
  fi
fi
if [[ -z "${FLASH_MOE_MODEL}" || "${FLASH_MOE_MODEL}" == ":latest" ]]; then
  if pick="$(smoke_flash_moe_autoresolve "${ROOT}" 2>/dev/null)"; then
    FLASH_MOE_MODEL="$(echo "${pick}" | sed -n '3p')"
  fi
fi
if [[ -z "${FLASH_MOE_MODEL}" || "${FLASH_MOE_MODEL}" == ":latest" ]]; then
  echo "No pulled tag for ${FLASH_MOE_GGUF}; pull a model or set FLASH_MOE_MODEL=your-tag:latest" >&2
  exit 1
fi

BIN="$(smoke_resolve_zerollama_bin "${ROOT}")"
export ZEROLLAMA_FLASH_MOE=1
export ZEROLLAMA_FLASH_MOE_SIDECAR="${FLASH_MOE_SIDECAR}"
export ZEROLLAMA_FLASH_MOE_SLOT_BANK="${FLASH_MOE_SLOT_BANK}"
export ZEROLLAMA_FLASH_MOE_LLAMA_SERVER_BIN="${FLASH_MOE_BIN}"
export ZEROLLAMA_LLAMA_SERVER=1
export ZEROLLAMA_LEGACY_RUNNER=1
export ZEROLLAMA_RUNTIME=0
export ZEROLLAMA_RUNTIME_EMBED=0
export ZEROLLAMA_RUNTIME_DARWIN_SIDECAR=0
unset ZEROLLAMA_RUNTIME_URL
export OLLAMA_HOST="${FLASH_MOE_HOST}"

if command -v fuser >/dev/null 2>&1; then
  fuser -k "${FLASH_MOE_PORT}/tcp" 2>/dev/null || true
elif command -v lsof >/dev/null 2>&1; then
  lsof -ti ":${FLASH_MOE_PORT}" | xargs kill -9 2>/dev/null || true
fi
sleep 1

LOG="${FLASH_MOE_OUT%.json}.log"
"${BIN}" serve --llama-server-backend >"${LOG}" 2>&1 &
SERVE_PID=$!
trap 'kill "${SERVE_PID}" 2>/dev/null || true' EXIT

_deadline=$((SECONDS + 180))
while (( SECONDS < _deadline )); do
  if curl -sf "http://${FLASH_MOE_HOST}/api/tags" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
    echo "zerollama serve exited early" >&2
    tail -40 "${LOG}" >&2 || true
    exit 1
  fi
  sleep 2
done
if ! curl -sf "http://${FLASH_MOE_HOST}/api/tags" >/dev/null 2>&1; then
  echo "timeout waiting for serve on ${FLASH_MOE_HOST}" >&2
  tail -40 "${LOG}" >&2 || true
  exit 1
fi

if ! grep -q 'moe-sidecar\|flash-moe\|slot-bank' "${LOG}" 2>/dev/null; then
  echo "warn: serve log missing flash-moe markers (may still work via env passthrough)" >&2
fi

GEN_JSON="$(curl -sf "http://${FLASH_MOE_HOST}/api/generate" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${FLASH_MOE_MODEL}\",\"prompt\":\"Say hi in one word.\",\"stream\":false,\"options\":{\"num_predict\":${FLASH_MOE_NUM_PREDICT}}}")"

echo "${GEN_JSON}" | python3 -c "
import json, sys
body = json.load(sys.stdin)
if 'error' in body:
    raise SystemExit(body['error'])
resp = body.get('response') or ''
if not resp.strip():
    raise SystemExit('empty response')
print('response_len=', len(resp))
print('eval_count=', body.get('eval_count'))
"

python3 - <<PY
import json, time
print(json.dumps({
    "ok": True,
    "tier": "e2e",
    "model": "${FLASH_MOE_MODEL}",
    "gguf": "${FLASH_MOE_GGUF}",
    "sidecar": "${FLASH_MOE_SIDECAR}",
    "host": "${FLASH_MOE_HOST}",
    "bin": "${FLASH_MOE_BIN}",
    "ts": time.time(),
}, indent=2))
PY
> "${FLASH_MOE_OUT}"

echo "flash_moe_smoke: tier2 PASS (report ${FLASH_MOE_OUT})"
