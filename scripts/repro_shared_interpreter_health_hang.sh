#!/usr/bin/env bash
# Minimal repro: GET /health hangs when training + embedded runtime share one CPython.
# Intermittent: often succeeds once or twice, then hangs. See docs/bugs/shared-interpreter-health-hang.md
#
# Uses loopback ports 19180 (Go), 19181 (runtime), 19650 (training TCP) — not :8080.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ZEROLLAMA_BIN:-$REPO_ROOT/zerollama}"
GO_HOST="${REPRO_GO_HOST:-127.0.0.1:19180}"
RT_PORT="${REPRO_RUNTIME_PORT:-19181}"
TRAIN_TCP="${REPRO_TRAINING_TCP:-:19650}"
HEALTH_TIMEOUT="${REPRO_HEALTH_TIMEOUT:-20}"
HEALTH_ROUNDS="${REPRO_HEALTH_ROUNDS:-5}"
STARTUP_WAIT="${REPRO_STARTUP_WAIT:-45}"

LLAMA_MODEL="${LLAMA_MODEL:-}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-}"

if [[ ! -x "$BIN" ]]; then
  echo "build first: cd $REPO_ROOT && CGO_ENABLED=1 go build -o zerollama ." >&2
  exit 1
fi
if [[ -z "$LLAMA_MODEL" || ! -f "$LLAMA_MODEL" ]]; then
  for cand in \
    "$REPO_ROOT/../Llama-OuteTTS-1.0-1B-Q8_0.gguf" \
    "/root/Llama-OuteTTS-1.0-1B-Q8_0.gguf"; do
    if [[ -f "$cand" ]]; then
      LLAMA_MODEL="$cand"
      break
    fi
  done
fi
if [[ -z "$LLAMA_MODEL" || ! -f "$LLAMA_MODEL" ]]; then
  echo "set LLAMA_MODEL to any small .gguf on disk" >&2
  exit 1
fi
if [[ -z "$LLAMA_SERVER_BIN" ]]; then
  for cand in "$REPO_ROOT/llama-server" /usr/bin/llama-server; do
    if [[ -x "$cand" ]]; then
      LLAMA_SERVER_BIN="$cand"
      break
    fi
  done
fi

RUNTIME_URL="http://127.0.0.1:${RT_PORT}"
LOG="${REPRO_LOG:-/tmp/repro-shared-interpreter-health.log}"
PIDFILE="${REPRO_PIDFILE:-/tmp/repro-shared-interpreter-health.pid}"

cleanup() {
  if [[ -f "$PIDFILE" ]]; then
    local pid
    pid=$(cat "$PIDFILE")
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    rm -f "$PIDFILE"
  fi
}
trap cleanup EXIT

if ss -tlnp 2>/dev/null | grep -q ":${RT_PORT}\b"; then
  echo "port ${RT_PORT} already in use; stop other test serves or set REPRO_RUNTIME_PORT" >&2
  exit 1
fi
if curl -sf -m 1 "http://127.0.0.1:8080/api/version" >/dev/null 2>&1; then
  echo "note: production may be on :8080 — this repro uses ${GO_HOST} only" >&2
fi

echo "== repro shared-interpreter /health hang =="
echo "binary: $BIN"
echo "ports:  go=${GO_HOST} runtime=${RUNTIME_URL} training_tcp=${TRAIN_TCP}"
echo "log:    $LOG"

unset ZEROLLAMA_RUNTIME_URL
export OLLAMA_HOST="$GO_HOST"
export ZEROLLAMA_RUNTIME_EMBED_PORT="$RT_PORT"
export ZEROLLAMA_RUNTIME_EMBED=1
export OLLAMA_TRAINING=true
export OLLAMA_TRAINING_TCP="$TRAIN_TCP"
export OLLAMA_TRAINING_PYTHONPATH="$REPO_ROOT"
export ZEROLLAMA_REPO="$REPO_ROOT"
export OLLAMA_NO_CLOUD=true
export LLAMA_MODEL
export LLAMA_SERVER_BIN
export ZEROLLAMA_RUNTIME_CONFIG="$REPO_ROOT/runtime/configs/single_gpu.yaml"
export ZEROLLAMA_DEVICE_COUNT=1
export OLLAMA_MAX_LOADED_MODELS=1

"$BIN" serve >"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "daemon pid=$(cat "$PIDFILE")"

deadline=$((SECONDS + STARTUP_WAIT))
train_ok=0
while (( SECONDS < deadline )); do
  if curl -sf -m 2 "http://${GO_HOST#*://}/api/train/status" >/dev/null 2>&1; then
    train_ok=1
    break
  fi
  sleep 1
done
if [[ "$train_ok" != 1 ]]; then
  echo "FAIL: /api/train/status never became ready (${STARTUP_WAIT}s)" >&2
  tail -30 "$LOG" >&2
  exit 1
fi
echo "PASS: /api/train/status responded"

if printf '{"cmd":"ping"}\n' | nc -w 5 "127.0.0.1" "${TRAIN_TCP#:}" | grep -q '"status":"ok"'; then
  echo "PASS: training TCP ping responded"
else
  echo "FAIL: training TCP ping" >&2
  exit 1
fi

root_code=$(curl -sS -m 3 -o /dev/null -w '%{http_code}' "${RUNTIME_URL}/" || echo 000)
echo "runtime / http=${root_code} (expect 404 = uvicorn listening)"

hang_round=0
for round in $(seq 1 "$HEALTH_ROUNDS"); do
  health_tmp=$(mktemp)
  t0=$SECONDS
  health_http=$(curl -sS -m "$HEALTH_TIMEOUT" \
    -o "$health_tmp" -w '%{http_code}' "${RUNTIME_URL}/health" 2>/dev/null || echo 000)
  dt=$((SECONDS - t0))
  health_bytes=$(wc -c <"$health_tmp" | tr -d ' ')
  rm -f "$health_tmp"
  echo "health try ${round}: http=${health_http} bytes=${health_bytes} time=${dt}s"
  if [[ "$health_http" != "200" || "$health_bytes" -lt 10 ]]; then
    hang_round=$round
    break
  fi
done

if [[ "$hang_round" -eq 0 ]]; then
  echo "PASS: /health OK for all ${HEALTH_ROUNDS} tries (shared-interpreter mitigation holds)"
  exit 0
fi

echo "FAIL: /health hung on try ${hang_round} (shared-interpreter bug reproduced)"
grep -E 'training worker started|embedded python runtime|go-coordination' "$LOG" | tail -8 || true
echo "see $LOG and docs/bugs/shared-interpreter-health-hang.md"
exit 1
