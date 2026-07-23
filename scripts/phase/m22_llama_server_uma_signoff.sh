#!/usr/bin/env bash
# M22 UMA ↔ llama-server (Darwin Metal) lab sign-off.
#
# Never binds or kills :11434 / :8081.
# Uses M22_PORT for llama-server (default 18082) — not Go serve.
#
# Prerequisites:
#   - Darwin + llama-server built with BUILD_UMA (libuma in libllama)
#   - uma_daemon up (UMAStatus.app)
#   - GGUF path (default: eliza-1-2b / m21-ggml blob)
#
# Usage:
#   ./scripts/phase/m22_llama_server_uma_signoff.sh
#   M22_SKIP_BUILD=1 ./scripts/phase/m22_llama_server_uma_signoff.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

M22_PORT="${M22_PORT:-18082}"
M22_HOST="127.0.0.1:${M22_PORT}"
M22_URL="http://${M22_HOST}"
LOG_DIR="${M22_LOG_DIR:-/tmp/m22-llama-server-uma-signoff}"
mkdir -p "${LOG_DIR}"

if [[ "${M22_PORT}" == "11434" || "${M22_PORT}" == "8081" || "${M22_PORT}" == "11435" ]]; then
  echo "error: refusing reserved/lab-serve port ${M22_PORT} (use e.g. 18082)" >&2
  exit 1
fi
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "skip: M22 llama-server UMA sign-off is Darwin-only" >&2
  exit 0
fi

_broker_ping() {
  python3 - "${UMA_SOCK:-/tmp/uma_daemon.sock}" <<'PY'
import os, socket, sys
path = sys.argv[1]
if not os.path.exists(path):
    sys.exit(1)
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(2)
try:
    s.connect(path)
    s.sendall(b"PING\n")
    r = s.recv(64)
except OSError:
    sys.exit(1)
sys.exit(0 if r.startswith(b"OK") else 1)
PY
}

_ensure_broker() {
  if _broker_ping; then
    return 0
  fi
  if [[ -d "${ROOT}/../bmtl/hardware_lab/lanes/m4/uma_toolkit/UMAStatus.app" ]]; then
    open "${ROOT}/../bmtl/hardware_lab/lanes/m4/uma_toolkit/UMAStatus.app" || true
    for _ in $(seq 1 40); do
      _broker_ping && return 0
      sleep 0.5
    done
  fi
  echo "FAIL: uma_daemon not running" >&2
  exit 1
}

_stop_lab() {
  local pids
  pids="$(lsof -nP -iTCP:"${M22_PORT}" -sTCP:LISTEN -t 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    # shellcheck disable=SC2086
    kill -TERM ${pids} 2>/dev/null || true
    sleep 2
    pids="$(lsof -nP -iTCP:"${M22_PORT}" -sTCP:LISTEN -t 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
      # shellcheck disable=SC2086
      kill -KILL ${pids} 2>/dev/null || true
    fi
  fi
}

_pick_bin() {
  if [[ -n "${M22_BIN:-}" && -x "${M22_BIN}" ]]; then
    echo "${M22_BIN}"
    return
  fi
  local cand
  for cand in \
    "${ROOT}/vendor/llama-cpp-"*/build/bin/llama-server \
    "${ROOT}/../llama.cpp/build/bin/llama-server"; do
    if [[ -x "${cand}" ]]; then
      echo "${cand}"
      return
    fi
  done
  echo ""
}

_pick_model() {
  if [[ -n "${M22_MODEL:-}" && -f "${M22_MODEL}" ]]; then
    echo "${M22_MODEL}"
    return
  fi
  python3 <<'PY'
import json, os
candidates = [
    os.path.expanduser("~/.ollama/models/manifests/registry.ollama.ai/library/m21-ggml/latest"),
    os.path.expanduser("~/.ollama/models/manifests/registry.ollama.ai/library/eliza-1-2b/latest"),
]
for man in candidates:
    if not os.path.isfile(man):
        continue
    d = json.load(open(man))
    for layer in d.get("layers") or []:
        if "model" in layer.get("mediaType", "") and "draft" not in layer.get("mediaType", ""):
            dig = layer["digest"].replace(":", "-")
            path = os.path.expanduser(f"~/.ollama/models/blobs/{dig}")
            if os.path.isfile(path):
                print(path)
                raise SystemExit(0)
raise SystemExit("no GGUF found (pull eliza-1-2b or set M22_MODEL)")
PY
}

_ensure_broker

BIN="$(_pick_bin)"
if [[ -z "${BIN}" ]]; then
  echo "error: no llama-server binary (build or set M22_BIN)" >&2
  exit 1
fi

if [[ "${M22_SKIP_BUILD:-}" != "1" ]]; then
  echo "== build llama-server (BUILD_UMA) =="
  BUILD_UMA=1 ./scripts/build/build_llama_server.sh
  BIN="$(_pick_bin)"
fi

MODEL="$(_pick_model)"
LIB_DIR="$(cd "$(dirname "${BIN}")" && pwd)"
LIB="${LIB_DIR}/libllama.dylib"

echo "== M22 llama-server UMA sign-off =="
echo "bin=${BIN} port=${M22_PORT} model=${MODEL}"

echo ""
echo "== [1] libllama carries uma client =="
if [[ ! -f "${LIB}" ]]; then
  echo "FAIL: missing ${LIB}" >&2
  exit 1
fi
if ! nm -gU "${LIB}" 2>/dev/null | grep -q 'uma_mlx_lease_begin'; then
  echo "FAIL: ${LIB} missing uma_mlx_lease_begin (rebuild with BUILD_UMA=1)" >&2
  exit 1
fi
echo "PASS: uma symbols in libllama"

echo ""
echo "== [2] require connect + warm completion =="
_stop_lab
: >"${LOG_DIR}/server.log"
env ZEROLLAMA_UMA_SCHED=require \
  ZEROLLAMA_UMA_SCHED_LOG=1 \
  DYLD_LIBRARY_PATH="${LIB_DIR}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}" \
  "${BIN}" --model "${MODEL}" --port "${M22_PORT}" --host 127.0.0.1 \
  --no-webui -c 1024 -ngl 99 \
  >>"${LOG_DIR}/server.log" 2>&1 &
for i in $(seq 1 90); do
  curl -sf -m 3 "${M22_URL}/health" >/dev/null 2>&1 && break
  sleep 0.5
done
if ! curl -sf -m 3 "${M22_URL}/health" >/dev/null; then
  echo "FAIL: llama-server not healthy on ${M22_URL}" >&2
  tail -40 "${LOG_DIR}/server.log" >&2
  exit 1
fi
curl -sS --max-time 180 "${M22_URL}/completion" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Say hi","n_predict":4,"temperature":0,"stream":false}' \
  | tee "${LOG_DIR}/warm.json" >/dev/null
python3 - "${LOG_DIR}/warm.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d.get("tokens_predicted", 0) >= 1 or d.get("content") is not None, d
print("PASS: warm completion")
PY
if ! grep -qE 'project=llama-server|uma_mlx: connected' "${LOG_DIR}/server.log"; then
  echo "FAIL: no uma connect evidence" >&2
  tail -30 "${LOG_DIR}/server.log" >&2
  exit 1
fi
if ! grep -qE 'llama-server-(prefill|decode)|lease begin phase=' "${LOG_DIR}/server.log"; then
  echo "FAIL: no lease begin (graph_compute UMA wrap missing?)" >&2
  exit 1
fi
echo "PASS: uma gate + leases"

echo ""
echo "== [3] HOLD_GPU competitor queues llama-server =="
_ensure_broker
: >"${LOG_DIR}/competitor.log"
python3 - "${UMA_SOCK:-/tmp/uma_daemon.sock}" <<'PY' >"${LOG_DIR}/competitor.log" 2>&1 &
import re, socket, sys, time
sock = sys.argv[1]

def tx(line, timeout=30.0):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(sock)
    s.sendall((line + "\n").encode())
    data = b""
    while b"\n" not in data:
        chunk = s.recv(8192)
        if not chunk:
            break
        data += chunk
    s.close()
    return data.decode(errors="replace").strip()

r = tx("SUBMIT name=m22-competitor HOLD_GPU")
m = re.search(r"ticket=(\d+)", r)
assert m, r
tid = int(m.group(1))
for _ in range(5000):
    j = tx(f"JOB {tid}")
    if "phase=holding" in j:
        break
    time.sleep(0.01)
else:
    raise SystemExit("hold timeout")
print(f"competitor holding ticket={tid}", flush=True)
time.sleep(5.0)
print(tx(f"RELEASE {tid}"), flush=True)
print(tx(f"WAIT {tid} 30"), flush=True)
PY
COMP_PID=$!
for i in $(seq 1 200); do
  if grep -q 'competitor holding' "${LOG_DIR}/competitor.log" 2>/dev/null; then
    break
  fi
  sleep 0.05
done
if ! grep -q 'competitor holding' "${LOG_DIR}/competitor.log"; then
  echo "FAIL: competitor never reached HOLD" >&2
  cat "${LOG_DIR}/competitor.log" >&2
  kill "${COMP_PID}" 2>/dev/null || true
  exit 1
fi
cat "${LOG_DIR}/competitor.log" | head -1
T0=$(python3 -c 'import time; print(time.time())')
curl -sS --max-time 180 "${M22_URL}/completion" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Say hi","n_predict":4,"temperature":0,"stream":false}' \
  | tee "${LOG_DIR}/contend.json" >/dev/null
T1=$(python3 -c 'import time; print(time.time())')
wait "${COMP_PID}"
ELAPSED=$(python3 -c "import sys; print(f'{float(sys.argv[1])-float(sys.argv[2]):.1f}')" "${T1}" "${T0}")
python3 -c "
import json
d=json.load(open('${LOG_DIR}/contend.json'))
assert d.get('tokens_predicted', 0) >= 1 or d.get('content') is not None, d
elapsed=float('${ELAPSED}')
assert elapsed >= 2.0, f'expected queue delay under HOLD, elapsed={elapsed}s'
print(f'PASS: completion under competitor HOLD (wall={elapsed}s)')
"
if ! grep -qE 'wait_ms=[1-9][0-9]{2,}' "${LOG_DIR}/server.log"; then
  # soft check — wall assert is primary
  echo "WARN: no large wait_ms line found (still OK if wall delay passed)" >&2
fi

echo ""
echo "== [4] ZEROLLAMA_UMA_SCHED=off ignores HOLD =="
_stop_lab
: >"${LOG_DIR}/server-off.log"
env ZEROLLAMA_UMA_SCHED=off \
  ZEROLLAMA_UMA_SCHED_LOG=1 \
  DYLD_LIBRARY_PATH="${LIB_DIR}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}" \
  "${BIN}" --model "${MODEL}" --port "${M22_PORT}" --host 127.0.0.1 \
  --no-webui -c 1024 -ngl 99 \
  >>"${LOG_DIR}/server-off.log" 2>&1 &
for i in $(seq 1 90); do
  curl -sf -m 3 "${M22_URL}/health" >/dev/null 2>&1 && break
  sleep 0.5
done
curl -sS --max-time 180 "${M22_URL}/completion" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"hi","n_predict":2,"temperature":0,"stream":false}' >/dev/null
: >"${LOG_DIR}/competitor-off.log"
python3 - "${UMA_SOCK:-/tmp/uma_daemon.sock}" <<'PY' >"${LOG_DIR}/competitor-off.log" 2>&1 &
import re, socket, sys, time
sock = sys.argv[1]

def tx(line, timeout=30.0):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(sock)
    s.sendall((line + "\n").encode())
    data = b""
    while b"\n" not in data:
        chunk = s.recv(8192)
        if not chunk:
            break
        data += chunk
    s.close()
    return data.decode(errors="replace").strip()

r = tx("SUBMIT name=m22-off-competitor HOLD_GPU")
m = re.search(r"ticket=(\d+)", r)
assert m, r
tid = int(m.group(1))
for _ in range(5000):
    j = tx(f"JOB {tid}")
    if "phase=holding" in j:
        break
    time.sleep(0.01)
else:
    raise SystemExit("hold timeout")
print(f"competitor holding ticket={tid}", flush=True)
time.sleep(5.0)
print(tx(f"RELEASE {tid}"), flush=True)
print(tx(f"WAIT {tid} 30"), flush=True)
PY
COMP_PID=$!
for i in $(seq 1 200); do
  grep -q 'competitor holding' "${LOG_DIR}/competitor-off.log" 2>/dev/null && break
  sleep 0.05
done
T0=$(python3 -c 'import time; print(time.time())')
curl -sS --max-time 60 "${M22_URL}/completion" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Say hi","n_predict":4,"temperature":0,"stream":false}' \
  | tee "${LOG_DIR}/off-contend.json" >/dev/null
T1=$(python3 -c 'import time; print(time.time())')
wait "${COMP_PID}"
ELAPSED=$(python3 -c "import sys; print(f'{float(sys.argv[1])-float(sys.argv[2]):.2f}')" "${T1}" "${T0}")
if grep -qE 'uma_mlx: connected|lease begin' "${LOG_DIR}/server-off.log"; then
  echo "FAIL: uma still active under ZEROLLAMA_UMA_SCHED=off" >&2
  rg -n 'uma_mlx' "${LOG_DIR}/server-off.log" | head -10 >&2 || true
  exit 1
fi
python3 -c "
import json
d=json.load(open('${LOG_DIR}/off-contend.json'))
assert d.get('tokens_predicted', 0) >= 1 or d.get('content') is not None, d
elapsed=float('${ELAPSED}')
assert elapsed < 2.0, f'off mode must not queue under HOLD, elapsed={elapsed}s'
print(f'PASS: off mode ungated under HOLD (wall={elapsed}s)')
"

_stop_lab
echo ""
echo "M22 llama-server UMA sign-off PASS (logs ${LOG_DIR})"
echo "Note: runtime subprocess + inprocess inherit this gate via the same libllama.dylib."
echo "Disable: ZEROLLAMA_UMA_SCHED=off (runtime) or BUILD_UMA=0 (compile out)."
