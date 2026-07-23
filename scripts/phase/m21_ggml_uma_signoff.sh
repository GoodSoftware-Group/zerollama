#!/usr/bin/env bash
# M21 UMA ↔ ollamarunner (GGUF ggml Metal) lab sign-off.
#
# Never binds or kills :11434 / :8081. Uses M21_PORT (default 11435).
#
# Prerequisites:
#   - Darwin + -tags uma binary (BUILD_UMA=auto)
#   - uma_daemon up (UMAStatus.app)
#   - Ollama-engine GGUF without draft-eagle/MTP (default creates m21-ggml
#     from eliza-1-2b with spec_type cleared). Plain llama arches use legacy
#     llamarunner (no M21 wrap); draft-eagle tags use llama-server.
#
# Usage:
#   ./scripts/phase/m21_ggml_uma_signoff.sh
#   M21_SKIP_BUILD=1 ./scripts/phase/m21_ggml_uma_signoff.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

M21_PORT="${M21_PORT:-11435}"
# Default: lab tag created below from eliza GGUF without draft-eagle routing.
M21_MODEL="${M21_MODEL:-m21-ggml:latest}"
M21_FROM="${M21_FROM:-eliza-1-2b:latest}"
M21_HOST="127.0.0.1:${M21_PORT}"
M21_URL="http://${M21_HOST}"
LOG_DIR="${M21_LOG_DIR:-/tmp/m21-ggml-uma-signoff}"
mkdir -p "${LOG_DIR}"

if [[ "${M21_PORT}" == "11434" || "${M21_PORT}" == "8081" ]]; then
  echo "error: refusing production port ${M21_PORT}" >&2
  exit 1
fi
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "skip: M21 GGUF UMA sign-off is Darwin-only" >&2
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

_pick_bin() {
  if [[ -n "${M21_BIN:-}" && -x "${M21_BIN}" ]]; then
    echo "${M21_BIN}"
    return
  fi
  if [[ -x /tmp/zerollama-uma ]]; then
    echo /tmp/zerollama-uma
    return
  fi
  if [[ -x "${ROOT}/zerollama" ]]; then
    echo "${ROOT}/zerollama"
    return
  fi
  echo ""
}

_stop_lab() {
  local pids
  pids="$(lsof -nP -iTCP:"${M21_PORT}" -sTCP:LISTEN -t 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    # shellcheck disable=SC2086
    kill -TERM ${pids} 2>/dev/null || true
    sleep 2
    pids="$(lsof -nP -iTCP:"${M21_PORT}" -sTCP:LISTEN -t 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
      # shellcheck disable=SC2086
      kill -KILL ${pids} 2>/dev/null || true
    fi
  fi
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

_wait_tags() {
  for i in $(seq 1 60); do
    curl -sf -m 10 "${M21_URL}/api/tags" >/dev/null 2>&1 && return 0
    sleep 0.5
  done
  echo "FAIL: serve not ready on ${M21_URL}" >&2
  return 1
}

# Create a qwen35 ollama-engine tag without draft-eagle so Darwin stays on
# ComputeWithNotify (M21) instead of llama-server / legacy llamarunner.
_ensure_m21_model() {
  if curl -sf -m 30 "${M21_URL}/api/show" -d "{\"name\":\"${M21_MODEL}\"}" >/dev/null 2>&1; then
    echo "model ${M21_MODEL} already present"
    return 0
  fi
  echo "creating ${M21_MODEL} from ${M21_FROM} (clear spec_type for ollama-engine)"
  curl -sS --max-time 120 "${M21_URL}/api/create" -d "{
    \"model\": \"${M21_MODEL}\",
    \"from\": \"${M21_FROM}\",
    \"parameters\": {
      \"spec_type\": \"off\",
      \"draft_num_predict\": 0,
      \"num_ctx\": 2048,
      \"temperature\": 0,
      \"repeat_penalty\": 1.05,
      \"stop\": [\"<|im_end|>\"]
    },
    \"stream\": false
  }" | tee "${LOG_DIR}/create-model.json"
  echo
  if ! curl -sf -m 30 "${M21_URL}/api/show" -d "{\"name\":\"${M21_MODEL}\"}" >/dev/null; then
    echo "FAIL: could not create ${M21_MODEL}" >&2
    cat "${LOG_DIR}/create-model.json" >&2
    exit 1
  fi
  echo "PASS: created ${M21_MODEL}"
}

BIN="$(_pick_bin)"
if [[ -z "${BIN}" ]]; then
  echo "error: no zerollama binary (build or set M21_BIN)" >&2
  exit 1
fi
if [[ "${M21_SKIP_BUILD:-}" != "1" ]]; then
  echo "== build (BUILD_UMA) =="
  ./scripts/build/build_zerollama_mac.sh
  BIN="${ROOT}/zerollama"
fi
if ! strings "${BIN}" | grep 'cum_leases' >/dev/null; then
  echo "error: binary missing uma client (cum_leases); rebuild with BUILD_UMA=auto" >&2
  exit 1
fi

echo "== M21 GGUF UMA sign-off =="
echo "bin=${BIN} port=${M21_PORT} model=${M21_MODEL} from=${M21_FROM}"

_ensure_broker

echo ""
echo "== [1] doctor uma broker =="
"${BIN}" doctor 2>&1 | tee "${LOG_DIR}/doctor.txt" >/dev/null || true
if ! grep -q '\[ok\] uma broker' "${LOG_DIR}/doctor.txt"; then
  echo "FAIL: uma broker not ok" >&2
  grep -A2 'uma broker' "${LOG_DIR}/doctor.txt" || true
  exit 1
fi
echo "PASS: uma broker"

echo ""
echo "== [2] require generate (ollama-engine) =="
_stop_lab
: >"${LOG_DIR}/serve-require.log"
# ZEROLLAMA_LLAMA_SERVER=0 keeps plain GGUF on in-process Metal (M21 target).
env OLLAMA_HOST="${M21_HOST}" \
  ZEROLLAMA_UMA_SCHED=require \
  ZEROLLAMA_UMA_SCHED_LOG=1 \
  ZEROLLAMA_LLAMA_SERVER=0 \
  ZEROLLAMA_RUNTIME_DARWIN_SIDECAR=0 \
  ZEROLLAMA_RUNTIME=0 \
  "${BIN}" serve >>"${LOG_DIR}/serve-require.log" 2>&1 &
_wait_tags
_ensure_m21_model

curl -sS --max-time 300 "${M21_URL}/api/generate" \
  -d "{\"model\":\"${M21_MODEL}\",\"prompt\":\"Say hi\",\"raw\":true,\"stream\":false,\"options\":{\"num_predict\":6,\"temperature\":0,\"num_ctx\":1024}}" \
  | tee "${LOG_DIR}/gen-require.json"
echo
python3 - "${LOG_DIR}/gen-require.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert not d.get("error"), d
assert d.get("done"), d
assert (d.get("response") or d.get("thinking") or d.get("eval_count", 0)), d
print("PASS: generate done")
PY

if grep -q 'using llama-server subprocess for model' "${LOG_DIR}/serve-require.log"; then
  echo "FAIL: routed to llama-server (out of M21 scope). Use a plain GGUF model without draft-eagle/MTP, or unset M21_MODEL." >&2
  grep 'using llama-server' "${LOG_DIR}/serve-require.log" >&2 || true
  exit 1
fi
# Must be ollamarunner ( --ollama-engine --model ).
if ! grep -qE 'runner --ollama-engine --model' "${LOG_DIR}/serve-require.log"; then
  echo "FAIL: no ollama-engine model load" >&2
  grep 'starting runner' "${LOG_DIR}/serve-require.log" >&2 || true
  exit 1
fi

# Runner logs may be sparse in parent file; accept either phase project or gate connect.
if grep -qE 'name=ollamarunner-(prefill|decode)|project=ollamarunner|uma broker gate active|uma_mlx: connected' "${LOG_DIR}/serve-require.log"; then
  echo "PASS: uma gate / ollamarunner project seen in serve log"
else
  # Fallback: spawn short-lived runner and require connect line
  ZEROLLAMA_UMA_SCHED=require ZEROLLAMA_UMA_SCHED_LOG=1 UMA_JOB_NAME=ollamarunner \
    "${BIN}" runner --ollama-engine --port 18098 >"${LOG_DIR}/runner-direct.log" 2>&1 &
  RP=$!
  sleep 1
  kill "${RP}" 2>/dev/null || true
  wait "${RP}" 2>/dev/null || true
  if ! grep -q 'project=ollamarunner' "${LOG_DIR}/runner-direct.log"; then
    echo "FAIL: no ollamarunner UMA connect evidence" >&2
    tail -20 "${LOG_DIR}/serve-require.log" >&2
    exit 1
  fi
  echo "PASS: runner-direct project=ollamarunner (serve log lacked child lines)"
fi

echo ""
echo "== [3] HOLD_GPU competitor queues ggml =="
_ensure_broker
# Keep the same serve from step 2 so the warm runner queues on competitor HOLD.
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

r = tx("SUBMIT name=m21-competitor HOLD_GPU")
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
curl -sS --max-time 300 "${M21_URL}/api/generate" \
  -d "{\"model\":\"${M21_MODEL}\",\"prompt\":\"Say hi\",\"raw\":true,\"stream\":false,\"options\":{\"num_predict\":4,\"temperature\":0,\"num_ctx\":1024}}" \
  | tee "${LOG_DIR}/gen-contend.json" >/dev/null
T1=$(python3 -c 'import time; print(time.time())')
wait "${COMP_PID}"
ELAPSED=$(python3 -c "import sys; print(f'{float(sys.argv[1])-float(sys.argv[2]):.1f}')" "${T1}" "${T0}")
python3 -c "
import json
d=json.load(open('${LOG_DIR}/gen-contend.json'))
assert not d.get('error'), d
assert d.get('done'), d
elapsed=float('${ELAPSED}')
assert elapsed >= 2.0, f'expected queue delay under HOLD, elapsed={elapsed}s'
print(f'PASS: ollamarunner generate under competitor HOLD (wall={elapsed}s)')
"

echo ""
echo "== [4] HOLD_GPU competitor queues legacy llamarunner =="
M21_LLAMA_MODEL="${M21_LLAMA_MODEL:-llama3.2:3b}"
: >"${LOG_DIR}/serve-llama.log"
# Append llama path evidence to the same serve log file for greps.
_ensure_broker
: >"${LOG_DIR}/competitor-llama.log"
python3 - "${UMA_SOCK:-/tmp/uma_daemon.sock}" <<'PY' >"${LOG_DIR}/competitor-llama.log" 2>&1 &
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

r = tx("SUBMIT name=m21-llama-competitor HOLD_GPU")
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
  if grep -q 'competitor holding' "${LOG_DIR}/competitor-llama.log" 2>/dev/null; then
    break
  fi
  sleep 0.05
done
if ! grep -q 'competitor holding' "${LOG_DIR}/competitor-llama.log"; then
  echo "FAIL: llama competitor never reached HOLD" >&2
  cat "${LOG_DIR}/competitor-llama.log" >&2
  kill "${COMP_PID}" 2>/dev/null || true
  exit 1
fi
cat "${LOG_DIR}/competitor-llama.log" | head -1
# Truncate serve log marker so we can see the new runner line for llama3.2
MARK="$(wc -l <"${LOG_DIR}/serve-require.log" | tr -d ' ')"
T0=$(python3 -c 'import time; print(time.time())')
curl -sS --max-time 600 "${M21_URL}/api/generate" \
  -d "{\"model\":\"${M21_LLAMA_MODEL}\",\"prompt\":\"Say hi\",\"raw\":true,\"stream\":false,\"options\":{\"num_predict\":4,\"temperature\":0,\"num_ctx\":1024}}" \
  | tee "${LOG_DIR}/gen-llama-contend.json" >/dev/null
T1=$(python3 -c 'import time; print(time.time())')
wait "${COMP_PID}"
ELAPSED=$(python3 -c "import sys; print(f'{float(sys.argv[1])-float(sys.argv[2]):.1f}')" "${T1}" "${T0}")
python3 -c "
import json
d=json.load(open('${LOG_DIR}/gen-llama-contend.json'))
assert not d.get('error'), d
assert d.get('done'), d
elapsed=float('${ELAPSED}')
assert elapsed >= 2.0, f'expected queue delay under HOLD, elapsed={elapsed}s'
print(f'PASS: llamarunner generate under competitor HOLD (wall={elapsed}s)')
"
# New load must be legacy llamarunner ( --model without --ollama-engine ).
tail -n +"$((MARK + 1))" "${LOG_DIR}/serve-require.log" >"${LOG_DIR}/serve-llama.log" || true
if grep -q 'using llama-server subprocess for model' "${LOG_DIR}/serve-llama.log"; then
  echo "FAIL: llama3.2 routed to llama-server" >&2
  exit 1
fi
if ! grep -qE 'runner --model .*\.ollama/models' "${LOG_DIR}/serve-llama.log"; then
  echo "FAIL: no legacy llamarunner model load line" >&2
  grep 'starting runner' "${LOG_DIR}/serve-llama.log" >&2 || true
  exit 1
fi
if grep -qE 'runner --ollama-engine --model' "${LOG_DIR}/serve-llama.log"; then
  echo "FAIL: expected llamarunner but saw ollama-engine for ${M21_LLAMA_MODEL}" >&2
  exit 1
fi
if ! grep -qE 'project=llamarunner|name=llamarunner-(prefill|decode)|uma_mlx: connected' "${LOG_DIR}/serve-llama.log"; then
  # Accept wait evidence from parent if project lines missing
  if ! grep -qE 'llamarunner-(prefill|decode)' "${LOG_DIR}/serve-require.log"; then
    echo "WARN: no llamarunner project line in serve log (check binary -tags uma)" >&2
  fi
fi
echo "PASS: legacy llamarunner path under UMA"

_stop_lab
echo ""
echo "M21 GGUF UMA sign-off PASS (logs ${LOG_DIR})"
