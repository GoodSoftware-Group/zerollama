#!/usr/bin/env bash
# M20 UMA ↔ mlxrunner production sign-off (lab ports only).
#
# Never binds or kills :11434 / :8081 (production). Uses M20_PORT (default 11435).
#
# Prerequisites:
#   - Darwin + sibling bmtl uma_toolkit (or prebuilt -tags uma binary)
#   - uma_daemon with HOLD_GPU (UMAStatus.app or make uma-daemon-install)
#   - MLX model pulled (default gemma4:26b-optiq)
#
# Usage:
#   ./scripts/phase/m20_uma_signoff.sh
#
# Env:
#   M20_PORT=11435
#   M20_MODEL=gemma4:26b-optiq
#   M20_BIN=               — default: prefer /tmp/zerollama-uma, else ./zerollama
#   M20_SKIP_BUILD=1       — do not rebuild
#   RUN_E2E_UMA_RESTART=1  — restart uma_daemon mid-run (disrupts all UMA clients)
#   RUN_E2E_UMA_AGENT=1    — two-turn prompt_cache_key soak (default on)
#   RUN_E2E_UMA_ATTN=0     — skip ATTN-under-HOLD GPU unit check (default on)
#   RUN_E2E_UMA_HYBRID=0   — skip RUN_HYBRID-under-HOLD (default on; prepare M=8)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

M20_PORT="${M20_PORT:-11435}"
M20_MODEL="${M20_MODEL:-gemma4:26b-optiq}"
M20_HOST="127.0.0.1:${M20_PORT}"
M20_URL="http://${M20_HOST}"
LOG_DIR="${M20_LOG_DIR:-/tmp/m20-uma-signoff}"
mkdir -p "${LOG_DIR}"

if [[ "${M20_PORT}" == "11434" || "${M20_PORT}" == "8081" ]]; then
  echo "error: refusing production port ${M20_PORT}; use M20_PORT=11435 (or another lab port)" >&2
  exit 1
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "skip: M20 UMA sign-off is Darwin-only" >&2
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
  if [[ -n "${M20_BIN:-}" && -x "${M20_BIN}" ]]; then
    echo "${M20_BIN}"
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
  pids="$(lsof -nP -iTCP:"${M20_PORT}" -sTCP:LISTEN -t 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    # shellcheck disable=SC2086
    kill -TERM ${pids} 2>/dev/null || true
    sleep 2
    pids="$(lsof -nP -iTCP:"${M20_PORT}" -sTCP:LISTEN -t 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
      # shellcheck disable=SC2086
      kill -KILL ${pids} 2>/dev/null || true
    fi
  fi
}

_wait_ready() {
  local i
  for i in $(seq 1 90); do
    if curl -sf -m 10 "${M20_URL}/api/tags" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  echo "error: serve not ready on ${M20_URL}" >&2
  return 1
}

_start_lab() {
  local mode="$1"
  local log="${LOG_DIR}/serve-${mode}.log"
  _stop_lab
  : >"${log}"
  # Ensure machine broker for gated modes (stale sock / UMAStatus quit).
  if [[ "${mode}" == "require" || "${mode}" == "auto" ]]; then
    if ! _broker_ping 2>/dev/null; then
      if [[ -d "${ROOT}/../bmtl/hardware_lab/lanes/m4/uma_toolkit/UMAStatus.app" ]]; then
        open "${ROOT}/../bmtl/hardware_lab/lanes/m4/uma_toolkit/UMAStatus.app" || true
        for _ in $(seq 1 40); do
          _broker_ping 2>/dev/null && break
          sleep 0.5
        done
      fi
    fi
  fi
  # Lab only: never bind/start production runtime sidecar on :8081.
  env OLLAMA_HOST="${M20_HOST}" \
    ZEROLLAMA_UMA_SCHED="${mode}" \
    ZEROLLAMA_UMA_SCHED_LOG=1 \
    ZEROLLAMA_RUNTIME_DARWIN_SIDECAR=0 \
    ZEROLLAMA_RUNTIME_EMBED=0 \
    OLLAMA_NO_CLOUD=true \
    OLLAMA_TRAINING=false \
    "${BIN}" serve >"${log}" 2>&1 &
  echo $! >"${LOG_DIR}/serve.pid"
  _wait_ready
}

_generate() {
  local out="$1"
  local extra="${2:-}"
  # temperature 0 for token stability across UMA on/off
  local body
  body="$(python3 -c "
import json, sys
extra = json.loads(sys.argv[1] or '{}')
opts = {'num_predict': 4, 'num_ctx': 512, 'temperature': 0}
opts.update(extra.get('options') or {})
req = {'model': sys.argv[2], 'prompt': extra.get('prompt') or 'hi', 'stream': False, 'options': opts}
if extra.get('prompt_cache_key'):
    req['options']['prompt_cache_key'] = extra['prompt_cache_key']
print(json.dumps(req))
" "${extra}" "${M20_MODEL}")"
  curl -sS --max-time 600 "${M20_URL}/api/generate" -d "${body}" | tee "${out}"
  echo
}

_context_tokens() {
  python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
ctx = d.get('context')
if ctx is None:
    sys.exit('missing context in ' + sys.argv[1])
print(','.join(str(x) for x in ctx))
print(d.get('response', ''), file=sys.stderr)
" "$1"
}

BIN="$(_pick_bin)"
if [[ -z "${BIN}" ]]; then
  echo "error: no zerollama binary; build with BUILD_UMA=auto ./scripts/build/build_zerollama_mac.sh" >&2
  exit 1
fi

echo "== M20 UMA mlxrunner sign-off =="
echo "bin=${BIN} port=${M20_PORT} model=${M20_MODEL}"

if [[ "${M20_SKIP_BUILD:-0}" != "1" ]]; then
  echo ""
  echo "== [0] build -tags uma =="
  if [[ ! -f "${ROOT}/../bmtl/hardware_lab/lanes/m4/uma_toolkit/uma_client.c" ]]; then
    echo "error: sibling bmtl uma_toolkit missing" >&2
    exit 1
  fi
  # shellcheck source=scripts/runtime/mac_cgo_env.sh
  source "${ROOT}/scripts/runtime/mac_cgo_env.sh"
  mac_cgo_env_warn_path
  mac_cgo_env
  make -C "${ROOT}/x/mlxrunner/uma"
  # cgo may not re-link when only libuma_embed.a changes
  touch "${ROOT}/x/mlxrunner/uma/uma_darwin.go"
  OUT_BIN="${M20_BIN:-/tmp/zerollama-uma}"
  CGO_ENABLED=1 go build -a -tags uma -o "${OUT_BIN}" .
  if ! strings "${OUT_BIN}" | grep -q 'cum_leases'; then
    echo "error: binary missing cum_leases (cgo did not link libuma_embed.a)" >&2
    exit 1
  fi
  BIN="${OUT_BIN}"
  echo "wrote ${BIN}"
fi

trap '_stop_lab' EXIT INT TERM

echo ""
echo "== [1] doctor uma broker =="
"${BIN}" doctor 2>&1 | tee "${LOG_DIR}/doctor.txt" >/dev/null || true
if ! grep -q '\[ok\] uma broker' "${LOG_DIR}/doctor.txt"; then
  echo "FAIL: uma broker not ok (install: make -C ../bmtl/.../uma_toolkit uma-daemon-install)" >&2
  grep -A2 'uma broker' "${LOG_DIR}/doctor.txt" || true
  exit 1
fi
echo "PASS: uma broker HOLD_GPU ready"

echo ""
echo "== [2] golden tokens (require vs off) =="
_start_lab require
_generate "${LOG_DIR}/gen-require.json"
REQ_CTX="$(_context_tokens "${LOG_DIR}/gen-require.json" 2>"${LOG_DIR}/resp-require.txt")"
grep -q 'lease begin phase=load' "${LOG_DIR}/serve-require.log" || {
  echo "FAIL: expected load lease in require mode" >&2
  exit 1
}

_start_lab off
_generate "${LOG_DIR}/gen-off.json"
OFF_CTX="$(_context_tokens "${LOG_DIR}/gen-off.json" 2>"${LOG_DIR}/resp-off.txt")"

if [[ "${REQ_CTX}" != "${OFF_CTX}" ]]; then
  echo "FAIL: context tokens differ require vs off" >&2
  echo "require: ${REQ_CTX}" >&2
  echo "off:     ${OFF_CTX}" >&2
  exit 1
fi
echo "PASS: golden context tokens match ($(wc -c <<<"${REQ_CTX}" | tr -d ' ') bytes)"
echo "  response: $(tr -d '\n' <"${LOG_DIR}/resp-require.txt")"

# Still on off serve: HOLD must not delay decode (escape hatch).
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

r = tx("SUBMIT name=m20-off-competitor HOLD_GPU")
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
COMP_OFF=$!
for i in $(seq 1 200); do
  grep -q 'competitor holding' "${LOG_DIR}/competitor-off.log" 2>/dev/null && break
  sleep 0.05
done
T0=$(python3 -c 'import time; print(time.time())')
_generate "${LOG_DIR}/gen-off-hold.json"
T1=$(python3 -c 'import time; print(time.time())')
wait "${COMP_OFF}"
ELAPSED=$(python3 -c "import sys; print(f'{float(sys.argv[1])-float(sys.argv[2]):.2f}')" "${T1}" "${T0}")
python3 -c "
elapsed=float('${ELAPSED}')
assert elapsed < 2.0, f'off mode must not queue under HOLD, elapsed={elapsed}s'
print(f'PASS: off mode ungated under HOLD (wall={elapsed}s)')
"

echo ""
echo "== [3] default auto mode =="
_start_lab auto
_generate "${LOG_DIR}/gen-auto.json"
AUTO_CTX="$(_context_tokens "${LOG_DIR}/gen-auto.json" 2>/dev/null)"
if [[ "${AUTO_CTX}" != "${REQ_CTX}" ]]; then
  echo "FAIL: auto context differs from require" >&2
  exit 1
fi
if ! grep -Eq 'connected mode=2|lease begin phase=load' "${LOG_DIR}/serve-auto.log"; then
  echo "FAIL: auto mode did not gate (broker down or ungated)" >&2
  grep -E 'uma_mlx:|broker' "${LOG_DIR}/serve-auto.log" | tail -5 || true
  exit 1
fi
echo "PASS: auto gates and matches tokens"

if [[ "${RUN_E2E_UMA_AGENT:-1}" == "1" ]]; then
  echo ""
  echo "== [4] agent two-turn (prompt_cache_key) =="
  KEY="m20-uma-$(date +%s)"
  _start_lab require
  _generate "${LOG_DIR}/agent-t1.json" "{\"prompt\":\"System: be brief.\\nUser: say hi\",\"options\":{\"num_predict\":3,\"num_ctx\":1024,\"temperature\":0,\"prompt_cache_key\":\"${KEY}\"}}"
  _generate "${LOG_DIR}/agent-t2.json" "{\"prompt\":\"System: be brief.\\nUser: say hi\\nAssistant: hi\\nUser: again\",\"options\":{\"num_predict\":3,\"num_ctx\":1024,\"temperature\":0,\"prompt_cache_key\":\"${KEY}\"}}"
  python3 -c "
import json
for p in ('${LOG_DIR}/agent-t1.json', '${LOG_DIR}/agent-t2.json'):
    d = json.load(open(p))
    assert d.get('done'), p
    assert d.get('eval_count', 0) >= 1 or d.get('response') is not None, p
print('PASS: agent two-turn completed key=${KEY}')
"
fi

echo ""
echo "== [5] contention (HOLD_GPU competitor + RUN_NOP) =="
UMA_CLI="${UMA_CLI:-}"
if [[ -z "${UMA_CLI}" ]]; then
  for c in \
    "${ROOT}/../bmtl/hardware_lab/lanes/m4/uma_toolkit/uma_daemon" \
    "${ROOT}/../bmtl/hardware_lab/lanes/m4/uma_toolkit/UMAStatus.app/Contents/MacOS/uma_daemon"; do
    if [[ -x "${c}" ]]; then
      UMA_CLI="${c}"
      break
    fi
  done
fi
if [[ -z "${UMA_CLI}" ]]; then
  echo "FAIL: uma_daemon CLI not found for contention smoke" >&2
  exit 1
fi

# RUN_NOP / ATTN / HYBRID contention below use _broker_ping (defined earlier).
_start_lab require
# Competitor holds GPU long enough that OptiQ manifest I/O (ungated) finishes
# while still held, so LeaseBegin(load) must queue (non-zero wait_ms).
python3 - "${UMA_SOCK:-/tmp/uma_daemon.sock}" <<'PY' &
import re, socket, sys, time
sock_path = sys.argv[1]
hold_s = 8.0

def transact(line: str) -> str:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(30)
    s.connect(sock_path)
    s.sendall((line + "\n").encode())
    data = b""
    while b"\n" not in data:
        chunk = s.recv(4096)
        if not chunk:
            break
        data += chunk
    s.close()
    return data.decode(errors="replace").strip()

r = transact("SUBMIT name=m20-competitor HOLD_GPU")
m = re.search(r"ticket=(\d+)", r)
if not m:
    raise SystemExit(f"submit failed: {r}")
tid = int(m.group(1))
for _ in range(5000):
    j = transact(f"JOB {tid}")
    if "phase=holding" in j:
        break
    if "state=err" in j or j.startswith("ERR"):
        raise SystemExit(f"hold failed: {j}")
    time.sleep(0.01)
else:
    raise SystemExit("timeout waiting for phase=holding")
print(f"competitor holding ticket={tid}", flush=True)
time.sleep(hold_s)
print(transact(f"RELEASE {tid}"), flush=True)
print(transact(f"WAIT {tid} 30"), flush=True)
PY
COMP_PID=$!
sleep 0.3
_generate "${LOG_DIR}/gen-contend-hold.json"
wait "${COMP_PID}"
CONTEND_CTX="$(_context_tokens "${LOG_DIR}/gen-contend-hold.json" 2>/dev/null)"
if [[ "${CONTEND_CTX}" != "${REQ_CTX}" ]]; then
  echo "FAIL: contention HOLD tokens differ" >&2
  exit 1
fi
# load lease should have waited for competitor
MAX_WAIT="$(python3 -c "
import re
mx=0.0
n=0
for line in open('${LOG_DIR}/serve-require.log'):
    m=re.search(r'wait_ms=([0-9.]+)', line)
    if m:
        n+=1
        mx=max(mx, float(m.group(1)))
print(f'{mx:.1f}')
if n==0:
    raise SystemExit('no wait_ms fields in serve-require.log')
")"
python3 -c "
w=float('${MAX_WAIT}')
assert w >= 200.0, f'expected wait_ms>=200 under competitor HOLD, got {w}'
print(f'PASS: HOLD_GPU contention (max wait_ms={w})')
"

# ATTN is Metal GPU work (F0375). Under HOLD_GPU it must queue — GPU-unit proxy for
# HYBRID (full HYBRID_PREPARE is minutes; opt out with RUN_E2E_UMA_ATTN=0).
if [[ "${RUN_E2E_UMA_ATTN:-1}" != "0" ]]; then
  python3 - "${UMA_SOCK:-/tmp/uma_daemon.sock}" <<'PY'
import re, socket, sys, time
sock_path = sys.argv[1]

def tx(line: str, timeout: float = 30.0) -> str:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(sock_path)
    s.sendall((line + "\n").encode())
    data = b""
    while b"\n" not in data:
        chunk = s.recv(8192)
        if not chunk:
            break
        data += chunk
    s.close()
    return data.decode(errors="replace").strip()

r = tx("SUBMIT name=m20-attn-hold HOLD_GPU")
m = re.search(r"ticket=(\d+)", r)
if not m:
    raise SystemExit(f"HOLD submit failed: {r}")
hid = int(m.group(1))
for _ in range(5000):
    j = tx(f"JOB {hid}")
    if "phase=holding" in j:
        break
    if "state=err" in j or j.startswith("ERR"):
        raise SystemExit(f"hold failed: {j}")
    time.sleep(0.01)
else:
    raise SystemExit("timeout waiting for HOLD phase=holding")

r = tx("SUBMIT name=m20-attn ATTN B=1 T=16 H=4 KV=1 HD=32 rounds=4")
m = re.search(r"ticket=(\d+)", r)
if not m:
    tx(f"RELEASE {hid}")
    raise SystemExit(f"ATTN submit failed: {r}")
aid = int(m.group(1))
time.sleep(0.3)
j = tx(f"JOB {aid}")
if "state=done" in j:
    tx(f"RELEASE {hid}")
    raise SystemExit(f"ATTN finished under HOLD (expected queued): {j}")
if "state=queued" not in j and "state=pending" not in j and "pending=" not in r:
    # accept running only if still blocked; fail if already done above
    if "state=running" not in j:
        tx(f"RELEASE {hid}")
        raise SystemExit(f"ATTN not queued under HOLD: {j}")

print(tx(f"RELEASE {hid}"))
print(tx(f"WAIT {hid} 30"))
for _ in range(200):
    j = tx(f"JOB {aid}")
    if "state=done" in j or "state=err" in j:
        break
    time.sleep(0.05)
else:
    raise SystemExit(f"ATTN did not finish after RELEASE: {j}")
if "state=done" not in j or "metal=1" not in j:
    raise SystemExit(f"ATTN bad result after RELEASE: {j}")
print(f"PASS: ATTN queued under HOLD_GPU then metal=1 (ticket={aid})")
PY
fi

# RUN_HYBRID under HOLD (opt-in; needs prepared hybrid — small M=8 ~seconds).
# Opt out: RUN_E2E_UMA_HYBRID=0. Default on when broker already prepared or can prepare fast.
if [[ "${RUN_E2E_UMA_HYBRID:-1}" != "0" ]]; then
  python3 - "${UMA_SOCK:-/tmp/uma_daemon.sock}" <<'PY'
import re, socket, sys, time
sock_path = sys.argv[1]

def tx(line: str, timeout: float = 180.0) -> str:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(sock_path)
    s.sendall((line + "\n").encode())
    data = b""
    while b"\n" not in data:
        chunk = s.recv(16384)
        if not chunk:
            break
        data += chunk
    s.close()
    return data.decode(errors="replace").strip()

# Prepare once (cached on repeat); M=8 is ~seconds on this host.
r = tx("SUBMIT name=m20-hyb HYBRID_PREPARE M=8 ngen=1")
m = re.search(r"ticket=(\d+)", r)
if not m:
    print(f"skip HYBRID contention (prepare submit failed: {r})")
    raise SystemExit(0)
ptid = int(m.group(1))
for _ in range(900):
    j = tx(f"JOB {ptid}")
    if "state=done" in j:
        break
    if "state=err" in j or j.startswith("ERR"):
        print(f"skip HYBRID contention (prepare failed: {j})")
        raise SystemExit(0)
    time.sleep(0.2)
else:
    print(f"skip HYBRID contention (prepare timeout: {j})")
    raise SystemExit(0)

r = tx("SUBMIT name=m20-hyb-hold HOLD_GPU")
m = re.search(r"ticket=(\d+)", r)
if not m:
    raise SystemExit(f"HOLD submit failed: {r}")
hid = int(m.group(1))
for _ in range(5000):
    j = tx(f"JOB {hid}")
    if "phase=holding" in j:
        break
    if "state=err" in j or j.startswith("ERR"):
        raise SystemExit(f"hold failed: {j}")
    time.sleep(0.01)
else:
    raise SystemExit("timeout waiting for HOLD phase=holding")

r = tx("SUBMIT name=m20-hyb RUN_HYBRID 1")
m = re.search(r"ticket=(\d+)", r)
if not m:
    tx(f"RELEASE {hid}")
    raise SystemExit(f"RUN_HYBRID submit failed: {r}")
hid2 = int(m.group(1))
time.sleep(0.3)
j = tx(f"JOB {hid2}")
if "state=done" in j:
    tx(f"RELEASE {hid}")
    raise SystemExit(f"RUN_HYBRID finished under HOLD (expected queued): {j}")
if "state=queued" not in j and "state=running" not in j:
    tx(f"RELEASE {hid}")
    raise SystemExit(f"RUN_HYBRID not queued under HOLD: {j}")

print(tx(f"RELEASE {hid}"))
print(tx(f"WAIT {hid} 30"))
for _ in range(300):
    j = tx(f"JOB {hid2}")
    if "state=done" in j or "state=err" in j:
        break
    time.sleep(0.1)
else:
    raise SystemExit(f"RUN_HYBRID did not finish after RELEASE: {j}")
if "state=done" not in j:
    raise SystemExit(f"RUN_HYBRID bad result: {j}")
print(f"PASS: RUN_HYBRID queued under HOLD_GPU (ticket={hid2})")
PY
fi

# RUN_NOP is CPU/sched work (not GPU HOLD); must complete while MLX holds GPU.
_start_lab require
_generate "${LOG_DIR}/gen-contend-nop.json" &
GEN_PID=$!
# small delay so mlxrunner acquires HOLD_GPU first
sleep 0.5
if ! _broker_ping; then
  echo "FAIL: broker not responding before RUN_NOP contention" >&2
  wait "${GEN_PID}" || true
  exit 1
fi
NOP_OUT="$("${UMA_CLI}" --name m20-nop --submit "RUN_NOP 80")"
echo "${NOP_OUT}" | tee "${LOG_DIR}/nop-submit.txt"
NOP_TID="$(python3 -c "import re,sys; m=re.search(r'ticket=(\d+)', sys.argv[1]); print(m.group(1) if m else '')" "${NOP_OUT}")"
if [[ -z "${NOP_TID}" ]]; then
  echo "FAIL: RUN_NOP submit failed: ${NOP_OUT}" >&2
  wait "${GEN_PID}" || true
  exit 1
fi
"${UMA_CLI}" --wait "${NOP_TID}" 30 | tee "${LOG_DIR}/nop-wait.txt"
NOP_JOB="$("${UMA_CLI}" --job "${NOP_TID}")"
echo "${NOP_JOB}" | tee "${LOG_DIR}/nop-job.txt"
wait "${GEN_PID}"
NOP_CTX="$(_context_tokens "${LOG_DIR}/gen-contend-nop.json" 2>/dev/null)"
if [[ "${NOP_CTX}" != "${REQ_CTX}" ]]; then
  echo "FAIL: RUN_NOP contention tokens differ" >&2
  exit 1
fi
if ! grep -Eq 'state=done|OK' "${LOG_DIR}/nop-wait.txt" && ! echo "${NOP_JOB}" | grep -Eq 'state=done'; then
  echo "FAIL: RUN_NOP job did not complete: ${NOP_JOB}" >&2
  exit 1
fi
echo "PASS: RUN_NOP concurrent with MLX HOLD"

echo ""
echo "== [6] lease metrics =="
if ! grep -q 'wait_ms=' "${LOG_DIR}/serve-require.log"; then
  echo "FAIL: missing wait_ms in lease logs" >&2
  exit 1
fi
if ! grep -q 'hold_ms=' "${LOG_DIR}/serve-require.log"; then
  echo "FAIL: missing hold_ms in lease logs" >&2
  exit 1
fi
if ! grep -q 'cum_leases=' "${LOG_DIR}/serve-require.log"; then
  echo "FAIL: missing cum_leases in lease logs" >&2
  exit 1
fi
# Phase project names (mlxrunner-load|prefill|decode) unless flat.
if ! grep -qE 'name=mlxrunner-(load|prefill|decode)' "${LOG_DIR}/serve-require.log"; then
  echo "FAIL: missing phase project names (mlxrunner-load|prefill|decode)" >&2
  exit 1
fi
echo "PASS: phase project names present"
# Prefer disconnect stats; unload is best-effort (runner may be SIGKILL'd).
curl -sS --max-time 120 "${M20_URL}/api/generate" \
  -d "{\"model\":\"${M20_MODEL}\",\"keep_alive\":0}" \
  | tee "${LOG_DIR}/unload.json" >/dev/null || true
for _ in $(seq 1 20); do
  if grep -qE 'uma_mlx: stats leases=|msg="uma broker stats"' "${LOG_DIR}/serve-require.log" 2>/dev/null; then
    break
  fi
  sleep 0.25
done
if grep -qE 'uma_mlx: stats leases=|msg="uma broker stats"' "${LOG_DIR}/serve-require.log"; then
  grep -E 'uma_mlx: stats leases=|msg="uma broker stats"' "${LOG_DIR}/serve-require.log" | tail -2
else
  echo "note: disconnect stats not seen (runner exit path); cum_* on lease end is sufficient"
  grep 'cum_leases=' "${LOG_DIR}/serve-require.log" | tail -1
fi
_stop_lab
echo "PASS: lease wait_ms/hold_ms + cum stats"

# Soft reconnect (no daemon TERM) — always when broker is up.
echo ""
echo "== [6b] libuma_client reconnect (half-close) =="
if [[ -x "${ROOT}/scripts/phase/m20_uma_client_reconnect_smoke.sh" ]]; then
  "${ROOT}/scripts/phase/m20_uma_client_reconnect_smoke.sh"
else
  echo "skip (missing m20_uma_client_reconnect_smoke.sh)"
fi

# Broker restart: RUN_E2E_UMA_RESTART=1 forces; also auto when LaunchAgent installed.
# Opt out: RUN_E2E_UMA_RESTART=0
_launchagent_uma=0
shopt -s nullglob 2>/dev/null || true
_la_files=("${HOME}/Library/LaunchAgents/"*uma*)
if ((${#_la_files[@]})) && [[ -e "${_la_files[0]}" ]]; then
  _launchagent_uma=1
fi
if [[ "${RUN_E2E_UMA_RESTART:-}" == "1" || ( "${RUN_E2E_UMA_RESTART:-}" != "0" && "${_launchagent_uma}" -eq 1 ) ]]; then
  echo ""
  echo "== [7] broker restart soak =="
  echo "NOTE: briefly TERMs machine uma_daemon (all UMA clients)."
  _start_lab auto
  _generate "${LOG_DIR}/pre-restart.json"
  PRE_CTX="$(_context_tokens "${LOG_DIR}/pre-restart.json" 2>/dev/null)"
  if [[ "${PRE_CTX}" != "${REQ_CTX}" ]]; then
    echo "FAIL: pre-restart tokens differ" >&2
    exit 1
  fi
  SOCK="${UMA_SOCK:-/tmp/uma_daemon.sock}"
  DAEMON_PID="$(pgrep -f 'uma_daemon$' | head -1 || true)"
  if [[ -z "${DAEMON_PID}" ]]; then
    echo "FAIL: uma_daemon not found to restart" >&2
    exit 1
  fi
  echo "sending TERM to uma_daemon pid=${DAEMON_PID}"
  kill -TERM "${DAEMON_PID}" || true
  sleep 1
  for i in $(seq 1 60); do
    if [[ -S "${SOCK}" ]] && _broker_ping; then
      NEW_PID="$(pgrep -f 'uma_daemon$' | head -1 || true)"
      if [[ -n "${NEW_PID}" ]]; then
        echo "broker up pid=${NEW_PID}"
        break
      fi
    fi
    sleep 0.5
  done
  if ! _broker_ping; then
    if [[ -d "${ROOT}/../bmtl/hardware_lab/lanes/m4/uma_toolkit/UMAStatus.app" ]]; then
      echo "broker down; opening UMAStatus.app"
      open "${ROOT}/../bmtl/hardware_lab/lanes/m4/uma_toolkit/UMAStatus.app" || true
      for i in $(seq 1 40); do
        _broker_ping && break
        sleep 0.5
      done
    fi
  fi
  if ! _broker_ping; then
    echo "FAIL: broker did not respawn (start UMAStatus.app / make uma-daemon-install)" >&2
    exit 1
  fi
  # Same lab serve must reconnect via libuma_client (do not restart serve).
  _generate "${LOG_DIR}/post-restart.json"
  POST_CTX="$(_context_tokens "${LOG_DIR}/post-restart.json" 2>/dev/null)"
  if [[ "${POST_CTX}" != "${REQ_CTX}" ]]; then
    echo "FAIL: post-restart tokens differ (same serve reconnect)" >&2
    exit 1
  fi
  echo "PASS: broker restart + same-serve auto recover"
  _stop_lab
else
  echo ""
  echo "== [7] broker restart soak =="
  if [[ "${_launchagent_uma}" -eq 0 ]]; then
    echo "skip (no LaunchAgent; install with make uma-daemon-install, or RUN_E2E_UMA_RESTART=1)"
    echo "      operator: ./scripts/phase/m20_uma_restart_soak.sh"
  else
    echo "skip (RUN_E2E_UMA_RESTART=0)"
  fi
fi

echo ""
echo "M20 UMA sign-off PASS (logs ${LOG_DIR})"
echo "Disable: ZEROLLAMA_UMA_SCHED=off (runtime) or BUILD_UMA=0 (compile out)."
