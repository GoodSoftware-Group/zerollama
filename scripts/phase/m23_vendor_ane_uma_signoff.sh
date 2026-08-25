#!/usr/bin/env bash
# M23 vendor e2e: ANE draft HOLD_ANE on vendor libllama-common (Darwin lab).
#
# Never binds or kills :11434 / :8081.
# Uses M23_PORT only if launching llama-server (default 18083).
#
# What this proves:
#   1) patch 0095 on vendor pin + ZEROLLAMA_UMA in llama-common
#   2) libllama exports uma_mlx_lease_begin_unit
#   3) ane_draft_session_step_once takes HOLD_ANE (queues under competitor)
#   4) HOLD_GPU ∥ HOLD_ANE still works (Metal decode admission independent)
#
# Usage:
#   ./scripts/phase/m23_vendor_ane_uma_signoff.sh
#   M23_SKIP_BUILD=1 ./scripts/phase/m23_vendor_ane_uma_signoff.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

FETCH_HEAD="$(grep '^FETCH_HEAD=' "${ROOT}/Makefile.sync" | cut -d= -f2)"
VENDOR="${M23_VENDOR:-${ROOT}/vendor/llama-cpp-${FETCH_HEAD}}"
BIN_DIR="${VENDOR}/build/bin"
LOG_DIR="${M23_LOG_DIR:-/tmp/m23-vendor-ane-uma-signoff}"
mkdir -p "${LOG_DIR}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "skip: M23 vendor ANE UMA sign-off is Darwin-only" >&2
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
  local app="${ROOT}/../bmtl/hardware_lab/lanes/m4/uma_toolkit/UMAStatus.app"
  if [[ -d "${app}" ]]; then
    # stale sock from dead daemon
    if [[ -S "${UMA_SOCK:-/tmp/uma_daemon.sock}" ]]; then
      rm -f "${UMA_SOCK:-/tmp/uma_daemon.sock}" || true
    fi
    open "${app}" || true
    for _ in $(seq 1 40); do
      _broker_ping && return 0
      sleep 0.5
    done
  fi
  echo "FAIL: uma_daemon not running (open UMAStatus.app)" >&2
  exit 1
}

echo "== M23 vendor ANE UMA sign-off =="
echo "vendor=${VENDOR}"

echo ""
echo "== [0] source + patch =="
test -f "${VENDOR}/common/ane_draft_session.mm"
grep -q 'ane_session_eval_ungated' "${VENDOR}/common/ane_draft_session.mm"
grep -q 'dlsym(RTLD_DEFAULT, "uma_mlx_lease_begin_unit")' "${VENDOR}/common/ane_draft_session.mm"
grep -q 'HOLD_ANE for ane_draft_session' "${VENDOR}/common/CMakeLists.txt"
test -f "${ROOT}/llama/patches/0095-darwin-uma-hold-ane-draft-eval.patch"
subj="$(git -C "${VENDOR}" log -1 --format=%s --grep='HOLD_ANE around ane_draft_session' 2>/dev/null || true)"
if [[ -z "${subj}" ]]; then
  # accept working tree wrap even if commit message drifted
  echo "WARN: vendor git subject for 0095 not found (tree wrap present)"
else
  echo "PASS: vendor commit: ${subj}"
fi
echo "PASS: vendor ANE HOLD_ANE wrap present"

_ensure_broker
python3 - <<'PY'
import socket, sys
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(2)
s.connect("/tmp/uma_daemon.sock")
s.sendall(b"INFO\n")
r = s.recv(4096).decode()
s.close()
if "HOLD_ANE" not in r and "holds=HOLD_GPU,HOLD_ANE" not in r:
    print("FAIL: broker lacks HOLD_ANE", r[:200], file=sys.stderr)
    sys.exit(1)
print("PASS: broker advertises HOLD_ANE")
PY

if [[ "${M23_SKIP_BUILD:-}" != "1" ]]; then
  echo ""
  echo "== [1] rebuild vendor llama-server (BUILD_UMA) =="
  make -C "${ROOT}/x/uma" llama
  BUILD_UMA=1 ./scripts/build/build_llama_server.sh
fi

LIB="${BIN_DIR}/libllama.dylib"
COMMON="${BIN_DIR}/libllama-common.dylib"
test -x "${BIN_DIR}/llama-server"
test -f "${LIB}"
test -f "${COMMON}"

echo ""
echo "== [2] uma unit symbols in libllama =="
if ! nm -gU "${LIB}" | grep -q 'uma_mlx_lease_begin_unit'; then
  echo "FAIL: ${LIB} missing uma_mlx_lease_begin_unit" >&2
  exit 1
fi
echo "PASS: uma_mlx_lease_begin_unit exported"

echo ""
echo "== [3] harness: ANE step queues under HOLD_ANE competitor =="
HARNESS_SRC="${LOG_DIR}/m23_ane_hold_harness.mm"
HARNESS_BIN="${LOG_DIR}/m23_ane_hold_harness"
cat >"${HARNESS_SRC}" <<'OBJC'
/* Lab harness: load vendor dylibs, run ane_draft_session_step_once under UMA. */
#import <Foundation/Foundation.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef bool (*bool_fn)(void);
typedef bool (*init_fn)(int, int, const char *, const char *);
typedef bool (*step_fn)(float);
typedef int (*acquire_fn)(void);
typedef int (*runtime_fn)(void);

int main(int argc, char **argv) {
  const char *bindir = argc > 1 ? argv[1] : ".";
  char path[1024];
  snprintf(path, sizeof(path), "%s/libllama.dylib", bindir);
  void *llama = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
  if (!llama) {
    fprintf(stderr, "dlopen libllama: %s\n", dlerror());
    return 2;
  }
  snprintf(path, sizeof(path), "%s/libllama-common.dylib", bindir);
  void *common = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
  if (!common) {
    fprintf(stderr, "dlopen libllama-common: %s\n", dlerror());
    return 2;
  }
  runtime_fn runtime = (runtime_fn)dlsym(llama, "uma_mlx_runtime_enabled");
  acquire_fn acquire = (acquire_fn)dlsym(llama, "uma_mlx_acquire");
  if (!runtime || !acquire) {
    fprintf(stderr, "missing uma acquire symbols\n");
    return 2;
  }
  if (acquire() != 0) {
    fprintf(stderr, "uma_mlx_acquire failed\n");
    return 3;
  }
  if (!runtime()) {
    fprintf(stderr, "uma runtime not active\n");
    return 3;
  }
  bool_fn supported = (bool_fn)dlsym(common, "ane_draft_session_supported");
  init_fn init = (init_fn)dlsym(common, "ane_draft_session_init");
  step_fn step = (step_fn)dlsym(common, "ane_draft_session_step_once");
  bool_fn ready = (bool_fn)dlsym(common, "ane_draft_session_ready");
  if (!supported || !init || !step || !ready) {
    fprintf(stderr, "missing ane_draft_session symbols\n");
    return 2;
  }
  if (!supported()) {
    fprintf(stderr, "ane_draft_session_supported=false (libane_bridge?)\n");
    return 4;
  }
  if (!init(64, 16, NULL, NULL)) {
    fprintf(stderr, "ane_draft_session_init failed\n");
    return 5;
  }
  if (!ready()) {
    fprintf(stderr, "ane session not ready\n");
    return 5;
  }
  const char *ready_path = getenv("M23_READY_FILE");
  if (ready_path && ready_path[0]) {
    FILE *rf = fopen(ready_path, "w");
    if (rf) {
      fputs("1", rf);
      fclose(rf);
    }
  }
  fprintf(stderr, "harness: ready — waiting for GO file\n");
  fflush(stderr);
  const char *go = getenv("M23_GO_FILE");
  if (go && go[0]) {
    for (int i = 0; i < 3000; i++) {
      if (access(go, F_OK) == 0)
        break;
      usleep(10000);
    }
    if (access(go, F_OK) != 0) {
      fprintf(stderr, "harness: GO timeout\n");
      return 7;
    }
  }
  CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
  bool ok = step(0.01f);
  CFAbsoluteTime t1 = CFAbsoluteTimeGetCurrent();
  printf("ok=%d wall_s=%.3f\n", ok ? 1 : 0, t1 - t0);
  return ok ? 0 : 6;
}
OBJC

clang++ -std=c++17 -fobjc-arc -O2 \
  -isysroot "$(xcrun --sdk macosx --show-sdk-path)" \
  -framework Foundation -framework CoreFoundation \
  -o "${HARNESS_BIN}" "${HARNESS_SRC}"

# Sync: harness writes READY → competitor HOLD_ANE → GO → harness step (queues) → RELEASE
rm -f "${LOG_DIR}/go" "${LOG_DIR}/ready"
: >"${LOG_DIR}/competitor.log"
python3 - "${UMA_SOCK:-/tmp/uma_daemon.sock}" "${LOG_DIR}" <<'PY' >"${LOG_DIR}/competitor.log" 2>&1 &
import os, re, socket, sys, time
sock, logdir = sys.argv[1], sys.argv[2]
ready = os.path.join(logdir, "ready")
go = os.path.join(logdir, "go")

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

for _ in range(600):
    if os.path.exists(ready):
        break
    time.sleep(0.05)
else:
    raise SystemExit("timeout waiting for harness READY")

r = tx("SUBMIT name=m23-competitor HOLD_ANE")
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
open(go, "w").write("1")
time.sleep(3.0)
print(tx(f"RELEASE {tid}"), flush=True)
print(tx(f"WAIT {tid} 30"), flush=True)
PY
COMP_PID=$!

export M23_GO_FILE="${LOG_DIR}/go"
export M23_READY_FILE="${LOG_DIR}/ready"
set +e
env ZEROLLAMA_UMA_SCHED=require \
  ZEROLLAMA_UMA_SCHED_LOG=1 \
  UMA_JOB_NAME=m23-ane-harness \
  M23_GO_FILE="${M23_GO_FILE}" \
  M23_READY_FILE="${M23_READY_FILE}" \
  DYLD_LIBRARY_PATH="${BIN_DIR}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}" \
  "${HARNESS_BIN}" "${BIN_DIR}" \
  >"${LOG_DIR}/harness.out" 2>"${LOG_DIR}/harness.err"
HRC=$?
set -e
wait "${COMP_PID}" || true
if ! grep -q 'competitor holding' "${LOG_DIR}/competitor.log"; then
  echo "FAIL: competitor never reached HOLD_ANE" >&2
  cat "${LOG_DIR}/competitor.log" >&2
  exit 1
fi

echo "--- harness.out ---"; cat "${LOG_DIR}/harness.out" || true
echo "--- harness.err (tail) ---"; tail -30 "${LOG_DIR}/harness.err" || true

if [[ "${HRC}" -ne 0 ]]; then
  echo "FAIL: harness exit ${HRC}" >&2
  exit 1
fi
WALL="$(python3 - <<PY
import re
t=open("${LOG_DIR}/harness.out").read()
m=re.search(r"wall_s=([0-9.]+)", t)
print(m.group(1) if m else "0")
PY
)"
python3 -c "
w=float('${WALL}')
assert w >= 2.0, f'expected ANE eval to queue under HOLD_ANE (>=2s), wall={w}'
print(f'PASS: ane_draft_session_step_once queued under HOLD_ANE (wall={w}s)')
"

echo ""
echo "== [4] HOLD_GPU ∥ HOLD_ANE (multi-unit) =="
python3 - <<'PY'
import re, socket, time, sys

def tx(cmd, timeout=10.0):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect("/tmp/uma_daemon.sock")
    s.sendall((cmd + "\n").encode())
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = s.recv(1024)
        if not chunk:
            break
        buf += chunk
    s.close()
    return buf.decode().strip()

def wait_holding(tid, label):
    t0 = time.time()
    while time.time() - t0 < 3.0:
        j = tx(f"JOB {tid}")
        if "phase=holding" in j and "state=running" in j:
            return j
        time.sleep(0.05)
    raise SystemExit(f"FAIL: {label} never holding: {tx(f'JOB {tid}')}")

g = tx("SUBMIT name=m23-gpu HOLD_GPU")
a = tx("SUBMIT name=m23-ane HOLD_ANE")
gid = int(re.search(r"ticket=(\d+)", g).group(1))
aid = int(re.search(r"ticket=(\d+)", a).group(1))
wait_holding(gid, "GPU")
wait_holding(aid, "ANE")
assert tx(f"RELEASE {gid}").startswith("OK")
assert tx(f"RELEASE {aid}").startswith("OK")
assert tx(f"WAIT {gid} 5").startswith("OK")
assert tx(f"WAIT {aid} 5").startswith("OK")
print("PASS: HOLD_GPU ∥ HOLD_ANE")
PY

echo ""
echo "M23 vendor ANE UMA sign-off PASS"
echo "log_dir=${LOG_DIR}"
