#!/usr/bin/env bash
# Mac UMA operator gate — M20–M23 admission (lab ports only).
#
# Never binds or kills :11434 / :8081.
#
# Default (ship ladder):
#   broker · doctor uma · M23 source · multi-unit · M23 vendor · M21 · M22
# Optional:
#   RUN_M20=1          — full mlxrunner M20 sign-off (heavy)
#   SKIP_M21=1         — skip GGUF ggml gate
#   SKIP_M22=1         — skip llama-server gate
#   SKIP_M23_VENDOR=1  — skip ANE HOLD harness
#   UMA_SKIP_BUILD=1   — pass *_SKIP_BUILD=1 to child gates (default 1)
#
# Usage:
#   ./scripts/phase/mac_uma_signoff.sh
#   RUN_M20=1 ./scripts/phase/mac_uma_signoff.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "skip: Mac UMA sign-off is Darwin-only" >&2
  exit 0
fi

UMA_SKIP_BUILD="${UMA_SKIP_BUILD:-1}"
LOG_DIR="${UMA_GATE_LOG_DIR:-/tmp/mac-uma-signoff}"
mkdir -p "${LOG_DIR}"
PASS=0
FAIL=0
SKIP=0

_run() {
  local name="$1"
  shift
  echo ""
  echo "======== ${name} ========"
  local rc=0
  set +e
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" -eq 0 ]]; then
    echo "PASS: ${name}"
    PASS=$((PASS + 1))
  else
    echo "FAIL: ${name} (exit ${rc})" >&2
    FAIL=$((FAIL + 1))
    return "${rc}"
  fi
}

_skip() {
  echo "SKIP: $1"
  SKIP=$((SKIP + 1))
}

_child_env() {
  if [[ "${UMA_SKIP_BUILD}" == "1" ]]; then
    export M20_SKIP_BUILD=1 M21_SKIP_BUILD=1 M22_SKIP_BUILD=1 M23_SKIP_BUILD=1
  fi
}

echo "== Mac UMA operator gate =="
echo "log_dir=${LOG_DIR} UMA_SKIP_BUILD=${UMA_SKIP_BUILD}"
_child_env

# --- broker ---
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

if ! _broker_ping; then
  app="${ROOT}/../bmtl/hardware_lab/lanes/m4/uma_toolkit/UMAStatus.app"
  if [[ -d "${app}" ]]; then
    if [[ -S "${UMA_SOCK:-/tmp/uma_daemon.sock}" ]]; then
      python3 -c 'import socket;s=socket.socket(socket.AF_UNIX);s.settimeout(0.3);s.connect("/tmp/uma_daemon.sock")' 2>/dev/null \
        || rm -f "${UMA_SOCK:-/tmp/uma_daemon.sock}" || true
    fi
    open "${app}" || true
    for _ in $(seq 1 40); do
      _broker_ping && break
      sleep 0.5
    done
  fi
fi
if ! _broker_ping; then
  echo "FAIL: uma_daemon not running (open UMAStatus.app)" >&2
  exit 1
fi
echo "PASS: uma_daemon up"

# --- doctor uma ---
doctor_uma_checks() {
  local bin="" c
  for c in /tmp/zerollama-uma "${ROOT}/zerollama"; do
    if [[ -x "$c" ]]; then bin="$c"; break; fi
  done
  python3 - <<'PY'
import socket, sys
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(2)
s.connect("/tmp/uma_daemon.sock")
s.sendall(b"INFO\n")
info = s.recv(4096).decode()
s.close()
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(2)
s.connect("/tmp/uma_daemon.sock")
s.sendall(b"HELP\n")
help_ = s.recv(4096).decode()
s.close()
assert "HOLD_GPU" in info or "HOLD_GPU" in help_, info
assert "HOLD_ANE" in info or "HOLD_ANE" in help_, info
print("PASS: broker INFO/HELP advertise HOLD_GPU + HOLD_ANE")
PY
  if [[ -n "${bin}" ]]; then
    "${bin}" doctor 2>&1 | tee "${LOG_DIR}/doctor.txt"
    if ! grep -qi "uma broker" "${LOG_DIR}/doctor.txt"; then
      echo "FAIL: doctor missing uma broker line" >&2
      return 1
    fi
    if grep -qiE '\[warn\].*uma broker|\[fail\].*uma broker' "${LOG_DIR}/doctor.txt"; then
      echo "FAIL: doctor uma broker not ok" >&2
      grep -i "uma broker" "${LOG_DIR}/doctor.txt" >&2 || true
      return 1
    fi
    if ! grep -qiE 'uma broker.*HOLD_ANE|HOLD_GPU\+ANE' "${LOG_DIR}/doctor.txt"; then
      echo "FAIL: doctor uma broker missing HOLD_ANE detail — rebuild zerollama (cmd/doctor_uma.go)" >&2
      grep -i "uma broker" "${LOG_DIR}/doctor.txt" >&2 || true
      return 1
    fi
    echo "PASS: zerollama doctor uma broker ok (${bin})"
  else
    echo "WARN: no zerollama binary — skipped CLI doctor"
  fi
  local fetch_head lib
  fetch_head="$(grep '^FETCH_HEAD=' "${ROOT}/Makefile.sync" | cut -d= -f2)"
  lib="${ROOT}/vendor/llama-cpp-${fetch_head}/build/bin/libllama.dylib"
  if [[ -f "${lib}" ]]; then
    nm -gU "${lib}" | grep -q uma_mlx_lease_begin_unit
    nm -gU "${lib}" | grep -q uma_mlx_lease_begin
    echo "PASS: vendor libllama UMA symbols (${lib})"
  else
    echo "WARN: vendor libllama.dylib missing — build with BUILD_UMA=1"
  fi
}
_run "doctor_uma" doctor_uma_checks

# --- M23 source + multiunit (fast) ---
_run "m23_source" ./scripts/phase/m23_ane_hold_source_smoke.sh
_run "m23_multiunit" ./scripts/phase/m23_uma_multiunit_client_smoke.sh

# --- GRAPH / grain / BUF client surface (F0624–F0627; fast) ---
_run "m24_graph" ./scripts/phase/m24_uma_graph_client_smoke.sh
_run "m25_grain" ./scripts/phase/m25_uma_grain_op_smoke.sh
_run "m27_buf" ./scripts/phase/m27_uma_buf_graph_smoke.sh
if [[ "${SKIP_M28_OPTIQ_CHAIN:-0}" == "1" ]]; then
  _skip "m28_optiq_chain"
else
  _run "m28_optiq_chain" ./scripts/phase/m28_uma_optiq_live_chain_smoke.sh
fi
if [[ "${SKIP_M29_OPTIQ_HOLD:-0}" == "1" ]]; then
  _skip "m29_optiq_hold"
else
  _run "m29_optiq_hold" ./scripts/phase/m29_uma_optiq_live_chain_hold_smoke.sh
fi
if [[ "${SKIP_M30_OPTIQ_PROBE:-0}" == "1" ]]; then
  _skip "m30_optiq_probe"
else
  _run "m30_optiq_probe" ./scripts/phase/m30_uma_optiq_graph_probe_smoke.sh
fi
if [[ "${SKIP_M31_OPTIQ_GRAPH_GEN:-1}" == "1" ]]; then
  _skip "m31_optiq_graph_generate"
else
  _run "m31_optiq_graph_generate" ./scripts/phase/m31_optiq_graph_generate_rematch.sh
fi
if [[ "${SKIP_M26_OPTIQ_FREEZE:-1}" == "1" ]]; then
  _skip "m26_optiq_freeze"
else
  _run "m26_optiq_freeze" ./scripts/phase/m26_mlxrunner_optiq_tokens_freeze.sh
fi

# --- M23 vendor e2e ---
if [[ "${SKIP_M23_VENDOR:-0}" == "1" ]]; then
  _skip "m23_vendor"
else
  _run "m23_vendor" ./scripts/phase/m23_vendor_ane_uma_signoff.sh
fi

# --- M21 GGUF ggml ---
if [[ "${SKIP_M21:-0}" == "1" ]]; then
  _skip "m21_ggml"
else
  # Distinct lab port so we never touch production or collide with a leftover M20 serve.
  M21_PORT="${M21_PORT:-11436}" \
    _run "m21_ggml" ./scripts/phase/m21_ggml_uma_signoff.sh || true
fi

# --- M22 llama-server ---
if [[ "${SKIP_M22:-0}" == "1" ]]; then
  _skip "m22_llama_server"
else
  M22_PORT="${M22_PORT:-18082}" \
    _run "m22_llama_server" ./scripts/phase/m22_llama_server_uma_signoff.sh || true
fi

# --- optional M20 ---
if [[ "${RUN_M20:-0}" == "1" ]]; then
  M20_PORT="${M20_PORT:-11435}" \
    _run "m20_mlx" ./scripts/phase/m20_uma_signoff.sh || true
else
  _skip "m20_mlx (set RUN_M20=1)"
fi

echo ""
echo "== Mac UMA gate summary: pass=${PASS} fail=${FAIL} skip=${SKIP} =="
if [[ "${FAIL}" -gt 0 ]]; then
  echo "Mac UMA operator gate FAIL" >&2
  exit 1
fi
echo "Mac UMA operator gate PASS"
