#!/usr/bin/env bash
# Lab: verify ANE draft session sources gate with HOLD_ANE (M23).
# Optional: rebuild llama-server (BUILD_UMA) and nm-check symbols.
# Never touches :11434 / :8081.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MM="${ROOT}/llama/llama.cpp/common/ane_draft_session.mm"
CM="${ROOT}/llama/llama.cpp/common/CMakeLists.txt"
test -f "$MM" && test -f "$CM"

grep -q 'ane_session_eval_ungated' "$MM"
grep -q 'dlsym(RTLD_DEFAULT, "uma_mlx_lease_begin_unit")' "$MM"
grep -q 'HOLD_ANE for ane_draft_session' "$CM"
test -f "${ROOT}/llama/patches/0095-darwin-uma-hold-ane-draft-eval.patch"
echo "PASS: in-tree ANE draft HOLD_ANE wrap + patch present"

# Broker should advertise HOLD_ANE (F0390)
python3 - <<'PY'
import socket, sys
try:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(1)
    s.connect("/tmp/uma_daemon.sock")
    s.sendall(b"HELP\n")
    r = s.recv(2048).decode()
    s.close()
except Exception as e:
    print(f"WARN: broker down ({e}) — source checks only")
    sys.exit(0)
if "HOLD_ANE" not in r:
    print("FAIL: broker lacks HOLD_ANE", file=sys.stderr)
    sys.exit(1)
print("PASS: broker HOLD_ANE ready")
PY

if [[ "${M23_REBUILD:-0}" == "1" ]]; then
  echo "== rebuild llama-server with BUILD_UMA (lab) =="
  make -C x/uma llama
  BUILD_UMA=1 ./scripts/build/build_llama_server.sh
  BIN="${LLAMA_SERVER_BIN:-}"
  if [[ -z "$BIN" ]]; then
    # resolve like build script
    ROOT_LLAMA="$(./scripts/build/build_llama_server.sh 2>/dev/null | true)"
    for cand in \
      "${ROOT}/../llama.cpp/build/bin/llama-server" \
      "${ROOT}/vendor"/llama-cpp-*/build/bin/llama-server; do
      [[ -x "$cand" ]] && BIN="$cand" && break
    done
  fi
  if [[ -n "${BIN:-}" && -x "$BIN" ]]; then
    libdir="$(dirname "$BIN")"
    if nm -gU "${libdir}/libllama-common.dylib" 2>/dev/null | grep -q 'uma_mlx_lease_begin_unit' \
      || nm -gU "${libdir}/libllama.dylib" 2>/dev/null | grep -q 'uma_mlx_lease_begin_unit'; then
      echo "PASS: uma_mlx_lease_begin_unit linked"
    else
      echo "WARN: could not nm uma symbols (may be internal/hidden)"
    fi
  fi
fi

echo "M23 ANE HOLD_ANE source smoke PASS"
