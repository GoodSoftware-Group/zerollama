#!/usr/bin/env bash
# Lab: ZEROLLAMA_UMA_GRAIN=op vs phase (wishlist 4.1 / F0625).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if ! python3 - <<'PY'
import socket, sys
try:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(1)
    s.connect("/tmp/uma_daemon.sock")
    s.sendall(b"HELP\n")
    r = s.recv(4096).decode()
    s.close()
except Exception as e:
    print(f"FAIL: broker down ({e})", file=sys.stderr)
    sys.exit(1)
if "HOLD_GPU" not in r:
    print(f"FAIL: broker lacks HOLD_GPU\n{r}", file=sys.stderr)
    sys.exit(1)
print("PASS: broker HOLD ready")
PY
then
  exit 1
fi

echo "== rebuild libuma_embed.a =="
make -C x/uma

echo "== Go grain=op / grain=phase =="
export ZEROLLAMA_UMA_GRAIN_SMOKE=1
export ZEROLLAMA_UMA_SCHED=require
CGO_ENABLED=1 go test -tags uma -count=1 -timeout 90s ./x/uma/ \
  -run 'TestGrainOpAllowsGraphBetweenEvals|TestGrainPhaseQueuesPeerGraph'

echo "x/uma grain=op smoke PASS"
