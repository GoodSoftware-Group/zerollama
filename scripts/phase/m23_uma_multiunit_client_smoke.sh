#!/usr/bin/env bash
# Lab: x/uma multi-unit leases (HOLD_GPU ∥ HOLD_ANE) against machine uma_daemon.
# Never touches :11434 / :8081.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if ! python3 - <<'PY'
import socket, sys
try:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(1)
    s.connect("/tmp/uma_daemon.sock")
    s.sendall(b"INFO\n")
    r = s.recv(2048).decode()
    s.close()
except Exception as e:
    print(f"FAIL: broker down ({e})", file=sys.stderr)
    sys.exit(1)
if "HOLD_ANE" not in r and "holds=HOLD_GPU,HOLD_ANE" not in r:
    print(f"FAIL: broker lacks HOLD_ANE — upgrade uma_daemon\n{r}", file=sys.stderr)
    sys.exit(1)
print("PASS: broker multi-unit HOLD ready")
PY
then
  exit 1
fi

echo "== rebuild libuma_embed.a =="
make -C x/uma

echo "== Go LeaseBeginUnit gpu∥ane =="
export ZEROLLAMA_UMA_MULTIUNIT_SMOKE=1
export ZEROLLAMA_UMA_SCHED=require
export UMA_JOB_NAME=xuma-multiunit
CGO_ENABLED=1 go test -tags uma -count=1 -timeout 60s ./x/uma/ -run TestMultiUnitLeaseGPUAndANE

echo "M23 x/uma multi-unit client smoke PASS"
