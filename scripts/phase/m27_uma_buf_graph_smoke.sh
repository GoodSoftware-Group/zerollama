#!/usr/bin/env bash
# Lab: x/uma BUF_* + RESIDUAL_ADD GRAPH (F0627).
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
if "RESIDUAL_ADD" not in r and "graph_ops=" not in r:
    print(f"FAIL: broker lacks GRAPH/RESIDUAL\n{r[:200]}", file=sys.stderr)
    sys.exit(1)
print("PASS: broker GRAPH ready")
PY
then
  exit 1
fi

echo "== rebuild libuma_embed.a =="
make -C x/uma

echo "== Go BUF + RESIDUAL_ADD =="
export ZEROLLAMA_UMA_BUF_SMOKE=1
export ZEROLLAMA_UMA_SCHED=require
export UMA_JOB_NAME=xuma-buf
CGO_ENABLED=1 go test -tags uma -count=1 -timeout 60s ./x/uma/ -run TestBufResidualGraph

echo "x/uma BUF GRAPH smoke PASS"
