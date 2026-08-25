#!/usr/bin/env bash
# Lab: x/uma GRAPH FormatGraph + Submit/Wait (wishlist GRAPH-MLX 0.4 / F0624).
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
    s.sendall(b"HELP\n")
    r = s.recv(4096).decode()
    s.close()
except Exception as e:
    print(f"FAIL: broker down ({e})", file=sys.stderr)
    sys.exit(1)
if "graph=" not in r and "GRAPH" not in r:
    print(f"FAIL: broker lacks GRAPH — upgrade uma_daemon\n{r}", file=sys.stderr)
    sys.exit(1)
print("PASS: broker GRAPH ready")
PY
then
  exit 1
fi

echo "== rebuild libuma_embed.a =="
make -C x/uma

echo "== Go FormatGraph + Submit/Wait =="
export ZEROLLAMA_UMA_GRAPH_SMOKE=1
export ZEROLLAMA_UMA_SCHED=require
export UMA_JOB_NAME=xuma-graph
CGO_ENABLED=1 go test -tags uma -count=1 -timeout 60s ./x/uma/ -run TestGraphFormatAndSubmit

echo "x/uma GRAPH client smoke PASS"
