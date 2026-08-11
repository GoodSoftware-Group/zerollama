#!/usr/bin/env bash
# Lab: mlxrunner decode-gap OptiQ GRAPH probe (F0635) + stable dump (F0636).
# Prefers make optiq-live-dump → /tmp/uma_optiq_live_dump.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

UMA_TOOLKIT="${UMA_TOOLKIT:-}"
if [[ -z "$UMA_TOOLKIT" ]]; then
  for cand in \
    "$ROOT/../bmtl/hardware_lab/lanes/m4/uma_toolkit" \
    "$HOME/Sites/inference/bmtl/hardware_lab/lanes/m4/uma_toolkit"
  do
    if [[ -d "$cand" ]]; then
      UMA_TOOLKIT="$cand"
      break
    fi
  done
fi
if [[ -z "${UMA_TOOLKIT}" || ! -d "$UMA_TOOLKIT" ]]; then
  echo "FAIL: set UMA_TOOLKIT" >&2
  exit 1
fi

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
if "GEMV_Q4_G64" not in r and "graph_ops=" not in r:
    print(f"FAIL: broker lacks GEMV\n{r[:200]}", file=sys.stderr)
    sys.exit(1)
print("PASS: broker GRAPH ready")
PY
then
  exit 1
fi

DUMP="${UMA_OPTIQ_DUMP_DIR:-/tmp/uma_optiq_live_dump}"
export UMA_OPTIQ_DUMP_DIR="$DUMP"

echo "== stable live dump (F0636) =="
make -C "$UMA_TOOLKIT" optiq-live-dump
if [[ ! -f "$DUMP/wz.bin" ]]; then
  echo "SKIP: no live OptiQ dump"
  exit 0
fi
echo "PASS: dump at $DUMP"

echo "== rebuild libuma_embed.a =="
make -C x/uma

echo "== MaybeProbeOptiqLiveChain (default dump path) =="
export ZEROLLAMA_UMA_OPTIQ_PROBE_SMOKE=1
export ZEROLLAMA_UMA_OPTIQ_GRAPH_PROBE=require
export ZEROLLAMA_UMA_SCHED=require
export UMA_JOB_NAME=xuma-optiq-probe
# Unset override to exercise default /tmp/uma_optiq_live_dump resolution when DUMP is default.
if [[ "$DUMP" == "/tmp/uma_optiq_live_dump" ]]; then
  unset UMA_OPTIQ_DUMP_DIR || true
fi
CGO_ENABLED=1 go test -tags uma -count=1 -timeout 300s ./x/uma/ -run TestOptiqMaybeProbe

echo "== pipeline hook wired =="
rg -n 'MaybeProbeOptiqLiveChain|optiqProbeAttempted|OPTIQ_GRAPH_PROBE|/tmp/uma_optiq_live_dump' \
  x/mlxrunner/pipeline.go x/uma/optiq_probe.go
echo "PASS: probe defaults + pipeline hook (F0635/F0636)"

echo "m30 optiq GRAPH probe PASS"
