#!/usr/bin/env bash
# Lab: x/uma Buf* + live OptiQ Wz→Wo GEMV chain (F0633).
# Dumps converted packs via uma_toolkit chain smoke, then Go rematch.
#
# Env:
#   UMA_TOOLKIT — path to hardware_lab/.../uma_toolkit (default: sibling bmtl tree)
#   SKIP if ornith blobs missing (chain smoke exits 0).
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
  echo "FAIL: set UMA_TOOLKIT to uma_toolkit tree" >&2
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
    print(f"FAIL: broker lacks GEMV_Q4_G64\n{r[:200]}", file=sys.stderr)
    sys.exit(1)
print("PASS: broker GRAPH ready")
PY
then
  exit 1
fi

DUMP="$(mktemp -d /tmp/uma_m28_optiq.XXXXXX)"
trap 'rm -rf "$DUMP"' EXIT
export UMA_OPTIQ_DUMP_DIR="$DUMP"

echo "== dump live chain packs (C) =="
make -C "$UMA_TOOLKIT" test_uma_optiq_live_chain_smoke
if ! "$UMA_TOOLKIT/test_uma_optiq_live_chain_smoke"; then
  echo "FAIL: C live chain dump" >&2
  exit 1
fi
if [[ ! -f "$DUMP/wz.bin" ]]; then
  echo "SKIP: no live OptiQ dump (blobs missing?)"
  exit 0
fi
echo "PASS: dump $DUMP (nw=$(wc -c <"$DUMP/wz.bin"))"

echo "== rebuild libuma_embed.a =="
make -C x/uma

echo "== Go live OptiQ chain =="
export ZEROLLAMA_UMA_OPTIQ_CHAIN_SMOKE=1
export ZEROLLAMA_UMA_SCHED=require
export UMA_JOB_NAME=xuma-optiq
CGO_ENABLED=1 go test -tags uma -count=1 -timeout 300s ./x/uma/ -run TestOptiqLiveChainGraph

echo "x/uma OptiQ live chain smoke PASS"
