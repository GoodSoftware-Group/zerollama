#!/usr/bin/env bash
# Lab: F0699/F0719 in-process OptiQ GRAPH generate rematch (cgo → libuma_optiq_graph_gen).
# Gate: GRAPH_GEN_TOKENS matches tokens_ref / got_gen suffix [12675,248046].
# F0719: Go test covers dump-compat (nil prompt) AND explicit prompt_ids from dump.
# Go path must use in-process (invalid BIN; no ALLOW_EXEC).
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
export UMA_TOOLKIT

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

DUMP="${ORNITH_OPTIQ_GENERATE_DIR:-/tmp/uma_optiq_generate_dump}"
export ORNITH_OPTIQ_GENERATE_DIR="$DUMP"
REF="${ORNITH_OPTIQ_TOKENS_REF:-/tmp/uma_mlxrunner_optiq_tokens_ref.json}"

echo "== ensure generate dump (meta.json) =="
if [[ ! -f "$DUMP/meta.json" ]]; then
  if [[ ! -f "$REF" ]]; then
    echo "FAIL: missing $REF (run m26 freeze first) and $DUMP/meta.json" >&2
    exit 1
  fi
  echo "NOTE: building dump via ornith_generate_parity"
  go run ./x/mlxrunner/cmd/ornith_generate_parity
fi
if [[ ! -f "$DUMP/meta.json" ]]; then
  echo "FAIL: still no $DUMP/meta.json" >&2
  exit 1
fi
echo "PASS: dump at $DUMP"

echo "== build libuma_optiq_graph_gen.dylib (F0699) =="
make -C "$UMA_TOOLKIT" libuma_optiq_graph_gen.dylib
test -f "$UMA_TOOLKIT/libuma_optiq_graph_gen.dylib"
echo "PASS: dylib present"

echo "== build + run F0697 cascade L0 generate smoke =="
make -C "$UMA_TOOLKIT" optiq-live-cascade-l0-multit-generate-smoke | tee /tmp/f0699_cascade.out
if ! rg -q 'GRAPH_GEN_TOKENS=12675,248046' /tmp/f0699_cascade.out; then
  echo "FAIL: expected GRAPH_GEN_TOKENS=12675,248046" >&2
  rg GRAPH_GEN_TOKENS /tmp/f0699_cascade.out || true
  exit 1
fi
echo "PASS: GRAPH_GEN_TOKENS rematch"

# Invalid BIN: Go must still PASS via in-process dylib (F0699 claim).
export UMA_OPTIQ_GRAPH_GENERATE_BIN="/nonexistent/optiq_graph_generate_bin"
unset UMA_OPTIQ_GRAPH_GENERATE_ALLOW_EXEC || true

echo "== Go RunOptiqGraphGenerate rematch (in-process; invalid BIN) =="
make -C x/uma
export ZEROLLAMA_UMA_OPTIQ_GRAPH_GENERATE_SMOKE=1
export ZEROLLAMA_UMA_OPTIQ_GRAPH_GENERATE=require
CGO_ENABLED=1 go test -tags uma -count=1 -timeout 45m ./x/uma/ -run TestOptiqGraphGenerateRematch

echo "== pipeline hook present =="
rg -n 'OptiqGraphGenerateEnabled|RunOptiqGraphGenerate|emitOptiqGraphGenerate|OPTIQ_GRAPH_GENERATE|in-process|uma_optiq_graph_generate' \
  x/mlxrunner/pipeline.go x/uma/optiq_graph_generate.go
echo "PASS: pipeline + uma helper wired (F0699/F0719 in-process any-prompt)"

echo "m31 optiq GRAPH generate rematch PASS"
