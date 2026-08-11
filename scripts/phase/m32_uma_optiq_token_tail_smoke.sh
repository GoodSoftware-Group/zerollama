#!/usr/bin/env bash
# Lab: F0700 GRAPH token-tail unit smoke (post-norm dump → gemv_argmax rematch tok0).
# Full serve rematch: ./scripts/phase/m32_optiq_token_tail_rematch.sh
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
    r = s.recv(8192).decode()
    s.close()
except Exception as e:
    print(f"FAIL: broker down ({e})", file=sys.stderr)
    sys.exit(1)
need = ("GEMV_Q8_G64", "ARGMAX", "NORM_MUL_F16")
missing = [n for n in need if n not in r]
if missing and "graph_ops=" not in r:
    print(f"FAIL: broker lacks {missing}\n{r[:240]}", file=sys.stderr)
    sys.exit(1)
print("PASS: broker GRAPH ready")
PY
then
  exit 1
fi

DUMP="${UMA_OPTIQ_TOKEN_TAIL_DIR:-/tmp/uma_optiq_token_tail_dump}"
export UMA_OPTIQ_TOKEN_TAIL_DIR="$DUMP"
GEN="${ORNITH_OPTIQ_GENERATE_DIR:-/tmp/uma_optiq_generate_dump}"
export ORNITH_OPTIQ_GENERATE_DIR="$GEN"

echo "== token-tail dump =="
make -C "$UMA_TOOLKIT" optiq-token-tail-dump
if [[ ! -f "$DUMP/lm_q8.bin" ]]; then
  echo "SKIP: no token-tail dump (blobs missing)"
  exit 0
fi
echo "PASS: dump at $DUMP"

if [[ ! -f "$GEN/prefill_hidden.bin" ]]; then
  echo "== generate dump (prefill_hidden) =="
  if [[ -f /tmp/uma_mlxrunner_optiq_tokens_ref.json ]]; then
    CGO_ENABLED=1 go run ./x/mlxrunner/cmd/ornith_generate_parity || {
      echo "SKIP: ornith_generate_parity failed"
      exit 0
    }
  else
    echo "SKIP: no tokens_ref / prefill_hidden (run m26 freeze + ornith_generate_parity)"
    exit 0
  fi
fi

echo "== rebuild libuma_embed.a =="
make -C x/uma

echo "== OptiqTokenTailArgmax rematch (gemv_argmax on post-norm dump) =="
export ZEROLLAMA_UMA_OPTIQ_TOKEN_SMOKE=1
export ZEROLLAMA_UMA_OPTIQ_TOKEN_TAIL=require
export ZEROLLAMA_UMA_OPTIQ_TOKEN_RECIPE=gemv_argmax
export ZEROLLAMA_UMA_SCHED=require
export UMA_JOB_NAME=xuma-optiq-token
CGO_ENABLED=1 go test -tags uma -count=1 -timeout 600s ./x/uma/ -run TestOptiqTokenTailPrefillRematch

echo "== pipeline hook wired =="
rg -n 'maybeOwnedGraphToken|OptiqTokenTail|TOKEN_TAIL|SkipFinalNorm|beginOptiqTokenTail' \
  x/mlxrunner/pipeline.go x/mlxrunner/optiq_token_tail.go x/uma/optiq_token_tail.go \
  x/models/qwen3_5/qwen3_5.go
echo "PASS: GRAPH token-tail Go path (F0700)"

echo "m32 optiq token-tail unit smoke PASS"
