#!/usr/bin/env bash
# Lab-only ANE FFN / SwiGLU checks. Never binds or kills :11434 / :8081.
#
# Usage:
#   ./scripts/ane/ane_ffn_lab_smoke.sh           # unit + host force smokes
#   ./scripts/ane/ane_ffn_lab_smoke.sh --print-env  # env block for lab serve
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${ROOT}/build/ane-probe-darwin/bin"
ANE_REPO="${ANE_REPO:-${HOME}/Sites/inference/ane}"

refuse_prod() {
  if lsof -nP -iTCP:11434 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "note: production :11434 is listening (left alone)" >&2
  fi
  if lsof -nP -iTCP:8081 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "note: production runtime :8081 is listening (left alone)" >&2
  fi
}

print_env() {
  cat <<EOF
# Canonical ANE FFN lab serve (shexp force+SwiGLU) — never :11434 / :8081
export OLLAMA_HOST=127.0.0.1:11435
export ZEROLLAMA_ANE_FFN=1
export ZEROLLAMA_ANE_FFN_MODE=force
export ZEROLLAMA_ANE_FFN_FORCE_ENABLE=1
export ZEROLLAMA_ANE_FFN_SWIGLU=1
export ZEROLLAMA_ANE_FFN_NAME=shexp
export ZEROLLAMA_ANE_FFN_IC=2048
export ZEROLLAMA_ANE_FFN_OC=512
export ZEROLLAMA_ANE_FFN_SEQ_MAX=512
export ZEROLLAMA_ANE_FFN_LAB_PORT=11435
export ZEROLLAMA_ANE_FFN_TELEMETRY=1
export ZEROLLAMA_ANE_FFN_REPLACE_DYLIB=${OUT}/libane_ffn_force.dylib
# Leave OVERLAP unset (quality broken on current pin).
# Dense eliza instead of shexp: NAME=ffn and omit IC/OC/INT8_*.
# Rebuild lab binary:
#   BUILD_MLX=0 BUILD_LLAMA_SERVER=0 BUILD_RUNTIME_KV_EXT=0 \\
#     ./scripts/build/build_zerollama_mac.sh ./zerollama-ane-ffn-lab
# Reinstall force dylib: ANE_REPO=… ./scripts/ane/ane_probe_build.sh
# OLLAMA_HOST=127.0.0.1:11435 ./zerollama-ane-ffn-lab serve
# Expect: metal_layout_replaced#N / swiglu_fp16_replaced#N (never on :11434)
EOF
}

if [[ "${1:-}" == "--print-env" ]]; then
  print_env
  exit 0
fi

refuse_prod

echo "== build probes =="
ANE_REPO="${ANE_REPO}" "${ROOT}/scripts/ane/ane_probe_build.sh"

echo "== fuse unit =="
"${OUT}/ane-prefill-ffn-fuse-unit-smoke"

echo "== policy =="
"${OUT}/ane-prefill-ffn-policy-smoke"

echo "== force matmul =="
"${OUT}/ane-prefill-ffn-force-smoke" --ic 512 --oc 256 --seq 64

echo "== force swiglu =="
"${OUT}/ane-prefill-ffn-swiglu-force-smoke" --ic 256 --hidden 128 --seq 64

echo "== force swiglu metal-layout (expert geom + pad seq) =="
"${OUT}/ane-prefill-ffn-swiglu-force-smoke" \
  --int8 --int8-in --metal-layout --ic 2048 --hidden 512 --seq 66 --iters 8 --warmup 2
"${OUT}/ane-prefill-ffn-swiglu-force-smoke" \
  --int8 --int8-in --metal-layout --ic 2048 --hidden 512 --seq 512 --iters 8 --warmup 2

echo "== metal package =="
( cd "${ROOT}" && go build -o /dev/null ./ml/backend/ggml/ggml/src/ggml-metal/ )

echo "PASS: ane_ffn_lab_smoke"
echo "For lab serve env: $0 --print-env"
