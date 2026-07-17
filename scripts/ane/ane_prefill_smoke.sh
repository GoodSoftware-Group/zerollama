#!/usr/bin/env bash
# Prefill lab smoke: ANE vs Metal matmul proxy at IC×OC×SEQ (Darwin only).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "skip: ane_prefill_smoke requires Darwin" >&2
  exit 0
fi

export ANE_REPO="${ANE_REPO:-${HOME}/Sites/inference/ane}"
./scripts/ane/ane_probe_build.sh

echo "== ane_prefill_smoke: build zerollama =="
go build -o "${ROOT}/.ane_prefill_smoke_bin" .
trap 'rm -f "${ROOT}/.ane_prefill_smoke_bin"' EXIT
BIN="${ROOT}/.ane_prefill_smoke_bin"

echo "== compare 256×256×512 =="
out="$("${BIN}" ane-prefill-bench --compare-metal --ic 256 --oc 256 --seq 512 --quick)"
echo "${out}"
echo "${out}" | grep -qE '"ok"[[:space:]]*:[[:space:]]*true'
echo "${out}" | grep -qE '"faster"[[:space:]]*:[[:space:]]*"ane"'
if echo "${out}" | grep -qE '"metal_mps"'; then
  echo "== MPS baseline present =="
  echo "${out}" | grep -qE '"mode"[[:space:]]*:[[:space:]]*"metal_mps_prefill_matmul"'
fi

echo "== sweep 256×256 quick grid =="
sweep="$("${BIN}" ane-prefill-sweep --ic 256 --oc 256 --quick)"
echo "${sweep}"
echo "${sweep}" | grep -qE '"ok"[[:space:]]*:[[:space:]]*true'
echo "${sweep}" | grep -qE '"ane_wins"[[:space:]]*:[[:space:]]*[1-9]'

echo "== lab status =="
"${BIN}" ane-lab-status

echo "== prefill handoff 256²×128 =="
handoff="$("${BIN}" ane-prefill-handoff-smoke --ic 256 --oc 256 --seq 128 --quick)"
echo "${handoff}"
echo "${handoff}" | grep -qE '"ok"[[:space:]]*:[[:space:]]*true'
echo "${handoff}" | grep -qE '"mode"[[:space:]]*:[[:space:]]*"metal_prefill_handoff"'

echo "ane_prefill_smoke: PASS"
