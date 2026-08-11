#!/usr/bin/env bash
# Opt-in ANE smoke: build probe + run via zerollama ane-probe (Darwin only).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "skip: ane_probe_smoke requires Darwin" >&2
  exit 0
fi

ANE_REPO="${ANE_REPO:-${HOME}/Sites/inference/ane}"
export ANE_REPO
RUN_SMOKE=1 ./scripts/ane_probe_build.sh

echo "== ane_probe_smoke: build zerollama =="
go build -o "${ROOT}/.ane_probe_smoke_bin" .
trap 'rm -f "${ROOT}/.ane_probe_smoke_bin"' EXIT

out="$("${ROOT}/.ane_probe_smoke_bin" ane-probe)"
echo "${out}"
echo "${out}" | grep -q '"ok":true'

bench="$("${ROOT}/.ane_probe_smoke_bin" ane-bench --quick)"
echo "${bench}"
echo "${bench}" | grep -q '"ok":true'

draft="$("${ROOT}/.ane_probe_smoke_bin" ane-draft-bench --quick)"
echo "${draft}"
echo "${draft}" | grep -q '"ok":true'

echo "ane_probe_smoke: PASS"
