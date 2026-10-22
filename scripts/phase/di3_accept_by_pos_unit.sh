#!/usr/bin/env bash
# DI3 host unit — verified accept_by_pos EWMA (no GPU / no llama-server).
# Exit: compile + run tools/server/unit/test-di3-accept-by-pos.cpp
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
UNIT="$ROOT/llama/llama.cpp/tools/server/unit"
SRC="$UNIT/test-di3-accept-by-pos.cpp"
OUT="${TMPDIR:-/tmp}/zerollama-di3-accept-by-pos-unit"

CXX="${CXX:-c++}"
"$CXX" -std=c++17 -O0 -Wall -Wextra -I"$UNIT" -o "$OUT" "$SRC"
"$OUT"
echo "di3_accept_by_pos_unit: PASS"
