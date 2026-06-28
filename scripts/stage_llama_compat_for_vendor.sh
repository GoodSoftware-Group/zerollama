#!/usr/bin/env bash
# Stage zerollama llama/compat into vendor llama.cpp for CMake builds.
#
# WHY: patch 0015 adds hook call sites; compat .cpp/.h live in zerollama (not vendor).
# Go CGO links compat via llama/compat/compat.go; llama-server needs target_sources wiring.
#
# Usage:
#   ./scripts/stage_llama_compat_for_vendor.sh /path/to/vendor/llama-cpp-c84b3020
set -euo pipefail

VENDOR_ROOT="${1:-}"
if [[ -z "${VENDOR_ROOT}" || ! -f "${VENDOR_ROOT}/CMakeLists.txt" ]]; then
  echo "usage: $0 /path/to/vendor/llama-cpp-<pin>" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPAT_SRC="${ROOT}/llama/compat"
STAGE="${VENDOR_ROOT}/src/ollama-compat"

mkdir -p "${STAGE}/models"
for f in \
  llama-ollama-compat.cpp \
  llama-ollama-compat.h \
  llama-ollama-compat-util.cpp \
  llama-ollama-compat-util.h; do
  ln -sf "${COMPAT_SRC}/${f}" "${STAGE}/${f}"
done
# Optional arch overlays (laguna etc.) ship with Go CGO only until upstream adds arch.

SRC_CMAKE="${VENDOR_ROOT}/src/CMakeLists.txt"
if ! grep -q 'ollama-compat/llama-ollama-compat.cpp' "${SRC_CMAKE}"; then
  python3 <<PY
from pathlib import Path
path = Path("${SRC_CMAKE}")
text = path.read_text()
needle = "            unicode.cpp\n"
insert = needle + (
    "            ollama-compat/llama-ollama-compat.cpp\n"
    "            ollama-compat/llama-ollama-compat-util.cpp\n"
)
if needle not in text:
    raise SystemExit("src/CMakeLists.txt layout changed; update stage_llama_compat_for_vendor.sh")
path.write_text(text.replace(needle, insert, 1))
PY
fi

if ! grep -q 'target_include_directories(llama PRIVATE ollama-compat)' "${SRC_CMAKE}"; then
  python3 <<PY
from pathlib import Path
path = Path("${SRC_CMAKE}")
text = path.read_text()
needle = "target_include_directories(llama PRIVATE .)\n"
insert = needle + "target_include_directories(llama PRIVATE ollama-compat)\n"
if needle not in text:
    raise SystemExit("src/CMakeLists include hook missing")
path.write_text(text.replace(needle, insert, 1))
PY
fi

MTMD_CMAKE="${VENDOR_ROOT}/tools/mtmd/CMakeLists.txt"
if ! grep -q 'OLLAMA_COMPAT_MTMD_BUILD' "${MTMD_CMAKE}"; then
  python3 <<PY
from pathlib import Path
path = Path("${MTMD_CMAKE}")
text = path.read_text()
needle = "target_include_directories(mtmd PRIVATE ../../vendor)\n"
insert = (
    needle
    + "target_include_directories(mtmd PRIVATE ../../src/ollama-compat)\n"
    + "target_compile_definitions(mtmd PRIVATE OLLAMA_COMPAT_MTMD_BUILD)\n"
)
if needle not in text:
    raise SystemExit("tools/mtmd/CMakeLists.txt layout changed")
path.write_text(text.replace(needle, insert, 1))
PY
fi

echo ">>> staged Ollama compat for ${VENDOR_ROOT}" >&2
