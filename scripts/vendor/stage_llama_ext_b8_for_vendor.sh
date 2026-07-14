#!/usr/bin/env bash
# Stage B8 cross-attention C API into vendor for patch 0018 (ANE dflash handoff).
#
# Patch 0018 adds ane_draft_hook.cpp but not llama_context_cross_* — those live in
# zerollama in-tree llama/llama.cpp/src until a dedicated vendor patch lands.
set -euo pipefail

VENDOR_ROOT="${1:-}"
if [[ -z "${VENDOR_ROOT}" || ! -f "${VENDOR_ROOT}/CMakeLists.txt" ]]; then
  echo "usage: $0 /path/to/vendor/llama-cpp-<pin>" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INTREE="${ROOT}/llama/llama.cpp/src"
VENDOR_SRC="${VENDOR_ROOT}/src"

if [[ ! -f "${VENDOR_ROOT}/common/ane_draft_hook.cpp" ]]; then
  exit 0
fi

if grep -q 'llama_context_cross_has_v_embd' "${VENDOR_SRC}/llama-context.cpp" 2>/dev/null; then
  exit 0
fi

if [[ ! -f "${INTREE}/llama-ext.h" || ! -f "${INTREE}/llama-context.cpp" ]]; then
  echo "error: in-tree B8 sources missing under ${INTREE}" >&2
  exit 1
fi

echo ">>> staging B8 cross-attention API → ${VENDOR_ROOT}" >&2
install -m 644 "${INTREE}/llama-ext.h" "${VENDOR_SRC}/llama-ext.h"

python3 <<'PY'
from pathlib import Path
import re
import sys

vendor_ctx = Path(sys.argv[1])
in_tree = Path(sys.argv[2])
text = vendor_ctx.read_text()
if "llama_context_cross_has_v_embd" in text:
    sys.exit(0)

src = in_tree.read_text()
m = re.search(
    r"(bool llama_context_cross_has_v_embd\(const llama_context \* ctx\) \{.*?"
    r"void llama_context_cross_upsert_row\(.*?\n\})\n",
    src,
    re.S,
)
if not m:
    raise SystemExit("B8 cross API block not found in in-tree llama-context.cpp")

block = m.group(1) + "\n"
anchor = "    return ctx->get_embeddings_pre_norm_ith(i);\n}\n\n"
if anchor not in text:
    raise SystemExit("llama-context.cpp anchor missing; update stage_llama_ext_b8_for_vendor.sh")
if "#include <algorithm>" not in text:
    text = text.replace("#include <cstring>\n", "#include <cstring>\n#include <algorithm>\n", 1)
vendor_ctx.write_text(text.replace(anchor, anchor + block, 1))
print("  grafted llama_context_cross_* into llama-context.cpp")
PY
"${VENDOR_SRC}/llama-context.cpp" "${INTREE}/llama-context.cpp"

echo ">>> staged B8 cross-attention API for ${VENDOR_ROOT}" >&2
