#!/usr/bin/env bash
# Linux vendor build: ane_draft_hook.cpp references g_handoff_step only declared under __APPLE__.
# WHY: A380 Vulkan llama-server build must not require Mac ANE bridge.
set -euo pipefail

ROOT="${1:-}"
if [[ -z "${ROOT}" || ! -f "${ROOT}/common/ane_draft_hook.cpp" ]]; then
  echo "patch_vendor_linux_ane_hook: bad root" >&2
  exit 1
fi

FILE="${ROOT}/common/ane_draft_hook.cpp"
MARK="ZEROLLAMA_LINUX_HANDOFF_STEP"

if grep -q "${MARK}" "${FILE}"; then
  exit 0
fi

python3 - <<PY
from pathlib import Path
path = Path("${FILE}")
text = path.read_text()
needle = "#include <vector>\n"
insert = needle + "\n#if !defined(__APPLE__)\n// ${MARK}\nstatic std::atomic<int> g_handoff_step { 0 };\n#endif\n"
if needle not in text:
    raise SystemExit("needle missing")
path.write_text(text.replace(needle, insert, 1))
print(f"patched {path}")
PY
