#!/usr/bin/env bash
# Restore zerollama-only ANE draft hook sources under llama/llama.cpp after vendor rsync.
# Why: vendor/ lacks ane_draft_* until llama/patches/0018 lands; rsync --delete would drop them.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CANON="${ROOT}/tools/ane-patches/canonical/common"
INTREE="${ROOT}/llama/llama.cpp/common"

if [[ ! -d "${CANON}" ]]; then
  echo "restore_ane_hook_intree: missing ${CANON}" >&2
  exit 1
fi

echo ">>> restore ANE hook sources → ${INTREE}" >&2
install -d "${INTREE}"
for f in ane_draft_hook.h ane_draft_hook.cpp ane_draft_session.h ane_draft_session.mm \
         ane_draft_session_stub.cpp ane_iosurface_map.h; do
  install -m 644 "${CANON}/${f}" "${INTREE}/${f}"
done

python3 "${ROOT}/tools/ane-patches/apply_speculative_ane_hook.py" "${INTREE}"
echo ">>> restore_ane_hook_intree: OK" >&2
