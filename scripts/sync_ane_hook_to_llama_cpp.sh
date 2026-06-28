#!/usr/bin/env bash
# Copy zerollama ANE draft hook + ggml IOSurface API into LLAMA_CPP_ROOT (unified sibling).
# Why: in-process ANE session lives in llama-common; sibling ../llama.cpp is the runtime build
# target for ./scripts/build_llama_server.sh (vendor tree may lag hook patches).
# Canonical sources: tools/ane-patches/canonical/common/ (survives vendor rsync).
# Idempotent — safe to re-run; apply_speculative_ane_hook.py fails loud on anchor drift.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FETCH_HEAD="$(grep '^FETCH_HEAD=' "${ROOT}/Makefile.sync" 2>/dev/null | cut -d= -f2 || echo c84b3020)"
LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-${ROOT}/../llama.cpp}"
CANON="${ROOT}/tools/ane-patches/canonical/common"

if [[ ! -f "${LLAMA_CPP_ROOT}/CMakeLists.txt" ]]; then
  echo "sync_ane_hook: missing ${LLAMA_CPP_ROOT}/CMakeLists.txt" >&2
  exit 1
fi
if [[ ! -d "${CANON}" ]]; then
  echo "sync_ane_hook: missing ${CANON}; run from zerollama checkout" >&2
  exit 1
fi

T_COMMON="${LLAMA_CPP_ROOT}/common"

echo "== sync_ane_hook: ${LLAMA_CPP_ROOT} (pin ref ${FETCH_HEAD}) =="

install -d "${T_COMMON}"
for f in ane_draft_hook.h ane_draft_hook.cpp ane_draft_session.h ane_draft_session.mm \
         ane_draft_session_stub.cpp ane_iosurface_map.h; do
  install -m 644 "${CANON}/${f}" "${T_COMMON}/${f}"
done

python3 "${ROOT}/tools/ane-patches/apply_speculative_ane_hook.py" "${LLAMA_CPP_ROOT}"
python3 "${ROOT}/tools/ane-patches/apply_iosurface_sibling.py" "${LLAMA_CPP_ROOT}"

echo "== sync_ane_hook: done — rebuild: ./scripts/build_llama_server.sh =="
