#!/usr/bin/env bash
# Stage ANE draft hook + B8 cross/IOSurface fixes into a llama.cpp tree (vendor or sibling).
# Why: sync_ane_hook_to_llama_cpp.sh skips vendor/ (patch 0018 owns those files).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CANON="${ROOT}/tools/ane-patches/canonical/common"
LLAMA_CPP_ROOT="${1:-}"

if [[ -z "${LLAMA_CPP_ROOT}" ]]; then
  FETCH_HEAD="$(grep '^FETCH_HEAD=' "${ROOT}/Makefile.sync" 2>/dev/null | cut -d= -f2 || echo c84b3020)"
  LLAMA_CPP_ROOT="${ROOT}/vendor/llama-cpp-${FETCH_HEAD}"
fi

if [[ ! -f "${LLAMA_CPP_ROOT}/CMakeLists.txt" && ! -f "${LLAMA_CPP_ROOT}/common/CMakeLists.txt" ]]; then
  echo "stage_ane_hook: missing llama.cpp tree at ${LLAMA_CPP_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${CANON}" ]]; then
  echo "stage_ane_hook: missing ${CANON}" >&2
  exit 1
fi

echo "== stage_ane_hook: ${LLAMA_CPP_ROOT} =="

install -d "${LLAMA_CPP_ROOT}/common"
if [[ "${STAGE_ANE_HOOK_STUB:-0}" == "1" ]]; then
  echo "  STAGE_ANE_HOOK_STUB=1: keep existing ane_draft_hook.* (no canonical overwrite)"
  for f in ane_draft_session.h ane_draft_session.mm ane_draft_session_stub.cpp ane_iosurface_map.h; do
    if [[ -f "${CANON}/${f}" ]]; then
      install -m 644 "${CANON}/${f}" "${LLAMA_CPP_ROOT}/common/${f}"
    fi
  done
else
  for f in ane_draft_hook.h ane_draft_hook.cpp ane_draft_session.h ane_draft_session.mm \
           ane_draft_session_stub.cpp ane_iosurface_map.h; do
    install -m 644 "${CANON}/${f}" "${LLAMA_CPP_ROOT}/common/${f}"
  done
fi

python3 "${ROOT}/tools/ane-patches/apply_speculative_ane_hook.py" "${LLAMA_CPP_ROOT}"
python3 "${ROOT}/tools/ane-patches/apply_iosurface_sibling.py" "${LLAMA_CPP_ROOT}"
python3 "${ROOT}/tools/ane-patches/verify_session_stub.py" \
  "${LLAMA_CPP_ROOT}/common/ane_draft_session.h" \
  "${LLAMA_CPP_ROOT}/common/ane_draft_session_stub.cpp"

# Optional: sync in-tree dflash/graph/context onto the target tree.
# Default OFF for patched vendor builds — in-tree dflash headers lag vendor
# graph APIs (embd_h / attn_k_dsa) and clobbering them breaks llama-server.
STAGE_ANE_SYNC_GRAPH="${STAGE_ANE_SYNC_GRAPH:-0}"
if [[ "${STAGE_ANE_SYNC_GRAPH}" == "1" ]]; then
  EXT_SRC="${ROOT}/llama/llama.cpp/src/llama-ext.h"
  CTX_H="${ROOT}/llama/llama.cpp/src/llama-context.h"
  CTX_CPP="${ROOT}/llama/llama.cpp/src/llama-context.cpp"
  MODEL_CPP="${ROOT}/llama/llama.cpp/src/llama-model.cpp"
  DFLASH_EXPORT_H="${ROOT}/llama/llama.cpp/src/llama-dflash-export.h"
  CPARAMS_H="${ROOT}/llama/llama.cpp/src/llama-cparams.h"
  GRAPH_H="${ROOT}/llama/llama.cpp/src/llama-graph.h"
  QWEN35_CPP="${ROOT}/llama/llama.cpp/src/models/qwen35.cpp"
  QWEN35MOE_CPP="${ROOT}/llama/llama.cpp/src/models/qwen35moe.cpp"
  DFLASH_DRAFT_CPP="${ROOT}/llama/llama.cpp/src/models/dflash-draft.cpp"
  SPEC_CPP="${ROOT}/llama/llama.cpp/common/speculative.cpp"
  LLAMA_H="${ROOT}/llama/llama.cpp/include/llama.h"
  for src in "${EXT_SRC}" "${CTX_H}" "${CPARAMS_H}" "${GRAPH_H}" "${DFLASH_EXPORT_H}"; do
    if [[ -f "${src}" && -d "${LLAMA_CPP_ROOT}/src" ]]; then
      install -m 644 "${src}" "${LLAMA_CPP_ROOT}/src/$(basename "${src}")"
    fi
  done
  if [[ -f "${QWEN35_CPP}" && -d "${LLAMA_CPP_ROOT}/src/models" ]]; then
    install -m 644 "${QWEN35_CPP}" "${LLAMA_CPP_ROOT}/src/models/qwen35.cpp"
  fi
  if [[ -f "${QWEN35MOE_CPP}" && -d "${LLAMA_CPP_ROOT}/src/models" ]]; then
    install -m 644 "${QWEN35MOE_CPP}" "${LLAMA_CPP_ROOT}/src/models/qwen35moe.cpp"
  fi
  if [[ -f "${DFLASH_DRAFT_CPP}" && -d "${LLAMA_CPP_ROOT}/src/models" ]] && rg -q 't_dflash_attn_out' "${DFLASH_DRAFT_CPP}" 2>/dev/null; then
    install -m 644 "${DFLASH_DRAFT_CPP}" "${LLAMA_CPP_ROOT}/src/models/dflash-draft.cpp"
  fi
  if [[ -f "${SPEC_CPP}" && -d "${LLAMA_CPP_ROOT}/common" ]] && rg -q 'llama_set_dflash_target_export' "${SPEC_CPP}" 2>/dev/null; then
    install -m 644 "${SPEC_CPP}" "${LLAMA_CPP_ROOT}/common/speculative.cpp"
  fi
  if [[ -f "${CTX_CPP}" && -d "${LLAMA_CPP_ROOT}/src" ]] && rg -q 'cross_n_enc' "${CTX_CPP}" 2>/dev/null; then
    install -m 644 "${CTX_CPP}" "${LLAMA_CPP_ROOT}/src/llama-context.cpp"
  fi
  # llama-dflash API lives in llama-context.cpp; drop stale standalone translation unit.
  rm -f "${LLAMA_CPP_ROOT}/src/llama-dflash.cpp"
  if [[ -f "${LLAMA_CPP_ROOT}/src/CMakeLists.txt" ]] && rg -q 'llama-dflash.cpp' "${LLAMA_CPP_ROOT}/src/CMakeLists.txt" 2>/dev/null; then
    python3 - <<'PY' "${LLAMA_CPP_ROOT}/src/CMakeLists.txt"
import pathlib, sys
p = pathlib.Path(sys.argv[1])
orig = p.read_text()
text = orig.replace("            llama-dflash.cpp\n", "")
if text != orig:
    p.write_text(text)
    print("  removed stale llama-dflash.cpp from CMakeLists.txt")
PY
  fi
  if [[ -f "${MODEL_CPP}" && -d "${LLAMA_CPP_ROOT}/src" ]] && rg -q 'llama_model_dflash_n_target_features' "${MODEL_CPP}" 2>/dev/null; then
    install -m 644 "${MODEL_CPP}" "${LLAMA_CPP_ROOT}/src/llama-model.cpp"
  fi
  if [[ -f "${LLAMA_H}" && -d "${LLAMA_CPP_ROOT}/include" ]] && rg -q 'llama_get_dflash_target_features' "${LLAMA_H}" 2>/dev/null; then
    install -m 644 "${LLAMA_H}" "${LLAMA_CPP_ROOT}/include/llama.h"
  fi

  # llama-context.cpp B8 cross helpers: copy from vendor when submodule cpp lags.
  VENDOR_CTX="${ROOT}/vendor/llama-cpp-$(grep '^FETCH_HEAD=' "${ROOT}/Makefile.sync" 2>/dev/null | cut -d= -f2 || echo c84b3020)/src/llama-context.cpp"
  SUB_CTX="${ROOT}/llama/llama.cpp/src/llama-context.cpp"
  if [[ -f "${VENDOR_CTX}" && -f "${SUB_CTX}" ]] && rg -q 'cross_upsert_row' "${VENDOR_CTX}" 2>/dev/null; then
    if ! rg -q 'cross_upsert_row' "${SUB_CTX}" 2>/dev/null; then
      install -m 644 "${VENDOR_CTX}" "${LLAMA_CPP_ROOT}/src/llama-context.cpp"
    elif [[ "${LLAMA_CPP_ROOT}/src/llama-context.cpp" -ef "${SUB_CTX}" ]]; then
      : # submodule already has cross helpers
    elif [[ ! -f "${LLAMA_CPP_ROOT}/src/llama-context.cpp" ]] || ! rg -q 'cross_upsert_row' "${LLAMA_CPP_ROOT}/src/llama-context.cpp" 2>/dev/null; then
      install -m 644 "${VENDOR_CTX}" "${LLAMA_CPP_ROOT}/src/llama-context.cpp"
    fi
  fi
else
  echo "  skip in-tree graph/context sync (STAGE_ANE_SYNC_GRAPH=0; vendor APIs kept)"
fi

echo "== stage_ane_hook: done =="
