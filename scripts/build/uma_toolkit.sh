#!/usr/bin/env bash
# UMA toolkit helpers (source only). Sibling checkout optional.
#
# Default layout (from zerollama repo root):
#   ../bmtl/hardware_lab/lanes/m4/uma_toolkit
#
# Override: BMTL_UMA_TOOLKIT=/path/to/uma_toolkit
#
#   source ./scripts/build/uma_toolkit.sh
#   uma_toolkit_root
#   uma_toolkit_present && echo ready

_UMA_TOOLKIT_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_UMA_TOOLKIT_REPO_ROOT="$(cd "${_UMA_TOOLKIT_SH_DIR}/../.." && pwd)"

uma_toolkit_root() {
  echo "${BMTL_UMA_TOOLKIT:-${_UMA_TOOLKIT_REPO_ROOT}/../bmtl/hardware_lab/lanes/m4/uma_toolkit}"
}

uma_toolkit_present() {
  local tk
  tk="$(uma_toolkit_root)"
  # Current bmtl layout: src/uma_client.c; legacy: uma_client.c at toolkit root.
  [[ -f "${tk}/src/uma_client.c" || -f "${tk}/uma_client.c" ]] \
    && [[ -d "${tk}/include" ]]
}
