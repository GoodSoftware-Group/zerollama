#!/usr/bin/env bash
# Apply zerollama IOSurface export patch to maderix/ane libane_bridge (idempotent).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANE_REPO="${ANE_REPO:-${HOME}/Sites/inference/ane}"
PATCH="${ROOT}/tools/ane-patches/0001-bridge-iosurface-export.patch"
BRIDGE="${ANE_REPO}/bridge"

if [[ ! -f "${BRIDGE}/ane_bridge.h" ]]; then
  echo "ane_bridge_patch: missing ${BRIDGE}" >&2
  exit 1
fi

if grep -q 'ane_bridge_input_surface_id' "${BRIDGE}/ane_bridge.h"; then
  echo "ane_bridge_patch: already applied"
  exit 0
fi

echo "ane_bridge_patch: applying IOSurface export patch to ${BRIDGE}"
(cd "${BRIDGE}" && patch -p1 < "${PATCH}")
echo "ane_bridge_patch: rebuilding libane_bridge"
make -C "${BRIDGE}" clean all
