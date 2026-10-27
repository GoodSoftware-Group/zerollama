#!/usr/bin/env bash
# Undo patch_vendor_linux_ane_hook.sh so vendor/ stays git-clean after Linux builds.
set -euo pipefail

ROOT="${1:-}"
if [[ -z "${ROOT}" ]]; then
  echo "restore_vendor_linux_ane_hook: bad root" >&2
  exit 1
fi

FILE="${ROOT}/common/ane_draft_hook.cpp"
DARWIN="${FILE}.darwin"

if [[ -f "${DARWIN}" ]]; then
  cp -f "${DARWIN}" "${FILE}"
  rm -f "${DARWIN}"
  echo "restored ${FILE} from Darwin backup" >&2
elif grep -q 'ZEROLLAMA_LINUX_ANE_DRAFT_STUB' "${FILE}" 2>/dev/null; then
  echo "warn: Linux ANE stub present but no ${DARWIN}; leave as-is" >&2
fi
