#!/usr/bin/env bash
# Deploy Python runtime from repo checkout to production sidecar path.
#
# Usage:
#   sudo ./scripts/deploy_runtime.sh
#   INSTALL_ROOT=/opt/zerollama ./scripts/deploy_runtime.sh
#
# Copies runtime package + configs; preserves .venv on target.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-/opt/zerollama}"
DEST="${INSTALL_ROOT}/runtime"
SERVICE="${ZEROLLAMA_RUNTIME_SERVICE:-zerollama-runtime}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "deploy_runtime: re-run with sudo" >&2
  exit 1
fi

if [[ ! -d "${ROOT}/runtime/runtime" ]]; then
  echo "deploy_runtime: missing ${ROOT}/runtime/runtime" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="${DEST}.bak-${STAMP}"
if [[ -d "${DEST}" ]]; then
  echo ">>> backup ${DEST} → ${BACKUP}"
  cp -a "${DEST}" "${BACKUP}"
fi

mkdir -p "${DEST}"
echo ">>> sync runtime sources → ${DEST}"
rsync -a \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  "${ROOT}/runtime/" "${DEST}/"

if [[ ! -x "${DEST}/.venv/bin/python" ]]; then
  echo "warn: ${DEST}/.venv missing — run runtime_uv_venv.sh on target host" >&2
fi

echo ">>> verify llama_fork CUDA probe present"
if ! grep -q 'cuda_fork_backend_capable' "${DEST}/runtime/llama_fork.py"; then
  echo "error: deployed llama_fork.py missing cuda_fork_backend_capable" >&2
  exit 1
fi

if systemctl is-active --quiet "${SERVICE}" 2>/dev/null; then
  echo ">>> restart ${SERVICE}"
  systemctl restart "${SERVICE}"
  sleep 2
  if curl -sf -m 15 "http://127.0.0.1:8081/health" >/dev/null 2>&1; then
    echo "OK: runtime /health after restart"
  else
    echo "warn: /health not up yet — check: journalctl -u ${SERVICE} -n 40" >&2
  fi
else
  echo ">>> ${SERVICE} not active (skipped restart)"
fi

echo ">>> deploy OK: ${DEST}"
echo ">>> backup: ${BACKUP}"
