#!/usr/bin/env bash
# Production zerollama serve — install as ~/bin/serve.sh
#
# WHY this wrapper exists:
#   serve_gpu_example.sh resolves repo root as $(dirname "$0")/.. . When operators
#   copied that file to ~/bin/serve.sh, .. became $HOME — not ~/zerollama — so
#   sched_watchdog_env.sh, training_uv_venv.sh, and PYTHONPATH never loaded.
#   Serve exited before :8080 or embed failed with ModuleNotFoundError: uvicorn
#   while the screen looked idle (logs only when SERVE_LOG is set).
#
# WHAT this does:
#   Sets ZEROLLAMA_REPO (default ~/zerollama), SERVE_LOG, execs in-repo example.
#
# Install:
#   cd ~/zerollama
#   cp scripts/serve/serve_production_wrapper.sh ~/bin/serve.sh
#   chmod +x ~/bin/serve.sh
#
# Start:
#   ~/bin/serve.sh
#   tail -f /tmp/zerollama-serve.log
#
# Verify:
#   curl -s http://127.0.0.1:8080/api/version | jq .
#   curl -s http://127.0.0.1:8081/health | jq '{status, profile: .gpu_profile.id}'
#
# Doc: docs/5080-runbook.md#production-serve-binserve-sh
set -euo pipefail

export ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-${HOME}/zerollama}"
export SERVE_LOG="${SERVE_LOG:-/tmp/zerollama-serve.log}"

# Never bind inference on the PVE hypervisor (CT 1564 only).
# shellcheck disable=SC1091
source "${ZEROLLAMA_REPO}/scripts/serve/refuse_pve_host.sh"

if [[ ! -f "${ZEROLLAMA_REPO}/scripts/serve/serve_gpu_example.sh" ]]; then
  echo "~/bin/serve.sh: missing ${ZEROLLAMA_REPO}/scripts/serve/serve_gpu_example.sh" >&2
  echo "WHY: set ZEROLLAMA_REPO to your zerollama checkout (symlink ~/zerollama is fine)." >&2
  exit 1
fi

exec bash "${ZEROLLAMA_REPO}/scripts/serve/serve_gpu_example.sh"
