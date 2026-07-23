#!/usr/bin/env bash
# M20 step 7 wrapper — briefly TERM machine-wide uma_daemon.
#
# Disrupts all UMA clients on this Mac. Does not touch production
# zerollama :11434 / :8081 (lab uses :11435).
#
# Usage:
#   ./scripts/phase/m20_uma_restart_soak.sh
#
# Requires explicit operator intent (TERMs the machine broker).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export M20_SKIP_BUILD="${M20_SKIP_BUILD:-1}"
export RUN_E2E_UMA_RESTART=1
exec "${ROOT}/scripts/phase/m20_uma_signoff.sh"
