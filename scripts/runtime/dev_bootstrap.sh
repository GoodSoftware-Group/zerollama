#!/usr/bin/env bash
# Fresh-clone bootstrap for macOS dev — safe defaults for any checkout path.
#
# Why: mac_setup used to run metal sign-off by default (needs pulled models) and
# failed when ../llama.cpp was missing. This entry point sets portable defaults.
#
# Usage (from repo root, any path):
#   ./scripts/runtime/dev_bootstrap.sh
#   MAC_SETUP_SIGNOFF=1 ./scripts/runtime/dev_bootstrap.sh   # after zerollama pull …
#   MAC_SETUP_TRAINING=1 ./scripts/runtime/dev_bootstrap.sh
#
# Same env as mac_setup.sh; see scripts/runtime/mac_setup.sh for full list.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MAC_SETUP_SIGNOFF="${MAC_SETUP_SIGNOFF:-0}"
export MAC_SETUP_LLAMA_CLONE="${MAC_SETUP_LLAMA_CLONE:-1}"
exec "${ROOT}/scripts/runtime/mac_setup.sh"
