#!/usr/bin/env bash
# Fresh-clone bootstrap for macOS dev — safe defaults for any checkout path.
#
# Why: mac_setup used to run metal sign-off by default (needs pulled models) and
# failed when ../llama.cpp was missing. This entry point sets portable defaults.
#
# Tier 0 (default): host deps → ggml build → install to ~/.local/bin → uv venv.
# Sibling Metal libllama is best-effort (MAC_SETUP_LLAMA_OPTIONAL=1).
#
# Usage (from repo root, any path):
#   ./scripts/runtime/dev_bootstrap.sh
#   MAC_SETUP_SIGNOFF=1 ./scripts/runtime/dev_bootstrap.sh   # after zerollama pull …
#   MAC_SETUP_TRAINING=1 ./scripts/runtime/dev_bootstrap.sh
#   MAC_SETUP_LLAMA_OPTIONAL=0 ./scripts/runtime/dev_bootstrap.sh  # require libllama
#
# Same env as mac_setup.sh; see scripts/runtime/mac_setup.sh for full list.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MAC_SETUP_SIGNOFF="${MAC_SETUP_SIGNOFF:-0}"
export MAC_SETUP_LLAMA_CLONE="${MAC_SETUP_LLAMA_CLONE:-1}"
export MAC_SETUP_HOST_DEPS="${MAC_SETUP_HOST_DEPS:-1}"
export MAC_SETUP_INSTALL="${MAC_SETUP_INSTALL:-1}"
export MAC_SETUP_SMOKE="${MAC_SETUP_SMOKE:-1}"
# Ggml-first: clone/build llama.cpp when possible, but do not fail tier 0 without it.
export MAC_SETUP_LLAMA_OPTIONAL="${MAC_SETUP_LLAMA_OPTIONAL:-1}"
# Do not rewrite llama/llama.cpp/common during tier-0 (ANE restore dirties the tree).
export BUILD_ANE_RESTORE="${BUILD_ANE_RESTORE:-0}"
exec "${ROOT}/scripts/runtime/mac_setup.sh"
