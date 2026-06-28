#!/usr/bin/env bash
# Serve zerollama with llama-server backend for Eliza speculative decoding.
#
# DFlash: create tags from modelfiles/eliza-1-*-dflash.modelfile (DRAFT + spec_type dflash).
# Ngram: set ZEROLLAMA_ELIZA_NGRAM=1 or use modelfiles/eliza-1-*-ngram.modelfile.
#
# Usage:
#   ./scripts/serve_eliza_fast.sh
#   ./scripts/serve_eliza_fast.sh --port 11434
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ROOT}/scripts/serve_llama_server_backend.sh" "$@"
