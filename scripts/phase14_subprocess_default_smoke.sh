#!/usr/bin/env bash
# Phase 14 smoke when llama_backend is the packaged subprocess default (no env, no YAML key).
#
# Prerequisite: serve with autoconfig (e.g. single_gpu.yaml) and WITHOUT
# ZEROLLAMA_RUNTIME_LLAMA_BACKEND and without uncommented llama_backend: in YAML.
#
#   source ./scripts/phase14_serve_env.sh
#   export LLAMA_MODEL=/path/to/small.q8_0.gguf
#   ./zerollama serve
#
#   export LLAMA_MODEL=/path/to/same.gguf
#   ./scripts/phase14_subprocess_default_smoke.sh
#
# LLAMA_SERVER_BIN on this shell is optional (Phase 14 uses the running serve's binary).
# Asserts /health llama_backend=subprocess and llama_backend_source=default.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_E2E_LLAMA_BACKEND_SOURCE=default
exec "${ROOT}/scripts/phase14_backend_smoke.sh"
