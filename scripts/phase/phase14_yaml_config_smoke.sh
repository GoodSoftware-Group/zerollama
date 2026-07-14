#!/usr/bin/env bash
# Phase 14 smoke when llama_backend is set in runtime YAML (not env).
#
# Prerequisite: set llama_backend in runtime YAML (typically inprocess via
# phase14_enable_yaml_inprocess.sh), restart serve WITHOUT
# ZEROLLAMA_RUNTIME_LLAMA_BACKEND, then:
#
#   source ./scripts/phase/phase14_serve_env.sh
#   export LLAMA_MODEL=/path/to/small.q8_0.gguf
#   ./zerollama serve
#
#   export LLAMA_MODEL=/path/to/same.gguf
#   export RUN_E2E_PROXY_MODEL=<pulled-local-tag>   # optional render-chat
#   ./scripts/phase/phase14_yaml_config_smoke.sh
#
# Asserts /health llama_backend_source=config; RUN_E2E_* backend flags are
# inferred from /health (inprocess or llama-cpp-python, not subprocess).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RUN_E2E_LLAMA_BACKEND_SOURCE=config
exec "${ROOT}/scripts/phase/phase14_backend_smoke.sh"
