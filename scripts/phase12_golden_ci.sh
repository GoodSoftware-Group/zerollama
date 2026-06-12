#!/usr/bin/env bash
# Phase 12 CI goldens (no GPU): render/parse parity before GPU smokes.
#
#   ./scripts/phase12_golden_ci.sh          # all (GPU preflight / local)
#   ./scripts/phase12_golden_ci.sh go       # Go -run Golden only
#   ./scripts/phase12_golden_ci.sh py       # Python tools meta (needs runtime installed)
#
# CI: go-server job runs full `go test ./server/...` (includes Golden); runtime-pytest
# runs full pytest (includes test_go_render_chat.py). This script is not duplicated there.
#
# Optional first step on a GPU host:
#   RUN_E2E_PREFLIGHT=1 ./scripts/gpu_smoke_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PART="${1:-all}"
case "$PART" in
  all|go|py) ;;
  -h|--help)
    sed -n '2,12p' "$0"
    exit 0
    ;;
  *)
    echo "usage: $0 [all|go|py]" >&2
    exit 1
    ;;
esac

_resolve_go() {
  if [[ -n "${GO:-}" ]]; then
    return 0
  fi
  local _ct_go="${ROOT}/../../usr/local/go/bin/go"
  if command -v go >/dev/null 2>&1; then
    GO=go
  elif [[ -x /usr/local/go/bin/go ]]; then
    GO=/usr/local/go/bin/go
  elif [[ -x "${_ct_go}" ]]; then
    GO="${_ct_go}"
  else
    echo "go not found; set GO= to a built toolchain" >&2
    exit 1
  fi
}

_run_check_gpu_scripts() {
  echo "== check GPU scripts =="
  "${ROOT}/scripts/check_gpu_scripts.sh"
}

_run_go_golden() {
  _resolve_go
  echo "== Phase 12 golden (Go) =="
  if [[ "$(uname -s)" == "Darwin" ]]; then
    # shellcheck source=scripts/mac_cgo_env.sh
    source "${ROOT}/scripts/mac_cgo_env.sh"
    mac_cgo_env_warn_path
    mac_cgo_env
  fi
  CGO_ENABLED=1 OLLAMA_NO_CLOUD=true "${GO}" test -count=1 ./server/... -run Golden
}

_run_py_golden() {
  echo "== Phase 12 tools meta (Python) =="
  # shellcheck source=scripts/runtime_uv_venv.sh
  source "${ROOT}/scripts/runtime_uv_venv.sh"
  local py=""
  if command -v uv >/dev/null 2>&1; then
    runtime_uv_venv
    py="${RUNTIME_UV_PYTHON}"
  elif [[ -x "${ROOT}/runtime/.venv/bin/python" ]]; then
    py="${ROOT}/runtime/.venv/bin/python"
  else
    py="python3"
  fi
  if ! "$py" -c "import runtime" 2>/dev/null; then
    echo "runtime package not importable; run: ./scripts/runtime_uv_venv.sh" >&2
    exit 1
  fi
  cd "${ROOT}/runtime"
  PYTHONPATH=. "$py" -m pytest tests/test_go_render_chat.py -q
}

if [[ "$PART" == "all" ]]; then
  _run_check_gpu_scripts
  _run_go_golden
  _run_py_golden
elif [[ "$PART" == "go" ]]; then
  _run_check_gpu_scripts
  _run_go_golden
else
  _run_py_golden
fi

echo "PASS: phase12_golden_ci ($PART)"
