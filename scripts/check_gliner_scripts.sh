#!/usr/bin/env bash
# Syntax-check GLiNER track scripts + pin/patch presence (CI-friendly, no ORT/GPU).
# WHY: keep GL0–GL5a operator surfaces from rotting after RelEx stays Parked.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fail=0

need_file() {
  local p="$1"
  if [[ ! -f "${ROOT}/${p}" ]]; then
    echo "missing: ${p}" >&2
    fail=1
  fi
}

need_exec() {
  local p="$1"
  need_file "${p}"
  if [[ -f "${ROOT}/${p}" && ! -x "${ROOT}/${p}" ]]; then
    echo "not executable: ${p}" >&2
    fail=1
  fi
}

need_file "GLINER_CPP_COMMIT"
need_file "gliner/server/CMakeLists.txt"
need_file "gliner/server/gliner-server.cpp"
need_file "gliner/patches/0001-token-logits-channel-first-transpose.patch"
need_file "gliner/patches/README.md"
need_file "docs/gliner-cpp.md"
need_file "docs/gliner-cpp-findings.md"
need_file "agentskills/entity-extract/SKILL.md"

SCRIPTS=(
  scripts/vendor/ensure_gliner_cpp.sh
  scripts/vendor/ensure_gliner_model.sh
  scripts/vendor/ensure_gliner_server_deps.sh
  scripts/vendor/ensure_onnxruntime.sh
  scripts/build/build_gliner_server.sh
  scripts/serve/serve_gliner_lab.sh
  scripts/smoke/gliner_cpp_smoke.sh
  scripts/smoke/gliner_server_smoke.sh
  scripts/smoke/gliner_go_e2e.sh
  scripts/smoke/gliner_cuda_smoke.sh
  scripts/smoke/gliner_token_smoke.sh
  scripts/smoke/gliner_token_go_e2e.sh
  scripts/check_gliner_scripts.sh
)

for s in "${SCRIPTS[@]}"; do
  need_exec "${s}"
  if [[ -f "${ROOT}/${s}" ]]; then
    if ! bash -n "${ROOT}/${s}"; then
      echo "bash -n failed: ${s}" >&2
      fail=1
    fi
  fi
done

# Pin must be a non-empty hex sha
PIN="$(tr -d '[:space:]' < "${ROOT}/GLINER_CPP_COMMIT" 2>/dev/null || true)"
if [[ ! "${PIN}" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
  echo "GLINER_CPP_COMMIT invalid: '${PIN}'" >&2
  fail=1
fi

if [[ "${fail}" -ne 0 ]]; then
  echo "GLINER_SCRIPTS_CHECK_FAIL" >&2
  exit 1
fi
echo "GLINER_SCRIPTS_CHECK_OK"
