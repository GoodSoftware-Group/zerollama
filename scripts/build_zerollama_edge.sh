#!/usr/bin/env bash
# Build zerollama with Phase 16 edge compile marker (-tags edge + version.EdgeBuild).
#
# WHY v0–v2 edge build:
#   v0 marker — version.EdgeBuild + serve-time --edge defaults (runtime env)
#   v1 stub   — GgmlRunnerLinked=false; zerollama runner subprocess rejected
#   v2 CGO    — server.go (!edge) excluded; edge links llama-server path only
#
# Usage:
#   ./scripts/build_zerollama_edge.sh
#   ./scripts/build_zerollama_edge.sh /path/to/zerollama-edge
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${ROOT}/zerollama-edge}"

if [[ -z "${VERSION:-}" ]]; then
  if git -C "${ROOT}" describe --tags --first-parent --abbrev=7 --long --dirty --always &>/dev/null; then
    VERSION="$(git -C "${ROOT}" describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e 's/^v//')"
  else
    VERSION="0.19.0"
  fi
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  # shellcheck source=scripts/mac_cgo_env.sh
  source "${ROOT}/scripts/mac_cgo_env.sh"
  mac_cgo_env_warn_path
  mac_cgo_env
fi

LDFLAGS=(
  "-X" "github.com/ollama/ollama/version.Version=${VERSION}"
  "-X" "github.com/ollama/ollama/version.EdgeBuild=true"
)

echo ">>> building edge-marked zerollama -> ${OUT}" >&2
echo ">>> tags: edge  ldflags: version=${VERSION} EdgeBuild=true" >&2
(
  cd "${ROOT}"
  go build -tags edge -ldflags "${LDFLAGS[*]}" -o "${OUT}" .
)

echo ">>> built ${OUT} (run: ${OUT} serve --edge)" >&2
