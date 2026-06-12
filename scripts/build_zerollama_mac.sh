#!/usr/bin/env bash
# Build zerollama on macOS with CGO (Metal ggml + embedded libpython for training).
#
# Usage:
#   ./scripts/build_zerollama_mac.sh
#   ./scripts/build_zerollama_mac.sh /path/to/output/binary
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${ROOT}/zerollama}"

# shellcheck source=scripts/mac_cgo_env.sh
source "${ROOT}/scripts/mac_cgo_env.sh"
mac_cgo_env_warn_path
mac_cgo_env

echo ">>> CC=${CC}" >&2
echo ">>> SDKROOT=${SDKROOT}" >&2
echo ">>> python3-embed: $(pkg-config --modversion python3-embed)" >&2

cd "${ROOT}"
GOFLAGS=-mod=mod go build -o "${OUT}" .
echo ">>> wrote ${OUT}" >&2
