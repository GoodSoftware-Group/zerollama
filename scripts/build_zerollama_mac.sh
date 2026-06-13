#!/usr/bin/env bash
# Build zerollama on macOS with CGO (Metal ggml + embedded libpython for training).
#
# Why go generate runs here: macOS Metal loads shaders from ggml-metal-embed.metal
# (generated from ggml-metal.metal). If ggml adds kernels without regenerating
# the embed, model load can succeed but first decode crashes (missing unary ops
# such as sigmoid for qwen35 SSM). See docs/qwen35-apple-silicon.md.
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
echo ">>> regenerating ggml-metal-embed.metal" >&2
GOFLAGS=-mod=mod go generate ./ml/backend/ggml/ggml/src/ggml-metal/
GOFLAGS=-mod=mod go build -o "${OUT}" .
echo ">>> wrote ${OUT}" >&2
