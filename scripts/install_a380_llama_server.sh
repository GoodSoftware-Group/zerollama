#!/usr/bin/env bash
# Install vendor Vulkan llama-server + libs for A380 production.
#
#   sudo ./scripts/install_a380_llama_server.sh
#
# Copies vendor/llama-cpp-*/build/bin/* → /usr/lib/ollama-zerollama (default).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/a380_llama_vendor.sh
source "${ROOT}/scripts/a380_llama_vendor.sh"

BUILD_DIR="$(a380_llama_vendor_build_dir)"
INSTALL_DIR="$(a380_llama_install_dir)"

if [[ ! -x "${BUILD_DIR}/llama-server" ]]; then
  echo "install_a380: missing ${BUILD_DIR}/llama-server — run:" >&2
  echo "  GGML_VULKAN=1 GGML_CUDA=OFF ./scripts/build_llama_server.sh" >&2
  exit 1
fi

if [[ "$(id -u)" -ne 0 ]]; then
  echo "install_a380: re-run with sudo" >&2
  exit 1
fi

mkdir -p "${INSTALL_DIR}"
echo ">>> install ${BUILD_DIR} → ${INSTALL_DIR}"
rsync -a --delete "${BUILD_DIR}/" "${INSTALL_DIR}/"
chmod -R a+rX "${INSTALL_DIR}"

export LLAMA_SERVER_BIN="${INSTALL_DIR}/llama-server"
a380_verify_fork_llama_server "${LLAMA_SERVER_BIN}"

mkdir -p /etc/zerollama
cat > /etc/zerollama/a380-llama.env <<EOF
# Zerollama patched llama-server (eliza vendor @ $(a380_llama_vendor_pin))
LLAMA_SERVER_BIN=${INSTALL_DIR}/llama-server
LLAMA_CPP_LIB=${INSTALL_DIR}/libllama.so
LD_LIBRARY_PATH=${INSTALL_DIR}:/usr/lib/ollama/vulkan
OLLAMA_LIBRARY_PATH=${INSTALL_DIR}:/usr/lib/ollama/vulkan
EOF
chmod 644 /etc/zerollama/a380-llama.env

echo ">>> wrote /etc/zerollama/a380-llama.env"
echo ">>> install OK: ${INSTALL_DIR}/llama-server"
