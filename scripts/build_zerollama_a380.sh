#!/usr/bin/env bash
# Build zerollama + patched vendor llama-server (Vulkan) on Intel Arc A380.
#
#   ./scripts/build_zerollama_a380.sh           # go + vendor llama (full stack)
#   ./scripts/build_zerollama_a380.sh --go-only # zerollama binary only
#   ./scripts/build_zerollama_a380.sh --llama-only
#
# After build: sudo ./scripts/install_a380_llama_server.sh && sudo systemctl restart zerollama
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DO_GO=1
DO_LLAMA=1
for arg in "$@"; do
  case "${arg}" in
    --go-only) DO_LLAMA=0 ;;
    --llama-only) DO_GO=0 ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: ${arg}" >&2; exit 1 ;;
  esac
done

FETCH_HEAD="$(grep '^FETCH_HEAD=' Makefile.sync | cut -d= -f2)"
VENDOR="${ROOT}/vendor/llama-cpp-${FETCH_HEAD}"

if [[ "${DO_LLAMA}" -eq 1 ]]; then
  if [[ ! -d "${VENDOR}/.git" ]]; then
    echo ">>> vendor clone + Ollama patches"
    "${ROOT}/scripts/rebase_vendor_unified.sh" --apply
  elif ! git -C "${VENDOR}" rev-parse HEAD >/dev/null 2>&1; then
    "${ROOT}/scripts/rebase_vendor_unified.sh" --apply
  else
  PATCH_COUNT="$(git -C "${VENDOR}" rev-list --count "$(grep '^FETCH_REF=' Makefile.sync | cut -d= -f2)..HEAD" 2>/dev/null || echo 0)"
    if [[ "${PATCH_COUNT}" -eq 0 ]]; then
      echo ">>> applying Ollama patches to vendor"
      "${ROOT}/scripts/rebase_vendor_unified.sh" --apply
    fi
  fi

  echo ">>> build vendor llama-server (Vulkan)"
  GGML_VULKAN=1 GGML_CUDA=OFF LLAMA_BUILD_WEBUI=OFF "${ROOT}/scripts/build_llama_server.sh"

  # shellcheck source=scripts/a380_llama_vendor.sh
  source "${ROOT}/scripts/a380_llama_vendor.sh"
  a380_verify_fork_llama_server "${VENDOR}/build/bin/llama-server"
fi

if [[ "${DO_GO}" -eq 1 ]]; then
  if ! command -v go >/dev/null 2>&1; then
    echo "install golang-go first: apt install golang-go gcc g++" >&2
    exit 1
  fi

  if [[ ! -f llama/llama.cpp/vendor/cpp-httplib/httplib.h ]]; then
    if [[ -d /root/llama.cpp/vendor/cpp-httplib ]]; then
      echo ">>> rsync cpp-httplib from sibling llama.cpp"
      rsync -a /root/llama.cpp/vendor/cpp-httplib/ llama/llama.cpp/vendor/cpp-httplib/
    else
      echo "missing cpp-httplib — clone ../llama.cpp or vendor manually" >&2
      exit 1
    fi
  fi

  export CGO_ENABLED=1
  export CGO_CFLAGS_ALLOW='-mfma|-mavx2|-O3'
  VERSION="$(grep 'var Version string' version/version.go | sed -n 's/.*"\(.*\)".*/\1/p')"
  go build -mod=mod -ldflags="-X=github.com/ollama/ollama/version.Version=${VERSION}" -o zerollama .
  echo ">>> built ${ROOT}/zerollama"
fi

echo ">>> next: sudo cp zerollama /usr/bin/zerollama"
echo ">>>       sudo ./scripts/install_a380_llama_server.sh"
echo ">>>       sudo cp scripts/zerollama-a380.service /etc/systemd/system/zerollama.service"
echo ">>>       sudo systemctl daemon-reload && sudo systemctl restart zerollama"
