#!/usr/bin/env bash
# macOS CGO build environment for zerollama (source only).
#
# Pins Xcode clang/SDK and python3-embed (Xcode Python3.framework 3.9 by default;
# Homebrew python@3.12 optional). No Homebrew required for a normal build.
#
#   source ./scripts/mac_cgo_env.sh && mac_cgo_env
#   eval "$(./scripts/mac_cgo_env.sh --export)"
#   ./scripts/build_zerollama_mac.sh
set -euo pipefail

_MAC_CGO_XCODE_PY_PC="/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/Current/lib/pkgconfig"
_MAC_CGO_XCODE_FW="/Applications/Xcode.app/Contents/Developer/Library/Frameworks"

mac_cgo_env_python_rpath() {
  local prefix
  prefix="$(pkg-config --variable=prefix python3-embed 2>/dev/null || true)"
  if [[ "$prefix" != *Python3.framework* ]]; then
    return 0
  fi
  local rpath_flag="-Wl,-rpath,${_MAC_CGO_XCODE_FW}"
  if [[ "${CGO_LDFLAGS:-}" != *"${_MAC_CGO_XCODE_FW}"* ]]; then
    export CGO_LDFLAGS="${CGO_LDFLAGS:+$CGO_LDFLAGS }${rpath_flag}"
  fi
}

mac_cgo_env() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "mac_cgo_env: darwin only" >&2
    return 1
  fi
  if ! xcode-select -p >/dev/null 2>&1; then
    echo "mac_cgo_env: install Xcode Command Line Tools: xcode-select --install" >&2
    return 1
  fi

  export CGO_ENABLED="${CGO_ENABLED:-1}"
  export SDKROOT="${SDKROOT:-$(xcrun --show-sdk-path 2>/dev/null)}"
  export CC="${CC:-$(xcrun --find clang 2>/dev/null)}"
  # Some dev shells export CXX=.../clang (elan/llvm); C++ shared libs need clang++ linkage.
  local _cxx_toolchain
  _cxx_toolchain="$(xcrun --find clang++ 2>/dev/null || true)"
  if [[ -z "${CXX:-}" || "${CXX}" == "${CC}" || "${CXX##*/}" == "clang" ]]; then
    export CXX="${_cxx_toolchain}"
  fi

  if [[ -f "${_MAC_CGO_XCODE_PY_PC}/python3-embed.pc" ]]; then
    export PKG_CONFIG_PATH="${_MAC_CGO_XCODE_PY_PC}${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
  fi
  if command -v brew >/dev/null 2>&1; then
    local py prefix
    for py in python@3.12 python@3.11 python3; do
      prefix="$(brew --prefix "$py" 2>/dev/null || true)"
      [[ -n "$prefix" && -f "${prefix}/lib/pkgconfig/python3-embed.pc" ]] || continue
      export PKG_CONFIG_PATH="${prefix}/lib/pkgconfig:${PKG_CONFIG_PATH}"
      break
    done
  fi

  if ! command -v pkg-config >/dev/null 2>&1; then
    echo "mac_cgo_env: pkg-config not found (install Xcode CLI tools)" >&2
    return 1
  fi
  if ! pkg-config --exists python3-embed 2>/dev/null; then
    echo "mac_cgo_env: python3-embed not found — xcode-select --install" >&2
    return 1
  fi
  mac_cgo_env_python_rpath
  return 0
}

mac_cgo_env_warn_path() {
  [[ "$(uname -s)" == "Darwin" ]] || return 0
  local xclang first
  xclang="$(xcrun --find clang 2>/dev/null || true)"
  first="$(command -v clang 2>/dev/null || true)"
  if [[ -n "$xclang" && -n "$first" && "$first" != "$xclang" ]]; then
    echo "warn: PATH clang is ${first} (not Xcode). Use ./scripts/build_zerollama_mac.sh or: eval \"\$(./scripts/mac_cgo_env.sh --export)\"" >&2
  fi
}

mac_cgo_env_print() {
  mac_cgo_env || return 1
  printf 'export CGO_ENABLED=%q\n' "${CGO_ENABLED}"
  printf 'export SDKROOT=%q\n' "${SDKROOT}"
  printf 'export CC=%q\n' "${CC}"
  printf 'export CXX=%q\n' "${CXX}"
  printf 'export PKG_CONFIG_PATH=%q\n' "${PKG_CONFIG_PATH}"
  if [[ -n "${CGO_LDFLAGS:-}" ]]; then
    printf 'export CGO_LDFLAGS=%q\n' "${CGO_LDFLAGS}"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  case "${1:-}" in
    --export)
      mac_cgo_env_print
      ;;
    --check)
      mac_cgo_env_warn_path
      mac_cgo_env
      echo "ok CC=${CC}"
      echo "ok SDKROOT=${SDKROOT}"
      echo "ok python3-embed=$(pkg-config --modversion python3-embed)"
      if [[ -n "${CGO_LDFLAGS:-}" ]]; then
        echo "ok CGO_LDFLAGS=${CGO_LDFLAGS}"
      fi
      ;;
    *)
      echo "usage: $0 --export | --check" >&2
      exit 1
      ;;
  esac
fi
