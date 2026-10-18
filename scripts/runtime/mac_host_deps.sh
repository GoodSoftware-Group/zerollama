#!/usr/bin/env bash
# Install user-local Mac host deps for zerollama (no sudo / Homebrew required).
#
# Installs into ~/.local (Go, pkgconf, uv, PATH) and ~/sdk (CMake.app).
# Homebrew is used when already present (faster), never required.
#
# Usage:
#   ./scripts/runtime/mac_host_deps.sh           # install missing deps
#   ./scripts/runtime/mac_host_deps.sh --check   # exit 1 if anything missing
#   ./scripts/runtime/mac_host_deps.sh --dry-run # list what would install where
#   source ./scripts/runtime/mac_host_deps.sh && mac_host_deps_ensure
#
# Env:
#   MAC_HOST_GO_VER=1.24.5
#   MAC_HOST_CMAKE_VER=3.31.6
#   MAC_HOST_PKGCONF_VER=2.3.0
#   MAC_HOST_PREFIX=$HOME/.local
#   MAC_HOST_SDK=$HOME/sdk
set -euo pipefail

_MAC_HOST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAC_HOST_PREFIX="${MAC_HOST_PREFIX:-${HOME}/.local}"
MAC_HOST_SDK="${MAC_HOST_SDK:-${HOME}/sdk}"
MAC_HOST_GO_VER="${MAC_HOST_GO_VER:-1.24.5}"
MAC_HOST_CMAKE_VER="${MAC_HOST_CMAKE_VER:-3.31.6}"
MAC_HOST_PKGCONF_VER="${MAC_HOST_PKGCONF_VER:-2.3.0}"
MAC_HOST_DRY_RUN="${MAC_HOST_DRY_RUN:-0}"

mac_host_deps_path_prepend() {
  export PATH="${MAC_HOST_PREFIX}/bin:${MAC_HOST_PREFIX}/go/bin:${PATH}"
  if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
}

mac_host_deps_ensure_path_profile() {
  local profile="${HOME}/.zprofile"
  local marker="# zerollama toolchain (user-local)"
  mkdir -p "${MAC_HOST_PREFIX}/bin"
  if [[ -f "${profile}" ]] && grep -qF "${marker}" "${profile}" 2>/dev/null; then
    return 0
  fi
  {
    echo ""
    echo "${marker}"
    echo "export PATH=\"${MAC_HOST_PREFIX}/bin:${MAC_HOST_PREFIX}/go/bin:\$PATH\""
  } >> "${profile}"
  echo ">>> appended PATH to ${profile}" >&2
}

_mac_host_arch() {
  case "$(uname -m)" in
    arm64|aarch64) echo "arm64" ;;
    x86_64|amd64) echo "amd64" ;;
    *) echo "$(uname -m)" ;;
  esac
}

_mac_host_go_ok() {
  command -v go >/dev/null 2>&1 || return 1
  local ver major minor
  ver="$(go env GOVERSION 2>/dev/null | sed 's/^go//')"
  [[ -n "${ver}" ]] || return 1
  major="${ver%%.*}"
  minor="${ver#*.}"
  minor="${minor%%.*}"
  # go.mod requires 1.24.1+
  if [[ "${major}" -gt 1 ]]; then
    return 0
  fi
  if [[ "${major}" -eq 1 && "${minor}" -ge 24 ]]; then
    return 0
  fi
  return 1
}

_mac_host_install_go() {
  if _mac_host_go_ok; then
    echo ">>> go ok: $(go version)" >&2
    return 0
  fi
  if command -v brew >/dev/null 2>&1; then
    echo ">>> brew install go@1.24 (or go)" >&2
    brew install go@1.24 2>/dev/null || brew install go
    mac_host_deps_path_prepend
    if _mac_host_go_ok; then
      echo ">>> go ok: $(go version)" >&2
      return 0
    fi
  fi
  local arch go_arch tarball
  arch="$(_mac_host_arch)"
  case "${arch}" in
    arm64) go_arch="arm64" ;;
    amd64) go_arch="amd64" ;;
    *)
      echo "error: unsupported arch ${arch} for Go tarball; install from https://go.dev/dl/" >&2
      return 1
      ;;
  esac
  tarball="go${MAC_HOST_GO_VER}.darwin-${go_arch}.tar.gz"
  echo ">>> installing Go ${MAC_HOST_GO_VER} → ${MAC_HOST_PREFIX}/go" >&2
  mkdir -p "${MAC_HOST_PREFIX}" /tmp
  curl -fsSL "https://go.dev/dl/${tarball}" -o "/tmp/${tarball}"
  rm -rf "${MAC_HOST_PREFIX}/go"
  tar -C "${MAC_HOST_PREFIX}" -xzf "/tmp/${tarball}"
  ln -sfn "${MAC_HOST_PREFIX}/go/bin/go" "${MAC_HOST_PREFIX}/bin/go"
  ln -sfn "${MAC_HOST_PREFIX}/go/bin/gofmt" "${MAC_HOST_PREFIX}/bin/gofmt"
  mac_host_deps_path_prepend
  _mac_host_go_ok || {
    echo "error: Go install finished but version still < 1.24" >&2
    return 1
  }
  echo ">>> go ok: $(go version)" >&2
}

_mac_host_install_cmake() {
  if command -v cmake >/dev/null 2>&1; then
    echo ">>> cmake ok: $(cmake --version | head -1)" >&2
    return 0
  fi
  if command -v brew >/dev/null 2>&1; then
    echo ">>> brew install cmake" >&2
    brew install cmake
    mac_host_deps_path_prepend
    command -v cmake >/dev/null 2>&1 && {
      echo ">>> cmake ok: $(cmake --version | head -1)" >&2
      return 0
    }
  fi
  local ver="${MAC_HOST_CMAKE_VER}"
  local name="cmake-${ver}-macos-universal"
  local tarball="${name}.tar.gz"
  echo ">>> installing CMake ${ver} → ${MAC_HOST_SDK}/${name}" >&2
  mkdir -p "${MAC_HOST_SDK}" "${MAC_HOST_PREFIX}/bin" /tmp
  curl -fsSL "https://github.com/Kitware/CMake/releases/download/v${ver}/${tarball}" -o "/tmp/${tarball}"
  rm -rf "${MAC_HOST_SDK}/${name}"
  tar -C "${MAC_HOST_SDK}" -xzf "/tmp/${tarball}"
  ln -sfn "${MAC_HOST_SDK}/${name}" "${MAC_HOST_SDK}/cmake"
  ln -sfn "${MAC_HOST_SDK}/cmake/CMake.app/Contents/bin/cmake" "${MAC_HOST_PREFIX}/bin/cmake"
  ln -sfn "${MAC_HOST_SDK}/cmake/CMake.app/Contents/bin/ctest" "${MAC_HOST_PREFIX}/bin/ctest"
  ln -sfn "${MAC_HOST_SDK}/cmake/CMake.app/Contents/bin/cpack" "${MAC_HOST_PREFIX}/bin/cpack"
  mac_host_deps_path_prepend
  command -v cmake >/dev/null 2>&1 || {
    echo "error: cmake not on PATH after install" >&2
    return 1
  }
  echo ">>> cmake ok: $(cmake --version | head -1)" >&2
}

_mac_host_install_pkgconfig() {
  if command -v pkg-config >/dev/null 2>&1; then
    echo ">>> pkg-config ok: $(pkg-config --version)" >&2
    return 0
  fi
  if command -v brew >/dev/null 2>&1; then
    echo ">>> brew install pkg-config" >&2
    brew install pkg-config
    mac_host_deps_path_prepend
    command -v pkg-config >/dev/null 2>&1 && {
      echo ">>> pkg-config ok: $(pkg-config --version)" >&2
      return 0
    }
  fi
  local ver="${MAC_HOST_PKGCONF_VER}"
  local name="pkgconf-${ver}"
  echo ">>> building pkgconf ${ver} → ${MAC_HOST_PREFIX}" >&2
  mkdir -p /tmp "${MAC_HOST_PREFIX}"
  curl -fsSL "https://distfiles.ariadne.space/pkgconf/${name}.tar.xz" -o "/tmp/${name}.tar.xz"
  rm -rf "/tmp/${name}"
  tar -C /tmp -xf "/tmp/${name}.tar.xz"
  (
    cd "/tmp/${name}"
    ./configure --prefix="${MAC_HOST_PREFIX}" --with-system-libdir=/usr/lib --with-system-includedir=/usr/include
    make -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
    make install
  )
  ln -sfn "${MAC_HOST_PREFIX}/bin/pkgconf" "${MAC_HOST_PREFIX}/bin/pkg-config"
  mac_host_deps_path_prepend
  command -v pkg-config >/dev/null 2>&1 || {
    echo "error: pkg-config not on PATH after install" >&2
    return 1
  }
  echo ">>> pkg-config ok: $(pkg-config --version)" >&2
}

_mac_host_install_uv() {
  if command -v uv >/dev/null 2>&1; then
    echo ">>> uv ok: $(uv --version)" >&2
    return 0
  fi
  echo ">>> installing uv → ${MAC_HOST_PREFIX}/bin" >&2
  curl -LsSf https://astral.sh/uv/install.sh | sh
  mac_host_deps_path_prepend
  # astral default is ~/.local/bin
  if [[ -x "${HOME}/.local/bin/uv" && ! -e "${MAC_HOST_PREFIX}/bin/uv" ]]; then
    ln -sfn "${HOME}/.local/bin/uv" "${MAC_HOST_PREFIX}/bin/uv"
  fi
  command -v uv >/dev/null 2>&1 || {
    echo "error: uv not on PATH after install" >&2
    return 1
  }
  echo ">>> uv ok: $(uv --version)" >&2
}

mac_host_deps_check() {
  mac_host_deps_path_prepend
  local missing=0
  if ! xcode-select -p >/dev/null 2>&1; then
    echo "FAIL: Xcode CLI tools missing — xcode-select --install" >&2
    missing=1
  else
    echo "OK: xcode-select $(xcode-select -p)" >&2
  fi
  if [[ ! -d /Applications/Xcode.app ]] && ! command -v brew >/dev/null 2>&1; then
    echo "WARN: no Xcode.app and no Homebrew — python3-embed may fail (install Xcode.app or brew python@3.12)" >&2
  fi
  if _mac_host_go_ok; then
    echo "OK: $(go version)" >&2
  else
    echo "FAIL: Go ≥1.24 required (go.mod)" >&2
    missing=1
  fi
  if command -v cmake >/dev/null 2>&1; then
    echo "OK: $(cmake --version | head -1)" >&2
  else
    echo "FAIL: cmake missing" >&2
    missing=1
  fi
  if command -v pkg-config >/dev/null 2>&1; then
    echo "OK: pkg-config $(pkg-config --version)" >&2
  else
    echo "FAIL: pkg-config missing" >&2
    missing=1
  fi
  if command -v uv >/dev/null 2>&1; then
    echo "OK: $(uv --version)" >&2
  else
    echo "FAIL: uv missing" >&2
    missing=1
  fi
  return "${missing}"
}

mac_host_deps_dry_run() {
  mac_host_deps_path_prepend
  echo "mac_host_deps dry-run (no downloads, no writes)"
  echo "  PREFIX:  ${MAC_HOST_PREFIX}"
  echo "  SDK:     ${MAC_HOST_SDK}"
  echo "  PATH +=  ${MAC_HOST_PREFIX}/bin : ${MAC_HOST_PREFIX}/go/bin"
  echo "  profile: ${HOME}/.zprofile  (marker: zerollama toolchain)"
  echo ""
  if _mac_host_go_ok; then
    echo "  go:         keep ($(go version 2>/dev/null | head -1))"
  else
    echo "  go:         INSTALL → ${MAC_HOST_PREFIX}/go  (tarball go${MAC_HOST_GO_VER}.darwin-$(_mac_host_arch))"
  fi
  if command -v cmake >/dev/null 2>&1; then
    echo "  cmake:      keep ($(cmake --version 2>/dev/null | head -1))"
  else
    echo "  cmake:      INSTALL → ${MAC_HOST_SDK}/cmake-${MAC_HOST_CMAKE_VER}-macos-universal + symlink ${MAC_HOST_PREFIX}/bin/cmake"
  fi
  if command -v pkg-config >/dev/null 2>&1; then
    echo "  pkg-config: keep ($(pkg-config --version 2>/dev/null))"
  else
    echo "  pkg-config: BUILD → pkgconf-${MAC_HOST_PKGCONF_VER} under ${MAC_HOST_PREFIX}"
  fi
  if command -v uv >/dev/null 2>&1; then
    echo "  uv:         keep ($(uv --version 2>/dev/null))"
  else
    echo "  uv:         INSTALL → ${MAC_HOST_PREFIX}/bin/uv (astral install.sh)"
  fi
  if command -v brew >/dev/null 2>&1; then
    echo "  brew:       present — prefer brew packages when installing"
  else
    echo "  brew:       absent — user-local tarballs/source only (no Homebrew required)"
  fi
}

mac_host_deps_ensure() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "warn: mac_host_deps targets Darwin; skipping" >&2
    return 0
  fi
  if [[ "${MAC_HOST_DRY_RUN}" == "1" ]]; then
    mac_host_deps_dry_run
    return 0
  fi
  mkdir -p "${MAC_HOST_PREFIX}/bin" "${MAC_HOST_SDK}"
  mac_host_deps_path_prepend
  mac_host_deps_ensure_path_profile
  echo "== Mac host deps (${MAC_HOST_PREFIX}) ==" >&2
  _mac_host_install_go
  _mac_host_install_cmake
  _mac_host_install_pkgconfig
  _mac_host_install_uv
  mac_host_deps_path_prepend
  if ! mac_host_deps_check; then
    echo "error: host deps still incomplete after install" >&2
    return 1
  fi
  echo "PASS: mac host deps ready" >&2
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  case "${1:-}" in
    --check)
      mac_host_deps_path_prepend
      mac_host_deps_check
      ;;
    --dry-run)
      MAC_HOST_DRY_RUN=1
      mac_host_deps_dry_run
      ;;
    ""|--ensure)
      mac_host_deps_ensure
      ;;
    *)
      echo "usage: $0 [--ensure|--check|--dry-run]" >&2
      exit 1
      ;;
  esac
fi
