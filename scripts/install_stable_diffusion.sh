#!/usr/bin/env bash
# Install stable-diffusion.cpp (Vulkan) and local GGUF image models for zerollama.
#
# Usage: ./scripts/install_stable_diffusion.sh [--build|--binary] [--models-only]
#
# Downloads:
#   sd15-vulkan      stable-diffusion-v1-5-Q4_0.gguf
#   sd15-q8-vulkan   stable-diffusion-v1-5-Q8_0.gguf
#   sd15-turbo-vulkan sd_turbo-f16-q8_0.gguf
#   sdxl-vulkan      sd_xl_base_1.0_0_Q4_0.gguf
#
# Models land under SD_ROOT/models/ (default /usr/share/zerollama/sd-cpp).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SD_ROOT="${SD_ROOT:-/usr/share/zerollama/sd-cpp}"
MODE="binary"
MODELS_ONLY=0
SD_REPO="${SD_REPO:-$SD_ROOT/stable-diffusion.cpp}"
SD_TAG="${SD_TAG:-master-746-2574f59}"
SD_RELEASE_HASH="${SD_RELEASE_HASH:-${SD_TAG##*-}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) MODE="build"; shift ;;
    --binary) MODE="binary"; shift ;;
    --models-only) MODELS_ONLY=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--build|--binary] [--models-only]"
      echo "  SD_ROOT=$SD_ROOT"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$SD_ROOT/bin" "$SD_ROOT/models"

install_binary_release() {
  local zip="sd-master-${SD_RELEASE_HASH}-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip"
  local url="https://github.com/leejet/stable-diffusion.cpp/releases/download/${SD_TAG}/${zip}"
  local tmp
  tmp="$(mktemp -d)"
  echo "Downloading $url ..."
  curl -fL --progress-bar -o "$tmp/$zip" "$url"
  python3 -m zipfile -e "$tmp/$zip" "$tmp/extract"
  local cli
  cli="$(find "$tmp/extract" -name 'sd-cli' -type f | head -1)"
  if [[ -z "$cli" ]]; then
    echo "sd-cli not found in release zip" >&2
    exit 1
  fi
  cp "$cli" "$SD_ROOT/bin/sd-cli"
  find "$tmp/extract" -maxdepth 2 -name '*.so*' -exec cp -a {} "$SD_ROOT/bin/" \; 2>/dev/null || true
  chmod +x "$SD_ROOT/bin/sd-cli"
  rm -rf "$tmp"
}

install_binary_build() {
  if [[ ! -d "$SD_REPO/.git" ]]; then
    git clone --depth 1 https://github.com/leejet/stable-diffusion.cpp.git "$SD_REPO"
  fi
  cmake -S "$SD_REPO" -B "$SD_REPO/build" -DSD_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build "$SD_REPO/build" --target sd-cli -j"$(nproc)"
  cp "$SD_REPO/build/bin/sd-cli" "$SD_ROOT/bin/sd-cli"
  find "$SD_REPO/build/bin" -name '*.so*' -exec cp -a {} "$SD_ROOT/bin/" \; 2>/dev/null || true
}

download_model() {
  local repo="$1"
  local file="$2"
  local dest="$SD_ROOT/models/$file"
  if [[ -f "$dest" ]]; then
    echo "skip (exists): $file"
    return 0
  fi
  echo "Downloading ${repo}/${file} ..."
  curl -fL --progress-bar -o "$dest" \
    "https://huggingface.co/${repo}/resolve/main/${file}"
}

if [[ "$MODELS_ONLY" -eq 0 ]]; then
  case "$MODE" in
    binary) install_binary_release ;;
    build) install_binary_build ;;
  esac
fi

# repo|filename
SD_MODELS=(
  "gpustack/stable-diffusion-v1-5-GGUF|stable-diffusion-v1-5-Q4_0.gguf"
  "gpustack/stable-diffusion-v1-5-GGUF|stable-diffusion-v1-5-Q8_0.gguf"
  "Green-Sky/SD-Turbo-GGUF|sd_turbo-f16-q8_0.gguf"
  "kostakoff/stable-diffusion-xl-base-1.0-GGUF|sd_xl_base_1.0_0_Q4_0.gguf"
)

for entry in "${SD_MODELS[@]}"; do
  repo="${entry%%|*}"
  file="${entry##*|}"
  download_model "$repo" "$file"
done

chmod +x "$REPO_ROOT/scripts/sd_external_image.sh"

echo ""
echo "Installed sd-cli: $SD_ROOT/bin/sd-cli"
echo "Models:"
ls -lh "$SD_ROOT/models/"*.gguf 2>/dev/null || true
echo ""
echo "Register: ./scripts/register_sd_models.sh"
echo "Generate examples:"
echo "  zerollama run sd15-turbo-vulkan \"a cat astronaut\""
echo "  zerollama run sd15-q8-vulkan \"oil painting of mountains\""
echo "  zerollama run sdxl-vulkan \"cinematic portrait, soft light\""
