#!/usr/bin/env bash
# Download one llama.cpp mtmd audio pack (text GGUF + mmproj) into an empty-enough
# directory and write a Modelfile so `zerollama create` keeps only those two files.
#
# Does not start a server and does not run create. Weights stay on disk until
# you create the tag against a server you already have.
#
#   ./scripts/audio/fetch_mtmd_audio.sh --list
#   ./scripts/audio/fetch_mtmd_audio.sh --dry-run voxtral-realtime
#   ./scripts/audio/fetch_mtmd_audio.sh voxtral-realtime ./models/voxtral-realtime
#   ./scripts/audio/fetch_mtmd_audio.sh --main-gpu 1 lfm3-audio ./models/lfm3-audio
set -euo pipefail

# name|tag|placement|urls...
# placement is "one" (fits one 24 GiB card, or a realtime encoder) or "pair"
# (layer-split; do not set main_gpu).
packs=(
  "voxtral-realtime|voxtral-realtime|one|https://huggingface.co/ggml-org/Voxtral-Mini-4B-Realtime-2602-GGUF/resolve/main/Voxtral-Mini-4B-Realtime-2602-Q4_K_M.gguf|https://huggingface.co/ggml-org/Voxtral-Mini-4B-Realtime-2602-GGUF/resolve/main/mmproj-Voxtral-Mini-4B-Realtime-2602-Q8_0.gguf"
  "lfm3-audio|lfm3-audio|one|https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF/resolve/main/LFM3-Audio-2B-Q4_K_M.gguf|https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF/resolve/main/mmproj-LFM3-Audio-2B-Q8_0.gguf"
  "lfm3-omni|lfm3-omni|one|https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF/resolve/main/LFM3-Omni-2B-Q4_K_M.gguf|https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF/resolve/main/mmproj-LFM3-Omni-2B-Q8_0.gguf"
  "voxtral-mini|voxtral-mini|one|https://huggingface.co/ggml-org/Voxtral-Mini-3B-2507-GGUF/resolve/main/Voxtral-Mini-3B-2507-Q4_K_M.gguf|https://huggingface.co/ggml-org/Voxtral-Mini-3B-2507-GGUF/resolve/main/mmproj-Voxtral-Mini-3B-2507-Q8_0.gguf"
  "qwen25-omni|qwen25-omni|one|https://huggingface.co/ggml-org/Qwen2.5-Omni-7B-GGUF/resolve/main/Qwen2.5-Omni-7B-Q4_K_M.gguf|https://huggingface.co/ggml-org/Qwen2.5-Omni-7B-GGUF/resolve/main/mmproj-Qwen2.5-Omni-7B-Q8_0.gguf"
  "voxtral-small-q4|voxtral-small|one|https://huggingface.co/bartowski/mistralai_Voxtral-Small-24B-2507-GGUF/resolve/main/mistralai_Voxtral-Small-24B-2507-Q4_K_M.gguf|https://huggingface.co/bartowski/mistralai_Voxtral-Small-24B-2507-GGUF/resolve/main/mmproj-mistralai_Voxtral-Small-24B-2507-f16.gguf"
  "qwen3-omni-q4|qwen3-omni|one|https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf|https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF/resolve/main/mmproj-Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf"
  "voxtral-small-q8|voxtral-small-q8|pair|https://huggingface.co/bartowski/mistralai_Voxtral-Small-24B-2507-GGUF/resolve/main/mistralai_Voxtral-Small-24B-2507-Q8_0.gguf|https://huggingface.co/bartowski/mistralai_Voxtral-Small-24B-2507-GGUF/resolve/main/mmproj-mistralai_Voxtral-Small-24B-2507-f16.gguf"
  "qwen3-omni-q8|qwen3-omni-q8|pair|https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf|https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF/resolve/main/mmproj-Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf"
  "ultravox-70b-q4|ultravox-70b|pair|https://huggingface.co/steampunque/Llama-3.3-70B-Instruct-MP-GGUF/resolve/main/Llama-3.3-70B-Instruct.Q4_K_H.gguf|https://huggingface.co/steampunque/ultravox-v0_6-llama-3_3-70b-MP-GGUF/resolve/main/ultravox-v0_6-llama-3_3-70b.mmproj.gguf"
)

usage() {
  cat <<'EOF'
usage: fetch_mtmd_audio.sh [--list] [--dry-run] [--main-gpu N] <pack> <dir>

Packs:
  voxtral-realtime   Voxtral Mini 4B Realtime Q4 + mmproj. Live socket, one GPU.
  lfm3-audio         LFM3-Audio 2B Q4 + mmproj. Live socket and file audio.
  lfm3-omni          LFM3-Omni 2B Q4 + mmproj. Live socket, vision, reasoning off.
  voxtral-mini       Voxtral Mini 3B Q4. File audio, one GPU.
  qwen25-omni        Qwen2.5-Omni 7B Q4. File audio, one GPU.
  voxtral-small-q4   Voxtral Small 24B Q4. File audio, one 24 GiB card.
  qwen3-omni-q4      Qwen3-Omni 30B-A3B Q4. File audio and vision, one 24 GiB card.
  voxtral-small-q8   Voxtral Small 24B Q8. File audio, both GPUs.
  qwen3-omni-q8      Qwen3-Omni 30B-A3B Q8. File audio and vision, both GPUs.
  ultravox-70b-q4    Ultravox v0.6 on Llama 3.3 70B Q4. File audio, both GPUs.

--dry-run prints each file's size and does not write anything. <dir> is optional.
--main-gpu N pins the tag to that device (N >= 1). Leave it unset for a pack
marked pair. main_gpu 0 is unset in zerollama and is rejected here.
EOF
}

list_packs() {
  printf '%-20s %-18s %s\n' PACK TAG PLACEMENT
  local row name tag place _
  for row in "${packs[@]}"; do
    IFS='|' read -r name tag place _ <<<"$row"
    printf '%-20s %-18s %s\n' "$name" "$tag" "$place"
  done
}

main_gpu=""
dry=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list|-h|--help)
      if [[ "$1" == "--list" ]]; then list_packs; else usage; fi
      exit 0
      ;;
    --dry-run)
      dry=1
      shift
      ;;
    --main-gpu)
      main_gpu="${2:-}"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ "$dry" -eq 1 && $# -eq 1 ]]; then
  set -- "$1" .
elif [[ $# -ne 2 ]]; then
  usage
  exit 2
fi
pack="$1"
dest="$2"

if [[ -n "$main_gpu" ]]; then
  if ! [[ "$main_gpu" =~ ^[0-9]+$ ]] || [[ "$main_gpu" -lt 1 ]]; then
    echo "main_gpu must be 1 or higher (0 is unset and layer-splits)" >&2
    exit 2
  fi
fi

row=""
for candidate in "${packs[@]}"; do
  IFS='|' read -r name _ <<<"$candidate"
  if [[ "$name" == "$pack" ]]; then
    row="$candidate"
    break
  fi
done
if [[ -z "$row" ]]; then
  echo "unknown pack $pack" >&2
  list_packs >&2
  exit 2
fi

IFS='|' read -r name tag place url1 url2 <<<"$row"
if [[ "$place" == "pair" && -n "$main_gpu" ]]; then
  echo "$pack needs both GPUs; omit --main-gpu" >&2
  exit 2
fi

content_length() {
  curl -fsIL "$1" | tr -d '\r' | awk 'tolower($1)=="content-length:" {n=$2} END {print n+0}'
}

if [[ "$dry" -eq 1 ]]; then
  total=0
  for url in "$url1" "$url2"; do
    n="$(content_length "$url")"
    total=$((total + n))
    awk -v name="$(basename "$url")" -v n="$n" 'BEGIN { printf "%8.2f GiB  %s\n", n/1024/1024/1024, name }'
  done
  awk -v n="$total" 'BEGIN { printf "%8.2f GiB  total\n", n/1024/1024/1024 }'
  exit 0
fi

mkdir -p "$dest"
dest="$(cd "$dest" && pwd)"

download() {
  local url="$1"
  local file="$dest/$(basename "$url")"
  echo "fetch $file"
  curl -fL --retry 3 --continue-at - -o "$file" "$url"
}

download "$url1"
download "$url2"

{
  echo "FROM ."
  if [[ -n "$main_gpu" ]]; then
    echo "PARAMETER main_gpu $main_gpu"
  fi
} >"$dest/Modelfile"

echo
echo "tag: $tag ($place)"
echo "create from $dest:"
echo "  zerollama create $tag -f $dest/Modelfile"
case "$pack" in
  voxtral-realtime)
    echo "then: go run ./scripts/audio/realtime_probe -model $tag -wav clip.wav"
    ;;
  lfm3-audio|lfm3-omni)
    echo "live transcript: go run ./scripts/audio/realtime_probe -model $tag -wav clip.wav"
    echo "conversation:    go run ./scripts/audio/realtime_probe -model $tag -intent \"\" -modalities audio -out reply.wav -wav clip.wav"
    echo "barge-in:        add -barge"
    ;;
esac
