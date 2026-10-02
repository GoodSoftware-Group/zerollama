#!/usr/bin/env bash
# Install Irodori-TTS-Server (OpenAI-compatible Japanese TTS + zero-shot clone).
# Assets on the models volume; sidecar listens on 127.0.0.1:8088 (see irodori-tts.service).
#
#   ./scripts/speech/install_irodori_tts.sh
#   sudo cp scripts/systemd/irodori-tts.service /etc/systemd/system/
#   sudo systemctl daemon-reload && sudo systemctl enable --now irodori-tts
#
# Env:
#   IRODORI_ROOT           — install dir (default /mnt/ollama_img/speech/irodori-tts-server)
#   IRODORI_VOICES_DIR     — voice refs (default /mnt/ollama_img/speech/irodori/voices)
#   IRODORI_HF_CHECKPOINT  — HF id (default Aratako/Irodori-TTS-v4.1-Small)
#   IRODORI_TTS_BACKEND    — uv extra: cu128|rocm|cpu (default cu128)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SPEECH_ROOT="${SPEECH_ROOT:-/mnt/ollama_img/speech}"
IRODORI_ROOT="${IRODORI_ROOT:-${SPEECH_ROOT}/irodori-tts-server}"
IRODORI_VOICES_DIR="${IRODORI_VOICES_DIR:-${SPEECH_ROOT}/irodori/voices}"
IRODORI_HF_CHECKPOINT="${IRODORI_HF_CHECKPOINT:-Aratako/Irodori-TTS-v4.1-Small}"
IRODORI_TTS_BACKEND="${IRODORI_TTS_BACKEND:-cu128}"
VOICES_CATALOG_DIR="${SPEECH_ROOT}/voices"
REPO_URL="${IRODORI_TTS_SERVER_REPO:-https://github.com/Aratako/Irodori-TTS-Server.git}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; install from https://docs.astral.sh/uv/" >&2
  exit 1
fi

mkdir -p "$IRODORI_VOICES_DIR" "$VOICES_CATALOG_DIR" "$(dirname "$IRODORI_ROOT")"

echo ">>> clone/update Irodori-TTS-Server → ${IRODORI_ROOT}"
if [[ -d "${IRODORI_ROOT}/.git" ]]; then
  git -C "$IRODORI_ROOT" fetch --depth 1 origin
  git -C "$IRODORI_ROOT" reset --hard origin/HEAD 2>/dev/null || git -C "$IRODORI_ROOT" pull --ff-only
else
  git clone --depth 1 "$REPO_URL" "$IRODORI_ROOT"
fi

echo ">>> uv sync --extra ${IRODORI_TTS_BACKEND}"
(
  cd "$IRODORI_ROOT"
  uv sync --extra "${IRODORI_TTS_BACKEND}"
  # Ensure ollama can exec without root-owned uv on PATH.
  if [[ ! -x .venv/bin/python ]]; then
    echo "ERROR: expected ${IRODORI_ROOT}/.venv/bin/python after uv sync" >&2
    exit 1
  fi
)

echo ">>> .env"
ENV_FILE="${IRODORI_ROOT}/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  if [[ -f "${IRODORI_ROOT}/.env.example" ]]; then
    cp "${IRODORI_ROOT}/.env.example" "$ENV_FILE"
  else
    touch "$ENV_FILE"
  fi
fi
# Idempotent key upserts
_upsert_env() {
  local key="$1" val="$2"
  if grep -qE "^${key}=" "$ENV_FILE" 2>/dev/null; then
    sed -i "s|^${key}=.*|${key}=${val}|" "$ENV_FILE"
  else
    printf '%s=%s\n' "$key" "$val" >>"$ENV_FILE"
  fi
}
_upsert_env IRODORI_HF_CHECKPOINT "$IRODORI_HF_CHECKPOINT"
_upsert_env IRODORI_VOICES_DIR "$IRODORI_VOICES_DIR"
_upsert_env IRODORI_ALLOW_NO_REF_VOICE true
_upsert_env IRODORI_MODEL_NAME irodori-tts
_upsert_env IRODORI_DEFAULT_VOICE none

echo ">>> voice catalogs + sample ref"
cp -f "${REPO_ROOT}/modelfiles/irodori/voices.json" "${VOICES_CATALOG_DIR}/irodori.json"
# Prefer an existing short Japanese sample if the upstream repo ships one; else leave a placeholder note.
if [[ ! -f "${IRODORI_VOICES_DIR}/sample.wav" ]]; then
  found=""
  for cand in \
    "${IRODORI_ROOT}/voices/sample.wav" \
    "${IRODORI_ROOT}/examples/sample.wav" \
    "${IRODORI_ROOT}/assets/sample.wav"; do
    if [[ -f "$cand" ]]; then
      cp -f "$cand" "${IRODORI_VOICES_DIR}/sample.wav"
      found=1
      break
    fi
  done
  if [[ -z "$found" ]]; then
    cat >"${IRODORI_VOICES_DIR}/README.txt" <<EOF
Place reference WAVs here (stem = voice id), e.g. sample.wav → voice "sample".
Or use voice=none with emotion=caption for Voice Design without a reference.
Clone via zerollama: backend_paths.tts_ref_audio → irodori.ref_wav.
EOF
    echo "WARN: no sample.wav yet — drop a Japanese reference clip at ${IRODORI_VOICES_DIR}/sample.wav" >&2
  fi
fi

# Ownership for ollama service user when present
if id ollama >/dev/null 2>&1; then
  chown -R ollama:ollama "$IRODORI_ROOT" "$IRODORI_VOICES_DIR" 2>/dev/null || true
fi

cat <<EOF
OK: Irodori-TTS-Server at ${IRODORI_ROOT}
  checkpoint: ${IRODORI_HF_CHECKPOINT}
  voices:     ${IRODORI_VOICES_DIR}
  catalog:    ${VOICES_CATALOG_DIR}/irodori.json

Start:
  sudo cp ${REPO_ROOT}/scripts/systemd/irodori-tts.service /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable --now irodori-tts
  curl -sS http://127.0.0.1:8088/health

Register zerollama tag:
  OLLAMA_MODELS=/mnt/ollama_img/models ${REPO_ROOT}/scripts/register_speech_models.sh
EOF
