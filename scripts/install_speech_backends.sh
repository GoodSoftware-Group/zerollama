#!/usr/bin/env bash
# Install Piper TTS + Whisper STT + multi-voice packs + remote-tts voice catalogs.
# Assets live on the models volume (root disk is tight on this host).
set -euo pipefail

SPEECH_ROOT="${SPEECH_ROOT:-/mnt/ollama_img/speech}"
PIPER_DIR="${SPEECH_ROOT}/piper"
WHISPER_DIR="${SPEECH_ROOT}/whisper"
VOICES_DIR="${SPEECH_ROOT}/voices"
REFS_DIR="${SPEECH_ROOT}/refs"
TMP="${TMPDIR:-/tmp}/zerollama-speech-install"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$PIPER_DIR" "$WHISPER_DIR" "$VOICES_DIR" "$REFS_DIR" "$TMP"

echo ">>> Piper binary"
if [[ ! -x "${PIPER_DIR}/piper" ]]; then
  curl -fsSL -o "${TMP}/piper_linux_x86_64.tar.gz" \
    "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"
  tar -xzf "${TMP}/piper_linux_x86_64.tar.gz" -C "$TMP"
  if [[ -d "${TMP}/piper" ]]; then
    rsync -a "${TMP}/piper/" "${PIPER_DIR}/"
  else
    find "$TMP" -maxdepth 2 -type f -name piper -executable -exec cp {} "${PIPER_DIR}/piper" \;
  fi
  chmod +x "${PIPER_DIR}/piper"
fi
"${PIPER_DIR}/piper" --help >/dev/null 2>&1 || "${PIPER_DIR}/piper" -h >/dev/null 2>&1 || true
ls -la "${PIPER_DIR}/piper"

# lang/locale/speaker/quality/voice_base — skip quietly if upstream 404s
download_piper_voice() {
  local lang="$1" locale="$2" speaker="$3" quality="$4" base="$5"
  local url_base="https://huggingface.co/rhasspy/piper-voices/resolve/main/${lang}/${locale}/${speaker}/${quality}"
  if [[ ! -f "${PIPER_DIR}/${base}.onnx" ]]; then
    echo ">>> Piper voice ${base}"
    if ! curl -fsSL -o "${PIPER_DIR}/${base}.onnx" "${url_base}/${base}.onnx"; then
      echo "WARN: skip ${base}.onnx (not found upstream)"
      rm -f "${PIPER_DIR}/${base}.onnx"
      return 0
    fi
  fi
  if [[ ! -f "${PIPER_DIR}/${base}.onnx.json" ]]; then
    if ! curl -fsSL -o "${PIPER_DIR}/${base}.onnx.json" "${url_base}/${base}.onnx.json"; then
      echo "WARN: skip ${base}.onnx.json"
      rm -f "${PIPER_DIR}/${base}.onnx" "${PIPER_DIR}/${base}.onnx.json"
      return 0
    fi
  fi
}

download_piper_voice en en_US lessac medium en_US-lessac-medium
download_piper_voice en en_US amy medium en_US-amy-medium
download_piper_voice en en_US joe medium en_US-joe-medium
download_piper_voice en en_US ryan medium en_US-ryan-medium
download_piper_voice en en_US arctic medium en_US-arctic-medium
download_piper_voice en en_US hfc_female medium en_US-hfc_female-medium
download_piper_voice en en_US hfc_male medium en_US-hfc_male-medium
download_piper_voice en en_US kusal medium en_US-kusal-medium
download_piper_voice en en_US l2arctic medium en_US-l2arctic-medium
download_piper_voice en en_US libritts high en_US-libritts-high
download_piper_voice en en_GB alba medium en_GB-alba-medium

ls -lh "${PIPER_DIR}"/*.onnx 2>/dev/null | head -40 || true

echo ">>> Voice catalogs (remote-tts)"
cp -f "${REPO_ROOT}/modelfiles/chatterbox/voices.json" "${VOICES_DIR}/chatterbox.json"
cp -f "${REPO_ROOT}/modelfiles/orpheus/voices.json" "${VOICES_DIR}/orpheus.json"
cp -f "${REPO_ROOT}/modelfiles/kokoro/voices.json" "${VOICES_DIR}/kokoro.json"
# Alias engine names used by scripts/tts_remote_server.py
ln -sfn chatterbox.json "${VOICES_DIR}/echo.json" 2>/dev/null || cp -f "${VOICES_DIR}/chatterbox.json" "${VOICES_DIR}/echo.json"
ls -la "${VOICES_DIR}/"

echo ">>> Whisper.cpp CLI (ubuntu x64)"
if [[ ! -x "${WHISPER_DIR}/whisper-cli" && ! -x "${WHISPER_DIR}/whisper" && ! -x "${WHISPER_DIR}/main" ]]; then
  curl -fsSL -o "${TMP}/whisper-bin-ubuntu-x64.tar.gz" \
    "https://github.com/ggml-org/whisper.cpp/releases/download/v1.9.1/whisper-bin-ubuntu-x64.tar.gz"
  tar -xzf "${TMP}/whisper-bin-ubuntu-x64.tar.gz" -C "$TMP"
  found=""
  for cand in whisper-cli whisper main; do
    p="$(find "$TMP" -type f -name "$cand" | head -1 || true)"
    if [[ -n "$p" ]]; then
      cp "$p" "${WHISPER_DIR}/$cand"
      chmod +x "${WHISPER_DIR}/$cand"
      found="$cand"
      break
    fi
  done
  if [[ -n "$found" ]]; then
    srcdir="$(dirname "$(find "$TMP" -type f -name "$found" | head -1)")"
    find "$srcdir" -maxdepth 1 -name '*.so*' -exec cp -a {} "${WHISPER_DIR}/" \; || true
  fi
fi
# Absolute ROOT — deriving from BASH_SOURCE breaks when /usr/local/bin/whisper
# is a symlink into this dir (dirname becomes /usr/local/bin → missing libs / loops).
cat > "${WHISPER_DIR}/whisper-run" <<WRAP
#!/usr/bin/env bash
set -euo pipefail
ROOT="${WHISPER_DIR}"
export LD_LIBRARY_PATH="\${ROOT}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
if [[ -x "\${ROOT}/whisper-cli" ]]; then
  exec "\${ROOT}/whisper-cli" "\$@"
fi
exec "\${ROOT}/main" "\$@"
WRAP
chmod +x "${WHISPER_DIR}/whisper-run"
ln -sfn whisper-run "${WHISPER_DIR}/whisper"
ls -la "${WHISPER_DIR}/"

echo ">>> Whisper GGML base.en"
if [[ ! -f "${WHISPER_DIR}/ggml-base.en.bin" ]]; then
  curl -fsSL -o "${WHISPER_DIR}/ggml-base.en.bin" \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
fi
ls -lh "${WHISPER_DIR}/ggml-base.en.bin"

sudo mkdir -p /usr/local/lib/zerollama-speech
sudo ln -sfn "${PIPER_DIR}" /usr/local/lib/zerollama-speech/piper
sudo ln -sfn "${WHISPER_DIR}" /usr/local/lib/zerollama-speech/whisper
sudo ln -sfn "${PIPER_DIR}/piper" /usr/local/bin/piper
# Real file (not symlink into WHISPER_DIR) so PATH lookup cannot recurse into whisper-run.
sudo tee /usr/local/bin/whisper >/dev/null <<WRAP
#!/usr/bin/env bash
exec ${WHISPER_DIR}/whisper-run "\$@"
WRAP
sudo chmod +x /usr/local/bin/whisper

echo ">>> smoke piper"
VOICE_BASE="en_US-lessac-medium"
echo "Hello from zerollama speech." | "${PIPER_DIR}/piper" \
  --model "${PIPER_DIR}/${VOICE_BASE}.onnx" \
  --config "${PIPER_DIR}/${VOICE_BASE}.onnx.json" \
  --output_file /tmp/piper-smoke.wav
ls -lh /tmp/piper-smoke.wav

echo ">>> smoke whisper"
export LD_LIBRARY_PATH="${WHISPER_DIR}:${LD_LIBRARY_PATH:-}"
"${WHISPER_DIR}/whisper" -m "${WHISPER_DIR}/ggml-base.en.bin" -f /tmp/piper-smoke.wav -otxt -of /tmp/whisper-smoke
cat /tmp/whisper-smoke.txt

echo "OK: speech assets under ${SPEECH_ROOT}"
echo "Remote TTS: on the GPU host run:"
echo "  TTS_ENGINE=chatterbox TTS_PORT=8090 python3 ${REPO_ROOT}/scripts/tts_remote_server.py"
echo "Then register models and set tts_url in modelfiles (or OLLAMA_TTS_URL)."
