#!/usr/bin/env bash
# Install Kokoro ONNX weights + venv for scripts/tts_remote_server.py on :8090.
# zerollama tag kokoro:latest is remote-tts → OLLAMA_TTS_URL (default 127.0.0.1:8090).
set -euo pipefail

SPEECH_ROOT="${SPEECH_ROOT:-/mnt/ollama_img/speech}"
KOKORO_DIR="${SPEECH_ROOT}/kokoro"
VENV="${KOKORO_DIR}/.venv"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# HF mirror of the official kokoro-onnx v1.0 pair (GitHub releases often 504 here).
MODEL_URL="${KOKORO_MODEL_URL:-https://huggingface.co/rumbleFTW/kokoro-v1.0-onnx/resolve/main/kokoro-v1.0.onnx}"
VOICES_URL="${KOKORO_VOICES_URL:-https://huggingface.co/rumbleFTW/kokoro-v1.0-onnx/resolve/main/voices-v1.0.bin}"
PY="${KOKORO_PYTHON:-python3.11}"

mkdir -p "$KOKORO_DIR"
cd "$KOKORO_DIR"

if [[ ! -f kokoro-v1.0.onnx ]]; then
  echo ">>> download kokoro-v1.0.onnx"
  curl -fL --retry 5 --retry-delay 2 -o kokoro-v1.0.onnx.tmp "$MODEL_URL"
  mv kokoro-v1.0.onnx.tmp kokoro-v1.0.onnx
fi
if [[ ! -f voices-v1.0.bin ]]; then
  echo ">>> download voices-v1.0.bin"
  curl -fL --retry 5 --retry-delay 2 -o voices-v1.0.bin.tmp "$VOICES_URL"
  mv voices-v1.0.bin.tmp voices-v1.0.bin
fi
ls -lh kokoro-v1.0.onnx voices-v1.0.bin

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo ">>> venv ${VENV} ($PY)"
  "$PY" -m venv "$VENV"
fi
# Lean deps only. misaki[en] pulls spacy+torch CUDA; espeakng-loader is enough.
"${VENV}/bin/pip" install -U 'pip<26' wheel
"${VENV}/bin/pip" install 'kokoro-onnx'

# Runnable copy for User=ollama (no /root access).
cp -f "${REPO_ROOT}/scripts/tts_remote_server.py" "${KOKORO_DIR}/tts_remote_server.py"

echo ">>> smoke synthesize"
KOKORO_MODEL="${KOKORO_DIR}/kokoro-v1.0.onnx" \
KOKORO_VOICES="${KOKORO_DIR}/voices-v1.0.bin" \
"${VENV}/bin/python" -c "
import importlib.util
spec = importlib.util.spec_from_file_location('tts_remote_server', '${KOKORO_DIR}/tts_remote_server.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
wav = mod.synthesize_kokoro('Hello from Kokoro.', 'af_heart', '')
assert len(wav) > 1000, len(wav)
open('/tmp/kokoro-sidecar-smoke.wav', 'wb').write(wav)
print('smoke ok', len(wav), 'bytes')
"

chown -R ollama:ollama "$KOKORO_DIR"

echo "OK: assets under ${KOKORO_DIR}"
echo "Install unit:"
echo "  sudo cp ${REPO_ROOT}/scripts/systemd/remote-tts.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload && sudo systemctl enable --now remote-tts"
