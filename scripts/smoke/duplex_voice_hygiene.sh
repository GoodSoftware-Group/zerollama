#!/usr/bin/env bash
# Duplex / spoken-reply hygiene smoke.
#
# Catches the failure mode where runtime /api/chat used User:/Assistant: lines
# and the ear spoke "Assistant: …" / ChatML into the mic reply.
#
# Tiers (opt-in; skip cleanly when the host is down):
#   default                 — /internal/render-chat is a real template + /api/chat hygiene
#   RUN_E2E_DUPLEX_TTS=1    — also TTS the chat reply; assert audible PCM
#   RUN_E2E_DUPLEX_FULL=1   — also speech→duplex socket→reply text+audio hygiene
#
# Env:
#   DUPLEX_SMOKE_HOST   base URL (default OLLAMA_HOST or http://127.0.0.1:2083)
#   DUPLEX_SMOKE_MODEL  chat tag (default qwen2.5:14b)
#   DUPLEX_SMOKE_VOICE  speech tag (default piper-lessac)
#   DUPLEX_SMOKE_ASR    duplex ASR tag (default qwen3-asr:latest)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

raw_host="${DUPLEX_SMOKE_HOST:-${OLLAMA_HOST:-http://127.0.0.1:2083}}"
case "${raw_host}" in
  http://*|https://*) HOST="${raw_host%/}" ;;
  *) HOST="http://${raw_host%/}" ;;
esac
MODEL="${DUPLEX_SMOKE_MODEL:-qwen2.5:14b}"
VOICE="${DUPLEX_SMOKE_VOICE:-piper-lessac}"
ASR="${DUPLEX_SMOKE_ASR:-qwen3-asr:latest}"
TMP="${TMPDIR:-/tmp}/duplex_voice_hygiene_$$"
mkdir -p "${TMP}"
cleanup() { rm -rf "${TMP}"; }
trap cleanup EXIT

if ! curl -sf -m 3 "${HOST}/api/version" >/dev/null; then
  echo "skip: ${HOST}/api/version unreachable" >&2
  exit 0
fi

echo ">>> host=${HOST} model=${MODEL}" >&2

echo ">>> /internal/render-chat must use a real chat template" >&2
curl -sf -m 10 "${HOST}/internal/render-chat" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"truncate\":false}" \
  > "${TMP}/render.json"
python3 - <<PY
import json, sys
d = json.load(open("${TMP}/render.json"))
p = d.get("prompt") or ""
if d.get("error"):
    print("render error:", d["error"], file=sys.stderr)
    sys.exit(1)
markers = ("<|im_start|>", "<|start|>", "[INST]", "<|user|>", "<start_of_turn>")
if not any(m in p for m in markers):
    print("expected a real chat template, got:", repr(p[:240]), file=sys.stderr)
    sys.exit(1)
print("render ok, bytes", len(p), "truncate_mode", d.get("truncate_mode"))
PY

echo ">>> /api/chat multi-turn hygiene (catches User:/Assistant: leaks)" >&2
cat > "${TMP}/chat_req.json" <<JSON
{
  "model": "${MODEL}",
  "stream": false,
  "messages": [
    {"role": "system", "content": "Reply in one short spoken sentence. Never write role labels."},
    {"role": "user", "content": "Can you see anything else?"},
    {"role": "assistant", "content": "Yes, I can see a keyboard and a mouse."},
    {"role": "user", "content": "Is there anything else nearby?"}
  ],
  "options": {"num_predict": 48, "num_ctx": 8192, "temperature": 0.2}
}
JSON
curl -sf -m 180 "${HOST}/api/chat" \
  -H 'Content-Type: application/json' \
  -d @"${TMP}/chat_req.json" \
  > "${TMP}/chat.json"
python3 - <<PY
import json, pathlib, sys
d = json.load(open("${TMP}/chat.json"))
if d.get("error"):
    print(d["error"], file=sys.stderr)
    sys.exit(1)
text = ((d.get("message") or {}).get("content") or "").strip()
pathlib.Path("${TMP}/reply.txt").write_text(text + "\n")
print("chat reply:", repr(text))
PY
( cd "${ROOT}" && go run ./scripts/smoke/duplex_hygiene_check.go -file "${TMP}/reply.txt" )

if [[ "${RUN_E2E_DUPLEX_TTS:-0}" != "1" && "${RUN_E2E_DUPLEX_FULL:-0}" != "1" ]]; then
  echo "PASS chat+render hygiene (set RUN_E2E_DUPLEX_TTS=1 / RUN_E2E_DUPLEX_FULL=1 for audio)" >&2
  exit 0
fi

echo ">>> /v1/audio/speech of chat reply (${VOICE})" >&2
python3 - <<PY
import json, pathlib, urllib.request
text = pathlib.Path("${TMP}/reply.txt").read_text().strip()
body = json.dumps({"model": "${VOICE}", "input": text, "response_format": "wav"}).encode()
req = urllib.request.Request("${HOST}/v1/audio/speech", data=body, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as r:
    pathlib.Path("${TMP}/reply.wav").write_bytes(r.read())
print("tts bytes", pathlib.Path("${TMP}/reply.wav").stat().st_size)
PY
python3 - "${TMP}/reply.wav" <<'PY'
import array, sys, wave
path = sys.argv[1]
with wave.open(path, "rb") as w:
    n, rate, sw, ch = w.getnframes(), w.getframerate(), w.getsampwidth(), w.getnchannels()
    raw = w.readframes(n)
if sw != 2 or ch < 1 or rate < 8000 or n < 1:
    print(f"bad wav rate={rate} sw={sw} ch={ch} n={n}", file=sys.stderr)
    sys.exit(1)
samples = array.array("h")
samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
rms = (sum(int(s) * int(s) for s in samples) / len(samples)) ** 0.5
dur = len(samples) / float(rate) / max(ch, 1)
print(f"tts pcm: rate={rate} dur={dur:.2f}s rms={rms:.1f} frames={len(samples)}")
if dur < 0.2 or dur > 30:
    print("unreasonable duration", file=sys.stderr)
    sys.exit(1)
if rms < 80:
    print("near-silent pcm", file=sys.stderr)
    sys.exit(1)
PY

if [[ "${RUN_E2E_DUPLEX_FULL:-0}" != "1" ]]; then
  echo "PASS chat+TTS hygiene" >&2
  exit 0
fi

echo ">>> duplex full: speak prompt → realtime duplex → hygiene" >&2
PROMPT_TEXT="What is two plus two? Answer with one short sentence."
python3 - <<PY
import json, pathlib, urllib.request
body = json.dumps({
    "model": "${VOICE}",
    "input": """${PROMPT_TEXT}""",
    "response_format": "wav",
}).encode()
req = urllib.request.Request("${HOST}/v1/audio/speech", data=body, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as r:
    pathlib.Path("${TMP}/prompt.wav").write_bytes(r.read())
print("prompt wav bytes", pathlib.Path("${TMP}/prompt.wav").stat().st_size)
PY

( cd "${ROOT}" && go run ./scripts/audio/realtime_probe \
  -host "${HOST}" \
  -model "${MODEL}" \
  -pipeline duplex \
  -tts "${VOICE}" \
  -asr "${ASR}" \
  -rate 16000 \
  -wav "${TMP}/prompt.wav" \
  -out "${TMP}/duplex_reply.wav" \
  -timeout 3m \
  -check-hygiene \
  -min-audio-bytes 2000 \
) | tee "${TMP}/duplex.json"

python3 - <<PY
import array, json, sys, wave
rep = json.load(open("${TMP}/duplex.json"))
reply = (rep.get("reply") or "").strip()
print("duplex reply:", repr(reply))
print("audio_bytes:", rep.get("audio_bytes"), "total_ms:", rep.get("total_ms"), "status:", rep.get("status"))
path = "${TMP}/duplex_reply.wav"
with wave.open(path, "rb") as w:
    n, rate, sw, ch = w.getnframes(), w.getframerate(), w.getsampwidth(), w.getnchannels()
    raw = w.readframes(n)
samples = array.array("h")
samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
rms = (sum(int(s) * int(s) for s in samples) / max(len(samples), 1)) ** 0.5
dur = len(samples) / float(rate) / max(ch, 1) if rate else 0.0
print(f"duplex pcm: rate={rate} dur={dur:.2f}s rms={rms:.1f}")
if dur < 0.15 or rms < 80:
    print("duplex audio not audible enough", file=sys.stderr)
    sys.exit(1)
low = reply.lower()
if "four" not in low and "4" not in low:
    print("warn: reply did not mention four/4 (soft):", reply, file=sys.stderr)
PY

echo "PASS duplex full hygiene" >&2
