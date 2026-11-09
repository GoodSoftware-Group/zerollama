---
name: text-to-speech
description: "Synthesize speech audio from text via a zerollama server's OpenAI-compatible speech API (Piper, Chatterbox, Orpheus, Kokoro, Irodori)."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, tts, audio, speech, piper, chatterbox, orpheus, kokoro, irodori]
    category: mlops
    related_skills: [zerollama-integration, speech-to-text, download-model, install-multimedia-model]
---

# Text-to-Speech Skill

Synthesize speech audio from text on a [zerollama](https://github.com/GoodSoftware-Group/zerollama)
server via the OpenAI-compatible `/v1/audio/speech` endpoint. Backends are
either **Piper** (CPU, always WAV) or a **remote-tts** bridge (Chatterbox,
Orpheus, Kokoro, Irodori), selected by the `model` tag.

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/v1/audio/speech -d '{}'   # 400/422 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/tags   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/v1/audio/voices   # 200/400 = route exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

## When to Use

- The user asks to generate spoken audio / a voiceover from text locally
- Picking a voice for a given speech model
- Debugging wrong/empty audio output or format mismatches

## Prerequisites

- zerollama server running (default `http://localhost:11434`)
- A speech-capable model pulled (`piper-lessac`, `chatterbox`, `orpheus`,
  `kokoro`, `irodori`, etc.) — check `GET /api/tags`
- Irodori needs the Irodori-TTS-Server sidecar (`:8088`; see
  `scripts/speech/install_irodori_tts.sh`)

## API Contract

`POST /v1/audio/speech`

| Field | Required | Notes |
|---|---|---|
| `model` | yes | Speech model tag, e.g. `piper-lessac`, `chatterbox`, `orpheus`, `kokoro`, `irodori` |
| `input` | yes | Text to synthesize, max 4096 Unicode characters |
| `voice` | no | Voice id from `GET /v1/audio/voices` |
| `response_format` | no | Piper: `wav` (anything else is 400). llama-server gen: `wav` or `pcm`. remote-tts forwards the value and returns the upstream `Content-Type`. No transcoding |
| `emotion` | no | zerollama extension. remote-tts forwards it (Orpheus: `excited`/`sad`; Irodori: Japanese caption / Voice Design). Kokoro scales speed: `calm` 0.85, `sad` 0.80, `serious` 0.90, `neutral` 1, `happy` 1.10, `excited` 1.20, `angry` 1.25. Any other word is 400. Piper ignores it |
| `sentence_silence` | no | Piper seconds of silence after each sentence, from 0 to 5. An omitted value stays at 0.2. Other speech models return 400 |
| `noise_scale` | no | Piper generator noise, from 0 to 2. 0 keeps the generator steady. An omitted value leaves the voice JSON, or 0.667 when that JSON omits it. Other speech models return 400 |
| `noise_w` | no | Piper phoneme-width noise, from 0 to 2. 0 keeps each phoneme's length steady. An omitted value leaves the voice JSON, or 0.8 when that JSON omits it. Other speech models return 400 |
| `speaker_id` | no | Piper speaker index, from 0 to 4095. An omitted value, or 0, leaves the default speaker. A single-speaker voice leaves it unused. Other speech models return 400 when it is above 0 |
| `phoneme_silence` | no | Map of one character to seconds of silence, from 0 to 5, at most 32 characters. Piper passes it as `--phoneme_silence` (a character already listed in the voice JSON keeps that pause). Kokoro inserts that silence after each matching character. Other speech models return 400 when the map is set |

Response is raw audio bytes (`audio/wav` or `audio/mpeg` depending on
backend), not JSON.

`GET /v1/audio/voices` (optional `?model=<tag>`) lists the voice catalog
across all speech models — always check this before hardcoding a voice id.

## How to Run

```bash
# 1. List available voices (optionally scoped to one model)
curl -s http://localhost:11434/v1/audio/voices
curl -s "http://localhost:11434/v1/audio/voices?model=piper-lessac"

# 2. Piper (CPU, always WAV)
curl -s http://localhost:11434/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"model":"piper-lessac","input":"Hello there.","voice":"amy"}' \
  -o speech.wav

# 3. Remote-tts with emotion (Orpheus)
curl -s http://localhost:11434/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"model":"orpheus","input":"Hello there.","voice":"tara","emotion":"excited"}' \
  -o speech.wav

# 4. Irodori (Japanese caption / Voice Design; sidecar :8088)
curl -s http://localhost:2083/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"model":"irodori","input":"こんにちは。","voice":"none","emotion":"落ち着いた自然な声"}' \
  -o irodori.wav
```

## Pitfalls

- **Arabic Piper voices diacritize first** — libtashkeel runs before phonemization. The model file comes from `backend_paths.tashkeel_model` or `OLLAMA_PIPER_TASHKEEL_MODEL`. When that is unset, Piper looks for `libtashkeel_model.ort` next to its executable. A missing file fails Arabic speech. Other languages leave the path unused.
- **eSpeak Piper voices phonemize from a data directory** — that directory comes from `backend_paths.espeak_data` or `OLLAMA_PIPER_ESPEAK_DATA`. When that is unset, Piper looks for `espeak-ng-data` next to its executable. A missing directory fails that voice. A voice that does not use eSpeak leaves the directory unused.
- **Speech tags can answer on the live socket** — Piper, remote-tts, Kokoro, and a llama-server gen projector accept `GET /v1/realtime`. `conversation.item.create` carries the text and `response.create` speaks it as `response.output_audio.delta`. `pipeline=duplex&voice=<speech>&asr=<whisper or llama ASR>&rate=<Hz>` transcribes a committed clip, asks `model` for a reply, and speaks it in sentences. Each reply includes up to 8 earlier turns from that socket. Thinking is off, and the chat template is rendered. The reply stops after 48 tokens. Duplex also emits `response.status` (`silence_started`, `transcribing`, `generating`, `first_token`, `audio_done`) and `conversation.item.input_audio_transcription.started` for client logs. Earlier turns are prefilled while the next person is speaking. The first words start as soon as the model finishes them — a comma, semicolon, colon, or about 16 characters — while the rest of the reply is still generating. A later stretch waits for about 48 characters. A Piper live socket keeps one ONNX process for the whole call and forwards each PCM chunk as it is synthesized. POST /v1/audio/speech reuses one Piper process for the same voice. When whisper-server 1.9.1 or newer is installed, one CPU process takes every clip for the call. Otherwise Whisper starts when the socket opens, skips timestamp tokens, uses one greedy pass, and the next CLI process starts when synthesis finishes. A load that is still in progress is kept, and the clip is written to it when that load finishes. The clip is written when the turn commits, and it keeps about 200 ms after the last loud sample. With server_vad, transcription starts on the first quiet chunk, and the chat model starts when that transcript is ready. The voice synthesizes during that wait. The audio is sent when the turn commits. When the CLI is used, the next Whisper starts when that synthesis finishes, still during the silence when the voice is already done. Quiet samples at the end of a speech packet count toward the turn's silence. If audio stops arriving, that same silence commits the turn, including on the llama-server proxy. A packet that ends loud starts the full silence. On the duplex socket, transcription starts about 100 ms into that wait. The chat model loads with the socket when the voice does not need that runner. A cancel omits sentences that were not sent. `turn=server_vad` ends the turn when the PCM goes quiet. The same query on the llama-server socket commits from the rate in `session.created`, and a conversation session also gets `response.create`. The probe's `-turn server_vad` streams the clip plus 500 ms of silence and leaves that commit to the server. `-pipeline duplex -tts <speech> -asr <transcriber>` asks `-model` to answer and speaks on commit. `-rate` is the input PCM rate, or the wav rate when omitted. `-voice` is the speaker id. `-turn server_vad` on that probe appends 500 ms of silence and lets the socket commit. A delta whose mime is not PCM is saved as those bytes. The heard text is `transcript` and the spoken answer is `reply`. `-barge` cancels that answer. The threshold is amplitude, from 0 to 1, default 0.02. `response.cancel` stops that reply. Music 3 stays on `POST /v1/audio/speech`.
- **Piper and local Kokoro repeat a short phrase from memory** — 48 characters or fewer, up to 32 clips. A hit sets `X-Speech-Cache: hit`. Serve pre-synthesizes `Okay.`, `Sure.`, `One moment.`, `Let me check.`, `Got it.`, and the duplex hesitation `um` when the voice files are present. `ZEROLLAMA_PIPER_WARM=0` skips Piper. `ZEROLLAMA_KOKORO_WARM=0` skips Kokoro. The Kokoro key also includes language, the requested speed, `emotion`, and `phoneme_silence`. A longer line is synthesized every time. `stream: true` sends the WAV as Piper synthesizes each line when the voice JSON has `audio.sample_rate`. Without that rate the finished WAV is returned. On a remote-tts tag, `stream: true` is forwarded and the upstream body is relayed as it arrives. A buffered remote clip keeps an upstream `X-Audio-Seconds` when the body has no length of its own. A WAV body uses its own length.
- **Piper, remote-tts, and Music 3 are speech routes** — chat, generate,
  embeddings, and transcription return 400. Speak on `POST /v1/audio/speech`.
  Piper and remote-tts also speak text on `GET /v1/realtime`. Music 3 stays
  on the speech POST.
- **Live and whole-clip models do not speak here** — a live-only model is 400
  and stays on `GET /v1/realtime`. A file-audio model without a gen tower is
  400 and stays on `POST /v1/audio/transcriptions`. An LFM3 model that also
  has the gen tower still speaks.
- **`response_format` is not transcoded** — Piper accepts `wav`. A
  llama-server gen projector accepts `wav` or `pcm`. `mp3`, `opus`, `aac`,
  and `flac` are 400 on those backends. `pcm` is
  `audio/pcm;rate=<Hz>;encoding=float;bits=32`. A buffered wav or pcm response
  includes `X-Audio-Seconds`. A stream sends that value as a trailer after the last chunk when the length is known. remote-tts returns the upstream
  `Content-Type`.
- **4096-character input limit** — chunk long text into multiple calls and
  concatenate audio client-side rather than sending one huge `input`.
- **Unknown voice id** — always resolve voices via `GET /v1/audio/voices`
  first; voice catalogs differ per model and guessing IDs 404s or falls
  back to a default you didn't intend. `?model=` on a tag that cannot speak
  returns the same 400 as chat or transcription.
- **Remote-tts backends need the upstream service reachable** — Chatterbox/
  Orpheus/Kokoro (`:8090`) and Irodori (`:8088`) are proxied to separate
  services; a connection error here is a backend availability issue, not a
  request-shape issue.
- **Irodori is Japanese-only** — English input will degrade; do not clone
  voice actors / celebrities without consent.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `speech-to-text` — the inverse operation (audio → text)
- `download-model` — pulling a speech model before use
- `install-multimedia-model` — pick Piper/Kokoro/Chatterbox/… and install it
