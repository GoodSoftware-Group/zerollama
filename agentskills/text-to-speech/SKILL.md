---
name: text-to-speech
description: "Synthesize speech audio from text via a zerollama server's OpenAI-compatible speech API (Piper, Chatterbox, Orpheus, Kokoro)."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, tts, audio, speech, piper, chatterbox, orpheus, kokoro]
    category: mlops
    related_skills: [zerollama-integration, speech-to-text, download-model]
---

# Text-to-Speech Skill

Synthesize speech audio from text on a [zerollama](https://github.com/GoodSoftware-Group/zerollama)
server via the OpenAI-compatible `/v1/audio/speech` endpoint. Backends are
either **Piper** (CPU, always WAV) or a **remote-tts** bridge (Chatterbox,
Orpheus, Kokoro), selected by the `model` tag.

## When to Use

- The user asks to generate spoken audio / a voiceover from text locally
- Picking a voice for a given speech model
- Debugging wrong/empty audio output or format mismatches

## Prerequisites

- zerollama server running (default `http://localhost:11434`)
- A speech-capable model pulled (`piper-lessac`, `chatterbox`, `orpheus`,
  `kokoro`, etc.) — check `GET /api/tags`

## API Contract

`POST /v1/audio/speech`

| Field | Required | Notes |
|---|---|---|
| `model` | yes | Speech model tag, e.g. `piper-lessac`, `chatterbox`, `orpheus`, `kokoro` |
| `input` | yes | Text to synthesize, max 4096 Unicode characters |
| `voice` | no | Voice id from `GET /v1/audio/voices` |
| `response_format` | no | `mp3`\|`opus`\|`aac`\|`flac`\|`wav`\|`pcm` — **Piper ignores this and always returns WAV**; remote-tts returns whatever the upstream sends via `Content-Type`, no server-side transcoding |
| `emotion` | no | zerollama extension, remote-tts only (e.g. Orpheus: `excited`, `sad`) |

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
```

## Pitfalls

- **`response_format` silently ignored on Piper** — Piper always returns
  WAV regardless of what you request; only remote-tts backends honor a
  format hint, and even then it's whatever the upstream returns (no
  transcoding).
- **4096-character input limit** — chunk long text into multiple calls and
  concatenate audio client-side rather than sending one huge `input`.
- **Unknown voice id** — always resolve voices via `GET /v1/audio/voices`
  first; voice catalogs differ per model and guessing IDs 404s or falls
  back to a default you didn't intend.
- **Remote-tts backends need the upstream service reachable** — Chatterbox/
  Orpheus/Kokoro are proxied to a separate service; a connection error here
  is a backend availability issue, not a request-shape issue.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `speech-to-text` — the inverse operation (audio → text)
- `download-model` — pulling a speech model before use
