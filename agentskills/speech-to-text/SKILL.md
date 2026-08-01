---
name: speech-to-text
description: "Transcribe audio to text via a zerollama server's OpenAI-compatible transcription API (Whisper or multimodal chat models)."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, stt, audio, transcription, whisper, multimodal]
    category: mlops
    related_skills: [zerollama-integration, text-to-speech, download-model]
---

# Speech-to-Text Skill

Transcribe an audio file to text on a [zerollama](https://github.com/GoodSoftware-Group/zerollama)
server via the OpenAI-compatible `/v1/audio/transcriptions` endpoint. Backed
by a Whisper model (e.g. `whisper-base`) or a multimodal chat model with
audio input.

## When to Use

- The user asks to transcribe an audio/voice file to text locally
- Choosing `response_format` for downstream use (plain text vs. structured
  JSON with segments/timestamps)
- Debugging garbled or empty transcription output

## Prerequisites

- zerollama server running (default `http://localhost:11434`)
- A transcription-capable model pulled (`whisper-base`, or a multimodal
  chat model that accepts audio) — check `GET /api/tags`

## API Contract

`POST /v1/audio/transcriptions` — `multipart/form-data`

| Form field | Required | Notes |
|---|---|---|
| `file` | yes | Audio file to transcribe |
| `model` | yes | STT model (e.g. `whisper-base`) or multimodal chat model |
| `language` | no | Optional language hint |
| `prompt` | no | Optional context prepended for transcription |
| `response_format` | no | `json` (default) \| `text` \| `verbose_json` |

Response content type matches `response_format`: `application/json` for
`json`/`verbose_json`, `text/plain` for `text`.

## How to Run

```bash
# Default JSON response
curl -s http://localhost:11434/v1/audio/transcriptions \
  -F file=@recording.wav \
  -F model=whisper-base

# Plain text, with a language hint
curl -s http://localhost:11434/v1/audio/transcriptions \
  -F file=@recording.wav \
  -F model=whisper-base \
  -F language=en \
  -F response_format=text

# Verbose JSON (segments/timestamps, if the backend supports it)
curl -s http://localhost:11434/v1/audio/transcriptions \
  -F file=@recording.wav \
  -F model=whisper-base \
  -F response_format=verbose_json
```

## Pitfalls

- **Wrong `model`** — this endpoint accepts both dedicated STT models
  (`whisper-base`) and multimodal chat models with audio input; check
  `GET /api/tags` and the model's `details.families` if unsure which path
  it takes.
- **Large files / long audio** — no chunking is done server-side; split
  very long recordings client-side if the backend times out.
- **`response_format` support varies by backend** — `verbose_json`
  (segments/timestamps) may not be implemented for every model; fall back
  to `json`/`text` if it errors.
- **`prompt` is context, not an instruction** — it biases transcription
  (e.g. spelling of proper nouns), it does not make the model summarize or
  translate; that requires a separate chat call on the transcript.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `text-to-speech` — the inverse operation (text → audio)
- `download-model` — pulling a transcription model before use
