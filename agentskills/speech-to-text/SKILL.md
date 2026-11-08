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
    related_skills: [zerollama-integration, text-to-speech, download-model, install-multimedia-model]
---

# Speech-to-Text Skill

Transcribe an audio file to text on a [zerollama](https://github.com/GoodSoftware-Group/zerollama)
server via the OpenAI-compatible `/v1/audio/transcriptions` endpoint. Backed
by a Whisper model (e.g. `whisper-base`), a Qwen3-ASR tag
(`modality_backends.transcribe: qwen3-asr` with a local weight directory),
a Parakeet or Nemotron-transcribe GGUF on llama-server, or a multimodal chat
model with audio input.

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/v1/audio/transcriptions -d '{}'   # 400/422 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/tags   # 200/400 = route exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

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
| `model` | yes | STT model (`whisper-base`, a Parakeet or Nemotron-transcribe GGUF) or multimodal chat model |
| `language` | no | Whisper code, name, or `auto`. Omitted leaves whisper.cpp at English. `auto` detects the language and still transcribes. An unknown code returns 400 |
| `prompt` | no | Optional context prepended for transcription |
| `response_format` | no | `json` (default) \| `text` \| `srt` \| `verbose_json` \| `vtt` |
| `timestamp_granularities[]` | no | `segment` (default) or `word`. `word` requires `verbose_json` and asks whisper.cpp to split segments on words |
| `stream` | no | `true` on Parakeet or Nemotron-transcribe returns `text/event-stream`. Requires `response_format=json` |
| `include[]` | no | `logprobs` on a Whisper model. Requires `json` or `verbose_json` |

Response content type matches `response_format`: `application/json` for
`json`/`verbose_json`, `text/plain` for `text` and `srt`, `text/vtt` for `vtt`.

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

# English translation. language is the source language.
# A file-audio chat model answers in English. Parakeet and Nemotron-transcribe return 400.
curl -s http://localhost:11434/v1/audio/translations \
  -F file=@recording.wav \
  -F model=whisper-base \
  -F language=es
```

## Pitfalls

- **Whisper, Qwen3-ASR, Parakeet, and Nemotron-transcribe are this route** — those
  tags return 400 on chat, generate, speech, and `GET /v1/realtime`.
- **Qwen3-ASR** — `modality_backends.transcribe: qwen3-asr` keeps one CPU
  `python3 -m runtime.asr.qwen3_serve` process. Point `backend_paths.qwen3_asr`
  (or `OLLAMA_QWEN3_ASR`) at a local weight directory and install `qwen-asr`
  in that interpreter. A Hub id is not downloaded. Missing Python, package, or
  directory falls back to the Whisper CLI. Language codes such as `en` are
  sent as names (`English`). `POST /v1/audio/translations` returns 400.
- **Wrong `model`** — this endpoint accepts both dedicated STT models
  (`whisper-base`) and multimodal chat models with audio input; check
  `GET /api/tags` and the model's `details.families` if unsure which path
  it takes.
- **Large files / long audio** — no chunking is done server-side; split
  very long recordings client-side if the backend times out.
- **`response_format` support varies by backend** — whisper.cpp
  `verbose_json`, `srt`, and `vtt` use segment start and end. Whisper
  `verbose_json` also fills `avg_logprob` and `no_speech_prob` from token
  probabilities, and `tokens` when whisper.cpp wrote token ids. `diarize=true`
  asks whisper.cpp to label a stereo clip. `suppress_nst=true` asks whisper.cpp
  to keep non-speech marks such as `[MUSIC]` out of the text. `no_fallback=true`
  keeps a failed segment at the requested temperature. `temperature_inc` is the
  step added after a failed segment. An omitted value stays at 0.2. `entropy_thold`
  is the token-entropy cutoff. A segment below it fails and whisper.cpp retries.
  An omitted value stays at 2.4. `logprob_thold` is the average log-probability cutoff.
  A segment below it fails when the clip still looks like speech, and whisper.cpp
  retries. An omitted value stays at -1. `no_speech_thold` is the silence cutoff.
  A segment is silence when its no-speech probability is above that value and its
  average log probability is still low. An omitted value stays at 0.6. `beam_size`
  is the number of beam-search candidates when the temperature is 0. A value of 1
  decodes greedily. An omitted value stays at 5. `best_of` is the number of candidates
  when the temperature is above 0. An omitted value stays at 5. `carry_initial_prompt=true`
  keeps the initial prompt in front of every window. An omitted value uses the prompt
  on the first window. `max_len` is the maximum characters in a segment. An omitted
  value, or 0, leaves the cap off. A positive value wraps a longer segment, and
  word timestamps keep that wrap on a word. `split_on_word=true` does the same
  without word timestamps. It requires a positive `max_len`. The wrap waits for a
  token that starts with a space, so a segment can run past `max_len` until that
  token. `word_thold` is the token-timestamp
  probability. A timestamp is kept when its probability is above that value. An
  omitted value stays at 0.01. `max_context` is the cap on earlier text tokens kept
  for the next window. An omitted value, or -1, leaves the library cap. 0 leaves
  earlier text out of the next window. `audio_ctx` is the number of encoder frames in
  each window. An omitted value, or 0, uses the full context. A standard Whisper
  model has 1500 frames. `offset_t` is milliseconds from the start of the clip.
  An omitted value, or 0, starts at the beginning. A start that leaves less than
  100 ms yields an empty transcript. `duration_ms` is the milliseconds to
  transcribe from that start. An omitted value, or 0, continues to the end of
  the clip. A window shorter than 100 ms yields an empty transcript. `suppress_regex`
  is a regular expression matched against each whole token. A match is left out
  of the transcript. An invalid expression fails the transcription. `offset_n` is
  added to each SRT cue number. An omitted value, or 0, starts at 1. VTT cues
  stay unnumbered. `no_timestamps=true` starts the decoder with the no-timestamps
  token and blocks timestamp tokens. Each segment advances 30 seconds. A
  first-release distilled model forces this on. `grammar` is GBNF and `grammar_rule`
  is the rule decoding starts from. Both are required together. Tokens the grammar
  rejects lose `grammar_penalty` logit points. An omitted penalty stays at 100. 0
  leaves those logits unchanged. A grammar uses beam search. A grammar whisper.cpp
  cannot parse fails the transcription. `print_special=true` keeps special tokens,
  such as `<|nospeech|>` and timestamp marks, in the segment text. An omitted
  value leaves them out. Word timestamps and logprobs still leave them out.
  `language=auto` detects the spoken language and still transcribes. `json`,
  `verbose_json`, and `diarized_json` include that code and `language_probability`,
  the softmax score of the detected language from 0 to 1. A translation keeps
  `language` as `en` and omits that score. An omitted language
  leaves whisper.cpp at English. A name such as `english` is sent as its code.
  An unknown code returns 400. `detect_language=true` reports the spoken language
  and skips the transcript, so the text is empty. `json`, `verbose_json`, and `diarized_json` include that
  language and `language_probability`. A request language is left unused. An English-only model has no
  language tokens, so detection fails the transcription. `dtw` is a whisper.cpp
  alignment preset that must match the loaded model: `tiny`, `tiny.en`, `base`,
  `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large.v1`, `large.v2`,
  `large.v3`, or `large.v3.turbo`. A mismatch fails the transcription. Flash
  attention is off for that request, because DTW timestamps require it. Word
  starts use the DTW time. An omitted value leaves flash attention on and keeps
  the token-offset word times. `vad=true` keeps speech and drops the rest.
  The Silero file comes from `backend_paths.vad_model` or `OLLAMA_WHISPER_VAD_MODEL`.
  Segment times are mapped back onto the original clip. No speech leaves the
  transcript empty. A missing model file fails the transcription. `vad_threshold`
  is the Silero probability that starts a speech region. Speech ends when the
  probability stays 0.15 below that value, floored at 0.01, for
  `vad_min_silence_duration_ms`. An omitted
  value stays at 0.5. 0 keeps a frame at any probability. A higher value keeps
  less audio. It requires `vad=true`. `vad_min_speech_duration_ms` leaves out a
  speech region of that many milliseconds or shorter. An omitted value stays at
  250. 0 keeps a region of any length. It requires `vad=true`.
  `vad_min_silence_duration_ms` keeps a shorter pause inside the speech region.
  An omitted value stays at 100. 0 ends a region on the first frame below the
  end threshold. Whisper.cpp then joins neighboring regions when the gap is
  under 200 ms. It requires `vad=true`. `vad_max_speech_duration_s` splits a
  speech region longer than that many seconds. Whisper.cpp uses the whole
  seconds, then subtracts one analysis window and twice the speech padding, so
  the cut lands earlier. The split prefers a silence longer than 98 ms. When no
  such silence exists, the region ends at the limit. An omitted value leaves the
  cap off. A value under 1 second leaves the cap off, so it is rejected. It
  requires `vad=true`. `vad_speech_pad_ms` grows each speech region by that
  many milliseconds on each side, stopping at the clip edges. When the gap
  between two regions is shorter than twice this value, the regions meet in the
  middle of the gap. An omitted value stays at 30. 0 adds no padding. The same
  padding is subtracted twice from `vad_max_speech_duration_s`. It requires
  `vad=true`. `vad_samples_overlap` is the extra seconds the decoder hears
  after each speech region except the last, stopping at the clip edge. An
  omitted value stays at 0.1. 0 copies the region as detected. Whisper.cpp still
  inserts 100 ms of silence between regions. It requires `vad=true`.
  `tinydiarize=true` lets a model trained for tinydiarize end a segment with
  ` [SPEAKER_TURN]`. `verbose_json` sets `speaker_turn_next`. It cannot be
  combined with `diarize` or `diarized_json`. Other models return 400.
  The cue text keeps `(speaker 0)`,
  `(speaker 1)`, or `(speaker ?)`, and `verbose_json` sets `speaker`. A mono
  clip is unchanged. `response_format=diarized_json` passes `--diarize` and
  puts the label in `speaker`. Segment `text` is the spoken words. Other models
  return 400. `duration`
  is the WAV length when the upload is a WAV, otherwise the last cue end. Other backends
  return one cue. Its end is the WAV length when that length is known, and time 0 otherwise. Parakeet and Nemotron-transcribe `json` and
  `verbose_json` include `usage` token counts. `input_tokens_details.cached_tokens`
  is copied when llama-server sent it. A WAV upload adds
  `usage.seconds` on `json` and `verbose_json` for Whisper, a chat model
  that hears, and Parakeet or Nemotron-transcribe. Fall back to `json` or `text`
  if a backend errors.
- **`prompt` is context, not an instruction** — it biases transcription
  (e.g. spelling of proper nouns), it does not make the model summarize or
  translate; that requires a separate chat call on the transcript. whisper.cpp
  receives `prompt` as `--prompt` and `temperature` as `--temperature`.
  llama-server ASR receives both on its transcription form. A file-audio chat
  model uses that temperature, and an omitted value stays 0. `max_tokens` caps
  llama-server ASR and a chat model that hears the clip. Whisper returns 400.

## Realtime

`GET /v1/realtime?model=<tag>&intent=transcription` is a WebSocket to llama-server
for `voxtral_rt` and `lfm3a` (LFM3-Audio, and LFM3-Omni with `lfm3vl`).
Send `input_audio_buffer.append` events with base64 PCM16 mono. The sample rate
is `audio.input.format.rate` on `session.created`. Omit `intent` for an LFM3
conversation. `PARAMETER main_gpu 1` keeps that model on one GPU.

`./scripts/audio/fetch_mtmd_audio.sh <pack> <dir>` downloads a text GGUF and its
mmproj (`--list` for the pack names). `go run ./scripts/audio/realtime_probe
-model <tag> -wav clip.wav -lang en` resamples to the session rate, sets the transcription language, and prints
`first_text_ms`. An empty `-intent` is a conversation: the probe sends
`response.create` after commit, `-text` adds a text turn (no wav required),
`-modalities audio` asks the model to speak, `-out reply.wav` saves that PCM,
`-instructions` and `-tools` (a JSON array or a path) are `session.update` before the first turn, `-tool-choice required` and `-max-output-tokens 16` ride on that update, tool calls are listed as `tool_calls`, `-tool-output` sends that text back as `function_call_output` and waits for the next reply, `-follow` sends another user message after that reply, `-follow-wav` speaks another clip there (and `-follow` text goes out before that clip),
and `-barge` sends `response.cancel` on the first reply byte. A gen projector
(`lfm3a_gen`, `qwen3tts_gen`, `pockettts_gen`, `soprano`, `drama3_gen`,
`chatterbox`) also speaks on `POST /v1/audio/speech`. `-mode speech -text "hello" -voice female -lang en -format pcm` times that route and sets `response_format`; the speech default omits it. `output_rate` is the `rate` in the content type, or the WAV fmt rate when the body is a WAV. `seconds` is that clip's length, or `X-Audio-Seconds` from the response header or trailer when the server sent one. A pcm `-out` file is a float32 WAV at that rate; without a rate the raw body is saved. `-speaker-ref clip.wav` sends a clone clip and is not combined with `-voice`. `-stream` records `first_audio_ms` while the clip is still arriving. On `-mode transcribe`, `-stream` posts `stream=true` and reads `transcript.text.delta` events. `-seed 7` and `-max-tokens 32` set the TTS seed and frame cap. `-top-k 8 -top-p 0.9 -repeat-penalty 1.1` set the acoustic sampler; omit them to leave top_k 0, top_p 0, and repeat_penalty 1.05. `GET /v1/audio/voices` includes `sample_rate` for the gen projector. `-mode transcribe -prompt "names and places" -temperature 0.2` times `POST /v1/audio/transcriptions`. `-format srt` or `-format vtt` requests subtitles, and `-format verbose_json -words` asks for word times. `-logprobs` sends `include[]=logprobs`. `-out` writes that body. `-mode translate` posts to `/v1/audio/translations`. The live socket is PCM only; images for
LFM3-Omni, Qwen3-Omni, and Phi-4 go on a chat turn.

File upload stays on `POST /v1/audio/transcriptions`. That path also serves
offline mtmd audio: Voxtral Small (`voxtral`), Qwen2.5-Omni (`qwen2.5o`),
Qwen3-Omni (`qwen3a` with `qwen3vl_merger`), Ultravox (`ultravox`), and Phi-4
multimodal (`phi4`). Voxtral Realtime does not. On two GPUs, leaving
`main_gpu` unset layer-splits the model across both, which is what Voxtral
Small Q8, Qwen3-Omni Q8, and Ultravox 70B Q4 need.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `text-to-speech` — the inverse operation (text → audio)
- `download-model` — pulling a transcription model before use
