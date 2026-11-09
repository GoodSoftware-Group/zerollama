# Quick voice stack

Speech-to-text, a short reply, and speech, on one duplex socket. The server already runs that path. This page is the small set of models and the one client setting that keep the reply inside the quiet wait.

## Models

| Step | Tag | Where it runs |
|------|-----|----------------|
| Hear | `whisper-base` | CPU. Whisper GGML, English base |
| Reply | a small chat tag already loaded | GPU or CPU, through the normal completion runner |
| Speak | `piper-lessac` | CPU. Piper ONNX. One process for the whole call |

Use a chat tag that is already local. A 0.5B is the size that finishes a short clause inside the quiet wait. `qwen2.5:0.5b` and an MLX tag such as `qwen2.5-0.5b-mlx:4bit` both go through the same completion runner. The reply stops after 48 tokens. The first words are spoken at a comma, semicolon, colon, or about 16 characters, while the rest of the reply is still generating.

Kokoro, Qwen3-ASR, and a llama-server speech projector are the heavier voices and recognizers. Parakeet is a separate GGUF on llama-server, and loading it unloads the resident chat model. Leave those off this stack.

Piper warms the acknowledgement clips when the ONNX is present: `Okay.`, `Sure.`, `One moment.`, `Let me check.`, `Got it.`, and the duplex hesitation `um`. A hit is replayed from memory. `ZEROLLAMA_PIPER_WARM=0` skips that warm.

## Socket

```
GET /v1/realtime?model=<chat tag>&pipeline=duplex&asr=whisper-base&voice=piper-lessac&rate=16000&turn=server_vad
```

`model` is the chat tag. `voice` and `asr` are required. `rate` is the input PCM16 rate in hertz, from 1 to 192000. Send little-endian mono PCM16 as base64 on `input_audio_buffer.append`. With `turn=server_vad` the server commits the turn; the client does not send `input_audio_buffer.commit`.

`turn=server_vad` on the URL uses a 300 ms quiet wait on duplex. A longer or shorter wait is a `session.update` after the socket opens:

```json
{
  "type": "session.update",
  "session": {
    "audio": {
      "input": {
        "turn_detection": {
          "type": "server_vad",
          "silence_duration_ms": 500
        }
      }
    }
  }
}
```

`silence_duration_ms` is 0 through 10000. 300 still catches a short pause. The server answers with `session.updated`.

Spoken audio comes back as `response.output_audio.delta`. `delta` is base64 PCM, `mime` is `audio/pcm`, and `rate` is the voice rate. `response.output_audio.done` ends the clip.

Duplex also streams `response.status` milestones for clients and logs: `silence_started` (server VAD quiet wait began; includes `silence_duration_ms`), `transcribing` (ASR started; also `conversation.item.input_audio_transcription.started`), `generating` (chat prompt submitted), `first_token` (first model content), and `audio_done` (final PCM flushed; pairs with `response.output_audio.done`).

**Hygiene smoke** (rejects replies that speak `Assistant:` / ChatML role leaks; optional audible TTS + full duplex):

```bash
./scripts/smoke/duplex_voice_hygiene.sh
RUN_E2E_DUPLEX_TTS=1 ./scripts/smoke/duplex_voice_hygiene.sh
RUN_E2E_DUPLEX_FULL=1 ./scripts/smoke/duplex_voice_hygiene.sh
```

## What the wait is made of

These timers are the ones the socket uses. They are not a measured end-to-end run.

- The turn commits after `silence_duration_ms` of quiet audio. The default is 300. The update above sets 500.
- Transcription of a loud packet with no further audio starts about 100 ms into that wait. The chat model starts when the transcript is ready, and Piper synthesizes during the rest of the wait. Audio is sent when the turn commits.
- When the first clause is ready at commit, that audio plays immediately. When it is not, a cached `um` in the same voice holds the floor, then one `One moment.` if the opening is still missing. The model's opening fades in over that hold and is the only text kept in history. A leading model filler such as `Well,` is spoken in the same phrase as the words that follow.
- The clip keeps about 200 ms of quiet audio after the last loud sample.
- A transcriber or chat model that is still loading on the first turn misses this window. Later turns reuse the loaded chat model and, when `whisper-server` is up, the same Whisper process.

A warm small chat model finishes the first clause inside that wait. A cold weight load does not.

## Install

The speech tags are manifests, not a `zerollama pull` of a GGUF. On the Linux host that keeps speech files on the models volume:

```bash
./scripts/install_speech_backends.sh
./scripts/register_speech_models.sh
```

`install_speech_backends.sh` downloads the Linux x86_64 Piper binary, the Whisper.cpp 1.9.1 CLI, `ggml-base.en.bin`, and the Lessac ONNX into `SPEECH_ROOT`. The default is `/mnt/ollama_img/speech`. The registered manifests point `backend_paths` at that same directory (`modelfiles/piper-lessac/config.json`, `modelfiles/whisper-base/config.json`). Override `SPEECH_ROOT` only when the manifest paths are edited to match.

That script installs the Whisper CLI (`whisper` / `whisper-cli`). A duplex call keeps one process for every clip when `whisper-server` 1.9.1 or newer is on `PATH`, or set as `OLLAMA_WHISPER_SERVER`. An older server treats the posted WAV as a filename and the call falls back to starting the CLI per clip. `OLLAMA_WHISPER_BIN` is that CLI (default `whisper`). `OLLAMA_PIPER_BIN` is Piper (default `piper`).

The chat tag is separate. Pull or register it the same way as any other completion model, and name it in `model=`.

On a Mac the Linux tarballs in `install_speech_backends.sh` are the wrong binaries. Install Piper and `whisper-server` for the host, put the GGML and the ONNX on disk, and point `backend_paths.whisper_model`, `piper_model`, and `piper_config` at those files before registering.

There is no Raspberry Pi image. The same three processes are the small-CPU stack when they fit. `zerollama serve --edge` is a thin chat daemon and does not add this voice path.
