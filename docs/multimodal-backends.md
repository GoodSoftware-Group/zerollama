# Optional multimodal backends

Ollama supports OpenAI-compatible routes for images (`/v1/images/*`), speech-to-text (`/v1/audio/transcriptions`), and text-to-speech (`/v1/audio/speech`). Beyond the built-in MLX image pipeline and multimodal audio-in-chat STT, you can select **subprocess backends** per model using manifest metadata.

For **video in chat** (VLM), see **[video-understanding.md](./video-understanding.md)** for goals, merge semantics, and security notes; this page lists env vars and manifest keys.

### Why both environment variables and manifest fields?

- **Env** is how operators set **fleet-wide guardrails** (max bytes, timeouts, default sampling) without republishing models.
- **Manifest (`config.json`)** is how **model authors** ship per-checkpoint recommendations (fps/stride/max frames) that match how the model was evaluated or documented.

Merging env + manifest avoids forcing every user to edit JSON for safety caps, while still letting a published model describe its preferred native sampling. See **ResolveVideoPolicy** in `server/modality/policy.go` and [video-understanding.md](./video-understanding.md).

## Manifest fields

In the model `config.json` (same layer as other `model.ConfigV2` fields):

- **`modality_backends`**: map of modality key → driver name.
  - `image`: `mlx-imagegen` (default, implicit), `external-image` (stable-diffusion.cpp; [sd-vulkan-a380.md](./sd-vulkan-a380.md)), `openvino-image` (OpenVINO GenAI; [sd-openvino-a380.md](./sd-openvino-a380.md)), or `comfyui` (proxy to a running ComfyUI server for edit/img2img/ControlNet/LoRA — see [comfyui-image-backend.md](./comfyui-image-backend.md)).
  - `transcribe`: `whisper` (whisper.cpp-style CLI), `qwen3-asr` (resident Qwen3-ASR via `qwen-asr`; Whisper CLI fallback), `llama-asr` (Parakeet or Nemotron-transcribe GGUF on llama-server), or omit for multimodal LLM audio models.
  - `speech`: `piper` (CPU ONNX TTS), `kokoro` (local Kokoro ONNX via one CPU `python3 -m runtime.tts.kokoro_serve` process; 24000 Hz), `remote-tts` (Chatterbox/Orpheus/Kokoro/Irodori HTTP — the registered `kokoro:latest` tag stays here), or `music3` (MiniMax Music 3 via mlx-audio / later music-cli). **Why music3 is separate:** songs are minutes of DiT, not Piper phonemes; Comfy’s port is GPL so it is not a backend. `music3` returns **202 JSON** on `/v1/audio/speech` (same job as `POST /v1/audio/generations`) — [music-c.md](./music-c.md). Local `kokoro` is not the optional llama-server `LLAMA_BUILD_KOKORO` GGUF route.
  - `video_understanding` (VLM): `native` (default) samples frames with **ffmpeg** and feeds them like images, or `sglang` to forward OpenAI `POST /v1/chat/completions` to a SGLang server when `OLLAMA_SGLANG_URL` is set.
  - `video_generation` (T2V / TI2V): `wan` runs [Wan](../scripts/video/wan_video_generate.py) via the training job queue; capability `video_gen`. `rife` is reserved (not shipped). Keyframe uploads: [media-uploads.md](./media-uploads.md); Wan semantics: [wan-t2v.md](./wan-t2v.md).
  - `mesh_generation` (**lab, not wired to HTTP yet**): TRELLIS.2 image→PBR GLB on astra (CUDA) or Darwin (Metal via trellis-mac); see [mesh-gen.md](./mesh-gen.md). Capability name reserved: `mesh_gen`.
- **`video_sampling`** (optional, native path only): per-model overrides for ffmpeg—`mode` (`fps` or `stride`), `fps`, `stride`, `max_frames`. Omitted fields use server env defaults (see below).
- **`tokens_per_image`** (optional): vision-token budget **per raster frame** for **context preflight** only (default `768` until projector metadata is wired through).
- **`vision_patch_size`** / **`vision_spatial_merge_size`** (optional): tune native **`grid_thw`** estimates on ffmpeg-expanded clips (Qwen3-VL defaults: 14 / 2).
- **`backend_paths`**: map of string keys → filesystem paths for weights/config:
  - `whisper_model`: GGML weights for Whisper.
  - `qwen3_asr`: local directory of Qwen3-ASR weights (never a Hub repo id). Used when `modality_backends.transcribe` is `qwen3-asr`.
  - `piper_model`: Piper ONNX file.
  - `tashkeel_model`: libtashkeel ONNX file. An Arabic voice uses it to add diacritics before phonemization.
  - `espeak_data`: espeak-ng data directory. An eSpeak voice uses it to turn text into phonemes.
  - `piper_config`: optional Piper JSON config.
  - `piper_voice_<name>`: optional per-voice ONNX (e.g. `piper_voice_alloy`); `<name>` is the OpenAI `voice` field with non-alphanumeric characters stripped. If set, it overrides `piper_model` for that request.
  - `kokoro_model`: local Kokoro ONNX file (`kokoro-v1.0.onnx`) for `speech=kokoro`. Must already exist; never a Hub id. With `kokoro_voices` set and no `piper_model`, an empty `modality_backends.speech` selects kokoro. `OLLAMA_KOKORO_MODEL` fills a missing path only after the tag is already kokoro.
  - `kokoro_voices`: local `voices-v1.0.bin` numpy archive for that backend. Voice ids are the `*.npy` member names. `af_` / `am_` use `en-us` and `bf_` / `bm_` use `en-gb`. An optional `lang` of `en`, `en-us`, `en-gb`, or `en-uk` overrides that prefix, including on the duplex `language` query. Other languages and the v1.1 prefixes (`zf_`, `jf_`, `ef_`, `ff_`, `hf_`, `if_`, `pf_`, and their `m_` forms) are 400 and are left out of the voice list. Short phrases share Piper's memory cache. `emotion` scales speed. `phoneme_silence` inserts silence after those characters.
  - `tts_url`: base URL or full `…/v1/audio/speech` for `remote-tts` (overrides fleet `OLLAMA_TTS_URL`).
  - `tts_upstream_model`: model id sent to the remote server (defaults to the local tag name).
  - `tts_default_voice`: used when the client omits `voice`.
  - `tts_voices_file`: JSON catalog (`[{id,name,…}]` or `{"voices":[…]}`) listed by `GET /v1/audio/voices`.
  - `tts_ref_audio`: path to a clone reference clip; sent as `X-TTS-Ref-Audio` to the remote server (and as `irodori.ref_wav` when `tts_flavor=irodori`).
  - `tts_flavor`: set to `irodori` so Go nests `emotion` → `irodori.caption` for Irodori-TTS-Server.
  - `music3_mlx_model`: mlx-community (or local) MiniMax Music 3 pack for `speech=music3`.
  - `wan_repo`, `wan_ckpt_dir`: Wan upstream tree and checkpoint directory (weights installed outside Ollama blobs).
  - `wan_gguf_path` (optional): GGUF weights when safetensors+offload OOM on 16 GB.
  - `sd_cli`, `sd_model`: [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (Vulkan; [sd-vulkan-a380.md](./sd-vulkan-a380.md))
  - `ov_model_dir`, `ov_python`, `external_image_bin`: OpenVINO GenAI ([sd-openvino-a380.md](./sd-openvino-a380.md))
  - `comfy_workflow_dir`: directory of ComfyUI API-format workflow template JSON files (one per named workflow, e.g. `t2i.json`, `edit.json`) for `modality_backends.image=comfyui`.
  - `comfy_default_workflow` (optional): workflow name used when the request omits `options.workflow`; falls back to `t2i`.
- **`image_generation`** (optional): per-model defaults for `external-image` / `openvino-image` — `width`, `height`, `steps`, `cfg_scale`, `sampler`, `diffusion_fa`, `vae_on_cpu`, `vae_tiling`. **Why manifest not env:** Intel ANV needs `diffusion_fa: true` for sd.cpp on Arc; SD-Turbo needs low steps/cfg — these vary per tag, not fleet-wide.

Example (SD 1.5 Vulkan on Arc):

```json
{
  "capabilities": ["image"],
  "modality_backends": { "image": "external-image" },
  "concurrency_groups": ["imagegen"],
  "backend_paths": {
    "sd_cli": "/usr/share/zerollama/sd-cpp/bin/sd-cli",
    "sd_model": "/usr/share/zerollama/sd-cpp/models/stable-diffusion-v1-5-Q4_0.gguf"
  },
  "image_generation": {
    "width": 512,
    "height": 512,
    "steps": 20,
    "cfg_scale": 7.0,
    "diffusion_fa": true,
    "vae_on_cpu": true
  }
}
```

Example (OpenVINO — per-model wrapper override):

```json
{
  "capabilities": ["image"],
  "modality_backends": { "image": "openvino-image" },
  "backend_paths": {
    "ov_model_dir": "/usr/share/zerollama/openvino-genai/models/sd15-int8-ov",
    "external_image_bin": "/usr/lib/ollama-zerollama/ov_external_image.sh"
  },
  "image_generation": { "width": 512, "height": 512, "steps": 20 }
}
```

**Why `external_image_bin` in manifest:** Vulkan SD uses fleet env `OLLAMA_EXTERNAL_IMAGE_BIN` → `sd_external_image.sh`; OpenVINO tags need a different wrapper without changing global env — manifest wins over env when set.

Example (Piper TTS):

```json
{
  "capabilities": ["speech"],
  "modality_backends": { "speech": "piper" },
  "backend_paths": { "piper_model": "/path/to/en_US-lessac-medium.onnx" }
}
```

Example (remote GPU TTS — Chatterbox / Orpheus / Kokoro):

```json
{
  "capabilities": ["speech"],
  "modality_backends": { "speech": "remote-tts" },
  "backend_paths": {
    "tts_upstream_model": "chatterbox",
    "tts_default_voice": "puppet",
    "tts_voices_file": "/mnt/ollama_img/speech/voices/chatterbox.json",
    "tts_ref_audio": "/mnt/ollama_img/speech/refs/puppet.wav"
  }
}
```

Set the remote server once via fleet env (or per-model `tts_url`):

```bash
# systemd drop-in / environment
OLLAMA_TTS_URL=http://cozmic:8090   # or http://127.0.0.1:8090 for local sidecar
```

On the GPU host, run the OpenAI-compatible sidecar:

```bash
TTS_ENGINE=chatterbox TTS_PORT=8090 python3 scripts/tts_remote_server.py
# or TTS_ENGINE=orpheus / kokoro / echo (smoke)
# Kokoro on this sidecar scales emotion the same way as local speech=kokoro.
```

### Irodori (Japanese voice cloning)

[Irodori-TTS](https://github.com/Aratako/Irodori-TTS) is a **Japanese-only** Flow Matching TTS with zero-shot clone + caption Voice Design. Zerollama fronts the official [Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server) on **`:8088`** (keep Chatterbox on `:8090`).

**Consent:** do not clone voice actors / celebrities / private individuals without explicit permission.

```bash
./scripts/speech/install_irodori_tts.sh
sudo cp scripts/systemd/irodori-tts.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now irodori-tts
OLLAMA_MODELS=/mnt/ollama_img/models ./scripts/register_speech_models.sh
# Drop reference clips into /mnt/ollama_img/speech/irodori/voices/*.wav (stem = voice id)
curl -sS http://127.0.0.1:2083/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"model":"irodori","input":"こんにちは。","voice":"none","emotion":"落ち着いた自然な声"}' \
  -o /tmp/irodori-smoke.wav
```

Go maps `emotion` → `irodori.caption` and `backend_paths.tts_ref_audio` → `irodori.ref_wav` when `tts_flavor=irodori` / `tts_upstream_model=irodori-tts`.

Clients call the same OpenAI route; optional zerollama extension `emotion` is forwarded for expressive engines:

```bash
curl -s localhost:11434/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"model":"orpheus","input":"Hello there.","voice":"tara","emotion":"excited"}' \
  -o speech.wav
curl -s localhost:11434/v1/audio/voices | jq .
```

Example (Whisper STT):

```json
{
  "modality_backends": { "transcribe": "whisper" },
  "backend_paths": { "whisper_model": "/path/to/ggml-base.bin" }
}
```

`POST /v1/audio/translations` is the same request with `--translate`, so the text is English and `verbose_json` sets `task` to `translate`. `language` on that route is the source language. Parakeet and Nemotron-transcribe return 400. A file-audio chat model is prompted to answer in English. `prompt` and `temperature` on `POST /v1/audio/transcriptions` are passed to that CLI as `--prompt` and `--temperature`. An omitted temperature stays at the whisper.cpp default of 0. A file-audio chat model receives that same temperature, and an omitted value stays 0. `max_tokens` caps llama-server ASR and a chat model that hears the clip. An omitted value leaves the cap off. Whisper returns 400. `response_format=verbose_json` also passes `-oj` and fills each segment's `start` and `end` from the JSON offsets. `srt` and `vtt` use those same offsets. `suppress_nst=true` passes `--suppress-nst`, which keeps marks such as `[MUSIC]` and `[BLANK_AUDIO]` out of the text. Other models return 400. `no_fallback=true` passes `--no-fallback`, so a failed segment stays at the requested temperature. Other models return 400. `temperature_inc` passes `--temperature-inc`, the step added after a failed segment. An omitted value stays at 0.2. Other models return 400. `entropy_thold` passes `--entropy-thold`. A segment whose token entropy is below that value fails, and whisper.cpp retries. An omitted value stays at 2.4. Other models return 400. `logprob_thold` passes `--logprob-thold`. A segment whose average log probability is below that value fails when the clip still looks like speech, and whisper.cpp retries. An omitted value stays at -1. Other models return 400. `no_speech_thold` passes `--no-speech-thold`. A segment is silence when its no-speech probability is above that value and its average log probability is below `logprob_thold`. An omitted value stays at 0.6. Other models return 400. `beam_size` passes `--beam-size`, the number of beam-search candidates when the temperature is 0. A value of 1 decodes greedily. An omitted value stays at 5. Other models return 400. `best_of` passes `--best-of`, the number of candidates when the temperature is above 0. An omitted value stays at 5. Other models return 400. `carry_initial_prompt=true` passes `--carry-initial-prompt`, so the initial prompt stays in front of every window. An omitted value uses the prompt on the first window. Other models return 400. `max_len` passes `--max-len`, the maximum characters in a segment. An omitted value, or 0, leaves the cap off. A positive value wraps a longer segment, and word timestamps keep that wrap on a word. `split_on_word=true` does the same without word timestamps. It requires a positive `max_len`. The wrap waits for a token that starts with a space, so a segment can run past `max_len` until that token. Other models return 400. `word_thold` passes `--word-thold`. A token timestamp is kept when its probability is above that value. An omitted value stays at 0.01. Other models return 400. `max_context` passes `--max-context`, the cap on earlier text tokens kept for the next window. An omitted value, or -1, leaves the library cap of 16384, which whisper.cpp also limits to half the model's text context. 0 leaves earlier text out of the next window. Other models return 400. `audio_ctx` passes `--audio-ctx`, the number of encoder frames in each window. An omitted value, or 0, uses the full context. A standard Whisper model has 1500 frames. A value above the loaded model's audio context fails the transcription. Other models return 400. `offset_t` passes `--offset-t`, milliseconds from the start of the clip. An omitted value, or 0, starts at the beginning. A start that leaves less than 100 ms yields an empty transcript. Other models return 400. `duration_ms` passes `--duration`, the milliseconds to transcribe from that start. An omitted value, or 0, continues to the end of the clip. A window shorter than 100 ms yields an empty transcript. Other models return 400. `suppress_regex` passes `--suppress-regex`, a regular expression matched against each whole token. A match is left out of the transcript. An invalid expression fails the transcription. Other models return 400. `offset_n` is added to each SRT cue number. An omitted value, or 0, starts at 1. VTT cues stay unnumbered. Other models return 400. `no_timestamps=true` passes `--no-timestamps`. The decoder starts with the no-timestamps token and cannot emit timestamp tokens. Each segment advances 30 seconds. A first-release distilled model forces this on. Other models return 400. `grammar` and `grammar_rule` pass `--grammar` and `--grammar-rule`. The grammar is GBNF, and decoding starts at that rule. `grammar_penalty` passes `--grammar-penalty`, the logit points subtracted from each token the grammar rejects. An omitted value stays at 100. 0 leaves those logits unchanged. A grammar uses beam search. A grammar whisper.cpp cannot parse fails the transcription. Other models return 400. `print_special=true` passes `--print-special`. Special tokens, such as `<|nospeech|>` and timestamp marks, stay in the segment text. An omitted value leaves them out. Word timestamps and logprobs still leave them out. Other models return 400. `language=auto` passes `-l auto`. Whisper.cpp detects the spoken language and still transcribes. `json`, `verbose_json`, and `diarized_json` include that code and `language_probability`, the softmax score of the detected language from 0 to 1. A translation keeps `language` as `en` and omits that score. An omitted language leaves whisper.cpp at English. A language name such as `english` is sent as its code. An unknown code returns 400. `detect_language=true` passes `--detect-language`. Whisper.cpp reports the spoken language and does not decode a transcript, so the text is empty. `json`, `verbose_json`, and `diarized_json` include that language and `language_probability`. A request language is left unused. An English-only model has no language tokens, so detection fails the transcription. Other models return 400. `dtw` passes `--dtw` and `--no-flash-attn`. The preset is `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large.v1`, `large.v2`, `large.v3`, or `large.v3.turbo`, and it must match the loaded model. A mismatch fails the transcription. Flash attention is off for that request, because DTW timestamps require it. Word starts use the DTW time. An omitted value leaves flash attention on and keeps the token-offset word times. Other models return 400. `vad=true` passes `--vad` and `--vad-model`. The Silero file comes from `backend_paths.vad_model` or `OLLAMA_WHISPER_VAD_MODEL`. VAD keeps speech and drops the rest. Segment times are mapped back onto the original clip. No speech leaves the transcript empty. A missing model file fails the transcription. `vad_threshold` passes `--vad-threshold`. Speech starts when the Silero probability is at least that value. It ends when the probability stays 0.15 below that value, floored at 0.01, for `vad_min_silence_duration_ms`. An omitted value stays at 0.5. 0 keeps a frame at any probability. A higher value keeps less audio. It requires `vad=true`. `vad_min_speech_duration_ms` passes `--vad-min-speech-duration-ms`. A speech region of this many milliseconds or shorter is left out. An omitted value stays at 250. 0 keeps a region of any length. It requires `vad=true`. `vad_min_silence_duration_ms` passes `--vad-min-silence-duration-ms`. A shorter pause stays inside the speech region. An omitted value stays at 100. 0 ends a region on the first frame below the end threshold. Whisper.cpp then joins neighboring regions when the gap is under 200 ms. It requires `vad=true`. `vad_max_speech_duration_s` passes `--vad-max-speech-duration-s`. A speech region longer than this many seconds is split. Whisper.cpp uses the whole seconds, then subtracts one analysis window and twice the speech padding, so the cut lands earlier. The split prefers a silence longer than 98 ms. When no such silence exists, the region ends at the limit. An omitted value leaves the cap off. A value under 1 second leaves the cap off, so it is rejected. It requires `vad=true`. `vad_speech_pad_ms` passes `--vad-speech-pad-ms`. Each speech region grows by this many milliseconds on each side, stopping at the clip edges. When the gap between two regions is shorter than twice this value, the regions meet in the middle of the gap. An omitted value stays at 30. 0 adds no padding. The same padding is subtracted twice from `vad_max_speech_duration_s`. It requires `vad=true`. `vad_samples_overlap` passes `--vad-samples-overlap`. Except for the last region, the decoder hears this many extra seconds after the region, stopping at the clip edge. An omitted value stays at 0.1. 0 copies the region as detected. Whisper.cpp still inserts 100 ms of silence between regions. It requires `vad=true`. `tinydiarize=true` passes `--tinydiarize`. A model trained for tinydiarize can end a segment with ` [SPEAKER_TURN]`, and `verbose_json` sets `speaker_turn_next`. The text file whisper.cpp writes omits that mark, so the response is read from the JSON. It cannot be combined with `diarize` or `diarized_json`. Other models return 400. `diarize=true` passes `--diarize`. On a stereo clip, whisper.cpp prefixes each cue with `(speaker 0)`, `(speaker 1)`, or `(speaker ?)`, and `verbose_json` copies that label to `speaker`. `response_format=diarized_json` passes `--diarize` as well. Each segment's `text` is the spoken words, and `speaker` is `0`, `1`, or `?`. A mono clip is unchanged. Parakeet, Nemotron-transcribe, and a chat model return 400. Other backends return one cue. Its end is the WAV length when that length is known, and time 0 otherwise.

### Parakeet and Nemotron-transcribe (llama-server)

File transcription for these families goes to the model's llama-server `POST /v1/audio/transcriptions`. The encoder graphs are already in the vendored llama.cpp pin (`tools/mtmd/models/parakeet.cpp`, `tools/mtmd/models/nemotron-transcribe.cpp`). Nemotron-transcribe fuses Streaming Sortformer into the ASR states. The HTTP body is the transcript text. It does not include speaker-turn labels.

Create from the GGUF. A `general.architecture` or `clip.projector_type` of `parakeet` or `nemotron_transcribe` is enough. `modality_backends.transcribe=llama-asr` forces the same route when the architecture string differs.

```bash
zerollama create parakeet-tdt -f - <<'EOF'
FROM /models/parakeet-tdt-0.6b.gguf
EOF
```

Same endpoint as Whisper:

```bash
curl -s localhost:11434/v1/audio/transcriptions \
  -F file=@meeting.wav \
  -F model=parakeet-tdt \
  -F language=en
```

Chat, generate, embeddings, speech, and `GET /v1/realtime` return 400 for these models. llama-server must be built (`LLAMA_SERVER_BIN`, or `./scripts/build/build_llama_server.sh`). `ZEROLLAMA_LLAMA_SERVER=0` refuses these loads. On a 16 GiB card keep the 0.6B GGUF. The resident chat model still owns the card. The broker unloads it when this model loads.

### Realtime WebSocket

`GET /v1/realtime` is a WebSocket proxy onto that model's llama-server socket. The runner stays loaded until the socket closes.

```bash
# Text GGUF + mmproj. Does not start a server.
./scripts/audio/fetch_mtmd_audio.sh --list
./scripts/audio/fetch_mtmd_audio.sh voxtral-realtime ./models/voxtral-realtime
zerollama create voxtral-realtime -f ./models/voxtral-realtime/Modelfile

# Resamples the wav to session.created's rate and prints first-text latency.
go run ./scripts/audio/realtime_probe -model voxtral-realtime -wav clip.wav
# -turn server_vad appends 500 ms of silence and does not commit. The server does.
go run ./scripts/audio/realtime_probe -model voxtral-realtime -wav clip.wav -turn server_vad -lang en
# Duplex replies when the clip is committed. -rate overrides the wav rate.
go run ./scripts/audio/realtime_probe -model llama3.2 -pipeline duplex -tts piper -asr whisper -wav clip.wav
# Duplex with server_vad: 500 ms of silence, and the socket commits.
go run ./scripts/audio/realtime_probe -model llama3.2 -pipeline duplex -tts piper -asr whisper -wav clip.wav -turn server_vad
# LFM3 conversation: commit does not answer; the probe then sends response.create.
go run ./scripts/audio/realtime_probe -model lfm3-audio -intent "" -text "what did I say" -modalities audio -out reply.wav -wav clip.wav
# -barge sends response.cancel on the first reply byte and reports cancel_ms.
go run ./scripts/audio/realtime_probe -mode transcribe -model voxtral-small -wav clip.wav
```

`intent=transcription` is live ASR. This pin's streaming encoders are `voxtral_rt` (Voxtral Mini 4B Realtime, live transcription only) and `lfm3a` (LFM3-Audio, and LFM3-Omni when the mmproj also has `lfm3vl`). Leave `intent` off for an LFM3 conversation. LFM3-Omni is launched with `--reasoning off`. A gen projector (`lfm3a_gen`, `qwen3tts_gen`, `pockettts_gen`, `soprano`, `drama3_gen`, `chatterbox`) also serves `POST /v1/audio/speech`, which llama-server speaks on `POST /tts` and returns wav. `GET /v1/audio/voices?model=` lists that file's `voice.*` tensors. LFM3-Audio's ids are `female` and `male`. Optional `lang` is forwarded to the TTS template. A base64 `speaker_ref` clones a voice on `qwen3tts_gen` and `pockettts_gen` and cannot be sent with `voice`. `stream: true` flushes wav or pcm as the projector synthesizes it. A non-negative `seed` is the TTS RNG seed, and `max_new_tokens` caps audio frames (`-1` leaves the cap off). Optional `top_k`, `top_p`, and `repeat_penalty` are the acoustic sampler; omit them to leave top_k 0, top_p 0, and repeat_penalty 1.05. `GET /v1/audio/voices` includes `sample_rate` for that projector. PCM is little-endian float32 at that rate (24000 for LFM3, Qwen3-TTS, Pocket TTS, and Chatterbox; 32000 for Soprano; 44100 for Drama3). `session.update` before the first turn sets `instructions`, Responses-API `tools`, `tool_choice`, `max_output_tokens`, and a live transcription language at `audio.input.transcription.language`. LFM3 uses that tag in the ASR prompt. `-intent transcription` also sets `session.type` to `transcription` before the first audio, so the conversation handler transcribes on commit. The proxy sends that `session.update` itself when the query is `intent=transcription` or `language` is set, before it copies client audio. File transcription accepts `prompt` and `temperature` (`0` to `1`; omitted stays `0`). A tool call comes back as `response.function_call_arguments.done`. The next turn is `conversation.item.create` with `function_call_output`, then `response.create`. A later user message is another `conversation.item.create` plus `response.create`. A later clip is another `input_audio_buffer.append`, `commit`, and `response.create`. Events are llama-server's OpenAI realtime subset: `session.update`, `input_audio_buffer.append`, `input_audio_buffer.commit`. `turn=server_vad` on this proxy commits after speech goes quiet. The sample rate is `audio.input.format.rate` on `session.created`. A session whose type is `realtime` also receives `response.create`. A transcription session receives only the commit. Put `output_modalities` on `session.update` when that reply should be audio or text. The probe's `-turn server_vad` does this, and it appends 500 ms of silence after the clip. `-pipeline duplex -tts <speech> -asr <whisper or llama ASR>` uses `-model` as the chat model. The input rate is `-rate`, or the wav rate when `-rate` is omitted. `-voice` is the speaker id. A duplex clip is committed without `response.create`, because that socket replies on the commit. The next reply includes up to 8 earlier turns from this socket. Thinking is off, and the chat template is rendered. The reply stops after 48 tokens. Earlier turns are prefilled while the next person is speaking. With `-turn server_vad` the probe still appends 500 ms of silence and does not send that commit. A delta whose `mime` is not `audio/pcm` is saved as those bytes. The probe reports the heard text as `transcript` and the spoken answer as `reply`. `-barge` cancels that answer, and the socket sends `response.done` with status `cancelled`. Quiet audio during a reply is not forwarded. A loud chunk sends `response.cancel`.

A Piper, local `kokoro`, remote-tts, or llama-server gen projector tag speaks on this socket. Send `conversation.item.create` with the text, then `response.create`. Audio comes back as `response.output_audio.delta`. Music 3 stays a job on `POST /v1/audio/speech`. `pipeline=duplex&voice=<speech tag>&asr=<whisper or llama ASR tag>&rate=<pcm16 Hz>` uses `model` as the chat model: a committed clip is transcribed, the chat model answers, and the voice speaks that answer one sentence at a time. Duplex streams `response.status` milestones (`silence_started` with `silence_duration_ms`, `transcribing`, `generating`, `first_token`, `audio_done`) and `conversation.item.input_audio_transcription.started` when ASR begins. `turn=server_vad` commits after speech goes quiet. Amplitude mode treats a sample as speech when its absolute value divided by 32768 is at least 0.02. When `OLLAMA_NEURAL_VAD` is a local Silero ONNX file, one CPU `python3 -m runtime.asr.vad_serve` process scores 8 kHz and 16 kHz packets for every duplex and llama-server-proxy socket (per-socket recurrent state; reset on commit or barge). Other rates, a missing file or package, or a score that misses a short timeout stay on amplitude at 0.02 (never the neural probability). Neural mode uses threshold 0.5 when the client omitted it; an explicit value is the probability cutoff. Duplex only attaches neural VAD when the input rate is 8 or 16 kHz. On duplex, 300 ms of quieter audio ends the turn when silence_duration_ms is omitted (500 ms on the llama-server realtime proxy), including the quiet tail of the packet that held the last speech, and 300 ms from before the speech is kept. When the first clause is not ready at commit, a cached um (then one One moment.) holds the floor until the model's opening arrives. If the sender stops after that tail, the llama-server proxy commits the rest of the silence on the clock. A packet that ends loud starts the full silence. On the duplex socket, transcription starts about 100 ms into that wait. `session.update` can set `audio.input.turn_detection` (`type` `server_vad` or `none`, plus `threshold` from 0 to 1, `silence_duration_ms`, and `prefix_padding_ms`); the session echoes that cutoff. Threshold 0 in amplitude mode treats any non-zero sample as speech. Silence 0 ends on the first quiet chunk. With turn detection off, the client commits. Prefix warmup and the reply set the same MLX `EnableMTP` when the chat model has a draft head (`qwen35` / `qwen35moe` force it off). Accepted tokens reach the sentence splitter; draft tokens do not. The first words are spoken as soon as the chat model finishes them. A comma, semicolon, colon, or about 16 characters at a word boundary starts the audio. A later stretch waits for about 48 characters, and later text keeps generating while that audio plays. A Piper live socket loads the ONNX voice when the socket opens, keeps that process for later sentences, and forwards each PCM chunk as it is synthesized. POST /v1/audio/speech reuses one Piper process for the same voice. A `qwen3-asr` tag keeps one CPU `python3 -m runtime.asr.qwen3_serve` process for every clip (file transcription and duplex). Install `qwen-asr` in that interpreter and point `backend_paths.qwen3_asr` (or `OLLAMA_QWEN3_ASR`) at a local weight directory. Missing Python, package, or directory falls back to the Whisper CLI. Whisper language codes map to names such as `English`. When whisper-server 1.9.1 or newer is installed for a Whisper tag, one CPU process takes every clip for the call. Otherwise Whisper starts when the socket opens, skips timestamp tokens, uses one greedy pass, and the next CLI process starts when synthesis finishes, so the model loads while the reply plays. A load that is still in progress is kept, and the clip is written to it when that load finishes. The clip is written when the turn commits, and it keeps about 200 ms after the last loud sample. With server_vad, transcription starts on the first quiet chunk, and the chat model starts when that transcript is ready. The voice synthesizes during that wait. The audio is sent when the turn commits. When the CLI is used, the next Whisper starts when that synthesis finishes, still during the silence when the voice is already done. If audio stops arriving, that same silence commits the turn. The chat model loads with the socket when the voice does not need that runner. `response.cancel` stops the completion and drops sentences that have not been sent, and those sentences are left out of later turns. With turn detection off, any new input audio does that too. With `server_vad`, only a loud chunk does. Qwen3-Omni and the other whole-clip models stay on `POST /v1/audio/transcriptions` unless this duplex query names them as the chat model.

The live ASR language is the `language` query. The asr template receives it when the stream opens. A template that does not read `language` gets the file-ASR hint ` (language: …)` appended. A later `session.update` that names a different language is rejected. The probe's `-lang` is that query, and the same tag is on `session.update` for a conversation. That behavior is in the llama-server source; the running binary picks it up on the next `llama-server` build.

Those realtime checkpoints stay on one GPU (`--split-mode none`). `PARAMETER main_gpu 1` puts one on the second card.

With two GPUs and no `main_gpu`, llama-server is started with `--split-mode layer` and the scheduler charges both cards. That is the path for a weight set that does not fit in 24 GiB: Voxtral Small 24B Q8 (`voxtral`, 23.3 GiB text plus the audio mmproj), Qwen3-Omni 30B-A3B Q8 (`qwen3a` plus `qwen3vl_merger`, 30.3 GiB text plus a 1.2 GiB mmproj), and Ultravox on Llama 3.3 70B at Q4 (`ultravox`, about 35–40 GiB plus the mmproj). Qwen3-Omni takes audio and images and returns text. Setting `main_gpu` pins any of these back to one card, and those quants will not load.

Voxtral Small, Qwen2.5-Omni (`qwen2.5o`), Qwen3-Omni, Ultravox, and Phi-4 multimodal (`phi4`) take a whole clip on `POST /v1/audio/transcriptions` or a chat turn with audio. A native llama-server chat sends that clip as `input_audio` after any images. Video on those models is sampled to frames, and when the model can hear, the soundtrack is attached as a wav too. Create from the text GGUF and its mmproj in one directory (`FROM .`). Voxtral Realtime has no file transcription. A gen-only projector (`qwen3tts_gen`, `pockettts_gen`, `soprano`, `drama3_gen`, `chatterbox`) speaks on `POST /v1/audio/speech` and on `GET /v1/realtime`. A live-only model on the speech route is told to use `GET /v1/realtime`. A file-audio model without a gen tower is told to use `POST /v1/audio/transcriptions`. An LFM3 model that also has the gen tower still speaks. Chat, generate, embeddings, transcription, rerank, score, decisions, and multidecode return 400 for a gen-only projector. A Piper or remote-tts tag with no weight file gets that 400 on the text routes and speaks text on the live socket. Music 3 gets that 400 on the text routes, and a live reply tells the client to use `POST /v1/audio/speech`. A Whisper tag with no weight file gets a transcription 400 on those routes, on `POST /v1/audio/speech`, and on the live socket, and still transcribes. The runtime chat and generate proxies, the SGLang chat proxy, and `POST /api/cache/warm` return that 400 before they load the model. `POST /api/load` can still warm it. A `keep_alive` of 0 still unloads it. An LFM3 model that also has the audio encoder still chats and opens the live socket.

### Host install (Linux)

Speech backends are **not llama GGUFs**. Piper uses ONNX; Whisper uses whisper.cpp `ggml-*.bin`. Weights live outside the Ollama blob store and are referenced from `backend_paths`. `/api/show` skips GGUF metadata for these config-only tags (same pattern as Wan / image backends).

On this dual-4090 lane, speech assets live on the models volume (root disk is tight):

```bash
./scripts/install_speech_backends.sh          # Piper multi-voice + Whisper + voice catalogs
OLLAMA_MODELS=/mnt/ollama_img/models ./scripts/register_speech_models.sh
sudo cp scripts/systemd/speech-backends.conf /etc/systemd/system/zerollama.service.d/
# Edit drop-in: set OLLAMA_TTS_URL=http://cozmic:8090 when using remote-tts tags
sudo systemctl daemon-reload && sudo systemctl restart zerollama
```

Registered tags: `piper-lessac:latest`, `whisper-base:latest`, `chatterbox:latest`, `orpheus:latest`, `kokoro:latest`, `irodori:latest`.

**Live OpenAPI:** this server serves `GET /docs` (Swagger UI), `GET /openapi.json`, and `GET /openapi.yaml`.

**Voice catalog:** Piper exposes every `piper_voice_*` key; remote tags load `tts_voices_file`. List with `GET /v1/audio/voices` (optional `?model=`).

**Recommended layout:** Chatterbox/Orpheus on GPU (`remote-tts`) for quality + emotion; **Irodori** for Japanese clone/Voice Design on `:8088`; local `speech=kokoro` (or remote-tts `kokoro:latest`) for efficient multi-preset; Piper as CPU fallback.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `OLLAMA_WHISPER_BIN` | Whisper.cpp `main` binary (default: `whisper` on `PATH`). |
| `OLLAMA_WHISPER_SERVER` | whisper.cpp `whisper-server` (default: `whisper-server` on `PATH`). A duplex call keeps one CPU process for every clip when this binary is 1.9.1 or newer. Missing falls back to `OLLAMA_WHISPER_BIN`. File transcription stays on the CLI. |
| `OLLAMA_WHISPER_MODEL` | Default GGML path if `backend_paths.whisper_model` is unset. |
| `OLLAMA_QWEN3_ASR` | Local Qwen3-ASR weight directory if `backend_paths.qwen3_asr` is unset. Must already exist; a Hub id is not downloaded. |
| `OLLAMA_QWEN3_ASR_DEVICE` | `device_map` for Qwen3-ASR (default `cpu`). Set `cuda:0` to load on a GPU. |
| `OLLAMA_WHISPER_VAD_MODEL` | Silero VAD ggml path if `backend_paths.vad_model` is unset. Used when file Whisper `vad=true`. Distinct from neural live turn-taking. |
| `OLLAMA_NEURAL_VAD` | Local Silero ONNX for live `server_vad` (duplex + llama-server proxy). One CPU `python3 -m runtime.asr.vad_serve` process; 8 kHz / 16 kHz only. Missing file or package leaves amplitude. No download. |
| `OLLAMA_WHISPER_EXTRA_ARGS` | Extra CLI tokens (space-separated) appended to Whisper. |
| `OLLAMA_WHISPER_TIMEOUT` | Max runtime for Whisper (Go duration, e.g. `15m`; default `10m`). |
| `OLLAMA_PIPER_BIN` | Piper executable (default: `piper`). |
| `OLLAMA_PIPER_TASHKEEL_MODEL` | libtashkeel ONNX path if `backend_paths.tashkeel_model` is unset. An Arabic voice uses it to add diacritics. |
| `OLLAMA_PIPER_ESPEAK_DATA` | espeak-ng data directory if `backend_paths.espeak_data` is unset. An eSpeak voice uses it to phonemize. |
| `OLLAMA_PIPER_TIMEOUT` | Max runtime for Piper (default `5m`). |
| `ZEROLLAMA_PIPER_WARM` | Pre-synthesize five short Piper fillers at serve. `0` skips the pass. |
| `OLLAMA_KOKORO_MODEL` | Local Kokoro ONNX when the tag is already kokoro and `backend_paths.kokoro_model` is unset. Does not select speech for other tags. |
| `OLLAMA_KOKORO_VOICES` | Local `voices-v1.0.bin` when the tag is already kokoro and `backend_paths.kokoro_voices` is unset. Does not select speech for other tags. |
| `OLLAMA_KOKORO_TIMEOUT` | Max runtime for Kokoro load and speak (default `5m`). |
| `ZEROLLAMA_KOKORO_WARM` | Pre-synthesize five short Kokoro fillers at serve. `0` skips the pass. Independent of `ZEROLLAMA_PIPER_WARM`. |
| `OLLAMA_TTS_URL` | Default base/full URL for `modality_backends.speech=remote-tts` when `backend_paths.tts_url` is unset. |
| `OLLAMA_TTS_TIMEOUT` | Max runtime for remote TTS HTTP (default `5m`). |
| `OLLAMA_EXTERNAL_IMAGE_BIN` | Script/binary for `modality_backends.image=external-image`. |
| `OLLAMA_EXTERNAL_IMAGE_TIMEOUT` | Max runtime for external image hook (default `10m`). |
| `OLLAMA_COMFYUI_URL` | Base URL for a running ComfyUI server (default `http://127.0.0.1:8188`) when `modality_backends.image=comfyui`. |
| `OLLAMA_COMFYUI_TIMEOUT` | Max wall time for one Comfy workflow run: queue + poll + download (default `10m`; heavy models like GLM-Image/FLUX.2-dev may need more on 16GB cards). |
| `OLLAMA_COMFYUI_WORKFLOWS_ROOT` | Base directory that a relative `backend_paths.comfy_workflow_dir` is joined against (default: daemon's current working directory). Set this or use absolute paths if `zerollama serve` doesn't run from the repo root. |
| `OLLAMA_SGLANG_URL` | Base URL for SGLang (e.g. `http://127.0.0.1:30000`) when `video_understanding` is `sglang`. |
| `OLLAMA_VIDEO_ALLOW_INSECURE_HTTP` | Set to `1` / `true` to allow `http://` for remote `video_url` fetches (default: **https only**). |
| `OLLAMA_FFMPEG` | `ffmpeg` binary for native video frame sampling (default: `ffmpeg` on `PATH`). |
| `OLLAMA_VIDEO_MAX_FRAMES` | Max frames sampled per video (default `32`). |
| `OLLAMA_VIDEO_SAMPLE_MODE` | `fps` (time-uniform) or `stride` (every Nth decoded frame); default `fps`. |
| `OLLAMA_VIDEO_STRIDE` | When mode is `stride`, N for `select=not(mod(n,N))` (default `30`). |
| `OLLAMA_VIDEO_SAMPLE_FPS` | `fps` filter value passed to ffmpeg when mode is `fps` (default `1`). |
| `OLLAMA_VIDEO_MAX_BYTES` | Max embedded or downloaded video size in bytes (default 256 MiB). |
| `OLLAMA_VIDEO_MAX_PER_MESSAGE` | Max `video_url` parts per message (default `1`). |
| `OLLAMA_VIDEO_MAX_IMAGES_PER_MESSAGE` | Max total images after expanding videos in one message (default `64`). |
| `OLLAMA_VIDEO_FFMPEG_TIMEOUT` | Max wall time for one ffmpeg run (default `5m`). |
| `OLLAMA_VIDEO_FETCH_TIMEOUT` | Max duration for a remote `video_url` HTTP GET (default `10m`). |

### Video understanding (VLM)

OpenAI-compatible chat accepts `content` parts with `type: "video_url"` and `video_url.url` (data URI or remote URL). Parts are merged into a **single** user message: **text**, then **`image_url` images in order**, then **raw video blobs** on `videos`. The server expands each video to PNG frames with ffmpeg and appends those frames to `images` **after** still images, preserving OpenAI array order.

Remote URLs default to **https only**; set `OLLAMA_VIDEO_ALLOW_INSECURE_HTTP=1` to allow `http://`. Fetches use the HTTP request **context** (client disconnect cancels in-flight downloads). **SSRF:** hostnames are resolved and loopback/private/link-local addresses are rejected before `GET`. This does not fully pin the TCP connection to those IPs (DNS rebinding is a known class of bypass in general); prefer allowlists or data URIs for high-security deployments.

When `modality_backends.video_understanding` is `sglang` and `OLLAMA_SGLANG_URL` is set, `POST /v1/chat/completions` requests that include `video_url` are proxied in full to `{OLLAMA_SGLANG_URL}/v1/chat/completions` (streaming responses are passed through). The upstream request uses the same JSON body as the client, including the **`model`** string: SGLang must accept that model id (or you must align names between Ollama and SGLang). The default path does **not** require SGLang.

`POST /api/chat` accepts the same `videos` field on messages (raw bytes); expansion uses the native ffmpeg path.

**Text-to-video / keyframe video:** `POST /v1/videos` (async jobs) for models with `video_gen` and `video_generation: wan`. Upload stills via `PUT /v1/media/{session}/{label}` then pass `options.media_session` + `options.keyframes`. **Why not inline base64:** see [media-uploads.md](./media-uploads.md). **Roadmap:** [ROADMAP.md](./ROADMAP.md), [wan-t2v.md](./wan-t2v.md).

### External image hook

When `modality_backends.image` is `external-image` or `openvino-image`, Ollama sets:

- `OLLAMA_IMAGE_PROMPT`, `OLLAMA_IMAGE_WIDTH`, `OLLAMA_IMAGE_HEIGHT`, `OLLAMA_IMAGE_SEED`
- `OLLAMA_IMAGE_OUTPUT`: path where your program must write a PNG.
- For SD models: `OLLAMA_SD_CLI`, `OLLAMA_SD_MODEL`, and optional `OLLAMA_SD_STEPS`, `OLLAMA_SD_CFG_SCALE`, `OLLAMA_SD_SAMPLER`, `OLLAMA_SD_DIFFUSION_FA`, `OLLAMA_SD_VAE_ON_CPU`, `OLLAMA_SD_VAE_TILING` (from manifest `backend_paths` + `image_generation`).
- For OpenVINO: `OLLAMA_OV_MODEL_DIR`, `OLLAMA_OV_PYTHON`, `OLLAMA_OV_DEVICE`, `OLLAMA_OV_STEPS`, … (from manifest + `ov_image_generate.py`).

**Why subprocess not in-process ggml:** diffusion UNet/VAE stacks are separate from chat GGUF runners; sd.cpp and OpenVINO GenAI ship their own schedulers and weight formats. Subprocess isolation keeps chat scheduler stable and lets operators swap backends without recompiling zerollama.

The model must still declare the `image` capability. List with `zerollama ls image`; bench wall time with `zerollama bench MODEL --force`.

### ComfyUI image backend

When `modality_backends.image` is `comfyui`, Zerollama proxies to a running [ComfyUI](https://github.com/comfyanonymous/ComfyUI) server.

**Why another image driver (alongside MLX, external-image, and OpenVINO):**

| Driver | Use when |
|--------|----------|
| `mlx-imagegen` (default) | Fast Z-Image / Klein 4B on the in-tree MLX subprocess |
| `comfyui` | Agent utility: edit, img2img, ControlNet, LoRA, Qwen/FLUX/GLM graphs |
| `external-image` | One-off `OLLAMA_EXTERNAL_IMAGE_BIN` script (no workflow discovery) |

Porting each HF DiT into MLX would take months; Comfy already packs those graphs. See **[comfyui-image-backend.md](./comfyui-image-backend.md)** for setup, model presets (`comfy/*`), and `options.*`. Discovery: `GET /api/image/workflows?model=<name>`.

## Audio formats

The Whisper adapter keeps the upload’s **filename extension** when present (e.g. `.webm`, `.mp3`, `.wav`); otherwise it **sniffs** common magic bytes (WAV, MP3, FLAC, Ogg, WebM/EBML). Your Whisper build must support the container you send.

`POST /v1/audio/speech` limits **`input`** to **4096 Unicode characters** (OpenAI-compatible). **`speed`** is mapped to Piper `--length_scale` (inverse: higher speed ⇒ shorter scale), clamped similarly to OpenAI’s 0.25–4.0 range. **`sentence_silence`** is seconds of silence after each sentence, passed as `--sentence_silence`. An omitted value stays at 0.2. Other speech models return 400. **`noise_scale`** is Piper generator noise, passed as `--noise_scale`. 0 keeps the generator steady. An omitted value leaves the voice JSON, or 0.667 when that JSON omits it. Other speech models return 400. **`noise_w`** is Piper phoneme-width noise, passed as `--noise_w`. 0 keeps each phoneme's length steady. An omitted value leaves the voice JSON, or 0.8 when that JSON omits it. Other speech models return 400. **`speaker_id`** is the Piper `--speaker` index. An omitted value, or 0, leaves the default speaker. A single-speaker voice leaves it unused. Other speech models return 400 when it is above 0. **`phoneme_silence`** is a map of one character to seconds of silence. Each value is from 0 to 5, and the map holds at most 32 characters. Piper passes it as `--phoneme_silence`, and a character already listed in the voice JSON keeps that pause. Kokoro splits the text on those characters, synthesizes each piece, and inserts that many seconds of silence. Other speech models return 400 when the map is set. An Arabic Piper voice diacritizes the text with libtashkeel before phonemization. The model file comes from `backend_paths.tashkeel_model` or `OLLAMA_PIPER_TASHKEEL_MODEL`. When that is unset, Piper looks for `libtashkeel_model.ort` next to its executable. A missing file fails Arabic speech. Other languages leave the path unused. An eSpeak Piper voice turns text into phonemes from an espeak-ng data directory. That directory comes from `backend_paths.espeak_data` or `OLLAMA_PIPER_ESPEAK_DATA`. When that is unset, Piper looks for `espeak-ng-data` next to its executable. A missing directory fails that voice. A voice that does not use eSpeak leaves the directory unused. Optional **`emotion`** is a zerollama extension. `remote-tts` forwards it. Piper ignores it. Kokoro accepts `calm`, `sad`, `serious`, `neutral`, `happy`, `excited`, or `angry` and scales the requested speed by 0.85, 0.80, 0.90, 1, 1.10, 1.20, or 1.25. Any other word is 400, and so is a product outside 0.5–2. Piper keeps a short phrase (48 characters or fewer) in memory after the first synthesis, up to 32 clips. A repeat returns that WAV and sets `X-Speech-Cache: hit`. Serve pre-synthesizes `Okay.`, `Sure.`, `One moment.`, `Let me check.`, and `Got it.` for each Piper tag whose ONNX file is present. `ZEROLLAMA_PIPER_WARM=0` skips that pass. A longer line is synthesized every time. `stream: true` runs Piper with `--output_raw` and sends a WAV header, then 16-bit mono PCM as each line is synthesized. That needs `audio.sample_rate` in the voice JSON. Without it, the response is the finished WAV. The same flag on a remote-tts tag is forwarded upstream, and the response body is relayed as it arrives. An upstream `X-Audio-Seconds` in that range is kept on a streamed response, and on a buffered response when the body itself has no length. A WAV body uses its own length. Local kokoro keeps an omitted speed at 1 and returns 400 when speed is outside 0.5–2, the input is empty, or the voice is not in the voices file. `stream: true` sends status only after the first PCM chunk, so that rejection is still 400. An empty `voice` uses that tag's `tts_default_voice` on the request, or `af_heart`. Voice-list entries leave `sample_rate` unset; duplex clips carry 24000 Hz int16 themselves. A short Kokoro phrase (48 characters or fewer) stays in the same memory cache as Piper, keyed by the ONNX path, voices path, voice, requested speed, language, emotion, and `phoneme_silence`. A hit sets `X-Speech-Cache: hit`. Serve pre-synthesizes `Okay.`, `Sure.`, `One moment.`, `Let me check.`, and `Got it.` for each Kokoro tag whose files are present. `ZEROLLAMA_KOKORO_WARM=0` skips that pass. A streamed hit is raw int16 PCM, not a WAV.

Piper returns **WAV**. `response_format` other than `wav` is **400**. A llama-server gen projector accepts `wav` and `pcm` and returns **400** for `mp3`, `opus`, `aac`, and `flac`. A `pcm` response uses `Content-Type: audio/pcm;rate=<Hz>;encoding=float;bits=32`. A rate llama-server already set is kept. A buffered wav or pcm response includes `X-Audio-Seconds`. A stream sends that value as a trailer after the last chunk when the length is known. Remote engines may return WAV or MP3 depending on upstream `Content-Type`. There is no server-side transcoding.

`GET /v1/audio/voices` lists selectable `voice` ids across speech models (or `?model=` for one tag). `?model=` on a live-only, file-audio, Whisper, or Parakeet tag returns the same 400 as the other audio routes.

### Transcription `response_format`

- **`json`** (default) and **`text`** return the transcript. Parakeet and Nemotron-transcribe `json` and `verbose_json` also include `usage` token counts from llama-server. When that body includes `input_tokens_details.cached_tokens`, the same object is copied onto the response. A streaming `transcript.text.done` already carries that field. A WAV upload with a byte rate sets `usage.seconds` on `json` and `verbose_json` for Whisper, for a chat model that hears, and for Parakeet or Nemotron-transcribe. Whisper and chat use `usage.type` `duration`. Parakeet and Nemotron keep `type` `tokens` and add `seconds` beside the token counts.
- **`stream=true`** on those same models returns `text/event-stream`. Each JSON object is preceded by `event:` set to its `type` (`transcript.text.delta`, then `transcript.text.done` with `usage`), and the stream ends with `data: [DONE]`. A WAV upload sets `usage.seconds` on that done event. `response_format` stays `json`. Whisper and chat models return 400.
- **`include[]=logprobs`** on a Whisper model adds a `logprobs` array of `token`, `logprob`, and `bytes` to `json` and `verbose_json`. Other models return 400.
- **`verbose_json`** returns `task`, `text`, `language`, `duration`, and segments. A Whisper model fills `start` and `end`, and `avg_logprob` plus `no_speech_prob` when the CLI wrote token probabilities. Each segment's `tokens` lists the whisper.cpp token ids when the JSON included them. `speaker` is set when the cue text starts with a diarization prefix. `duration` is the WAV length when the upload is a WAV. Otherwise it is the last cue end. Other backends use one segment. Its end is the WAV length when that length is known, and time 0 otherwise.
- **`diarized_json`** on a Whisper model passes `--diarize`. Segments use `id`, `type` `transcript.text.segment`, `start`, `end`, and `speaker` (`0`, `1`, or `?`). `text` is the spoken words. A mono clip leaves `speaker` empty. Other models return 400.
- **`srt`** and **`vtt`** use those same times. `timestamp_granularities=word` with `verbose_json` passes `--split-on-word` and adds a `words` array from whisper.cpp token times.
