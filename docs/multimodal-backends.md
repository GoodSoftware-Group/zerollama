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
  - `image`: `mlx-imagegen` (default, implicit), `external-image` (stable-diffusion.cpp; [sd-vulkan-a380.md](./sd-vulkan-a380.md)), or `openvino-image` (OpenVINO GenAI; [sd-openvino-a380.md](./sd-openvino-a380.md)).
  - `transcribe`: `whisper` (whisper.cpp-style CLI) or omit for multimodal LLM audio models.
  - `speech`: `piper` for Piper TTS.
  - `video_understanding` (VLM): `native` (default) samples frames with **ffmpeg** and feeds them like images, or `sglang` to forward OpenAI `POST /v1/chat/completions` to a SGLang server when `OLLAMA_SGLANG_URL` is set.
  - `video_generation` (T2V): `wan` runs [Wan](../scripts/video/wan_video_generate.py) via the training job queue; capability `video_gen`. See [wan-t2v.md](./wan-t2v.md).
- **`video_sampling`** (optional, native path only): per-model overrides for ffmpeg—`mode` (`fps` or `stride`), `fps`, `stride`, `max_frames`. Omitted fields use server env defaults (see below).
- **`tokens_per_image`** (optional): vision-token budget **per raster frame** for **context preflight** only (default `768` until projector metadata is wired through).
- **`vision_patch_size`** / **`vision_spatial_merge_size`** (optional): tune native **`grid_thw`** estimates on ffmpeg-expanded clips (Qwen3-VL defaults: 14 / 2).
- **`backend_paths`**: map of string keys → filesystem paths for weights/config:
  - `whisper_model`: GGML weights for Whisper.
  - `piper_model`: Piper ONNX file.
  - `piper_config`: optional Piper JSON config.
  - `piper_voice_<name>`: optional per-voice ONNX (e.g. `piper_voice_alloy`); `<name>` is the OpenAI `voice` field with non-alphanumeric characters stripped. If set, it overrides `piper_model` for that request.
  - `wan_repo`, `wan_ckpt_dir`: Wan upstream tree and checkpoint directory (weights installed outside Ollama blobs).
  - `wan_gguf_path` (optional): GGUF weights when safetensors+offload OOM on 16 GB.
  - `sd_cli`, `sd_model`: [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (Vulkan; [sd-vulkan-a380.md](./sd-vulkan-a380.md))
  - `ov_model_dir`, `ov_python`, `external_image_bin`: OpenVINO GenAI ([sd-openvino-a380.md](./sd-openvino-a380.md))
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

Example (Whisper STT):

```json
{
  "modality_backends": { "transcribe": "whisper" },
  "backend_paths": { "whisper_model": "/path/to/ggml-base.bin" }
}
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `OLLAMA_WHISPER_BIN` | Whisper.cpp `main` binary (default: `whisper` on `PATH`). |
| `OLLAMA_WHISPER_MODEL` | Default GGML path if `backend_paths.whisper_model` is unset. |
| `OLLAMA_WHISPER_EXTRA_ARGS` | Extra CLI tokens (space-separated) appended to Whisper. |
| `OLLAMA_WHISPER_TIMEOUT` | Max runtime for Whisper (Go duration, e.g. `15m`; default `10m`). |
| `OLLAMA_PIPER_BIN` | Piper executable (default: `piper`). |
| `OLLAMA_PIPER_TIMEOUT` | Max runtime for Piper (default `5m`). |
| `OLLAMA_EXTERNAL_IMAGE_BIN` | Script/binary for `modality_backends.image=external-image`. |
| `OLLAMA_EXTERNAL_IMAGE_TIMEOUT` | Max runtime for external image hook (default `10m`). |
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

**Text-to-video:** `POST /v1/videos` (async jobs) for models with `video_gen` and `video_generation: wan`. **Roadmap:** [ROADMAP.md](./ROADMAP.md), [wan-t2v.md](./wan-t2v.md).

### External image hook

When `modality_backends.image` is `external-image` or `openvino-image`, Ollama sets:

- `OLLAMA_IMAGE_PROMPT`, `OLLAMA_IMAGE_WIDTH`, `OLLAMA_IMAGE_HEIGHT`, `OLLAMA_IMAGE_SEED`
- `OLLAMA_IMAGE_OUTPUT`: path where your program must write a PNG.
- For SD models: `OLLAMA_SD_CLI`, `OLLAMA_SD_MODEL`, and optional `OLLAMA_SD_STEPS`, `OLLAMA_SD_CFG_SCALE`, `OLLAMA_SD_SAMPLER`, `OLLAMA_SD_DIFFUSION_FA`, `OLLAMA_SD_VAE_ON_CPU`, `OLLAMA_SD_VAE_TILING` (from manifest `backend_paths` + `image_generation`).
- For OpenVINO: `OLLAMA_OV_MODEL_DIR`, `OLLAMA_OV_PYTHON`, `OLLAMA_OV_DEVICE`, `OLLAMA_OV_STEPS`, … (from manifest + `ov_image_generate.py`).

**Why subprocess not in-process ggml:** diffusion UNet/VAE stacks are separate from chat GGUF runners; sd.cpp and OpenVINO GenAI ship their own schedulers and weight formats. Subprocess isolation keeps chat scheduler stable and lets operators swap backends without recompiling zerollama.

The model must still declare the `image` capability. List with `zerollama ls image`; bench wall time with `zerollama bench MODEL --force`.

## Audio formats

The Whisper adapter keeps the upload’s **filename extension** when present (e.g. `.webm`, `.mp3`, `.wav`); otherwise it **sniffs** common magic bytes (WAV, MP3, FLAC, Ogg, WebM/EBML). Your Whisper build must support the container you send.

`POST /v1/audio/speech` limits **`input`** to **4096 Unicode characters** (OpenAI-compatible). **`speed`** is mapped to Piper `--length_scale` (inverse: higher speed ⇒ shorter scale), clamped similarly to OpenAI’s 0.25–4.0 range.

Piper returns **WAV** audio regardless of OpenAI `response_format` (other formats are not transcoded yet).

### Transcription `response_format`

- **`json`** (default) and **`text`** are fully supported.
- **`verbose_json`** returns a JSON object with `task`, `text`, optional `language`, `duration` (0 when unknown), and a single **segment** placeholder when word timestamps are unavailable.
- **`srt`** / **`vtt`** are not generated yet; the API returns JSON with a `text` field and logs a debug message.
