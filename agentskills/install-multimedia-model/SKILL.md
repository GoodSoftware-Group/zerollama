---
name: install-multimedia-model
description: "Help a user pick and install one multimedia model into zerollama (music gen, TTS, STT, image gen, video gen, or media understanding) — present modality options with disk/platform notes, install only the chosen one, never bulk-download everything."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, multimedia, install, music, tts, stt, image, video, whisper, wan, comfyui]
    category: mlops
    related_skills: [download-model, generate-image, generate-video, text-to-speech, speech-to-text, video-understanding-chat, model-suggester]
---

# Install Multimedia Model Skill

Guide a user who wants **one** multimedia capability on
[zerollama](https://github.com/GoodSoftware-Group/zerollama) — music, speech,
image, video, or media understanding — to a concrete option, then install
**only that option**. Do **not** install every multimedia stack; disk and
VRAM budgets are large (Music 3 alone is ~14 GB; Wan/Comfy/LTX are larger).

This skill is the **picker + install** half. After install, hand off to the
matching usage skill (`generate-image`, `generate-video`, `text-to-speech`,
`speech-to-text`, `video-understanding-chat`, or music HTTP in
`docs/music-c.md`).

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/tags   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/v1/audio/generations -d '{}'   # 400/422 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/v1/audio/speech -d '{}'   # 400/422 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/v1/audio/transcriptions -d '{}'   # 400/422 = route exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

## When to Use

- User says they want to **make music**, **speak text**, **transcribe
  audio**, **generate images/video**, or **understand a video clip**
- User asks "what multimedia models can I install?" without naming one
- Choosing between MLX vs Comfy vs Wan/LTX for a host's disk/GPU

## Agent workflow (always)

1. **Map intent → modality** (table below).
2. **List options** for that modality (disk, platform, quality notes).
3. **Ask the user to pick one** (or one default if they say "just pick").
4. **Check what's already registered** before downloading:
   `curl -s http://127.0.0.1:11434/api/tags | jq -r '.models[].name'`
5. **Install only the chosen option** with the command in that section.
6. **Confirm** the tag appears in `/api/tags`, then point them at the
   usage skill / smoke curl.

Never run every install script in this file unless the user explicitly
asks for multiple modalities one by one.

## Intent → modality

| User says… | Modality | Options section |
|---|---|---|
| make music / song / lyrics → audio | Music gen | [Music](#music-generation) |
| speak / TTS / voiceover / narrate | TTS | [Text-to-speech](#text-to-speech) |
| transcribe / STT / whisper / speech→text | STT | [Speech-to-text](#speech-to-text) |
| generate / draw / edit an image | Image gen | [Image generation](#image-generation) |
| generate a video / text-to-video / clip | Video gen | [Video generation](#video-generation) |
| watch / understand / describe a video | Video understand | [Media understanding](#media-understanding) |

---

## Music generation

Local songs on Mac = **mlx-audio MiniMax Music 3** (not MiniMax cloud, not
Comfy GPL). See `docs/music-c.md`.

| Option | Disk | Platform | Why pick it |
|---|---|---|---|
| `MiniMax-Music3-8bit` → tag `minimax-music3:lab` | ~14 GB | Apple Silicon (MLX) | **Default first listen** |
| MXFP4 pack | ~8 GB | Apple Silicon | Tighter disk; weaker lyrics |
| Full Omni/CUDA pack | tens of GB | CUDA lab | Rematch gold later — not first listen |

**Install (default 8-bit):**

```bash
# From zerollama repo root
uv venv --python 3.11 .venv-music
uv pip install --python .venv-music \
  "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git@784b29e2691a93ca7483147d86f61859dfaa6296" \
  huggingface_hub
hf download mlx-community/MiniMax-Music3-8bit \
  --local-dir ~/.zerollama/models/MiniMax-Music3-8bit
./scripts/audio/register_music3_models.sh
# Restart serve if routes were missing before this build, then:
curl -s http://127.0.0.1:11434/api/tags | grep -i music
```

**Smoke (optional, before HTTP):**

```bash
.venv-music/bin/python scripts/audio/music3_mlx_generate.py \
  --model ~/.zerollama/models/MiniMax-Music3-8bit \
  --duration 10 --seed 7 --out /tmp/music3_10s.wav
```

**Use after install:** `POST /v1/audio/generations` (async 202) — not sync
`/v1/audio/speech` bytes. Lyrics = `input`, caption = `instructions`.

---

## Text-to-speech

Speech backends are **not llama GGUFs** — ONNX / remote HTTP. Host install
script + register:

| Option | Disk / cost | Why pick it |
|---|---|---|
| `piper-lessac` | Small (CPU ONNX) | **Default** local TTS; always WAV |
| `kokoro` | Medium | Efficient multi-preset remote-tts |
| `chatterbox` | Larger + GPU sidecar | Quality + emotion |
| `orpheus` | Larger + GPU sidecar | Emotion (`excited`/`sad`) |
| `irodori` | Sidecar `:8088` | Japanese clone / Voice Design |

**Install (speech pack + register):**

```bash
# Weights + Piper/Whisper bins (Linux host layout; override SPEECH_ROOT on Mac if needed)
SPEECH_ROOT="${SPEECH_ROOT:-$HOME/.zerollama/speech}" ./scripts/install_speech_backends.sh
OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}" ./scripts/register_speech_models.sh
# Irodori only (extra):
./scripts/speech/install_irodori_tts.sh
```

Registered tags: `piper-lessac`, `whisper-base`, `chatterbox`, `orpheus`,
`kokoro`, `irodori`. Remote-tts engines also need
`TTS_ENGINE=<name> TTS_PORT=8090 python3 scripts/tts_remote_server.py` (or
`OLLAMA_TTS_URL`) when using non-Piper tags.

**Use after install:** `text-to-speech` skill → `POST /v1/audio/speech`.

---

## Speech-to-text

| Option | Disk | Why pick it |
|---|---|---|
| `whisper-base` | Small GGML | **Default** transcription |
| Larger Whisper GGML | Bigger | Better accuracy if disk allows |
| Multimodal chat w/ audio | Model-sized | Already have a VL/audio LLM |

`install_speech_backends.sh` + `register_speech_models.sh` (above) register
`whisper-base`. Point `backend_paths.whisper_model` / `OLLAMA_WHISPER_BIN`
if the binary isn't on `PATH` (`docs/multimodal-backends.md`).

**Use after install:** `speech-to-text` skill →
`POST /v1/audio/transcriptions`.

---

## Image generation

| Option | Disk | Platform | Why pick it |
|---|---|---|---|
| `x/z-image-turbo` (or `z-image-turbo`) | ~12 GB | MLX Mac / CUDA MLX | **Default** fast local T2I |
| `x/flux2-klein` / Klein-class MLX | Medium | MLX | Higher quality, still MLX path |
| `comfy/qwen-image`, `comfy/flux2-*`, … | Comfy weights | Needs ComfyUI `:8188` | LoRA / ControlNet / heavy DiT |

**MLX pull (preferred on Mac):**

```bash
zerollama pull x/z-image-turbo
# confirm
curl -s http://127.0.0.1:11434/api/tags | grep -i z-image
```

**Comfy register (weights live in ComfyUI's tree, not Ollama blobs):**

```bash
./scripts/register_comfy_models.sh                    # all presets
# or one:
./scripts/register_comfy_models.sh comfy/qwen-image modelfiles/comfy-qwen-image/config.json
# User must still download ComfyUI weights into Comfy's models/ dir
```

**Use after install:** `generate-image` skill.

---

## Video generation

| Option | Disk | Platform | Why pick it |
|---|---|---|---|
| Wan 2.1 1.3B (`wan2.1-t2v-1.3b`) | Smaller Wan | Mac/CUDA via install script | **Default** first T2V |
| Wan 2.2 TI2V 5B | Larger | More VRAM | Better quality / i2v |
| LTX distilled / MLX LTX | Varies | See `docs/ltx-t2v.md` | Alternate stack |
| H3 / Wan2GP CUDA | Large | CUDA | Product CUDA lane |

**Install Wan (default profile — ask before `all`):**

```bash
./scripts/video/install_wan_video.sh --profile 1.3b   # or 2.2 | all
./scripts/video/register_wan_models.sh
curl -s http://127.0.0.1:11434/api/tags | grep -i wan
```

LTX / H3: `scripts/video/install_ltx_*.sh`, `install_h3_wan2gp.sh` + matching
`register_*` — only if the user picked those options.

**Use after install:** `generate-video` skill → async `POST /v1/videos`.

---

## Media understanding

Not a separate weight pack — use a **vision/video-capable chat model**:

| Option | Install | Why |
|---|---|---|
| Local VLM with `vision` / `video` capability | `zerollama pull <vlm>` | Frames via `video_url` in chat |
| Already-pulled VL model | nothing | Just use it |

```bash
# Find candidates already local
curl -s http://127.0.0.1:11434/api/tags | jq '.models[] | select(.capabilities[]? == "vision" or .capabilities[]? == "video") | .name'
# Or pull a known VLM the user chose, then:
# see video-understanding-chat skill
```

**Use after install:** `video-understanding-chat` skill.

---

## Pitfalls

- **Never bulk-install** music + Wan + Comfy + speech + LTX in one go —
  tens of GB and conflicting GPU holds.
- **Config-only tags ≠ weights downloaded** — Comfy/Wan/Music/speech
  register scripts write manifests; weights come from `hf download`,
  install scripts, or Comfy's own folder.
- **Music ≠ TTS** — `minimax-music3` is async song gen; Piper/Kokoro are
  sync speech. Don't send a song request through `/v1/audio/speech`
  expecting immediate WAV (Music returns 202 JSON).
- **Don't bind production `:11434` / `:8081` to "verify"** — lab ports or
  CLI smokes (`/tmp` WAV) only; see workspace port rules.
- **Comfy needs a live ComfyUI** at `OLLAMA_COMFYUI_URL` after register.
- **Remote-tts needs the sidecar** (`scripts/tts_remote_server.py` or
  Irodori `:8088`) or tags will fail at request time even if registered.

## Related

- `download-model` — generic GGUF pull / Comfy / Wan register overview
- `generate-image` / `generate-video` / `text-to-speech` / `speech-to-text` /
  `video-understanding-chat` — use the model after install
- `model-suggester` — capability/VRAM fit for chat models (not multimedia packs)
- Docs: `docs/music-c.md`, `docs/multimodal-backends.md`,
  `docs/comfyui-image-backend.md`, `docs/wan-t2v.md`, `docs/ltx-t2v.md`
