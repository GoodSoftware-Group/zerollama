# Media Roadmap

This file holds the detailed backlog for **media generation** and related multimodal product surfaces.

Research-only parity work lives in [../RESEARCH_ROADMAP.md](../RESEARCH_ROADMAP.md).

---

## Video generation — DiT media toolkit (Wan + LTX + H3)

Open work only:

| Milestone | Status | Open work |
|-----------|--------|-----------|
| **v1.1** | **Open** | Cancel + TTL + kill running Wan jobs |
| **v1.2b** | **Open** | `mmgp` layer offload for 16 GB TI2V |
| **v1.3** | **Open** | Optional Wan CPU worker sidecar |
| **v1.4b** | **Open** | Mac `video-c` multi-family (`wan|h3`) |
| **Later** | **Open** | Diffusers / other runner values |
| **Later** | **Open** | Pure-C diffusion + VRAM paging |

Primary docs: [../wan-t2v.md](../design/media/wan-t2v.md), [../ltx-t2v.md](../design/media/ltx-t2v.md), [../video-c.md](../design/media/video-c.md), [../h3-cuda-port.md](../design/media/h3-cuda-port.md).

## Music generation — MiniMax Music 3

Open work only:

| Milestone | Status | Open work |
|-----------|--------|-----------|
| **m1** | **Open** | Host DAV from `dav.pth` → 44.1 kHz → resample 32 kHz |
| **m2** | **Open** | Music-vocab GGUF on Metal llama.cpp + C depth/fusion |
| **m3** | **Open** | DiT one-chunk then short `--generate` |
| **Later** | **Open** | Dedicated `music_generation` capability if TTS catalog stays confusing |
| **A7** | **Open** | Prune music `.wav` artifacts and hold the GPU for the full Wan2GP timeout. Detail below. |

Reference: [../music-c.md](../design/media/music-c.md).

## Speech, transcription, and realtime voice

Audit of the local audio path on 9 Nov 2026 (`/v1/audio/*`, `GET /v1/realtime`, Piper, Kokoro, whisper, Qwen3-ASR, Gemma4 audio input, Music 3 jobs). Open defects only.

| ID | Status | Open work |
|----|--------|-----------|
| **A1** | **Done** | One writer per realtime socket: voicepipe async out-queue + atomic frames; wsproxy backend writes under `writeMu`. Draft/commit no longer block STT on unread status frames. |
| **A2** | **Done** | `POST /v1/audio/transcriptions` and `POST /v1/audio/speech` cap bodies at 32 MiB (`MaxBytesReader` + limited file read). |
| **A3** | **Done** | Piper `Shared` keeps one process per flag set with refcounts; switching voice/flags no longer kills an in-flight reader. |
| **A4** | **Done** | Serve shutdown calls `modality.CloseAudioShared()` (Piper, Kokoro, Qwen3-ASR, neural VAD). Whisper-server remains per duplex socket. |
| **A5** | **Done** | Reject bad WAV headers in gemma4 `decodeWAV` / harden `resampleLinear` against zero channels, bit depth &lt; 8, and sample rate &lt; 2. |
| **A6** | **Done** | Hand-start hardening for `scripts/tts_remote_server.py`: default `TTS_HOST=127.0.0.1`, optional `TTS_API_KEY`, 32 MiB body cap, ref-audio confined to `TTS_REF_AUDIO_DIR` / `TTS_REF_AUDIO` parent, generic 500 bodies, Chatterbox model singleton. |
| **A7** | **Done** | Music `.wav` artifacts prune with video TTL; job timeout capped to `videoExclusiveWatchLimit` (wan2gp modelfile 10800s). |
| **A8** | **Later** | Transcription cloud and remote-model branches call `ChatHandler`, which rejects both. The chat fallback stores the clip on `Images`, so the vision check runs and WebM/MP4 demux does not. An unknown `response_format` comes back as JSON. `GET /v1/audio/voices` uses `BackendFor` while speech uses `ResolveSpeechBackend`. `ZEROLLAMA_MUSIC_CLI` is forwarded and never read. |

Left in place after the same pass: `/v1/realtime` sits behind the global CORS and allowed-hosts middleware, upload filenames are reduced to a base name, and speech input stays capped at 4096 runes (20000 when instructions or a duration is set).

Also seen while running `go test -race` on the server package, outside the audio handlers: `Scheduler.processOnePending` writes a pending-request field while `schedPendingAttrs` reads it from `GetRunner`, and `discover.bootstrapGgmlDevices` writes from a goroutine that `effectiveGgmlFreeVRAMForSuggest` reads with no lock. The regression workflow does not pass `-race`.

### Native full-duplex (Haru) — later

| ID | Status | Open work |
|----|--------|-----------|
| **H1** | **Design** | Haru-4B native full-duplex (7.5 Hz frame clock). Fourth realtime path beside Voxtral / speech / cascade duplex. Design note only; listen gate before any Go bridge. [haru-duplex.md](../design/media/haru-duplex.md). |

## Image generation — MLX fast path + ComfyUI utility (partial)

Open work only:

| Milestone | Status | Open work |
|-----------|--------|-----------|
| **v0.1** | **Open** | Calibrate one golden workflow on ship GPU |
| **v0.2** | **Open** | Cancel / interrupt Comfy on client disconnect |
| **v0.3** | **Open** | Optional `upscale` template + checkpoint override |
| **Later** | **Open** | Async job shape for multi-minute GLM / FLUX.2 |

Reference: [../comfyui-image-backend.md](../design/media/comfyui-image-backend.md), [../imagegen-zimage-turbo.md](../design/media/imagegen-zimage-turbo.md).

## Mesh / GLB generation — TRELLIS lab (HTTP not started)

Open work only:

| Milestone | Status | Open work |
|-----------|--------|-----------|
| **g1** | **Open** | Service bake without `nvdiffrast` |
| **g2** | **Open** | `POST /v1/meshes` + exclusive GPU hold |
| **Later** | **Open** | Lit Mac preview; Pixal3D / PartCrafter / further catalog work |

Reference: [../mesh-gen.md](../design/media/mesh-gen.md).
