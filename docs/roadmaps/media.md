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
| **A1** | **Open** | One writer per realtime socket. `session.send` holds the session lock across the socket write (`server/voicepipe/session.go`), and the pong reply writes with no lock. `writeFrame` sends the header and the payload as two writes, so a pong can land in the middle of another frame. The llama-server proxy has the same split: `server/wsproxy/turn.go` writes some backend frames outside `writeMu` while the idle timer writes inside it. `TestQuietStartsTranscription`, `TestChatStartsDuringSilence`, `TestSpeechResumeDropsEarlyTranscript`, `TestReadyOpeningSkipsHold`, `TestLateOpeningPlaysHold`, and `TestBargeDropsHold` fail on master because `announceSilence` blocks on the test pipe before transcription starts. |
| **A2** | **Open** | Cap `POST /v1/audio/transcriptions` and `POST /v1/audio/speech`. `ParseMultipartForm(25<<20)` in `middleware/openai.go` only sets the in-memory spill point, and the handler then keeps the audio in memory again. Speech JSON has no `MaxBytesReader`. Other handlers already cap the body at 32 MiB. |
| **A3** | **Open** | One Piper process per flag set. `postPiper` in `server/modality/piper.go` is a single shared process. `piperphrase.Shared.procFor` kills it when the next request uses a different voice, speed, or flag, which cuts off the request still reading that process. |
| **A4** | **Open** | Stop Kokoro, Qwen3-ASR, neural VAD, and whisper-cli when serve exits. They are started with `exec.Command` and no parent-death signal. Shutdown in `server/routes.go` never calls `CloseShared` (tests are the only caller). A Mac kill leaves the children, and Qwen3-ASR keeps its VRAM. The shared loopback shell is **R7** in [core-serving.md](core-serving.md#code-structure). |
| **A5** | **Open** | Reject a bad WAV header in `model/models/gemma4/process_audio.go` before decode. Zero channels or a bit depth under 8 divides by zero. A sample rate of 0 or 1 makes `resampleLinear` allocate a huge buffer and can kill the runner. |
| **A6** | **Open** | Harden `scripts/tts_remote_server.py` for a hand start. The default bind is `0.0.0.0`, there is no auth and no body cap, `X-TTS-Ref-Audio` is a server-side file path, and errors return the exception text. The systemd unit already sets `TTS_HOST=127.0.0.1`. Chatterbox reloads the model on every request. |
| **A7** | **Open** | Music `.wav` files written into the video artifact folder (`server/music_generate.go`) are never pruned. `pruneExpiredVideoArtifacts` only deletes `.mp4` and `.json`. The Wan2GP Music 3 manifest timeout is 4 hours (`modelfiles/minimax-music3-wan2gp`) and `videoExclusiveWatchLimit` is 3 hours, so the GPU hold can end while the job is still running. |
| **A8** | **Later** | Transcription cloud and remote-model branches call `ChatHandler`, which rejects both. The chat fallback stores the clip on `Images`, so the vision check runs and WebM/MP4 demux does not. An unknown `response_format` comes back as JSON. `GET /v1/audio/voices` uses `BackendFor` while speech uses `ResolveSpeechBackend`. Duplex rejects `input_audio_buffer.clear`. Opt-in ear dumps are mode `0644`. `ZEROLLAMA_MUSIC_CLI` is forwarded and never read. |

Left in place after the same pass: `/v1/realtime` sits behind the global CORS and allowed-hosts middleware, upload filenames are reduced to a base name, and speech input stays capped at 4096 runes (20000 when instructions or a duration is set).

Also seen while running `go test -race` on the server package, outside the audio handlers: `Scheduler.processOnePending` writes a pending-request field while `schedPendingAttrs` reads it from `GetRunner`, and `discover.bootstrapGgmlDevices` writes from a goroutine that `effectiveGgmlFreeVRAMForSuggest` reads with no lock. The regression workflow does not pass `-race`.

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
