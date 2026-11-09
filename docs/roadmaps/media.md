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

Reference: [../music-c.md](../design/media/music-c.md).

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
