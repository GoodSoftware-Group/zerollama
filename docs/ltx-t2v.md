# LTX text-to-video (zerollama v1.4 slice)

Local **LTXV** generation reuses the OpenAI async Videos API (`POST /v1/videos`) and the training **`run_script`** queue — same control plane as Wan. The runner is sibling **[Wan2GP](https://github.com/deepbeepmeep/Wan2GP)** (`shared.api` / `--process`), not Gradio `wgp.py` as the product UI.

**Shipped first:** **LTX Video 0.9.8 Distilled 13B + quanto bf16_int8** (`ltxv_distilled`). **Not** LTX-2 / Gemma TE on ~24 GiB host RAM boxes.

Related: [wan-t2v.md](./wan-t2v.md), [wangp-borrowings.md](./wangp-borrowings.md), [ROADMAP — Video generation](./ROADMAP.md#video-generation--wan-t2v-v1-shipped), [h3-cuda-port.md](./h3-cuda-port.md) (H3 later).

## Why LTXV distilled first

| Choice | Why |
|--------|-----|
| **LTXV 13B distilled + quanto** | 6 steps, long-prompt model, Wan2GP mmgp profile 5 fits 16 GB VRAM classes better than LTX-2 |
| **Not LTX-2 on CT 1564** | LTX-2 preloads **Gemma-3-12B** TE → host RAM often worse than Wan TI2V; ~24 GiB + prod serve is a wall |
| **Wan2GP, not Gradio** | Product is `/v1/videos` + exclusive GPU QoS; Gradio competes for ports/UX |
| **`modality_backends.video_generation: "ltx"`** | Multi-family registry (ROADMAP v1.4); Wan stays `"wan"` |
| **Config-only tag** | Multi‑GB ckpts under `~/.zerollama/third_party/wan2gp/ckpts/` — not GGUF blobs |

## Architecture

```text
Client  POST /v1/videos  (model=ltxv-13b-distilled:16g)
   │
   ▼
Go  VideoCreateHandler
   │  backend=ltx → admitLtxHostRAM → exclusive lease
   │  buildLtxVideoPayload → run_script
   ▼
training.py  run_local_script
   │  wan2gp venv python  scripts/video/ltx_video_generate.py
   ▼
Wan2GP shared.api / wgp.py --process
   │  model_type=ltxv_distilled, profile 5, attention sdpa
   ▼
$OLLAMA_MODELS/generated/<job_id>.mp4
```

## Install

```bash
# Sibling checkout expected at /root/Wan2GP (or set WAN2GP_REPO)
./scripts/video/install_ltx_wan2gp.sh            # venv + LTXV distilled/quanto + VAE/T5
./scripts/video/install_ltx_wan2gp.sh --weights-only   # skip pip if venv exists
./scripts/video/register_ltx_models.sh
```

Weights land under `$WAN2GP_ROOT/ckpts/` (default `~/.zerollama/third_party/wan2gp/ckpts`). DiT ~13 GB quanto + T5 ~5 GB + VAE ~2.5 GB.

**Lab only:** never bind Wan2GP Gradio to production `:11434` / `:8081`. Dry-run validates weights + settings via the thin wrapper (does not import Gradio/`wgp.py`):

```bash
./scripts/video/install_ltx_wan2gp.sh --dry-run
# or
LTX_DRY_RUN=1 WAN2GP_REPO=/root/Wan2GP ... python scripts/video/ltx_video_generate.py
```

The install script reuses `~/.zerollama/third_party/wan/venv` when present (symlink) so you do not need a second PyTorch tree for dry-run / headless generate.
## Manifest / env

| Field / env | Role |
|-------------|------|
| `backend_paths.wan2gp_repo` | Sibling Wan2GP tree |
| `backend_paths.wan2gp_venv` | Python venv with torch + Wan2GP deps |
| `backend_paths.wan2gp_ckpt_dir` | Checkpoint dir (`ckpts`) |
| `video_generation.profile` | `ltxv-13b-distilled` |
| `video_generation.quant` | `quanto` |
| `video_generation.steps` | `6` (distilled lock) |
| `LTX_*` / `WAN2GP_*` | Wrapper env (see `ltx_video_generate.py`) |
| `LTX_DRY_RUN=1` | Validate settings/weights; no DiT allocate |
| `ZEROLLAMA_LTX_MIN_HOST_RAM_GIB` | Raise-only host floor (default **12** GiB, mmgp+GPU-VAE class) |

## Admission / QoS

- Same **exclusive GPU** lease as Wan (`video_exclusive.go`).
- Host floor starts at **12 GiB** (Wan mmgp + GPU VAE class). Tune after measured peaks.
- Full generate needs free VRAM (~prod often holds ~6.5 GiB on `:11434`). **Unload production listeners only when the operator requests it.**

## API

Same as Wan: `POST /v1/videos` → poll `GET /v1/videos/:id` → `GET …/content`. Keyframes are **not** supported on this LTXV T2V tag (TI2V/control later).

## E2E after unload

1. Operator stops/unloads prod inference on `:11434` / `:8081` (or free enough VRAM).
2. `POST /v1/videos` with `model: "ltxv-13b-distilled:16g"` and a **long** prompt (LTXV expects rich prompts).
3. Poll until `completed`; download mp4 from `/content`.

## Out of scope (this slice)

- LTX-2 / Gemma, MiniMax H3 product path, Gradio UI, native `h3_cuda`
- Generating while production holds the GPU without an unload window
