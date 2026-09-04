# LTX text-to-video (zerollama v1.4 slice)

Local **LTXV** generation reuses the OpenAI async Videos API (`POST /v1/videos`) and the training **`run_script`** queue — same control plane as Wan. The runner is sibling **[Wan2GP](https://github.com/deepbeepmeep/Wan2GP)** (`shared.api` / `--process`), not Gradio `wgp.py` as the product UI.

**Shipped first:** **LTX Video 0.9.8 Distilled 13B + quanto bf16_int8** (`ltxv_distilled`) and **2B distilled FP8** (`ltxv_2b_distilled`, official `ltxv-2b-0.9.8-distilled-fp8.safetensors`). **Not** LTX-2 / Gemma TE on ~24 GiB host RAM boxes.

**Mac GRAPH / toolkit (parallel):** bmtl [WISHLIST_LTX_MEDIA.md](../../bmtl/hardware_lab/lanes/m4/uma_toolkit/docs/WISHLIST_LTX_MEDIA.md) under [WISHLIST_DIT_MEDIA.md](../../bmtl/hardware_lab/lanes/m4/uma_toolkit/docs/WISHLIST_DIT_MEDIA.md).

Related: [wan-t2v.md](./wan-t2v.md), [wangp-borrowings.md](./wangp-borrowings.md), [ROADMAP — Video generation](./ROADMAP.md#video-generation--dit-media-toolkit-wan--ltx--h3), [h3-cuda-port.md](./h3-cuda-port.md) (H3 parallel).

## Why LTXV distilled first

| Choice | Why |
|--------|-----|
| **LTXV 13B distilled + quanto** | 6 steps, long-prompt model, Wan2GP mmgp profile 5 fits 16 GB VRAM classes better than LTX-2 |
| **LTXV 2B distilled FP8** | Prompt-iteration / cartoon-prototype tag (`ltxv-2b-distilled:lab`). ~4.5 GiB DiT, 512² / 8 steps. Wan2GP has no native 2B type — we install a **finetune JSON** that reuses the LTXV loader. |
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
./scripts/video/install_ltx_wan2gp.sh --venv-only
./scripts/video/install_ltx_wan2gp.sh --2b-only
./scripts/video/register_ltx_models.sh
```

On Mac the installer clones sibling `../Wan2GP` if it is missing (override with `WAN2GP_REPO`). zsh does not treat `#` as a comment unless `setopt interactivecomments` — put flags alone on the line.

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
| `video_generation.profile` | `ltxv-13b-distilled` or `ltxv-2b-distilled` |
| `video_generation.quant` | `quanto` (13B) or `fp8` (2B) |
| `video_generation.steps` | `6` (13B distilled lock) or `8` (2B distilled recipe) |
| `LTX_*` / `WAN2GP_*` | Wrapper env (see `ltx_video_generate.py`) |
| `LTX_DRY_RUN=1` | Validate settings/weights; no DiT allocate |
| `ZEROLLAMA_LTX_MIN_HOST_RAM_GIB` | Raise-only host floor (default **12** GiB for 13B, **8** GiB for 2B) |

## Admission / QoS

- Same **exclusive GPU** lease as Wan (`video_exclusive.go`).
- Host floor starts at **12 GiB** for 13B (Wan mmgp + GPU VAE class) and **8 GiB** for 2B. Tune after measured peaks.
- Full generate needs free VRAM (~prod often holds ~6.5 GiB on `:11434`). **Unload production listeners only when the operator requests it.**

## API

Same as Wan: `POST /v1/videos` → poll `GET /v1/videos/:id` → `GET …/content`. Keyframes are **not** supported on these LTXV T2V tags (TI2V/control later).

| Tag | Use |
|-----|-----|
| `ltxv-13b-distilled:16g` | Quality 768×512, 6 steps, Linux 16g CUDA (Wan2GP) |
| `ltxv-2b-distilled:lab` | Wan2GP 2B FP8 (CUDA/PyTorch) |
| `ltxv-2b-mlx:lab` | **Fast Darwin prototype:** 768×480, 17 frames, **4 steps**. Expects cartoon drift. |
| `ltxv-13b-mlx:lab` | **Darwin anime:** 1280×720, 41 frames, **8 steps**, first-frame I2V. `./scripts/video/install_ltx_mlx.sh --13b-only` |

### Measured on M4 Max (2026-08-22, 2B distilled bf16, seed 42, 4 steps)

| Config | T5 | DiT | VAE | Total |
|--------|----|----|-----|-------|
| 480×768 × 17f | 0.2 s | 2.6 s | 1.0 s | **4.0 s** |
| 480×768 × 65f | 0.2 s | 8.5 s | 3.3 s | 12.6 s |
| 480×768 × 129f | 0.2 s | 18.7 s | 6.5 s | 26.1 s |
| 720×1280 × 97f | 0.2 s | 44.9 s | 16.3 s | **62 s** |

Model load ≈ 3 s warm. Character identity + style hold through 129 frames.
720p is production-quality anime line art. This replaces the Wan C path as the
Darwin anime-shorts runner (Wan 160²×9f was ~30 s for blob-level output; LTX
2B is ~4 s for usable frames at higher res). One transient SIGKILL (rc=137)
was observed on the very first long-frame run with a cold page cache — rerun
succeeded; not reproduced since.

### 90s anime look (validated recipe)

**Motion rule (measured 2026-08-22):** the 4-step distilled model renders a
*frozen still* unless the prompt contains explicit motion verbs — "walks
slowly" scored consecMAD 0.17 (frozen); "walks briskly toward the camera,
hair swaying, neon signs flickering" scored 4.21 (25×). Style tokens must
**lead** the prompt or motion language dilutes the aesthetic; extra steps
(8 vs 4) do not add motion. Budget ~2× render time for moving shots
(720p×97f: 174 s moving vs 82 s static).

Validated prompt template:

> 1990s Japanese anime with soft painterly cel shading and warm analog film
> grain: <subject> <strong motion verb(s)> <scene>, <secondary motion>,
> melancholic city-pop mood, slight VHS softness

Post: `./scripts/video/ltx_post_anime.sh IN OUT` (default `LTX_LOOK=90s`):
light denoise → soft unsharp → warm curves (red-lift mids, blue-rolled
shadows) → vignette → animated grain. No posterization — 90s anime has rich
color. `LTX_LOOK=cel` + color count remains available for the hard
propaganda-poster variant.

### Flat 90s TV-cel look (Evangelion-era, validated recipe)

For hard-shadow flat-cel interiors (crisp ink, saturated fills — the "Shinji
on the phone" look), swap the soft/painterly language for flat-cel language
**plus a show-style anchor and character-design anchors** — without them the
2B prior drifts to western preschool cartoon:

> mid-1990s Japanese TV anime cel in the style of Neon Genesis Evangelion,
> crisp thin ink outlines, flat saturated colors, hard two-tone shadows:
> <subject with 90s anchors — "almond-shaped eyes", "short messy black hair">,
> <scene with color anchors — "blue bedspread", "pink pillow">, <motion>

Post for this look: `LTX_LOOK=eva ./scripts/video/ltx_post_anime.sh IN OUT`.
Known gap: 2B T2V line weight stays softer than ink; lock it with **13B + I2V**.

### 13B optimization notes (measured 2026-08-25)

`--bits 8|4` landed in ltx-mlx (`mlx.nn.quantize` after weight load; packed
uint32 params skip the dtype cast). Findings:

| Config | 97f I2V | DiT step | Quality |
|--------|---------|----------|---------|
| 13B bf16, ≤65f | ✅ | 11.5 s | reference |
| 13B bf16, 97f | **OOM** | — | — |
| 13B 8-bit, 97f | ✅ | ~32 s (**slower**) | near-lossless, verified |
| 13B 4-step bf16 | — | 4.6 s | face quality visibly degrades |

Why quantized is slower here: DiT GEMMs at video token counts are
compute-bound, so dequant overhead beats the bandwidth saving (quant wins
only for memory-bound LLM decode). **Use `--bits 8` solely to unlock 97f
13B I2V; stay bf16 otherwise.** 8 steps is the operating point — 4-step
degrades faces. Quantize-in-RAM adds ~15 s/run (persisting the 8-bit
checkpoint needs 14 GB disk; only worth it if space frees up).

Recommended tiers: **2B** for iteration (4 s/shot) · **13B bf16 8-step**
for keeper plates ≤65f (2 min/shot, best faces) · **13B 8-bit** only for
97f shots (5 min). The ~33f stillness ceiling holds on both models — it is
architectural (length-dependent motion prior), not a capacity effect.

### I2V style-lock from a reference cel (validated)

The exact-style path: condition on a real 90s cel frame and animate it.

```bash
# input image AND --width/--height MUST be divisible by 32, else the CLI
# exits rc=0 with NO output and no error (silent no-op — check for the file!)
ffmpeg -i ref.png -vf "scale=512:768:flags=lanczos" ref_32.png
ltx_mlx.cli --image ref_32.png --width 512 --height 768 --frames 97 \
  --prompt "1990s TV anime, <subject keeps doing <action>, subtle motion verbs>"
```

Measured (512×768×97f, 4 steps): 106.8 s total (DiT 35.9 + VAE 57.4 —
portrait VAE decode is the slow stage). Frame 0 ≈ the reference exactly;
motion (character sits up, cord sways) at consecMAD ≈ 4 while style, scene,
and identity hold. This beats prompt-only styling for any series with a
fixed art plate: draw/generate one canonical keyframe per character/room,
then animate from it.

**Motion-duration limit (measured):** unconstrained I2V motion warps
hand-drawn geometry ("nightmare fuel", consecMAD 4.01, character fully
re-posed by f90). The fix is **limited animation** — the actual 90s TV
technique. Prompt: "limited animation: <subject> stays <pose>, only chest
rises and falls, blinks slowly, mouth moves as he talks, cord sways gently,
everything else perfectly still, camera locked". Result: consecMAD 0.45,
drawing holds, mouth/eyes animate. **Reliable window is ~33 frames (1.4 s
at 24 fps)** — at 97f the sigma schedule (length-dependent) re-injects
motion energy and drift returns (per-segment MAD 2→11). For longer holds:
stretch to 12 fps effective (`setpts=2*PTS` → 2.75 s, period-correct —
limited animation ran on twos/threes), or chain last-frame → next I2V.

LTXV wants **long, descriptive prompts**. Distilled models lock CFG≈1 and ignore negatives in Wan2GP — color clamps belong in the **positive** prompt (or post-quantize frames).

## E2E after unload

1. Operator stops/unloads prod inference on `:11434` / `:8081` (or free enough VRAM).
2. `POST /v1/videos` with `model: "ltxv-13b-mlx:lab"` (Mac anime) or `"ltxv-2b-mlx:lab"` (fast prototype) or `"ltxv-13b-distilled:16g"` (Linux CUDA) and a **long** prompt. Optional first-frame still via `options.keyframes[0]` or `options.first_frame_image` on MLX tags. `last_frame_image` (or a second keyframe) is **400** — community `ltx-mlx` has `--image` only (mlx-serve #260).
3. Poll until `completed`; download mp4 from `/content`.

## Out of scope (this slice)

- LTX-2 / Gemma, MiniMax H3 product path, Gradio UI, native `h3_cuda`
- Generating while production holds the GPU without an unload window
