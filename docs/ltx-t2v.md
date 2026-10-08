# LTX text-to-video (zerollama v1.4 slice)

Local **LTXV** generation reuses the OpenAI async Videos API (`POST /v1/videos`) and the training **`run_script`** queue — same control plane as Wan. The runner is sibling **[Wan2GP](https://github.com/deepbeepmeep/Wan2GP)** (`shared.api` / `--process`), not Gradio `wgp.py` as the product UI.

**Shipped:** **LTX Video 0.9.8 Distilled 13B + quanto** (`ltxv_distilled`), **2B distilled FP8** (`ltxv_2b_distilled`), and on astra-class hosts **LTX-2.5 22B distilled** (`ltx2_25_22B_distilled`) with Gemma4 TE + start/end keyframe control.

**Mac GRAPH / toolkit (parallel):** bmtl [WISHLIST_LTX_MEDIA.md](../../bmtl/hardware_lab/lanes/m4/uma_toolkit/docs/WISHLIST_LTX_MEDIA.md) under [WISHLIST_DIT_MEDIA.md](../../bmtl/hardware_lab/lanes/m4/uma_toolkit/docs/WISHLIST_DIT_MEDIA.md).

Related: [wan-t2v.md](./wan-t2v.md), [wangp-borrowings.md](./wangp-borrowings.md), [ROADMAP — Video generation](./ROADMAP.md#video-generation--dit-media-toolkit-wan--ltx--h3), [h3-cuda-port.md](./h3-cuda-port.md) (H3 parallel).

## Why LTXV distilled first

| Choice | Why |
|--------|-----|
| **LTXV 13B distilled + quanto** | 6 steps, long-prompt model, Wan2GP mmgp profile 5 fits 16 GB VRAM classes better than LTX-2 |
| **LTXV 2B distilled FP8** | Prompt-iteration / cartoon-prototype tag (`ltxv-2b-distilled:lab`). ~4.5 GiB DiT, 512² / 8 steps. Wan2GP has no native 2B type — we install a **finetune JSON** that reuses the LTXV loader. |
| **LTX-2.5 on astra (2×4090)** | Gemma4-12B TE + 22B DiT (~50 GB pack) — host floor **72 GiB MemAvailable** (see [WHY below](#why-ltx-25-host-floor-is-72-gib)); dual-GPU **best-fit** placement ([cuda_device.py](../scripts/video/cuda_device.py)); start/end stills via `options.keyframes` |
| **Not LTX-2.5 on CT 1564** | ~24 GiB host + prod serve is a wall; use LTXV 13B/2B there |
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
# LTX-2.5 (control) on SSD — astra:
./scripts/video/install_ltx2_wan2gp.sh            # distilled int8 + gemma4 + VAEs + union-control LoRA
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
| `video_generation.profile` | `ltxv-13b-distilled`, `ltxv-2b-distilled`, or `ltx2.5-22b-distilled` |
| `video_generation.quant` | `quanto` (13B), `fp8` (2B), `int8_convrot` (LTX-2.5) |
| `video_generation.steps` | `6` (13B), `8` (2B / LTX-2.5 distilled) |
| `LTX_*` / `WAN2GP_*` | Wrapper env (see `ltx_video_generate.py`) |
| `LTX_IMAGE_START` / `LTX_IMAGE_END` | LTX-2.5 keyframe control (from `options.keyframes`) |
| `LTX_DRY_RUN=1` | Validate settings/weights; no DiT allocate |
| `ZEROLLAMA_LTX13B_MIN_HOST_RAM_GIB` / `…_LTX2B_…` / `…_LTX25_…` | Per-profile MemAvailable floor (defaults **12** / **8** / **72** GiB) |
| `ZEROLLAMA_LTX_MIN_HOST_RAM_GIB` | Legacy: raises **13B class only** (not 2B/2.5) unless `…_FORCE=1` |
| `ZEROLLAMA_VIDEO_ARTIFACT_TTL` | Keep completed `{id}.mp4`+`.json` servable after job wipe (default **24h**); prune deletes after expiry |
| `ZEROLLAMA_VIDEO_ARTIFACT_PRUNE_INTERVAL` | Background prune tick (default **15m**; `0`/`off` disables) |
| `DELETE /v1/videos/{id}` | Client ack after download — deletes mp4+json immediately (204) |

## Admission / QoS

- Same **exclusive GPU** lease as Wan (`video_exclusive.go`).
- Host floor: **12 GiB** (13B), **8 GiB** (2B), **72 GiB** (LTX-2.5 MemAvailable). Tune after measured peaks.
- Dual-GPU placement: see [WHY dual-GPU best-fit](#why-dual-gpu-best-fit--migrate-only-on-solvable-contention).
- Full generate needs free VRAM (~prod often holds ~6.5 GiB on `:11434`). **Unload production listeners only when the operator requests it.**
- **ltx-mlx tags on Linux:** immediate **400** — Apple MLX cannot run on NVIDIA; message points at `ltxv-*-distilled` Wan2GP tags. **Why not remap:** silent CUDA alias hid the platform mismatch and made Mac vs Linux tags mean different runners.

### WHY LTX-2.5 host floor is 72 GiB

**Incident (astra, 2026-10-02):** `ltx2.5-22b-distilled:48g` ran ~19 min with oscillating `encoding_text` progress, then the kernel **OOM-killed `zerollama-serve`** (~54 GiB process RSS; unit `MemoryPeak≈54.2G`). Swap was already full. Journal showed mmgp re-hooking / reloading the 19 GiB DiT in a loop — not “slow diffusion.”

| Old gate | Why it failed |
|----------|----------------|
| **48 GiB MemAvailable** | Peak RSS alone was ~54 GiB while staging DiT+Gemma4; desktop + irodori + page cache need headroom beyond the bare pack size |

**Fix:** refuse submit below **72 GiB MemAvailable** with a 503 that names CUDA alternatives (`ltxv-13b-distilled:16g` / `ltxv-2b-distilled:lab`). Override only via `ZEROLLAMA_LTX_MIN_HOST_RAM_GIB` (+ `…_FORCE=1` to lower). Progress updates are **monotonic** in `training.py` / load-phase floors in `ltx_video_generate.py` so mmgp sub-step % cannot bounce the OpenAI job bar.

### WHY post-generate `os._exit` + early-complete

**Incident (astra, 2026-10-02):** `ltxv-2b-distilled:lab` wrote a valid H.264 mp4, then Wan2GP/mmgp/torch **atexit/`__del__` re-hooked** and spiked host RSS. `GET /v1/videos/:id` stalled ~1–2 min (embedded `job_status` waiting on the GIL behind SCRIPT stdout floods), then the kernel **OOM-killed `zerollama.service`** before clients saw `status=completed` / could `GET …/content`. `/tmp` tmpfs full of smoke logs (~32 GiB `h3_smoke_gen.log`) made swap pressure worse — keep `/tmp` clear on tmpfs hosts.

| Layer | Fix |
|-------|-----|
| `ltx_video_generate.py` | After copy + `TRAINING_COMPLETE`, **`os._exit(0)`** — skip destructor cascade |
| `training.py` `run_script` | On `TRAINING_COMPLETE` + artifact → **`complete_job` immediately**, then **SIGKILL** after `ZEROLLAMA_RUN_SCRIPT_TEARDOWN_GRACE_SEC` (default 2 s); ignore fail after COMPLETED |
| `training.py` stdout | **Ring-buffer** last 200 lines + **rate-limit** `SCRIPT:` journal prints — one LTX job previously produced ~11M journal lines and pinned multi‑GiB `stdout` strings inside embed `JOB_QUEUE` |

### WHY dual-GPU best-fit (+ migrate only on solvable contention)

Astra has **two 24 GiB GPUs**. Always picking “freest” empties one card but packs poorly; always pinning sticky TTS to GPU0 ignores cases where DiT must reclaim that card.

| Rule | Behavior | Why |
|------|----------|-----|
| **Best-fit** (default `ZEROLLAMA_CUDA_POLICY=best_fit`) | Among GPUs with `free ≥ need`, pick the **smallest** free that still fits | Packs compute; keeps the **largest open chunk** for the next DiT job |
| **Migrate iff solvable** | If nothing fits, move **irodori** only (`irodori-cuda.env` + SIGTERM; systemd `Restart=` reloads env). Desktop/kwin/Steam are immovable | Pay a one-time reload cost only when contention is real and fixable — not on every video job |
| **Need MiB** | `LTX_VRAM_NEED_MIB` (22B=18000, 13B=14000, 2B=8000) from Go payload | Placement must know contiguous free target; mmgp does not merge the two cards |
| Escapes | `ZEROLLAMA_CUDA_POLICY=freest`, `ZEROLLAMA_CUDA_MIGRATE=0`, `LTX_CUDA_DEVICE` / `H3_CUDA_DEVICE` / `WAN_CUDA_DEVICE` | Labs / debugging without surprising migrations |

Code: [`scripts/video/cuda_device.py`](../scripts/video/cuda_device.py) (LTX / Wan / H3 wrappers). Irodori default pack: `CUDA_VISIBLE_DEVICES=0` + optional `/mnt/ollama_img/speech/irodori-cuda.env` ([`irodori-tts.service`](../scripts/systemd/irodori-tts.service)).

### WHY jobs die at ~27% (`loading_model`)

**27% is not diffusion** — the wrapper pins Wan2GP's `loading_model` phase to a ~27% floor. Failures there are load-time, before steps run.

| Failure | Why | Fix |
|---------|-----|-----|
| `[Errno 13] Permission denied: 'loras/ltxv'` | Wan2GP creates `loras/<family>` under the repo; tree was root-owned while serve runs as `ollama` | `ensure_wan2gp_writable_dirs` + `chown` `repo/loras` (and HF `ckpts/.cache`) to `ollama`; LTX `WorkingDir` = Wan2GP repo |
| `'str' object has no attribute 'update'` (2B FP8) | Official Lightricks FP8 embeds `metadata['config']` as a JSON **string**; mmgp calls `.update` expecting a dict. 13B quanto packs use `config_base64` and are fine | `patch_mmgp_lightricks_string_config()` in [`ltx_video_generate.py`](../scripts/video/ltx_video_generate.py) forces a parsed transformer `forcedConfigPath` |
| Not 512×512 | Manifest default size for the lab tag; unrelated to the mmgp/config crash | Raise size via request/options when you want 768² |

## API

Same as Wan: `POST /v1/videos` → poll `GET /v1/videos/:id` → `GET …/content`.

| Tag | Use |
|-----|-----|
| `ltxv-13b-distilled:16g` | Quality 768×512, 6 steps, Linux 16g CUDA (Wan2GP) |
| `ltxv-2b-distilled:lab` | Wan2GP 2B FP8 (CUDA/PyTorch); default **512×512** (lab iteration size — not the failure mode) |
| `ltx2.5-22b-distilled:48g` | **Control:** 1280×704, 8 steps, A/V + start/end keyframes (`options.keyframes`) |
| `ltxv-2b-mlx:lab` | **Fast Darwin prototype:** 768×480, 17 frames, **4 steps**. Expects cartoon drift. |
| `ltxv-13b-mlx:lab` | **Darwin anime:** 1280×720, 41 frames, **8 steps**, first-frame I2V. `./scripts/video/install_ltx_mlx.sh --13b-only` |

**LTX-2.5 control:** pass one still (`keyframes[0]` → `image_start`) or two (`[0]`/`[-1]` → start/end). LTXV 0.9.8 Wan2GP tags stay T2V-only (400 on keyframes).

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

- Full LTX-2.5 research stack (non-distilled / MSR / edit-anything) as default tags
- Gradio UI as product surface
- Generating while production holds the GPU without an unload window
