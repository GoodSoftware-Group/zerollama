# video-c — Pure-C multi-family video (`x/video-c`)

Local text-to-video via a **strict C11** `video-cli` (symlink `wan-cli`) that:

1. Reads **GGUF** / safetensors weights (Wan: converted offline; H3: VAE + pruned DiT)
2. Holds tensors in a **compute backend** (named buffers + GEMM/ops)
3. Runs DiT / VAE / T5 through that backend (Wan generate shipped; H3 host `--generate` shipped, slow)
4. Encodes frames with system **`ffmpeg`** (H3 audio decode writes WAV; H3 video decode writes mp4/ppm)

Compat: `x/wan-c` → `x/video-c` symlink. Old docs that say “wan-c” mean this tree.

## Product rule — client-optional runner

Clients call `POST /v1/videos` with a **model tag** only. They do **not** choose Python vs video-c.

| Who | Chooses |
|-----|---------|
| **Client** | Model name + prompt / size / … |
| **Manifest** | `modality_backends.video_generation` + `video_generation.*` |
| **Operator** | Wan **Python** tags (`wan2.1-t2v:1.3b`) vs Wan **C** tag (`wan2.1-t2v-c:lab` with `backend_paths.video_cli`). Env `ZEROLLAMA_VIDEO_CLI` still forces C on any Wan tag. |

## Families

| `--family` | Status |
|------------|--------|
| `wan` (default) | Full T2V generate (UMA / host / CUDA twin lab) on M4 Max |
| `h3` | **Host vertical (M4):** `--info`, AudioVAE, video VAE, packed DiT `--generate` (pruned int8 ConvRot; host-slow) |

H3 rematch references (not product runners):

| Sibling | Role |
|---------|------|
| [antirez/h3.c](https://github.com/antirez/h3.c) (`../h3.c`) | Native Metal/C — host layout we vendored; full Metal generate |
| [mrbizarro/minimax-h3-mlx](https://github.com/mrbizarro/minimax-h3-mlx) (`../minimax-h3-mlx`) | MLX Python — bit-exact packing/scheduler/AdaLN/VAE/TE vs diffusers; best dump source for video-c rematch |

Weights (`MiniMaxAI/MiniMax-H3` / `FL2VA`) stay operator-supplied. FL2VA and Ref2VA share the same weight blobs (mlx README); only pipeline metadata differs.
Mac lab snapshot: `~/.zerollama/models/MiniMax-H3` (`FL2VA/audio_vae` ~0.56 GiB +
`video_vae/source` ~9.7 GiB + tokenizer/config). Stock bf16 DiT + 32B TE (~62 GiB each)
do not fit. Pruned FL2VA DiT is on disk:

`~/.zerollama/third_party/h3/dit/MiniMax-H3-FL2VA-pruned_int8_convrot.safetensors`
(~20.6 GiB, `DeepBeepMeep/MiniMax-H3`, `quantization_format=int8_convrot` /
`convrot_format=comfy_quant_v1`). `h3_st_store_load_f32` dequants I8 + `weight_scale`
+ **regular** Hadamard un-rotate (`gs=256`, comfy-kitchen `_build_hadamard`, not Sylvester FWHT).
Host `--generate` streams one block at a time. 768² × 50 layers is tens of minutes on M4.

### H3 Darwin / M4 Max (current)

```bash
make -C x/video-c test                 # weightless rematch
make -C x/video-c test-h3-weights      # AudioVAE + video VAE if pack present (~5s+; ViT decode tens of s)
./x/video-c/video-cli --family h3 --info -d ~/.zerollama/models/MiniMax-H3
./x/video-c/video-cli --family h3 --decode-audio -d ~/.zerollama/models/MiniMax-H3 -o /tmp/h3.wav
./x/video-c/video-cli --family h3 --encode-audio -d ~/.zerollama/models/MiniMax-H3
./x/video-c/video-cli --family h3 --encode-video -d ~/.zerollama/models/MiniMax-H3
./x/video-c/video-cli --family h3 --decode-video -d ~/.zerollama/models/MiniMax-H3 -o /tmp/h3_vae.mp4
# lab canvas (not 768×50L). Omit -o to skip VAE. Full 50L is correct (verified
# against ComfyUI _forward); a truncated layer count breaks audio (see above).
./x/video-c/video-cli --family h3 --generate --width 128 --height 128 --layers 50 --steps 8 --reuse 2 \
  -d ~/.zerollama/models/MiniMax-H3
./x/video-c/video-cli --family h3 --tokenize --prompt "A red fox walking through snow"
./x/video-c/video-cli --family h3 --present --prompt "A red fox walking through snow" --pictures 1
./x/video-c/video-cli --family h3 --embed --prompt "A red fox walking through snow"
```

H3 tokenize is a thin wrap of sibling BMTL (`bmtl_tokenizer_load_bmtl`, Qwen2 pretok
+ NFC). Do **not** port antirez ObjC/ICU. Export `tokenizer.json` with
`export_bmtl_tokenizer.py --scheme qwen2` to
`~/.zerollama/third_party/h3/minimax_h3.bmtl_tok`. Encode rematch:
antirez `tests/test_tokenizer.c` IDs. Decode is not in BMTL (encode-only).
FL2VA presentation (`h3_present_fl2va`): `"<Picture i>: "` + vision_start +
`image_pad`×merged grid + vision_end + prompt; AdaLN tags (vision block = video);
mRoPE `[3,T]` rematch of antirez `h3_multimodal.c`. `--embed` runs that then a
hash 4B hidden through ClipProj (no TE shards).

### H3 serve daemon — resident weight store (cold-start fix)

H3 host generate is cold-start-bound: dequantizing the ~22 GB `pruned_int8_convrot`
DiT (f32 expands ~4×) takes minutes. The answer is a **resident daemon** that keeps
every dequantized weight cached across requests.

```bash
WAN_PROFILE=1 H3_MLOCK=1 ./x/video-c/video-cli --family h3 --generate \
  --serve-sock /tmp/h3_serve.sock -d ~/.zerollama/models/MiniMax-H3
```

- **Why `H3_MLOCK=1`:** the caches total ~57.7 GiB (DiT 38 + TE 10.5 + vvae 9.0 +
  avae 0.24). On a shared 128 GB Mac they get **swapped out between requests** —
  the "slow second request" was swap-in, not reload. `mlock` pins them; without it
  the store persists but pays page-in.
- **Request** (tab-separated, 8 fields back-compat; reuse/adaln optional):
  `out_mp4 \t prompt \t frames \t width \t height \t seed \t steps \t layers \t reuse \t adaln_t_sigma`
  → `ok\n` / `err: …\n`.
- **Measured:** warm request ~13 s vs ~224 s cold — **~17×**, signature bit-identical
  (`latent_rms=29.3413 a_rms=1.45345`). Per-request compute: DiT ~3.1 s, audio VAE
  ~1.8 s, text_cond ~1.4 s, video VAE ~0.9 s, encode ~0.14 s; video+audio VAE
  overlap (`H3_PARALLEL_VAE=1`).
- **Why so fast:** all matmuls Accelerate `sgemm`; independent loops
  `dispatch_apply`-parallelized bit-exactly; weights served as immutable `const`
  pointers (`h3_st_store_get_f32`) — never copied per request. Rules in
  `x/video-c/AGENTS.md`.


Vendored host code: `family_h3/h3_host.c` (layout, sigma schedules, canvas) + `h3_reuse.c`
(velocity reuse: `--reuse 2` skips odd DiT evals and extrapolates like antirez)
+ `h3_st_store.c` (multi-shard safetensors inventory via `safetensors_min`;
`h3_st_store_load_f32`, optional `h3_st_store_open_ex(..., recursive=1)`)
+ `h3_audio_vae_host.c` / `h3_audio_vae_decode.c` (BigVGAN host decode + encoder)
+ `h3_dit_host.c` (timestep sinusoid, 3-axis RoPE, `condition_proj` / linear / SwiGLU / concat QKV)
+ `h3_video_vae_host.c` / `h3_video_vae_encode.c` / `h3_video_vae_decode.c` (CNN encode + streamed ViT decode)
`--info` probe (exit 0) needs `FL2VA/transformer/config.json`, tokenizer, and
`audio_vae` + `video_vae/source` shards. DiT + text_encoder shards are **optional**
for probe (required later for generate). Prints host geometry for 512² / 22-frame.

Schedule rematch: antirez `h3_schedule_build(20)` ≡ minimax-h3-mlx scheduler
`linspace(1,0,n=21)` + shift 12/3 (`tools/dump_h3_mlx_schedule.py`, `test_h3_st_store`).
Packing rematch: frame align + `h3_adapt_canvas` ≡ MLX
(`tools/dump_h3_mlx_packing.py`, `test_h3_packing_mlx`).
Audio VAE rematch (no weights): hop=800, Kaiser-sinc, plus antirez Metal-primitive
host refs (weight-norm / Conv1d / alias-free Snake) in `test_h3_audio_vae_host`.
Real-weight host decode: `h3_audio_vae_decode_host` against
`~/.zerollama/models/MiniMax-H3/FL2VA/audio_vae` (`make test-h3-weights`,
T=4 ~5 s CPU; skips if weights absent; mean/rms/absmax regression). CLI:
`--decode-audio -o out.wav`. Host encode (`h3_audio_vae_encode_host`) pads PCM
to hop=800, runs DAC residuals + strided downsample + pre_block (LN/QKV/causal
SDPA/GeGLU) + `mean_proj` denorm → `[32,2,T]`. CLI: `--encode-audio`.
DiT extras: `h3_dit_linear`, `h3_dit_timestep_mlp`, `h3_dit_swiglu_ffn`,
`h3_dit_qkv_split` (default concat / Comfy; `H3_QKV_INTERLEAVED=1` for
`(heads,3,head_dim)`). Video VAE geometry:
spatial 16 / temporal 4 / 24 ch / clip 17 / token_drop 3 / decoder 36×2048.
Host CNN encode (`h3_video_vae_encode_host`) against `FL2VA/video_vae/source`
(32²×1 smoke, no 256-tile blend). CLI: `--encode-video`. Host ViT decode
(`h3_video_vae_decode_host`) streams one of 36 blocks (~50–130 MiB weights at
a time); **T=2 → 5**, **T=7 → 22**, **T=12 → 39** frames at 2×2 latent (32² px)
with antirez 5-frame temporal overlap between 7-token chunks. Encode 32²×1 (or
tiled 48×32 with `H3_VAE_TILE_PIXELS=32`), repeat last time to T=2, then decode
Encoder spatial tiles reuse one weight load. Optional `-o frame.ppm`
writes RGB frame 0. Spatial tiles match antirez (256 px, 64 px overlap; 512² → 3
tiles). Host ViT **refuses tiles > 16×16 latent** unless `H3_VAE_ALLOW_LARGE=1`. Optional `-o out.mp4` uses ffmpeg when present (`PATH`, then
`$HOME/.homebrew/bin/ffmpeg`, `/opt/homebrew/bin/ffmpeg`); otherwise a playable **AVI**
(RGB24, optional AudioVAE PCM) at `out.avi`. `.ppm` still writes frame 0.
`--encode-video -o` encodes RGB (synthetic or `--in frame.ppm`), pads latent to T=2, decodes, and writes
the same media. AdaLN **curve-table** (`adaln_t_table` `[1001,64]`): data-time \(t=1-\sigma\) in
\([0,1]\), lerp, then `adaln_proj` with **no SiLU** (Comfy `apply_silu: false`;
the table already holds rank-k `silu(time_embedder(t))`). `H3_ADALN_SILU=1`
restores a second SiLU (wrong for this pack). Comfy oracle (int8 pack, NVFP4 32B TE, 1344×768, 5 frames, 8 steps):
**photoreal fox**, not gray (`ComfyUI/output/video/h3_oracle_t2v_00001_.mp4`).
The correct model runs **all 50 blocks**; the old 24-layer default was an **audio
bug** and has been removed. Verified against ComfyUI's own `_forward`
(`comfy/ldm/minimax/model.py` on this exact pruned int8 export, MPS bf16): the
host stage path matches the reference — raw audio hidden `h_audio_rms` ~6–8e3,
final `RMSNorm` out ~0.37, curve-table `scale_rms=0.852` / `shift_rms=0.010`
(bit-identical to Comfy), `ha_rms≈0.3`, and **audio velocity rms ≈ 0.35–0.7**
vs host 50L `vel_audio≈1.0`. Truncating to 24 blocks cuts the residual stack, so
the final AdaLN/RMSNorm sees a wrong hidden state and the audio velocity
explodes ~20–70× (24L `vel_rms≈17–21` vs 50L `≈1–2`), integrating to
`a_rms=45` and a **~93%-clipped waveform**. Host 50L now yields
`latent_rms=1.18298 a_rms=0.504888` and `clipped=0/12800` on the regression
gate. The "rank-1 / per-ch_std collapse" reading of 50L was a misdiagnosis:
low `per-ch_std` (0.2–0.7) at small canvases is the correct model's on-manifold
behavior — the model is trained for the 768²+ canvas (Comfy oracle is
1344×50L), and tiny 32²/128² packs are not visually meaningful. Host QKV split
defaults to **concat** (Comfy `Attention.split`);
`H3_QKV_INTERLEAVED=1` restores MLX `(heads,3,hd)`; both are numerically fine
at 50L. L0 wiring was verified clean against Comfy on the same packed `x`
(`H3_DUMP_L0` + `tools/h3_l0_comfy_rematch.py`): AdaLN modulate **2e-8**,
f32-dequant QKV **7e-7**, full block `y` **2e-4**; Kitchen `int8_linear` QKV vs
host ~0.02 rms. `x_in_rms` growing to ~6–7e3 by L49 is **expected** (the final
RMSNorm rescales it); `gate_msa` crossing ~1 at L25–L30 is a trained behavior,
not a blow-up. Comfy
`return [-video_out, -audio_out]` plus CONST Euler
(`denoised = x − σ·out`, `x += d·(σ′−σ)`) is the same update as host
`x += (σ−σ′)·head` on the **unnegated** head — not a sign bug.
`H3_DIT_BF16_ACT=1` / `H3_DIT_ACT_INT8=1` (Comfy `int8_linear` activation
fake-quant on DiT GEMMs whose `in_dim` is a multiple of 256) leave the 50L
velocity at ~1.3–1.5 — compute dtype is not the issue.
`H3_DUMP_L0` / `H3_DUMP_LAST` / `H3_GATE_CLIP` / `H3_RES_CENTER` /
`H3_FFN_CLIP_TEXT` are lab-only debug switches (default off), unchanged.
QK-RMSNorm + **split-half RoPE** matches Comfy `apply_rope_split_half` (pairs `(i, i+half)` on the first 96 of 128; pass-through tail). Host now prefers pack `rope.inv_freq` over the theta formula.
ClipProj rematch: affine + `sink_out` + fp16 residual MLP (`mlp.0`/`mlp.2`) against
control + `mmh3-4b-ClipProj-celeb-mlp` under `~/.zerollama/third_party/h3/clipproj/`
([h3-clipproj.md](./h3-clipproj.md)).
DiT host rematch: timestep embed + RoPE + `condition_proj` + patchify/unpatchify
`(1,2,2)` + audio pack/unpack + SiLU/RMSNorm + AdaLN modulate/`split_final` +
ImageNet pixel norm in `test_h3_dit_host` / `test_h3_adaln_host`. Spatial/temporal
RoPE grids public on `h3_host` (`test_h3_host`). `--info` prints DiT constants,
ModulationCache MiB, and optional ClipProj probe.

Packed DiT + rectified-flow: `h3_dit_denoise` (video shift 12 / audio 3;
per-row AdaLN `t = 1-σ`). Default **Euler**. Lab `H3_SAMPLER=res_multistep` is
Comfy `sample_res_multistep` (η=0): first / last-σ=0 steps are CONST Euler
(`denoised = x+σv`); middle steps are the arXiv 2308.02157 2nd-order update.
Oracle fox used this sampler, not host Euler. Tiny pack smoke: `H3_CONVROT_PACK=1` in
`test_h3_dit_forward`; CLI `--family h3 --dit-denoise` (4-token T2VA layout:
text + stereo audio_t=1 + 1 video patch, Comfy RoPE grids — not all-zero
positions). AdaLN `unique_t` is **sorted** like Comfy.
Host perf (2026-08): one **workspace arena per forward pass**
(`h3_dit_ws_create`) feeds all blocks — the block path moves ~800 MB of
activations per block-step and Darwin page-faults fresh mallocs of that size;
RoPE cos/sin are built **once per forward** (`h3_dit_rope_tables`) and applied
in place; the per-layer probe/stats prints are gated behind `H3_DIT_DEBUG`;
big weights (`adaln_proj`, ~49.5 MB/block) are read by pointer from the weight
cache instead of a per-call memcpy. All bit-exact — gate string unchanged.
Opt-in `H3_SDPA_BLAS=1` runs attention as per-head `cblas_sgemm` QK^T/P·V +
vDSP softmax (no gather; sgemm walks packed Q/K/V via `lda=heads·hd`):
768² 1-step **1119 → 744 ms/layer (1.50×)**, `vel_rms` identical to 4 digits,
tiny-gate signature drifts only in the 5th digit (accumulation order). Not
default while the gate is bit-exactness-tracked.
`--family h3 --generate --prompt TEXT` maps tokens through Qwen3-VL-4B (or
hash TE) + ClipProj into DiT text rows and keeps presentation **AdaLN tags**
(vision pads = video). Omit `--prompt` for dummy sine cond. H3TE dumps may
append `uint8[nt]` tags after the floats (T2VA text-only is all `1`).
T2VA pack is Comfy `PackedLayout`: **text | audio | video**, same RoPE grids (tiny
`seq=30` `Σt=339.666…`; 128² `seq=60` `Σh=384` `Σw=576`). **32² has no spatial
RoPE** (2×2 latent / 2×2 patch → one site, `h=w=0`); only the two frame `t`
values differ. 128² fox 24L 1-step (`H3_DIT_DEBUG=1`): video→text `mass_txt` is
**not** zero — L1 **0.66**, L13 **0.63**, L19 **0.48**; L20 is video-only
(`mass_vid=1` `mass_txt=0`); L23 **0.17**. Mid-stack also attends audio (L16
`mass_vid+txt≈0.07`). Same 128² 24L 1-step **dummy** (`nt=12` sine): mean
`mass_txt=0.058` vs fox **0.202**. L1 is a text-slot look for both (dummy
**0.74**, fox 0.66); from **L8–L19** fox keeps reading text (L13 **0.63** vs
dummy **0.03**) while dummy stays video. 4B cond **steers attention**,
not just occupancy. 4B `nt=6` **is read**. Do **not** use Wan UniPC
(`x0 = x − σ v`). `--generate` defaults to **50 DiT layers** (the full model,
verified against ComfyUI `_forward`; the old 24-layer default was the audio
bug — see above); `--dit-denoise` stays 1-layer (smoke test). Host f32 50L
velocity matches Comfy (~1–2), so the latent is on-manifold
(`latent_rms≈1.2`, `a_rms≈0.5`) and the audio decodes clean
(`clipped=0/12800`). `--width 768` packs T=2, latent 48×48, nv=1152 (host-slow).
`H3_WEIGHT_CACHE_MB` defaults to 81920 (80 GiB, I8 ConvRot only); `0` disables.
Darwin TE: `h3_qwen_te_4b_forward` streams Qwen3-VL-**4B-Instruct** tap **24**
(ClipProj calibration; no final RMSNorm) into ClipProj. Full last-hidden:
`H3_QWEN_TE_LAYERS=36 H3_QWEN_TE_FINAL_NORM=1`. Hash-embed is the fallback.
T=2 VAE decode tiles at **256 px** (16×16 latent). The old picker preferred 320 px
on 768² and tripped the ViT guard. Larger tiles still need `H3_VAE_ALLOW_LARGE`.
Stderr: `video-c: vae [Ns] tile i/N vit L/36` (wall seconds). `H3_VAE_QUIET=1` silences it.
After DiT, `--generate` logs `latent spatial WxH std=… ac1=…` (t=0, mean over channels). A `.pgm` sidecar is written next to `-o *.ppm`, or `H3_LATENT_PREVIEW=path`. `--steps>=2` uses the MLX linspace sigma grid (`h3_serving_schedule_build`); 1-step stays the antirez `[1,0]` jump.
**1-step `per-ch_std` is velocity energy, not a picture.** The old lab numbers
below are 24L (truncated): the huge velocity (~17–21 vs ~1–2 at 50L) walked the
latent ~10× past the VAE's ~1.3 encoded scale, so 24L decode was magenta/gray
"not a fox" — an artifact of the truncation, not the field direction. The 50L
default fixes it: tiny 32² 2-step gate is `latent_rms=1.18298 a_rms=0.504888`
(`per-ch_std` small, `ac1≈1` — on-manifold, not a picture either; a 2×2 latent
cannot hold a fox). `final_layer.video_out` is not a host GEMM miss:
4-token `--dit-denoise --layers 1` `vel_rms=75.41` matches numpy
`norm→(1+scale)+shift→video_out` on `H3_DUMP_LAST` **to 0.00**. Lab
`H3_VEL_RMS=1.3` (default off) rescales each step's `v` to encoded-orange RMS;
at 24L it landed on-manifold (`latent_rms=0.62`) but decoded gray-beige —
because the 24L field direction was already wrong. Do not ship velocity RMS
scaling (50L is correct without it).
**50L is the shipped model (open).** The old "24L / 50L science — closed"
conclusion (gate `latent_rms=17.2124 a_rms=45.3436`, "host 50L is gray", "do not
raise host default layers") was reversed: every one of those runs used the
truncated 24-block stack. Its own data already showed 50L velocity ≈ 1–2
(correct, matching ComfyUI `_forward`) while 24L ≈ 17–21 (20× too large) —
which is exactly why the audio clipped at 93%. Comfy's 50 DiTBlocks give the
same `L49 x_rms≈6.6e3`, `video_head_rms≈1.9` class as host 50L: that is the
correct model, and the final RMSNorm absorbs the residual magnitude. Comfy
**simple/8 shift-12 sigmas match** host linspace; default stays Euler.
**1344×768 fox canvas (nv=2016 seq=2071, oracle `nt=39`, seed 42, Comfy latents injected):**
pack/RoPE match Comfy (packed diff ~3.6e-7, `pos.bin` exact). Host **24L**
`vel_rms=16.88` (off-manifold, energetic speckle); host **50L** `vel_rms=1.926`
(on-manifold). Comfy-native L0 on the same canvas is `x_rms=62.0233` vs host
`x_out_rms=62.03`. Canvas confirms: 24L is the broken count, 50L is correct.
Oracle mp4 prompt JSON is this int8 pack, `weight_dtype=default`,
Qwen3-VL-32B NVFP4, 1344×768×5, `res_multistep` + `simple`/8, `BasicGuider`.
Comfy-native **`forward` + `audio_scale=4`** at 128² 50L (MPS fallback): σ=1
**`video_out_neg_rms=1.84982`**. Layer vid-div matches host: L23 **0.5910/986** vs
host **0.5914/979**; L47 **0.780/3979** vs **0.790/4019**; L48 **0.769/4960** vs
**0.777/4972**; L49 **0.916/6643** vs **0.914/6578**. Comfy L49 `mean_cos=0.916` —
late token-token cosine is **not** a host bug. Comfy `video_out` patch-row
`mean_cos=0.677` vs host **0.659**. Comfy spatial stats are on **model `video_out`**
(`ac1=0.043` `per-ch_std=1.297` `std=0.242`); host `--generate` `latent spatial`
`ac1=0.889` is the **post-Euler latent** (`x += σ v` at 1-step), not `v`. Compare
`vel spatial` on unpatched `vpred` to Comfy `ac1`. Host 128² 50L 1-step:
`vel spatial std=0.226 ac1=-0.139 per-ch_std=1.291` vs Comfy `0.242 / 0.043 / 1.297`
(independent RNGs). Same host noise injected into Comfy `forward`: **cosine=-0.9986**
(`cosine_neg=0.9986`) — host `v` is Comfy `video_out` with opposite sign. That matches
CONST `denoised = x − σ·model_out` vs host `x + σ·v`. Do not flip the Euler step.
Post-step `latent spatial ac1=0.889` is Euler `x += σ v`, not a dead field.
Host **8-step `H3_SAMPLER=res_multistep`** 128² 50L oracle TE: `latent_rms=1.047`
`a_rms=0.861`; last-step `vel spatial ac1=0.008` `per-ch_std=1.108` (field stays
alive). VAE decode writes a 128² frame (8×8 latent — not a fox, not gray). **256²**
same recipe (`nv=128` `seq=183`, ~4 min): `latent_rms=0.879` `a_rms=0.744`; every
step `vel spatial ac1≈0.00–0.05`. Decode is `/tmp/h3_256_res8.ppm`. Same recipe **`-o .mp4`**: `/tmp/h3_256_res8.mp4`
(5 frames + PCM, `clipped=3/12800`). **512²** same recipe (~9 min, `nv=512`):
`latent_rms=0.916` `a_rms=0.439`; last `vel spatial ac1=0.127`. Wrote
`/tmp/h3_512_res8.mp4`. Audio decode is near-silent (`clipped=0` `rms=0.002`)
because the **512 DiT audio latent is off-manifold** (same dump→`--decode-audio`:
wav `rms=0.002`), not mux. 256² 8-step dump decodes `rms=0.40`. **`H3_AUDIO_CARRY`**
matches Comfy `process_latent_in` (audio ×4), uncarry for the network, Euler/res
on carried x, then ÷4. Unset: on when `nv>8` (32² gate stays native
`latent_rms=1.18298 a_rms=0.504888`). Host wrap-on-native-x (no ×4) is obsolete:
256² oracle 8-step wav **`rms=0.761`** (`a_rms=2.79`, `clipped=3848/12800`; video
`0.878` unchanged vs 0.879); 512² wav **`rms=0.661`** (`a_rms=2.26`,
`clipped=2076/12800`; video `0.917` vs 0.916) vs old carry `0.100` / no-carry
`0.002`. `H3_AUDIO_CARRY=0` forces the native audio ODE.
**768²** product TE (Qwen3-VL-**4B**+ClipProj, `nt=6`, same 8-step carry): `/tmp/h3_768_4b.mp4`
is still a sharp fox (`latent_rms=0.917` `a_rms=2.67` wav `rms=0.676`). Oracle 32B dump is
not required for a readable 768 clip. **768²** 8-step 32B dump (`nv=1152`): `latent_rms=0.899` `a_rms=2.70`;
last `vel spatial ac1=0.115`. Wrote `/tmp/h3_768_carry_in.mp4` (sharp fox; mux wav
`rms=0.681` `clipped=533/12800`). Prior wrap-on-native clip was `/tmp/h3_768_res8.mp4`
(`latent_rms=0.905` `a_rms=1.44` wav `rms=0.130` `clipped=10`). Video field unchanged.
**1344×768** 8-step oracle canvas (`nv=2016`, carry ×4-in): `latent_rms=0.903`
`a_rms=2.34`; L0 `x_out_rms=62.48` (Comfy 62.02); last `vel spatial ac1=0.143`.
Wrote `/tmp/h3_1344x768_carry_in.mp4` (sharp fox; mux wav **`rms=0.703`**
`clipped=1064/12800`). Old wrap `/tmp/h3_1344x768_res8.mp4` was `a_rms=1.14` wav
`rms=0.031`. Gate still `latent_rms=1.18298 a_rms=0.504888`.



LTX product remains Wan2GP ([ltx-t2v.md](./ltx-t2v.md)); no `video-c --family ltx` yet.

## Ownership (Darwin vs CUDA)

| Track | Owner | Scope |
|-------|--------|--------|
| **Darwin** | Mac / UMA path | Default `make -C x/video-c`; H3 `--info`; product wire |
| **CUDA** | Parallel owner | `backend_cuda*`, `make cuda-lab` / rematch, [cuda-uma-toolkit.md](./cuda-uma-toolkit.md), [research/h3-cuda/](../research/h3-cuda/) |

## Backends (wins + unlocks)

| Backend | Host | Role |
|---------|------|------|
| **UMA** (default on Darwin) | Mac `uma_daemon` | Production Apple path — GRAPH recipes + UmaBuffers |
| **CUDA in-process** | Linux lab (`backend_cuda.c`) | Twin for 5080 CTs — see [cuda-uma-toolkit.md](./cuda-uma-toolkit.md) |
| **Host local** | `UMA_WAN_LOCAL=1` | No broker; CPU kernels |

Residency: [`dit_pager`](./dit-pager.md) (`WAN_DIT_RESIDENT`) — N-block LRU above the backend.

Lab CUDA smokes (CT; other owner):

```bash
export LD_LIBRARY_PATH=/root/nvidia-host:/usr/lib/ollama/cuda_v13:/usr/local/cuda/lib64
make -C x/dit_pager test
make -C x/video-c cuda-lab
make -C x/video-c cuda-block0-rematch
make -C x/video-c cuda-multiblock-rematch
make -C x/video-c cuda-latent-unipc-rematch
```

Never bind lab tools to production `:11434` / `:8081`.

## Prerequisites (UMA / Mac)

- **`uma_daemon` running** (one per Mac). Do not start a second broker.
- Wan 2.1 T2V 1.3B under `~/.zerollama/third_party/wan/Wan2.1-T2V-1.3B` for Wan generate
- ffmpeg on PATH (or `~/.homebrew/bin/ffmpeg` — video-c looks there)

## Install

```bash
./scripts/video/install_wan_c.sh
source ~/.zerollama/third_party/video-c/env.sh
```

Lab without broker:

```bash
UMA_WAN_LOCAL=1 ./x/video-c/video-cli --family wan --ckpt-dir … --prompt "…" --out /tmp/t.mp4
./x/video-c/video-cli --family h3 --info -d ~/.zerollama/models/MiniMax-H3
./x/video-c/video-cli --family h3 --decode-audio -d ~/.zerollama/models/MiniMax-H3 -o /tmp/h3.wav
```

## Zerollama API

Wan C product tag: `wan2.1-t2v-c:lab` (`./scripts/video/register_wan_models.sh`). Manifest points at repo `x/video-c/video-cli` + `~/.zerollama/third_party/wan/Wan2.1-T2V-1.3B`. Jobs run `wan_c_generate.py` → `video-cli --family wan`. On Darwin, serve sets `UMA_WAN_LOCAL=1` unless `UMA_SOCK` / `UMA_WAN_LOCAL` is already set (do not bounce the user’s `uma_daemon`). Python Wan tags stay the default when `video_cli` is unset. Override the binary with `ZEROLLAMA_VIDEO_CLI`.

H3 tags: `minimax-h3-tiny:lab` (5×32², **50 DiT layers**, 2-step Euler gate) and `minimax-h3-768:lab` (5×768², **50L**, 8-step `res_multistep`). Profiles `h3-tiny-t2va` / `h3-768-t2va`. Register with `./scripts/video/register_h3_models.sh`. Jobs run `wan_c_generate.py` → `video-cli --family h3 --generate`. Manifest `backend_paths.video_cli` is repo-relative (`x/video-c/video-cli`); `h3_ckpt_dir` defaults to `~/.zerollama/models/MiniMax-H3`. Override the binary with `ZEROLLAMA_VIDEO_CLI`. The 768 tag defaults to a 4h job timeout (tiled VAE decode) and sets `H3_SAMPLER=res_multistep` when steps ≥ 8. Truncating layers (old 24L default) explodes velocity; do not ship it. Request `options.layers` to override (cap 50).

## Parked

- Sibling Metal `../h3.c` BF16 directory shards (this lab uses the 21 GiB pruned int8 ConvRot pack). Comfy `pruned_bf16` is **[1025,8] AdaLN** — video-c refuses it (`H3_DIT_ST`).
- Shell-out to antirez `./h3` (blocked: no BF16 DiT on disk; do not pull 32B TE for that path)

## Parity / tools

```bash
python3 x/video-c/tools/export_umt5_spm.py $CKPT/google/umt5-xxl/spiece.model -o umt5.vocab
python3 x/video-c/tools/parity_dump.py --prompt "a cat" --out /tmp/wan_dumps
```

Speed notes: [wan-c-speed-gap.md](./wan-c-speed-gap.md) · [wan-c-blowaway-ideas.md](./wan-c-blowaway-ideas.md).
