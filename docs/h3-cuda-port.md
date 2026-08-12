# MiniMax-H3 CUDA engine research (from antirez/h3.c)

Research notes for a **CUDA** MiniMax-H3 engine, using [antirez/h3.c](https://github.com/antirez/h3.c) (Metal, Apple Silicon) as the reference architecture — not a hash-table port.

**Status:** research only (2026-08-12). Lab CUDA backend **API-complete** (`make all` / `int8` / `fixture` / `audio-vae` green, `has_int8_mlp=1`). Toy DiT block **CUDA↔CPU golden** parity + **real AudioVAE decode** (fp32 pack, shape+finite) shipped. Still no ~21 GB DiT packs on this host.

**Related:** [sglang-multimodal-borrowings.md](./sglang-multimodal-borrowings.md) (text/VLM, not DiT video), sibling checkouts `/root/sglang`, `/root/Wan2GP`, `/root/vllm`.

---

## 1. Verdict

| Question | Answer |
|----------|--------|
| What is h3.c? | Native **Metal** MiniMax-H3 (joint video+audio DiT): FL2VA + Ref2VA |
| Best CUDA path short-term? | **Wan2GP** pruned INT8 (+ GGUF/NVFP4 text encoder), or **SGLang Diffusion** |
| Native C CUDA port? | Implement `h3_gpu.h` behind CUDA (`h3_cuda.cu`); keep portable C host |
| This box (5080 16 GB + ~24 GiB RAM + swap)? | Full DiT+TE stage still **blocked by host RAM**; toy fixture + audio VAE OK |

---

## 2. Host probe (CT 1564 / RTX 5080)

Measured 2026-08-11:

| Resource | Value | Note |
|----------|-------|------|
| GPU | RTX 5080, sm_120, **16303 MiB** | ~6400 MiB already used by production `zerollama` + llama-server |
| Free VRAM | ~9.5 GiB | Do **not** kill :11434 / :8081 |
| Host RAM | **~24 GiB** + ~24 GiB swap | Still short of ~31 GB DiT+TE stage |
| Local weights | audio VAE fp32 (~0.6 GB) under `/tmp/h3c-research/weights/` | No DiT/TE packs; toy fixture in lab `misc/fixtures/` |
| CUDA toolchain | `/usr/local/cuda/bin/nvcc` present | OK for stub builds |

### Minimum working-set estimate (Wan2GP pruned path)

| Component | Disk (DeepBeepMeep/MiniMax-H3) | Role |
|-----------|--------------------------------|------|
| DiT FL2VA pruned rank8 **int8_convrot** | **21.06 GB** | Denoiser (best consumer default) |
| Qwen3-VL text encoder **Q2_K GGUF** | **8.49 GB** | Prompt encode (lowest TE option) |
| Video VAE fp8mix | **2.79 GB** | Decode |
| Audio VAE fp32 | **0.61 GB** | Decode |
| **Host if DiT+TE staged on CPU** | **~31 GB+** | Typical mmgp offload |

**RAM wall:** ~31 GB staged weights vs **19 GiB** RAM → swap thrash / OOM even before a full generate. h3.c-style **SSD streaming** (2 DiT blocks resident) is the only native way this class of host works; PyTorch offload alone is not enough here.

Full BF16 DiT is **66.28 GB** per variant; Qwen TE BF16 is **51.51 GB**. Official SGLang assumes multi-GPU + large host RAM.

Production listeners (:11434 / :8081) must stay up — any H3 lab work uses separate processes and does not free those ports.

---

## 3. h3.c architecture (port seam)

```text
Portable C (reuse as-is)
  h3.c / h3_host / h3_safetensors / h3_weights
  h3_dit / h3_dit_schedule / h3_text_encoder
  h3_video_vae / h3_audio_vae / h3_vision_encoder / h3_multimodal / h3_ffmpeg
        │
   h3_gpu.h   ← 93 entry points (backend contract)
        │
   ┌────┴────┐
Metal (shipped)     CUDA (to implement)
h3_gpu.m            h3_cuda.cu / .cuh
h3_shaders.metal    kernels + cuBLAS / FA / CUTLASS
h3_metal.m          h3_cuda_probe.c
h3_tokenizer.m      ICU / HF tokenizer C or stub
```

Upstream sizes (shallow clone 2026-08-11): `h3_gpu.h` 613, `h3_gpu.m` 4711, `h3_shaders.metal` 4332 lines.

Public API (`h3.h`): `h3_load_dir` → `h3_generate` with FL2VA/Ref2VA params (`steps`, `layers`, `reuse`, `ssd_streaming`, …).

---

## 4. Op checklist (`h3_gpu.h` → CUDA)

Priority **P0** = one DiT block BF16 parity. **P1** = full denoise + text. **P2** = VAEs/media. **P3** = Metal fast-path fusions (int8 / token reduction).

### P0 — context, memory, DiT core

| Op | CUDA sketch | Notes |
|----|-------------|-------|
| `h3_gpu_create` / `free` | `cudaSetDevice`, stream, workspace | Map Metal device → CUDA device 0 lab |
| `h3_gpu_is_m5` / `has_nax_mlp` / `has_int8_mlp` | capability / compile flags | Return 0 until int8 path lands |
| `h3_gpu_tensor_new_*` / `from_*` / `free` | `cudaMalloc` + host mirror optional | BF16 = `__nv_bfloat16` |
| `h3_gpu_tensor_load_*` / `read_file` / `stream_file` | `cudaMalloc` + `pread`/`cudaMemcpyAsync` | **SSD streaming** = P0 for 16 GB |
| `h3_gpu_begin` / `continue` / `submit` | stream sync / multi-stream | Metal splits DiT into 2 command buffers |
| `h3_gpu_linear_bf16` | cublasGemmEx BF16 | Hot path |
| `h3_gpu_mlp_bf16` | gemm + SwiGLU | Or `linear` + `swiglu_bf16` |
| `h3_gpu_swiglu_bf16` / `silu_mul_bf16` | custom elementwise | |
| `h3_gpu_adaln_bf16` / `gate_bf16` | custom | DiT AdaLN |
| `h3_gpu_grouped_qkv_rope_bf16` | custom layout | Checkpoint is `[head,qkv,dim]` |
| `h3_gpu_sdpa_bf16` | FlashAttention / cuDNN | Head-major optional later |
| `h3_gpu_copy_bf16` / casts | memcpy / cast kernels | |
| stats / profile | cudaEvent | Optional |

### P1 — text encoder + sampler

| Op | CUDA sketch |
|----|-------------|
| `h3_gpu_embedding_bf16` | gather |
| `h3_gpu_rms_norm_bf16` / `head_rms_norm_bf16` | |
| `h3_gpu_text_qk_rope_bf16` / `rope_text_bf16` | |
| `h3_gpu_gqa_causal_bf16` | FlashAttention causal GQA |
| `h3_gpu_add_bf16` / `sub_bf16` | |
| `h3_gpu_euler_bf16` | sampler step on device |
| patch linears (`patch_linear_bf16*`) | small gemms |

### P2 — vision + video/audio VAE (mostly F32)

| Op | Notes |
|----|-------|
| `h3_gpu_vision_qkv_rope_bf16`, `gelu_bf16`, `layer_norm_bf16`, `sdpa_bf16` | Qwen vision tower |
| `h3_gpu_conv3d_f32`, VAE pad/group_norm | Video VAE |
| `h3_gpu_conv1d_*`, `snake1d`, `alias_free_snake`, audio SDPA | AudioVAE / BigVGAN-ish |
| `h3_gpu_linear_f32`, norms, `scale_add`, `clip`, `geglu` | Shared F32 |

### P3 — Metal performance path (defer)

| Op | Role |
|----|------|
| `h3_gpu_mlp_int8_bf16`, `linear_int8_*`, `quantize_weight_int8` | M5 int8 default |
| `h3_gpu_grouped_qkv_linear_rope_int8`, `gate_adaln_quantize_int8` | Fused int8 |
| `h3_gpu_sdpa_bf16_head_major_output`, `linear_int8_head_major_bf16` | Layout fusion |
| `h3_gpu_token_pool_*` / `token_expand_*` | `--token-reduction` |
| `h3_gpu_mlp_nax_bf16` | Metal 4 TensorOps only → skip / map to CUTLASS |
| fused `gate_adaln`, `adaln_linear` | Fewer launches |

Full ordered API list: 93 symbols in upstream `h3_gpu.h` (see clone under `/tmp/h3c-research/h3.c`).

---

## 5. Wan2GP — primary CUDA H3 path

Upstream: [deepbeepmeep/Wan2GP](https://github.com/deepbeepmeep/Wan2GP) (local: `/root/Wan2GP`, tip tracked as `origin/main`). Branding: **WanGP**. Fork of Wan-Video/Wan2.1 turned into a multi-model “GPU Poor” app (Gradio UI, CLI, API, MCP).

### Why it matters vs antirez/h3.c

| | h3.c (Metal) | Wan2GP (CUDA) |
|--|--------------|---------------|
| Audience | Mac M3/M5 Max, unified memory | Nvidia (and AMD) consumer GPUs |
| Runtime | Native C + Metal / MPSGraph | PyTorch + **mmgp** offload (`mmgp==3.7.12`) |
| H3 claim | ~40 GB peak unified on M5 E2E | **5–6 GB VRAM** for 5 s (124 frames); **8–9 GB** for 15 s @ 832×480 |
| Host RAM | File-map / SSD stream DiT | Weights staged in system RAM; Profile 5 = min RAM |
| Accelerators | int8 TensorOps, reuse, layers | Spectrum, First Block Cache, Sol-Attn (SM89+ incl. **SM120**), LoRA turbo profiles |

**Recommendation:** treat Wan2GP as the **working CUDA H3 engine** for product/lab video. Keep `h3_cuda` as a long-term native port research track, not the first way to get pixels.

### H3 surface in Wan2GP (`models/minimax_h3/`)

- **FL2VA / Ref2VA**, full 33B and **pruned ~20B**
- Checkpoints from `DeepBeepMeep/MiniMax-H3` (int8_convrot, pruned rank8, W4A8 community packs)
- Text encoder: Qwen3-VL-32B (BF16 / quanto int8 / **NVFP4** / **GGUF Q4/Q2**)
- VAEs: fp16 video + **fp8mix** (lower RAM); audio VAE fp32
- Offload via `mmgp.offload` in `minimax_h3_main.py` / `pipeline.py`
- Memory priority: default **Lower VRAM**; switch to **Lower RAM** when host is tight
- CLI profiles (`docs/CLI.md`): Profile **5** = minimum RAM; `--perc-reserved-mem-max`

### CT 1564 fit

| Claim / need | This host |
|--------------|-----------|
| VRAM 5–9 GB for H3 gen | **Feasible** on 16 GB **if** production llama-server (~6 GB) is paused/unloaded by the operator |
| Host RAM for pruned int8 + Q2 TE | Still **tight** at 19 GiB; use Profile 5 + Lower RAM + Q2/NVFP4 TE + fp8mix VAE; W4A8 DiT if available |
| `mmgp` installed? | **No** in default python yet (`ModuleNotFoundError`) — need Wan2GP venv / `scripts/install.sh` |
| Torch | `2.10.0+cu128`, CUDA OK |

Do not download ~21 GB+ packs onto this CT until RAM strategy is explicit (more RAM, or accept swap thrash for a smoke).

### Other CUDA stacks

| Stack | Path | Role vs Wan2GP |
|-------|------|----------------|
| **SGLang Diffusion** | `/root/sglang` | Official multi-GPU / FP8 serve; heavier host |
| **vLLM-omni recipes** | `/root/vllm` | Serving recipes |

Borrow **schedules** from h3.c README (`--reuse`, `--layers`, `--ssd-streaming`) into any native CUDA engine; for day-one video, use Wan2GP rather than transliterating `.metal`.

---

## 6. Lab layout + status

Lab tree: **`/tmp/h3c-research/h3-cuda`** (active) with durable mirror **`research/h3-cuda/`**.

```text
h3-cuda/
  include/h3_gpu.h …
  src/cuda/
    h3_cuda_runtime.cu   # create/begin/submit + pinned staging
    h3_cuda_tensor.cu    # alloc + double-buffered pinned file-stream
    h3_cuda_gemm.cu      # linear / mlp / adaln / patch / adaln_linear
    h3_cuda_attn.cu      # sdpa + grouped QKV RoPE
    h3_cuda_dit_ops.cu   # gate_adaln, add/sub/euler
    h3_cuda_p1_text.cu   # embedding / GQA / text RoPE
    h3_cuda_p2_elem.cu   # F32/BF16 elem + norms + token_pool
    h3_cuda_token.cu     # token_pool_adaln + token_expand(+adaln)
    h3_cuda_conv.cu      # conv1d / transpose1d / weight_norm (L2)
    h3_cuda_audio.cu     # snake / alias-free / audio SDPA / vision RoPE
    h3_cuda_vae.cu       # VAE pad / conv3d / group_norm+silu
    h3_cuda_f32_dit.cu   # adaln/gate/qkv_rope/sdpa F32 + ungrouped qkv_rope_bf16
    h3_cuda_int8.cu      # portable int8 + cuBLAS INT8→I32→BF16 (scratch reuse)
    h3_cuda_stubs.cu     # empty (all entry points implemented)
  tests/… bench_linear_int8.cu …
```

**Verified on RTX 5080 (sm_120):**

| Target | Result |
|--------|--------|
| `make smoke` / `dit` / `p1` / `pack` | prior P0–pack path |
| `make p2` | silu/clip/norms, token_pool, pinned multi-chunk H2D |
| `make token` | token_pool_adaln, expand_delta/adaln, conv1d |
| `make audio` | snake, alias-free, causal SDPA, attn pool, vision RoPE |
| `make vae` | pad (reflect+zero front), conv3d, group_norm+silu |
| `make f32dit` | adaln/gate/qkv_rope/sdpa F32, video_qkv_rope, ungrouped qkv_rope_bf16 |
| `make int8` | quantize, linear_int8, head-major, mlp_int8, gate_adaln_q, qkv_int8; cuBLAS-size path |
| `make dit8` | synthetic int8 DiT block |
| `make bench8` | BF16 vs int8 linear microbench (`128×512×256`) |
| `make fixture` | toy DiT safetensors → CUDA vs CPU goldens (`misc/fixtures/h3_dit_toy_f32.safetensors`) |
| `make audio-vae` | portable `h3_audio_vae.c` + CUDA vs real FL2VA pack (shape+finite PCM) |
| `make link-dit` | portable `h3_dit.c` ↔ `libh3_cuda.a` (`has_int8=1`) |
| `make all` | full matrix green (does not require AudioVAE weights on disk) |

**Token-reduction path:** pool → residual+baseline+AdaLN; expand restores originals via delta vs baseline; expand_adaln fuses restore+AdaLN.

**AudioVAE starters:** `conv1d_f32` / stride / transpose / `weight_norm_f32` (time-major [B,L,C], OIK/IOK).

**Audio/vision (green via `make audio`):** `snake1d_f32`, `alias_free_snake_f32`, `audio_qkv_split_f32`, `sdpa_causal_f32`, `audio_attention_pool_f32`, `vision_qkv_rope_bf16`. Also fixed `weight_norm_f32` to Metal L2×magnitude.

**Video VAE (green via `make vae`):** `vae_encoder_pad_f32`, `conv3d_f32` (NDHWC + OIHWKD), `vae_encoder_group_norm_silu_f32`.

**F32 DiT helpers (green via `make f32dit`):** `adaln_f32`, `gate_f32`, `qkv_rope_f32` (ungrouped), `video_qkv_rope_f32`, `sdpa_f32`, plus ungrouped `qkv_rope_bf16`.

**Int8 (green via `make int8` / `dit8` / `bench8`):** portable per-row act / per-out weight scales. Mid/large GEMMs use **cuBLAS INT8→INT32** + scale epilogue with reused scratch (`H3_DISABLE_CUBLAS_INT8=1` → shared-A kernel). On 5080, `128×512×256` bench: BF16 ~0.013 ms vs int8 ~0.016 ms (int8 includes dynamic act quantize). `has_nax_mlp=0`. **100/100** `h3_gpu.h` entry points implemented.

**Toy fixture parity (green via `make fixture`):** `scripts/generate_toy_dit_fixture.py` writes dims matching Metal `tests/test_metal.c` (seq=32, hidden=256, …) plus CPU goldens `x.attn_out` / `x.mlp_out` / `x.h_out`. CUDA one-block smoke matches within ~6e-4 relative. Antirez MLX fixtures remain unpublished.

**On-disk pack:** `MiniMax-H3-audio_vae_fp32.safetensors` (~605 MB) + `FL2VA/audio_vae/config.json` under `/tmp/h3c-research/weights/`. `make audio-vae` (T=4 ~0.21 s, T=37 ~0.50 s) runs portable antirez decoder through CUDA — **16 submissions**, ~0.5 GiB peak alloc, non-zero finite stereo @ 32 kHz. No MLX waveform golden on this host (antirez fixtures unpublished). Do **not** download ~21 GB DiT packs here without streaming/offload plan.

**Still open:** full pruned DiT+TE stage (~31 GB); MLX golden AudioVAE / DiT parity; need ≥64 GiB host or SSD streaming end-to-end.

**Next:** optional encode smoke / round-trip; keep DiT packs off this CT.

## 7. Zerollama product fit (later)

**Product intent:** support **MiniMax H3** and **LTX** families eventually behind the existing OpenAI async Videos API (`POST /v1/videos`), not Gradio. Roadmap: **v1.4** pluggable `runner` / multi-family registry — [ROADMAP — Video generation](./ROADMAP.md#video-generation--wan-t2v-v1-shipped). Suggested order: **LTX first** (lighter host/VRAM profiles in Wan2GP), then **H3** (needs quantized TE + DiT offload; host RAM is the CT 1564 wall). **LTXV distilled slice:** [ltx-t2v.md](./ltx-t2v.md).

H3/LTX are **not** GGUF/llama.cpp modalities. Product shape:

- Same job path as Wan: training `run_script` + wrapper + VRAM broker handoff  
- Thin Go registry (`model` / `runner` → family); Python from Wan2GP patterns or sibling invoke — not vendoring `wgp.py`  
- Lab ports only for any standalone serve; never compete with `:11434` / `:8081` without operator unload  

Native `h3_cuda` remains optional long-term; day-one H3 pixels = Wan2GP/mmgp-class path.

---

## 8. Next actions

1. Raise host RAM (or move lab to ≥64 GiB box) before downloading 21 GB+ DiT packs.  
2. Keep `/tmp/h3c-research/h3.c` as Metal reference; durable CUDA lab mirror in `research/h3-cuda/`. **Full `h3_gpu.h` surface green** (`make all` / `dit8` / `fixture` / `audio-vae`). NAX stays Apple-only.  
3. Optional: Wan2GP dry-run on a larger host with pruned INT8 + Q2 TE + Sol-Attn (SM120).  
4. Do not compete with production VRAM on :11434 without an explicit unload window from the operator.  
5. Native path next: MLX golden parity when fixtures exist; SSD streaming for DiT only on a larger host.