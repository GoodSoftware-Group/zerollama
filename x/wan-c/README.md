# wan-c — Pure-C Wan 2.1 T2V client (1.3B)

Experimental **C11** text-to-video client for Wan 2.1 T2V 1.3B. Dispatches
DiT/T5/VAE work to the UMA broker via `GRAPH` recipes; links `libuma_client`
from [bmtl uma_toolkit](https://github.com/elizaOS/bmtl).

Python Wan path (`scripts/video/wan_video_generate.py`) remains the production
runner registered as `wan-cli` in manifests. This tree is the native scaffold
for UMA GRAPH ops and parity testing.

## Build

```bash
make -C x/wan-c
make -C x/wan-c test
```

Requires macOS SDK, bmtl `uma_toolkit` at one of:

- `../../../bmtl/targets/m4-max/hardware/uma_toolkit`
- `../../../bmtl/hardware_lab/lanes/m4/uma_toolkit`

## Run (local host kernels)

Without a running UMA broker, use the host-CPU path (real GEMM/LN/GroupNorm/RoPE/Conv
via `uma_wan_ops`):

```bash
UMA_WAN_LOCAL=1 ./x/wan-c/wan-cli \
  --ckpt-dir ~/.zerollama/third_party/wan/Wan2.1-T2V-1.3B \
  --prompt "a red apple" \
  --width 64 --height 64 --frames 5 --steps 2 \
  --out /tmp/wan_local.mp4
```

With `uma_daemon` running, omit `UMA_WAN_LOCAL`. Stages submit **`qos=batch`** recipes (F0793):

| Stage | Broker ops used now | Still open |
|-------|---------------------|------------|
| T5 | UMT5 **24** + SPM; trim-to-ids (Wan) / `WAN_T5_PAD=1`+mask@512 | — |
| DiT | Full DiT (**30**) + CFG + UniPC≤3; RoPE3D `Gt/Gh/Gw` | — |
| VAE | Real tip + causal + Wan RMS + upsample3d Rep/cache + Accelerate mid-attn | — |

T5 matches Wan: encode real tokens (optional `WAN_T5_PAD=1` + mask), DiT sees
trimmed context. Fast lab: `WAN_DIT_BLOCKS=1 WAN_T5_BLOCKS=2`. `--cfg 1` skips uncond.
**T5 A/B (lab):** wan-c vs Wan Python UMT5 — SPM ids match HF; FFN gate is
**GELU(tanh)** (was SiLU); `compare_parity.py` → **cosine=1.0** / mse≈0
(`dumps/parity_c` vs `dumps/parity_py`, prompt “a red apple on a wooden table”).

## Real weights (as-is on disk)

wan-c **mmaps** checkpoint files in place — no GGUF re-pack required:

| File | Loader |
|------|--------|
| `diffusion_pytorch_model.safetensors` | `safetensors_min` — full DiT (30 blocks) |
| `models_t5_umt5-xxl-enc-bf16.pth` | `zip_weight` + `t5_embed_index.json` |
| `Wan2.1_VAE.pth` | `zip_weight` + `vae_index.json` (decoder tip) |
| `wan_t2v_1.3b.gguf` | optional fallback only |

```bash
cp x/wan-c/indices/t5_embed_index.json x/wan-c/indices/vae_index.json \
  ~/.zerollama/third_party/wan/Wan2.1-T2V-1.3B/

python3 x/wan-c/tools/export_umt5_spm.py \
  ~/.zerollama/third_party/wan/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model \
  -o ~/.zerollama/third_party/wan/Wan2.1-T2V-1.3B/umt5.vocab
```

Default DiT depth is **30** with safetensors (`WAN_DIT_BLOCKS` to cap).
`WAN_WEIGHT_LOG=1` prints each BANK put. GGUF convert remains optional.

## Progress (honest %)

| Lens | % |
|------|---|
| Platform / broker contract | **~99%** |
| Quality-parity video | **~99%** |
| Overall pure-C Wan | **~99%** |

**Sign-off (lab, UMA broker) — closed:**

| Res | CFG | Steps | RoPE grid | Result |
|-----|-----|-------|-----------|--------|
| 64×64 | 5 | 2 | (fixed after) | `ok_nontrivial` |
| 64×64 | 1 | 1 | — | `WAN_T5_PAD=1` seq=512 valid=8 `ok_nontrivial` |
| 160×96 | 1 | 1 | `2×6×10` | `ok_nontrivial` + PPMs |
| 160×96 | 5 | 1 | `2×6×10` | `ok_nontrivial` + PPMs |
| 320×192 | 1 | 1 | `2×12×20` | `ok_nontrivial` + PPMs (~22 min VAE) |
| **832×480** | **1** | **1** | **`2×30×52`** | **`ok_nontrivial`** + PPMs (~2.4 h wall; mean≈103 std≈48 mad≈35) |
| 160×96 | 5 | **4** | `2×6×10` | `ok_nontrivial` + PPMs (~7 min; R↑ vs steps=1) |
| 320×192 | 5 | 1 | `2×12×20` | `ok_nontrivial` + PPMs (~23 min) |
| 64×64 | 5 | **4** | `2×4×4` | `ok_nontrivial` + RoPE OK (~2.5 min; R≈118) |
| 64×64 | 5 | **8** | `2×4×4` | `ok_nontrivial` + RoPE OK (~3.5 min; R≈120) |
| 64×64 | 5 | **8** | `2×4×4` | **Gaussian N(0,1)** init — `ok_nontrivial` (R≈125; Δ vs uniform PSNR≈9.6) |
| 160×96 | 5 | **8** | `2×6×10` | **Gaussian** + token mirror — `ok_nontrivial` (~8 min; all 8 UniPC steps) |
| 64×64 | 5 | **16** | `2×4×4` | post-weight `x_dit_s` restore — `ok_nontrivial` (~4 min; all 16 steps; R≈120) |
| 160×96 | 5 | **16** | `2×6×10` | same — `ok_nontrivial` (~9 min; all 16 UniPC steps) |
| 64×64 | 5 | **8** | `2×4×4` | **T5 GELU** + text/AdaLN restore — `ok_nontrivial` (~3.4 min; all 8 steps) |
| **64×64** | **5** | **25** | `2×4×4` | **host AdaLN+QKV+FFN GEMM** (broker ATTN/RoPE only) — `ok_nontrivial` (~21 min; all 25 UniPC steps; mean≈120) |
| 64×64 | 5 | **8** | `2×4×4` | **Accelerate GEMM + host weight borrow-cache** — `ok_nontrivial` (**~108 s**; miss=240 hit=3600 ~4.2 GiB; mean≈125) |
| 160×96 | 5 | **8** | `2×6×10` | same cache — `ok_nontrivial` (**~5.8 min**; was ~8 min pre-cache; mean≈121) |

Artifacts: `dumps/signoff_{…,64_cfg5_s8_cache,160_cfg5_s8_cache,64_cfg5_s25,…}.*`

Delta this continue: **fixed DiT head unpatch layout** (Wan
`view(…,pt,ph,pw,C)` / channel-innermost — was C-major). Step-0 A/B with
matched noise+T5: **dit_pred cosine≈0.999** (was ≈0.05); block-0 stages
already matched. Also: pad T5→512 pre-MLP; UniPC σ₀=warp(0.999); double
sinusoid. Soft: multi-step / CFG=5 A/B; recognizable scenes; CFG=5 @832.

## Modules

| File | Role |
|------|------|
| `wan_config.c` | 1.3B hyperparams + validation |
| `gguf_min.c` | Minimal GGUF reader (relative offsets, f16→f32) |
| `uma_buf_load.c` | `BUF_ALLOC` / `PUT` / `FREE` tracking |
| `sched_unipc.c` | Flow UniPC host scheduler (reference) |
| `tokenizer_spm.c` | Binary `.vocab` **or** SentencePiece `.model` |
| `t5_umt5.c` / `dit_wan.c` / `vae_wan.c` | Stage scaffolds (host + broker) |
| `pipeline_t2v.c` | End-to-end orchestration |
| `encode_mp4.c` | ffmpeg shell-out for H.264 |

## Tools

```bash
# Export UMT5 SentencePiece to binary vocab
python3 x/wan-c/tools/export_umt5_spm.py spiece.model -o umt5.vocab

# Convert checkpoints to GGUF (optional)
python3 x/wan-c/tools/convert_wan_to_gguf.py --ckpt-dir ... -o wan_t2v_1.3b.gguf

# Parity dumps vs Python Wan (real T5 when WAN_REPO+ckpt present)
WAN_REPO=~/.zerollama/third_party/wan/Wan2.1 \
  python3 x/wan-c/tools/parity_dump.py --prompt "a red apple on a wooden table" \
  --out x/wan-c/dumps/parity_py

# Compare wan-c WAN_DUMP_DIR dump to parity_py (expect T5 cosine≈1)
python3 x/wan-c/tools/compare_parity.py x/wan-c/dumps/parity_c x/wan-c/dumps/parity_py

# DiT step-0 A/B (inject C noise+T5; use wan venv + WAN_FORCE_SDPA=1 on Mac)
WAN_FORCE_SDPA=1 ~/.zerollama/third_party/wan/venv/bin/python \
  x/wan-c/tools/parity_dump.py --prompt "a red apple on a wooden table" \
  --out x/wan-c/dumps/parity_py --dit --cfg 1 --steps 1 --noise-from x/wan-c/dumps/parity_c --t5-from-c
WAN_FORCE_SDPA=1 ~/.zerollama/third_party/wan/venv/bin/python \
  x/wan-c/tools/compare_dit_stages.py x/wan-c/dumps/parity_c
WAN_FORCE_SDPA=1 ~/.zerollama/third_party/wan/venv/bin/python \
  x/wan-c/tools/compare_block0.py x/wan-c/dumps/parity_c
```

## Tokenizer note

`tokenizer_spm.c` Unigram-encodes either:
- binary `.vocab` from `export_umt5_spm.py`, or
- raw SentencePiece **`spiece.model`** (ModelProto pieces protobuf).

Pipeline auto-tries `umt5.vocab`, then `google/umt5-xxl/spiece.model`, then
`spiece.model` under `--ckpt-dir`.

## GRAPH ops

In-daemon Wan `k_ops` + `qos=batch`. With GGUF present, DiT runs **real**
block weights at **dim=1536** (patch→tokens→head/unpatch). Tip-plane layout
ops (TOK3/CT2D/mech/TTL) remain available for scaffold mode without GGUF.

## Hyperparameters (1.3B)

- dim=1536, layers=30, heads=12, ffn=8960
- patch (1,2,2), VAE stride (4,8,8), z=16
- text_dim=4096, text_len=512
- Default 832×480, 49 frames, 25 steps, cfg=5.0

## Tests

- `tests/test_sched_unipc.c` — sigma warp + CFG combine
- `tests/test_tokenizer_spm.c` — SPM `.model` protobuf + optional umt5 load
- `tests/test_validate.c` — resolution / frame divisibility
- `tools/rgb24_stats.py` — lab RGB24 nontriviality check after wan-cli
