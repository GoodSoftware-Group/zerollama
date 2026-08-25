# MiniMax-H3 MLX borrowings (Darwin rematch)

**Sibling:** `../minimax-h3-mlx` — [mrbizarro/minimax-h3-mlx](https://github.com/mrbizarro/minimax-h3-mlx) (fork of PipeNetwork).  
**Product path:** Pure-C [video-c](./video-c.md) on M4 (`--family h3`), not this Python stack as the serve runner.  
**Also:** [antirez/h3.c](https://github.com/antirez/h3.c) for Metal/C host + generate reference.

## Why it matters for video-c

| MLX finding | video-c implication |
|-------------|---------------------|
| Packed 1-D sequence `[text \| cond \| audio \| video]` | Same geometry as vendored `h3_layout_build` (already rematched) |
| Video σ-shift 12 / audio 3 | `h3_schedule_build` / serving schedule |
| AdaLN precompute drops **13B** (~26 GiB) | Plan ModulationCache before resident DiT; do not keep `adaln_proj` live every step |
| TE uses **unnormalized** hidden after **layer 50/64** | Skip layers 50–63 + lm_head + vision tower (~50 GiB vs 66 GiB) |
| FL2VA ≈ Ref2VA weights | One weight conversion / mmap plan covers both tasks |
| CFG distilled (no negative / guidance) | No dual forward for CFG |
| Attention FLOPs dominate wall | Quant helps fit, not 5× speed; streaming + reuse still primary |

## Rematch surfaces (use for dumps, not Gradio)

| Test / module | Use |
|---------------|-----|
| `tests/test_packing_parity.py` | Cross-check layout tags / (t,h,w) vs `test_h3_host` / `test_h3_packing_mlx` |
| `tests/test_dit_parity.py` / `test_dit_smoke.py` | Block forward goldens when weights present |
| `tests/test_audio_vae_parity.py` | First real-weight vertical (~0.6 GiB); video-c host decode: `test_h3_audio_vae_decode` |
| `tests/test_video_vae_parity.py` | After audio |
| `tests/test_text_encoder_parity.py` | Layer-50 pre-norm assert |
| `minimax_h3_mlx/adaln.py` | Cache + drop pattern |
| `minimax_h3_mlx/scheduler.py` | Sigma trajectory — rematched in video-c `test_h3_st_store` / `dump_h3_mlx_schedule.py` |
| canvas / frame align | `dump_h3_mlx_packing.py` → `fixtures/h3_mlx_packing.json` / `test_h3_packing_mlx` |
| Kaiser-sinc / hop=800 | `dump_h3_mlx_audio_vae.py` → `fixtures/h3_mlx_audio_vae.json` / `test_h3_audio_vae_host` |
| weight-norm / Conv1d / Snake | antirez `tests/test_audio_gpu.c` host refs → `test_h3_audio_vae_host` |
| `minimax_h3_mlx/adaln.py` | ModulationCache sizing / modality_row → `test_h3_adaln_host` |
| `minimax_h3_mlx/dit.py` timestep / RoPE | `test_h3_dit_host` / `fixtures/h3_mlx_timestep.json` |
| `condition_proj` 5120→5376 | `h3_dit_condition_proj` |
| `packing.py` patchify/unpatchify | `h3_dit_patchify_video` / round-trip in `test_h3_dit_host` |
| `packing.py` spatial/temporal RoPE grids | `h3_rope_spatial_axis` / `h3_rope_temporal_*` |
| audio pack/unpack | `h3_dit_pack_audio` / `h3_dit_unpack_audio` |
| AdaLN linear split | `h3_adaln_split_block` / `h3_adaln_split_final` |
| AdaLN modulate / gated residual | `h3_dit_modulate` / `h3_dit_gated_residual` |
| ImageNet PIXEL_MEAN/STD | `h3_dit_pixel_{normalize,denormalize}` |
| SiLU / silu_mul / RMSNorm | `h3_dit_silu*` / `h3_dit_rmsnorm` |

## Non-goals

- Shipping MLX generate behind `/v1/videos` (M3 Ultra–class UMA; M4 Max is compute-bound for full H3).
- Vendoring the Python package into the Go binary.
- Competing with production `:11434` / `:8081`.

## Last checked

2026-08-13 — `--info` probe-ok without DiT/TE; host AudioVAE decode + WAV CLI;
host AudioVAE encode; DiT linear/SwiGLU/interleaved QKV; video-VAE geometry.
DiT/TE still too large for free disk.
