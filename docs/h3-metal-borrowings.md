# MiniMax-H3 Metal borrowings (zuiho-kai/h3-metal)

**Sibling:** `../h3-metal` — [zuiho-kai/h3-metal](https://github.com/zuiho-kai/h3-metal), fork of [antirez/h3.c](https://github.com/antirez/h3.c).  
**Tip noted:** `4ba47a3` (2026-10-28).  
**Product paths stay put:** Darwin pixels are video-c ConvRot (`--family h3`). CUDA pixels are Wan2GP (`runner: h3-wan2gp`). This checkout is the Mac Metal speed lab.

## Brought

| Piece | Where |
|-------|--------|
| FastH3 4-step sigma grid | `h3_fasth3_schedule_build` in `x/video-c/family_h3/h3_host.c` |

Video sigmas: `0.99992, 0.97283, 0.92308, 0.8, 0`. Audio: `0.99967, 0.89952, 0.75, 0.5, 0`. The function is not on the `--generate` path. The 50-layer ConvRot gate stays `latent_rms=1.18314 a_rms=0.504881`.

## Left in the sibling

| Piece | Why it stays out |
|-------|------------------|
| VSA block-sparse attention (`h3_gpu_vsa_sparse_bf16`) | Metal 4 TensorOps, `head_dim` 128, M5. Top 12 of 120 video tiles at sparsity 0.9. |
| `H3_VAE_BF16=1` decoder linears | Their warm-time win on the official BF16 VAE. video-c decodes the streamed f32 ViT. |
| FastH3 weight remap | `../h3-metal/tools/fasth3_keymap.md`. Needs `FastH3-4step/transformer` next to the HF snapshot. |
| int8 TensorOps / row-wise FC2 | Already in antirez `h3.c`; the fork tunes them. video-c dequants ConvRot to f32. |

Audio VAE, Qwen text encoder, and the tokenizer match antirez `8974cc0` byte for byte.

## Lab build

Weights are operator-supplied. This host’s `~/.zerollama/models/MiniMax-H3` has VAEs and the tokenizer, not the BF16 DiT. Do not pull the ~62 GiB DiT/TE shards onto a full disk.

```bash
# This Mac's PATH clang is Lean's. The Xcode clang is the one that sees the SDK.
make -C ../h3-metal -j8 CC="$(xcrun -f clang)" h3
../h3-metal/h3 --help
# when MiniMax-H3 BF16 shards are on disk:
H3_VAE_BF16=1 ../h3-metal/h3 -d ./MiniMax-H3 --info
```

`make -C ../h3-metal CC="$(xcrun -f clang)" test` runs the weightless host, audio-GPU, and VSA tests and skips the rest when fixtures are absent.

## Last checked

2026-10-28 — sibling cloned at `4ba47a3`. Schedule rematch is `tests/test_h3_host`.
