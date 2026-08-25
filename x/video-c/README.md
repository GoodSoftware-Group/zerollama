# video-c — Pure-C multi-family video (Wan + H3)

Evolved from **wan-c**. Dispatches DiT/T5/VAE work to the UMA broker via `GRAPH`
recipes; links `libuma_client` from [bmtl uma_toolkit](https://github.com/elizaOS/bmtl).

Python Wan (`scripts/video/wan_video_generate.py`) remains the default product path.
Operators may set `ZEROLLAMA_VIDEO_CLI` so Wan jobs use this binary — clients never
choose the runner. See [docs/video-c.md](../../docs/video-c.md).

Compat: `x/wan-c` → this tree; `wan-cli` → `video-cli`.

## Build

```bash
make -C x/video-c
make -C x/video-c test              # weightless
make -C x/video-c test-h3-weights   # AudioVAE + video VAE if MiniMax pack present
```

Requires macOS SDK, bmtl `uma_toolkit` at one of:

- `../../../bmtl/targets/m4-max/hardware/uma_toolkit`
- `../../../bmtl/hardware_lab/lanes/m4/uma_toolkit`

## Run

```bash
UMA_WAN_LOCAL=1 ./x/video-c/video-cli --family wan \
  --ckpt-dir ~/.zerollama/third_party/wan/Wan2.1-T2V-1.3B \
  --prompt "a red apple" \
  --width 64 --height 64 --frames 5 --steps 2 \
  --out /tmp/wan_local.mp4

./x/video-c/video-cli --family h3 --info -d ~/.zerollama/models/MiniMax-H3
./x/video-c/video-cli --family h3 --decode-audio \
  -d ~/.zerollama/models/MiniMax-H3 -o /tmp/h3.wav
./x/video-c/video-cli --family h3 --encode-audio \
  -d ~/.zerollama/models/MiniMax-H3
./x/video-c/video-cli --family h3 --clipproj
./x/video-c/video-cli --family h3 --encode-video \
  -d ~/.zerollama/models/MiniMax-H3 -o /tmp/h3_roundtrip.mp4
./x/video-c/video-cli --family h3 --decode-video \
  -d ~/.zerollama/models/MiniMax-H3 -o /tmp/h3_frame.ppm
./x/video-c/video-cli --family h3 --encode-video --in /tmp/h3_frame.ppm \
  -d ~/.zerollama/models/MiniMax-H3 -o /tmp/h3_from_ppm.mp4
./x/video-c/video-cli --family h3 --tokenize --prompt "A red fox walking through snow"
./x/video-c/video-cli --family h3 --present --prompt "hi" --pictures 1 --merge-h 2 --merge-w 2
./x/video-c/video-cli --family h3 --embed --prompt "A red fox walking through snow"
./x/video-c/video-cli --family h3 --generate --prompt "A red fox" \
  -d ~/.zerollama/models/MiniMax-H3 -o /tmp/h3_tiny.ppm
# Resident serve daemon (warm requests reuse the dequantized weight cache):
WAN_PROFILE=1 H3_MLOCK=1 ./x/video-c/video-cli --family h3 --generate \
  --serve-sock /tmp/h3_serve.sock -d ~/.zerollama/models/MiniMax-H3
```

H3 tiny T2VA generate is shipped (5×32²). `/v1/videos` runs the **full 50 DiT
layers** (the pruned int8 export has 50 blocks; a truncated default previously
broke audio — see AGENTS.md). `--width 768` packs 48×48 latent (nv=1152).
`--decode-video -o` and `--encode-video -o` write **mp4** (ffmpeg, including
`~/.homebrew/bin`) or a playable **AVI** when ffmpeg is missing.

`/v1/videos` tags: `minimax-h3-tiny:lab` (32²) and `minimax-h3-768:lab` (768², 50 layers) via `./scripts/video/register_h3_models.sh`.

Rematch reference: sibling `../h3.c` (antirez) and `../minimax-h3-mlx`.

## H3 serve daemon — resident weight store

The H3 host pipeline is cold-start-bound: loading + dequantizing the ~22 GB
`pruned_int8_convrot` DiT (f32 expands ~4×) takes minutes and dominated latency.
The answer is a **resident daemon** that keeps every dequantized weight cached in
RAM across requests, so warm requests pay only compute.

```bash
WAN_PROFILE=1 H3_MLOCK=1 ./video-cli --family h3 --generate \
  --serve-sock /tmp/h3_serve.sock -d ~/.zerollama/models/MiniMax-H3
```

**Why `H3_MLOCK=1`:** the dequantized caches total ~57.7 GiB on this Mac
(DiT 38 + TE 10.5 + video VAE 9.0 + audio VAE 0.24). On a shared 128 GB machine
these pages are otherwise **swapped out between requests** — the "slow second
request" was swap-in of the DiT cache, not re-dequantization. `mlock` pins them
(`ulimit -l` is unlimited here). Without it the store still persists (no reload)
but pays page-in on a cold/warm chassis.

Request protocol (tab-separated, 8 fields backward-compatible; reuse/adaln optional):

```
out_mp4 \t prompt \t frames \t width \t height \t seed \t steps \t layers \t reuse \t adaln_t_sigma
```

Reply is `ok\n` or `err: <msg>\n`. `--reuse` 1=evaluate every step (best quality),
2/3 extrapolate velocity. `adaln_t_sigma` 0=`t=1-σ` (default), 1=`t=σ`, -1=env.

**Measured (quiet machine):** warm served request ~13 s vs ~224 s cold
single-process — **~17×**, signature bit-identical (`latent_rms=1.18314
a_rms=0.504881` for seed=1, "A red fox walking through snow"). The per-request
compute is: DiT denoise ~3.1 s (BLAS f32), audio VAE ~1.8 s, text_cond ~1.4 s,
video VAE ~0.9 s, media encode ~0.14 s — video+audio VAE decode overlap via
`H3_PARALLEL_VAE=1` (default).

**Why so fast:** every matmul is Accelerate `cblas_sgemm`; independent loops
(dequant, conv output positions, snake activation, attention heads) are
`dispatch_apply`-parallelized bit-exactly; weights are handed out as immutable
`const` pointers (`h3_st_store_get_f32`) — never copied per request. See
`AGENTS.md` for the simple/fast/correct rules.

## H3 (Darwin / M4 Max)

Weightless rematch lives in `family_h3/` (`h3_host`, AdaLN, DiT ops, ClipProj).
`--info` succeeds on a VAE-only snapshot (no DiT/TE shards). Decode needs
`FL2VA/audio_vae`. Text encode uses the sibling **BMTL C tokenizer** (Gigatoken
lessons already in `uma_toolkit`), not a from-scratch BPE. Export once:

```bash
python3 ../bmtl/tools/export_bmtl_tokenizer.py \
  ~/.zerollama/models/MiniMax-H3/FL2VA/tokenizer/tokenizer.json \
  --scheme qwen2 -o ~/.zerollama/third_party/h3/minimax_h3.bmtl_tok
```

## Ownership

| Track | Scope |
|-------|--------|
| Darwin / UMA | This Mac track |
| CUDA (`make cuda-lab`) | Parallel owner — [docs/cuda-uma-toolkit.md](../../docs/cuda-uma-toolkit.md) |

Further Wan detail: [wan-c-speed-gap.md](../../docs/wan-c-speed-gap.md).
