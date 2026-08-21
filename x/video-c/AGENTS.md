# AGENTS.md — video-c (H3 host port)

Guidelines for editing this tree: keep it **simple**, **fast**, **correct**. Every
change to the H3 path should be judged against all three.

## Correctness gate (non-negotiable)

The H3 generation output is deterministic. The regression gate is a tiny T2VA run:

- seed=1, prompt `A red fox walking through snow`, defaults (frames/width/height/
  steps/layers = 0) must print:

  ```
  latent_rms=1.18314 a_rms=0.504881
  ```

If that string moves, the change is a regression, not an optimization. Run it via
the serve socket or `--generate` after any numeric change. (The H3 family files
are part of the wan-c→video-c reorg and have been in flux; re-verify this gate
whenever the denoise/audio-latent path changes.)

**Layers gate:** `--generate` must default to the full `H3_DIT_NUM_LAYERS`
(50). Running fewer blocks cuts the residual stack; the final AdaLN/RMSNorm sees
a wrong hidden state and the audio velocity explodes ~70× → ~93%-clipped waveform. Do not reintroduce a layer cap (the old 24-layer default with its
`L47–L48 v_rms cliff` reasoning was the audio bug).

## Performance invariants (why the hot path looks the way it does)

- **Workspace, not malloc-per-block** (`h3_dit_ws_create`, `h3_dit_block.c`):
  one scratch arena per forward pass feeds all 50 blocks. The block path moves
  ~800 MB of qkv/fused/hid/q/k/v/attn per block-step; Darwin mmaps those sizes,
  so fresh mallocs page-fault every time. Buffers are fully overwritten each
  use (sgemm beta=0), so reuse is bit-exact — the gate string above is
  unchanged.
- **RoPE tables are built once per forward** (`h3_dit_rope_tables`) and shared
  by every block/step: position_ids never change, and per-block `sinf/cosf`
  over seq×192 was pure waste. Rotary applies in place
  (`h3_dit_apply_rotary_heads_inplace`) — no temp gather/scatter.
- **Diagnostics are gated by `H3_DIT_DEBUG`.** `attn_probe_video`, the
  per-layer stats line, and `log_vid_div` used to run unconditionally (~14G
  double ops + 400 stderr lines per generate). Do not ungated them.
- **SDPA is BLAS by default** (`h3_sdpa_blas`): per-head `cblas_sgemm` QK^T /
  P·V + vDSP softmax. Q/K/V stay packed — sgemm walks them via
  `lda=heads·hd`, no gather pass. Measured at 768² 1-step: **1119 → 744
  ms/layer (1.50×)**; vs scalar it agrees to ~1e-8 (test) / gate moved only in
  the 5th digit (accumulation order), which is why the gate string above is
  the BLAS one. `H3_SDPA_BLAS=0` restores the bit-exact scalar path (tests pin
  it for the bit-identity check). The softmax rows are chunked across threads
  (`dispatch_apply`, ncore chunks): each row keeps its exact serial op
  sequence, so it is bit-identical — micro-bench at the 768² shape:
  serial 70 ms → 36 ms per SDPA call. `VECLIB_MAXIMUM_THREADS` tuning was
  tried and is a wash (default threading already optimal).
- **Block outputs ping-pong** between `packed`/`tmp` in `h3_dit_forward` —
  no per-block memcpy back; one conditional copy after the loop lands the
  final hidden in `packed`. Bit-exact.
- **AdaLN scratch lives in the workspace** (`proj`/`six`/`adaln_b`, sized to
  the nuniq≤8 maxima) — no per-block mallocs left in the block path.
- **Weight-cache lookups are hashed** (`h3_wc_htab_*` FNV-1a open addressing
  in `h3_st_store.c`) — replaces the O(n) strcmp scan over ~600 tensors.
- **Measured out (do not revisit lightly):** making QK-norm/RoPE read strided
  off the fused qkv buffer to skip `qkv_split`'s three memcpys saves ~100 MB
  of traffic per block-step ≈ 0.2% of a 744 ms layer — not worth the
  invasiveness.
- **Big weights are read by pointer** (`h3_st_store_get_f32`) — e.g. the
  49.5 MB `adaln_proj.linear.weight` used to be memcpy'd out of the weight
  cache every block-step (~19.8 GB/generate). Keep small loads on `load_n`;
  anything ≥ tens of MB must use the pointer path.

## Build / test / serve

```bash
make video-cli        # build
make test             # unit + fixture tests (must be rc=0)
```

Serve daemon (primary interface — the resident store is what makes warm requests
fast):

```bash
WAN_PROFILE=1 H3_MLOCK=1 ./video-cli --family h3 --generate \
  --serve-sock /tmp/h3_serve.sock -d ~/.zerollama/models/MiniMax-H3
```

Request protocol (tab-separated, 8 fields backward-compatible; reuse/adaln optional):

```
out_mp4 \t prompt \t frames \t width \t height \t seed \t steps \t layers \t reuse \t adaln_t_sigma
```

## Simple

- **One code path per job.** No parallel implementations of the same flow. The
  CLI `--generate` and the serve daemon share `h3_run_generate` — keep it that way;
  a forked copy will drift (it already did, silently keeping an old VAE path).
- **Debug is opt-in and invisible by default.** All diagnostics are env-gated
  (see table). Never add bare `fprintf` to a hot path. When adding a new knob,
  read the env once, keep it off by default.
- **Defaults are the happy path.** The binary works with no env set. Tuning knobs
  layer on top.

## Fast

- **Cache once, never copy.** The weight store is resident in the daemon: tensors
  dequantized once, optionally `mlock`ed, and served as `const` pointers. Use
  `h3_st_store_get_f32` for weight access in request paths; never `malloc` +
  `load_f32` (memcpy) per request for big weights.
- **BLAS for every matmul.** All linear/conv matmuls go through Accelerate
  `cblas_sgemm` (`h3_dit_linear` → `h3_dit_condition_proj`). New matmul code must
  use BLAS, not hand loops.
- **Threads for independent work.** `dispatch_apply` over output positions /
  heads (dequant, conv, snake activation, attention, and the video+audio VAE
  decodes overlap). Only parallelize when each output is computed with the same
  loop order as serial — see Correctness.
- **Measure buckets, not wall time.** Use `h3_prof_add_ms`/`WAN_PROFILE` for
  per-stage timing, and only on a quiet machine. This lab box is heavily shared
  (load >20 on 16 cores) — absolute wall times there are noise.

## Correct

- **Bit-exact parallelism.** A loop may be parallelized only over *independent*
  outputs, each computed with *identical* arithmetic order to the serial version.
  Then results are bit-identical regardless of scheduling. If outputs share
  reduction state (e.g. one `scores` buffer), give each task its own.
- **Concurrency is earned.** The store registry and profiler are the only shared
  mutable globals and both are mutex-guarded — because the parallel VAE decodes
  needed that. Per-request store access stays single-threaded. Don't add locks
  speculatively; add them where concurrency is real.
- **Cached weights are immutable.** `get_f32` returns `const` pointers owned by
  the store; callers must not free or mutate them (`wfree` is a no-op for
  store-owned pointers). A weight must never be folded/rewritten in place.
- **Small-M gemm reality.** At lab seq (`seq≈24–33`) the DiT/VAE are BLAS- and
  memory-bound; the naive O(seq²) attention only matters at real scale (seq≥64)
  where it is parallelized over heads, threshold-gated so the lab path is
  byte-identical.

## Env knobs (dev aids)

| Var | Effect |
|-----|--------|
| `H3_TEXT_COND` | Load H3TE dump (`--text-cond`) instead of 4B+ClipProj. Optional uint8[nt] tags after floats. |
| `H3_MLOCK=1` | mlock all resident weight caches (57.7 GiB on this Mac) |
| `H3_PARALLEL_VAE=0` | disable overlapping video+audio VAE decode |
| `H3_STORE_DBG` / `H3_AVAE_DBG` / `H3_BLK_MS` / `H3_MLOCK_DBG` | debug prints (store, audio stages, per-layer DiT, per-store mlock) |
| `H3_DUMP_EMBED` / `H3_EMBED_ONLY` | dump packed `h` after patch+refiner; skip DiT blocks |
| `H3_VIDEO_LATENT` / `H3_AUDIO_LATENT` | load f32 CTHW / (2,C,T); `--decode-audio` can load `H3_AUDIO_LATENT` |
| `H3_DUMP_NOISE_DIR` / `H3_DUMP_VEL` / `H3_DUMP_AUDIO_LATENT` | noise bins / unpatched `vpred` / post-DiT audio `(2,C,T)` |
| `H3_DIT_LAYERS` / `VIDEO_H3_LAYERS` / `H3_ADALN_T_SIGMA` | layers override / AdaLN time index (0 = t=1-σ) |
| `H3_SAMPLER=res_multistep` | Comfy `sample_res_multistep` (η=0). Default Euler when `nv≤8` (tiny gate). Unset env + `nv>8` uses res_multistep. `H3_SAMPLER=euler` forces Euler. |
| `H3_AUDIO_CARRY` | `1`/`0` force Comfy AV wrap. Unset: on when `nv>8`. When on: `process_latent_in` ×(12/3), forward on uncarried x, Euler/res on carried x, `process_latent_out` ÷4. |
| `H3_FFN_CLIP` / `H3_FFN_CLIP_TEXT` / `H3_DIT_BF16_ACT` / `H3_DIT_ACT_INT8` / `H3_VEL_RMS` / `H3_LATENT_PREVIEW` | clip / bf16 residual+embed / ConvRot act fake-quant / scale velocity RMS / .pgm |
| `WAN_PROFILE=1` | cumulative stage profiler report |

## Toolkit relationship (bmtl uma_toolkit)

- `video-c` is the complete host-f32 H3 pipeline (resident serve daemon). The
  toolkit (`bmtl/hardware_lab/lanes/m4/uma_toolkit`) is the broker/GRAPH plane.
  Two separate worlds: video-c computes on host Accelerate; the toolkit composes
  on its daemon. When a piece is "ported" to the toolkit, keep it **bit-exact**
  with the host port (cross-check via `make h3-convrot-smoke`, rms=`0.0809541`).
- `src/uma_h3/` in the toolkit holds the ported reader (`safetensors_min` +
  `uma_convrot` + `uma_h3_weight`). If you change the reader math here, update
  the port too — they must stay in lockstep or dequantized weights drift.
- The real DiT/VAE GRAPH compose is blocked on a daemon `GEMM_F16` bug for
  `M<64` shapes (see toolkit F1078). Do not work around it with client-side
  toggles; the fix belongs in the toolkit core.
