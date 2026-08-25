# wan-c vs Python MPS — speed gap

Lab host (Mac UMA broker): **832×480 / 5 frames / 8 UniPC steps / cfg=5**.

| Path | Wall | Notes |
|------|------|-------|
| Python MPS (approx) | ~210 s | load~55 + DiT~45 + VAE; DiT ~5–6 s/step |
| wan-c pre RPC cuts | ~440 s | README sign-off ~7.3 min |
| wan-c Phase 1 (`WAN_PROFILE=1`) | **~408 s** | BANK_BINDS, sticky BUF, RoPE skip, FFN chunk=3120 |
| wan-c + ASCII CPU↔GPU fuse | ~426 s | **slower** — fewer submits (8691→4851) but longer waits (F0994 lesson) |
| wan-c + warm HEADT (client pad) | ~467 s | **slower VAE** (~115 s vs ~97 s); default **off** |

Still **~1.9×** Python wall (target ≤1.2×). Kept **same-device** RoPE Q+K fuse only.

## Profile (`WAN_PROFILE=1`, 832 s8 Phase 1) — non-overlapping wall view

Use `real` as truth; nested buckets double-count. Approximate exclusive shares:

| Stage | ~ms | Share of ~408 s |
|-------|-----|-----------------|
| DiT cond+uncond | ~308 s | **~76%** |
| VAE decode | ~97 s | **~24%** |
| T5 | ~1.5 s | <1% |

Inside DiT (nested under GRAPH waits):

| Bucket | ms | n |
|--------|----|---|
| `graph` submit+wait | ~324 s | 8691 |
| `dit_self` | ~152 s | 480 |
| `dit_ffn` | ~100 s | 480 |
| `dit_cross` | ~49 s | 480 |
| `bank_bind` / `dit_bind` | ~0.1 s | 520 / 480 |
| `buf_skip` | — | 22826 (sticky BUF_ALLOC) |

**Conclusion:** C-side BIND/PUT tax is now tiny. ASCII multi-op fuse that **mixes `@CPU!` and `@GPU!`** increases per-ticket latency at T=3120 (same finding as F0994 `WAN_DIT_ONE_GRAPH`). Remaining gap needs a real **`DIT_BLOCK` Metal CB** plus on-broker VAE feat_cache (no client pad shuttle).

## Landed in wan-c (Phase 0–1 + RoPE + warm experiment)

- `WAN_PROFILE=1` — [`wan_profile.c`](../x/wan-c/wan_profile.c)
- F0703 `BANK_BINDS` one IPC/block ([`dit_wan.c`](../x/wan-c/dit_wan.c))
- Sticky `BUF_ALLOC` when size matches (`WAN_BUF_STICKY=0` restores re-assert)
- Skip per-block text mirror on persist; RoPE freq PUT sticky by geometry
- `WAN_DIT_FFN_CHUNK` default threshold **4096** (T=3120 → one FFN_GELU)
- `wan_graph_rope3_qk` — one GRAPH for Q+K RoPE (same-device)
- Sticky dual text ctx `x_dit_tctx{0,1}` across UniPC (`tctx_hit`)
- **Warm HEADT (opt-in `WAN_VAE_WARM_HEADT=1`)** — F1012 broker feat_cache (no sil shuttle); rematch @160 MAD≈4e-5. Exclusive 832 s2 VAE **102 s host vs 106 s warm** — no win; keep opt-in. Legacy shuttle `WAN_VAE_WARM_HEADT_SHUTTLE=1` still worse.
- Bank key `decoder.conv1.weight` (36 keys) — ready for a future cold causal tip
- Toolkit: `UMA_ATTN_FLASH_TK_MIN` (default 8192). **Do not** set 3000 for Wan T=3120 — measured regress.

## Blow-away idea scorecard

[docs/wan-c-blowaway-ideas.md](./wan-c-blowaway-ideas.md) — 10 ideas; only sticky tctx landed as a small win; flash ungating and client warm HEADT **failed**; rest need Metal.

## Toolkit asks (open — needed for ≤1.2×)

File against `uma_toolkit` wishlist / CHANGELOG:

1. **`DIT_BLOCK` (or half-block) k_op** — LN+AdaLN+QKV+RoPE+ATTN+O+cross+FFN_GELU in one Metal CB / one GRAPH wait. Do **not** chase longer ASCII recipes (F0994 + 2026-08-08 rebench: rematch OK, slower at prod T).
   **Partial 2026-08-08:** gated `RESIDUAL_ADD` (`y+=(x+up?)*gate`) + FFN_GELU compose; O-proj gated resid. wan-c default on (`WAN_DIT_NO_GATED_RESID=1` rollback). 832 s2: GRAPH n 2092→1852.
   **Partial+ 2026-08-08:** Metal GPU `LAYERNORM_MUL` / `AFFINE_MUL_ADD`; FFN one sticky GRAPH LN+AdaLN+FFN+gated resid; self/cross LN `@GPU!`. 832 s2: GRAPH n **1732**; wall **~149–154 s** (≈flat vs gated). Rematch MAD≈0.001 @160.
   **Partial++ 2026-08-08:** fuse self O+bias+gated resid; fuse cross ATTN+O+bias+resid; bias AFFINE `@GPU!`. GRAPH n **1252** (bit-exact @160 vs prior). 832 s2 wall **noisy** — not a clear win; next is Metal ROPE3 / ATTN residency, not more ASCII length.
   **Partial+++ 2026-08-08:** Metal `ROPE3` (`uma_wan_rope3_f32` + GPU chain); wan-c `@GPU!`. Rematch MAD≈0.002 @160; 832 s2 wall **~148 s** (`dit_self`~25 s, `vae`~92 s, `graph` n=1252). Next: QKV→RMS→RoPE→ATTN→O same sticky CB.
   **Partial++++ 2026-08-08:** self-attn two sticky GRAPHs (`LN→QKV→RMS→RoPE` + `ATTN→O→gated`); rollback `WAN_DIT_NO_SELF_FUSE=1`. Rematch **bit-exact** @160 vs no-fuse; `graph` n **892**; 832 s2 wall **~152 s** (≈flat vs brick4; VAE noise).
   **Partial+++++ 2026-08-08:** cross-attn two sticky GRAPHs; rollback `WAN_DIT_NO_CROSS_FUSE=1`. Rematch bit-exact @160; `graph` n **652**; 832 s2 wall **~148 s**.
   **Partial++++++ 2026-08-08:** VAE same-C dual-HEADT one GRAPH (cold; fixed F1012 slots); rollback `WAN_VAE_NO_RESID_FUSE=1`. Rematch bit-exact @160; `graph` n **639**; 832 s2 wall **~152 s** (`vae`~96 s) — **flat**. ASCII ticket merges saturated.
   **Brick 9 2026-08-08:** profile split `dit_self_pre` / `dit_self_attn`. Cross-submit Metal CB HOLD / `DIT_SELF_PRE` composer rejected for wall (commit tax ≪ ATTN).
   **Brick 10 / F1014 2026-08-09:** TG-tiled MH QK+PV **default ON** (~2.2× @T=3120). 832 s2: `dit_self_attn` **18.1→10.8 s**; wall ≈146 vs ≈149 (VAE-bound); rgb MAD **0**. Rollback `UMA_ATTN_TG=0`.
   **Brick 11 2026-08-09:** `WAN_VAE_STAGE_PROF=1` tip map @832 s2 — `vae` **96.9 s**; `vae_conv_host` **44 s** + HEADT cblas **26 s** ≈ **73%**.
   **Brick 12 / F1017 2026-08-09:** BNNS depth-fold CONV3D **default ON**. 832 s2:
   `vae` **95.8→39.5 s**, `conv_host` **46.6→5.3 s**, wall **143→115 s**; rgb MAD≈0.
   Rollback `UMA_WAN_CONV_BNNS=0`.
   **Brick 13a 2026-08-09:** `WAN_DIT_HOST_FFN_CORE` rematch OK; `dit_ffn` **30→37 s**
   — **kill** for wall (IPC). Opt-in only.
   **Brick 13b / F1018 2026-08-09:** BNNS CONV2D + host resample **default ON**
   (wan-cli only; no daemon). 832 s2: `vae_resample` **7.6→1.2 s**, `vae` **40→33 s**;
   MAD≈0. Rollback `WAN_VAE_HOST_RESAMPLE=0`. **Next:** DiT without daemon bounce.
2. ~~**Flash / tiled full-T ATTN**~~ — **done F1014:** TG default ON; 832 s2
   `dit_self_attn` **18.1→10.8 s** (~1.7×); wall ≈flat (VAE ~92 s). Rollback
   `UMA_ATTN_TG=0`.
3. ~~**HEADT + feat_cache on broker**~~ — **F1012 opt-in only (Brick 8 miss):** sticky
   mid/out exists; default-on warm @832 s2 wall **~190 s** / `vae`~117 s vs
   `NO_WARM` **~177 s** / `vae`~109 s; rematch vs host-warm MAD≈9.6 @160. Keep
   `WAN_VAE_WARM_HEADT=1` opt-in. Legacy shuttle: `WAN_VAE_WARM_HEADT_SHUTTLE=1`.
4. ~~**VAE host CONV (Brick 12)**~~ — **done F1017:** BNNS default ON; 832 s2 wall
   **143→115 s**, `vae` **96→40 s**. Rollback `UMA_WAN_CONV_BNNS=0`.
5. ~~**VAE resample CONV2D (Brick 13b)**~~ — **done F1018:** host BNNS default ON;
   resample **7.6→1.2 s**. Rollback `WAN_VAE_HOST_RESAMPLE=0`.

Secondary (wan-c after toolkit): mid-attn CHW↔token without GET/PUT shuttle — **deprioritized** (Brick 11 `vae_ipc` ~0.1 s).

## How to rebench

```bash
WAN_PROFILE=1 /usr/bin/time -p ./x/wan-c/wan-cli \
  --ckpt-dir ~/.zerollama/third_party/wan/Wan2.1-T2V-1.3B \
  --uma-sock /tmp/uma_daemon.sock --seed 42 \
  --prompt "a red apple on a wooden table" \
  --width 832 --height 480 --frames 5 --steps 8 --cfg 5 \
  --out /tmp/wan_prof_832.mp4
```

Success: wall **≤ ~250 s** (~1.2× Python ~210 s) with `ok_nontrivial` unchanged.
