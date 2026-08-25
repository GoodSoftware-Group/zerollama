# Blow-away Python — idea scorecard (2026-08-09)

Target: wan-c **faster** than Python MPS (~210 s @ 832×480 s8 cfg5). Baseline wan-c Phase1 **~408 s**.

## Lab rule — do not disturb main `uma_daemon`

Broker-side wan experiments ship as **EXT / `uma_opworker` plugins** (`EXT_CALL`,
`EXT_REGISTER`, F0712 / F0779). Rebuild **wan-cli** + **opworker** only.

- **Do not** `make src/uma_daemon`, `make uma-daemon`, `--stop`, or `pkill uma_daemon`
  unless the user explicitly asks to bounce the main broker.
- Host kernels in `uma_wan_ops.c` linked into **wan-cli** are fine without a daemon bump.
- In-daemon `CONV3D@CPU!` / HEADT picks up shared-ops changes only on a **user-driven**
  daemon upgrade — prefer an EXT BNNS/CONV path if the broker must stay stock.

| # | Idea | Tried? | Result |
|---|------|--------|--------|
| 1 | `DIT_BLOCK` Metal CB | **blocked for wall** | ASCII saturated; see Brick 9 |
| 2 | Flash / tiled ATTN @ T=3120 | **yes** | rematch OK; **slower**. Gate **8192** |
| 3 | Sticky BANK once | **done** | not the wall |
| 4–6 | DiT GPU fuse bricks | **done** | n **652**; wall flat ~148 s |
| 7 | VAE dual-HEADT | **done** | n **639**; wall flat |
| 8 | F1012 warm default-on | **miss** | MAD≈9.6; wall **regress** |
| 9 | Profile ATTN wall | **done** | `dit_self_attn` dominates self |
| 10 | TG-tiled MH QK+PV (F1014) | **win** | ATTN ~2×; s2 wall VAE-bound |
| 11 | VAE tip stage map | **done** | cblas ≈73% of `vae` |
| 12 | BNNS depth-fold CONV3D (F1017) | **win** | `vae` 96→40 s |
| 13a | Host FFN_GELU CORE | **miss** | rematch OK; `dit_ffn` **regress** |
| 13b | BNNS CONV2D + host resample (F1018) | **win** | below |

## Brick 12 — F1017 BNNS depth-fold CONV3D (2026-08-09)

**Shipped in shared `uma_wan_ops.c` (default ON).** Rollback `UMA_WAN_CONV_BNNS=0`.

| Path @832 s2 | `vae` | `conv_host` | wall |
|--------------|------:|------------:|-----:|
| im2col | 95.8 s | 46.6 s | 143 s |
| BNNS | **39.5 s** | **5.3 s** | **115 s** |

## Brick 13a — host FFN CORE (2026-08-09) — **kill for wall**

`WAN_DIT_HOST_FFN_CORE=1`: broker LN/AdaLN/resid + host Accelerate FFN_GELU
(no bias). Rematch **bit-exact**. `dit_ffn` **30→37 s** (GET/PUT tax). Opt-in only.

## Brick 13b — F1018 BNNS CONV2D + host resample (2026-08-09)

**Shipped (wan-cli default ON; no daemon bounce):**
- BNNS in `uma_wan_conv2d_f32` (`UMA_WAN_CONV_BNNS`)
- Host nearest+CONV2D resample default ON; rollback `WAN_VAE_HOST_RESAMPLE=0`

| Path @832 s2 | `vae_resample` | `vae` | rgb MAD |
|--------------|---------------:|------:|--------:|
| broker F1001 | 7.6 s | 40.3 s | — |
| host BNNS | **1.2 s** | **32.8 s** | **≈0** |

160: resample 0.36→0.047 s. Microbench ~8–16× vs im2col.

**Daemon bounce (operator OK, 2026-08-09):** rebuilt `metallib` + `src/uma_daemon`,
`make uma-daemon` → pid **75695**. ATTN micro healthy:
`wan-graph-attn-dit-smoke` T=3120 **live_s=0.072** (F1014 class).

**Post-bounce exclusive 832 s2:** wall still **983 s** — bounce did **not** restore
Brick12 wall. VAE/FFN still fine; DiT GRAPH still bloated:

| bucket | post-bounce | note |
|--------|------------:|------|
| wall | 983 s | |
| `vae` / resample | **39.9 / 1.4 s** | F1018 OK |
| `dit_ffn` | **30.9 s** | OK |
| `dit_self_pre` | 420 s | LN→QKV fuse |
| `dit_self_attn` | 151 s | ~1.26 s/call vs micro 0.072 s |
| `dit_cross` / `graph` | 332 / 944 s | |

Raw ATTN is fine; **fused DiT persist graphs** are the wall. Next: profile
`dit_self_pre` / fused ATTN→O GRAPH vs standalone ATTN (not another bounce).

See also: [wan-c-speed-gap.md](./wan-c-speed-gap.md) · bmtl F1017 · F1018.
