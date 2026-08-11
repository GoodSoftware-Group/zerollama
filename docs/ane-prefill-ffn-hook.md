# ANE prefill FFN ggml intercept (lab policy)

**Status:** policy + session parity + Metal shadow + host force + pack + sync-and-resume + **shexp/dense name filter**.

**Audience:** lab on non-production ports. Never enable on **:11434** / **:8081**.

Related: [ane-hybrid-path.md](./ane-hybrid-path.md), [ane-ggml-iosurface-hook.md](./ane-ggml-iosurface-hook.md), `ml/backend/ggml/ggml/src/ggml-metal/ane_ffn_policy.h`.

---

## Why a separate namespace

| Namespace | Role |
|-----------|------|
| `ZEROLLAMA_ANE_DRAFT_*` | Speculative draft handoff after `llama_decode` |
| `ZEROLLAMA_ANE_FFN_*` | Future prefill/shexp **`MUL_MAT`** intercept |

Do not reuse draft env for FFN.

---

## Target op

| Op | First intercept? | Notes |
|----|------------------|-------|
| `GGML_OP_MUL_MAT` | **Yes** | Dense / shared-expert (`*_shexp`) SwiGLU slices |
| `GGML_OP_MUL_MAT_ID` | **No** | Routed MoE — needs expert ids + streamed weights |

Validated sessions (smoke only today):

- `ane_prefill_session` — single fp16-blob matmul or fused SwiGLU (`--swiglu`)
- Geometry that wins vs MPS: **rectangular** (e.g. 2048×512), not full 2048²

**Expert SwiGLU lab (Jul 2026, M4 Max)** — `2048→512→2048` fused 1×1-conv, CPU golden:

| SEQ | eval_ms | golden_cosine | notes |
|----:|--------:|--------------:|-------|
| 64 (proxy 64→32) | ~0.07 | ≥0.99999 | compile OK |
| 128 | ~0.13 | ≥0.999999 | qwen3.6-mtp expert-up width |
| 512 | ~0.35 | ≥0.999999 | full expert FFN in one ANE eval |

Single expert-up matmul @ SEQ=512: ~0.18 ms eval, cosine ≥0.999999. Map ~0.04 ms. Dense 2048² still loses to MPS — do not chase.

---

## Env (fail-closed)

| Var | Default | Meaning |
|-----|---------|---------|
| `ZEROLLAMA_ANE_FFN` | off | Master switch |
| `ZEROLLAMA_ANE_FFN_MODE` | `shadow` when on | `shadow` = log/count only; `force` = skip Metal when host replace succeeds |
| `ZEROLLAMA_ANE_FFN_FORCE_ENABLE` | off | Extra latch for force (required) |
| `ZEROLLAMA_ANE_FFN_FORCE_HOST` | off | Skip GPU sync (assume CPU-visible acts already) |
| `ZEROLLAMA_ANE_FFN_REPLACE_DYLIB` | unset | Path to `libane_ffn_force.dylib` (dlopen register) |
| `ZEROLLAMA_ANE_FFN_NAME` | unset (any) | Weight-name filter: `shexp`, `ffn`/`dense`, `any`, or comma substrings |
| `ZEROLLAMA_ANE_FFN_SWIGLU` | off | Fuse up+gate+GLU+down into one ANE SwiGLU eval (force path) |
| `ZEROLLAMA_ANE_FFN_INT8` | off | Force SwiGLU: int8 weight blobs |
| `ZEROLLAMA_ANE_FFN_W8A8` | off | Force SwiGLU: W8A8 on hid (implies INT8); auto-calibrates `hid_scale` |
| `ZEROLLAMA_ANE_FFN_W8A8_X` | off | Dual W8A8 on x+hid (implies W8A8); separate `x_scale` |
| `ZEROLLAMA_ANE_FFN_INT8_IN` | off | Host int8 act write (`write_acts_int8`); implies W8A8_X |
| `ZEROLLAMA_ANE_FFN_IC` / `_OC` | 0 (any) | Geometry filter |
| `ZEROLLAMA_ANE_FFN_SEQ_MAX` | 0 (any) | Reject when `seq > max` |
| `ZEROLLAMA_ANE_FFN_LAB_PORT` | unset | If set, serve port must match; unknown port → refuse |
| `ZEROLLAMA_ANE_FFN_TELEMETRY` | off | Extra logging |
| `ZEROLLAMA_ANE_FFN_LOG` | unset | Telemetry file path (default `/tmp/ane-ffn-force.log` when TELEMETRY=1) |
| `ZEROLLAMA_ANE_FFN_WCACHE_SLOTS` | 32 | LRU dequant weight slots (1–64). Eliza 24 layers ≈ 150 MB/slot; set ≥ layers to avoid decode thrash |
| `ZEROLLAMA_ANE_FFN_SCACHE_SLOTS` | 32 | LRU ANE SwiGLU **session** slots in force dylib (1–64). Decode reuses larger prefill pad |

**Name presets** (match `src0->name`, e.g. `blk.0.ffn_up_shexp.weight`):

| Value | Matches |
|-------|---------|
| unset / `any` | all names |
| `shexp` | `ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp` |
| `ffn` / `dense` | `ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight` (not shexp/exps) |
| `ffn_up_shexp,ffn_down_shexp` | custom comma-separated substrings |

**Hard refuse:** ports **11434** and **8081** in both shadow and force.

```bash
# lab example (does not wire Metal yet — policy smoke only)
export ZEROLLAMA_ANE_FFN=1
export ZEROLLAMA_ANE_FFN_MODE=shadow
export ZEROLLAMA_ANE_FFN_IC=2048
export ZEROLLAMA_ANE_FFN_OC=512
export ZEROLLAMA_ANE_FFN_SEQ_MAX=512
export ZEROLLAMA_ANE_FFN_LAB_PORT=11435
./build/ane-probe-darwin/bin/ane-prefill-ffn-policy-smoke
```

---

## Integration plan

1. **Done:** CPU golden on blob matmul + fused SwiGLU; policy unit smoke; Go `envconfig.ANEFFN*`.
2. **Done:** `ane_ffn_shadow_note_mul_mat` in `ggml_metal_op_mul_mat` — counts + rate-limited stderr; Metal always still runs.
3. **Done:** `ane_ffn_force_try_mul_mat` (dims-only Metal hook) — fail-closed / deferred without host buffers.
4. **Done (lab):** `ane_ffn_force_try_mul_mat_host` + `ane_ffn_force_replace` — compile-once `ane_prefill_session`, pack acts, eval, write dst, return **true**. Proven by `ane-prefill-ffn-force-smoke`. Not linked into ggml-metal by default (no `ane_bridge` dep on Metal package).
5. **Done (lab):** ggml↔channel pack (`ane_ffn_force_pack`) + `ane_ffn_force_try_mul_mat_tensors` in Metal `mul_mat`. Optional `ZEROLLAMA_ANE_FFN_FORCE_HOST=1` skips sync; otherwise **`ggml_metal_encoder_sync_and_resume`** end+commit+wait then resumes a fresh cmd_buf. Replace via `ZEROLLAMA_ANE_FFN_REPLACE_DYLIB`.
6. **Done:** shexp/dense **name filter** (`ZEROLLAMA_ANE_FFN_NAME=shexp|ffn|…`) on weight `src0->name`.
7. **Done (lab):** fused SwiGLU intercept — at `ffn_up` **or** `ffn_gate` look ahead through `GLU(SWIGLU)→MUL_MAT(down)` (lookahead ≤48). Accepts **up→gate** / **gate→up**; MoE **holey** skip-scan. Opt-in `ZEROLLAMA_ANE_FFN_SWIGLU=1`. Topology match allows Q4; force-replace dequants via `to_float` then packs F32 W. With SWIGLU on, **no partial** single-matmul replace of up/gate/down.
8. **Done (lab):** fuse topology unit smoke (up-first + gate-first + holey) + shadow `swiglu_fuse#N` telemetry when clean chain matches.
9. **Done (lab):** optional post-mm **scale MULs** (`up_s`/`gate_s`/`down_s`) — fuse lookahead ≤7, fold scales into W, write `fuse->dst`. Lab script: `./scripts/ane/ane_ffn_lab_smoke.sh`.
10. **Done (lab):** hot-path cuts — **weight cache** by ggml data ptr (dylib session reuse), **fp16 acts pack** (`ane_ffn_pack_acts_ggml_to_channel_f16` + `replace_swiglu_fp16`).
11. **Done (lab):** restored `llama_vocab::get_suppress_tokens` (gemma4 logits bias) so Mac link succeeds; binary `./zerollama-ane-ffn-lab` embeds ANE SwiGLU hooks.
12. **Done (lab):** force replace selects best expert kernel via `ZEROLLAMA_ANE_FFN_{INT8,W8A8,W8A8_X,INT8_IN}` → `create_swiglu_int8_w8a8` + auto tile + abs>0 gate. Proven by `ane-prefill-ffn-swiglu-force-smoke --int8 --int8-in`.
13. **Done (lab):** shadow e2e on `:11435` — tag `ane-ffn-lab-eliza` (`spec_type=off` → ollama-engine Metal); dense FFN `swiglu_fuse#N` with `ic=2048 hidden=6144 seq=66` (prefill) / `seq=1` (decode). Production `:11434` untouched.
14. **Done (lab):** prepack int8 replace + staging reuse; force `ane_only` ≈ slice (~0.21 ms). Leftover ~0.6 ms is host fp16→f32 out — keep fp16/mapped Y (F0741).
15. **Done (lab):** force weight **dequant** via `ggml_get_type_traits()->to_float` (Q4_K_M etc.) into the SwiGLU/single-matmul weight cache; acts still F16/F32. Enables force on `ane-ffn-lab-eliza` without an F16 GGUF.
16. **Done (lab):** ggml `pack_acts_…_i8` + `try_swiglu_int8_fp16` in tensors (`INT8_IN`); blocked transpose; ggml i8 ~4.2 ms vs f16 ~5.3 ms — host layout tax ≫ ANE (F0742).
17. **Done (lab):** ANE SwiGLU **seq pad** — multiples of **64** (min 64); lengths ≡32 (mod 64) reject at 2048→512 int8-in. Logical 66/96→128. Metal pack uses `session_seq` (F0744).
18. **Done (lab):** Metal layout pack/unpack on ANE IOSurfaces — ggml-shaped wall **~0.66 ms** (~6.7× vs host); tensors prefer Metal (F0743).
19. **Done (lab):** force e2e on `:11435` — `swiglu_fp16_replaced#` at seq 66/1 after seq-pad dylib; initially ~50% layers Metal.
20. **Done (lab):** bail telemetry + fix — Q4_K_M mixed `ffn_down=Q6_K` was falsely rejected as `weight_type_mismatch`; per-tensor dequant → **100%** `swiglu_fp16_replaced` (0 bail / 0 fuse).
21. **Done (lab):** multi-slot LRU weight cache (`ZEROLLAMA_ANE_FFN_WCACHE_SLOTS`, default 32) + `wcache_hit#` / `wcache_miss#` telem — stops re-dequant thrash across layers on decode.
22. **Done (lab):** pad-safe `write_int8`/`reeval` + Metal layout @ seq 66/96/512 cosine ≥0.999999; dylib installed (F0744).
23. **Done (lab):** multi-slot ANE **session** LRU in `libane_ffn_force` (`ZEROLLAMA_ANE_FFN_SCACHE_SLOTS`, default 32) + `scache_hit#`/`miss#`. Decode reuses prefill when `sess.seq ≥ pad(decode)`.
24. **Done (lab):** serve e2e `:11435`/`ane-ffn-lab-eliza` — `scache … seq=128` (pad 66→128), `swiglu_fp16_replaced` ×23 @seq=66 + decode @seq=1; 0 bail (F0744).
25. **Done (lab):** MoE shexp **holey fuse** — skip-scan to GLU/down past router/`MUL_MAT_ID`; `n_encode_skip` + clear COMPUTE on glu/down. Unit smoke `n_encode_skip_holey=2`.
26. **Done (lab):** `metal_layout_replaced#` e2e on `:11435`/`qwen3.6-mtp` (`INT8_IN`+`NAME=shexp`+`OC=512`) — pad 18→64; ×38 @seq=18 + ×161 @seq=1; 0 bail / 0 fuse_miss (F0745). Not `MUL_MAT_ID`.
27. **Done (lab):** dense eliza A/B (`ane-ffn-lab-eliza`, 67→16 tok, temp 0) — **Metal wins**. Warm force eval ~820–1100 ms vs Metal ~170–180 ms (~5×); warm force total ~1.2–1.6 s vs Metal ~0.7 s. Force still correct (1609 replaced, 0 bail) but not a speed win at 2048×6144.
28. **Done (lab):** profile (`ZEROLLAMA_ANE_FFN_PROFILE=1`) — warm per-replace ≈ sync **1.2 ms** + pack **0.06** + ANE eval **0.9** + unpack **0.1** ≈ **2.3 ms**; ×24 layers ≈ **55 ms/tok** FFN-only vs Metal **~11 ms/tok** full model. ANE eval alone already ~2× Metal’s whole forward; sync_and_resume every layer doubles it again. Cold dylib dominated by ANE session compile (~100 ms+/layer once).
29. **Done (lab):** shexp A/B (`ane-ffn-lab-shexp` = `qwen3.6-mtp` blob + `spec_type=off`, ollama-engine; INT8_IN+NAME=shexp+IC=2048+OC=512). Same prompt, ~19→14 tok, temp 0 (discard cold). **Metal still wins, but close:** warm force eval ~**290–350 ms** vs Metal ~**254–258 ms** (~1.15–1.4×); warm total ~0.7 s both. Correct text in `thinking`. Decode hits `metal_layout_replaced` (sess_seq=64). Also many `bail reason=policy` (Metal fallback) — not a latency win. Micro: ane_only ~0.11 ms vs metal-layout wall ~0.5 ms (host tax). Do not use raw `qwen3.6-mtp` (draft-mtp → llama-server abort on this lab).
30. **Done (lab):** holey **quality** — early write to `down` at gate ⇒ gibberish; **defer replace to down encode** + scache ggml weight ids; fp16/INT8_IN greedy exact match on short prompt (F0746). `HOLEY=0` disables replace.
31. **Done (lab):** shexp sync floor (F0747) — `bail reason=policy` fixed by deferred `want_try(fuse.ic,hidden)`; `FORCE_HOST=1` and early staging+memcpy **fail quality**; sync~1.2 ms/layer remains the latency blocker. Next speed lever: async ANE∥MoE or finer waits — not skipping sync.
32. **Never:** auto-enable on production serve; never intercept `MUL_MAT_ID` in v1.

```bash
# Shadow on lab serve (Metal package builds; full zerollama may need llama pin sync):
export OLLAMA_HOST=127.0.0.1:11435
export ZEROLLAMA_ANE_FFN=1
export ZEROLLAMA_ANE_FFN_MODE=shadow
export ZEROLLAMA_ANE_FFN_NAME=shexp
export ZEROLLAMA_ANE_FFN_IC=2048
export ZEROLLAMA_ANE_FFN_OC=512
export ZEROLLAMA_ANE_FFN_SEQ_MAX=512
export ZEROLLAMA_ANE_FFN_LAB_PORT=11435
export ZEROLLAMA_ANE_FFN_TELEMETRY=1
# OLLAMA_HOST=127.0.0.1:11435 ./zerollama serve   # lab only — never :11434
# stderr: ane_ffn_shadow: match#N … name=blk.*.ffn_*_shexp.weight
```

Force path (lab, after building probes) — sync is default. Works on Q4_K denser FFN via host dequant cache:

```bash
export ZEROLLAMA_ANE_FFN=1
export ZEROLLAMA_ANE_FFN_MODE=force
export ZEROLLAMA_ANE_FFN_FORCE_ENABLE=1
export ZEROLLAMA_ANE_FFN_SWIGLU=1
export ZEROLLAMA_ANE_FFN_NAME=ffn
# export ZEROLLAMA_ANE_FFN_FORCE_HOST=1   # optional: skip GPU sync (CPU-visible already)
export ZEROLLAMA_ANE_FFN_REPLACE_DYLIB=$PWD/build/ane-probe-darwin/bin/libane_ffn_force.dylib
export ZEROLLAMA_ANE_FFN_LAB_PORT=11435
export ZEROLLAMA_ANE_FFN_TELEMETRY=1
export OLLAMA_HOST=127.0.0.1:11435
# Run in your own terminal (agent background shells exit and kill serve):
# ./zerollama-ane-ffn-lab serve
# curl …/api/generate -d '{"model":"ane-ffn-lab-eliza",…}'
# stderr: ane_ffn_force: swiglu_fp16_replaced#N …
```

Go package test (cgo tests unsupported in this package) — use:

```bash
./build/ane-probe-darwin/bin/ane-prefill-ffn-policy-smoke
./build/ane-probe-darwin/bin/ane-prefill-ffn-force-smoke
./build/ane-probe-darwin/bin/ane-prefill-ffn-swiglu-force-smoke
./build/ane-probe-darwin/bin/ane-prefill-ffn-fuse-unit-smoke
go build -o /dev/null ./ml/backend/ggml/ggml/src/ggml-metal/
```

---

## Smokes

```bash
./scripts/ane/ane_ffn_lab_smoke.sh
./scripts/ane/ane_ffn_lab_smoke.sh --print-env   # lab :11435 env block
./scripts/ane/ane_probe_build.sh
./build/ane-probe-darwin/bin/ane-prefill-ffn-policy-smoke
./build/ane-probe-darwin/bin/ane-prefill-ffn-force-smoke --ic 512 --oc 256 --seq 64
./build/ane-probe-darwin/bin/ane-prefill-ffn-swiglu-force-smoke --ic 256 --hidden 128 --seq 64
./build/ane-probe-darwin/bin/ane-prefill-ffn-swiglu-force-smoke --int8 --int8-in --ic 2048 --hidden 512 --seq 512
./build/ane-probe-darwin/bin/ane-prefill-ffn-fuse-unit-smoke
./build/ane-probe-darwin/bin/ane-prefill-ffn-slice-smoke --ic 2048 --oc 512 --seq 512 --quick
./build/ane-probe-darwin/bin/ane-prefill-ffn-slice-smoke --swiglu --ic 2048 --oc 512 --seq 512 --quick
# or via Go:
./zerollama ane-prefill-ffn-slice-smoke --expert-up --ic 2048 --seq 512 --swiglu --quick
```
