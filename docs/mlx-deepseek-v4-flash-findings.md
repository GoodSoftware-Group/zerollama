# MLX DeepSeek-V4 Flash — findings & learnings

**Why this doc:** The first mlxrunner graph looked complete (names registered, tests green) and was **not a valid Flash forward**. Capture the traps so the next pass does not re-stub CSA or re-break hash MoE. Operator how-to: [mlx-deepseek-v4-flash.md](./mlx-deepseek-v4-flash.md).

**Date:** Aug 2026 · **ROADMAP:** **M26** · **Oracle:** `llama/llama.cpp/src/models/deepseek4.cpp` · **Pack:** mlx-lm `DeepseekV4ForCausalLM` 2-bit DQ (~90 GiB)

---

## Problem (why we looked)

Local disk already had `mlx-DeepSeek-V4-Flash-2bit-DQ` (`architectures: ["DeepseekV4ForCausalLM"]`, 19 shards). UMA/`tri_processor` ran it. Zerollama create/load said **unsupported architecture**. The pack is not GGUF; ggml Metal cannot grow a Flash graph from a llama.cpp pin without the same CSA/HCA work.

**Why not “just create and see”:** default `create` copies blobs. Use `--link` so the 90 GiB pack stays on the original volume.

---

## What v1 stubbed (and why that was invalid)

| Stub | Why it felt reasonable | Why it is wrong |
|------|------------------------|-----------------|
| CSA/HCA as full causal MHA | Ratio-0 path exists; warn once | Most layers are ratio **4 or 128**. Uncompressed MHA ignores compressor + indexer. Logits cannot match UMA. |
| Hash MoE weights = uniform `1/k` | `tid2eid` already picks experts | llama.cpp still runs `sqrtsoftplus(gate)` then `get_rows(probs, tid2eid)`. Uniform mix is the wrong convex combination. |
| Add `e_score_correction_bias` then gather those scores | DeepSeek V3 “bias for routing” | llama.cpp: bias is **selection only**; mix weights are **unbiased** `sqrtsoftplus`. |
| Inverse RoPE = `RoPE(x, -offsets)` | “Negate the angle” | MLX `fast_rope` uses `offsets[b] + 0..T-1`. Negating the start is not `R(-(offset+t))`. Need `-θ` on the last `qk_rope_head_dim` dims (`ggml_rope_ext_back`). |
| Skip YaRN | Ratio-0 uses plain theta | Compressed layers use yarn `factor` 16 + `compress_rope_theta`. |
| `wo_a` as 2-D `Linear` | mlx-lm file is 2-D `(o_group_dim, o_lora*groups)` | llama.cpp **reshapes** to `[o_group_dim, o_lora, groups]` and batched-mul. One matmul on `[B,L,G,D]` is a different map. |
| `LinearFactory` `tensorQuant: nil` | Global 4-bit gs64 | Experts are **2-bit**; packed width collides with 4-bit/gs64. Shape guess can pick the wrong bits with no error. |
| `tid2eid` transpose if `dim0 < dim1` | “Smaller dim is top-k” | llama.cpp is `{n_expert_used, n_vocab}`. Prefer “first dim == topk” over “smaller dim.” |
| HC `fn` always `x @ fn.T` | PyTorch `[out, in]` | ggml is `{hc_dim, mix_dim}`. mlx-lm is usually `[mix, hc*d]`. **Detect layout from shape.** |
| Tests without the 90 GiB file | CI must stay cheap | Green tests only proved **shapes**. They cannot catch a stubbed CSA. |

**Learning:** “Register the arch + load tensor names” is necessary and **not sufficient**. Flash’s identity is the **compressor**, not MLA LoRA.

---

## What we shipped (and why each piece)

| Piece | Why |
|-------|-----|
| `base.Register("DeepseekV4ForCausalLM")` + `imports.go` | Missing side-import → “unsupported architecture” even if the package exists (`tiny-agent` Qwen2 lesson). |
| GatherQMM 2-bit affine on stacked `switch_mlp` | Do not dequant 256 experts. |
| Hash: `tid2eid` **ids**, `sqrtsoftplus(gate)` **weights** | Matches `build_moe_ffn(..., selected_experts)`. |
| Bias only on `Argpartition`, gather unbiased | Matches `selection_probs` vs `probs` in llama-graph. |
| Grouped `wo_a` (G independent QMM/matmul) | Matches reshape + `mul_mat` per group. |
| Inverse **NORM** RoPE on **last** 64 of 512 | `LLM_ARCH_DEEPSEEK4` is `LLAMA_ROPE_TYPE_NORM` (consecutive pairs), not NEOX. Offset still `n_nope`. NEOX-on-the-tail produced number-salad on the linked pack. |
| YaRN freqs + **mscale=1** on compressed layers | mlx-lm/HF `attention_factor=1`. llama.cpp `dsv4_rope_attn_factor` is **not** this checkpoint. |
| Compressed RoPE at **write-pos `k*ratio`** | llama.cpp `state_write_pos = source_start`. `0..n-1` made CSA/HCA keys look like a short consecutive sequence vs Q at chat length. |
| CSA overlap compressor (coff=2) + indexer top-k | Ratio 4: raw SWA concat masked compressed keys. |
| HCA softmax compressor (coff=1) | Ratio 128: no indexer. |
| `ApplyQuantizationFromConfig` → factory | 2-bit gs32 vs 4-bit gs64 must not be inferred from packed columns alone. |
| `RotatingKVCache(sliding_window)` | llama.cpp `set_swa_pattern(0)` ⇒ **all** trunk layers are SWA. |
| Parser stays deepseek3 | Flash is not a new chat format. |

---

## Measurement / process lessons

1. **A stub with a slog.Warn is still a ship blocker** if most layers hit the stub. One warn at load does not make first-token quality “TBD.”
2. **Do not A/B until the pack is linked into mlxrunner.** Graph bugs are invisible until `--link` (or a copy) exists.
3. **Negating RoPE offsets is a popular wrong inverse.** Prove inverse by round-trip on the **same** kernel/helper, not by sign-flipping the cache offset tensor.
4. **`k_rot` Hadamard is cache-shift, not Flash identity.** llama.cpp applies it when rotating KV; skipping it on a non-shifting MLX cache is OK; cargo-culting it without the shift matrix is not.
5. **Uniform expert weights look “fair” and fail hash layers.** Layers 0–2 still have a gate MLP; hash only replaces **which** experts, not the mix.
6. **CI cannot hold 90 GiB.** Unit tests must encode **invariants** (hash layout, bias split, grouped out dim, quantFor bits). UMA token match is an operator gate, not `go test`.
7. **Do not remap every `*.scale` as affine quant.** mlxrunner maps `foo.weight.scale` → `foo.weight_scale`. HC mix tensors are named `model.hc_head.scale` / `attn_hc.scale`. Treating those as quant scales drops `loadHC` (`missing model.hc_head` after a 2610-tensor load).
8. **DSv4 RoPE is NORM, not NEOX.** llama.cpp lists `LLM_ARCH_DEEPSEEK4` with `LLAMA_ROPE_TYPE_NORM`. Half-split (NEOX) on the last 64 dims still “looks like RoPE” and still emits `0.1. 000. 2 The only 0p` garbage.
9. **Do not leave default PLD on a graph whose extra state is not in `cache.Cache`.** Haiku → Chinese replacement chars then `panic: speculation: cache restore to 195 failed` was PLD verify + empty compressed RoPE (`FromValues` `&bts[0]` on len 0), not UMA OOM. Parking PLD stops the crash; it does **not** make logits match UMA.
10. **This pack is mlx-lm, not GGUF.** llama.cpp `attn_factor≈0.78` and ggml `[dst,src]` comb layout do not apply. HF/mlx-lm: **`attention_factor=1`** (no cos/sin mscale) and **`softmax(-1)` + `comb.T @ residual`**. Using llama numerics on this checkpoint produced worksheet blanks / `UnUnUn`.
11. **Compressed RoPE is write-pos `k*ratio`.** Matches mlx-lm pool rows at window starts.
12. **CSA concat must not use the rotating ring as chronological K.** Roll SWA to oldest-first before mask+SDPA.
13. **Coherent Chinese + `: 2,5` is not UniAI.** After HC/RoPE numerics match mlx-lm, a haiku prompt still ignored English and tailed into a **sub-8-token** score loop (`### 评论` / `: 2,5`). mlx-serve loop-stop's min period 8 never fired. Short-period (2–7 ×6) is a decode halt, not a logit gold.
14. **Do not clamp shared-expert SwiGLU on this pack.** mlx-lm leaves shared experts unlimited; `swiglu_limit` is routed-only. Clamping the shared residual + MLX family temp **0.8** produced a fluent Chinese “social skills” FAQ for `Write me a haiku`. Default DSv4 sampling is **greedy** (mlx-lm sign-off); request `--temperature` still wins.

---

## Deferred (and why)

| Item | Why later |
|------|-----------|
| Compressor state inside `cache.Cache` snapshots | Needed for MLX speculate rewind. Until then **PLD is parked** on this arch (`ParkSpeculation`). Default-on PLD + SWA 128 restored to prefill+draft (~195) panics: `speculation: cache restore to N failed`, after empty `FromValues` in `ropeCompressed`. |
| llama.cpp compressor write-pos RoPE | **Shipped** as `k*ratio` (source_start). Still not the full dsv4 cache plan (dummy blocks / k_rot). |
| Path-based create / no blob copy | **Shipped** as `create --experimental --link` (`source_dir`). |
| MTP `num_nextn_predict_layers` | Pack has a nextn block; trunk graph first. |
| Indexer Hadamard `k_rot` | Only for cache shift. |

---

## Related

- [mlx-deepseek-v4-flash.md](./mlx-deepseek-v4-flash.md)
- llama.cpp `deepseek4.cpp` (`build_csa_lid_attention`, `build_overlap_compressed_kv_from_state`, `build_hc_pre`)
