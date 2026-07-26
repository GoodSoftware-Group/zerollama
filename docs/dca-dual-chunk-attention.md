# Dual Chunk Attention (DCA)

**Status (Jul 2026):** **Native ggml CUDA DCA is the product path** (Qwen2 / Qwen2.5). SGLang is a **lab logit oracle only**, not the serve path.

## Why DCA

Qwen’s official long-context path (1M class) uses **Dual Chunk Attention** ([arXiv:2402.17463](https://arxiv.org/abs/2402.17463)): rewrite RoPE so relative offsets stay in the pretrained window, then run up to **three** FlashAttention passes (intra / succ / inter) and **LSE-merge**. YaRN alone is not the same algorithm.

Upstream llama.cpp has YaRN + GQA FA but **no** DualChunk RoPE / multi-pass FA. This fork ports dense DCA into `vendor/llama-cpp-86d86ed4` (patches under `llama/patches/`).

## Operator path (native — preferred)

1. Convert / stamp a Qwen2 or Qwen2.5 Instruct(-1M) GGUF so it carries DCA keys (see below).
2. Build / run stock **llama-server** from the patched vendor tree (`./scripts/build/build_llama_server.sh`) or **zerollama-runtime** with `LLAMA_SERVER_BIN` pointing at that binary.
3. Serve normally — when `dca.chunk_size > dca.local_size`, the Qwen2 graph uses DualChunk RoPE + 3× FA + LSE merge. YaRN frequency scaling is forced off; `dca.original_context_length` feeds length temperature `s(L)` only.

```bash
# Example: runtime + patched llama-server
export LLAMA_SERVER_BIN=/path/to/vendor/llama-cpp-86d86ed4/build/bin/llama-server
# zerollama serve with ZEROLLAMA_RUNTIME=1 as usual
```

### Algorithm (dense)

| Tensor | RoPE position |
|--------|----------------|
| **P_k** / **P_q_intra** | `pos % chunk_len` |
| **P_q_succ** | `min((pos % chunk_len) + chunk_len, chunk_size)` |
| **P_q_inter** | `chunk_size` (constant) |

`chunk_len = chunk_size - local_size`. Length temperature: `s(L) = max(1, 0.1·log(L/L0)+1)`.

Decode KV ranges (`n = (S−1)//chunk_len`): intra `[n·c, S)`; succ `[(n−1)·c, n·c)` if `n≥1`; inter `[0, (n−1)·c)` if `n≥2`. Merge normalized O with LSE soft-max weights. CUDA graphs are disabled when FA exports LSE.

### Prefill

Prefill uses the **same three mask stages** over the current ubatch (intra causal; succ/inter non-causal) with per-token DualChunk RoPE and LSE merge — equivalent to SGLang’s `chunk_len` stride loop for dense attention, without a separate product engine. Empty succ/inter stages are gated to weight 0.

### HF / GGUF config

| HF field | GGUF key |
|----------|----------|
| `dual_chunk_attention_config.chunk_size` | `{arch}.attention.dca.chunk_size` |
| `dual_chunk_attention_config.local_size` | `{arch}.attention.dca.local_size` |
| `dual_chunk_attention_config.original_max_position_embeddings` | `{arch}.attention.dca.original_context_length` |

Keys: patch **0094** (gguf-py). C++ load + DualChunk graph: **0095+**. Convert stamps HF config when present; `scripts/gguf/stamp_dca_metadata.py` can sidecar-stamp.

Models in scope: **Qwen2 and Qwen2.5** (`QWEN2` / `qwen2.cpp`). Sparse vertical/slash and Qwen3 are deferred.

## Oracle gate (required before ship)

SGLang dense (`--attention-backend dual_chunk_flash_attn`, sparse off) is the **logit oracle**, never the serve path.

```bash
# n=0: native DCA ≈ stock FA (intra only)
python3 scripts/dca_oracle_logits.py --mode n0 \
  --native-url http://127.0.0.1:8080 --stock-url http://127.0.0.1:8082

# n≥1: native ≈ SGLang dense
python3 scripts/dca_oracle_logits.py --mode n1 \
  --native-url http://127.0.0.1:8080 --sglang-url http://127.0.0.1:30000
```

| Case | Expectation |
|------|-------------|
| `S < chunk_len` (`n=0`) | Native DCA ≈ stock llama FA |
| `n≥1` / `n≥2` | Native ≈ SGLang dense on same weights / prompt / greedy |
| Drift | Fail smoke (`--max-abs` / atol/rtol) |

### Validation status (5080, Jul 2026)

- **n=0 PASS** on Qwen2.5-3B GGUF stamped `chunk_size=256 local_size=64` (`chunk_len=192`): top-8 logprobs vs stock FA, max \|Δ\| ≈ 0.05, greedy `Paris`.
- Fixes needed for that gate: FA→LSE graph barrier + GPU LSE buffer; prefer **fattn-tile** when exporting LSE (MMA leaves meta uninit when `np>1` → NaN); larger `graph_max_nodes` when DCA on (**0098**).
- **Chunk-boundary smoke** (same stamp): S≈181/193 ≈ stock (no NaN); S≈251 diverges from stock as expected once succ stage engages.
- **n≥1 vs SGLang**: needs a local HF Instruct-1M tree + working `sglang` env (`scripts/serve/sglang_dca_example.sh`). Helper `scripts/dca_unit_ref.py` covers mask/RoPE/`s(L)`/`soft_max` merge math offline.
- Offline unit: `python3 scripts/dca_unit_ref.py`

## Legacy SGLang proxy

`modality_backends.inference=sglang` + `server/sglang_chat_proxy.go` remain for experiments / video; **do not** treat them as the Qwen long-ctx product path.

## References

- SGLang oracle: `sglang/.../dual_chunk_flashattention_backend.py`, `rope_variant.py` (`DualChunkRotaryEmbedding`)
- Vendor: `vendor/llama-cpp-86d86ed4` — `llama-dca.h`, `qwen2.cpp`, FA LSE export (`ggml_flash_attn_ext_set_lse`)
- Watchlist: [llama-fork-watchlist.md](./llama-fork-watchlist.md) Lab Q
