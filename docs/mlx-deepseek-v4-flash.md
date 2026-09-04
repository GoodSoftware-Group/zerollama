# MLX DeepSeek-V4 Flash

**Why this doc:** The Mac lab already has an mlx-lm **2-bit DQ** pack (`DeepseekV4ForCausalLM`, ~90 GiB). UMA/`tri_processor` can run it. Zerollama mlxrunner could not: no registered arch, no Flash attention, no stacked MoE. This is the product graph in `x/models/deepseekv4`. Findings (wrong stubs, measurement traps): [mlx-deepseek-v4-flash-findings.md](./mlx-deepseek-v4-flash-findings.md).

**Status (Aug 2026):** Graph **loads names + CSA/HCA/MoE/HC**. Not UMA-logit gold. Default **PLD is parked**. Default **sampling is greedy** (mlx-lm sign-off); `--temperature` still overrides. Shared-expert SwiGLU is **unclamped**. Register with **`create --experimental --link`**. Rebuild `./zerollama`; do **not** auto-restart production `:11434`.

**ROADMAP:** **M26**. **Code:** `x/models/deepseekv4/` (side-import `x/mlxrunner/imports.go`). **Oracle:** `llama/llama.cpp/src/models/deepseek4.cpp`. **Pack:** `/Users/user1/models/mlx-DeepSeek-V4-Flash-2bit-DQ` (local; not vendored).

---

## Why MLX at all (not GGUF, not UMA-only)

| Option | Why not the product path |
|--------|--------------------------|
| **Wait for GGUF** | Flash is CSA/HCA + hyper-connections + hash MoE. Public GGUF + zerollama ggml Metal does not implement this graph. |
| **Only UMA `tri_processor`** | Fine for a lab oracle. Agents talk to `:11434` mlxrunner, not the UMA toolkit. |
| **Dequant 256 experts to fp16** | ~80 GiB extra resident on 128 GiB UMA. **2-bit GatherQMM** is the point of the DQ pack. |
| **Stub CSA as full MHA** | Most layers are `compress_ratios` **4 or 128**. Full causal MHA is a **different function**. First tokens are not “a bit worse.” |

**Why llama.cpp is the oracle, not mlx-lm Python:** Flash kernels and HC/Sinkhorn live in `deepseek4.cpp`. mlx-lm is the **weight layout** (`attn.wq_a`, `ffn.switch_mlp`, `attn.compressor.*`). Match names to mlx-lm; match numerics to llama.cpp.

---

## What Flash actually is

`config.json` `compress_ratios` (43 trunk layers on this pack): **0, 0, 4, 128, 4, 128, …, 4, 0**.

| Ratio | Name | Why |
|-------|------|-----|
| **0** | Raw MLA | First two + last layer. LoRA Q, single-head KV, SWA on raw cache (`sliding_window` 128). |
| **4** | **CSA** | Overlap compressor (coff=2) + lightning indexer (`index_topk` 512). Attention = SWA raw **concat** selected compressed keys. |
| **128** | **HCA** | Softmax compressor over 128 tokens, no indexer. Same concat pattern. |

Shared around every block: **hyper-connections** (`hc_mult=4`, Sinkhorn mix) and **MoE** (256 routed, top-6, 1 shared, `sqrtsoftplus`). Layers `0..num_hash_layers-1` (3) use **`tid2eid`** instead of learned top-k **for expert ids only**.

**Why inverse RoPE after attention:** grouped `wo_a` is defined on **derotated** heads (`ggml_rope_ext_back`). Skipping it or negating `SeqOffsets` is not an inverse.

**Why NORM pairs, not NEOX:** llama.cpp `LLM_ARCH_DEEPSEEK4` → `LLAMA_ROPE_TYPE_NORM` (consecutive `(x[2i], x[2i+1])` on the last 64 dims). NEOX half-split is a different map.

**Why YaRN only on compressed layers:** llama.cpp / mlx-lm disable yarn on ratio-0. Compressed uses `compress_rope_theta` + yarn freqs. **Do not scale cos/sin** (`attention_factor=1` on this mlx-lm pack). Compressed keys RoPE at **write-pos `k*ratio`**. mHC is **HF `softmax(-1)` + `comb.T @ residual`**, not ggml dest-fastest comb.

---

## Operator

### Disk / create

Default `zerollama create --experimental` **copies** safetensors into blobs. That still needs ~one extra model of free space.

**In-place (Flash / any large mlx-lm tree):**

```bash
# From anywhere (do not paste the comments; zsh treats * in comments as a glob)
OLLAMA_HOST=127.0.0.1:11435 ./zerollama create --experimental --link dsv4-flash:q2 \
  --dir /Users/user1/models/mlx-DeepSeek-V4-Flash-2bit-DQ
```

**Do not** omit `--link` on this pack. The copy path can reclaim source shards after import.

### Rebuild

The arch is linked only if `x/mlxrunner/imports.go` is compiled into the binary you serve. Rebuild after pulling this graph; old binaries still say unsupported architecture.

### Ports

Lab only (`OLLAMA_HOST=127.0.0.1:11435`). Do not contend with production `:11434` / `:8081`.

### Parser

The pack's `chat_template.jinja` defaults to **`thinking_mode=chat`** (`</think>` after Assistant), not open-ended `<think>`. DeepSeek CoT is often Chinese. HF `generation_config.json` `temperature: 1` / `top_p: 1` is identity, not a recipe — baking it made "hello" wander into Chinese health-tips then collapse. Those identity values are now omitted; MLX family `top_p` 0.95 applies. Rebuild; no re-`--link`. `./zerollama run dsv4-flash:q2 --think=false` if the CLI still opens Thinking.

---

## Tensor map (mlx-lm names)

| Role | Prefix |
|------|--------|
| Q LoRA | `model.layers.i.attn.wq_a` / `wq_b` / `q_norm` |
| KV | `attn.wkv` / `kv_norm` |
| Out | `attn.wo_a` / `wo_b` / `attn_sink` |
| CSA/HCA | `attn.compressor.{wkv,wgate,ape,norm}` |
| Indexer (ratio 4) | `attn.indexer.wq_b`, `weights_proj`, `indexer.compressor.*` |
| HC | `attn_hc` / `ffn_hc` / `model.hc_head` |
| MoE | `ffn.gate`, `ffn.gate.tid2eid` (hash), `e_score_correction_bias` (noaux), `ffn.switch_mlp.{gate,up,down}_proj` stacked `[E,…]` |
| Shared | `ffn.shared_experts.*` |

Quant: global affine **4-bit gs64**; experts **2-bit** (gate often **gs32**, up/down **gs64**). **Why per-tensor must reach `LinearFactory`:** packed column counts for 2-bit/gs128 and 4-bit/gs64 collide. Guessing bits from shape silently corrupts experts.

---

## What is still wrong vs llama.cpp (do not claim UMA parity)

| Gap | Why it matters |
|-----|----------------|
| Compressor token state is **on the layer**, not `cache.Cache` | Speculative rewind / snapshot restore will not roll CSA/HCA buffers. PLD is parked. Greedy decode is the intended path. |
| No cache-shift Hadamard `k_rot` | llama applies it when shifting a stationary KV ring. mlxrunner **physically** rolls SWA keys to oldest-first before CSA concat (decode wrap used to attend the wrong 128 tokens). |
| Tests do not load the pack | CI stays cheap; it cannot catch graph bugs. A/B vs UMA is the gate. |

---

## Tests

```bash
CGO_ENABLED=1 go test ./x/models/deepseekv4
```

Shape/quant/sinkhorn/hash-layout/grouped `wo_a` only. **Why no 90 GiB test:** CI machines and laptops cannot check out the pack; identity vs UMA is an operator lab.

---

## Related

- [findings](./mlx-deepseek-v4-flash-findings.md)
- [MLX routing](./mlx-routing-policy.md) — safetensors → mlxrunner, not Python runtime
- [Apple Silicon](./apple-silicon-metal.md)
- llama.cpp: `llama/llama.cpp/src/models/deepseek4.cpp`
