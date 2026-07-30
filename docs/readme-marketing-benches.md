# Megaprompt tokenize benches (README evidence)

**Date:** 2026-07-30 · **Host:** Apple Silicon · **Patches:** 0106–0126 in sibling `../llama.cpp`  
**Harness:** `./scripts/bench/run_tokenize_bpe_identity_bench.sh --bench`  
**Method:** same binary; **fast** = accelerated path; **legacy** = `LLAMA_BPE_FORCE_LEGACY=1` (bit-identical — identity gates green)  
**Seeds:** `mega_1mib` (mixed), `mega_1mib_ascii`, `mega_1mib_chat` — **1 MiB** each  

## Results (median ms / 1 MiB)

| Vocab | Seed | Legacy | Fast | Speedup |
|-------|------|-------:|-----:|--------:|
| **Qwen2** | mixed | **269** | **65** | **4.2×** |
| **Qwen2** | ascii | **386** | **95** | **4.1×** |
| **Qwen2** | chat markers | **389** | **81** | **4.8×** |
| **Qwen3.5** | mixed | 100 | **28** | **3.5×** |
| **Qwen3.5** | ascii | 114 | **20** | **5.8×** |
| **Qwen3.5** | chat | 98 | **26** | **3.8×** |
| **Gemma4** | mixed | 102 | **30** | **3.4×** |
| **Gemma4** | ascii | 134 | **36** | **3.7×** |
| **Gemma4** | chat | 117 | **31** | **3.7×** |
| **GPT-2** | mixed | **377** | **56** | **6.7×** |
| **GPT-2** | ascii | **353** | **50** | **7.1×** |
| **GPT-2** | chat | 239 | **72** | **3.3×** |
| Llama3 BPE | mixed | 43 | 25 | 1.8× |
| Llama3 BPE | ascii | 23 | 21 | 1.1× |

## What this proves

- Legacy encode still costs **hundreds of ms** on Qwen2 / GPT-2 agent megaprompts **before any forward**.
- Fast path is typically **~3–7×** on those vocabs; chat-marker seeds **~3.3–4.8×**.
- Absolute ms vary with host load (quiet labs have shown ~8–9 ms/MiB ascii); cite **speedups + legacy latency** for marketing.

## Reproduce

```bash
LLAMA_CPP_ROOT=../llama.cpp ./scripts/bench/run_tokenize_bpe_identity_bench.sh --bench
```

Related: [faster-bpe-tokenize.md](./faster-bpe-tokenize.md) · [findings](./faster-bpe-tokenize-findings.md)

## Related claims (other arms)

| Claim | Status | Notes |
|-------|--------|-------|
| Prompt cache (L3) turn-2 TTFT | Doc’d | [gpu-profiles-l3.md](./gpu-profiles-l3.md); `l3_cache_smoke.sh` on lab ports when GPU free |
| Decode ~+7% vs upstream Metal | Prior lab | `./scripts/phase/m4_upstream_vs_zerollama_bench.sh` — **lab ports only** (script stops `:11434`) |
