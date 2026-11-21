# Megakernel-zoo borrowings (decode-step ideas)

**Why this doc:** [dangxingyu/megakernel-zoo](https://github.com/dangxingyu/megakernel-zoo) publishes agent-written CUDA engines that run one LLM decode step as a single persistent kernel, plus a playbook of what those runs measured. Zerollama does **not** serve those engines. We keep a short bring / watch / skip list so decode-bandwidth work stays on **ggml / llama-server** on the CUDA lanes we actually have (`sm_120`, `sm_89`).

**Lab (what to run):** [megakernel-cuda-lab.md](../research/megakernel-cuda-lab.md)

**Related:** [cuda-lanes.md](../design/serving/cuda-lanes.md), [decode-graph-invalidation.md](../design/serving/decode-graph-invalidation.md), [deepgemm-borrowings.md](deepgemm-borrowings.md) (grouped GEMM / MoE fuse — complementary), [llama-fork-watchlist.md](llama-fork-watchlist.md) (5080 quant A/B already measured).

**Write-up:** [A Megakernel per Model](https://dangxingyu.github.io/megakernel/). Apache-2.0.

**Optional sibling:** `../megakernel-zoo` — clone only to read `autoresearch/skills/megakernel/playbook.md` and an engine’s `.cu`. **Do not** `pip install` it into the runtime, and **do not** point `mkserve` at `:8080` / `:8081` / `:11434`.

**Last checked:** 2026-11-19 — README tip (19 engines, 8 models, CUDA 13.2, H100 `sm_90a` / B200 `sm_100a`).

---

## Non-goals

| Item | Why it stays out of zerollama |
|------|-------------------------------|
| `mkserve` as a runtime backend | Engines are one Hugging Face checkpoint on one GPU. Product serve is GGUF through llama-server, any pulled model. |
| Running their kernels on 5080 / 4090 | They compile for `sm_90a` or `sm_100a` and refuse any other pair. 5080 is `sm_120`, 4090 is `sm_89`. |
| A persistent megakernel in ggml | Their fast path holds 8 sequences and 2,048 tokens, reads the prompt one token per pass, and does greedy argmax on device. Sampling, tools, speculative decode, and long context need the host in the loop. |
| Track-B 12-bit weight codes as a GGUF type | Lossless per-checkpoint repack (exponent window + exact escapes). Our byte reduction is already Q4 / Q8 / NVFP4 / MXFP4, which change the function. |
| Device-side request scheduler | Their serving engines (Qwen3-8B, Inkling only) admit, page KV, and match prefixes inside the kernel. Prefix match is a 64-bit hash with no token compare. L3 radix stays the product path. |
| Quoting their 1.1–2.0× vs vLLM/SGLang on our cards | Those rows are H100/B200, BF16-class checkpoints, HTTP, one sequence. Different bytes, different launch tax. |

---

## Useful to have (problem decomposition)

Delivery path is llama.cpp measurement and, only if a lab rung says the gap is real, a ggml patch. Their sources are the textbook, not the binary.

| Idea | What they measured | Zerollama stance |
|------|--------------------|------------------|
| **Bytes per decode step vs a measured copy rate** | Dense models landed within ~10–20% of a practical HBM floor (H100 ~3.1 TB/s of 3.35 nominal; B200 ~7.3 of 8). MoE steps were 2–5× the byte time: routing, norms, and phase hand-offs dominate when few experts are live. | **Try** — [lab MK-B / MK-M](../research/megakernel-cuda-lab.md#mk-b--implied-decode-bandwidth). This decides whether fusion can matter here. |
| **One launch per step, token fed back on device** | Replacing per-op launches was about −12% (~0.9 ms) on Qwen3-8B after the kernels themselves were already custom. Host argmax + readback was ~46–170 µs/token. | **Try as a graph A/B** — ggml CUDA graphs are the analogue. [MK-G](../research/megakernel-cuda-lab.md#mk-g--cuda-graph-on-vs-off). If graphs already ate that gap, stop. |
| **No calls and no per-thread spills on the inner path** | A phase function that spilled registers became an L2 round-trip queued behind the weight stream. A 160-byte descriptor copied onto every thread stack made an empty op cost 5.2 µs vs 0.8 µs for the barrier. | **Carry into an nsys reading** only if MK-B shows a large gap. Rule for any future fused ggml op: keep the inner path in registers / shared memory. |
| **Sampling and HTTP are a separate tax** | Greedy lives in the kernel. `mkserve` sampling costs 1–11%. `llama-bench` excludes sampling; their headline is HTTP. | **Try** — [MK-H](../research/megakernel-cuda-lab.md#mk-h--host-and-sampling-tax). Device argmax helps only the greedy gap, and chat samples. |
| **Time an engine only after logits match a reference** | FP64 reference, relative error within 2× Hugging Face BF16, batch independence, bit-identity between builds. | **Standing rule** for any later kernel A/B. Speed numbers without a token match are not a result. |
| **Attention grows with depth inside an otherwise flat step** | ~100 µs from position 64 to 1088 at batch 1 on Qwen3-8B, on top of a flat weight stream. | **Try** — `llama-bench -d` in [MK-D](../research/megakernel-cuda-lab.md#mk-d--context-depth). Separates KV traffic from weight traffic. |

---

## Already covered

| Their theme | Where we already are |
|-------------|----------------------|
| Collapse decode launches | ggml CUDA graphs, invalidated on KV change ([decode-graph-invalidation.md](../design/serving/decode-graph-invalidation.md)) |
| Fewer weight bytes | Q4 / Q8 / K-quants; NVFP4 / MXFP4; native FP8 ([cuda-lanes.md](../design/serving/cuda-lanes.md)). 5080 Llama-3.1-8B: stock **q8_0** beat f16 on decode ([llama-fork-watchlist.md](llama-fork-watchlist.md)) |
| MoE larger than a card | Flash-MoE slot-bank, FreeToken lab — residency, not a fused megakernel |
| Prefix reuse | L3 radix. Their 64-bit hash without a token compare is a limitation we should not copy |
| Speculative / draft verify | They tried an exact draft track on one run and the operator ended it. Our MTP / ngram / DFlash path stays separate |

---

## Operator checklist

When someone links megakernel-zoo (or a “we fused the whole model into one kernel” post):

1. Read the GPU. `sm_90a` / `sm_100a` claims stop at this doc. `sm_120` / `sm_89` claims go to the [lab](../research/megakernel-cuda-lab.md).
2. Ask which tax they removed: launch gaps, weight bytes, or MoE skeleton. Those are three different experiments.
3. Compare against `llama-bench` with graphs on, not against a PyTorch eager baseline. Their −65% first step was “leave Hugging Face ops”; we already did that by using ggml.
4. Keep prefill chunked. Their fast engines consume the prompt one token per pass; that is a decode bench, not a serve design.
5. Do not bind `mkserve` to a production port.
