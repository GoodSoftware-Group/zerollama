# Megakernel decode lab — CUDA lanes

**Why this lab:** [megakernel-zoo](https://github.com/dangxingyu/megakernel-zoo) showed that a single persistent decode kernel can beat vLLM/SGLang on H100 and B200, mostly by sitting close to HBM bandwidth. Those binaries do not run on our cards. This lab asks the same question on the GGUF stack we ship: **is decode already at the memory floor, or is there a launch / skeleton gap a fused kernel could still close?**

**Idea donor:** [megakernel-borrowings.md](../upstream/megakernel-borrowings.md)

**Lanes:** [cuda-lanes.md](../design/serving/cuda-lanes.md) — RTX 5080 (`sm_120`, 16 GiB, CT 1564) and RTX 4090 (`sm_89`, including dual-4090). Metal is out of scope; Apple decode has its own kernel story.

**Status:** written 2026-11-19. Rungs below are **not run yet**. Fill the results table at the bottom on the host that has the GPU. Do not paste numbers from the zoo.

---

## What a “yes” looks like

Their playbook, translated into a stop/go for ggml:

| If the measurement says | Then |
|-------------------------|------|
| Dense decode already moves ≥ ~70% of the measured device-copy rate | Fusion headroom is small. Leave the kernel alone. Byte wins stay on the quant path (Q8 / Q4 / NVFP4). |
| CUDA graphs off are within ~5% of graphs on | Launch gaps are not the tax. A persistent kernel that exists to delete launches will not pay. |
| Graphs off are ≥ ~15% slower | Graphs are already doing the job their first persistent-kernel win did (~0.9 ms of launch gaps on Qwen3-8B). Keep the graph invalidation path honest; do not add a second capture runtime. |
| Implied traffic is under ~50% of the copy rate, and graphs are already on | There is a real gap. Next step is one `nsys` trace of a decode step, read with their rule: spills and per-op calls show up as extra L2 traffic behind the weight stream. Still a ggml change, not an `mkserve` port. |
| MoE implied traffic (active experts only) is far under the copy rate | Same shape as their North / Flint / Inkling result: the step is a skeleton of routing and hand-offs. The watch item is fused MoE in ggml ([deepgemm-borrowings.md](../upstream/deepgemm-borrowings.md)), not a model-specific engine. |
| Greedy HTTP is ≥ ~15% below `llama-bench`, sampling adds more | The remaining tax is the server loop. Device argmax would touch only the greedy slice. Chat samples, so this does not justify an in-kernel tokenizer loop. |

---

## Safety

- `llama-bench` does not listen. Prefer it.
- Any `llama-server` in this lab binds **127.0.0.1:18082**. Never `:11434`, `:8080`, `:8081`, or `:2083`.
- Do **not** run `scripts/phase/l1_cuda_concurrent_bench.sh` for these rungs. It kills whatever is on `:8080` and any `zerollama` process.
- On a one-GPU host, production may already occupy the card. `nvidia-smi` first. If the only GPU is busy, stop and ask the operator. Do not kill the listener.
- On dual-4090, set `CUDA_VISIBLE_DEVICES` to a card that is not serving. `scripts/phase/l2_cuda_direct_bench.sh` defaults that to `1`.

```bash
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.used --format=csv
```

`compute_cap` should be `12.0` (5080) or `8.9` (4090). Anything else: record it and still run the bench, but do not compare the percentage-of-peak against the table below without writing the new peak down.

Published device bandwidth (the ceiling, not the expected GEMV rate):

| GPU | Compute capability | Published bandwidth |
|-----|--------------------|---------------------|
| RTX 5080 | 12.0 | 960 GB/s |
| RTX 4090 | 8.9 | 1008 GB/s |
| RTX 3090 | 8.6 | 936 GB/s |

A contiguous device copy lands near that number. Decode GEMV of small tiles lands lower. Their useful floor was a measured copy, about 90% of nominal on H100, and they judged engines against that — not against the brochure.

---

## MK-0 — copy floor

Run once per card, while the GPU is otherwise idle. This is the denominator for every later rung.

PyTorch is only a ruler here. Skip it when the machine has no CUDA torch; record “unmeasured” and use 85% of the published figure as a clearly labeled stand-in.

```bash
python3 - <<'PY'
import time
import torch
assert torch.cuda.is_available(), "no CUDA torch"
n = 1 << 30  # 1 GiB
a = torch.empty(n // 2, dtype=torch.float16, device="cuda")
b = torch.empty_like(a)
torch.cuda.synchronize()
for _ in range(5):
    b.copy_(a)
torch.cuda.synchronize()
iters = 20
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(iters):
    b.copy_(a)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
gbps = (n * iters) / dt / 1e9
name = torch.cuda.get_device_name()
cc = torch.cuda.get_device_capability()
print(f"device={name} cc={cc[0]}.{cc[1]} copy={gbps:.1f} GB/s")
PY
```

Write the GB/s into the results table as `copy_GBps`.

---

## MK-B — implied decode bandwidth

**Question:** for a dense GGUF, how close is steady decode to that copy rate?

`llama-bench` prints a model `size`. That size is the right byte count for a dense model (attention KV is extra and shows up in MK-D). Implied traffic:

```text
GB/s ≈ (size_bytes) × (tg tokens/s) / 1e9
fraction ≈ implied_GBps / copy_GBps
```

Use the production KV type on 5080 (`q8_0`). `llama-bench` does not sample and does not speak HTTP, which is what we want for this rung.

```bash
# MK_BENCH = the llama-bench next to the llama-server you actually ship
# MK_MODEL = a dense GGUF that fits. 5080 reference pair from the Jul 2026
#            fork watch: Llama-3.2-3B and Llama-3.1-8B, Q4_K_M and Q8_0.
#            F16 only when the card has room after KV.

"$MK_BENCH" \
  -m "$MK_MODEL" \
  -ngl 99 -fa on \
  -ctk q8_0 -ctv q8_0 \
  -p 0 -n 128 -r 3 \
  -o json
```

Repeat for Q4_K_M and Q8_0 of the same architecture when both files exist. The Jul 2026 5080 result (Llama-3.1-8B, FA on, q8_0 KV: **q8_0 weights ~157 tg**, **f16 ~113 tg**) already hints that fewer weight bytes win. This rung turns that into a fraction of `copy_GBps` so we can see whether the faster quant is merely “less to read” or still leaving a kernel gap.

**Read it:**

- Fraction ≥ ~0.70 on Q8 and Q4 → decode is bandwidth-bound. A megakernel’s dense win (the last 10–20% down to the floor) is the most we could hope to buy, and only if the trace shows waste. Default decision: stop.
- Q4’s fraction is high and Q8’s is low → Q8 is compute- or dequant-bound. Look at MMQ vs the Blackwell MMA path ([cuda-lanes.md](../design/serving/cuda-lanes.md) P2) before any fusion project.
- Both fractions under ~0.50 with graphs left on → go to the nsys follow-up in the decision table. Do not start by writing a kernel.

Keep prefill as its own line so a decode conclusion cannot leak into it:

```bash
"$MK_BENCH" -m "$MK_MODEL" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -p 512 -n 0 -r 3 -o json
```

Their fast engines read the prompt one token per pass. We keep chunked prefill. A bad pp number is a prefill bug, not a reason to fuse decode.

---

## MK-G — CUDA graph on vs off

**Question:** did ggml graphs already take the launch-gap win?

Same binary, same model, same KV types. Second run disables capture:

```bash
GGML_CUDA_DISABLE_GRAPHS=1 "$MK_BENCH" \
  -m "$MK_MODEL" \
  -ngl 99 -fa on \
  -ctk q8_0 -ctv q8_0 \
  -p 0 -n 128 -r 3 \
  -o json
```

Compare tg128. Their persistent kernel was worth about 12% once the math was already in custom CUDA, because the eager launcher had been inserting ~0.9 ms of gaps. If our off/on ratio is near 1.0, that gap is gone. If off is much slower, the graph is load-bearing: keep [decode-graph invalidation](../design/serving/decode-graph-invalidation.md) on whenever KV is rewritten, and do not stack a second graph cache.

---

## MK-D — context depth

**Question:** how much of the step is the KV stream rather than the weight stream?

```bash
"$MK_BENCH" \
  -m "$MK_MODEL" \
  -ngl 99 -fa on \
  -ctk q8_0 -ctv q8_0 \
  -p 0 -n 128 -d 0,1024,4096 -r 3 \
  -o json
```

They saw ~100 µs/token of attention growth from position 64 to 1088 on an otherwise flat Qwen3-8B step. Record tg at each depth. A flat line means weights dominate (MK-B is the whole story). A steep drop means long-context decode is KV bandwidth and cache type, which is the L2 watchlist, not a megakernel.

---

## MK-M — MoE on 4090 (active bytes)

**Question:** is gpt-oss-20b decode limited by the bytes of the experts it actually touches?

Their MoE engines were 2–5× slower than the active-byte floor, because routing and phase hand-offs dominated. Dual-4090 already has a correctness-and-speed pair: NVFP4 ~212 tok/s and MXFP4 ~220 tok/s at 8k ([cuda-lanes.md](../design/serving/cuda-lanes.md)). Re-bench with `llama-bench` only if that JSON is gone. One card, lab-quiet:

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
# paths from cuda-lanes.md (astra)
#   .../gpt-oss-20b-nvfp4.gguf
#   .../gpt-oss-20b-mxfp4.gguf

"$MK_BENCH" -m "$MK_MOE_MODEL" -ngl 99 -fa on -p 0 -n 128 -r 3 -o json
```

Do **not** multiply tok/s by the whole file size. Routed experts are a fraction of the file. Read `expert_count` and `expert_used_count` from the load log (or `gguf-dump`) and count bytes as:

```text
dense tensors + (expert_used_count / expert_count) × routed expert tensors
```

If that split is annoying to do by hand, report two numbers and label them: `file_GBps` (upper bound, all experts) and `active_GBps` (the one that counts). A high `file_GBps` that exceeds the copy rate is a sign the file-size estimate is wrong, not that the kernel beat physics.

**Read it:** active fraction near the copy rate → MoE decode is already streaming. Active fraction far below → the ggml MoE path is paying a skeleton (many small launches per layer). That is the DeepGEMM “fuse the expert GEMMs” watch, scoped to sm_89 / sm_120, and it still has to beat the MXFP4 number we have.

---

## MK-H — host and sampling tax

**Question:** how much of a real reply is outside the kernel `llama-bench` measures?

`llama-bench` excludes tokenization and sampling. Their HTTP path added on the order of 12–18% versus the in-process kernel, and sampling another 1–11%. Run this only after MK-B, and only on a lab server.

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
"$LLAMA_SERVER_BIN" \
  -m "$MK_MODEL" \
  --host 127.0.0.1 --port 18082 \
  -ngl 99 -fa on -c 2048 \
  -ctk q8_0 -ctv q8_0 \
  --no-webui
```

Time two completions, 128 new tokens, prompt sent as text, client on the same machine. First call is warmup.

| Leg | Request |
|-----|---------|
| Greedy | `temperature: 0` |
| Sampled | `temperature: 0.8`, `top_p: 0.95` |

Tokens per second from the streamed response, over the decode window only (after the first token). Compare with the MK-B tg128 of the same GGUF.

- Greedy HTTP within ~10% of `llama-bench` → the host loop is cheap at this size. Stop.
- Greedy HTTP ≥ ~15% slower → profile the server, not the kernel. A device argmax does not fix HTTP framing, grammar, or tool calls.
- Sampled meaningfully slower than greedy → that is the sampling tax. Expected. It is not a reason to move argmax into a persistent kernel, because sampled chat cannot stay inside that kernel.

Stop the lab server when finished. Leave anything on `:8080` / `:8081` / `:11434` alone.

---

## Follow-up only if MK-B says the gap is real

One capture, one decode, graphs left **on**:

```bash
nsys profile -t cuda,osrt -o /tmp/mk-decode --force-overwrite true \
  "$MK_BENCH" -m "$MK_MODEL" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -p 0 -n 32 -r 1
```

Read the timeline for gaps between ggml ops inside the replay, and for kernels that move little data but take a long time (norms, RoPE, residual adds, router). Their playbook’s inner-loop rule applies to whatever op is on the critical path: extra calls and local-memory spills become L2 trips sitting behind the weight loads. A fused epilogue is worth sketching only for an op the trace actually names.

Do not capture on a card that is serving production.

---

## Explicitly not a rung

| Temptation | Why it is absent |
|------------|------------------|
| Clone the zoo and `mkserve serve` on 5080 or 4090 | The engine refuses the GPU. CUDA 13.2 will not change the architecture check. |
| Port one `.cu` into `llama/patches/` | Shapes, MMA, and shared-memory budget are compiled for Hopper or B200. sm_120 and sm_89 need a different kernel even when the schedule idea survives. |
| 12-bit exact weight codec | Smaller win than the quants we already ship, and it is a per-checkpoint table built at load. |
| Device-side batch scheduler and prefix hash | Serving engines were slower at steady decode (Inkling 11.0 ms vs 3.3 ms) in exchange for tail latency. Our tail work is the Go queue and L3, which we can change without giving up sampling. |
| Autoresearch agent loop against a production GPU | The evaluator wants a quiet card for hours and an FP64 reference of one checkpoint. Fine as a personal project. It is not a zerollama gate. |

---

## Results

Fill on the machine that ran the rung. One row per GPU.

| Date | Host / GPU | CC | copy GB/s | Model | Quant | tg128 | Implied GB/s | Fraction of copy | Graphs-off / graphs-on | tg @ d=0 / 1024 / 4096 | Notes |
|------|------------|----|-----------|-------|-------|-------|--------------|------------------|------------------------|------------------------|-------|
| | | | | | | | | | | | |

MoE rows (4090), if run:

| Date | Model | tok/s | file GB/s | active GB/s | Fraction of copy (active) | Notes |
|------|-------|-------|-----------|-------------|---------------------------|-------|
| | | | | | | |

HTTP row, if run:

| Date | Model | llama-bench tg128 | greedy HTTP | sampled HTTP | Notes |
|------|-------|-------------------|-------------|--------------|-------|
| | | | | | |
