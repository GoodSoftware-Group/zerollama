# FreeToken MoE lab (arXiv:2608.16157)

Offline simulator of FreeToken serving policies for zerollama. **No serve, no production ports.** Paper: [FreeToken](https://arxiv.org/abs/2608.16157). Code: `x/freetokenlab`.

We do **not** vendor FlashML. This lab answers: which policies are worth wiring into Flash-MoE / llama-server next.

## Run

```bash
go test ./x/freetokenlab/...
go run ./x/freetokenlab/cmd/sim
go run ./x/freetokenlab/cmd/sim -k 8 -experts 256 -ram 128
go run ./x/freetokenlab/cmd/sim -trace x/freetokenlab/testdata/anemll_mini.jsonl
./zerollama freetoken           # MoE headers + agent prefill lab (no weight load)
./zerollama freetoken --json
```

## What the sim encodes

| Policy | Paper | Lab |
|--------|--------|-----|
| \(q^\star = m\,B_P/B_H\) miss split | §3.2 | `QStarFillCount` / `SplitMisses` |
| LRU expert cache vs static / prefill-hot | Fig. 4b | `SimulateCache` + `SweepMissRates` (11% / 37% / 50% of pool) |
| `--moe-prefetch-temporal` | Flash-MoE | `PolicyLRUPrefetch` (install last-step experts before lookup) |
| Real routing dump | anemll `write_trace` | `LoadAnemllJSONL` (`n_tokens>1` prefill) |
| Full-layer prefill double-buffer | §3.1 | `OverlapPrefill` |
| Semantic anchors at tool/think cuts | §3.1 | `PrefillTokensWithSemanticAnchor` |

Hardware numbers are FreeToken Table 1. **`mac-uma`** sets \(B_P=B_H\) so \(q^\star\) fills every miss (unified DRAM: no residual host bandwidth for a parallel CPU expert path).

## Measured here (synthetic)

On `go run ./x/freetokenlab/cmd/sim` (80 MiB expert, 4 unique misses/layer):

- Discrete PCIe boxes (5090/4090/4060): concurrent CPU experts **cut miss latency** vs all-PCIe-fill.
- **5090 desktop** \(B_P\approx B_H\): little residual; policy ≈ all fill (paper’s dual-channel DDR5 case).
- **Mac UMA**: all fill. Steal **LRU + prefill overlap + semantic checkpoints**, not \(q^\star\).
- Default sim prints **two Zipf regimes**. **i.i.d.** (no locality): static / prefill-hot beat LRU. **75% stay:** LRU miss ~0.12 at 11% slots vs static ~0.44 — same qualitative win as Fig. 4b. `--moe-prefetch-temporal` does not help when LRU already holds the stay-set.
- Tool-edit at 24k prefix + 400 new tokens: sparse 4k checkpoints replay thousands of tokens; a boundary checkpoint prefills **400**.

anemll JSONL replay: `LoadAnemllJSONL` / `-trace`. Mini fixture parses. This host’s `qwen3.6-mtp` is 22 GiB with no sidecar — not captured in this session.

## Capture a real anemll trace (lab only)

anemll already writes JSONL from `write_trace`. **Do not** attach this to production `:11434`. Use `llama-cli` on a lab prompt:

```bash
# needs FLASH_MOE llama-cli from anemll-flash-llama.cpp, GGUF, optional sidecar
./build/bin/llama-cli -m "$GGUF" --moe-trace /tmp/moe.jsonl --moe-trace-harness \
  --prompt "Write a 3-step plan." -n 64
go run ./x/freetokenlab/cmd/sim -trace /tmp/moe.jsonl
```

This host has `qwen3.6-mtp:latest` (**22 GiB**, 256 experts, **no sidecar**). A capture would load the full GGUF; skip unless you explicitly want that RAM hit.

## chat_compress vs FreeToken anchors

`partitionChatTail` keeps the **recent tail** and inserts a **new summary** where the old head was. That is **not** a surviving-prefix edit:

| Edit | Radix reuse | Re-prefill |
|------|-------------|------------|
| Suffix strip (OpenClaw / FreeToken) | exact prefix up to boundary | new suffix only |
| **zerollama keep-tail `summary`** | system/developer prefix only | **summary + tail** |
| **`mode=placeholder`** | exact messages until first elided or peeled turn | rest of thread |

Placeholder **elides newest fat tools first** so the start of the thread stays byte-identical (FreeToken prefix KV). Oldest peel is last resort. `summary` stays opt-in. `/api/chat` runs this before prompt render. `keep_tail_tokens` applies to **summary** only.

Synthetic agent fixture (`ChatCompressLabCompare`, `num_ctx=4096`, no inference):

| Policy | reuse | recompute |
|--------|------:|----------:|
| none | 0 | 4627 |
| placeholder | **3017** | **15** |
| summary | 8 | 46 |
| suffix-strip+anchor (next 400 tokens) | 3017 | **400** |
| suffix-strip + sparse ckpt@4k | 3017 | ≥400 |

Placeholder **fits** by eliding the newest fat tools (~15 tokens of new prefill) and keeps a long exact prefix. `summary` still inserts a new system blob, so reuse stays ~8. On a *follow-up* turn, suffix-strip+anchor still prefills only 400 if KV kept that prefix. Echo `compression.elide_from` on the next append-only request so a larger `num_ctx` cannot restore an already-elided tool and split prefix KV.

`zerollama doctor` / `zerollama freetoken` print this line.

Do **not** load `qwen3.6-mtp` on this Mac while production Metal serve is up (22 GiB GGUF would contend for GPU).

## Next

1. Operator capture of a MoE trace when GPU is free (`llama-cli --moe-trace`, not `:11434`).
2. CUDA **5080-est** (`BP=49`, `BH=63.2`) is an interpolation until CT 1564 measures expert DMA vs host GEMM. `AdviseProfile("5080-est")` still wants a CPU miss-split that anemll does not expose.
3. `zerollama doctor` `freetoken MoE policy` uses local GGUF `expert_count` / `expert_used_count` when present (this host: 256 / k=8 → slots~19). Prefetch stays off on Mac UMA unless a real trace is i.i.d.
4. `./zerollama freetoken` prints MoE header advice plus the agent prefill table (placeholder vs summary vs suffix-strip). Do not load `qwen3.6-mtp` on production Metal serve.
5. Native `zerollama run` / `--experimental` / Go `api.ChatThread` echo `elide_from`. HTTP agents with a stable `prompt_cache_key` (including `/v1/responses` and `/v1/messages` extra_body) get the same cut server-side (per model, 256-key LRU, 30m; `cache_reset` clears). Explicit `compression.elide_from` still wins.

See [flash-moe.md](./flash-moe.md).
