# Flare borrowings (agent KV / capacity policy ideas)

**Why this doc:** NVIDIA’s [flare-local-inference](https://github.com/NVIDIA/flare-local-inference) is a research local server for **SM120/SM121** (Blackwell / DGX Spark class): Ignition prepares with PyTorch + FlashInfer/b12x, Helios (Rust) serves captured CUDA graphs. It targets long chats, coding agents, and repeated tool use. Zerollama does **not** vendor Helios/Ignition or adopt FlashInfer as a serve backend. We keep a short bring / watch / skip list so agent prefix-cache and capacity policy stay aimed at **L3 + ggml / llama-server**, not a second PyTorch→Rust stack.

**Related:** [gpu-profiles-l3.md](./gpu-profiles-l3.md), [vllm-borrowings.md](./vllm-borrowings.md), [radix-prefix-share.md](./radix-prefix-share.md), [decode-graph-invalidation.md](./decode-graph-invalidation.md), [runtime-env.md](./runtime-env.md), [cuda-lanes.md](./cuda-lanes.md) (5080 = sm_120 overlap), [upstream-siblings.md](./upstream-siblings.md), [ROADMAP — L3](./ROADMAP.md#local-voice--llama-borrowings-eliza-v3).

**Sources (read, do not depend on at serve time):**

| Tree | Upstream | Role for us |
|------|----------|-------------|
| Flare | [NVIDIA/flare-local-inference](https://github.com/NVIDIA/flare-local-inference) | **Primary** idea donor — agent prefix checkpoints, capacity admission, KV+recurrent resume |
| Core design | [docs/core-design.md](https://github.com/NVIDIA/flare-local-inference/blob/main/docs/core-design.md) | Architecture invariants (prepare vs serve, cache ownership) |
| CLI / memory limits | [docs/cli-design.md](https://github.com/NVIDIA/flare-local-inference/blob/main/docs/cli-design.md) | `--kv-cache-limit`, snapshot slots, prefix-cache sizing |
| API metrics | [docs/api.md](https://github.com/NVIDIA/flare-local-inference/blob/main/docs/api.md) | `cached_tokens`, host vs GPU reuse counters |

**Optional sibling (Mac lab / CT):** `../flare-local-inference` — clone only when rematching agent-cache policy. **Do not** bind Flare, Ignition, or Helios into `:11434` / `:8081` / production ports.

**Last checked:** 2026-10-23 — README + AGENTS + core-design / CLI / API skim (no local tip SHA; clone optional).

---

## Non-goals (explicit)

| Item | WHY not in zerollama |
|------|----------------------|
| Vendor Helios / Ignition / Heliopause / `flare` CLI | Product path is Go daemon + Python runtime + GGUF/`llama-server` (+ MLX); not PyTorch prep → Rust graph packages |
| FlashInfer / b12x as serve kernels | Coupled to Flare by design; our GEMM/attn stays ggml/Metal/MLX ([deepgemm-borrowings.md](./deepgemm-borrowings.md) same rule) |
| Eager-serve fallback or runtime graph recapture | Flare forbids this; we already invalidate graphs on L3 clear — do not import their package format |
| Orbit multi-host TP / cross-node prep | Single-GPU / UMA product; fleet is LocalAI-style routing, not Flare rank packages |
| “No env vars for behavior” config philosophy | We use `ZEROLLAMA_*` + YAML profiles; do not rewrite config surface to match Flare |
| Claiming Flare tok/s or DGX Spark numbers on 5080/Metal | Different stack and memory topology; measure L3 on our lanes |

---

## Useful to have (problem decomposition)

Delivery path is always **L3 policy + llama.cpp / our patches**, not Helios sources.

| Idea | Why it matters | Zerollama stance |
|------|----------------|------------------|
| **Turn-boundary prefix checkpoints** (instruction / prompt / answer) | Tool loops re-hit system+tools; retaining ~N recent exchanges beats “fill all free VRAM with arbitrary prefixes” | **Bring next** — rematch against `prefix_cache_policy` / retention interval; optional snapshot-slot budget per concurrency (Flare: ~8 slots ≈ 4 exchanges; skip retention when backing exhausted) |
| **KV + recurrent (+ selector) as one resume contract** | Prefix reuse that drops GDN/mamba/hybrid state is silently wrong | **Taken as invariant** — keep hybrid Radix gate + MLX recurrent compact honest; attn+recurrent `seq_cp` live probe still open ([vllm-borrowings.md](./vllm-borrowings.md)) |
| **`--kv-cache-limit {memory\|context\|concurrency}`** | Operator dial: fill free memory vs one-context history vs scale with concurrency | **Watch → docs/CLI** — map onto existing admission / VRAM reserve language in [runtime-env.md](./runtime-env.md); no Helios flag port |
| **Admit or typed-fail — no silent capacity shrink** | Prepared capacities must not quietly omit variants | **Taken** — Phase 11 admission + L3 policy rejects; reinforce in smokes when outside prepared ctx/slots |
| **Always expose cache reuse in usage** (`cached_tokens`, Anthropic `cache_read_input_tokens`) | Agents tune prompts on hit rate | **Mostly taken** — OpenAI/Anthropic usage paths; ensure host vs GPU split stays visible where LMCache/disk tiers apply (SGLang `sglext` pattern) |
| **Host prefix store + GPU checkpoint tiers** | Long agent history leaves GPU for live rows; restore from host/disk | **Already covered directionally** — L3 RAM/disk + LMCache `file://`/`redis://`; watch Flare host_hits / host_reused_tokens naming for metrics polish only |
| **Shared runtime policy; model recipes own compute** | Avoid per-model scheduler allowlists | **Taken as habit** — L3/vLLM path uses `KVCacheSpec` geometry, not model-name flags |

---

## Already covered (skip re-implementing)

| Flare theme | Where we already are |
|-------------|----------------------|
| GPU prefix cache for agent / tool loops | L3 slot bridge + `prompt_cache_key` ([gpu-profiles-l3.md](./gpu-profiles-l3.md)) |
| Cross-request shared system prefix | Radix donor / `POST /kv/seq-copy` ([radix-prefix-share.md](./radix-prefix-share.md)) |
| Selective retention / SWA / draft × cache | `prefix_cache_policy.py`, vLLM borrowings |
| CUDA graph break on KV invalidate | Epoch + native / HTTP invalidate ([decode-graph-invalidation.md](./decode-graph-invalidation.md)) |
| Speculative draft/verify metrics | Runtime MTP / SPECULATIVE + usage fields |
| SM120 hardware class | 5080 production ([cuda-lanes.md](./cuda-lanes.md)) — kernels via ggml pin, not Flare packages |

---

## Bring next (ordered)

| ID | Work | Owner | Notes |
|----|------|--------|-------|
| **FL1** | **Turn-boundary checkpoint policy** | Python L3 | Prefer retaining checkpoints at chat turn / tool-result boundaries under memory pressure; document skip-when-no-backing (Flare snapshot slots). Measure on 5080 tool-heavy chat vs current retention interval alone. |
| **FL2** | **KV-limit mental model in operator docs** | Docs | One short table: memory-fill vs context-cap vs concurrency×context — map to existing env/YAML; no new Helios-compatible flags required. |
| **FL3** | **Host vs GPU reuse metrics polish** | Go / Python | When disk/LMCache contributes, expose split counters (or log fields) analogous to Flare `reused_tokens` vs `host_reused_tokens`; do not require Flare Prometheus names. |

---

## Watch (upstream)

| Signal | Action |
|--------|--------|
| Flare docs on tiered prefix-cache offload / Engram host cache | Rematch against L3 disk + LMCache finalize; borrow **eviction under live-row pressure**, not package format |
| Learned drafter packaging / MTP design notes | Compare to runtime SPECULATIVE / draft-spec×cache guards only |
| Blackwell FlashInfer kernels landing ideas in ggml-org | Prefer [LLAMA_CPP_PIN.md](../runtime/LLAMA_CPP_PIN.md) / fork watch — not FlashInfer bind |
| Recurrent / GDN / mHC recipe state layouts | Only if a GGUF family we ship needs the same resume contract |

---

## Operator / agent checklist

When someone links Flare:

1. Ask: **policy idea** (checkpoints, KV limits, metrics) or **serve stack** (Helios/FlashInfer)? Only policy can affect our product.
2. Map to **Bring next** / **Useful to have** — if it is Rust packages, Orbit TP, or b12x, stop.
3. Prefer rematch against [gpu-profiles-l3.md](./gpu-profiles-l3.md) + [vllm-borrowings.md](./vllm-borrowings.md) before opening a new subsystem.
4. Never start `flare run` / Ignition serve on production ports; optional clone is read-only research.

---

## Citation (if we reference publicly)

Apache-2.0 — [NVIDIA/flare-local-inference](https://github.com/NVIDIA/flare-local-inference). Prefer citing Flare for **agent cache / capacity-contract** discussion; do not imply zerollama runs Helios.
