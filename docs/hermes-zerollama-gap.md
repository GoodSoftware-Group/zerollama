# Hermes ↔ zerollama gap analysis

**Audience:** Hermes maintainers wiring OpenAI `/v1` + native zerollama APIs; zerollama contributors prioritizing harness gaps.

**Related:** [openai-harness-qos-wire-shapes.md](./openai-harness-qos-wire-shapes.md) (M15c), [hermes-gap-closure-findings.md](./hermes-gap-closure-findings.md) (M15e), [agent-qos-and-project-tracking.md](./agent-qos-and-project-tracking.md), [inference-wishlist-host.md](./inference-wishlist-host.md), [gpu-profiles-l3.md](./gpu-profiles-l3.md).

**Date:** Jul 2026 (post-M15e gap closure).

---

## Why this doc exists

A Hermes wishlist compared zerollama’s advertised APIs to what the harness actually calls. Several items were labeled “blocked by `/v1` rejecting custom fields” or “missing.” After audit:

- **QoS flat `extra_body` 400s are fixed** (M15c) — outdated as a blanket claim.
- Several “unused” features are **shipped and reachable** — Hermes just needs the right field name or a native `/api/*` call.
- A few items were **real product gaps**; **M15e shipped them** (`think` bind, timeout, `preempted_reason`, multi-GPU can-load topology, cache-pin REST, public batch).

---

## Summary table

| Wishlist item | Reality | Hermes action |
|---------------|---------|---------------|
| Flat `qos_class` / `project_*` 400 | **Fixed (M15c)** — folded into `options.zerollama` | Keep sending; prefer nested when possible |
| `think` on `/v1` | **Shipped (M15e)** — bound + mapped; wins over aliases | Prefer native `think` or aliases |
| `keep_alive` | **Works on `/v1`** | Send `keep_alive: "30m"` (or via `extra_body`) |
| `/api/pin` | **Shipped** native lease API | Call `/api/pin` (not OpenAI chat) |
| `/api/cache/pin` | **Shipped (M15e)** — prefix-cache lease | Pin L3/MLX branch by `prompt_cache_key` |
| `/api/can-load`, `/api/propose-load` | **Shipped** dry-run + topology | Preflight before heavy jobs |
| `/api/metrics` | **Shipped** Prometheus text | Scrape alongside Hermes metrics |
| `/v1/audio/*`, `/v1/images/*` | **Shipped** (model-gated) | Point at local speech/image models |
| KV / prompt cache | **Shipped** key + optional `/api/cache/pin` | Stable key every turn |
| Batch inference | **Shipped** `POST /v1/chat/completions/batch` | Same-model text-only; group by model client-side; see §8 wire format |
| `preempted_reason` | **Shipped** wait-abort (M15e) + mid-stream `done_reason=preempted` on MLX soft cancel (M15f) | Retry; do not treat truncated stream as final |
| Per-call server timeout | **Shipped** — `timeout` field | Prefer over client-only deadlines |
| Structured JSON / grammar | **Shipped** `response_format` + runtime forward (M15f); GBNF via `format.type=gbnf` | Prefer `json_schema`; tools+grammar → 400 |
| Multi-GPU in can-load | **Shipped** (topology report) | Host topology fields; not a TP planner |

---

## A. “Already in zerollama, not used by Hermes”

### 1. Reasoning / `think`

**Why Hermes thought it was blocked:** trap 77 400s on unknown top-level keys — `"think"` was allowlisted but unbound (accept-and-drop). **M15e binds and maps it.**

| Works |
|-------|
| top-level `"think": true` / `"high"` (native; wins over aliases) |
| `enable_thinking: true` |
| `reasoning_effort: "high"` / `reasoning: { "effort": "…" }` |
| `chat_template_kwargs.enable_thinking` |
| Native `/api/chat` `"think"` |

**Status:** **shipped (M15e)** — top-level `"think"` binds on `ChatCompletionRequest` and maps in `FromChatRequest` with precedence over `reasoning_*` / `enable_thinking`. Aliases still work.

### 2. `keep_alive` + `/api/pin`

| Mechanism | Surface | Why |
|-----------|---------|-----|
| `keep_alive` | `/v1/chat/completions` field + `extra_body` merge | Unload TTL after last use — stops 5m default thrash |
| `POST /api/pin` | Native only | Session **eviction lease** (Orient/Decide); does not load; stronger than keep_alive |

**Status:** `keep_alive` **usable from Hermes `/v1` today**. Pin is **shipped but native-only** — Hermes must call `/api/pin` if it wants lease semantics (multi-runtime GGUF pin is fail-closed; see [inference-wishlist-host.md](./inference-wishlist-host.md)).

### 3. `/api/can-load` + `/api/propose-load`

**Shipped.** Dry-run capacity without `GetRunner`. Propose = batch can-load + co-residency honesty.

**Why Hermes isn’t using them:** client only speaks OpenAI chat today — not a missing server feature.

**Status:** **shipped but Hermes-blocked** (won’t call non-`/v1` unless wired).

### 4. `/api/metrics`

**Shipped** Prometheus text at `GET /api/metrics`. Capability advertised on `/api/version`.

**Status:** **shipped**; Hermes can scrape in parallel with its own metrics.

### 5–6. Audio + images

| Route | Status |
|-------|--------|
| `POST /v1/audio/speech` | Shipped (Piper / remote-tts) |
| `POST /v1/audio/transcriptions` | Shipped (Whisper / multimodal) |
| `GET /v1/audio/voices` | Shipped |
| `POST /v1/images/generations` (+ edits) | Shipped (image-capable models) |

**Why “cloud-only” in Hermes:** harness routing, not zerollama absence. Needs local speech/image model tags configured.

**Status:** **shipped & usable** (model-gated).

---

## B. “Shipped in M15e (was missing)”

### 7. KV / prompt cache API

**What exists:** request-scoped **`prompt_cache_key`** (OpenAI top-level, options, or `extra_body`) → L3 slot pin / MLX live-session / radix donor policy. Optional blob fetch `GET /api/kv/blob/:digest`. Model residency via `/api/pin`. **M15e:** dedicated **`POST /api/cache/pin`** / **`DELETE /api/cache/pin/:id`** keyed on `prompt_cache_key` (MLX trie eviction skip + L3 disk TTL bump). Does **not** force idle llama-server `id_slot` retention.

**Status:** **shipped (M15e)** for key + optional cache-pin lease. Hermes’s largest win remains sending the **same** `prompt_cache_key` every turn.

### 8. Batch inference

**Shipped (M15e):** `POST /v1/chat/completions/batch` → Go validates → Python `generate_batch`. Cap `min(8, llama_parallel_slots)` (Go hard-caps at 8; Python may tighten further). Same model required; tools/vision/think rejected; non-stream only for v1. Internal `/internal/generate-batch` remains for smokes. OpenAPI: `ChatCompletionsBatchRequest` / `ChatCompletionsBatchResponse`.

#### Wire format (stable)

**Request**

```json
{
  "model": "llama3.2",
  "stream": false,
  "requests": [
    { "messages": [{ "role": "user", "content": "a" }], "max_tokens": 64 },
    { "messages": [{ "role": "user", "content": "b" }], "prompt_cache_key": "job-1" }
  ]
}
```

- Top-level `model` is required unless every item sets the same `model`.
- Nested items are OpenAI-shaped chat bodies (`messages` required). Per-item `model` must match the shared model.
- Rejected: `stream: true`, mixed models, `tools` / non-`none` `tool_choice`, vision / logprobs / think, empty `requests`, size above cap.
- Requires the Python runtime path (`modality_backends.inference` or `ZEROLLAMA_RUNTIME`).

**Response** (wrapper — not a bare list)

```json
{
  "object": "chat.completion.batch",
  "model": "llama3.2",
  "count": 2,
  "completions": [
    {
      "id": "chatcmpl-…",
      "object": "chat.completion",
      "created": 1720000000,
      "model": "llama3.2",
      "system_fingerprint": "fp_zerollama_runtime",
      "choices": [
        { "index": 0, "message": { "role": "assistant", "content": "…" }, "finish_reason": "stop" }
      ],
      "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
    }
  ]
}
```

- `completions[i]` corresponds to `requests[i]` (same order).
- Each element is a normal `chat.completion` object; `usage` may be zero placeholders on the runtime path; optional `vram_num_ctx` may appear.

#### Hermes client notes

- **Group by model client-side** before calling — the server does **not** accept mixed models in one POST (400). One HTTP call per model group; reassemble by original index.
- Chunk groups larger than `min(8, llama_parallel_slots)` into multiple POSTs.
- Do not treat this as OpenAI’s async Batch API (`/v1/batches`); it is synchronous fan-out decode batching.

**Status:** **shipped** as a Hermes product API (text-only, same-model). Mixed-model server-side grouping is a non-goal until a client needs it after adopting this schema.

### 9. `preempted_reason`

**Shipped (M15e):** when a request is aborted while deferred behind higher QoS (`waitForSlot` / global defer / fulfillment), error bodies may include `preempted_reason` (e.g. `lower_wait_interactive`).

**Shipped (M15f mid-stream):** MLX soft cancel of lower-class inflight when interactive admits → terminal chunk `done_reason: "preempted"` + `preempted_reason` (OpenAI `finish_reason: "preempted"`). **Not** ggml/Python hard kill.

**Status:** **shipped** for wait-abort / busy + MLX mid-stream soft cancel.

### 10. Per-call server timeout

**Shipped (M15e):** `timeout` on `GenerateRequest` / `ChatRequest` / `/v1` (and `extra_body`). Server wraps context with `WithTimeout`; `DeadlineExceeded` → **504** `{"error":"request timeout","timeout_seconds":N}` (distinct from 499 cancel).

**Status:** **shipped**.

### 11. Structured output (grammar / JSON schema)

**Shipped on `/v1`:** `response_format: { "type": "json_schema", "json_schema": {…} }` (and `json_object`) → `ChatRequest.Format` → llama-server grammar/schema path.

**M15f:** Python runtime proxies previously **dropped** format — now forward `json_schema`/`grammar` on `/completion`. GBNF: `format: {"type":"gbnf","grammar":"root ::= ..."}` (native or `extra_body.format`). **tools + grammar → HTTP 400.**

Top-level `"format"` on `/v1` is bound (not passthrough-only) as of M15f for GBNF.

**Status:** **shipped & usable** on native + runtime-proxied GGUF. MLX has no grammar sampler yet.

### 12. Multi-GPU awareness in `/api/can-load`

**Shipped (M15e).** Runtime `/internal/vram-estimate` embeds `vram_estimate.topology` (`device_count`, `tensor_parallel`, `split_mode`, `tensor_split`, `main_gpu`); Go copies those onto `CanLoadResponse` / `ProposeLoadPlan`. Ggml path reports best-effort `device_count` from `discover.GPUDevices` with an explicit note that multi-GPU fit is heuristic (not a TP plan).

**WHY not a planner:** TP layout is configured in Python runtime YAML/env; can-load surfaces that host topology so Hermes can decide *whether* to use multi-GPU hosts, without inventing a second fit engine on the Go path.

**Status:** **shipped** (topology report). Hermes still cannot ask can-load to *invent* a TP split plan — that remains runtime config.

---

## Recommended Hermes adoption order (WHYs)

1. **`prompt_cache_key` every turn** — largest TTFT win already on the wire; no new zerollama feature.
2. **`keep_alive` + optional `/api/pin`** — kill multi-second reload without waiting on new APIs.
3. **`think` / `enable_thinking` / `reasoning_*`** — surface coder-next traces (`think` is bound on `/v1` as of M15e).
4. **`response_format.json_schema`** — kill judge/tool parse failures.
5. **`/api/can-load` before heavy pulls** — fail closed before inference; read topology fields when targeting multi-GPU hosts.
6. **Scrape `/api/metrics`** — fill per-model scheduling gap cheaply.
7. **Use per-call `timeout` + `preempted_reason`** — server-enforced deadline and QoS wait abort signal (M15e).
8. **Use `/api/cache/pin` + `/v1/chat/completions/batch`** when prefix residency or fan-out decode batching helps (M15e).

Deep learnings (WHYs per fix): [hermes-gap-closure-findings.md](./hermes-gap-closure-findings.md). OpenAPI: `GET /openapi.json` (`server/openapi/openapi.yaml`).

---

## Non-goals of this doc

- Changing Hermes client code in this repo (lives in the Hermes tree).
- Promising multi-resident Python GGUF (`stable_multi_model_swap` remains false).
- Treating trap 77 removal as the fix for `think` — folding/binding is required, not just allowlisting.
