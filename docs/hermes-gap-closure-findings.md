# Hermes gap closure (M15e) — findings & learnings

**Audience:** contributors extending OpenAI `/v1` or agent harness APIs; anyone auditing “Hermes blocked” claims.

**Related:** [hermes-zerollama-gap.md](./hermes-zerollama-gap.md), [openai-harness-qos-wire-shapes.md](./openai-harness-qos-wire-shapes.md) (M15c), [agent-qos-and-project-tracking.md](./agent-qos-and-project-tracking.md), [inference-wishlist-host.md](./inference-wishlist-host.md).

**Status:** Shipped Jul 2026 (M15e). Batch wire-format polish Jul 2026 (OpenAPI `ChatCompletionsBatchResponse`). OpenAPI: `server/openapi/openapi.yaml` → `GET /openapi.json`.

---

## Why this doc exists

Hermes’s wishlist mixed three classes of item:

1. **Already shipped under another name** (e.g. `enable_thinking` vs unbound `think`).
2. **Native `/api/*` only** (can-load, pin, metrics) — not missing, just not on `/v1`.
3. **Real product gaps** — accept-and-drop fields, no server timeout, no wait-abort reason, no public batch, no cache-pin lease, no can-load topology.

M15e closed class (3). This doc records **why** each fix took the shape it did, so the next harness request does not re-open false gaps or invent a second planner.

---

## Finding 1 — Allowlisting ≠ binding (`think`)

**What:** Top-level `"think"` was on the trap-77 passthrough allowlist so Hermes stopped getting 400, but `ChatCompletionRequest` never had a `Think` field. Gin/JSON bind ignored it; `FromChatRequest` never saw it.

**Why that is worse than a loud 400:** operators believe reasoning is on; the model runs without think; A/B “thinking off” arms look identical.

**Learning:** Grow passthrough + struct field + `FromChatRequest` mapping **together**. Precedence must be explicit: native `Think` > `reasoning_*` > `enable_thinking` aliases — otherwise SDK flat aliases overwrite an intentional `think: false`.

**Non-goal:** Removing trap 77. Unknown keys must still 400.

---

## Finding 2 — Client disconnect ≠ server timeout

**What:** Without a per-call `timeout`, only HTTP cancel/`context.Canceled` stopped work. Hermes client deadlines often left the Go/Python side running until the socket closed late — or not at all on keep-alive proxies.

**Why 504 vs 499:** `DeadlineExceeded` is a **server policy** (“this call may not exceed N”); `Canceled` is **client abandoned**. Hermes retry logic treats them differently (escalate vs drop).

**Learning:** Apply `applyRequestTimeout` on **every** inference ingress that can hold a slot:

| Path | Wired |
|------|--------|
| `/api/generate` handler + runtime generate proxy | yes |
| `/api/chat` handler + **runtime chat proxy** | yes (audit: chat proxy was initially missed) |
| `/v1/chat/completions` runtime proxy | yes |

**Audit lesson:** Middleware that aborts before `ChatHandler` must re-apply the same timeout wrap — otherwise the common Python-proxy path silently ignores `timeout`.

---

## Finding 3 — `preempted_reason` is wait-abort, not mid-stream kill

**What:** When `waitForSlot` / global defer / fulfillment wait returns because the request context ended, wrap with `qosDeferAbortError{policy}` so `handleScheduleError` can emit `preempted_reason`.

**Why not hard-preempt in-flight decode:** killing an auxiliary generation mid-token would corrupt streams and force complex rollback. Hermes asked for “why did I wait and die,” not “steal the GPU from a running decode.”

**Learning:** Preserve `%w` / `errors.As` through `reserveScheduleQoS` → `scheduleRunner`. Wrapping with `fmt.Errorf("… %v", err)` would drop the policy string.

---

## Finding 4 — Topology report ≠ TP planner

**What:** `CanLoadResponse` / `ProposeLoadPlan` now carry `device_count`, `tensor_parallel`, `split_mode`, `tensor_split`, `main_gpu` from runtime `vram_estimate.topology` (ggml: best-effort `device_count` only).

**Why not invent a Go TP plan:** Tensor split is already configured in Python YAML/env. A second planner on the Go path would drift from what llama-server actually loads and lie about co-residency.

**Learning:** Surface **what the host already uses**; let Hermes decide *whether* to schedule heavy jobs on multi-GPU nodes. Document ggml as heuristic so thrash-sensitive clients do not treat `device_count=2` as “fits both GPUs.”

---

## Finding 5 — Model pin ≠ cache pin

**What:** `/api/pin` leases **model residency** (block eviction / soft UnloadAllRunners). `/api/cache/pin` leases a **`prompt_cache_key`** (MLX trie eviction skip + L3 disk TTL bump).

**Why separate REST:** Hermes wants prefix KV to survive idle gaps **without** claiming exclusive model residency (and without fail-closed multi-GGUF pin rules). Collapsing both into one API would either over-promise idle `id_slot` retention or under-serve model Orient/Decide leases.

**Honest limitation (document in `notes`):** does **not** force llama-server idle slot retention while no request is in flight — SlotAllocator does not hold empty slots today. Primary wins remain: MLX branch + L3 file TTL.

**Learning:** Hook MLX via `mlxrunner.CacheKeyPinned` package var (server `init`) — mlxrunner cannot import `server` (cycle). Python notified best-effort via `/internal/cache/pin`.

---

## Finding 6 — Public batch is a thin proxy, not a Go scheduler

**What:** `POST /v1/chat/completions/batch` → Go validates same-model / no tools-vision-think / max 8 → Python `generate_batch`.

**Why thin:** Go `pending_queue` is about **model load**, not decode fan-out. Phase 15 already batches decode in-process; inventing a second batch queue in Go would duplicate policy and fight VRAM broker.

**Constraints that keep the API honest:**

- Same model (one GGUF / one runner).
- No tools / vision / think (those need the interactive chat path).
- Non-stream only for v1 (streaming batch is a separate product).
- Response is a **wrapper** `{object:"chat.completion.batch", model, count, completions:[…]}` — not a bare OpenAI list. Documented in OpenAPI + [hermes-zerollama-gap.md](./hermes-zerollama-gap.md) §8.

**Hermes client:** group requests by model (and chunk ≤8) before POSTing; map `completions[i]` back to `requests[i]`. Server-side mixed-model grouping is deferred — same-model decode batching is the product.

**Learning — underspecified schemas block adoption:** shipping the route with OpenAPI text “OpenAI-shaped list / wrapper” was enough for zerollama operators who read the Python return dict, but not for Hermes. Aux work needed a **discriminator** (`object`) and an **order guarantee** before client grouping could start. **Why document before extending the API:** mixed-model server grouping looks convenient, but it would hide VRAM broker + load policy inside a chat endpoint; client grouping is the honest contract.

**Learning:** Reject empty `"tools": []` as present is acceptable false-positive — batch tools are unsupported anyway.

**Learning — not OpenAI Batch API:** this is synchronous fan-out decode (same request/response), not `/v1/batches` job IDs + polling. Clients that map to OpenAI’s async Batch product will mis-implement reassembly.

---

## Adoption order (WHYs for Hermes)

1. **Stable `prompt_cache_key` every turn** — largest TTFT win; already on the wire.
2. **`keep_alive` + optional `/api/pin`** — stop 5m default thrash.
3. **Bound `think` / aliases** — reasoning traces without accept-and-drop.
4. **`timeout` + `preempted_reason`** — server deadline + QoS-aware retry.
5. **`/api/can-load` topology** before heavy multi-GPU jobs.
6. **`/api/cache/pin` + `/v1/chat/completions/batch`** when prefix residency or fan-out decode helps.

---

## Code map

| Concern | Location |
|---------|----------|
| `/v1` think + timeout bind | `openai/openai.go`, `openai/chat_extras.go` |
| Request timeout helpers | `server/request_timeout.go` |
| QoS wait-abort wrap | `server/qos_preempt.go`, `mlx_sidecar_gate.go`, `fulfillment.go` |
| Can-load topology | `server/can_load.go`, `runtime/runtime/engine.py` (`vram_estimate.topology`) |
| Cache pin | `server/cache_pin.go`, `x/mlxrunner/cache_pin_hook.go`, `runtime/runtime/cache_pins.py` |
| Batch proxy | `server/runtime_v1_chat_batch_proxy.go`, `runtime/runtime/server/openai_v1.py` |
| OpenAPI | `server/openapi/openapi.yaml` |

---

## Finding 7 — Mid-stream preempt signal (M15f)

**What (wait-abort was not enough):** Hermes can catch `preempted_reason` on 499/503/504, but once tokens stream, a cancelled decode used to look like a normal finish (`done_reason=stop`) or a bare error — agents acted on half tool calls.

**What we shipped:** terminal chunk `done_reason: "preempted"` + `preempted_reason`, mapped to OpenAI `finish_reason: "preempted"` (zerollama extension string).

**Soft cancel scope — MLX only:** when interactive waits with policy `interactive_wait_inflight_lower`, the gate cancels the lower-class session's infer context. Victim Completion path emits the preempted done chunk instead of a generic cancel error.

**Architectural blockers for ggml/Python hard kill (still non-goals):**

- `UnloadAllRunners(Forced)` still defers when `refCount>0` — stream safety predates Phase B.
- Python `model_swap` waits for idle; no mid-decode abort of llama-server `/completion`.
- True hard kill needs subprocess abort + KV resume protocol — separate design.

**Learning:** Prefer an honest terminal signal on the one path that can soft-cancel today over lying that the whole fleet mid-stream preempts.

---

## Finding 8 — Runtime proxy dropped structured output (M15f)

**What:** Native Go llama-server path already forwarded `format` as `json_schema`/`grammar`. Python runtime `/completion` payloads never included those keys — so CUDA/runtime-proxied models ignored Hermes `response_format`.

**Fix:** merge `format` / OpenAI `response_format` into options; `apply_format_to_completion_payload` on llama-server HTTP. GBNF via `{"type":"gbnf","grammar":"..."}`. tools+grammar → **400**.

**Interactions (explicit):**

| Combo | Behavior |
|-------|----------|
| `think` + grammar | Grammar constrains final content only (thinking unconstrained) |
| `tools` + grammar | HTTP 400 |
| `stream` + grammar | Token-by-token (llama-server), same as schema |
| Invalid GBNF | Prefer fail at request/start (llama-server validates) |
| Speculative decode + grammar | Unverified — out of scope |

**Non-goals:** MLX logits grammar sampler; inventing OpenAI `response_format.type=gbnf` (use native `format` / `extra_body.format`).

---

## Non-goals (still)

- Hard mid-decode kill for ggml / Python-proxied llama-server (see Finding 7).
- Ggml multi-GPU TP planner / inventing `tensor_split` on the Go path.
- Idle llama-server `id_slot` lease while no request is in flight.
- Multi-resident Python GGUF (`stable_multi_model_swap` remains false).
- MLX GBNF/schema-constrained sampling.
