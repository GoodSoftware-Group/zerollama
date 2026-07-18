# Inference wishlist — host capacity & admission

**Audience:** operators and `@elizaos/plugin-ollama` / Orient Inventory / Decide clients.  
**Related:** [scheduling-vram-policy.md](./scheduling-vram-policy.md), [phase11-runtime-admission.md](./phase11-runtime-admission.md), [agent-qos-and-project-tracking.md](./agent-qos-and-project-tracking.md), live OpenAPI `/docs`.

This tracks the **zerollama host** wishlist (capacity + admission). Plugin → acquaintance wiring (`CounterpartRecord.kindExt`, hire maps) lives in the ElizaOS plugin repo — not here.

---

## Why this exists

Agent fleets (Orient Inventory, Decide, hire maps, autotune) need to answer:

1. **What is this host configured for?** (`NUM_PARALLEL`, `MAX_LOADED_MODELS`, queue caps) — without SSHing to read env.
2. **Can I admit model X at num_ctx Y without starting inference?** — so Decide can budget concurrency before a graph run.
3. **Is the host wedged or did the model refuse?** — empty done chunks and runner deaths must not look like semantic failure.
4. **When should I retry?** — busy 503s need a standard hint (`Retry-After`).

**Why Phase A only:** stable multi-model swap, pin/reserve leases, and a propose sidecar are scheduler redesigns (Python is single-resident GGUF; Go eviction + VRAM broker thrash). Shipping capacity APIs first unblocks Decide without pretending swap is solved.

**Why not a public unload API:** Phase 8 already chose automatic eviction over operator unload endpoints. can-load reports `needs_eviction`; it does not add `/api/unload`.

---

## Status

| # | Item | Status | Why this status |
|---|------|--------|-----------------|
| 1 | Stable multi-model swap | **missing** (Phase B) | Python `ModelSwapGate` = one GGUF; Go multi-runner + cross-stack broker still thrash under alternating tags |
| 2 | Readable runtime config | **shipped** | Env-only was guesswork for fleet clients |
| 3 | Capacity dry-run | **shipped** | Loopback `/internal/vram-estimate` was not a public product API |
| 4 | Queue & metrics | **shipped** | Status had queues; scrapers need Prometheus text |
| 5 | Pin / reserve N models | **missing** (Phase B) | `fulfillment` / `keep_alive` are request-scoped, not session leases |
| 6 | Admission 503 + Retry-After | **shipped** (Retry-After half) | Priority queues (`qos_class`, `options.priority`) already existed; clients lacked retry hint |
| 7 | Accounting on errors | **shipped** | Success had Metrics; errors were bare `{"error"}` |
| 8 | Health without false empty | **shipped** | Empty done ≠ refusal; runner death ≠ semanticOk fail |
| 9 | Concurrent residency advertised | **shipped** | Ops docs existed; progressive probe needed `/api/version` |
| 10 | Autotune multi-model / propose | **missing** (Phase B) | Per-GGUF estimate/calibrate only; no co-residency planner |

---

## APIs

| Method | Path | Role | Why |
|--------|------|------|-----|
| `GET` | `/api/status` | Live queues + **`inference.config`** | Fleet pollers already hit status; config belongs next to load |
| `POST` | `/api/can-load` | Dry-run admit; **never** `GetRunner` | Calling GetRunner would queue real loads and defeat “probe” |
| `GET` | `/api/metrics` | Prometheus text | Standard scrape; JSON counters stay on status |
| `GET` | `/api/version` | `zerollama.capabilities` flags | Same progressive-probe pattern as `mlx_qos` |

Busy schedule errors (`ErrMaxQueue`, Darwin Metal contention) and proxied runtime `503` set **`Retry-After: 2`** and JSON `retry_after`.

**Why constant 2s:** Phase A prefers a stable client contract over inventing queue-depth→delay math. Operators can still back off exponentially on top.

### Capability flags (`GET /api/version` → `zerollama.capabilities`)

| Flag | Meaning |
|------|---------|
| `runtime_config` | status exposes config knobs |
| `can_load` | `POST /api/can-load` exists |
| `metrics` | `GET /api/metrics` exists |
| `admission_retry_after` | busy 503s carry Retry-After |
| `error_timings` | error JSON may include durations / TTFT |
| `empty_gen_classify` | empty / host_unstable classification |
| `priority_queues` | documents existing qos/priority (not new queues) |
| `residency_policy` | same-model multi-copy false; NUM_PARALLEL = slots |
| `stable_multi_model_swap` / `pin_reserve` / `propose_sidecar` | **false** — honest gaps |

---

## can-load contract (read carefully)

```bash
curl -s localhost:8080/api/can-load -d '{"model":"llama3.2:latest","options":{"num_ctx":8192}}'
```

| Field | Meaning | Why |
|-------|---------|-----|
| `confidence` | `exact` (runtime VRAM estimate) or `heuristic` (ggml count/group) | Clients must not treat heuristic as hard VRAM truth |
| `can_load` | Scheduler would accept under known checks | Includes “admit by evicting someone” |
| `already_loaded` | **This** model/GGUF is warm | Not “some” model is loaded (Python is single-resident) |
| `needs_eviction` | Load would unload/swap another resident | Thrash-sensitive graphs must require `!needs_eviction` |
| `busy` | Queue full or loads paused | Still HTTP 200 — dry-run is structured, not 503 |

**Fail closed:** missing GGUF or unavailable VRAM estimate → `can_load: false` + notes.  
**Why fail closed:** fail-open soft-admit lied to Decide and caused thrash; capacity probes prefer “unknown/no” over false yes.

**Why never GetRunner:** ggml “requireFull” fit probes spawn real runner subprocesses; that is not a dry-run and can itself hit `ErrMaxQueue`.

---

## Empty generation vs host unstable

| Case | Signal | Why |
|------|--------|-----|
| Thinking-only (empty `response`) | leave alone | Thinking models are valid; not infra failure |
| Tiny `num_predict` / empty done, `eval_count==0` | `done_reason=empty_generation` | Distinguishes short predict from refusal |
| Runner/process exit | error NDJSON `cause=host_unstable` | Clients set `kindExt.hostUnstable` without blaming semanticOk |

Matchers for `host_unstable` are **tight** (`runner exited`, `signal: killed`, …). Broad substrings like `"llama server"` / `"subprocess"` were rejected — they false-positive on config errors.

---

## Findings / learnings (Phase A)

**Why write these down:** audit regressions (false `already_loaded`, fail-open admit, ggml-only metrics) were easy to reintroduce; this list is the contract for future host capacity work.

1. **Dual inference stacks need dual dry-run confidence.** Runtime has real VRAM math (`ProbeVramEstimate`); ggml does not expose a subprocess-free layout probe. Shipping one `confidence` field beats inventing a fake “exact” for ggml.
2. **`already_loaded := llama_server` is a lie on single-resident runtimes.** Any warm GGUF made every can-load look warm. Match `/health.model_swap.loaded_gguf` (or ggml `/api/ps` name) to the requested path.
3. **Fail-open admit when estimate fails is worse than fail-closed.** Soft-admit looked helpful in tests and broke Decide trust. Prefer `can_load: false` + `notes`.
4. **`can_load` and `needs_eviction` are orthogonal.** Admit-with-swap is valid for opportunistic work; hire maps / meta-autotune must gate on `!needs_eviction`.
5. **Metrics that only instrument the ggml path undercount on Linux.** Runtime proxy is the common path; counters must update on proxy success/fail too.
6. **Priority queues were already shipped.** Wishlist item 6 was half Retry-After; rebuilding queues would duplicate `qos_class` / `options.priority`. Advertise, don’t fork.
7. **GPU discovery on every `/api/status` is expensive for fleet pollers.** Effective `MAX_LOADED` when env is `0` needs a short-TTL GPU count cache.
8. **Empty gen ≠ semantic refusal.** Agents that map empty text → “model said no” poison Inventory. Classify at the host and bridge via `kindExt.hostUnstable` only for infra causes.
9. **Honest capability flags beat silent gaps.** `stable_multi_model_swap` / `pin_reserve` / `propose_sidecar` stay `false` so progressive probes do not invent Phase B.

---

## Until Phase B

- Serialize interviews under swap thrash
- Prefer self-propose autotune (`--meta-model` = target)
- Treat concurrency sweep as source of truth for budgets
- Soft-pin with `keep_alive:-1` + `options.zerollama.fulfillment=complete`
- On runner deaths: mark counterpart `kindExt.hostUnstable: true`

## Phase B (deferred)

| Item | Direction | Why deferred |
|------|-----------|--------------|
| Stable swap | Sticky warm-set / hysteresis; reduce cross-stack broker thrash | Scheduler redesign |
| Pin/reserve | Session lease API for N model keys | Stronger than request `fulfillment` |
| Propose | `/api/propose-load` using can-load + warm set | Needs stable multi-model residency first |

---

## Code map

| Area | Path | Why this file |
|------|------|---------------|
| Status config | `server/inference_config.go`, `server/inference_status.go` | Readable knobs next to live queues |
| can-load | `server/can_load.go` | Public dry-run; never GetRunner |
| Metrics | `server/metrics.go` | Prometheus text without new dep |
| Empty / error classify | `server/empty_gen.go`, `server/stream_keepalive.go` | Separate empty-gen from host_unstable |
| Version caps | `server/mlx_qos.go` (`zerollamaVersionCapabilities`) | Progressive probe + honest Phase B false |
| Runtime proxy metrics / Retry-After | `server/runtime_proxy.go`, `server/runtime_chat_forward.go` | Linux path must count; busy needs Retry-After |
| API types | `api/types.go` (`InferenceConfigStatus`, `CanLoad*`) | Shared contract for clients |
| OpenAPI | `server/openapi/openapi.yaml` | Live `/docs` for Orient/Decide |
| Tests | `server/wishlist_phase_a_test.go` | Fail-closed / already_loaded / Retry-After |
