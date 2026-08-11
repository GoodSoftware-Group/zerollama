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
5. **Can I keep a model warm across turns without thrashing the other stack?** — session pin + thrash dampen, without lying that two Python GGUFs stay resident.

**Why Phase A first:** capacity APIs unblocked Decide without pretending multi-resident swap was solved.  
**Why Phase B next:** dry-run alone did not stop Decide from thrashing ggml↔runtime or stacking conflicting interview models; pin/propose give *honest* leases and plans.  
**Phase B (shipped Jul 2026):** pin/propose + thrash dampen + honest single-resident contract + audit hardening (broker respects pins; fail-closed pin+runtime). **`stable_multi_model_swap` stays false** until multi-GGUF Python.

**Why not a public unload API:** Phase 8 already chose automatic eviction over operator unload endpoints. can-load / propose report `needs_eviction`; they do not add `/api/unload`.

---

## Status

| # | Item | Status | Why this status |
|---|------|--------|-----------------|
| 1 | Stable multi-model swap | **partial** (dampen only) | B0 skips unload when same GGUF warm **and ggml empty**; B1 ggml hysteresis; Python still one GGUF → flag stays **false** |
| 2 | Readable runtime config | **shipped** | Env-only was guesswork for fleet clients |
| 3 | Capacity dry-run | **shipped** | Loopback `/internal/vram-estimate` was not a public product API |
| 4 | Queue & metrics | **shipped** | Status had queues; scrapers need Prometheus text |
| 5 | Pin / reserve N models | **shipped** | TTL lease; broker respects pins; global key budget; runtime GGUF soft-pin (503 on conflict); fail closed on multi-runtime GGUF |
| 6 | Admission 503 + Retry-After | **shipped** (Retry-After half) | Priority queues already existed; clients lacked retry hint |
| 7 | Accounting on errors | **shipped** | Success had Metrics; errors were bare `{"error"}` |
| 8 | Health without false empty | **shipped** | Empty done ≠ refusal; runner death ≠ semanticOk fail |
| 9 | Concurrent residency advertised | **shipped** | Ops docs existed; progressive probe needed `/api/version` |
| 10 | Autotune multi-model / propose | **shipped** (plan API) | `POST /api/propose-load`; not a calibrator |

---

## APIs

| Method | Path | Role | Why |
|--------|------|------|-----|
| `GET` | `/api/status` | Live queues + **`inference.config`** + **`inference.pins`** | Fleet pollers already hit status |
| `POST` | `/api/can-load` | Dry-run admit; **never** `GetRunner` | Probe must not enqueue loads |
| `POST` | `/api/propose-load` | Batch can-load + co-residency plan | Decide needs multi-model honesty |
| `POST` | `/api/pin` | Session eviction lease (no load) | Stronger than request `fulfillment` |
| `DELETE` | `/api/pin/:id` | Release pin early | Explicit lease end |
| `GET` | `/api/metrics` | Prometheus text | Standard scrape |
| `GET` | `/api/version` | `zerollama.capabilities` flags | Progressive probe |

Busy schedule errors (`ErrMaxQueue`, Darwin Metal contention), proxied runtime `503`, and **pin/runtime VRAM conflicts** set **`Retry-After: 2`** and JSON `retry_after`.

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
| `pin_reserve` | **true** — `/api/pin` |
| `propose_sidecar` | **true** — `/api/propose-load` |
| `stable_multi_model_swap` | **false** — Python single-GGUF; B0/B1 only dampen thrash |

**Why keep `stable_multi_model_swap=false` after Phase B:** thrash dampen and soft-pin are *mitigations*, not multi-resident swap. Lying here would re-break Orient hire maps.

---

## Honest client contract (Phase B)

| Backend | Pin N models | Propose `fits_without_eviction` for N models |
|---------|--------------|-----------------------------------------------|
| ggml multi-runner | Yes — eviction protect up to `ZEROLLAMA_PIN_MAX` / `MAX_LOADED` (global distinct-key budget) | Heuristic; thrash if over capacity |
| runtime (Python) | **At most 1** distinct GGUF host-wide; second → `400`; conflicting chat → `503` `runtime_pin_gguf` | Batch with **>1 runtime GGUF** → `co_resident=false`, `serialize_required=true` |
| mixed | Pin protects ggml keys; residual pinned ggml blocks runtime resume (`503` `runtime_pin_ggml`) | Same serialize flag when any 2+ runtime paths |

Clients that need two interview models on a runtime host must **serialize** (or wait for Phase B+ multi-resident).

Env: `ZEROLLAMA_WARM_HYSTERESIS` (default 3m), `ZEROLLAMA_PIN_MAX`, `ZEROLLAMA_PIN_TTL` (default 30m).

### Pin + VRAM broker semantics (audit)

| Situation | Host behavior | Why |
|-----------|---------------|-----|
| Runtime chat, same GGUF already warm, **ggml empty** | Skip `UnloadAllRunners` (B0) | Avoid needless cross-stack thrash every turn |
| Runtime chat, same GGUF warm, **ggml still loaded** | Unload unprotected ggml; do **not** skip | Leftover ggml + llama-server = OOM risk |
| Runtime chat after unload, **pinned/in-use ggml remains** | `503` `cause=runtime_pin_ggml` **before** `ResumeInference` | Pin means keep warm; silently stacking stacks is worse than a busy retry |
| Active pin holds GGUF A; request wants GGUF B | `503` `cause=runtime_pin_gguf` | Go soft-pin of Python residency; Python still one GGUF |
| Training / exclusive `fulfillment=benchmark` | `UnloadAllRunnersForced` (pins ignored) | Training OOM and clean benches must reclaim the GPU |
| Two pin leases, two distinct runtime GGUFs | Second `POST /api/pin` → `400` | Host-wide single-resident contract, not per-lease only |

**What pin is not:** it does not load models, does not create multi-GGUF Python residency, and does not stop an in-flight HTTP stream (`refCount>0` still defers unload — stream safety predates Phase B).

### Prefix-cache pin (M15e) — distinct from model pin

```bash
curl -s localhost:11434/api/cache/pin -d '{"prompt_cache_key":"hermes:thread:42","ttl_seconds":3600}'
curl -s -X DELETE localhost:11434/api/cache/pin/<pin_id>
```

| | `/api/pin` | `/api/cache/pin` |
|--|------------|------------------|
| Key | model name(s) | `prompt_cache_key` |
| Effect | Block model eviction / soft UnloadAllRunners | Skip MLX trie eviction + bump L3 disk TTL |
| Idle llama-server slots | N/A | **Not** retained while no request in flight |

**Why separate:** Hermes wants prefix KV to survive idle gaps without claiming exclusive model residency (and without fail-closed multi-GGUF pin rules). See [hermes-gap-closure-findings.md](./hermes-gap-closure-findings.md).

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
| `device_count` / `tensor_parallel` / `split_mode` / `tensor_split` / `main_gpu` | Host topology from runtime estimate (or ggml `device_count` heuristic) | Hermes multi-GPU awareness without `/health` scrape; **not** a TP planner |

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

## Findings / learnings (Phase A + B)

**Why write these down:** audit regressions (false `already_loaded`, fail-open admit, ggml-only metrics, lying multi-runtime pin, pin ignored by broker) were easy to reintroduce. Each bullet is a decision that looked “obvious” until it burned a fleet client.

### Phase A

1. **Dual inference stacks need dual dry-run confidence.** Runtime has real VRAM math (`ProbeVramEstimate`); ggml does not expose a subprocess-free layout probe. **Why:** one `can_load` shape for both stacks is fine; one confidence value is not.
2. **`already_loaded := llama_server` is a lie on single-resident runtimes.** Match `/health.model_swap.loaded_gguf` to the requested path. **Why:** “something is loaded” ≠ “your model is warm.”
3. **Fail-open admit when estimate fails is worse than fail-closed.** **Why:** Decide treated soft-yes as budget truth and oversubscribed VRAM.
4. **`can_load` and `needs_eviction` are orthogonal.** **Why:** admit-by-evict is valid for ops; thrash-sensitive graphs need both fields.
5. **Metrics that only instrument the ggml path undercount on Linux.** **Why:** production chat is often runtime-proxied.
6. **Priority queues were already shipped.** Advertise, don’t fork. **Why:** second queue implementation would diverge QoS.
7. **GPU discovery on every `/api/status` is expensive for fleet pollers.** **Why:** status is high-frequency; discovery is not.
8. **Empty gen ≠ semantic refusal.** **Why:** Orient `semanticOk` must not fail on infra empties.

### Phase B

9. **Honest capability flags beat silent gaps.** `stable_multi_model_swap` stays false even after B0/B1. **Why:** dampen ≠ multi-resident; hire maps that key on the flag must stay conservative.
10. **Pin of N ≠ N residents on runtime.** Fail closed when pinning two runtime GGUFs (per request **and** cross-lease). **Why:** Python holds one GGUF; a “successful” dual pin would be a lie.
11. **Propose must set `serialize_required`** when co-residency is impossible — never imply two runtime models stay warm. **Why:** Decide plans from this API; optimistic co-resident caused interview thrash.
12. **Orphan loopback serves steal the API** from `0.0.0.0` production; refuse second bind with a clear error; install CT binary at CT `/usr/bin/zerollama`, not the Proxmox host path. **Why:** operators debugged “stale API” for hours when an orphan `:8080` answered.
13. **Pins must survive the VRAM broker.** First Phase B ship kept pins only in `findRunnerToUnload`; `UnloadAllRunners` still wiped them → pin was a no-op on runtime-default hosts. **Why fix:** `/api/pin` without broker respect is worse than no pin (clients trust it).
14. **B0 skip-unload is unsafe if ggml still holds VRAM.** Require resident GGUF match **and** empty ggml scheduler. **Why:** skipping unload with leftover ggml is how dual-stack OOM happens.
15. **Per-request pin caps lie without a global budget.** Distinct model keys across leases ≤ `PIN_MAX` / `MAX_LOADED`. **Why:** N leases × M models each overflows residency bookkeeping.
16. **Pinned ggml + runtime must fail closed.** After pin-respecting unload, residual ggml → 503 **before** `ResumeInference`. **Why:** resuming Python on top of pinned ggml races OOM; busy retry is the honest signal.
17. **Runtime pin must block other GGUFs in Go.** Store `RuntimeGGUFs` on the lease; reject cross-lease second GGUF; 503 conflicting chat/generate. **Why:** pin of a runtime model that only protected a never-loaded ggml key did not stop ModelSwapGate from swapping.
18. **`UnloadAllRunners` vs `Forced` is intentional duality.** Soft path respects pin/fulfillment; training OOM and exclusive benchmark must reclaim GPU. **Why:** one unload API cannot mean both “keep my lease” and “clear the card.”
19. **In-use defer (`refCount>0`) still wins over Forced for stream teardown.** **Why:** killing an active NDJSON image/chat stream orphans the client; training may still wait — stream safety predates pins.

---

## Phase B+ (still deferred)

| Item | Direction | Why deferred |
|------|-----------|--------------|
| True stable swap | Multi-GGUF Python / multi llama-server | Scheduler redesign |
| Propagate pin into Python `ModelSwapGate` | Hard refuse swap inside runtime | Go soft-pin + 503 is enough until multi-GGUF |
| Co-residency autotune calibrator | Planner that picks ctx for N models | Needs multi-resident first |
| Plugin / Orient wiring | Other repo | Host APIs only here |
| Imagegen forced-evict of pins | Optional Forced path for MLX | Pin semantics currently win; operators release pins |

---

## Code map

| Area | Path | Why this file |
|------|------|---------------|
| Status config | `server/inference_config.go`, `server/inference_status.go` | Readable knobs + pins |
| can-load | `server/can_load.go` | Public dry-run; never GetRunner |
| propose | `server/propose.go` | Honest batch plan |
| pin | `server/pin.go`, `server/fulfillment.go` | TTL protect keys + RuntimeGGUFs |
| Broker dampen + pin conflicts | `server/vram/broker.go`, `server/runtime_broker.go` | B0 skip; 503 before resume |
| Hysteresis / pin-aware unload | `server/sched.go` | B1 victim pick; `UnloadAllRunners` vs `Forced` |
| Training reclaim | `server/training_policy.go`, `vram.PrepareForTraining` | Forced unload ignores pins |
| Single-serve | `cmd/cmd.go` `claimLoopbackGuards` | Hold 127.0.0.1/::1 when bind is wildcard so post-start steals get EADDRINUSE |
| Metrics | `server/metrics.go` | Prometheus text |
| Empty / error classify | `server/empty_gen.go` | Separate empty-gen from host_unstable |
| Version caps | `server/mlx_qos.go` | Progressive probe |
| OpenAPI | `server/openapi/openapi.yaml` | Live `/docs` |
| Tests | `server/wishlist_phase_a_test.go`, `wishlist_phase_b_test.go` | Pin-vs-broker + B0 + 503 causes |
