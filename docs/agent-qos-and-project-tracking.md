# Agent QoS, project tracking, and cross-backend safety

**Audience:** operators running multiple agent harnesses on one zerollama node; contributors wiring clients on MLX, GGUF/CUDA, llama-server, or vanilla Ollama.

**Related:** [mlx-agent-prompts.md](./mlx-agent-prompts.md), [openai-harness-qos-wire-shapes.md](./openai-harness-qos-wire-shapes.md) (M15c findings), [hermes-zerollama-gap.md](./hermes-zerollama-gap.md), [scheduling-vram-policy.md](./scheduling-vram-policy.md), [mlx-routing-policy.md](./mlx-routing-policy.md), [fleet-scheduling.md](./fleet-scheduling.md).

---

## Why this doc exists

Jul 2026 production logs showed three separate problems that looked like one “MLX crash”:

1. **Two concurrent agent streams** slipped through QoS defer checks within milliseconds — a TOCTOU race between “may I run?” and “I hold the gate slot.”
2. **Operators could not tell** which Discord thread, audit pipeline, or sidecar owned a loaded model — `zerollama ps` showed VRAM and model name only.
3. **Optimizations meant for zerollama** (eliza metadata, 30m keep-alive, prefix-mm-cache) were safe on MLX but could confuse or slow unrelated production stacks (vanilla Ollama, vLLM, llama.cpp proxies) if sent unconditionally.

This doc describes the **generic harness contract** (`options.zerollama`), **fleet observability** (`project_id` / `project_name`), and **backend-aware branching** so Mac MLX, CUDA GGUF, and third-party servers each get what helps them — nothing else.

---

## Progressive client ladder (detect before optimize)

**Why a ladder:** Vanilla Ollama ignores unknown `options` fields, but some OpenAI-compatible proxies reject them. Clients should probe once per process, then send only what the server advertises.

```text
GET /api/version
  distribution != "zerollama"  → Tier 1: format, keep_alive, num_ctx only
  distribution == "zerollama"  → Tier 2: + prompt_cache_key + options.zerollama
  zerollama.capabilities.runner_paths → which backends this node may use
```

**Example detection (any language):**

```bash
curl -s http://127.0.0.1:11434/api/version | jq '{
  distribution,
  mlx_qos: .zerollama.capabilities.mlx_qos,
  session_gate: .zerollama.capabilities.session_qos_gate,
  runner_paths: .zerollama.capabilities.runner_paths
}'
```

| Tier | When | Send |
|------|------|------|
| 0 | Server unreachable | Retries only |
| 1 | Vanilla Ollama | `format`, `keep_alive`, `num_ctx`, `timeout` |
| 2 | Zerollama | + stable `prompt_cache_key` + `options.zerollama` |

**Shipped clients (Jul 2026):**

| Client | `project_id` | Detection |
|--------|--------------|-----------|
| Hermes (`hermes-lean`) | `hermes-lean` | `GET /api/version` → `is_zerollama` |
| ruby-trivia | `ruby-trivia` | `probeInferenceRuntime()` |
| simpleagent | `simpleagent` (configurable) | health + version probe |

---

## Harness QoS contract (`options.zerollama`)

**Why not MLX-only field names:** CUDA and llama-server paths share the same session gate and `/api/ps` metadata. Names like `mlx_session_class` remain as legacy aliases but new clients should use `options.zerollama`.

**OpenAI `/v1/chat/completions`:** Prefer nesting under `options.zerollama` or `extra_body.zerollama` / `extra_body.options.zerollama`. Flat top-level (or flat `extra_body`) `qos_class` / `project_id` / `project_name` / … are also accepted — the OpenAI Python SDK promotes `extra_body` onto the HTTP root, and the server folds those keys into `options.zerollama` (including the Python runtime proxy path). Precedence: nested `options.zerollama` wins over top-level `zerollama`, which wins over flat aliases. Invented top-level keys still 400 (trap 77).

**Why flat aliases exist (M15c):** Hermes aux tasks used OpenAI `extra_body` and hit trap 77 while nested `/api/chat` worked for the same QoS contract. Full wire-shape findings: [openai-harness-qos-wire-shapes.md](./openai-harness-qos-wire-shapes.md).

| Field | Purpose | Why |
|-------|---------|-----|
| `qos_class` | `interactive` \| `auxiliary` \| `background` | Prevents background batch jobs from clobbering live agent KV |
| `fulfillment` | `complete` \| `benchmark` | Request-scoped no-degradation contract (see below) |
| `session_group` | Harness namespace (`hermes-agent`, `ruby-trivia`) | Collapses ephemeral aux/bg keys onto shared branches where safe; Radix prefers same-group donors on ties |
| `session_parent` | Parent thread `prompt_cache_key` | `wait_parent` while parent key is hot (multiplex-aware); Radix prefers that donor on equal-length ties |
| `project_id` | Stable client id | Fleet / `zerollama ps` — which app owns GPU time |
| `project_name` | Human label (Discord channel, audit phase) | Operator grep without parsing cache keys |
| `cache_scope` | `thread` \| `shared` \| `auto` | MLX trie vs shared branch policy |
| `cache_level` | `auto` \| `gpu` \| `dram` \| `disk` | KV tier (`auto` = heuristics; `gpu`/`dram` = no disk; `disk` = allow blobs when policy permits) |
| `cache_reset` | `true` | Force miss under the **same** `prompt_cache_key` this request |

### `preempted_reason` (M15e)

**Why:** Hermes needs to know whether a failed/canceled call waited behind higher-priority QoS versus a generic cancel or queue-full busy — so it can retry faster or escalate.

When a request is **aborted while deferred** (client cancel or per-call timeout during `waitForSlot` / global defer / fulfillment wait), error bodies may include:

```json
{"error":"request canceled","preempted_reason":"lower_wait_interactive"}
```

Typical `preempted_reason` values: `lower_wait_interactive`, `wait_parent`, `fulfillment_exclusive`, `interactive_preempt_cooldown`, and other `mlxDeferPolicy` / fulfillment reason strings.

**Not mid-stream hard preemption for ggml/Python:** an in-flight background generation on those paths is not hard-killed when interactive admits; wait-abort fields only explain *wait abort* / busy paths.

**M15f (MLX soft mid-stream):** when interactive admits and a lower-class session is **inflight** on the same model key, the gate cancels that request. The victim’s terminal stream chunk is `done_reason: "preempted"` + `preempted_reason` (OpenAI `finish_reason: "preempted"`). Hermes must treat that as retryable incomplete output — not a finished tool call.

**Also:** per-call `timeout` → HTTP **504** `{"error":"request timeout","timeout_seconds":N}` (may also carry `preempted_reason` if the deadline fired while deferred). See [hermes-gap-closure-findings.md](./hermes-gap-closure-findings.md).

### Fulfillment modes (`options.zerollama.fulfillment`)

**Why:** Normal QoS defers lower-priority work but does not give a SQL-transaction-like guarantee that *this* request finishes without degradation (eviction, peer VRAM pressure, concurrent interactive noise). Fulfillment is **request-scoped** (`begin` → infer → `release`) — not a multi-minute fleet lease.

| Mode | Aliases | Behavior |
|------|---------|----------|
| **`complete`** | `guarantee`, `reliable` | Elevate to interactive; wait for a clear slot on this model; block aux/bg (and same-model peers) while held; floor `keep_alive` to 30m when unset; protect this model from eviction. Other interactive sessions on **other** models may still run. |
| **`benchmark`** | `bench`, `speed`, `exclusive` | Everything `complete` does, plus **exclusive GPU**: wait until no competing inflight/hot interactive sessions; unload peer runners before load; block **all** other traffic (including other interactive) until release; floor `keep_alive` to 2h when unset. |

**Example (bench suite):**

```json
{
  "model": "llama3.2:3b",
  "prompt": "Hello",
  "options": {
    "zerollama": {
      "fulfillment": "benchmark",
      "project_id": "zerollama-bench",
      "project_name": "tok/s gate"
    }
  }
}
```

**Example (critical agent turn — finish without eviction):**

```json
{
  "model": "gemma4",
  "messages": [{"role": "user", "content": "..."}],
  "options": {
    "zerollama": {
      "fulfillment": "complete",
      "qos_class": "interactive",
      "project_id": "hermes-lean"
    }
  }
}
```

Probe: `GET /api/version` → `zerollama.capabilities.fulfillment` and `zerollama.qos.fulfillment`.

Full field table and examples: [mlx-agent-prompts.md § Harness QoS API](./mlx-agent-prompts.md#harness-qos-api-generic-jul-2026).

---

## Project tracking (`zerollama ps`)

**Why:** One Mac or 5080 often runs Hermes Discord + ruby-trivia audit + manual `curl` at once. Without project labels, “who evicted my model?” requires log archaeology.

**CLI columns (when session metadata present):**

```text
NAME      PROJECT                    SESSION                         SIZE  ...
gemma4    hermes-lean/discord:dm:1   hermes:agent:main:discord:dm:1  ...
qwen3.6   ruby-trivia/audit          ruby-trivia:bg:audit              ...
```

**API:** `GET /api/ps` → `models[].zerollama.sessions[]` with `session_key`, `session_class`, `session_group`, `session_parent`, `project_id`, `project_name`, `cache_scope`, `cache_level`, `fulfillment`, `inflight`, `hot_until`.

**Why gate stores project fields:** Sessions are registered at `reserveScheduleQoS` claim time — the same moment we fix the TOCTOU race — so ps reflects in-flight work, not post-hoc log parsing. Multiplex key-hot entries appear even when they are not the fairness “primary.”

---

## Session gate and TOCTOU fix (Jul 2026)

**Why the race mattered:** Two handlers could both pass `waitForSlot` / defer checks, then both call `GetRunner` before either registered on the gate. On MLX this caused concurrent `switchToPath` and subprocess death mid-decode.

**Fix:** `reserveScheduleQoS` **claims the gate slot immediately** after policy allows scheduling, before waiting for a runner subprocess. Handlers `defer releaseQoS()` on all exit paths.

```text
Request → reserveScheduleQoS (claim gate) → GetRunner (load/wait) → infer → releaseQoS
```

Code: `server/schedule_qos.go`, `server/routes.go` (`scheduleRunner`).

---

## Session → cache great loop (Jul 2026)

**Why a “loop” not more knobs:** Agent harnesses (many threads on one connection) need to *declare* intent, have the server *schedule* on it, *retain* KV accordingly, and *hit* cache when safe. Extra client TTLs and `cold:` key prefixes were rejected — they fight L3/`keep_alive` and fragment the trie.

| Axis | Field | Server behavior | Why this shape |
|------|-------|-----------------|----------------|
| Identity | `prompt_cache_key` | Live extend / L3 slot pin / ps label | Same key = same conversation owner |
| Relation | `session_parent`, `session_group` | `wait_parent`; Radix prefer on equal-length ties | Spawns must not clobber parent KV; donors stay hash-verified |
| Validity | `cache_reset: true` | Force miss **under the same key** this turn | Soft “new branch” without inventing a second key namespace |
| Tier | `cache_level` | `auto`\|`gpu`\|`dram`\|`disk` | Retention hint; `auto` = no surprise vs heuristics |
| Urgency | existing `qos_class` / `fulfillment` | Unchanged | Defer vs exclusive are orthogonal to cache identity |

```text
declare (parent / reset / level)
    → schedule (multiplex-aware wait_parent + key hot-map)
    → retain (cache_level → disk persist policy)
    → hit (MLX trie / L3 resume / Radix prefer parent|group)
```

### Multiplex-aware `wait_parent`

**Why the primary slot alone failed:** With many agents on one runner, the “primary” holder is whoever claimed last. A child’s `session_parent` could still be inflight/cooldown while primary pointed at an unrelated thread — `wait_parent` never fired.

**Fix:** Per-model **key hot-map** (cap 64). Parent defer checks the map, not only the primary slot. Primary fairness display is **derived from** the map after each begin/end.

**Why normalize parent keys:** MLX may rewrite aux/bg keys to `aux:{model}` / `bg:{model}[:group]`. Children usually send the **raw** parent id. `parentKeyCandidates` expands `session_parent` through `injectMLXSessionKey` so `wait_parent` still matches.

### `cache_reset` contract

**Why under the same key:** Harnesses already have a stable interactive key; inventing `cold:…` forces clients to manage two namespaces and breaks ps continuity.

| Path | Behavior | Why |
|------|----------|-----|
| MLX | Skip live extend + trie hit for this turn | Enough to force re-prefill; trie branches may remain for later content matches |
| GGUF L3 | Deny `cache_prompt` / resume; bump decode-graph epoch; `seed_seq_pos(0)` or in-process `seq_rm` | Soft deny alone left residual slot pos / graphs that could soft-resume |
| Radix | **Skipped** when `cache_reset` | Cross-slot seed would undo “no KV reuse this turn” |

**Honesty:** `gpu` ≈ `dram` today (both forbid disk persist). Advertise both so clients can pin intent before spill exists.

### `cache_level`

| Value | Effect | Why |
|-------|--------|-----|
| `auto` (default) | Existing heuristics | No surprise for Tier-1 clients |
| `gpu` / `dram` | Forbid disk blob persist | Keep warm path off SSD |
| `disk` / `ssd` | Allow disk when policy permits | Still cannot override draft-spec hard deny |

---

## Findings / learnings (session → cache)

**Why write these down:** the first multiplex/`cache_*` ship was a coherent API with soft edges. An audit found concurrency and reset holes that would reappear as “QoS is flaky” / “reset didn’t clear.”

1. **Primary-slot `inflight++` + overwrite leaks fairness.** `begin` used to bump primary and set `sessionKey`; `end` only decremented when keys still matched. Concurrent A then B left A’s end as a no-op → stuck hot primary. **Why fix:** derive primary from keyHot; end only mutates the key entry.
2. **`waitForSlot` then separate `begin` is still TOCTOU-ish** for claim ordering, but keyHot accounting removes the leak that made races sticky. Full atomic waitAndBegin remains optional polish.
3. **`cache_reset` that only denies resume is soft on GGUF.** Slot `seq_pos` and CUDA graphs can still look “warm.” **Why hard invalidate:** bump epoch + zero seq_pos / `seq_rm` so the next completion cannot pretend continuity.
4. **Radix after deny undoes reset.** Admission historically seeded when full-prompt resume was denied (SWA window case). Reset must short-circuit Radix. **Why:** “no reuse this turn” means no donor seed either.
5. **`session_parent` must resolve to the gate key, not only the client string.** Aux rewrite breaks exact match. **Why candidates:** children should not need to know inject rules.
6. **`gpu` vs `dram` identical is intentional honesty**, not a bug. **Why document:** capability text must say so or harnesses invent false VRAM pinning.
7. **Zero-value gate maps panic under incomplete test schedulers.** Defensive `slots`/`keyHot` init in `beginLocked`. **Why:** `assignment to entry in nil map` on debug routes was worse than a soft no-op.
8. **Harness updates stay out of scope for the server loop.** Server advertises; clients probe `/api/version`. **Why:** progressive ladder already ships in Hermes / ruby-trivia / simpleagent.

---

## Backend-aware branching (don't hurt other systems)

**Why one gate name (`mlxGate`) is misleading:** The gate serializes **keyed interactive sessions on every text runner**, not MLX only. Branching inside the gate keeps unrelated production traffic safe.

| Backend | Session gate | Gate key rewriting | eliza / prefixHash injection | MLX prefill trie |
|---------|--------------|-------------------|------------------------------|------------------|
| **MLX** safetensors | Yes | Aux/bg → shared branches | When harness hints present | Yes |
| **GGUF** ggml / llama-server | Yes (keyed text only) | **Preserves client key** | When harness hints present | No |
| **GGUF** unkeyed | **No-op** | N/A | No | No |
| **Embedding-only** | Skipped | N/A | No | No |
| **Vanilla Ollama** (other host) | N/A | Client sends Tier 1 only | Client must not send Tier 2 | N/A |

### Rules (code: `server/inference_path.go`)

1. **`modelSupportsSessionQoS`** — MLX always; GGUF with `ModelPath` and completion capability (manifest-based, not GGUF file read at schedule time). Embedding-only models skip the gate.

2. **`gateSessionKey`** — MLX may rewrite aux/bg keys onto `aux:{model}` / `bg:{model}` branches for trie sharing. **GGUF keeps the client `prompt_cache_key`** so llama-server L3 and ps labels stay aligned.

3. **Unkeyed GGUF** — `reserveScheduleQoS` returns immediately (no wait behind MLX interactive). **Why:** CUDA batch endpoints without session keys must not stall 90s behind a Mac agent cooldown they don't participate in.

4. **`agentSessionMetadataEnabled`** — Server adds `eliza` / `prefixHash` only when:
   - `options.zerollama` is present, or
   - client already sent `eliza`, or
   - `prompt_cache_key` uses a known harness prefix (`hermes:`, `ruby-trivia:`, `simpleagent:`, `conv:`).

   Plain keys like `openwebui:user:42` get normalized `prompt_cache_key` only — no extra metadata. **Why:** unknown fields on third-party servers cause 400s or silent mis-routing.

5. **Client-side mirror (Hermes custom profile):** `keep_alive`, `enable_prefix_mm_cache`, and `eliza` blocks are sent **only when `is_zerollama`** is true after `/api/version` probe.

### Runner path advertisement

`GET /api/version` → `zerollama.capabilities.runner_paths`:

- `mlx`, `gguf_ggml` — always listed on full builds
- `gguf_llama_server` — when `--llama-server-backend`, edge mode, or unlinked ggml
- `runtime` — when Python sidecar URL is configured

**Why expose paths:** Clients and fleet routers can choose Tier 2 hints only when the target node actually runs a backend that consumes them.

---

## What stays MLX-specific

These optimizations **must not** be assumed on CUDA/Arc GGUF:

- `capMLXScheduleOptions`, `mlxKeepAliveFloor`, `PromptTokens` passthrough
- Prompt chain / trie / `tryExtendLiveSession` / rotating-KV restore
- MLX subprocess crash logging (`mlx runner subprocess exited`)

GGUF paths benefit from **L3 `prompt_cache_key`** (llama-server `cache_n`) and **QoS defer** when clients send Tier 2 metadata — not from MLX trie logic.

Doc: [mlx-agent-prompts.md](./mlx-agent-prompts.md), [gpu-profiles-l3.md](./gpu-profiles-l3.md).

---

## Operator checklist

1. **Rebuild and restart** zerollama after pulling these changes.
2. **Restart agents** (Hermes gateway, ruby-trivia, simpleagent) so they probe `/api/version` and send Tier 2 options.
3. **Verify ps:**
   ```bash
   ./zerollama ps
   curl -s http://127.0.0.1:11434/api/ps | jq '.models[].zerollama'
   ```
4. **Confirm unrelated stacks unchanged:** point a vanilla Ollama client at the same host — it should not receive `eliza` or `enable_prefix_mm_cache` from zerollama server-side unless it sends harness keys.
5. **Watch logs** after concurrent agent + batch load:
   - `mlx session deferred` / `mlx global defer` — expected for intentional QoS
   - `mlx runner subprocess exited` — MLX crash with exit code + stderr (Jul 2026)

---

## Code map

| Area | Path | Why this file |
|------|------|---------------|
| Inference path detection | `server/inference_path.go` | Backend-aware gate / eliza branching |
| QoS reserve + TOCTOU claim | `server/schedule_qos.go` | Claim before GetRunner |
| Session gate + key hot-map | `server/mlx_sidecar_gate.go` | Multiplex `wait_parent`; primary from keyHot |
| QoS parse + version caps | `server/mlx_qos.go` | `cache_reset` / `cache_level` + progressive probe; advertises flat OpenAI aliases |
| OpenAI bind + fold (M15c) | `openai/chat_extras.go`, `openai/chat_unknown_fields.go` | SDK-flattened `extra_body` → `options.zerollama`; trap 77 allowlist |
| Runtime v1 options forward | `server/runtime_manifest.go` `runtimeV1ProxyOptions` | Sidecar must see folded zerollama (not only Go routing) |
| Agent cache / eliza gating | `server/agent_prompt_cache.go` | Tier-2 metadata only when harness hints |
| MLX trie reset | `x/mlxrunner/cache.go` | `cacheReset` skips live extend + trie hit |
| GGUF admission + hard reset | `runtime/runtime/engine.py` | Deny resume; skip Radix; epoch + seq clear |
| Cache field extract | `runtime/runtime/cache_bridge.py` | Shared Python helpers for level/reset/parent |
| Radix prefer parent/group | `runtime/runtime/kv/prefix_block_pool.py` | Tie-break only; still hash-verified |
| `/api/ps` session snapshot | `server/sched.go` | Fleet visibility |
| CLI ps columns | `cmd/cmd.go` | Operator PROJECT/SESSION |
| Contract tests | `server/session_cache_contract_test.go`, `runtime/tests/test_session_cache_contract.py`, `runtime/tests/test_radix_engine_guard.py` | Inflight leak, parent normalize, reset↛Radix |

---

## Non-goals

- Replacing per-node schedulers with a global multi-GPU optimizer ([fleet-scheduling.md](./fleet-scheduling.md) picks **which node**, not how MLX trie works).
- Requiring every Ollama client to send `options.zerollama` — Tier 1 remains valid forever on compatible servers.
- Hardcoding Hermes-only behavior in the Go server — harness prefixes are conventions; `project_id` is the stable contract.
- Client harness rewrites (Hermes / ruby-trivia / simpleagent) as part of the server “great loop” — those stay in their repos; server advertises capabilities.
- Distinct `gpu` vs `dram` spill tiers until a real VRAM↔host spill path exists — both forbid disk today on purpose.
