# Agent QoS, project tracking, and cross-backend safety

**Audience:** operators running multiple agent harnesses on one zerollama node; contributors wiring clients on MLX, GGUF/CUDA, llama-server, or vanilla Ollama.

**Related:** [mlx-agent-prompts.md](./mlx-agent-prompts.md), [scheduling-vram-policy.md](./scheduling-vram-policy.md), [mlx-routing-policy.md](./mlx-routing-policy.md), [fleet-scheduling.md](./fleet-scheduling.md).

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

| Field | Purpose | Why |
|-------|---------|-----|
| `qos_class` | `interactive` \| `auxiliary` \| `background` | Prevents background batch jobs from clobbering live agent KV |
| `fulfillment` | `complete` \| `benchmark` | Request-scoped no-degradation contract (see below) |
| `session_group` | Harness namespace (`hermes-agent`, `ruby-trivia`) | Collapses ephemeral aux/bg keys onto shared branches where safe |
| `session_parent` | Parent thread `prompt_cache_key` | Spawns defer behind main thread; observability for subagents |
| `project_id` | Stable client id | Fleet / `zerollama ps` — which app owns GPU time |
| `project_name` | Human label (Discord channel, audit phase) | Operator grep without parsing cache keys |
| `cache_scope` | `thread` \| `shared` \| `auto` | MLX trie vs shared branch policy |

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

**API:** `GET /api/ps` → `models[].zerollama.sessions[]` with `session_key`, `session_class`, `project_id`, `project_name`, `inflight`, `hot_until`.

**Why gate stores project fields:** Sessions are registered at `reserveScheduleQoS` claim time — the same moment we fix the TOCTOU race — so ps reflects in-flight work, not post-hoc log parsing.

---

## Session gate and TOCTOU fix (Jul 2026)

**Why the race mattered:** Two handlers could both pass `waitForSlot` / defer checks, then both call `GetRunner` before either registered on the gate. On MLX this caused concurrent `switchToPath` and subprocess death mid-decode.

**Fix:** `reserveScheduleQoS` **claims the gate slot immediately** after policy allows scheduling, before waiting for a runner subprocess. Handlers `defer releaseQoS()` on all exit paths.

```text
Request → reserveScheduleQoS (claim gate) → GetRunner (load/wait) → infer → releaseQoS
```

Code: `server/schedule_qos.go`, `server/routes.go` (`scheduleRunner`).

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

| Area | Path |
|------|------|
| Inference path detection | `server/inference_path.go` |
| QoS reserve + TOCTOU fix | `server/schedule_qos.go` |
| Session gate | `server/mlx_sidecar_gate.go` |
| QoS parse + version API | `server/mlx_qos.go` |
| Agent cache / eliza gating | `server/agent_prompt_cache.go` |
| `/api/ps` session snapshot | `server/sched.go` |
| CLI ps columns | `cmd/cmd.go` |
| Tests | `server/inference_path_test.go`, `server/agent_cache_runtime_test.go`, `server/process_sessions_test.go` |

---

## Non-goals

- Replacing per-node schedulers with a global multi-GPU optimizer ([fleet-scheduling.md](./fleet-scheduling.md) picks **which node**, not how MLX trie works).
- Requiring every Ollama client to send `options.zerollama` — Tier 1 remains valid forever on compatible servers.
- Hardcoding Hermes-only behavior in the Go server — harness prefixes are conventions; `project_id` is the stable contract.
