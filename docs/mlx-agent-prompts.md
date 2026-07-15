# MLX agent prompts — context, truncation, tokenize, and observability

**Audience:** operators running MLX safetensors models (e.g. Gemma4, Hermes agents) on Apple Silicon; contributors debugging long-prefill or cold-reload loops.

**Related:** [mlx-routing-policy.md](./mlx-routing-policy.md), [scheduling-vram-policy.md](./scheduling-vram-policy.md#prompt-truncation-in-responses-jun-2026), [phase17-llama-server.md](./phase17-llama-server.md#pre-tokenized-prompts-prompttokens), [apple-silicon-metal.md](./apple-silicon-metal.md).

---

## Why this doc exists

Agent clients (Hermes, Mercury, Eliza) often send **the same 100k+ token megaprompt every turn** — system prompt + tool docs + `papers_kb/` + history. On MLX that causes:

1. **Wrong context window** — HF `config.json` can store `vocab_size` in `text_config.max_position_embeddings` (Gemma4 multimodal exports). Zerollama treated that as 262144 ctx → oversized KV budget, no tail truncate, multi-minute prefill.
2. **Double tokenize** — server tokenizes for truncation budget, MLX `Prepare` re-encodes unless `PromptTokens` is passed.
3. **Cold reload every 5 minutes** — default `keep_alive` evicts the MLX subprocess during agent think-time; reload costs 3–10s before prefill even starts.
4. **Empty SSE streams** — clients time out during long MLX prefill because no bytes leave the server until the first token.

These fixes are **generic** (not Hermes-specific): any MLX model + any client that resends huge prompts benefits.

---

## Decision flow

```text
Request arrives (chat/generate)
  → enrichMLXModelConfig (safetensors config.json → ContextLen)
  → capMLXScheduleOptions (num_ctx / num_predict vs model max)
  → mlxKeepAliveFloor (30m default when keep_alive unset)
  → chatPrompt (render → message drop → tail truncate by token ID)
  → mlxEnsurePromptTokens (always tokenize once for MLX passthrough)
  → CompletionRequest.PromptTokens → mlxrunner Prepare (skip re-encode)
  → TextGenerationPipeline prefill → first token
```

---

## Context length correction

**Why:** Multimodal HF exports sometimes put `vocab_size` (262144) into `text_config.max_position_embeddings` instead of the real context window (131072). VRAM tier defaults on 128 GB Mac then inherit 262144 → tail truncate never fires because budget looks infinite.

**Fix:** `x/server/show.go` `buildModelInfo` ignores `text_config.max_position_embeddings` when it equals `text_config.vocab_size`; falls back to top-level `max_position_embeddings`. `enrichMLXModelConfig` applies the corrected value at model load and schedule time.

**Log to watch:**
```
mlx context_length updated from safetensors config  from=262144 to=131072
num_ctx capped to mlx model maximum               from=262144 to=131072
```

Code: `x/server/show.go`, `server/mlx_model_config.go`, `server/images.go`.

---

## Tail truncation

**Why:** Message-level dropping cannot shrink a **single** megaprompt (one user message with 131k tokens). Tail truncate drops tokens from the **front** so the latest user turn survives.

**Budget:** `effectiveChatPromptBudget` = min(request `num_ctx`, model max, runner ctx) minus `num_predict` reserve.

**Log to watch:**
```
prompt tail-truncated to fit context budget  dropped_tokens=65787 kept_tokens=65530
large mlx prompt; prefill may take several minutes  prompt_tokens=65530 num_ctx=131072
mlx prepare tail-truncated prompt to fit context  (runner backstop if server budget missed)
```

Set `"truncate": false` on the request for HTTP 400 instead of silent drop — see [scheduling-vram-policy.md](./scheduling-vram-policy.md#prompt-truncation-in-responses).

**API response (Jul 2026):** final chunks set `prompt_truncated: true` and `original_prompt_tokens` to the **pre**-drop size (e.g. 131317, not 65530). **Why:** earlier paths logged `prompt tail-truncated` but left clients to infer overflow from `prompt_eval_count` alone; runtime-proxy requests also now detect llama-server context shift.

Code: `server/prompt.go` (`tailTruncatePrompt`, `findChatPromptStartIdx`), `server/truncation.go`, `x/mlxrunner/pipeline.go` (`Prepare` backstop).

---

## Single tokenize path (`PromptTokens`)

**Why:** Tail truncate operates on **token IDs**. Detokenizing truncated IDs back to text and re-tokenizing in MLX can diverge (special tokens, BOS, byte boundaries). MLX MTP budget checks need the exact slice the server computed.

**Flow:**
1. `chatPrompt` returns `promptTokens []int` after truncate (or explicit MLX tokenize when truncate skipped).
2. Routes set `CompletionRequest.PromptTokens` via `mlxCompletionPromptTokens`.
3. `x/mlxrunner/client.go` sends `Tokens` on the completion JSON; `Prepare` skips `Tokenizer.Encode`.

**Log to watch (debug):**
```
mlx prompt pre-tokenized for passthrough  tokens=65530
```

Code: `server/prompt.go`, `server/mlx_model_config.go`, `server/routes.go`, `x/mlxrunner/client.go`, `x/mlxrunner/pipeline.go`.

---

## Tokenize cache (bad clients)

**Why:** Even with passthrough, one request can hit `/v1/tokenize` twice (binary search in `findChatPromptStartIdx` + `tailTruncatePrompt` on the same rendered string). Agent loops resend **identical** megaprompts every turn — ~500ms HTTP round-trip each time without cache.

**Not a substitute for truncation** — caching 131k-token entries is ~512 KB each; bounded to 16 entries / ~2M tokens total (~8 MiB).

| Knob | Default | Why |
|------|---------|-----|
| Max entries | 16 | Agent sessions rarely reuse more than a few distinct prompt shapes |
| Max tokens stored | 2_000_000 | Cap memory on 128 GB hosts with multiple models |
| TTL | 15 m | Matches typical agent session length; stale tokenizer reload clears process anyway |

**Log to watch (debug):**
```
mlx tokenize cache hit  prompt_len=847291 tokens=65530
```

Code: `x/mlxrunner/tokenize_cache.go`, `x/mlxrunner/client.go` (`Client.tokenizeCache`).

---

## Agent prompt chain splice (multi-turn MLX cache)

**Why:** Gemma4/Hermes chat templates are **not token-prefix-stable** — each turn replaces the trailing **generation stub** (`<|turn>model\n<|channel>thought…`) with the prior assistant reply, so full-render string prefix matching fails. MLX prefix trie matching then misses on turn 2 (`cached_tokens ≈ 0`).

**Fix:** When `prompt_cache_key` is set, cache the **stable prefix** (render minus gen stub) + token IDs per thread. On append-only history (message fingerprint prefix match), tokenize only the stable delta + new gen stub.

**Log to watch (debug):**
```
mlx agent prompt chain splice  key=hermes:agent:main:cli:1 tokens=65542
mlx prompt chain miss  reason=render_prefix_mismatch  (system prompt edited or compression rewrote history)
```

Code: `server/mlx_prompt_chain.go`, wired from `server/prompt.go`, `server/routes.go` (chat + generate-with-messages).

---

## M15a: live session + restore (Jul 2026)

**Why:** Production Hermes turn 2 often reported **99% `cached_tokens`** but still took **60–90s** to first token — and turn 3+ sometimes collapsed to **~16k cached** (~175s) when `messages_dropped` climbed. Root causes:

| Symptom | Why it happened |
|---------|-----------------|
| **99% cached, slow TTFT** | Trie restore succeeded (`cache hit`, `utilization_pct≈99`) but **live-session** never fired (`fast_path=false`). Gen tokens from turn 1 sat in KV past the new prompt LCP; on Gemma4 OptiQ **rotating KV** cannot `Restore(nil, offset)` once the window has wrapped (~65k tokens) — every turn paid trie page-in/materialize. |
| **`fast_path` never in logs** | Live session requires the **same** `prompt_cache_key` on consecutive turns **and** `lastSessionInputs` from a completed prior turn. Missing key → trie-only path (still can hit 99%, but slower). |
| **~16,384 cached on turn 3+** | Message-level truncation (`messages_dropped` 5→11) rewrites the prompt **from the start**. Trie matches a partial shared prefix but rotating KV can only restore to **snapshot boundaries**. Coarse 2048-token intervals left large gaps; stale prompt-chain splice could misalign token IDs. |
| **`mlx_cache_warn` noise** | Short `/api/generate` sidecar calls matched ~29 trie tokens but KV restore failed on rotating layers — expected, not operator error. |

**Fixes (automatic — no env knobs):**

| Mechanism | Behavior | Why |
|-----------|----------|-----|
| **Live-session LCP** | Per `prompt_cache_key`, store prior **prompt-only** token IDs; next turn: longest common prefix with new inputs, rewind past gen tokens, prefill delta only. | Gen stub replacement changes the tail, not the stable prefix — compare prompts, not trie leaves that include generation. |
| **`rewindCachesViaSnapshots`** | When rotating KV declines live rewind, page in trie snapshots on the **active branch** (no per-cache `Free()` — partial free misaligned layers and forced `freeAll`). | Mirrors `switchToPath` page-in without a full trie round trip; safe for wrapped sliding windows. |
| **`bestRestorableOffset`** | Pick the **largest snapshot boundary ≤ LCP** on the trimmed active path (gen tokens past LCP excluded). Falls back to `capTrieMatchForRestore` only when no interior snapshots exist. | Blind mid-edge cap on the gen-extended leaf rewound too far (e.g. 10 → 8 in tests; 65k → 16k in production). |
| **Same-branch restore** | Trie match extends active path and live KV covers restore point → rewind only, skip leaf page-out/in. Snapshot fallback when live rewind fails on rotating layers. | Avoids ~75s page-in when the runner never switched branches between turns. |
| **`tunePrefillConfig`** | When `prompt_cache_key` + `fast_path`, `same_branch`, or high cache ratio + short tail, relax `materializeEvery` / `clearCacheEvery` (8–16 chunks) on the tail only. | Long-prompt cold defaults (`materializeEvery=1`) were correct for turn 1 but wasted seconds re-evaluating KV already resident after restore. |
| **Trie snapshot interval** | Agent + rotating KV: **1× `sliding_window`** (1024 for Gemma4 OptiQ), min 1024, plus end-of-prompt when `prompt_cache_key` set. Was 2× window (2048). | Finer boundaries reduce restore loss when prefix diverges (message drops, branch changes). |
| **Prompt chain on truncate** | `messages_dropped > 0` → invalidate stable-prefix cache + `prompt_chain_miss` `reason=messages_truncated`. | Cached stable tokens referred to messages no longer in the render; splicing would poison trie matching. |
| **Message fingerprint** | Equal message **count** now checks fingerprint (not only append-only extend). | In-place edits to the last message falsely hit splice when count unchanged. |
| **Warn policy** | `mlx_cache_warn` only at **Warn** for long agent prompts (≥8k matched + key set); short sidecar + expected rotating caps → **Debug**. | Production logs were dominated by harmless `/api/generate` restore misses. |

**Log to watch:**

```bash
# Live session (needs prompt_cache_key on both turns)
jq -c 'select(.event=="mlx_cache" and .fast_path==true)' ~/.ollama/logs/gemma-agent.jsonl

# Same-branch trie restore (fast_path false but hot tail tuning applies)
jq -c 'select(.event=="mlx_cache" and .same_branch==true)' ~/.ollama/logs/gemma-agent.jsonl

# Truncation → expect cold/partial cache, not 99%
jq -c 'select(.event=="prompt_chain_miss" and .reason=="messages_truncated")' ~/.ollama/logs/gemma-agent.jsonl
jq -c 'select(.event=="response_out") | {dropped:.messages_dropped,cached:.cached_prompt_tokens,ms:.duration_ms}' \
  ~/.ollama/logs/gemma-agent.jsonl | tail -10
```

**Example mlxrunner lines:**

```
cache hit  fast_path=true  rewound_from=6587  rewound_to=6586  cached=6586  left=42
cache hit  same_branch=true  cached=65042  utilization_pct=99.3
mlx prefix trie match but KV restore missed  (debug — short sidecar or expected cap)
```

| `gemma-agent.jsonl` field | Meaning |
|---------------------------|---------|
| `fast_path` | Live KV extended via LCP + snapshot rewind — skipped trie page-in/out |
| `same_branch` | Trie same-branch restore — skipped leaf page-out/in |
| `rewound_to` | Actual KV offset after rotating cap (may be < LCP when no snapshot at exact boundary) |
| `messages_dropped` | Oldest messages dropped to fit `num_ctx`; correlates with cache collapse when it increases |

Code: `x/mlxrunner/cache.go` (`tryExtendLiveSession`, `bestRestorableOffset`, `rewindCachesViaSnapshots`), `x/mlxrunner/prefill_config.go`, `x/mlxrunner/pipeline.go`, `server/mlx_prompt_chain.go`, `server/prompt.go`, `agentstats/runner_line.go`.

---

## MLX sidecar defer (Jul 2026)

**Why:** Unkeyed `/api/generate` on the same MLX model (~7.8k tok background work every ~20s) shares one `kvCache` with Hermes agent chat. Even with M15a trie restore, sidecar `begin()` → `switchToPath` clobbers **live** KV between agent turns and blocks `fast_path`.

**Behavior (automatic):**

| Request | Gate |
|---------|------|
| `/v1/chat/completions` or `/api/generate` with `prompt_cache_key` on MLX | Marks model **agent-hot** for the full inference (prefill + decode) |
| Any request with a **different** session key on same MLX runner | **Defers** (polls every 50ms) until hot session finishes and 90s cooldown expires |
| Unkeyed request on same MLX runner | **Defers** — treated as a different key |

**Log to watch:**

```bash
jq -c 'select(.event=="mlx_sidecar_defer")' ~/.ollama/logs/gemma-agent.jsonl
```

Code: `server/mlx_sidecar_gate.go`, wired from `server/routes.go` (`scheduleRunner`, `ChatHandler`, `GenerateHandler`).

---

## MLX keep-alive floor

**Why:** MLX model load takes 3–10s on Apple Silicon. Default Ollama `keep_alive` is 5 m — any agent pause longer than that evicts the runner; next turn pays cold load + full prefill again.

**Behavior:** When `keep_alive` is **unset** on an MLX model, floor to **30 m** (or `OLLAMA_KEEP_ALIVE` if higher). Explicit `keep_alive: 0` still unloads immediately.

Code: `server/mlx_model_config.go` (`mlxKeepAliveFloor`), wired in `server/routes.go` `scheduleRunner`.

---

## SSE stream keepalive (long prefill)

**Why:** Some agent clients (Hermes empty-stream guard) abort when **no SSE bytes** arrive for ~60s. MLX prefill on 65k+ tokens can take 1–8 minutes before the first token.

**Behavior:** After the stream starts, emit periodic `status: keepalive` NDJSON chunks until the first token (or error). Disabled when `OLLAMA_STREAM_KEEPALIVE_INTERVAL=0` or `DebugRenderOnly`.

| Env | Default |
|-----|---------|
| `OLLAMA_STREAM_KEEPALIVE_INTERVAL` | `15` (seconds) |

Code: `server/stream_keepalive.go`, `server/routes.go` (chat + generate).

---

## Operator logs (Jun 2026)

Structured logs added so each iteration is diagnosable without attaching a debugger.

### Inference phases (every request)

```
inference request in   route=/api/chat model=gemma4:26b-mlx-mtp stream=true ...
inference phase        phase=runner_ready   phase_elapsed=3.2s  request_elapsed=3.2s
inference phase        phase=prompt_ready   phase_elapsed=537ms request_elapsed=3.7s
inference phase        phase=first_token    phase_elapsed=87s   request_elapsed=91s
inference response out duration=92s prompt_tokens=65530 original_tokens=131317 truncated_tokens=65787 ...
```

**Why phases:** separates load vs template vs prefill vs decode — the `03:12` audit showed ~537ms `prompt_ready` but 8 min to first token (prefill), not template parsing.

### MLX prefill summary

```
prefill complete  prompt_tokens=65536 cached_tokens=0 prefill_tokens=65536 elapsed=87.3s tok_per_sec=751
prefill snapshots disabled for long prompt  (MTP snapshot path skipped)
MTP disabled for long prompt; using standard decode
peak memory  size=114.80 GiB
```

**Why:** `tok_per_sec` regressions across rebuilds are visible without MLX-internal profilers. Per-chunk `forward_ms` / `materialize_ms` breakdown is at **debug** (`OLLAMA_DEBUG=1`). Gemma4 OptiQ uses **2048-token** prefill chunks automatically (2× sliding window).

### Runner reload reason

**Before:** `needs_reload=true` with no explanation.

**After:**
```
runner needs reload  reload_reason=num_ctx_exceeds_loaded_kv loaded_ctx=131072 want_ctx=262144
runner needs reload  reload_reason=ping_failed
runner needs reload  reload_reason=runner_options_changed loaded_num_ctx=... want_num_ctx=...
```

**Why:** cold reload every agent turn was often `num_ctx` mismatch after context cap fix — this log names the branch.

Code: `server/sched.go` (`needsReload`), `server/inference_access_log.go`.

### `/api/show` failures (Hermes model picker probes)

Clients like Hermes burst `POST /api/show` to validate model names. **404** means the name is not in the local catalog; **500** means the name resolved but metadata loading failed.

```
api/show failed  route=/api/show model=some:tag status=500 show_stage=gguf_metadata error=gguf_metadata: ...
api/show model not found  route=/api/show model=missing:tag status=404
api/show bad request  route=/api/show model= status=400 error=...
```

**`show_stage` values:** `parse_name`, `list_manifests`, `load_model`, `parse_manifest`, `gguf_metadata`, `projector_metadata`.

**Why:** 500s were opaque in access logs; stage + error pinpoints corrupt GGUF, missing projector blob, bad manifest digest, etc.

Reproduce: `curl -s localhost:11434/api/show -d '{"name":"MODEL"}' | jq`

Code: `server/show_log.go`, `server/routes.go` (`ShowHandler`, `GetModelInfo`).

---

## Gemma agent stats file (JSONL)

Production Hermes traffic on `:11434` writes a **session-scoped JSONL log**. Each `./zerollama serve` start writes a `serve_start` line (version, pid, …) and rotates the prior session to **`gemma-agent.jsonl.prev`** when non-empty:

**Default path:** `~/.ollama/logs/gemma-agent.jsonl`

```bash
# tail while Hermes runs
tail -f ~/.ollama/logs/gemma-agent.jsonl | jq -c .

# compare current vs previous serve session (e.g. before/after a binary upgrade)
jq 'select(.event=="serve_start")' ~/.ollama/logs/gemma-agent.jsonl ~/.ollama/logs/gemma-agent.jsonl.prev
```

**Disable:** `ZEROLLAMA_GEMMA_AGENT_LOG=0 ./zerollama serve`

**Custom path:** `ZEROLLAMA_GEMMA_AGENT_LOG=/tmp/gemma.jsonl ./zerollama serve`

Events recorded when `prompt_cache_key` is set (Hermes) or model name contains `gemma`, plus all MLX runner cache/prefill lines:

| `event` | Meaning |
|---------|---------|
| `serve_start` | New serve session: `version`, `edge_build`, `pid`, `goos`/`goarch`; `previous_log` when `.prev` was rotated |
| `request_in` / `response_out` | Route, model, key, tokens, cache hits, duration |
| `request_parsed` | `prompt_cache_key` backfilled after `/v1` middleware (when peek missed nested `extra_body`) |
| `phase` | `runner_ready`, `prompt_ready`, `first_token` timings |
| `prompt_chain_splice` / `prompt_chain_miss` | Stable-prefix tokenize path |
| `mlx_cache` / `mlx_prefill` | mlxrunner trie restore + prefill summary (`fast_path`, `same_branch`, `rewound_to` when present) |
| `runner_reload` | Cold reload reason (`num_ctx`, ping failed, …) |

Serve logs `gemma agent stats log enabled path=...` at startup when active.

---

## Smoke (isolated port)

Hermes and other apps use production `:11434`. Run the two-turn prefix cache smoke on **`:11435`** so traffic does not share the MLX runner:

```bash
MLX_SMOKE_START_SERVE=1 ./scripts/mlx/mlx_prefix_cache_smoke.sh
# or: OLLAMA_HOST=127.0.0.1:11435 ./zerollama serve   # separate terminal
#     ./scripts/mlx/mlx_prefix_cache_smoke.sh
```

Pass criteria: turn 2 `cached_tokens` ≥ 4000 and elapsed ≤ 90s.

---

## What to fix on the client (Hermes / agents)

Server-side caps help but **root cause** is often client context:

- Exclude static corpora (`papers_kb/`) from every turn; pin via retrieval instead.
- Pass stable `options.num_ctx` ≤ model max (**131072** for Gemma4 text on Mac MLX/GGUF).
- Pass **`prompt_cache_key`** per agent thread for prefix reuse:
  - **Hermes** (`provider: custom` → zerollama): `plugins/model-providers/custom` sets `hermes:{session_id}` on `/v1/chat/completions`. Keys are duplicated into **`extra_body`** (OpenAI SDK promotes them to the HTTP JSON root) and into `options` for `/api/chat` after middleware conversion.
  - **OpenAI-compatible ingress:** `/v1/chat/completions` uses `BindChatCompletionRequest` to merge nested `extra_body.prompt_cache_key` when the SDK does not flatten unknown fields.
  - **GGUF + Python runtime:** L3 slot bridge skips repeat system prefill when `ZEROLLAMA_AGENT_CACHE_RUNTIME=auto` (Darwin default when runtime URL is set).
  - **MLX (e.g. `gemma4:26b-optiq`):** mlxrunner trie snapshots every **`sliding_window`** (1024 for OptiQ) plus end-of-prompt when `prompt_cache_key` is set — **why:** rotating KV can only restore at snapshot boundaries; finer intervals reduce the ~16k partial-hit ceiling when `messages_dropped` changes the prefix. Live-session (`fast_path=true`) additionally needs the key on **every** turn. Not L3.
- Set `model.context_length` and `model.ollama_num_ctx: 131072` in `~/.hermes/config.yaml` so Hermes compression budget matches runtime (GGUF metadata may advertise 262144).

Example Hermes config:

```yaml
model:
  default: gemma4:26b-optiq
  provider: custom
  context_length: 131072
  ollama_num_ctx: 131072
providers:
  custom:
    base_url: http://localhost:11434/v1
    models:
      gemma4:26b-optiq:
        context_length: 131072
```

---

## Harness QoS API (generic, Jul 2026)

Any client can detect zerollama via `GET /api/version` (`distribution: zerollama`, `zerollama.capabilities.mlx_qos`) and progressively add scheduling hints. Vanilla Ollama ignores unknown `options` fields.

### `options.zerollama`

| Field | Values | Purpose |
|-------|--------|---------|
| `qos_class` | `interactive` \| `auxiliary` \| `background` | Scheduling intent (aliases: `primary`, `aux`, `bg`) |
| `qos_priority` | `0–100` | Class inferred when `qos_class` omitted (`≥70` interactive) |
| `session_group` | string | Harness id for shared cache branch (`aux:{model}:{group}`) |
| `session_parent` | string | Parent thread `prompt_cache_key` (logging / future parent-aware defer) |
| `project_id` | string | Client harness id for fleet / `zerollama ps` (aliases: `client_id`, `project`) |
| `project_name` | string | Human label — Discord channel, audit phase, workspace name (aliases: `client_name`) |
| `cache_scope` | `auto` \| `thread` \| `shared` | `thread` = keep key as-is (live KV); `shared` = collapse ephemeral to shared branch |

Legacy flat fields still honored: `mlx_session_class`, `mlx_session_parent`.

### Examples

**Interactive agent (any harness):**

```json
{
  "options": {
    "prompt_cache_key": "myapp:agent:thread:abc",
    "zerollama": { "qos_class": "interactive", "cache_scope": "thread" }
  }
}
```

**Auxiliary worker (compression, subagent):**

```json
{
  "options": {
    "zerollama": {
      "qos_class": "auxiliary",
      "session_group": "my-harness",
      "session_parent": "myapp:agent:thread:abc",
      "cache_scope": "shared"
    }
  }
}
```

**Background batch (`/api/generate`):**

```json
{
  "options": {
    "prompt_cache_key": "myapp:bg:questions",
    "zerollama": { "qos_class": "background", "session_group": "my-harness", "cache_scope": "shared" }
  }
}
```

OpenAI SDK: nest under `extra_body.zerollama` (merged into `options` by `BindChatCompletionRequest`).

Server-inferred fallbacks (no harness deploy): `hermes:agent:*` → interactive; timestamped `hermes:YYYYMMDD_*` → auxiliary; unkeyed `/api/generate` → background.

### Multimodal coverage (Jul 2026)

| Modality | Routes | Default class |
|----------|--------|---------------|
| `text` | `/api/generate`, `/api/chat`, `/v1/chat/completions` | route/stream inferred |
| `vision` | chat with images (not video) | auxiliary (stream) |
| `video_understanding` | chat with `videos[]` / `video_spans` | auxiliary (stream) |
| `image_generation` | image-capable `/api/generate`, `/v1/images/*` | **background** |
| `video_generation` | `POST /v1/videos` (Wan) | **background** (waits behind interactive MLX) |

Image and video generation default to `background` + `cache_scope: shared` unless you set explicit `qos_class`. Wan video jobs wait for hot interactive agent threads before queuing; MLX imagegen uses the same per-runner gate as chat.

**Image generation QoS:**

```json
{
  "model": "z-image-turbo",
  "prompt": "a red cube",
  "options": {
    "zerollama": { "qos_class": "background", "session_group": "my-harness" }
  }
}
```

**Video generation QoS (`POST /v1/videos`):**

```json
{
  "model": "wan2.1-t2v-1.3b",
  "prompt": "ocean waves at sunset",
  "options": {
    "zerollama": { "qos_class": "background", "session_group": "my-harness" }
  }
}
```

Detect supported modalities via `GET /api/version` → `zerollama.qos.modalities`.

### Project tracking (`zerollama ps`, Jul 2026)

**Why:** Multiple harnesses on one node (Hermes Discord, ruby-trivia audit, manual scripts) share GPU. Model name and VRAM alone do not answer “who is blocking my load?”

Send stable **`project_id`** (app/repo) and optional **`project_name`** (thread, phase, channel):

```json
{
  "options": {
    "prompt_cache_key": "hermes:agent:main:discord:dm:123",
    "zerollama": {
      "qos_class": "interactive",
      "project_id": "hermes-lean",
      "project_name": "discord:dm:123"
    }
  }
}
```

**CLI:** `./zerollama ps` adds **PROJECT** and **SESSION** when gate metadata is active.

**API:** `GET /api/ps` → `models[].zerollama.sessions[]`.

**Client convention:** probe `GET /api/version` (`distribution: zerollama`) before sending Tier 2 fields — vanilla Ollama ignores unknown options but should not receive eliza / prefix-mm-cache / 30m keep-alive floors from clients. Full branching rules: [agent-qos-and-project-tracking.md](./agent-qos-and-project-tracking.md).

---

## Tests

| Test | Package | What it proves |
|------|---------|----------------|
| `TestCapMLXScheduleOptions` | `server` | num_ctx capped to manifest max |
| `TestEffectiveChatPromptBudgetCapsToModelMax` | `server` | budget uses model max, not inflated request ctx |
| `TestMLXKeepAliveFloor` | `server` | 30m floor; explicit keep_alive honored |
| `TestBuildModelInfo` (bogus text_config max_pos) | `x/server` | vocab_size ≠ context window |
| `TestTokenizeCacheHitMiss` | `x/mlxrunner` | LRU hit/miss + copy safety |
| `TestTryExtendLiveSession*` / `TestBestRestorableOffset*` | `x/mlxrunner` | Live-session LCP, rotating snapshot rewind, restore boundary pick |
| `TestMLXPromptChain*` / `TestMLXPromptChainInvalidate` | `server` | Stable-prefix splice, fingerprint on equal count, invalidate on truncate |
| `TestInferencePath*` / `TestGateSessionKey*` / `TestAgentSessionMetadataEnabled` | `server` | GGUF preserves client keys; eliza gated; embedding-only skips QoS |
| `TestReserveScheduleQoSClaimsBeforeRunnerWait` | `server` | TOCTOU: gate claimed before runner wait |
| `TestStreamKeepalive*` | `server` | keepalive session lifecycle |

---

## Non-goals

- Replacing client-side context management with unbounded server caches.
- Routing MLX through Python runtime or L3 llama-server slot bridge.
- Gemma4-specific hardcoded 262144 → 131072 overrides (generic config.json fix only).
