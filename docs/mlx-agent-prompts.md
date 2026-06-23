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

Set `"truncate": false` on the request for HTTP 400 instead of silent drop — see [scheduling-vram-policy.md](./scheduling-vram-policy.md).

Code: `server/prompt.go` (`tailTruncatePrompt`, `findChatPromptStartIdx`), `x/mlxrunner/pipeline.go` (`Prepare` backstop).

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

**Why:** `tok_per_sec` regressions across rebuilds are visible without MLX-internal profilers.

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

## What to fix on the client (still recommended)

Server-side caps help but **root cause** is often client context:

- Exclude static corpora (`papers_kb/`) from every turn; pin via retrieval instead.
- Pass stable `options.num_ctx` ≤ model max (131072 for Gemma4 text).
- Use `prompt_cache_key` / L3 for **repeat system prefix** on GGUF runtime models (MLX uses trie cache inside mlxrunner, not L3 slot bridge).

---

## Tests

| Test | Package | What it proves |
|------|---------|----------------|
| `TestCapMLXScheduleOptions` | `server` | num_ctx capped to manifest max |
| `TestEffectiveChatPromptBudgetCapsToModelMax` | `server` | budget uses model max, not inflated request ctx |
| `TestMLXKeepAliveFloor` | `server` | 30m floor; explicit keep_alive honored |
| `TestBuildModelInfo` (bogus text_config max_pos) | `x/server` | vocab_size ≠ context window |
| `TestTokenizeCacheHitMiss` | `x/mlxrunner` | LRU hit/miss + copy safety |
| `TestStreamKeepalive*` | `server` | keepalive session lifecycle |

---

## Non-goals

- Replacing client-side context management with unbounded server caches.
- Routing MLX through Python runtime or L3 llama-server slot bridge.
- Gemma4-specific hardcoded 262144 → 131072 overrides (generic config.json fix only).
