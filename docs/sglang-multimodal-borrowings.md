# SGLang-inspired multimodal borrowings (Jun 2026)

This document records **what** zerollama adopted from [SGLang](https://github.com/sgl-project/sglang) multimodal and usage patterns, and **why** each piece exists. It is not a commitment to run SGLang or achieve full parity.

**Why borrow at all:** SGLang invests heavily in agent-scale VLMs (repeat clips, OpenAI-shaped usage, prefix reuse). Zerollama’s **non-goals** remain RadixAttention v1, required SGLang sidecars, and bit-for-bit stack parity. We port **narrow patterns** that fit the existing Go + Python + ffmpeg native path.

**Related:** [video-understanding.md](./video-understanding.md) (native pipeline), [video-parity.md](./video-parity.md) (matrix), [gpu-profiles-l3.md](./gpu-profiles-l3.md) (KV prefix cache), [ROADMAP.md](./ROADMAP.md) Option 2.

---

## Shipped (Tier 1 + follow-up)

### 1. Pooled HTTP transport for `video_url`

**Where:** `openai/video_url.go` — process-wide cloned `http.Transport` (`MaxIdleConnsPerHost: 8`).

**Why:** Agent threads often re-fetch the same HTTPS clip across turns. A fresh client per request discarded TLS session reuse and added latency. Per-request deadlines still come from the Gin `context` and `OLLAMA_VIDEO_FETCH_TIMEOUT` — the transport is shared; timeouts are not frozen at init.

### 2. Remote `video_url` body cache

**Where:** `openai/video_fetch_cache.go` — LRU 32 entries, 30 m TTL, keyed by full URL.

**Why:** After SSRF checks pass once, repeat GETs of the same signed URL should not hit the CDN again. This sits **before** ffmpeg; it saves network and disk write of the container blob.

**Observability:** `video url fetch cache hit` logs **hostname only** (not query params) so operators see hits without leaking tokens.

### 3. Video expansion cache (global)

**Where:** `server/modality/video_cache.go` — keyed by `(sampling policy fingerprint, video SHA-256)`.

**Why:** Repeat agent turns send the same video bytes; ffmpeg subprocess work is the dominant cost. Global LRU (32 entries) shares wins across all clients on the node.

### 4. Session-scoped expansion cache

**Where:** `server/modality/session_video_cache.go` — keyed by `prompt_cache_key` / `eliza.conversationId` + same video key as global cache.

**Why:** Global LRU can evict a clip when many distinct videos flow through the node. Agent threads with a stable L3 session key keep expanded PNG frames warm for **their** clips even after global eviction (up to 16 videos per session, 256 sessions, 30 m sliding TTL on hit).

**Lookup order:** session → global → ffmpeg / `ExternalVideoDecodeHook`.

**Requires:** `options.prompt_cache_key` (or eliza aliases) on the chat request — same keys as [L3 prefix cache](./gpu-profiles-l3.md).

### 5. Preprocessed expansion fast path

**Where:** `server/modality/video_frames.go` — messages with `video_spans` and no raw `videos` skip ffmpeg.

**Why:** SGLang accepts already-processed multimodal layouts; clients that send expanded frames + span metadata should not pay decode twice. Invalid spans (`frame_count` sum > `len(images)`) return **400** before render.

### 6. Multimodal usage breakdown (OpenAI)

**Where:** `server/modality/token_budget.go`, `api.Metrics`, `openai.Usage.prompt_tokens_details`.

**Why:** OpenAI and SGLang expose per-modality prompt token estimates (`image_tokens`, `video_tokens`, `audio_tokens`) for billing and debugging. Zerollama uses an **upper-bound heuristic** (default 768 tokens per still frame / video frame / audio clip) after expansion; values are not from the vision tower.

**Gating:** Estimates run only when `ChatRequestHasMultimodalPayload()` — avoids inventing counts on text-only turns.

**Scoping:** Only the **latest user message** is counted (same as preflight for pre-expanded spans). Agent threads echo multimodal history; counting old frames would inflate `usage` and access-log modality fields every follow-up.

### 7. `cached_tokens` in usage

**Where:**

- Go llama-server path: `llm/llama_server.go` ← `timings.cache_n`
- Python runtime: `runtime/runtime/llama_timings.py` → `prompt_eval_count`, `prompt_eval_cached_count`, `cached_prompt_tokens`
- HTTP: `api.Metrics.cached_prompt_tokens` → OpenAI `usage.prompt_tokens_details.cached_tokens`
- Access log: `inference response out` includes `cached_prompt_tokens` when &gt; 0

**Why:** Prefix KV reuse (L3 `cache_prompt` / llama-server slot cache) is invisible without this field. Operators and agents need to see **how much** of the prompt was served from cache vs re-evaluated — same semantics as OpenAI and llama-server.

**Wire note:** Python subprocess emits both `prompt_eval_cached_count` (Go `CompletionResponse`) and `cached_prompt_tokens` (API alias). Go unmarshaling accepts either.

**SGLang `sglext` (Jun 2026):** When prefix cache hits are present, OpenAI `/v1/chat/completions` responses also include `sglext.cached_tokens_details` with `device` (GPU/slot cache), optional `host` (HiCache tier), and optional `storage` + `storage_backend` (L3). Streaming with `stream_options.include_usage` attaches the same object on the final usage chunk. Access log mirrors `cached_tokens_device`, `cached_tokens_host`, `cached_tokens_storage` when wired.

### 7b. `limit_mm_data_per_request`

**Where:** `OLLAMA_LIMIT_MM_DATA_PER_REQUEST` JSON (e.g. `{"image":4,"video":1,"audio":1}`) → `server/modality/limit_mm_data.go` preflight on `/api/chat` before ffmpeg expand.

**Why:** SGLang rejects oversized MM attachments on the active turn before ViT work. Zerollama counts only the **latest user message** (agent history echoes are excluded); pre-expanded `video_spans` count as videos, not still images.

**Gating:** No-op when env unset or all caps zero. Runs for image-only turns when `image` cap is set (not tied to video preflight).

### 7c. `precomputed_embedding` ingest (partial)

**Where:** `/api/chat` message `images` JSON object `{format:"precomputed_embedding", feature:[[...],...]}` or `precomputed_embeddings` array; requires `padded_input_ids` on the same message.

**Why:** SGLang clients that already ran the ViT on the client/GPU can skip server-side encode when pretokenized layout is supplied. Zerollama maps each feature row to a llamarunner embed chunk at padded vision slots.

**Limits:** Exactly one precomputed item per message (SGLang rule); cannot mix raw image bytes and precomputed on the same turn. Requires **`padded_input_ids`** on that message. **ollama-engine** supports Qwen3-VL / qwen25vl / **glmocr** (+ `grid_thw` patch grid); **Gemma3 / Gemma4 / mllama / deepseekocr** (rows, no grid); **mistral3** (per-row strips); **llama4** (optional `grid_thw` `[1,tile_h,tile_w]` for multi-tile + global chunk); **lfm2** (single-tile rows only). **llama-server** rejects precomputed (base64 rasters only).

**Observability:** grep `precomputed_embedding runner inject` or `processor_output runner inject` (`engine=ollama` on ollama-engine). Set **`enable_prefix_mm_cache: true`** in options with **`prompt_cache_key`** to pin session ViT overlay (logs a hint if the flag is set without a session key).

### 7d. `processor_output` ingest (partial)

**Where:** `/api/chat` message `images` object `{format:"processor_output", pixel_values:..., image_grid_thw:[T,H,W]}` or `processor_outputs` array; requires `padded_input_ids`.

**Why:** SGLang clients run the HF processor locally and send patch tensors so the server skips PNG decode + processor — only vision tower + LLM prefill run on ollama-engine.

**Limits:** Exactly one item per message; cannot mix with raw bytes or precomputed_embedding. **ollama-engine:** Qwen3-VL / qwen25vl / **glmocr** (patch grid `[T,H,W]`); **Gemma3 / Gemma4 / mistral3 / llama4 / lfm2** (`[1, H, W]` pixels; Gemma3 H=W=`vision.image_size`; llama4/lfm2 single-tile only); **mistral3** precomputed uses per-row strips. **deepseekocr** processor_output deferred (SAM pipeline). **ggml llamarunner** and **llama-server** reject processor_output (mtmd/subprocess need rasters).

### 8. Gemma4 video span placeholders

**Where:** `model/renderers/gemma4.go` — HF-style `<|video|>` / `<|image|>` when `RenderImgTags` is false (tests / HF path). Production chat sets `RenderImgTags=true` → per-frame `[img-N]` tags.

**Why:** Frame-list semantics are the production default; HF placeholders document the alternate layout for models that group clips.

### 9. Audio on correct field

**Where:** `openai/openai.go` — `input_audio` → `api.Message.AudioClips` (not stuffed into `Images`).

**Why:** Usage breakdown and preflight must distinguish audio from vision; mixing into `Images` broke token estimates and capability checks.

### 10. Preflight for pre-expanded `video_spans`

**Where:** `server/modality/preflight.go` — when a message has `video_spans` + `images` but no raw `videos`, vision frame counts come from spans (SGLang-style pre-expanded path). Raw `videos` path unchanged.

**Why:** Clients that already expanded frames must still pass vision budget checks; skipping those turns understated VRAM/context pressure.

### 11. Policy golden tests (Phase D partial)

**Where:** `server/modality/video_policy_golden_test.go` — deterministic policy fingerprints, multi-clip span order, policy-isolated cache keys via `ExternalVideoDecodeHook` (no ffmpeg fixtures in git).

**Why:** Regression gate for expansion policy without shipping MP4 blobs or requiring GPU in CI.

### 12. ffmpeg golden test (Phase D partial, optional)

**Where:** `server/modality/video_ffmpeg_golden_test.go` — lavfi-generated MP4, asserts frame count + PNG output + global cache hit. Skips when ffmpeg is absent.

**Why:** Hook-based tests do not exercise real ffmpeg argv; this closes the gap without checked-in video binaries.

### 13. mllama preflight (Phase C partial)

**Where:** `server/modality/preflight.go` — `PreflightMllamaSingleImage` before ffmpeg for `mllama` family models.

**Why:** Llama 3.2 Vision allows one raster per message; failing before expand names `max_frames=1` instead of opaque post-expand errors.

### 14. Agent two-turn session cache test + live smoke

**Where:** `server/modality/session_video_cache_test.go` (`TestExpandVideosInChatRequest_agentSecondTurn`); `scripts/video/video_agent_cache_smoke.sh`.

**Why:** Unit LRU tests do not model the agent pattern (turn 2 resends clip on latest message only). Live smoke (`RUN_E2E_VIDEO_AGENT=1`) checks ffmpeg + `video sample session cache hit` in serve logs.

### 15. Capability and preflight on pre-expanded `video_spans`

**Where:** `server/modality/multimodal.go` (`ChatRequestHasVideoPayload`); `server/routes.go` capability gate; `server/modality/preflight.go`.

**Why:** Early Tier 1 only gated on raw `videos[]`. SGLang-style clients that send `images` + `video_spans` without raw blobs would skip vision checks and under-budget preflight — non-vision models could accept frames, and context limits would surface only after expensive work.

### 16. Multi-turn preflight scoping (latest user)

**Where:** `server/modality/preflight.go` — `PreflightVideoVisionBudget` and `PreflightMllamaSingleImage` use `lastUserMessageIndex`.

**Why:** Agents echo full chat history. Counting pre-expanded frames on every historical user turn inflated vision token estimates and blocked innocent follow-ups (e.g. turn 2 text-only after turn 1 video). Raw `videos` on any message still count — those bytes will expand on this request.

### 17. OpenAI session cache keys

**Where:** `openai/openai.go` — `ChatCompletionRequest.prompt_cache_key` and `options` merged into `api.ChatRequest.Options`.

**Why:** Session expansion cache and L3 slot pinning already read `options.prompt_cache_key` on the native path. OpenAI `/v1/chat/completions` clients had no way to pass the key without a second API shape — repeat `video_url` clips could not use per-thread ffmpeg cache from OpenAI-only integrations.

### 18. OpenAI `video_url` agent session test

**Where:** `openai/video_agent_session_test.go` — `FromChatRequest` + `ExpandVideosInChatRequest` two-turn with `prompt_cache_key`.

**Why:** Native `/api/chat` agent tests do not prove OpenAI content-array `video_url` wiring shares the same session LRU.

### 19. Qwen3-VL video span rendering (Phase B partial)

**Where:** `model/renderers/qwen3vl.go`, `qwen3vl_video_test.go`.

**Why:** Phase B asks for span-aware templates. Qwen3-VL matches SGLang with **per-frame** vision blocks in HF mode; production uses `[img-N]` because mtmd consumes a flat raster list — not a single grouped video token (unlike Gemma4 `<|video|>`).

### 20. `padded_input_ids` layout cache (partial)

**Where:** `server/modality/session_video_cache.go`, `video_frames.go` — `padded_input_ids` stored in **session** expansion LRU (`layouts` map keyed by video digest); global LRU shares frames + `grid_thw` only.

**Why:** Agent turn 1 may send pretokenized layout from an upstream processor; turn 2 often resends only the video clip. Restoring `padded_input_ids` from session cache keeps preflight/usage accurate without clients retransmitting the full id vector. Layout is **not** global — it encodes the client-specific pretokenized prompt, not just the clip bytes. Single-clip messages only (multi-clip messages do not cache layout).

**Observability:** `video layout session cache hit` when turn 2 restores layout without client resend.

### 21. `padded_input_ids` runner stub (partial)

**Where:** `server/modality/preprocessed.go` (`LatestUserPaddedLayout`, `LogPaddedLayoutRunnerStub`); `server/routes.go` after expand; `server/inference_access_log.go`; `api.DebugInfo` on `_debug_render_only`.

**Why:** SGLang preprocessed clients need a clear contract: layout is accepted for preflight/usage but native render still expands/templates images until a family processor hook exists. Operators grep `padded_input_ids runner stub` or `padded_layout_consume=deferred` on `inference response out`.

### 22. Pre-expanded session layout cache (partial)

**Where:** `server/modality/preprocessed_layout_cache.go` — fingerprint of single-clip `images` + `video_spans`; session `layouts` map (same store as raw-video layout).

**Why:** SGLang clients that send pre-expanded frames on turn 1 (`images` + `video_spans` + `padded_input_ids`) often omit layout on turn 2. Fingerprint keys on frame bytes + span metadata so restore works without raw `videos[]`.

**Observability:** `preprocessed layout session cache hit` on turn 2 restore.

### 23. Qwen3-VL partial layout consume (HF path + runner inject)

**Where:** `model/renderers/qwen3vl.go`, `server/modality/padded_layout_consume.go`, `server/modality/build_padded_prompt.go`, `server/routes.go`, `runner/llamarunner/padded_inputs.go`, `runner/llamarunner/padded_families.go`, `llm/padded_prompt_llama_server.go`.

**Why:** SGLang `padded_input_ids` already includes vision placeholder token positions. **Runner inject (Jun 2026):** routes splice latest-user `padded_input_ids` into the rendered template (`BuildPaddedCompletionPromptTokens` uses **last** `<|im_start|>user` block — works with multimodal chat history). Renderer skips duplicate `<|vision_start|>…` blocks **or** `[img-N]` tags when padded layout is present. llamarunner replaces each `<|vision_start|>…<|vision_end|>` block with full mtmd chunks for the next image. **llama-server subprocess (Jun 2026):** maps each vision block to a media marker + `multimodal_data`. Deferred when an **earlier** turn also sent `padded_input_ids`.

**Observability:** `padded_input_ids runner inject` (ollama-engine + subprocess llamarunner; ollama-engine adds `engine=ollama`); `padded_input_ids llama-server inject` (subprocess); `layout_consume=qwen3vl_hf_runner_inject` on `inference response out`.

**Audit hardening (Jun 2026 follow-up):** see §25 below — tool-turn splice, safe placeholder skip, llama-server truncation with media, inject fallback.

### 25. `padded_input_ids` audit hardening (tool turns + llama-server + safe fallback)

**Where:** `server/modality/build_padded_prompt.go`, `padded_layout_consume.go`, `server/ggml_padded_prompt.go`, `server/routes.go`, `model/renderers/qwen3vl.go`, `llm/padded_prompt_llama_server.go`, `llm/llama_server.go`, `runtime/runtime/server/openai_v1.py`.

**Why this pass exists:** The first runner-inject ship assumed user blocks in the rendered template always matched `role=user` messages 1:1. Qwen3-VL **tool** turns render as `<|im_start|>user\n<tool_response>…` — they look like user blocks but are not real user messages. That skewed span counts, multi-turn splice returned `(nil, false)` **silently**, and the renderer had already stripped vision placeholders → broken multimodal prompts with no vision tags and no injected ids.

| Fix | Why |
|-----|-----|
| **Exclude tool pseudo-user spans** in `qwenVLUserContentSpans` | Align rendered spans with `userMessageIndices` so dual-padded agent history + tool loops splice correctly |
| **`MessageSkipsVisionPlaceholdersForChat`** | Latest user with `padded_input_ids` skips duplicate placeholders; **prior** padded users keep placeholders when any `role=tool` exists — inject failure can still render `[img-N]` / vision blocks for history |
| **`deferred_multimodal_history` on splice failure** | `ggmlPaddedCompletionPromptTokens` downgrades mode; routes propagate it for logs even when no `prompt_tokens` inject — operators grep mismatch instead of silent wrong prompts |
| **Warn on span mismatch** | `padded_input_ids splice failed: user span count mismatch` when prior-user padded layout cannot align |
| **Pretokenized truncation with media** (`llama-server`) | Previously skipped truncate when `len(media)>0`; huge pretokenized layouts could exceed `num_ctx`. Token-level truncate now runs; cut window expands to whole `<|vision_start|>…<|vision_end|>` blocks so media markers stay paired |
| **Runner inject without vision blocks** | If client sends images but pretokenized ids lack `151652` (`vision_start`), detokenize + standard `[img-N]` media path instead of silent token-only failure |
| **Unclosed vision blocks** | Incomplete `vision_start` without matching `vision_end` stays as text tokens — no phantom `multimodal_data` slot |
| **OpenAI `finish_reason: cancelled`** | SSE disconnect prefill abort now surfaces `cancelled` (not mapped to `stop`) so agents distinguish user abort from natural completion |

**Tests:** `build_padded_prompt_test.go` (tool turn + dual padded), `padded_layout_consume_test.go` (prior user + tools), `padded_prompt_llama_server_test.go` (vision-aware truncate, unclosed block).

### 26. Non-stream prefill cancel on HTTP disconnect (partial)

**Where:** `runtime/runtime/server/disconnect_stream.py` (`run_sync_on_disconnect`); `runtime/runtime/server/app.py` (non-stream `/api/generate`, `/api/chat`, `/v1/chat/completions`); `runtime/runtime/engine.py` (`generate` + `prefill_cancel`); `runtime/runtime/worker/llama_inprocess.py`.

**Why:** Streaming endpoints already called `prefill_abort_set()` when `request.is_disconnected()` during long chunked prefill. Non-stream agents (common for tool loops) could close the HTTP connection while the runtime kept burning GPU until prefill finished — same failure mode SGLang addresses with abortable prefill.

**Scope:** In-process **ctypes** backend honors cancel between native prefill chunks. **llama-server subprocess** closes the HTTP connection on cancel (aborts in-flight `/completion`). **llama_cpp_python** wheel checks cancel between stream chunks; **non-stream wheel** routes through internal streaming when `prefill_cancel` is set (Jun 2026). Non-stream cancel returns HTTP **499** with `done_reason=cancelled` in access logs when abort wins the race.

### 27. ViT embed cache sizing observability

**Where:** `server/modality/vit_embed_cache.go`, `server/routes.go` (after video expand).

**Why:** `OLLAMA_IMAGE_EMBED_CACHE_SIZE` defaults to 4; 32-frame video agents exceed that every turn. llamarunner now auto-grows per §28; this log fires only when frames exceed `OLLAMA_IMAGE_EMBED_CACHE_MAX`.

### 28. ViT embed cache auto-grow per turn (partial prefix-mm)

**Where:** `runner/llamarunner/image.go` (`growCacheForDistinctFrames`), `runner/ollamarunner/vision_embed_cache.go`, `runner/llamarunner/runner.go`, `envconfig/config.go` (`ImageEmbedCacheMax`).

**Why:** Requiring operators to set `OLLAMA_IMAGE_EMBED_CACHE_SIZE=32` before the first video request is fragile. SGLang keeps all clip frames in prefix-mm cache; zerollama grows the per-runner LRU to `min(frame_count, OLLAMA_IMAGE_EMBED_CACHE_MAX)` at the start of each `[img-N]` or padded-input multimodal encode. Still per-runner session (not cross-request radix sharing).

**Env:** `OLLAMA_IMAGE_EMBED_CACHE_MAX` (default 64). Initial slots remain `OLLAMA_IMAGE_EMBED_CACHE_SIZE` (default 4).

### 29. Session-keyed ViT embed overlay (`enable_prefix_mm_cache` session pin)

**Where:** `runner/llamarunner/image.go` (`MultimodalTokenize` + `GetPrecomputedChunks` session overlay), `runner/ollamarunner/vision_embed_cache.go`, `vision_embed_preprocessed.go`, `runner/ollamarunner/runner.go`, `runner/llamarunner/runner.go`, `llm/server_shared.go` (`SessionViTOverlay`), `server/modality/prefix_mm_cache.go` (`SessionViTOverlayEnabled`), `server/routes.go`, `openai/openai.go`.

**Why:** Server-side session video LRU already pins expanded frames per `prompt_cache_key`; the runner global ViT LRU (4–64 slots) can still evict clip frames between agent turns under fleet load. SGLang's `enable_prefix_mm_cache` keeps encoder outputs hot per conversation — zerollama mirrors that with a per-runner session overlay (32 sessions, 30m TTL) keyed by the same `ExtractPromptCacheKey` string as expansion/layout caches.

**Operator:** set `prompt_cache_key` on every agent turn. Session overlay defaults **ON** when the key is set; set `enable_prefix_mm_cache: false` to disable session pin and rely on global LRU only. Without a session key, `/api/chat` logs a hint and overlay stays off.

**Cache layers (turn 2+):** PNG bytes → global LRU + session overlay on both runners; `precomputed_embedding` → global + session on ollama-engine and ggml llamarunner; `processor_output` → global + session on ollama-engine only (llamarunner still requires PNG/mtmd).

**What differs:** overlay is per loaded model runner (not cross-runner radix); llamarunner stores `MtmdChunk` / `visionChunk` slices; ollama-engine stores materialized float32 vision tensors + metadata.

### 30. Gemma4 `padded_input_ids` runner inject

**Where:** `server/modality/build_padded_prompt.go` (`BuildGemma4PaddedCompletionPromptTokens`), `server/modality/padded_layout_consume.go`, `server/ggml_padded_prompt.go`, `runner/llamarunner/padded_inputs.go`, `llm/padded_prompt_llama_server.go`, `llm/llama_server.go`, `model/renderers/gemma4.go`.

**Why:** SGLang preprocessed clients send pretokenized layouts for Gemma4 VLMs; native path previously logged `deferred_non_qwen3vl` and re-rendered `[img-N]` tags. Gemma4 splices `padded_input_ids` into `<|turn>user\n…<turn|>` blocks and injects ViT embeds at runtime-resolved `<|image|>` soft token ids (`gemma4_img_runner_inject`) on **ollama-engine**, **ggml llamarunner**, and **llama-server subprocess** (`prompt_string` + `multimodal_data`).

**Operator:** same `prompt_cache_key` + `padded_input_ids` contract as Qwen3-VL; grep `padded_layout_consume=gemma4_img_runner_inject`.

**What differs from Qwen3-VL:** no vision_start/end blocks — soft tokens per raster (`<|image|>`), per clip (`<|video|>` → N frame injects), or per audio (`<|audio|>`); tool responses are not pseudo-user spans (Gemma4 folds tools into assistant turns).

### 31. `grid_thw` runner hints (partial)

**Where:** `server/modality/grid_thw_raster.go`, `server/prompt.go`, `llm.ImageData.GridTHW`, `runner/llamarunner/grid_thw_hint.go`, `runner/ollamarunner/grid_thw_hint.go`.

**Why:** SGLang/Qwen clients attach `[T,H,W]` on `video_spans`; zerollama used it for preflight/usage only. Each expanded frame now carries optional `[1,H,W]` on runner `Images[]` so operators can compare client layout vs mtmd embed counts.

**Operator:** grep `vision grid hints` (Info summary) or `vision grid hint` (per-frame debug). Hints are observability only — mtmd still encodes from pixels until llama.cpp exposes grid overrides. **Go seam (Jun 2026):** `llama.MtmdContext.MultimodalTokenize(..., gridTHW)` + llamarunner `ImageContext.MultimodalTokenize` pass `img.GridTHW`; debug log when hint present until upstream lands. Upstream handoff: [mtmd-grid-thw-handoff.md](./mtmd-grid-thw-handoff.md).

### 32. Ollama-engine padded inject (Mac default)

**Where:** `runner/ollamarunner/padded_inputs.go`, `runner/ollamarunner/padded_gemma4.go`, `runner/ollamarunner/runner.go`, `server/ggml_padded_prompt.go`, `server/modality/build_padded_prompt.go` (`BuildMllamaPaddedCompletionPromptTokens`).

**Why:** Qwen3-VL, Gemma4, and mllama are in `OllamaEngineRequired()` — Mac Metal default loads **ollama-engine** (`ollamarunner`), not subprocess llamarunner. Padded inject on llamarunner/llama-server alone missed the primary native path.

**Families:**

| Consume mode | Splice | Runner inject |
|--------------|--------|---------------|
| `qwen3vl_hf_runner_inject` | Qwen3-VL HF user spans | `EncodeMultimodal` at each `151652…151653` block |
| `gemma4_img_runner_inject` | `<\|turn>user\n…<turn|>` | Soft tokens `<\|image|>`, `<\|video|>` (N frames), `<\|audio|>` |
| `mllama_img_runner_inject` | Llama3 `<\|start_header_id\|>user…<\|eot_id\|>` | Slot token `128256` (`<\|image|>` placeholder) |
| `gemma3_img_runner_inject` | `<start_of_turn>user\n…<end_of_turn>` | `<start_of_image>` token `255999` |
| `llama4_img_runner_inject` | `<\|header_start\|>user…<\|eot\|>` | `<\|image_start\|>…<\|image_end\|>` block (`200080`) → `PostTokenize` tile expand |
| `lfm2_img_runner_inject` | `<\|im_start\|>user\n…<\|redacted_im_end\|>` | `<\|image_start\|>…<\|image_end\|>` block or contiguous `<image>` runs → `PostTokenize` tile expand |
| `glmocr_img_runner_inject` | `<\|user\|>\n…` (until next role tag) | image_start…image_end block (`59256…59257`) → `PostTokenize` M-RoPE expand |
| `mistral3_img_runner_inject` | `[INST] … [/INST]` | `[IMG]…[IMG_BREAK]…[IMG_END]` block (`10/12/13`) → `PostTokenize` Pixtral expand |
| `deepseekocr_img_runner_inject` | content-order user spans (no role wrappers) | contiguous `<image>` token runs (`128815`) → `PostTokenize` |

**Note:** `qwen2vl` / `qwen25vl` families reuse `qwen3vl_hf_runner_inject` (vision tokens `151652…151653`) via `isQwen3VLModel` — no separate consume mode.

**Operator:** same `prompt_cache_key` + `padded_input_ids` contract; grep consume modes above. llama-server subprocess handles Qwen3-VL + Gemma4 when `ZEROLLAMA_LLAMA_SERVER=1`; mllama, Gemma3, Llama4, LFM2, GLM-OCR, Mistral3, and **DeepSeek-OCR** use ollama-engine.

**What differs from llamarunner:** ollama-engine calls `model.MultimodalProcessor.EncodeMultimodal` + `PostTokenize` in-process (no mtmd subprocess); Gemma4 and LFM2 token ids resolved at sequence start via tokenizer.

### 33. LFM2 padded inject

**Where:** `server/modality/build_padded_prompt.go` (`BuildLfm2PaddedCompletionPromptTokens`), `runner/ollamarunner/padded_lfm2.go`, `runner/ollamarunner/padded_inputs.go`, `model/renderers/lfm2.go`.

**Why:** LFM2-VL (`lfm2` / `lfm2moe`) is `OllamaEngineRequired` and uses ChatML user spans identical to Qwen3-VL HF delimiters, but vision layout uses `<|image_start|>…<|img_row_N_col_M|>…<|image_end|>` (or flat `<image>` tokens when `vision.use_image_special_tokens=false`). SGLang preprocessed clients send those ids in `padded_input_ids`; without runner inject the engine would double-encode placeholders the renderer skipped.

**Inject:** one `EncodeMultimodal` per image block (or per contiguous `<image>` run); `PostTokenize` expands thumbnail + grid markers from model metadata. Token ids resolved from vocabulary at sequence start (`lfm2VisionTokens`).

### 34. GLM-OCR padded inject

**Where:** `server/modality/build_padded_prompt.go` (`BuildGlmocrPaddedCompletionPromptTokens`), `runner/ollamarunner/padded_glmocr.go`, `model/renderers/glmocr.go`.

**Why:** GLM-OCR (`glmocr`) is `OllamaEngineRequired`. SGLang clients send pretokenized image_start + vision token runs + image_end in `padded_input_ids`.

**Splice:** `<|user|>\n` content until the next `<|assistant|>`, `<|observation|>`, or `<|system|>` tag.

**Inject:** block skip on start/end tokens (GGUF defaults `59256` / `59257`); `PostTokenize` expands vision embeds with M-RoPE position cache.

### 35. Mistral3 / Pixtral padded inject

**Where:** `server/modality/build_padded_prompt.go` (`BuildMistral3PaddedCompletionPromptTokens`), `runner/ollamarunner/padded_mistral3.go`, `server/prompt.go` (skip `[img-N]` on padded users).

**Why:** Mistral3/Pixtral (`mistral3`) is `OllamaEngineRequired` and uses jinja `[INST] … [/INST]` user blocks with `[IMG]…[IMG_BREAK]…[IMG_END]` vision layouts in `padded_input_ids`.

**Inject:** one `EncodeMultimodal` per image at the first `[IMG]` of each block; skip through `[IMG_END]`; `PostTokenize` expands row patches.

### 36. DeepSeek-OCR padded inject

**Where:** `server/modality/build_padded_prompt.go` (`BuildDeepseekOcrPaddedCompletionPromptTokens`), `runner/ollamarunner/padded_deepseekocr.go`, `server/prompt.go` (`[img-N]` skip).

**Why:** DeepSeek-OCR (`deepseekocr`) is `OllamaEngineRequired`. The deepseek-ocr chat template concatenates raw message content (no `[INST]` wrappers). SGLang clients send pretokenized `<image>` token runs (`128815`) in `padded_input_ids`.

**Splice:** locate each `role=user` `msg.Content` substring in rendered prompt order (after `chatPrompt` img-tag injection or skip).

**Inject:** one `EncodeMultimodal` per contiguous `128815` run; `PostTokenize` expands SAM+CLIP embeds.

### 37. Video agent infer smoke (live VLM gate)

**Where:** `scripts/video/video_agent_infer_smoke.sh`, `scripts/video/video_agent_infer_gate_report.sh`.

**Why:** `video_agent_cache_smoke.sh` live mode uses `_debug_render_only` — it proves ffmpeg/session expansion caches but **not** real vision prefill or turn-2 prefix reuse. Agents need proof that `cached_prompt_tokens` > 0 on turn 2 when the same clip + `prompt_cache_key` returns. Expand-only smoke gave false confidence on Mac ollama-engine where `llama_cache.enabled` may be false but runner input-cache still hits.

**Legs:**

| Leg | Env | Proves |
|-----|-----|--------|
| Raw video two-turn | `RUN_E2E_VIDEO_AGENT_INFER=1` | Real VLM prefill; turn-2 `cached_prompt_tokens` ≥ min (L3 subprocess or ollama-engine input cache) |
| OpenAI shape | (same run) | `usage.prompt_tokens_details.cached_tokens` wiring (advisory — different message shape) |
| Preprocessed padded | `VIDEO_AGENT_INFER_PREPROC=1` + `VIDEO_AGENT_GO_LOG` | Turn-2 `padded_input_ids` restore (`preprocessed layout session cache hit`); preproc turn-2 cache |

**Why log read is after all HTTP:** preproc legs append log lines; parsing before preproc caused false "layout cache miss" failures.

**Why `VIDEO_AGENT_INFER_SOFT=1`:** MLX-only or KV-off hosts should soft-pass when inference succeeds but `cached_prompt_tokens` is 0 — wiring check without blocking CI on cache-less configs.

**Operator:**

```bash
RUN_E2E_VIDEO_AGENT_INFER=1 VIDEO_SMOKE_MODEL=qwen3-vl:latest \
  VIDEO_AGENT_GO_LOG=/tmp/zerollama-go.log \
  ./scripts/video/video_agent_infer_smoke.sh
VIDEO_AGENT_INFER_PREPROC=1 ...  # optional padded + grid_thw leg
./scripts/video/video_agent_infer_gate_report.sh /tmp/video-agent-infer-smoke.json
```

### 24. ViT embed cache

**Where:** `runner/llamarunner/image.go` (`ImageContext`), `envconfig/config.go` (`ImageEmbedCacheSize`).

**Why:** SGLang's `enable_prefix_mm_cache` avoids re-encoding frames already seen by the vision encoder. The llamarunner `ImageContext` already has a 4-slot LRU keyed by frame byte hash (added upstream); that cache is too small for video agents resending 32-frame clips across turns. The fix: `NewImageContext` accepts a `cacheSize` argument; the server reads `OLLAMA_IMAGE_EMBED_CACHE_SIZE` (default 4) at load time. **Auto-grow (Jun 2026):** each multimodal turn grows the LRU up to `OLLAMA_IMAGE_EMBED_CACHE_MAX` (default 64) when frame count exceeds initial slots — see §28.

**Observability:** `slog.Debug("loading image embeddings from cache")`; `vision embed cache auto-grown for multimodal turn` when expanded; `vision embed cache may be undersized` when frames exceed max cap (server preflight log).

**What differs from SGLang:** SGLang caches at the *radix-attention block* level (tensor keys); zerollama caches the raw `MtmdChunk` embed slice by frame bytes. Equivalent benefit for repeat single-turn or multi-turn same-clip scenarios; no cross-request sharing.

---

## Observability (cache hits)

| Log / field | Meaning |
|-------------|---------|
| `video url fetch cache hit` | Remote container bytes reused |
| `video sample session cache hit` | Expanded frames reused for this `prompt_cache_key` |
| `video sample global cache hit` | Expanded frames reused (any session) |
| `video layout session cache hit` | `padded_input_ids` restored on agent turn 2 (single raw-video clip) |
| `preprocessed layout session cache hit` | `padded_input_ids` restored on pre-expanded turn 2 (images + video_spans) |
| `padded_input_ids runner stub` | Layout acknowledged; render still uses images (`layout_consume=deferred`) |
| `padded_layout_consume=qwen3vl_hf_runner_inject` | ggml runner consumed pretokenized ids + ViT inject at image_pad |
| `padded_layout_consume=gemma4_img_runner_inject` | ggml runner or llama-server consumed pretokenized ids + inject at `<|image|>` / `<|video|>` / `<|audio|>` soft tokens |
| `padded_layout_consume=mllama_img_runner_inject` | ollama-engine consumed pretokenized ids + inject at mllama slot `128256` |
| `padded_layout_consume=gemma3_img_runner_inject` | ollama-engine consumed pretokenized ids + inject at Gemma3 `<start_of_image>` token `255999` |
| `padded_layout_consume=llama4_img_runner_inject` | ollama-engine consumed pretokenized ids + inject at `<|image_start|>…<|image_end|>` block |
| `padded_layout_consume=lfm2_img_runner_inject` | ollama-engine consumed pretokenized ids + inject at LFM2 image_start…image_end block or `<image>` runs |
| `padded_layout_consume=glmocr_img_runner_inject` | ollama-engine consumed pretokenized ids + inject at GLM-OCR image_start…image_end block |
| `padded_layout_consume=mistral3_img_runner_inject` | ollama-engine consumed pretokenized ids + inject at Pixtral IMG…IMG_END block |
| `padded_layout_consume=deepseekocr_img_runner_inject` | ollama-engine consumed pretokenized ids + inject at DeepSeek-OCR image token runs |
| `padded_layout_consume=deferred_multimodal_history` | Splice/inject failed; render kept or restored vision placeholders for history |
| `padded_input_ids splice failed` | User span count ≠ user messages (often tool turns before fix; grep after upgrade) |
| `padded_input_ids runner inject` | llamarunner built inputs from `prompt_tokens` (not re-tokenize) |
| `padded_input_ids llama-server inject` | llama-server subprocess: vision blocks → `prompt_string` + `multimodal_data` |
| `video sample` | ffmpeg actually ran (mode, fps, frame_count) |
| `vision embed cache auto-grown for multimodal turn` | Runner expanded ViT LRU slots for this turn (≤ `OLLAMA_IMAGE_EMBED_CACHE_MAX`) |
| `vision embed session cache hit` | ViT encoder skipped — frame embed restored from `prompt_cache_key` overlay |
| `vision embed global cache hit` | ViT encoder skipped — embed restored from per-runner global LRU (PNG on llamarunner; PNG/preprocessed/processor on ollama-engine) |
| `precomputed_embedding global cache hit` | Precomputed rows skipped — materialized from global LRU (ollama-engine + llamarunner) |
| `precomputed_embedding session cache hit` | Precomputed rows skipped — restored from session overlay (llamarunner; ollama-engine uses `vision embed session cache hit`) |
| `processor_output global cache hit` | HF pixels skipped — vision tower output from global LRU (ollama-engine) |
| `vision grid hints` | Info summary after multimodal encode (`hinted`, `matched`, `mismatched`) |
| `vision grid hint match` | Client `grid_thw` embed count matches mtmd output for that frame |
| `vision grid hint mismatch` | Hint vs mtmd differ — expected until mtmd accepts grid overrides |

**Why Info for sampling:** Operators troubleshooting production need effective ffmpeg policy without enabling global debug.

---

## Deferred (not shipped)

| SGLang pattern | Why deferred |
|----------------|--------------|
| ViT / encoder cache | **Partial (Jun 2026):** global LRU + auto-grow + **session overlay** (default ON with `prompt_cache_key`; opt out via `enable_prefix_mm_cache: false`); precomputed/processor global+session on both runners; radix sharing deferred |
| Non–Qwen3-VL `padded_input_ids` runner inject | **Shipped (Jun 2026)** on ollama-engine, **llama-server subprocess**, and **ggml llamarunner** for all native Go VLMs above. Still `deferred_non_qwen3vl` for text-only families (gemma3n, glm4moelite) and mtmd-only qwen2vl when not routed through family gate |
| `grid_thw` → vision forward | **Partial (Jun 2026):** client explicit `[1,H,W]` via `GridTHWExplicit`; **`mtmd_bitmap_set_grid_hint`** on M-RoPE (Qwen-VL); server ffmpeg estimates preflight-only; non-M-RoPE families deferred |
| HiCache host/storage cache breakdown | **Partial (Jun 2026):** `sglext.cached_tokens_details` + access-log fields wired; host/storage counts populate when L3/HiCache backends report them (device-only on native paths today) |
| `limit_mm_data_per_request` | **Shipped (Jun 2026):** `OLLAMA_LIMIT_MM_DATA_PER_REQUEST` on `/api/chat` latest-user preflight |
| `precomputed_embedding` ingest | **Partial (Jun 2026):** all native ollama-engine VLMs above except glmocr requires `grid_thw`; llamarunner embed-chunk path; llama-server rejects |
| `processor_output` ingest | **Partial (Jun 2026):** Qwen3-VL/qwen25vl/glmocr/mistral3 + Gemma3/Gemma4/llama4/**lfm2** (single-tile); llamarunner/llama-server reject |
| Non-stream prefill cancel (subprocess backends) | **Shipped (Jun 2026):** ctypes + llama-server HTTP close + wheel non-stream via internal stream |
| int8 linear-attn checkpoint pool | Phase 15 / model-specific |
| Breakable CUDA graphs | Phase 15 / upstream llama.cpp |

---

## Operator checklist

1. **Agent thread with repeat video:** set `options.prompt_cache_key` (or `eliza.conversationId`) on every turn — enables session expansion cache **and** L3 KV reuse. OpenAI clients: `prompt_cache_key` on `/v1/chat/completions` or `options` on `/api/chat`. Add **`enable_prefix_mm_cache: true`** when you want SGLang-compatible session ViT pinning (overlay still requires the session key).
2. **OpenAI clients:** inspect `usage.prompt_tokens_details` for `image_tokens`, `video_tokens`, `cached_tokens`; optional SGLang clients may read `sglext.cached_tokens_details`.
3. **Fleet logs:** grep `inference response out` for `image_tokens`, `video_tokens`, `audio_tokens`, `cached_prompt_tokens`, `cached_tokens_device`.
4. **L3 path:** `curl -s :8081/health | jq .llama_cache` — prefix policy and slot stats.
5. **Optional SGLang:** unchanged — full-body proxy when `video_understanding=sglang`; native path does not require it.

6. **CI gate:** `./scripts/video/video_expand_cache_smoke.sh` — unit tests for expansion/session/URL caches and preflight (no GPU).
7. **Agent loop:** `./scripts/video/video_agent_cache_smoke.sh` — two-turn resend-clip test; live E2E with `RUN_E2E_VIDEO_AGENT=1` + `VIDEO_SMOKE_MODEL`.
8. **Video + L3 gate:** `./scripts/video/video_l3_agent_gate.sh`; add `RUN_E2E_L3=1` for L3 text smoke on GPU hosts.
9. **Video + inference cache:** `RUN_E2E_VIDEO_AGENT_INFER=1 ./scripts/video/video_agent_infer_smoke.sh` — real VLM prefill; strict pass needs turn-2 `/api/chat` `cached_prompt_tokens` (L3 subprocess or ollama-engine input cache). Optional `VIDEO_AGENT_INFER_PREPROC=1` + `VIDEO_AGENT_GO_LOG` for padded preproc infer (§37). Optional `VIDEO_AGENT_INFER_PREFIX_MM_WARN=1` + `VIDEO_AGENT_GO_LOG` greps prefix-mm hint without session key. `./scripts/video/video_agent_infer_gate_report.sh` for sign-off. Grep logs: `vision embed session cache hit`, `vision grid hints`, `preprocessed layout session cache hit`, `precomputed_embedding runner inject`, `processor_output runner inject`.
10. **Pre-expanded layout:** live `RUN_E2E_VIDEO_AGENT=1` also checks turn-2 `images` + `video_spans` restore (`preprocessed layout session cache hit` in `VIDEO_AGENT_GO_LOG`).
11. **Preprocessed ingest:** send `precomputed_embedding` or `processor_output` with `padded_input_ids` on ollama-engine VLMs; grep inject log lines above. llama-server and llamarunner processor paths reject — use PNG or precomputed embed chunks on ggml path.

---

## Code map

| Concern | Primary files |
|---------|----------------|
| URL fetch + SSRF | `openai/video_url.go` |
| URL body LRU | `openai/video_fetch_cache.go` |
| ffmpeg expansion | `server/modality/video_frames.go` |
| Global expand LRU | `server/modality/video_cache.go` |
| Session expand LRU | `server/modality/session_video_cache.go` |
| Token estimates | `server/modality/token_budget.go`, `server/modality/preflight.go`, `server/modality/preprocessed.go` |
| Grid layout | `server/modality/vision_grid_compute.go`, `server/modality/vision_grid_tokens.go`, `server/modality/grid_thw_raster.go`, `runner/llamarunner/grid_thw_hint.go`, `runner/ollamarunner/grid_thw_hint.go` |
| Video payload detection | `server/modality/multimodal.go` (`ChatRequestHasVideoPayload`) |
| Policy golden tests | `server/modality/video_policy_golden_test.go` |
| CI smoke | `scripts/video/video_expand_cache_smoke.sh` |
| Agent + L3 gate | `scripts/video/video_agent_cache_smoke.sh`, `scripts/video/video_l3_agent_gate.sh`, `scripts/video/video_agent_infer_smoke.sh`, `scripts/video/video_agent_infer_gate_report.sh` |
| mtmd grid_thw seam | `llama/llama.go` (`MultimodalTokenize`), `runner/llamarunner/image.go`, `docs/mtmd-grid-thw-handoff.md` |
| Agent smoke | `scripts/video/video_agent_cache_smoke.sh` |
| OpenAI agent session test | `openai/video_agent_session_test.go` |
| Qwen3-VL span render tests | `model/renderers/qwen3vl_video_test.go` |
| Padded prompt splice + runner inject | `server/modality/build_padded_prompt.go`, `server/modality/padded_layout_consume.go`, `runner/llamarunner/padded_inputs.go`, `runner/llamarunner/padded_families.go`, `runner/ollamarunner/padded_inputs.go`, `runner/ollamarunner/padded_{lfm2,glmocr,mistral3,deepseekocr}.go`, `llm/padded_prompt_llama_server.go` |
| Padded inject audit (tool spans, fallback) | `server/ggml_padded_prompt.go`, `model/renderers/qwen3vl.go`, `server/routes.go` |
| ViT embed cache | `runner/llamarunner/image.go`, `envconfig/config.go` (`ImageEmbedCacheSize`), `server/modality/vit_embed_cache.go` |
| Session ViT embed overlay | `runner/llamarunner/image.go` (`MultimodalTokenize`), `runner/ollamarunner/vision_embed_cache.go`, `llm/server.go` (`PromptCacheKey`), `server/modality/session_video_cache.go` (`ExtractPromptCacheKey`), `server/modality/prefix_mm_cache.go` |
| Precomputed ingest | `server/modality/precomputed_embedding.go`, `api/preprocessed_parse.go`, `runner/ollamarunner/precomputed_embedding.go`, `runner/llamarunner/precomputed_embedding.go`, `model/models/*/precomputed.go` |
| Processor output ingest | `server/modality/processor_output.go`, `api/types.go` (`ProcessorOutput`), `runner/ollamarunner/precomputed_embedding.go` (`appendPaddedProcessorOutputImage`), `model/models/*/processor_output.go` |
| llama-server HTTP cancel | `runtime/runtime/worker/llama_server_http.py`, `runtime/runtime/worker/llama_server.py` |
| Chat wiring | `server/routes.go` |
| OpenAI usage | `openai/openai.go` |
| llama-server timings | `llm/llama_server.go` |
| Runtime timings | `runtime/runtime/llama_timings.py`, `runtime/runtime/engine.py` |
| Access log | `server/inference_access_log.go` (`cached_prompt_tokens`, `image_tokens`, `video_tokens`, `audio_tokens`) |
