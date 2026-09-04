# mlx-serve borrowings

**Upstream:** [ddalcu/mlx-serve](https://github.com/ddalcu/mlx-serve) · sibling `../mlx-serve`  
**Last checked:** 2026-08-24 (README + `src/pld_index.zig` + performance/cli docs)

Zig + mlx-c Mac server. **Do not vendor Zig or the MLX Core app.** Steal engine defaults we can land in `x/mlxrunner`.

## Brought

| Item | Where | Why |
|------|--------|-----|
| **PLD host-read Eval** | `mlx.Array.Ints`/`Int` | Parked decode `ExpandDims` tokens are lazy; reading without Eval SIGSEGV'd the runner (`qwen3:0.6b-mlx`). |
| **Runtime acceptance gate** | `speculationSession.applyRuntimeGate` | Sticky disable: PLD accept &lt;0.30 after 5 rounds; MTP &lt;0.70 after 8. Novel decode stays at plain AR. |
| **Decode-only attn quant** | `x/models/nn/decode_quant.go`, `MakeLinearLayer` | Dense bf16/fp16/fp32 **Q/K/V/O** get a 4-bit copy used only when L=1. Prefill + speculative fused forwards stay dense. Quantized checkpoints unchanged. |
| **SWA spec-read trim** | `x/mlxrunner/cache/rotating.go` | Batched attention reads `window+L-1` keys (mlx-serve `slidingViewFor`). Prefill and fused spec share the same view. |
| **Draft companion layouts** | `readDraftConfig` | Auto-detect `draft/`, `drafter/`, `assistant/` config layers (mlx-serve folder names). |
| **Paged KV FP8** | `cache/kv_pack.go` | Idle trie snapshots pack to FP8 when large; live decode KV stays dense (no fused kv-attn kernel yet). |
| **Round-cost persist** | `round_cost.go` | Spec width (cost + acceptance) per **context bucket**, under `mlx-round-cost/` next to `OLLAMA_MODELS`; next Load starts tuned. |
| **Last-run persist** | `last_run.go` | Last 8 decodes in `*.last.json` (accept, PLD park, hint, ctx). Doctor + `GET /v1/status` `Tune` surface it; one parked novel chat is not a warning. |
| **Loop-stop** | `loop_stop.go` | Triple repeat of an 8–48 token cycle ends decode as length (`repetition_loop` log). |
| **Draft load warn** | `loadDraftCompanion` | Broken in-manifest companion logs and keeps PLD/AR; `ZEROLLAMA_MLX_MTP=require` still fails closed. |
| **Vision engine pixel cap** | `imageproc.EngineMaxPixels` | 1536² before the VL tower (Qwen 2.5/3, GlmOCR). Pack 16 Mpx bounds ignored; 1080p screenshots pass. |
| **Prefill working-set cap** | `capPrefillChunkForWorkingSet` | When active RAM is ≥75% of recommended, auto 1024-token chunks + per-chunk eval/clear. |
| **Split fused-verify SDPA** | `nn.runFastSDPA` | qL&gt;8 at hd 192/256 (or qL×GQA&gt;32) splits into vector-kernel passes so MLX does not take the slow unfused path. |
| **Continue final message** | `template.ApplyContinueFinal` | `/v1` + `/api/chat` `continue_final_message`: prompt ends on trailing assistant text (no new turn). Tool-call replies ignored. `/v1/messages` infers the same (mlx-serve). |
| **Draft-head 4-bit at load** | `quantizeDraftCompanion` | Dense bf16/fp16 MTP/assistant linears become affine 4-bit; target still verifies. `ZEROLLAMA_MLX_DRAFT_QUANT=off` keeps checkpoint dtype. |
| **Spec depth +1** | `depthSearchPastFrontier=2` | After split-SDPA verify, EV search may pick two past the trusted frontier (mlx-serve “one position deeper”). |
| **`/v1/models` context** | `openai.ToListCompletion` / `ToModel` | Top-level `context_length` + `max_model_len` + `model_max_tokens` twin `meta.context_length` (oh-my-pi / mlx-serve). |
| **`/v1/models` capabilities** | `attachOpenAIModelCaps` | mlx-serve names: `chat` / `tool_use` / `streaming` / `vision` / `reasoning` / `json_schema` / `embeddings`, plus `input_modalities` and `meta.architecture`. |
| **`/v1/models` `supports_mtp`** | `ToListCompletion` / `ToModel` | True when GGUF has nextn, the manifest has `mtp/config.json`, or MLX in-weight `mtp.*` / `num_nextn_predict_layers`. Top-level and `meta.supports_mtp`. Gemma `drafter/` stays false. |
| **Anthropic `stop_sequence` echo** | `ToMessagesResponse` | Matched request stop is `stop_reason: stop_sequence` plus the string (stream `message_delta` too). OpenAI `finish_reason` stays `stop`. |
| **Responses / chat JSON format** | `formatFromStructuredOutput` | `json_object` → `"json"` on chat **and** `/v1/responses`. Flat `schema` or nested `json_schema.schema` both apply. |
| **Loop-stop signal** | `loop_stop.go` + `finish_details` | Exact triple cycle **or** near-repeat tail; OpenAI `finish_reason` stays `length`, `finish_details.type=repetition_loop`. |
| **Always `cached_tokens`** | `openai.promptTokensDetailsFromMetrics` | usage always includes `prompt_tokens_details.cached_tokens` (0 if miss). |
| **`n>1` 400** | `rejectNBest` | Chat and `/v1/completions`. mlx-serve same. `n=1` or omit is fine. Extra-body `n` counts. |
| **`parallel_tool_calls`** | `FilterParallelToolCalls` | `false` keeps the first tool call. OpenAI `/v1` chat + Responses; Anthropic `disable_parallel_tool_use`. Omit/true unchanged. |
| **Tool autocorrect** | `CoerceToolCalls` | Default-on schema type coerce + hoist a required arg buried in a nested object (mlx-serve). `ZEROLLAMA_TOOL_AUTOCORRECT=0` passes args through. |
| **Completions `echo`** | `CompleteWriter.applyEcho` | Prepends `prompt` to generated text (stream + non-stream). |
| **`logit_bias`** | `sample.SetSlotLogitBias` | OpenAI map of token-id → additive logit. MLX draw + llama-server `logit_bias` pairs. mlx-serve still unparsed. |
| **Jinja include stub** | `convert.ChatTemplateIncludeStubPath` | `{% include 'chat_template.jinja' %}` loads the sidecar (transformers ≥5). |
| **Overflow names both counts** | `llm.ContextOverflowMessage` | 400 body has prompt tokens and window. |
| **`POST /api/load` / `/api/unload`** | `LoadHandler` / `UnloadHandler` | Native prewarm + expire. LocalAI `POST /backend/load` only. No mlx-serve `/v1/load-model` aliases. |
| **Loop-stop trim** | `applyLoopTrim` | Non-stream drops the triple 8–48-rune cycle; `ZEROLLAMA_MLX_LOOP_TRIM=0` keeps it. |
| **Logprob UTF-8** | `utf8LogprobToken` | BPE fragments become U+FFFD in `token`; `bytes` stays exact. |
| **HF `generation_config` sampling** | `api.SamplingMapFromGenerationConfig` | Omitted request fields use the checkpoint's temperature/top-k/top-p (then PARAMETER, then request). `/v1` chat, completions, and Responses honor the same sampling keys (`top_k`, `min_p`, `typical_p`, `repetition_penalty`/`repeat_penalty`). `max_completion_tokens` aliases `max_tokens`. MLX `sample.TypicalP` is llama.cpp locally typical sampling (`1.0` = off). |
| **Per-request PLD/MTP** | `enable_pld` / `enable_mtp` / `enable_drafter` | Body flags on `/v1` chat, completions, Responses, and `/api/chat`/`generate`. `enable_drafter` is the Gemma companion name for the same draft slot as `enable_mtp`. Nil keeps process defaults. **MoE checkpoints park MTP** unless the request sets `enable_mtp`/`enable_drafter` (mlx-serve `defaultEnableMtp`). |
| **MLX family top_p** | `modelOptions` | 0.95 when the checkpoint and request omit top_p (mlx-serve applyFamilySamplingDefaults). |
| **Join text parts** | `openai.joinedTextParts` | Content-array `text` parts concatenate in order (#195). |
| **`n>1` 400** | `rejectNBest` | Chat and `/v1/completions`. Extra-body `n` counts. |
| **Responses `background` 400** | `FromResponsesRequest` | `background:true` is 400 (no async store). Omit or false. |
| **Responses store chain 400** | `rejectUnsupportedResponsesFields` | `previous_response_id`, `conversation`, and `truncation=auto` 400 (no ResponseStore; overflow is a named 400). |
| **Responses `include` / `max_tool_calls`** | `rejectUnsupportedResponsesFields` | Non-empty `include` 400 (no encrypted reasoning / file_search extras). `max_tool_calls` &lt;1 is 400; `=1` keeps one tool call. Usage `input_tokens_details.cached_tokens` is the prefix-cache hit (0 on miss). |
| **Embeddings `dimensions` overflow** | `applyEmbeddingDimensions` | `dimensions` larger than the vector is 400 (OpenAI truncate+renorm only when smaller). |
| **MLX `stop`** | `flushStopHold` | Request `stop` sequences halt MLX decode (same FindStop/hold as ollamarunner). Stop text is not streamed. |
| **Reserved sample ban** | `reservedSampleBanIDs` | FIM/reserved/pad/bos ids get −inf at sample time (mlx-serve `reservedOutputIds`). Think/tool/`im_end` stay legal. Logprobs stay raw. `ZEROLLAMA_MLX_SUPPRESS_RESERVED=0` disables. |
| **`mtp/` companion** | `draftCompanionLayouts` | In-manifest `mtp/config.json` (Qwen 3.6 mlx-serve packs). |
| **Qwen in-checkpoint MTP** | `qwen3_5.mtpHead` | Keep `mtp.*` weights (enorm/hnorm/fc + full-attn layers). `SelfDraft` + extra KV slots. MoE still parks unless `enable_mtp`. |
| **Long-period loop-stop** | `loopLongRepeats` | Exact 10× cycle of period 9–64 tokens (mlx-serve `isDegenerateTailLoopRange`). Triple 8–48 still fires first. |
| **FIM completions** | `template.WrapFIM` | `/v1/completions` `suffix` on prompt-only templates (typical MLX) becomes Qwen FIM markers. MLX tags advertise `insert`. |
| **Stream lead indent** | `streamLeadHold` | All-whitespace first SSE chunk is held and prepended to the first real text (mlx-serve `streamContentLead` / FIM indent). |
| **LTX last-frame 400** | `validateLtxMLXLastFrame` | mlx-serve #260: `last_frame_image` or two stills on ltx-mlx is 400 (CLI `--image` only). `first_frame_image` aliases `keyframes[0]`. |
| **Non-text on chat/generate** | `textSurfaceWrongModalityMessage` | Embedding (etc.) on `/api/chat` names the kind and the endpoint (`/v1/embeddings`). Same for generate. |
| **Image edits unhonored fields** | `rejectUnsupportedImageOpenAI` | `mask`, `n>1`, `response_format=url`, `stream:true` are 400 (mlx-serve named 400s). `n=1` / omit / `b64_json` OK. |
| **Responses `[DONE]`** | `ResponsesWriter` | Stream ends with `data: [DONE]` after `response.completed` (proxies that key off the chat sentinel). |
| **Leaked tool markup** | `trimLeakedToolMarkup` | Unparsed `<tool_call>` / LFM2 `<\|tool_call_start\|>` / Gemma `<\|tool_call>` / FunctionGemma `<start_function_call>` / Olmo `<function_calls>` / ATEM / Harmony wrappers are cut from assistant text so they never ride as content. |
| **Glimmer invoke name salvage** | `glimmerTrailingIdentifier` | ATEM `name=` keeps the trailing identifier (error-echo prefixes dropped). A leftover `<\|` never becomes the tool name. |
| **`reasoning_budget_tokens`** | `thinkFromReasoningBudget` | 0 off, &gt;0 on. Outranks `reasoning_effort` / `enable_thinking`. Native `think` still wins. Not a `num_predict` cap. `/v1` chat + Responses (incl. extra_body). |
| **Named `tool_choice`** | `applyToolChoice` | `none` omits tools. Named function (chat `{function.name}` / Responses `{name}` / Anthropic `type:tool`) keeps that tool. `required`/`any` keep the list. Unknown name is 400. |
| **Legacy `functions`** | `applyLegacyFunctions` | OpenAI `functions` / `function_call` map onto `tools` / `tool_choice` when those are empty. Extra-body copies fold. Tools win if both are set. |
| **Completions `best_of>1`** | `rejectBestOf` | Same as `n>1` (no n-best). `best_of=1` or omit is fine. |
| **Truncated tool JSON** | `salvageJSONToolCall` | Mangled Qwen `<tool_call>` JSON keeps the name and ships `{}` (never partial args). OpenAI emit/ingest empty arguments as `{}`. |
| **Length beats tool_calls** | `chatFinishReason` | `done_reason=length` stays `finish_reason=length` even if a tool call was parsed (mlx-serve). Anthropic `max_tokens`. Responses `status=incomplete` + `max_output_tokens`. |
| **Think-tag leaks** | `trimThinkTagLeaks` | Trailing `</think>` and a pos-0 unclosed `<think>` are stripped from content and thinking. Tool markup is also cut from thinking. |
| **`</think>` in tool args** | `thinkCloseIsToolCallPayload` | A closer inside an unclosed `<tool_call>` (etc.) stays in the argument. Real `</think>` still ends thinking. |
| **Think-off prompt suffix** | `ApplyNoThinkTailSuffix` | Think-off Muse commits ` to=user<|message|>` (no tools). Unclosed `<think>` is closed from the rendered bytes. `lfm2-thinking` gets `<think></think>`. Parsers seed from those bytes (`promptOpensThink`). |
| **Muse drop prior reasoning** | `GlimmerRenderer` | Only assistant turns after the last user keep `to=self` history (mlx-serve `dropPriorTurnReasoning`). |
| **`service_tier` 400** | `rejectServiceTier` | `flex` / `scale` / `priority` 400. omit / `auto` / `default` OK. Chat, completions, Responses. Extra-body copies fold. |
| **`store:true` 400** | `rejectStore` | Chat and Responses. omit / `false` OK. Extra-body copies fold. |
| **`tool_choice` required hint** | `AppendRequiredToolCallHint` | `required` / `any` / named function append a last-user “must invoke a tool” line (prompt-side; no grammar). `auto` / `none` unchanged. |
| **Tight output budget** | `AppendOutputBudgetGuidance` | When `num_predict`/`max_tokens` is set and &lt; 12288, last user gets a concise-reply line (mlx-serve `outputBudgetGuidance`). Omit / −1 / ≥12288 unchanged. |
| **XML tool dialect first** | `qwenXMLFunctionFirst` | A `<tool_call>` body with `<function=` is parsed as Qwen 3.5 XML before JSON, so a package.json in a parameter cannot become the tool name. |
| **LFM2 truncated pythonic** | `LFM2Parser` | Marker-gated call without `)` / end tag ships **name + `{}`** (never partial args). Complete args without `<|tool_call_end|>` still parse on done. |
| **Missing `</tool_call>`** | Qwen3 / 3.5 / VL / coder | On `done`, an open `<tool_call>` body is still parsed (mlx-serve delimiter-drop). Mid-stream stays buffered. |
| **FunctionGemma missing close** | `FunctionGemmaParser` | On `done`, `<start_function_call>` without `<end_function_call>` still parses `call:name{args}` (optional trailing `}`). |
| **DeepSeek missing tool end** | `DeepSeek3Parser` | On `done` (or `tool▁calls▁end` without `tool▁call▁end`), an open call still parses. Truncated JSON ships **name + `{}`**. |
| **Cogito missing tool end** | `CogitoParser` | Same delimiter-drop as DeepSeek; truncated fenced JSON ships **name + `{}`**. |
| **Ministral truncated `[ARGS]`** | `MinistralParser` | On `done`, incomplete JSON after `[ARGS]` ships **name + `{}`**. |
| **Olmo3 missing `</function_calls>`** | `Olmo3Parser` | On `done`, an open `<function_calls>` body still parses (last-chunk `done` used to drop the call). Truncated `name(args` ships **name + `{}`**. Unparsed `<function_calls>` is cut from assistant text. |
| **GLM missing `</tool_call>`** | `GLM46Parser` / 4.7 | On `done`, an open XML `<tool_call>` still parses. Complete args keep values; truncated / schema-invalid bodies ship **name + `{}`** (never partial `arg_value`). The turn does not 500. |
| **Cohere truncated `<\|START_ACTION\|>`** | `CohereParser` | On `done`, incomplete action JSON ships **tool_name + `{}`**. Unparsed `<\|START_ACTION\|>` is cut from assistant text. |
| **History tool-call `id`** | `api.EnsureToolCallID` | Empty history / OpenAI / llama-server tool-call ids become stable `call_N` (mlx-serve extra-context). Client-supplied ids are kept. Template args stay objects for ranging. |
| **Stream logprobs on content** | `ToChunks` / `contentLogprobs` | Mixed thinking+content SSE puts `logprobs` on the content/tool chunk, never on reasoning. Empty content (reasoning-only) drops them (`dropPending`). Completions use the four-array OpenAI shape (`tokens` / `token_logprobs` / `top_logprobs` / `text_offset`). Native + chat entries include vocab **`id`** when known. Completions **`logprobs: 0`** still returns chosen-token logprobs; **&lt;0 or &gt;5** is 400. |
| **`<tool_calls:` wrapper** | Qwen XML coder / 3.5 | Colon-suffixed plural opener only (hy3). `<tool_calls>` without `:` stays content. |
| **Inferred JSON tool names** | `LagunaParser` | Bare `{name, arguments}` only becomes a call if the name is **declared** (mlx-serve `filterInferredBySchema`). Tagged `<tool_call>` is never filtered. |
| **Gemma dropped `<\|"\|>`** | `quoteGemma4BareValues` | Unquoted values run to the next `key:` or the object's final `}`, never the first `,`/`}` inside markup. JSON scalars stay typed. Truncated `call:name{…` on `done` ships **name + `{}`**. |
| **Laguna truncated tagged JSON** | `LagunaParser` | Unclosed / mangled `<tool_call>` JSON ships **name + `{}`**. Standalone inferred JSON still requires a declared name. |
| **Glimmer truncated ATEM** | `GlimmerParser` | Missing `</atem:invoke>` / `</atem:function_calls>` still parses. Completed `<atem:parameter>` pairs are kept; a truncated last param is dropped. Non-ATEM tool bodies salvage the recipient name + `{}`. |
| **Harmony truncated tool JSON** | `HarmonyMessageHandler` | On `done`, invalid tool JSON ships **name + `{}`** (the turn does not 500, and prior content is kept). |

Prompt n-gram score parks PLD on novel text; if the generated tail later echoes the prompt (`tailMatchFraction` > 0.7 over 16 tokens) it re-enables. A runtime-frozen stretch never comes back. Escape hatch: `ZEROLLAMA_MLX_PLD=off`.

`ZEROLLAMA_MLX_MTP=require` still means a **draft head**, not PLD.

## Knobs (how to tune)

Defaults are meant to be left alone. `zerollama doctor` prints the sheet (effective value + when to touch each). Decode logs `mlx tune` / `tune=` on runtime gates.

| Env | Default | Touch when |
|-----|---------|------------|
| `ZEROLLAMA_MLX_PLD` | on | `off` for AR benches only |
| `ZEROLLAMA_MLX_MTP` | auto | `require` to refuse load without a draft head |
| `ZEROLLAMA_MLX_DRAFT_QUANT` | on | `off` if draft accept collapses vs the checkpoint dtype |
| `ZEROLLAMA_MLX_MTP_HISTORY` | committed | `auto` / `last_window` if long-prompt draft RAM spikes |
| `ZEROLLAMA_MLX_MTP_HISTORY_LAST_WINDOW` | 8192 | raise if MTP accept drops on long chats |
| `ZEROLLAMA_MLX_GREEDY_DRAFT_CHAIN` | on | `off` to A/B T=0 draft; sampled requests ignore it |
| `ZEROLLAMA_MLX_BATCHED_GREEDY_ACCEPT` | on | `off` to A/B T=0 accept vs Bernoulli p/q |
| `ZEROLLAMA_MLX_GREEDY_TRIO_MAX_CONTEXT` | 12288 | `0` unlimited; mtplx saw a loss at 16k/32k |
| `OLLAMA_MLX_PREFILL_CHUNK` | auto | pin a bench; unset to keep working-set / SWA caps |
| `OLLAMA_MLX_PREFILL_SNAPSHOT_INTERVAL` | auto | `0` disables MTP interior snaps |
| `OLLAMA_MLX_PREFILL_CLEAR_CACHE_EVERY` | 4 | `1` if peak RAM climbs |
| `OLLAMA_MLX_PREFILL_MATERIALIZE_EVERY` | 4 | `1` on prefill OOM |
| `ZEROLLAMA_MLX_SUPPRESS_RESERVED` | on | `0` if a FIM bench must emit hole/pad tokens |

Learned spec width lives in `mlx-round-cost/` next to `OLLAMA_MODELS`, **keyed by context bucket** (2k/4k/8k/16k/32k) so a short chat does not overwrite a long-prompt table. Doctor warns if a table is stuck at scheduled=0. After each request, `*.last.json` keeps the last 8 decodes; doctor warns only when parking is the usual path (not a single novel chat). Prefill chunk shrinks also log `mlx tune`.

## Next (research, not started)

| Item | Note |
|------|------|
| Gemma 4 HF sibling auto-download (`*-drafter`) | We load if the companion is **in the manifest**; no Bonjour/app auto-fetch |
| Live MLX KV quant 4/8 | Default off upstream; needs fused packed SDPA or decode regresses |
| Same-weight bench vs mlx-serve / LM Studio | lab `:11435` vs their `:11234` |
| LTX-Video 2.5 4-bit MLX pack | vs our LTXV 0.9.8 / 2B MLX |

## Skip

Menu-bar app, Bonjour LAN share, Telegram, Linux VM sandbox, Hunyuan3D, rewriting mlxrunner in Zig. Remote store already covers LAN weights (`docs/remote-model-storage.md`).
