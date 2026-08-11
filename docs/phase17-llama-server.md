# Phase 17 — Go → llama-server (upstream GGUF path)

Zerollama’s **Mac default** remains **in-process ggml Metal**. Phase 17 ports upstream Ollama’s **Go → llama-server** stack so plain text GGUF can run in upstream shape without the Python runtime hop — while keeping training, admission, fleet, and Phase 15 experiments.

See [upstream-ollama-diff.md](./upstream-ollama-diff.md) and [ROADMAP.md](./ROADMAP.md#phase-17--upstream-gguf-path-alignment-directional).

---

## Why Phase 17 exists

Vanilla [ollama/ollama](https://github.com/ollama/ollama) deleted the in-process ggml runner for text GGUF and routes everything through **Go → llama-server**. Zerollama forked earlier and invested in:

- **ggml Metal** as the Mac hot path (~**+7% decode** vs upstream llama-server on M4 Max)
- **Python runtime** for PagedAttention, training handoff, and admission policy
- **Eliza cloud**, fleet scheduling, and native KV experiments

Rebasing wholesale would lose those differentiators. Phase 17 instead **cherry-picks upstream integration pieces** (llama-server wrapper, discovery, renderer parity, pin alignment) so future merges stay cheap — **without** flipping Mac default or deleting `runtime/`.

```text
Upstream default:     Client → Go → llama-server → libllama
Zerollama Mac default: Client → Go → ggml Metal (ollama-engine)
Zerollama Phase 17:    Client → Go → llama-server  (--llama-server-backend / Linux auto)
Zerollama harness:     Client → Go → Python runtime → llama-server  (--llama-cpp-backend)
```

---

## Status

| Item | State | Why it matters |
|------|--------|----------------|
| `llm/llama_server.go` | **Done** | Upstream-shaped subprocess runner; DisableJinja, context shift, MTP |
| `LeadingBOSForRenderer` | **Done** | llama-server with `--no-jinja` must not double-emit BOS tokens Go already rendered |
| `discover/llama_server.go` | **Done** | CUDA arch + ROCm gfx filtering matches upstream scheduler inputs |
| Linux auto-default | **Done** | Plain text + vision GGUF when `ZEROLLAMA_LLAMA_SERVER=auto` (Linux serve default) |
| Mac default | **Unchanged (ggml)** | M7 bench: ggml ~166 vs llama-server ~155 tok/s @ 4k ctx |
| `LLAMA_CPP_VERSION=b9781` | **Done** | Vendor + in-tree sync @ b9781 (Jun 2026) |
| Native `gpu-discover` | **Done** | Enriches llama-server probe with PCI/CC/gfx from crash-isolated subprocess |
| Integrated GPU (`gfx1151`) | **Done** | Strix Halo 8060S on allowlist; `OLLAMA_IGPU_ENABLE` for others |
| Metal discovery retry | **Done** | Retries with `GGML_METAL_TENSOR_DISABLE=1`; persists via `RunnerEnvOverrides` |
| `/api/status` `inference.backend` | **Done** | Fleet + operator visibility: `llama_server`, `gguf_path`, `edge`, `runtime_chat` |
| `/api/version` `edge_build` | **Done** | Compile marker for `-tags edge` / `build_zerollama_edge.sh`; CLI `zerollama -v` |
| `/api/status` `inference.backend.ggml_linked` | **Done (v1–v2)** | `false` for `-tags edge`; v2 drops in-process ggml CGO from edge link |

---

## Enable the path

```bash
# Build llama-server into zerollama tree (Metal)
./scripts/build_ollama_llama_server_darwin.sh

# Build zerollama
./scripts/build/build_zerollama_mac.sh

# Serve (sets ZEROLLAMA_LEGACY_RUNNER=1 so Python runtime does not steal text GGUF)
./scripts/serve_llama_server_backend.sh
# equivalent:
./zerollama serve --llama-server-backend
```

**Linux auto (plain serve, no flag):**

```bash
./scripts/build_llama_server.sh
./scripts/serve_linux_auto.sh
# equivalent on Linux when llama-server is discoverable:
./zerollama serve
```

Smoke (Linux auto policy): `./scripts/phase17_linux_auto_smoke.sh` or `RUN_E2E_P17_LINUX_AUTO=1 ./scripts/gpu_5080_session.sh`

**Edge-marked binary (`-tags edge`, Phase 16 v1):**

```bash
./scripts/build_zerollama_edge.sh
./zerollama-edge serve --edge   # or plain serve — edge defaults apply
./zerollama-edge -v             # prints edge build: true
```

Edge builds set `inference.backend.ggml_linked=false`, stub `zerollama runner`, and require llama-server for GGUF. **v2 (Jun 2026):** `-tags edge` also excludes in-process ggml CGO (`llm/server.go`); edge main dep tree has no `llama`/`model` packages. CI: `./scripts/phase16_edge_build_smoke.sh`. Doc: [phase16-thin-edge.md](./phase16-thin-edge.md).

Binary discovery (`llm.FindLlamaServer`) checks, in order:

- `LLAMA_SERVER_BIN` if set
- `build/llama-server-darwin/bin/llama-server` (relative to executable or cwd)
- Packaged `lib/ollama/` layouts

Reuse upstream’s build if you already have it:

```bash
BUILD_UPSTREAM_GO=0 ./scripts/build_upstream_ollama_mac.sh
LLAMA_SERVER_BIN=../ollama-upstream/build/llama-server-darwin/bin/llama-server \
  ./zerollama serve --llama-server-backend
```

Smoke: `./scripts/phase17_llama_server_smoke.sh` (requires pulled local tag — auto-resolved from smallest text GGUF blob, or set `P17_MODEL=your-tag:latest`)

**Vision/thinking:** with `--llama-server-backend` or Linux `auto`, all GGUF (split mmproj, inline vision tensors `v.*`, thinking) routes through `NewLlamaServerRunner`. Split mmproj needs `--mmproj`. Thinking models use `enable_thinking` chat-template kwargs. Linux auto sends vision GGUF to llama-server the same as text — no plain-text-only restriction.

Vision E2E (opt-in, needs projector model pulled):

```bash
RUN_E2E_P17_VISION=1 P17_VISION_MODEL=llava:latest ./scripts/phase17_llama_server_vision_smoke.sh
```

---

## Routing policy

| Flag / env | Text GGUF path | Why |
|------------|----------------|-----|
| (default Mac) | ggml Metal | Faster on ship hardware; sidecar stays for tokenize/VRAM |
| Linux serve (default) | Go → llama-server (`auto`) | All GGUF when binary found — upstream parity |
| `--llama-server-backend` or `ZEROLLAMA_LLAMA_SERVER=1` | Go → llama-server (all GGUF) | Explicit opt-in on any OS: text, vision, thinking |
| `ZEROLLAMA_LLAMA_SERVER=0` | Disable auto + explicit | Operators forcing ggml |
| `--edge` / `ZEROLLAMA_EDGE=1` | Go → llama-server + runtime chat off | Phase 16 upstream-shaped edge — [phase16-thin-edge.md](./phase16-thin-edge.md); scheduler rejects ggml when llama-server off |
| `--llama-cpp-backend` | Go → Python → llama | Phase 12–15 harness; not long-term default |
| Vision/thinking without explicit flag (Mac) | ggml or Python | Mac stays ggml default; Linux auto includes vision |

`ApplyLlamaServerBackendDefaults()` sets `ZEROLLAMA_LEGACY_RUNNER=1` when unset so Darwin sidecar runtime routing does not intercept eligible text models.

`OLLAMA_NEW_ENGINE` is **deprecated** (no longer affects engine routing) — use `ZEROLLAMA_LLAMA_SERVER`, `--edge`, or Linux `auto`.

---

## GPU discovery (hybrid)

**Why upstream replaced ggml bootstrap:** llama-server prints compiled CUDA archs, ROCm gfx targets, and free VRAM on startup — data the ggml runner subprocess never exposed consistently.

**Why zerollama did not wholesale-replace bootstrap:** Mac default still uses ggml; spawning llama-server on every Mac boot when Phase 17 is off added latency and broke tests when a binary happened to exist on disk.

| Condition | Discovery path |
|-----------|----------------|
| `llama-server` on disk **and** (`GOOS=linux` **or** `ZEROLLAMA_LLAMA_SERVER=1`) | `discover/llama_server.go` (15s timeout) |
| Otherwise | ggml ollama-engine `/info` bootstrap (Mac default) |
| llama-server fails or returns empty | Fall back to ggml bootstrap |

ROCm devices are filtered against bundled rocBLAS gfx targets. CUDA devices are skipped when compute capability is absent from the compiled arch list — **why:** scheduling an unsupported GPU guarantees a crash at first load, not a clean 503.

Native `gpu-discover` runs in a short-lived **`zerollama gpu-discover`** subprocess (Linux/Windows CGO). **Why:** llama-server stderr alone lacks reliable PCI IDs and driver versions; merging native probe JSON avoids mis-identifying duplicate GPUs across CUDA and GGML enumerations.

Mac skips native probe (stub) — Metal discovery uses llama-server stdout + ggml fallback.

---

## LeadingBOS (DisableJinja parity)

**Problem:** Models with Go renderers (Gemma4, LFM2, Cogito, …) use `DisableJinja` on the llama-server path. The rendered prompt often **includes** the BOS token textually. llama-server would prepend BOS again → duplicated token, broken tool/thinking parsers.

**Fix:** Each renderer implements `LeadingBOS()`. Routes pass `CompletionRequest.LeadingBOS` so llama-server knows what Go already emitted.

| Renderer | Leading BOS | Why non-empty |
|----------|-------------|---------------|
| `gemma4*` | `<bos>` | Gemma chat templates expect explicit BOS in prompt |
| `lfm2*` | `<\|startoftext\|>` | LFM2 vocab uses startoftext, not `<bos>` |
| `cogito`, `deepseek3.1` | `<\|redacted_begin_of_sentence\|>` | DeepSeek-family BOS alias |
| `functiongemma` | `<bos>` | Matches template render output |
| Most others | `""` | BOS handled inside template or absent |

Code: `model/renderers/renderer.go` → `LeadingBOSForRenderer`; `server/renderer_resolution.go` → `leadingBOSForModel`.

---

## Pre-tokenized prompts (`PromptTokens`)

**Why:** `chatPrompt` tail-truncates megaprompts by dropping tokens from the **front** (keep latest user turn). Re-tokenizing the truncated string can diverge from the truncation math (byte-level detokenize edge cases, special tokens).

When truncation produces a token slice, routes pass `CompletionRequest.PromptTokens` (via `mlxCompletionPromptTokens` for safetensors) so the runner ingests exact IDs. MLX MTP uses this to skip re-tokenize on long prefills.

**MLX always tokenizes once:** even when truncate is off, `chatPrompt` captures token IDs for safetensors models so `Prepare` never re-encodes the full prompt string.

**Tokenize cache:** `x/mlxrunner/tokenize_cache.go` memoizes `/v1/tokenize` per runner client (bounded LRU). **Why:** binary search + tail truncate can hit the same rendered string twice in one request; agent loops resend identical megaprompts every turn.

See [mlx-agent-prompts.md](./mlx-agent-prompts.md) for context cap, keep-alive floor, SSE keepalive, and operator logs.

---

## Padded multimodal inject (llama-server)

**Why:** SGLang preprocessed clients send `padded_input_ids` with Qwen3-VL vision token ids (`151652` start, `151653` end) already in the layout. Re-tokenizing the rendered string would duplicate or misplace vision placeholders. The llama-server path maps each complete vision block to one media marker in `prompt_string` plus a matching `multimodal_data` entry (base64 raster).

**Flow:**

1. Routes build pretokenized `PromptTokens` via `BuildPaddedCompletionPromptTokens` (same splice as ggml llamarunner).
2. `completionPromptForRequest` detokenizes text spans between vision blocks; one marker per `<|vision_start|>…<|vision_end|>`.
3. `Completion` attaches `multimodal_data` in marker order.

**Truncation with media:** pretokenized prompts truncate at token level even when images are attached. **Why:** agent megaprompts can exceed `num_ctx`; skipping truncate because media was present left oversized layouts on llama-server. Middle discard expands to whole vision blocks so markers and payloads stay paired.

**Fallback:** if `qwen3vl_hf_runner_inject` is active but pretokenized ids lack vision blocks while `Media` is non-empty, detokenize to string and use the standard `[img-N]` → marker replacement path. **Why:** partial client layouts should not fail silently with token-only prompts and no attached rasters.

**Unclosed vision blocks:** `vision_start` without matching `vision_end` is kept as text tokens — no extra media slot. **Why:** truncating through half a block previously emitted a marker with no payload.

**Observability:** `padded_input_ids llama-server inject` log line (`prompt_tokens`, `media`, `prompt_string_len`).

Code: `llm/padded_prompt_llama_server.go`, `llm/llama_server.go` (`completionPromptForRequest`, `Completion` multimodal switch).

**Gemma4 (`gemma4_img_runner_inject`):** same splice as ggml runner (`BuildGemma4PaddedCompletionPromptTokens`). llama-server resolves `<|image|>`, `<|video|>`, and `<|audio|>` soft token ids via subprocess `/tokenize`; maps each slot to `prompt_string` media marker(s) + `multimodal_data` (`<|video|>` → one marker per frame in the clip). **Fallback:** pretokenized ids without multimodal soft tokens but with `Media` → detokenize + standard marker replacement (same pattern as Qwen3-VL).

**All native VLM families (Jun 2026):** llama-server subprocess now mirrors ollama-engine padded inject for `mllama_img_runner_inject`, `gemma3_img_runner_inject`, `llama4_img_runner_inject`, `lfm2_img_runner_inject`, `glmocr_img_runner_inject`, `mistral3_img_runner_inject`, and `deepseekocr_img_runner_inject`. Each maps pretokenized vision blocks or soft tokens to `prompt_string` media markers + ordered `multimodal_data`; partial layouts fall back to detokenize + `[img-N]` replacement when `Media` is present.

**Ollama-engine (Mac default):** all native Go multimodal families below run padded inject in-process via `runner/ollamarunner/padded_inputs.go` (`EncodeMultimodal` + `PostTokenize`):

| Consume mode | Families |
|--------------|----------|
| `qwen3vl_hf_runner_inject` | Qwen3-VL, **qwen25vl**, **qwen2vl** (family gate) |
| `gemma4_img_runner_inject` | Gemma4 |
| `mllama_img_runner_inject` | mllama |
| `gemma3_img_runner_inject` | Gemma3 |
| `llama4_img_runner_inject` | Llama4 |
| `lfm2_img_runner_inject` | LFM2 / lfm2moe |
| `glmocr_img_runner_inject` | GLM-OCR |
| `mistral3_img_runner_inject` | Mistral3 / Pixtral |
| `deepseekocr_img_runner_inject` | DeepSeek-OCR |

Session ViT overlay + input-cache prefix hits surface as `cached_prompt_tokens` on turn 2 when `prompt_cache_key` is stable. Live gate: `RUN_E2E_VIDEO_AGENT_INFER=1 ./scripts/video_agent_infer_smoke.sh` (+ optional `VIDEO_AGENT_INFER_PREPROC=1`, requires `VIDEO_AGENT_GO_LOG` for layout-cache grep). **Why infer smoke:** expand-only `_debug_render_only` does not exercise real vision prefill. See [sglang-multimodal-borrowings.md](./sglang-multimodal-borrowings.md) §32–37 and [video-understanding.md](./video-understanding.md#live-gate).

**Still `deferred_non_qwen3vl`:** text-only architectures (e.g. **gemma3n**, **glm4moelite**) and VLMs without a native Go `MultimodalProcessor` path on ollama-engine.

**`grid_thw` runner hints:** `llm.ImageData.GridTHW` per raster from `video_spans`; Info log `vision grid hints` after encode on llamarunner (mtmd) and ollama-engine (Mac default). **Go seam:** `MultimodalTokenize(..., gridTHW)` on llamarunner (debug until mtmd C API); mtmd forward override deferred — [mtmd-grid-thw-handoff.md](./mtmd-grid-thw-handoff.md).

---

## SSE stream keepalive (long prefill)

**Why:** Agent HTTP clients may abort when no stream bytes arrive during multi-minute MLX prefill.

Env `OLLAMA_STREAM_KEEPALIVE_INTERVAL` (default `15`, `0` = off). Emits `status: keepalive` until first token. Code: `server/stream_keepalive.go`.

---

## llama.cpp pin (`c84b3020`)

Zerollama pins **elizaOS/llama.cpp** @ **`c84b3020`** (`LLAMA_CPP_VERSION`, `LLAMA_CPP_COMMIT`). This supersedes upstream Ollama v0.30.11 tag **`b9781`** while keeping mergeability with Phase 17.

**Why elizaOS unified base:** one sibling/runtime tree with dflash/QJL/checkpoint features; zerollama adds **19 Ollama/zerollama patches** on top — see [ggml-b9509-migration.md](./ggml-b9509-migration.md).

**Vendor tree:** `vendor/llama-cpp-c84b3020/` + `./scripts/sync_vendor_llama.sh` → in-tree `ml/backend/ggml/ggml` and `llama/llama.cpp`.

**Why `sync_vendor_llama.sh` checks patch count:** syncing bare pin (no commits on top) ships upstream-only ggml while `build-info.cpp` still reports `c84b3020` — CGO then misses `ggml_backend_dev_reset`, no-alloc scheduler, kv-ext, and `/kv/seq-copy`.

**Patch doctor:** `./scripts/llama_patch_doctor.sh` · `/health.llama_patches` · `zerollama doctor` (llama.cpp patches check)

Workflow: [llama/README.md](../llama/README.md) · [runtime/LLAMA_CPP_PIN.md](../runtime/LLAMA_CPP_PIN.md)

---

## M7 benchmark (Metal, Jun 2026)

Fair run: idle GPU, `llama3.2:3b`, `num_ctx=4096`, 6 epochs.

| Arm | Host | Generate tok/s |
|-----|------|----------------|
| Upstream Go → llama-server | `:11435` | ~158.3 |
| Zerollama ggml Metal | `:11436` | ~164.1 |
| Zerollama `--llama-server-backend` | `:11434` | ~158 |

**Decision:** keep **ggml Metal** as Mac default; Phase 17 is for **mergeability** and upstream parity, not immediate throughput wins.

Reproduce: `./scripts/m4_upstream_vs_zerollama_bench.sh`

---

## Remaining work

1. ~~Smoke-test `--llama-server-backend`~~ — `phase17_llama_server_smoke.sh` (uses pulled tag via `P17_MODEL` / `RUN_E2E_PROXY_MODEL`)
2. ~~`llama/compat/` vs `llama/patches/` dedup~~ — 0016 hooks, 0017 ggml deltas
3. ~~Port `discover/llama_server.go`~~ — hybrid bootstrap shipped
4. ~~LeadingBOS + PreservedTokens~~ — wired for llama-server renderers/parsers
5. ~~**Vendor sync to b9781**~~ — done; sibling rebuild + Metal sign-off optional
6. ~~**Policy:** Mac default stays ggml (M7 bench); Linux `auto` routes all GGUF~~
7. ~~**Cohere2 MoE MLX**~~ — done (#16670)
8. ~~**Cline providers.json**~~ — done (#16402)
9. ~~**Vision/thinking on llama-server**~~ — explicit or Linux `auto`; vision E2E: `phase17_llama_server_vision_smoke.sh`
10. ~~**Launch model inventory**~~ — [launch-model-inventory.md](./launch-model-inventory.md)
11. ~~**Phase 16 edge mode**~~ — [phase16-thin-edge.md](./phase16-thin-edge.md)
12. **L2 fork merge** — coordinate pin with borrowings L2 when bench gates pass (**Jun 2026: FAIL merge @ 8k CUDA** — stock faster; see `./scripts/phase17_l2_pin_status.sh` and [gpu-profiles-l2.md](./gpu-profiles-l2.md))
13. **Flash-MoE (anemll)** — **Partial (Jun 2026):** flag passthrough, Modelfile `moe_*` options, `build_flash_moe_llama_server.sh`, **`flash_moe_smoke.sh`**, doctor check — [flash-moe.md](./flash-moe.md). **Why llama-server only:** slot-bank lives in anemll fork, not ggml Metal. **Open:** `pull` sidecar extract, vendor pin merge.
14. **ANE probe (maderix)** — **Partial (Jun 2026):** subprocess smoke + doctor — [ane-probe.md](./ane-probe.md). **Why subprocess:** private ANE APIs must not ship inside main Go binary. **Open:** hybrid inference research, not hot path.

---

## Operator troubleshooting

| Symptom | Likely cause | Fix |
|---------|----------------|-----|
| `ggml runner not linked in edge build` | `-tags edge` binary without llama-server routing | `./scripts/build_llama_server.sh`; `zerollama serve --edge` or `ZEROLLAMA_LLAMA_SERVER=1` |
| `ggml runner disabled in edge mode` | Edge mode without llama-server routing | Set `ZEROLLAMA_LLAMA_SERVER=1`/`auto`, or place `llama-server` on disk for Linux auto |
| `using llama-server subprocess` missing in logs | Model routed to ggml or runtime | Check `GET /api/status` → `inference.backend`; use `--llama-server-backend` or Linux auto |
| `:8081` runtime answers during edge smoke | Runtime not disabled | `ZEROLLAMA_RUNTIME=0`; use `phase16_edge_smoke.sh` env |
| Linux plain serve uses ggml | llama-server not on disk | Build llama-server; `./scripts/serve_linux_auto.sh`; `zerollama doctor` |

**Ship-hardware bundle (5080-class):**

```bash
export LLAMA_SERVER_BIN=/path/to/llama-server
export RUN_E2E_PROXY_MODEL=llama3.2:3b
RUN_E2E_PREFLIGHT=0 RUN_E2E_UPSTREAM_GGUF=1 ./scripts/gpu_5080_session.sh
```

**Phase 15 writable bind:** still blocked upstream — watch `./scripts/phase15_upstream_kv_watch.sh` and `phase15_llama_kv_ext_pin_check.sh` for new symbols in `llama.h`.

---

## Launch integrations (agent config)

**Why not document only in upstream diff:** launch is operator-facing; inventory changes how every integration writes config.

| Topic | Doc |
|-------|-----|
| `LaunchModel`, resolve, drift guard | [launch-model-inventory.md](./launch-model-inventory.md) |
| Cline `providers.json` | [launch-model-inventory.md](./launch-model-inventory.md#launch-drift-guard-liveconfigmatches) |
| OMP multi-model YAML | [launch-model-inventory.md](./launch-model-inventory.md#managed-single-model-catalog-omp) |

---

## Related docs

- [upstream-ollama-diff.md](./upstream-ollama-diff.md) — architecture comparison + cherry-pick map
- [phase16-thin-edge.md](./phase16-thin-edge.md) — Phase 16 edge mode (`--edge`)
- [sglang-multimodal-borrowings.md](./sglang-multimodal-borrowings.md) — padded multimodal inject + audit §25
- [mtmd-grid-thw-handoff.md](./mtmd-grid-thw-handoff.md) — upstream `grid_thw` → mtmd forward API sketch
- [llama-cpp-backend.md](./llama-cpp-backend.md) — Python runtime test harness
- [apple-silicon-metal.md](./apple-silicon-metal.md) — Mac operator guide
- [llama/README.md](../llama/README.md) — pin bump workflow
