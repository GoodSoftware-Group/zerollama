# Qwen 3.5 / 3.6 on Apple Silicon

**Audience:** macOS operators loading **qwen35**, **qwen35moe**, or library tags like **`qwen3.6:latest`** on M-series Macs.

**Related:** [apple-silicon-metal.md](./apple-silicon-metal.md), [llama/compat/README.md](../llama/compat/README.md), [ROADMAP.md](./ROADMAP.md#apple-silicon--metal-track).

---

## Why this doc exists

Qwen 3.5/3.6 GGUFs hit **three separate Mac-only failure modes** in mid-2026 zerollama builds. Each looked like “Metal is broken” or “qwen35 isn’t supported,” but the root causes were different layers:

| Symptom | Layer | Why it happened |
|---------|--------|-----------------|
| ~~SIGABRT in `ggml_backend_sched_reserve`~~ **Fixed Jun 2026** | Go **ollama engine** (Metal ggml backend) | `newTensor` eagerly allocated graph intermediates while `sched_reserve` also assigned buffers → `GGML_ASSERT(tensor->buffer == NULL)` on qwen35moe worst-case reserve. **Fix:** defer graph tensor alloc; `.Persistent()` for KV/recurrent contexts only. |
| `rope.dimension_sections has wrong array length; expected 4, got 3` | **llama.cpp loader** + missing compat | Published Ollama GGUFs store M-RoPE sections as **3** ints; llama.cpp expects **4** (padded). The fix lived in `llama/compat/` but was only wired into **llama-server** CMake builds—not the **in-process llamarunner** CGO path. |
| `kernel_unary_f32_f32 was not found` → SIGSEGV on first token | **Embedded Metal shaders** | macOS compiles shaders from `ggml-metal-embed.metal` at runtime. That file is **generated** from `ggml-metal.metal`; when ggml bumps without `go generate`, sigmoid/unary kernels (needed by qwen35 SSM/gated paths) are missing from the embed. |

Fixing one layer often exposed the next. This doc is the operator map.

---

## Recommended path today (M4 Max class)

1. **Rebuild** with a current tree (see [Build](#build)) — includes Metal embed regen and the Jun 2026 `sched_reserve` allocator fix.
2. **Restart serve** so runners pick up the new binary (old subprocesses keep the old `.dylib`/shader embed until replaced).
3. **Use a practical context** — these models advertise 262K train context; start with **`num_ctx` 2048–8192** unless you have headroom. The log line `n_ctx_seq (2048) < n_ctx_train (262144)` is **informational**, not an error.
4. **Expect thinking/VL variants** — monolithic qwen35 VL blobs include vision tensors; the compat layer strips/hides them for the text loader and uses mtmd for vision when configured. **Thinking models** (e.g. `qwen3.6:latest`) may return short replies in the **`thinking`** field with an empty **`response`** — that is normal, not a failed generate.

```bash
./scripts/build_zerollama_mac.sh
./zerollama serve
# API example — cap context for first test
curl http://127.0.0.1:11434/api/generate -d \
  '{"model":"qwen3.6:latest","prompt":"hi","stream":false,"options":{"num_ctx":2048,"num_predict":32}}'
```

---

## Build

**Why `go generate` is part of the Mac build:** The Metal backend does not ship a precompiled `.metallib` in the Go binary—it embeds **Metal source** and JIT-specializes kernels (e.g. `kernel_unary_f32_f32_op=102` = sigmoid for gated SSM). If `ggml-metal-embed.metal` is stale, load succeeds but **first decode** crashes.

```bash
./scripts/build_zerollama_mac.sh
```

That script (since Jun 2026) runs:

```bash
GOFLAGS=-mod=mod go generate ./ml/backend/ggml/ggml/src/ggml-metal/
```

before `go build`. After any edit to `ggml-metal.metal` or ggml vendor sync, regenerate manually if you bypass the script:

```bash
GOFLAGS=-mod=mod go generate ./ml/backend/ggml/ggml/src/ggml-metal/
go build -a -o zerollama .   # -a when embed changed, to force CGO relink
```

---

## Architecture routing (which runner loads qwen35)

```text
  qwen35 / qwen35moe / qwen3next GGUF
           │
           ├─ OllamaEngineRequired() (default for qwen35*) ──► Go ollama-engine
           │                      (ggml Metal backend on darwin since Jun 2026)
           │
           ├─ OLLAMA_NEW_ENGINE=0 + not required ──► legacy llamarunner (llama archs)
           │
           ├─ ZEROLLAMA_LLAMA_SERVER=1 ──► llama-server subprocess + llama/compat
           │
           └─ legacy investigation ──► llamarunner CGO + in-process compat
```

**Why Go ollama-engine on darwin (Jun 2026):** `sched_reserve` no longer aborts on qwen35moe after graph tensors defer allocation to the scheduler and KV contexts use `Persistent()`. M4 Max sign-off: `qwen3.6:latest`, 41/41 GPU layers, `--ollama-engine`.

**Why legacy llamarunner still exists:** llama-server / CGO path for Phase 17 upstream alignment, compat-layer debugging, and models not yet on the Go engine graph. It is **not** the default for `qwen35*` on Mac anymore.

**Dual Metal:** If the Python runtime sidecar holds Metal (`llama_server=true`), ggml loads may return HTTP **503** until handoff — see [apple-silicon-metal.md](./apple-silicon-metal.md#scheduler-errors-http-status). `qwen35_mac_smoke.sh` calls training-handoff first.

---

## Go engine sched_reserve fix (Jun 2026)

**Symptom:** Process abort during load (no Go stack trace) — log ends at `llm_load: ollama-engine layout phase` / worst-case reserve.

**Why:** During `LoadOperationFit`, the runner builds a worst-case forward graph and calls `ggml_backend_sched_reserve`. Graph intermediate tensors must have `tensor->buffer == NULL` so the scheduler can assign device memory. `newTensor` used to eagerly `ggml_backend_tensor_alloc` every tensor when `allocMemory` was true, including graph scratch — the scheduler then hit `GGML_ASSERT(tensor->buffer == NULL)`.

**Fix:**

| Piece | Path | Why |
|-------|------|-----|
| Defer graph alloc | `ml/backend/ggml/ggml.go` `newTensor` | Only **input** tensors and **persistent** contexts get eager buffers |
| KV/recurrent mark | `Context.Persistent()` + kvcache `*.go` | KV cells and recurrent state must exist before forward — not scheduler scratch |
| Remove arch blocklist | `runner/ollamarunner/runner.go` `reserveWorstCaseGraph` | Skipping qwen35 reserve hid the bug; root fix makes worst-case reserve safe |

**Verify:** `./scripts/build_zerollama_mac.sh && ./zerollama serve` → load `qwen3.6:latest` → log `offloaded N/N layers to GPU`, runner cmd includes `--ollama-engine`, no abort.

---

## Compat layer: what it fixes for qwen35

Published blobs differ from llama.cpp-native GGUF in several ways. `llama/compat/` patches metadata **in memory** at load time (see handlers `handle_qwen35`, `handle_qwen35moe`):

| Published GGUF | llama.cpp expects | Compat action |
|----------------|-------------------|---------------|
| `rope.dimension_sections` length **3** | length **4** | Pad with trailing `0` |
| `attention.head_count_kv` array per layer | scalar `UINT32` | Collapse to max non-zero |
| Embedded `v.*` / `mtp.*` / `mm.*` tensors | separate mmproj / dropped MTP | Skip or translate tensors |
| `blk.N.ssm_dt` | `blk.N.ssm_dt.bias` | Rename |
| MTP expert shards | merged expert weights | Merge + disable mmap when needed |

**Why compat wasn’t enough alone on Mac:** CMake `llama/server` builds applied compat automatically; the **default Mac binary** uses CGO directly against `llama/llama.cpp/` and needed the same hooks linked via `llama/compat/compat.go` + patched `llama-model-loader.cpp` / `clip.cpp`.

Disable compat (debug): `OLLAMA_LLAMA_CPP_COMPAT=0`.

---

## VL manifests and `PrimaryFamily()`

**Why this exists:** Qwen 3.5 VL GGUFs include both a **clip** projector layer and a **qwen35** LLM layer. At create time, whichever layer is processed first can become `ModelFamily` in the manifest — often `clip`. Renderers, parsers, and thinking defaults then pointed at the wrong architecture.

**Fix (Jun 2026):** `server/model_family.go` — `PrimaryFamily()` prefers LLM architectures (`qwen35`, `qwen35moe`, …) over projector-only strings like `clip`. Projector-only manifests return `""` (no LLM to route on).

**Operator impact:** Existing VL tags work without re-pull when `ModelFamilies` includes both `clip` and `qwen35`. Re-create the model on current tree to persist correct defaults in the manifest.

---

## Log lines explained

| Log | Meaning |
|-----|---------|
| `tensor API disabled for pre-M5 and pre-A19 devices` | Expected on M4 Max. Metal “tensor API” is off unless M5/A19 or `GGML_METAL_TENSOR_ENABLE=1`. **Not** the crash cause. |
| `n_ctx_seq (N) < n_ctx_train (262144)` | Runtime context is smaller than training max; normal when you set `num_ctx`. |
| `detected Ollama-format qwen35moe GGUF; applying compatibility fixes` | Compat shim active; `dimension_sections` should show `arr[i32,4]`. |
| `compat tensor transform: op=F32 add-one norm shift` | MTP norm weights adjusted for llama.cpp layout—expected during load. |
| `fused Gated Delta Net enabled` | qwen35 SSM path; first decode needs **sigmoid** Metal kernel (`op=102`). |
| `control-looking token … was not control-type` | Published GGUF `token_type` wrong for FIM markers; llama.cpp overrides. **Harmless** — [Token warnings](#token-warnings-jun-2026). |
| `embeddings required but some input tokens were not marked as outputs` | Embedding context + chat batch mismatch; llama.cpp overrides. **Harmless** — [Token warnings](#token-warnings-jun-2026). |
| `offloaded 0/N layers to GPU` with slow inference | Bootstrap GPU discovery returned empty (fixed Jun 2026). Rebuild; expect `library=Metal` at startup — [apple-silicon-metal.md](./apple-silicon-metal.md#gpu-bootstrap-discovery-jun-2026). |
| `kernel_unary_f32_f32 was not found` | Stale `ggml-metal-embed.metal` — rebuild with `go generate` (see [Build](#build)). |

---

## Memory and model size

| Model (example) | Notes |
|-----------------|--------|
| `qwen3.6:latest` (qwen35moe ~35B-A3B) | MoE—smaller active weights; still large KV + recurrent (SSM) state. |
| `qwen3.6-27b:q8_0` (qwen35 VL) | ~27GB weights + vision; may OOM on unified memory even after load fixes. |

**Why cap `num_ctx`:** KV and recurrent buffers scale with context. A 262K context on a 27B Q8 model is not realistic on a single Mac without aggressive swapping.

---

## Manifest `num_ctx` vs request `options.num_ctx` (Jun 2026)

**Why this matters:** `/api/create` with `"parameters": {"num_ctx": …}` **persists** the value in the model manifest (`/api/show`, Modelfile). That is **not** the same as passing `"options": {"num_ctx": …}` on a single chat/generate call.

| Source | When it applies | KV / memory |
|--------|-----------------|-------------|
| **Manifest default** (`parameters.num_ctx` from create) | Merged in `modelOptions()` and passed to **`llama.Load`** on every load/reload | **Pre-allocated at load time** for the full context — large values (e.g. 262144) can hang or OOM before the first token on ggml/llamarunner |
| **Request `options.num_ctx`** | Per call; may trigger **`needsReload`** if different from the loaded runner's options | Intended for **on-demand** large context (Hermes auto-detection, runtime Phase 13 clamp). Safer default: keep manifest at **4096–8192**, raise per request when needed |

**Symptoms operators hit:**

- **`/api/show`** shows `num_ctx 262144` but **`/api/ps`** shows `context_length: 4096` — manifest updated, **warm runner not reloaded** (fixed: create now evicts loaded runners; or run stop / `keep_alive:0`).
- **Generation hangs after create with huge manifest `num_ctx`** — load-time KV pre-allocation for 262K on qwen35moe; **revert manifest default** to 4096 and use request options for long context.

**Recommended pattern (M4 Max, qwen3.6):**

```bash
# Persist a modest default (fast load)
curl http://localhost:11434/api/create -d '{
  "model": "qwen3.6:latest",
  "from": "qwen3.6:latest",
  "parameters": {"num_ctx": 4096}
}'

# Long context only when needed (Hermes or manual)
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3.6:latest",
  "messages": [{"role":"user","content":"…"}],
  "stream": false,
  "options": {"num_ctx": 262144}
}'
```

**Truncation signal:** if input still exceeds effective context, final responses include `prompt_truncated` / `messages_truncated` — set `"truncate": false` for HTTP 400 instead of silent drop. See [CHANGELOG](../CHANGELOG.md).

**Unload before trusting `/api/ps`:**

```bash
curl http://localhost:11434/api/generate -d '{"model":"qwen3.6:latest","prompt":"","keep_alive":0}'
curl http://localhost:11434/api/ps   # should be empty
```

Doc: [scheduling-vram-policy.md — ggml scheduler unload](./scheduling-vram-policy.md#go-ggml-scheduler-keep_alive-unload-and-num_ctx-at-load).

---

## Token warnings (Jun 2026)

Two warning lines often appear on **successful** qwen3.6 loads. They look alarming but are **not** inference failures.

### `control-looking token … was not control-type`

**Example:**

```text
load: control-looking token: 248060 '<|fim_prefix|>' was not control-type; this is probably a bug in the model. its type will be overridden
```

**Why:** Ollama-published qwen35/qwen35moe GGUFs store every entry in `tokenizer.ggml.token_type` as **NORMAL** (`1`), including fill-in-the-middle and repo markers (`<|fim_prefix|>`, `<|fim_suffix|>`, `<|repo_name|>`, etc.). llama.cpp knows these strings should be **CONTROL** tokens and upgrades them at load time.

**Impact:** None for normal chat. FIM/repo tokens behave correctly after override. The warning is llama.cpp telling you the **blob metadata** is wrong, not that zerollama mis-tokenized your prompt.

**Where:** `llama/llama.cpp/src/llama-vocab.cpp` (auto-detect by token text). Compat fixes other qwen35 metadata (`rope.dimension_sections`, MTP tensors) but does not rewrite `token_type` today.

**Action:** Ignore on chat models. For FIM workflows, verify output quality; no rebuild required.

### `embeddings required but some input tokens were not marked as outputs -> overriding`

**Example:**

```text
init: embeddings required but some input tokens were not marked as outputs -> overriding
```

**Why:** The llamarunner always creates llama contexts with **`embeddings=true`** so the same loaded model can serve `/api/embed` (hidden-state extraction) without a second context. In that mode llama.cpp expects **every prefill token** flagged as an output row. Chat generation only marks the **last** prefill token (standard causal LM). llama.cpp detects the mismatch, logs once, and marks all rows itself.

**Impact:** Chat still returns 200. Slight extra work on prefill (same as what the override would do anyway). Not related to vision image embeddings.

**Where:** `llama/llama.go` (`NewContextParams` → `params.embeddings = true`); `llama/llama.cpp/src/llama-batch.cpp`; `runner/llamarunner/runner.go` (`output := i+1 == len(seq.inputs)`).

**Action:** Ignore for chat. If the line disappears in a future release, it means we split embedding vs generative context flags — behavior should stay the same.

---

## Troubleshooting checklist

1. **Confirm binary age** — `./zerollama --version` / rebuild time; restart serve after rebuild.
2. **`unknown architecture qwen35`** — llama.cpp pin too old; rebuild with current `LLAMA_CPP_VERSION` / vendor sync ([ggml-b9509-migration.md](./ggml-b9509-migration.md)).
3. **`dimension_sections` length error** — compat not linked; ensure tree includes `llama/compat` CGO import and `llama-model-loader.cpp` hooks.
4. **Load OK, crash on first token, `kernel_unary_*`** — run `go generate` on `ggml-metal`, `go build -a`.
5. **Still OOM** — lower `num_ctx`, use a smaller quant, or a smaller variant.
6. **`total_vram="0 B"`, `offloaded 0/N layers`** — GPU bootstrap discovery bug (fixed Jun 2026); rebuild and confirm `inference compute library=Metal` at serve start. **Why it looked like qwen35-only:** empty bootstrap discovery forced CPU layout even when Metal worked in a later runner subprocess.
7. **SIGABRT during qwen35 load on Go engine** — stale tree before Jun 2026 `sched_reserve` fix; rebuild current tree. If abort persists, check [Go engine sched_reserve fix](#go-engine-sched_reserve-fix-jun-2026).
8. **HTTP 503 darwin Metal contention** — runtime sidecar holds Metal; handoff before ggml load. See [apple-silicon-metal.md](./apple-silicon-metal.md#scheduler-errors-http-status).

**Opt-in smoke** (handoffs runtime Metal, validates Go ollama-engine + generate):

```bash
RUN_E2E_QWEN35_MODEL=qwen3.6:latest ./scripts/qwen35_mac_smoke.sh
```

**Why thinking models need special assertion:** `qwen3.6:latest` may put one-word replies in `thinking` with an empty `response` — the smoke accepts either field.

---

## Full Metal sign-off (Jun 2026)

**Why a separate gate from daily serve:** Sign-off uses **`OLLAMA_HOST=:8080`** + runtime **`:8081`** (CI layout), starts its own stack, and runs Phase 13–15 plus optional qwen35 — not the default `:11434` daily path.

```bash
LLAMA_CPP_ROOT=../llama.cpp ./scripts/build_llama_server.sh
RUN_E2E_QWEN35=1 RUN_E2E_QWEN35_MODEL=qwen3.6:latest ./scripts/metal_signoff.sh
```

**Order inside the script:** qwen35 runs **after Phase 14** and **before Phase 15**. **Why:** Phase 15 stops the runtime sidecar on exit; qwen35 needs `:8081` for training-handoff and resume after ggml unload.

**M4 Max PASS (Jun 2026):** coordination, Phase 13 snapshot, Phase 14 inprocess, qwen35 generate + unload, Phase 15 KV + multiseq.

Standalone qwen35 only (when you already have `:8080`/`:8081` up):

```bash
RUN_E2E_QWEN35_MODEL=qwen3.6:latest ./scripts/qwen35_mac_smoke.sh
```

---

## Future direction (Phase 17)

Upstream Ollama routes default GGUF through **Go → llama-server** with compat at CMake fetch time. Zerollama’s Mac default remains **in-process ggml** for now. When Phase 17 lands for Mac, qwen35 may move to llama-server subprocess with the same compat layer—**without** requiring the separate CGO compat package, but **with** the same metadata semantics documented here.

Track: [ROADMAP.md — M10](./ROADMAP.md#apple-silicon--metal-track).
