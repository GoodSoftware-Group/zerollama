# Qwen 3.5 / 3.6 on Apple Silicon

**Audience:** macOS operators loading **qwen35**, **qwen35moe**, or library tags like **`qwen3.6:latest`** on M-series Macs.

**Related:** [apple-silicon-metal.md](./apple-silicon-metal.md), [llama/compat/README.md](../llama/compat/README.md), [ROADMAP.md](./ROADMAP.md#apple-silicon--metal-track).

---

## Why this doc exists

Qwen 3.5/3.6 GGUFs hit **three separate Mac-only failure modes** in mid-2026 zerollama builds. Each looked like “Metal is broken” or “qwen35 isn’t supported,” but the root causes were different layers:

| Symptom | Layer | Why it happened |
|---------|--------|-----------------|
| SIGSEGV in `ggml.New()` / `_Cfunc_ggml_backend_get_default_buffer_type` | Go **ollama engine** (Metal ggml backend) | `qwen35*` is in `OllamaEngineRequired()`; the new Go path initializes Metal before load finishes. C segfaults don’t return Go errors, so there is no fallback. |
| `rope.dimension_sections has wrong array length; expected 4, got 3` | **llama.cpp loader** + missing compat | Published Ollama GGUFs store M-RoPE sections as **3** ints; llama.cpp expects **4** (padded). The fix lived in `llama/compat/` but was only wired into **llama-server** CMake builds—not the **in-process llamarunner** CGO path. |
| `kernel_unary_f32_f32 was not found` → SIGSEGV on first token | **Embedded Metal shaders** | macOS compiles shaders from `ggml-metal-embed.metal` at runtime. That file is **generated** from `ggml-metal.metal`; when ggml bumps without `go generate`, sigmoid/unary kernels (needed by qwen35 SSM/gated paths) are missing from the embed. |

Fixing one layer often exposed the next. This doc is the operator map.

---

## Recommended path today (M4 Max class)

1. **Rebuild** with a current tree (see [Build](#build)).
2. **Restart serve** so runners pick up the new binary (old subprocesses keep the old `.dylib`/shader embed until replaced).
3. **Use a practical context** — these models advertise 262K train context; start with **`num_ctx` 2048–8192** unless you have headroom. The log line `n_ctx_seq (2048) < n_ctx_train (262144)` is **informational**, not an error.
4. **Expect thinking/VL variants** — monolithic qwen35 VL blobs include vision tensors; the compat layer strips/hides them for the text loader and uses mtmd for vision when configured.

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
           ├─ darwin (default) ──► legacy llamarunner (CGO llama.cpp + Metal)
           │                      + llama/compat in-process (KV/tensor fixes)
           │
           ├─ linux/windows ──► Go ollama engine if OllamaEngineRequired()
           │                  (unless OLLAMA_NEW_ENGINE=0 and not required)
           │
           └─ OLLAMA_NEW_ENGINE=1 ──► forces Go engine (avoid on Mac for qwen35*)
```

**Why darwin uses legacy runner for qwen35\*:** Until the Go Metal backend handles qwen35-class models reliably, forcing legacy llama.cpp avoids a process-killing C segfault during `ggml.New()`. Phase 17 may reunify on llama-server; until then Mac qwen35 is **legacy + compat**, not **Go engine**.

**Override (debug only):** `OLLAMA_NEW_ENGINE=1` forces the Go path—expect crashes on qwen35 MoE/VL on pre-M5 Metal until upstream fixes land.

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
| `kernel_unary_f32_f32 was not found` | Stale `ggml-metal-embed.metal` — rebuild with `go generate` (see [Build](#build)). |

---

## Memory and model size

| Model (example) | Notes |
|-----------------|--------|
| `qwen3.6:latest` (qwen35moe ~35B-A3B) | MoE—smaller active weights; still large KV + recurrent (SSM) state. |
| `qwen3.6-27b:q8_0` (qwen35 VL) | ~27GB weights + vision; may OOM on unified memory even after load fixes. |

**Why cap `num_ctx`:** KV and recurrent buffers scale with context. A 262K context on a 27B Q8 model is not realistic on a single Mac without aggressive swapping.

---

## Troubleshooting checklist

1. **Confirm binary age** — `./zerollama --version` / rebuild time; restart serve after rebuild.
2. **`unknown architecture qwen35`** — llama.cpp pin too old; rebuild with current `LLAMA_CPP_VERSION` / vendor sync ([ggml-b9509-migration.md](./ggml-b9509-migration.md)).
3. **`dimension_sections` length error** — compat not linked; ensure tree includes `llama/compat` CGO import and `llama-model-loader.cpp` hooks.
4. **Load OK, crash on first token, `kernel_unary_*`** — run `go generate` on `ggml-metal`, `go build -a`.
5. **Still OOM** — lower `num_ctx`, use a smaller quant, or a smaller variant.
6. **HTTP 503 darwin Metal contention** — runtime sidecar holds Metal; handoff before legacy qwen35 load. See [apple-silicon-metal.md](./apple-silicon-metal.md#scheduler-errors-http-status).

**Opt-in smoke:**

```bash
RUN_E2E_QWEN35_MODEL=qwen3.6:latest ./scripts/qwen35_mac_smoke.sh
```

---

## Future direction (Phase 17)

Upstream Ollama routes default GGUF through **Go → llama-server** with compat at CMake fetch time. Zerollama’s Mac default remains **in-process ggml** for now. When Phase 17 lands for Mac, qwen35 may move to llama-server subprocess with the same compat layer—**without** requiring the separate CGO compat package, but **with** the same metadata semantics documented here.

Track: [ROADMAP.md — M10](./ROADMAP.md#apple-silicon--metal-track).
