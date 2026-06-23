# MLX vs ggml Metal vs Python runtime — routing policy

**Audience:** operators and contributors choosing an inference path on Apple Silicon (and anywhere MLX is built).

**Related:** [apple-silicon-metal.md](./apple-silicon-metal.md), [handoff-phase12-runtime-tools.md](./handoff-phase12-runtime-tools.md), [phase14-inprocess-llama.md](./phase14-inprocess-llama.md), [upstream-ollama-diff.md](./upstream-ollama-diff.md).

---

## Why three paths exist

| Path | Weight format | Process | Best for |
|------|---------------|---------|----------|
| **ggml Metal** (default) | GGUF | Go runner + llama.cpp Metal | Pulled library models, vision/think on ggml, lowest friction |
| **Python runtime** | GGUF | Embedded/sidecar FastAPI + llama-server or inprocess | Phase 12 tools on runtime-routed text, Phase 11/13 admission, manifest `options.gguf` |
| **MLX engine** | safetensors (`ModelFormat: safetensors`) | Go → `mlxrunner` / `imagegen` | Experimental MLX-native weights, MLX image pipelines |

These are **not interchangeable**. Converting GGUF → MLX or routing MLX through the Python runtime is out of scope today.

---

## Decision flow

```text
Is ModelFormat safetensors (IsMLX)?
  yes → mlxrunner / imagegen (sched.go). Never runtime-default.
  no  → GGUF
        Modelfile inference: zerollama-runtime OR Phase 12 default-on eligible?
          yes → Python runtime (proxy / embed). ggml load deferred when safe.
          no  → ggml runner (Metal on Mac, CUDA on Linux)
```

**Default-on (Phase 12):** When `ZEROLLAMA_RUNTIME` default is on and runtime is embedded/URL set, **text completion GGUF** models may proxy to Python without an explicit Modelfile backend. **Excluded:** MLX, embedding-only, vision, video, image, audio, thinking-only legacy paths, empty path.

**`--llama-cpp-backend`:** Forces eligible text GGUF through Python runtime (skips ggml load). Same defer path as explicit runtime backend — see [llama-cpp-backend.md](./llama-cpp-backend.md). Upstream Ollama achieves similar GGUF coverage via **Go → llama-server** without Python ([upstream-ollama-diff.md](./upstream-ollama-diff.md)).

---

## Code guarantees (M4)

MLX models are **never** silently routed to the Python runtime by default-on policy:

```69:70:server/runtime_inference_routing.go
	if m.ModelPath == "" || m.IsMLX() {
		return false
```

Scheduler loads MLX via `mlxrunner.NewClient` / `imagegen.NewServer`, not ggml:

```645:668:server/sched.go
		if !req.model.IsMLX() {
			// ... ggml runner ...
		} else {
			// mlxrunner or imagegen
		}
```

Explicit `inference: zerollama-runtime` on an MLX model does **not** route to Python (`modelUsesRuntimeInference` returns false for `IsMLX()`). Operators should not set that backend on safetensors models.

```go
// modelUsesRuntimeInference — MLX never uses Python GGUF runtime
if m.IsMLX() {
    return false
}
```

---

## Environment and manifest knobs

| Goal | Setting |
|------|---------|
| Force runtime for one request (smoke) | Header `X-Zerollama-Runtime: 1` or `ZEROLLAMA_RUNTIME=1` |
| Force all models to runtime | `ZEROLLAMA_RUNTIME_ALL=1` (still respects legacy chat gates) |
| Opt model into runtime | Modelfile `MODALITY inference zerollama-runtime` |
| Opt model out of runtime default | Modelfile `MODALITY inference ggml` (future) or legacy caps |
| Keep ggml only | `ZEROLLAMA_LEGACY_RUNNER=1` |
| Route text GGUF via Python llama.cpp (experimental) | `./zerollama serve --llama-cpp-backend` or `ZEROLLAMA_LLAMA_CPP_BACKEND=1` |
| MLX build | [apple-silicon-metal.md](./apple-silicon-metal.md#mlx-engine-optional) — **why** safetensors + `libmlxc.dylib`; rebuild at `MLX_VERSION` via `BUILD_MLX=1 ./scripts/build_zerollama_mac.sh` or `./scripts/build_production_mac.sh` |
| LM Studio MLX import | `OLLAMA_LMSTUDIO_IMPORT` (default on); `OLLAMA_LMSTUDIO_LIST_ALL=1` lists MLX even when disk tight — **why:** MLX repacks ~full model size into `OLLAMA_MODELS`; GGUF symlinks are near-zero copy |

---

## LM Studio MLX disk policy (Jun 2026)

**Why:** LM Studio caches MLX safetensors (`config.json` + weights) separately from GGUF. Importing MLX into zerollama **copies** tensors into new blobs (~model size + 512 MiB headroom). Listing models the operator cannot import wastes time; failing at pull with a clear error is better than a mid-import OOM.

| Behavior | Setting |
|----------|---------|
| Hide MLX models when disk insufficient | Default (`OLLAMA_LMSTUDIO_LIST_ALL` unset) |
| List all discoverable models anyway | `OLLAMA_LMSTUDIO_LIST_ALL=1` (pull still enforces space) |
| Pull fails with human-readable error | Always when `HasDiskForDirImport` fails |

**Full guide:** [lmstudio-import.md](./lmstudio-import.md) — three import paths, naming, troubleshooting, code map.

Code: `internal/lmstudio/lmstudio.go` (`ImportCopyBytes`), `server/lmstudio_catalog.go`, `server/lmstudio_import.go`.

---

## Mac operator checklist

1. **General chat (GGUF):** `zerollama serve` — ggml Metal unless manifest/default routes to runtime.
2. **Tools on runtime-routed text:** ensure runtime embed + Phase 12 path; see [apple-silicon-metal.md](./apple-silicon-metal.md).
3. **safetensors / MLX create:** build MLX component; model uses `IsMLX()` path automatically.
4. **Session gate:** `./scripts/gpu_metal_session.sh` (smoke + snapshot); optional `RUN_E2E_PHASE14=1` for inprocess Metal.
5. **Agent megaprompts (MLX):** see [mlx-agent-prompts.md](./mlx-agent-prompts.md) — context cap, truncate, tokenize cache, keepalive logs.

---

## MLX agent prompts (Jun 2026)

**Why separate from routing:** `IsMLX()` picks the subprocess; **M15** hardening fixes what happens **inside** that path when clients send 100k+ tokens every turn.

| Symptom | Likely cause | Log / fix |
|---------|--------------|-----------|
| `num_ctx=262144`, no truncate | Bogus HF `text_config.max_position_embeddings` | Rebuild; expect `num_ctx capped to mlx model maximum` |
| Two `/v1/tokenize` per request | Budget search + tail truncate | Tokenize LRU + `PromptTokens` passthrough |
| Cold reload every ~5m | Default keep_alive | MLX 30m floor when unset; or set explicit `keep_alive` |
| Client empty-stream timeout | Long prefill, no SSE bytes | `OLLAMA_STREAM_KEEPALIVE_INTERVAL=15` (default) |
| 8 min to first token @ 131k | No truncate + full prefill | Client: shrink context; server: `prompt tail-truncated` |

Full guide: [mlx-agent-prompts.md](./mlx-agent-prompts.md).

---

## Non-goals

- Replacing ggml Metal with MLX for all GGUF models.
- Runtime loading safetensors without a separate MLX integration.
- NVML-style VRAM probes on Mac (use `metal-unified`; see apple-silicon guide).
