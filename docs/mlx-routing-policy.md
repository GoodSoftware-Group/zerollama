# MLX vs ggml Metal vs Python runtime — routing policy

**Audience:** operators and contributors choosing an inference path on Apple Silicon (and anywhere MLX is built).

**Related:** [apple-silicon-metal.md](./apple-silicon-metal.md), [handoff-phase12-runtime-tools.md](./handoff-phase12-runtime-tools.md), [phase14-inprocess-llama.md](./phase14-inprocess-llama.md).

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
| MLX build | [development.md](./development.md#mlx-engine-optional) — `cmake --install build --component MLX` |

---

## Mac operator checklist

1. **General chat (GGUF):** `zerollama serve` — ggml Metal unless manifest/default routes to runtime.
2. **Tools on runtime-routed text:** ensure runtime embed + Phase 12 path; see [apple-silicon-metal.md](./apple-silicon-metal.md).
3. **safetensors / MLX create:** build MLX component; model uses `IsMLX()` path automatically.
4. **Session gate:** `./scripts/gpu_metal_session.sh` (smoke + snapshot); optional `RUN_E2E_PHASE14=1` for inprocess Metal.

---

## Non-goals

- Replacing ggml Metal with MLX for all GGUF models.
- Runtime loading safetensors without a separate MLX integration.
- NVML-style VRAM probes on Mac (use `metal-unified`; see apple-silicon guide).
