# llama.cpp backend (experimental)

Route **eligible GGUF text** through **pinned sibling [llama.cpp](../runtime/LLAMA_CPP_PIN.md)** (Python runtime) instead of the in-process **ggml Metal** runner.

This is a **zerollama test harness** toward [Phase 17 upstream GGUF alignment](./ROADMAP.md#phase-17--upstream-gguf-path-alignment-directional). Upstream Ollama uses **Go → llama-server** directly (no Python hop). See [upstream-ollama-diff.md](./upstream-ollama-diff.md).

---

## Enable

**Flag (recommended for testing):**

```bash
./zerollama serve --llama-cpp-backend
```

**Env (same behavior):**

```bash
export ZEROLLAMA_LLAMA_CPP_BACKEND=1
./zerollama serve
```

**Helper script (Mac):**

```bash
./scripts/serve/serve_llama_cpp_backend.sh
```

Sets `ZEROLLAMA_RUNTIME=1`, `ZEROLLAMA_AUTO_CONFIG=1`, and `LLAMA_CPP_ROOT=../llama.cpp` when present.

---

## What changes

| Request | `--llama-cpp-backend` | Default serve |
|---------|------------------------|---------------|
| Text GGUF chat/generate | Python runtime → llama.cpp (`inprocess` or `subprocess` per YAML) | ggml Metal runner **or** runtime (Phase 12 default-on) |
| Vision / thinking / embed | Still ggml when required | Same |
| safetensors (MLX) | mlxrunner (unchanged) | Same |

Scheduler **skips ggml runner load** for deferred models (`ErrRuntimeInferenceModel` path) — same as explicit `zerollama-runtime` Modelfile backend.

**Revert to ggml-only:** unset the flag; or `ZEROLLAMA_LEGACY_RUNNER=1` forces ggml even for runtime-tagged models.

**Long-term:** Phase 17 ports upstream’s `llm/llama_server.go` so default GGUF is **Go → llama-server**, not Go → Python → llama. This flag remains useful until that lands.

---

## Prerequisites

```bash
LLAMA_CPP_ROOT=../llama.cpp ./scripts/build/build_llama_server.sh
./scripts/mlx/ensure_mlx_sources.sh   # only if also testing MLX
./zerollama doctor
```

On Darwin, `apple_silicon.yaml` sets `llama_backend: inprocess` when autoconfig picks darwin. Override:

```bash
export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=subprocess   # llama-server subprocess
# or inprocess (default from YAML)
```

Upstream pins llama.cpp at **`b9509`** (repo-root `LLAMA_CPP_VERSION`). Zerollama’s runtime pin is older — see [LLAMA_CPP_PIN.md](../runtime/LLAMA_CPP_PIN.md) and [upstream-ollama-diff.md](./upstream-ollama-diff.md#pin-and-integration-gaps).

---

## Benchmark vs ggml

Start each server in its own terminal. Use a **text-only** model you already pulled (e.g. `llama3.2:3b`).

**ggml baseline** — restart serve with legacy runner so requests do not hit the Python sidecar:

```bash
ZEROLLAMA_LEGACY_RUNNER=1 ./zerollama serve
go run ./cmd/bench -model llama3.2:3b -epochs 3 -format csv -output ggml.csv
```

**llama.cpp backend (Python runtime):**

```bash
./scripts/serve/serve_llama_cpp_backend.sh
go run ./cmd/bench -model llama3.2:3b -epochs 3 -format csv -output llamacpp.csv
```

**Upstream Ollama (Go → llama-server)** on another port:

```bash
cd ../ollama-upstream && go build -o ollama .
OLLAMA_HOST=127.0.0.1:11435 ./ollama serve
go run ./cmd/bench -host 127.0.0.1:11435 -model llama3.2:3b -epochs 3 -format csv -output upstream.csv
```

Equivalent without `-host`: `OLLAMA_HOST=127.0.0.1:11435 go run ./cmd/bench ...`

If you see `could not admit request (KV pool or pause)`, the request reached the **Python runtime** (`:8081`) and admission failed (GPU busy or sidecar paused). For ggml-only runs use `ZEROLLAMA_LEGACY_RUNNER=1` on serve, or check `curl -s http://127.0.0.1:8081/health | jq`.

Check runtime health: `curl -s http://127.0.0.1:8081/health | jq '.llama_backend, .llama_backend_source'`

---

## Upstream Ollama comparison

```bash
./scripts/gpu/clone_upstream_ollama.sh
./scripts/build/build_upstream_ollama_mac.sh
OLLAMA_HOST=127.0.0.1:11435 ../ollama-upstream/ollama serve
```

| Path | Hops | When to use |
|------|------|-------------|
| Zerollama ggml (default Mac) | Go → ggml runner | Daily dev; vision/thinking |
| Zerollama `--llama-cpp-backend` | Go → Python → llama | PA/admission experiments; pre–Phase 17 benchmark |
| Upstream serve | Go → llama-server | Target default for plain text GGUF |

Full architecture diff: [upstream-ollama-diff.md](./upstream-ollama-diff.md).
