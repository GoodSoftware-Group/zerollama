# Embedded Python runtime (single process)

`zerollama serve` can run the **inference runtime inside the same OS process** as Go, using embedded CPython (same mechanism as GPU training).

## How it works

```text
zerollama (Go)
  ├── CGO libpython (one interpreter)
  ├── training.py (optional, OLLAMA_TRAINING)
  └── runtime package → uvicorn on 127.0.0.1:8081 (daemon thread)
         ↑
  Go proxies /api/generate, /api/chat, /v1/... to this loopback URL
```

No separate `zerollama-runtime` process is required when embed mode is on.

**Why embed instead of always sidecar:** one OS process for operators (single restart, shared logs), and optional sharing of one CPython with training. **VRAM:** automatic Go broker (Phase 8) coordinates runtime vs legacy runners; see [testing-smoke.md](./testing-smoke.md). **Product shape:** inference and training are separate **queues** sharing GPU time—see [ROADMAP.md](./ROADMAP.md#product-model-queues-stakeholders-and-gpu-time).

## Enable

```bash
export ZEROLLAMA_RUNTIME_EMBED=1    # default on if ZEROLLAMA_RUNTIME_URL is unset
export LLAMA_SERVER_BIN=.../llama-server
export LLAMA_MODEL=.../model.gguf
zerollama serve
```

## Disable embed (external sidecar)

```bash
export ZEROLLAMA_RUNTIME_EMBED=0
export ZEROLLAMA_RUNTIME_URL=http://127.0.0.1:8081
zerollama-runtime serve   # separate terminal
zerollama serve
```

## Build requirements

Same as training: `python3-dev`, `pkg-config python3-embed`, and:

```bash
cd runtime && pip install -e ".[serve]"
```

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `ZEROLLAMA_RUNTIME_EMBED` | on if no `ZEROLLAMA_RUNTIME_URL` | In-process runtime |
| `ZEROLLAMA_RUNTIME_EMBED_PORT` | `8081` | Loopback port |
| `OLLAMA_TRAINING_PYTHONPATH` | auto | Repo root (contains `runtime/`) |

## GPU sharing (runtime vs legacy runner)

Both use the same GPU. **Target:** a **Go-owned VRAM broker** evicts the other stack before each load—no operator `unload`/`resume` API. **Today (interim):** internal hooks only; see [testing-smoke.md](./testing-smoke.md) if you must free VRAM manually before the broker ships.

Single-GPU hosts: prefer `runtime/configs/single_gpu.yaml` over `dual_4090.yaml`. **Why:** tensor split (`-sm tensor -ts 1,1`) requires multiple GPUs.
