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

## Port conflicts (`address already in use` on :8081)

Embed starts uvicorn on loopback **8081** (or `ZEROLLAMA_RUNTIME_EMBED_PORT`). If another `zerollama serve` or `zerollama-runtime serve` already holds the port:

1. uvicorn logs `ERROR: [Errno 98] ... address already in use`
2. **Current zerollama:** Go refuses embed when the port is busy **or** `/health` on `:8081` lacks matching `embed_boot` (avoids attaching to a stale runtime).

**Fix:** stop the stale process before starting serve:

```bash
ss -tlnp | grep ':8081'
pkill -f 'zerollama serve'    # or kill the sidecar PID
```

`scripts/serve_production_wrapper.sh` (installed as `~/bin/serve.sh`) and `scripts/serve_gpu_example.sh` warn (and the example script may `pkill` stale zerollama). **WHY wrapper:** copying the example to `~/bin` breaks repo-root detection — see [5080-runbook — Production serve](./5080-runbook.md#production-serve-binserve-sh). **`systemctl stop ollama` does not stop zerollama** — common footgun on cudallama-style `~/bin/serve.sh` wrappers.

## Remote clients (Go API vs embedded runtime)

| Listener | Default | Remote clients? |
|----------|---------|-----------------|
| Go daemon (`zerollama serve`) | `127.0.0.1:11434` upstream; **5080 CT uses `0.0.0.0:8080`** | **Yes** — set `OLLAMA_HOST=0.0.0.0:8080` (or your CT IP) |
| Embedded Python runtime | `127.0.0.1:8081` | **No** — loopback only; Go proxies |

**Why two ports:** Go owns registry, scheduling, training HTTP, Eliza cloud merge, and VRAM broker. Python runtime is an internal inference worker. Remote apps (`ZEROLLAMA_API_ENDPOINT`, `OLLAMA_HOST`, Open WebUI) must target **Go `:8080`**, not `:8081`.

**Verify bind after restart:**

```bash
ss -tlnp | grep 8080    # expect *:8080 or 0.0.0.0:8080
curl -s http://<host-ip>:8080/api/tags
```

**Logs when stdout is redirected:** CT `~/bin/serve.sh` uses `>> /tmp/zerollama-serve.log` — `tail -f` that file; screen will look idle. See [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md#production-serve-binserve-sh).
