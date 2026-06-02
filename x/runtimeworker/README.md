# `runtimeworker` — embedded inference runtime

Starts **`runtime`** (FastAPI + uvicorn) on a **daemon thread** inside the `zerollama` process via CGO (`python3-embed`). Shares the same CPython interpreter as **`trainingworker`** when both are enabled.

## Why loopback HTTP inside one process?

- Reuses all existing `/api/generate`, streaming, and `/v1/chat/completions` Python code without rewriting the stack in C.
- Go keeps using the same HTTP proxy paths; `runtimeworker.BaseURL()` returns `http://127.0.0.1:8081` (configurable).

## Enable

```bash
export ZEROLLAMA_RUNTIME_EMBED=1   # default on when ZEROLLAMA_RUNTIME_URL is unset
export LLAMA_SERVER_BIN=.../llama-server
export LLAMA_MODEL=.../model.gguf
zerollama serve
```

Disable sidecar-only embed: `ZEROLLAMA_RUNTIME_EMBED=0` and set `ZEROLLAMA_RUNTIME_URL` to an external process.

## Build

Same as training: `python3-dev`, `pkg-config python3-embed`, runtime deps `pip install -e runtime/.[serve]'`.
