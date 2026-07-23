# GGUF ggml Metal gated through UMA broker (M21)

**Status:** PoC (Jul 2026). **Admission only** — same machine-wide `uma_daemon` as [mlx-uma-sched.md](./mlx-uma-sched.md). Does **not** run ggml ops inside `uma_sched`.

```text
ollamarunner (in-process Metal)
  Acquire → libuma_client
  ComputeWithNotify → HOLD_GPU → graph_compute_async + sched_synchronize → RELEASE
```

## Scope

| Covered | Not covered (yet) |
|---------|-------------------|
| Darwin **ollamarunner** CGO Metal (`ml/backend/ggml` `ComputeWithNotify`) | llama-server / Python runtime Metal |
| Eager synchronize under HOLD (no RELEASE race) | C++ `ggml-metal` vendor wrap |
| Same `ZEROLLAMA_UMA_SCHED` / `BUILD_UMA` as M20 | Coarse prefill/decode project names (optional later) |

Default ticket project: **`ollamarunner`** (override with `UMA_JOB_NAME`).

## Why Go wrap (not C++ Metal)

Reuse `x/mlxrunner/uma` + `-tags uma` without vendor patches. llama-server needs a separate cmake/`libuma_client` link later.

## Lab smoke

```bash
# lab :11435 — GGUF text model (e.g. llama3.2:3b / eliza-1-2b)
OLLAMA_HOST=127.0.0.1:11435 ZEROLLAMA_UMA_SCHED=require ZEROLLAMA_UMA_SCHED_LOG=1 \
  ZEROLLAMA_RUNTIME_DARWIN_SIDECAR=0 ./zerollama serve
# generate; serve log should show uma broker gate + ollamarunner HOLD tickets
```

Never bind production `:11434` / `:8081` from agent lab scripts.

## Wishlist

`bmtl/.../uma_toolkit/WISHLIST_GGML.md`
