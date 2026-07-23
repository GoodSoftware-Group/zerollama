# GGUF ggml Metal gated through UMA broker (M21)

**Status:** PoC (Jul 2026). **Admission only** — same machine-wide `uma_daemon` as [mlx-uma-sched.md](./mlx-uma-sched.md). Does **not** run ggml ops inside `uma_sched`.

```text
ollamarunner / llamarunner (Darwin Metal)
  Acquire → libuma_client
  LeaseBegin(prefill|decode) → HOLD_GPU
    ollamarunner: ComputeWithNotify (async + sync)
    llamarunner:  llama.Decode + Synchronize
  → RELEASE
```

## Scope

| Covered | Not covered (yet) |
|---------|-------------------|
| Darwin **ollamarunner** (`ComputeWithNotify`) | llama-server / Python runtime Metal |
| Darwin **llamarunner** (`llama.Decode` + sync under `RunGPU`) | C++ `ggml-metal` vendor wrap |
| Eager synchronize under HOLD (no RELEASE race) | |
| Same `ZEROLLAMA_UMA_SCHED` / `BUILD_UMA` as M20 | |
| Coarse `LeaseBegin(prefill\|decode)` project names | |

Default ticket projects: **`ollamarunner`** / **`llamarunner`** (override with `UMA_JOB_NAME`).

## Why Go wrap (not C++ Metal)

Reuse `x/mlxrunner/uma` + `-tags uma` without vendor patches. llama-server needs a separate cmake/`libuma_client` link later.

## Lab smoke

```bash
./scripts/phase/m21_ggml_uma_signoff.sh
# or:
OLLAMA_HOST=127.0.0.1:11435 ZEROLLAMA_UMA_SCHED=require ZEROLLAMA_UMA_SCHED_LOG=1 \
  ZEROLLAMA_LLAMA_SERVER=0 ZEROLLAMA_RUNTIME_DARWIN_SIDECAR=0 ./zerollama serve
```

Gate covers:
1. **ollamarunner** — creates **`m21-ggml:latest`** from `eliza-1-2b` with `spec_type=off` (avoids Darwin draft-eagle → llama-server).
2. **llamarunner** — plain `llama3.2:3b` (legacy Go runner, CGO `llama_decode`).

Requests use `"raw":true` to skip native Jinja `ApplyChatTemplate`.

## Disabling

Same knobs as [mlx-uma-sched.md](./mlx-uma-sched.md#disabling-uma-build--runtime):

- **Runtime:** `ZEROLLAMA_UMA_SCHED=off` (no HOLD; no rebuild)
- **Build:** `BUILD_UMA=0 ./scripts/build/build_zerollama_mac.sh` (no `-tags uma`)

## Wishlist

`bmtl/.../uma_toolkit/WISHLIST_GGML.md`
