# UMA admission on Darwin (operator overview)

**Admission only** — machine-wide `uma_daemon` decides *when* Metal may run (`HOLD_GPU` / `RELEASE`). Kernels stay in MLX / ggml / llama.cpp.

| Track | Surface | Gate |
|-------|---------|------|
| **M20** | mlxrunner | `./scripts/phase/m20_uma_signoff.sh` |
| **M21** | ollamarunner + llamarunner | `./scripts/phase/m21_ggml_uma_signoff.sh` |
| **M22** | llama-server + **runtime inprocess/subprocess** via `libllama` | `./scripts/phase/m22_llama_server_uma_signoff.sh` |

Python runtime **subprocess** (`llama-server` child) and **inprocess** (ctypes → same `libllama.dylib`) both inherit M22 when the dylib is UMA-linked. No separate Python HOLD wrap.

Client glue lives in **`x/uma`** (`libuma_embed.a` for Go `-tags uma`, `libuma_llama.a` for llama-server).

## Defaults

- **Runtime:** `ZEROLLAMA_UMA_SCHED` unset → **`auto`** (gate if broker up; else ungated)
- **Build:** `BUILD_UMA=auto` on Mac scripts when sibling `bmtl/.../uma_toolkit` exists

## Disable

| Knob | Effect |
|------|--------|
| `ZEROLLAMA_UMA_SCHED=off` (`0` / `false` / `disabled` / `none`) | No connect, no HOLD — **no rebuild** |
| `BUILD_UMA=0` | Compile out client (`-tags uma` / `libuma_llama.a`) |

Details: [mlx-uma-sched.md § Disabling](./mlx-uma-sched.md#disabling-uma-build--runtime).

## Lab ports

Never use production **`:11434`** / **`:8081`**. Sign-offs use `:11435` (Go) and `:18082` (llama-server).

## Docs

- [mlx-uma-sched.md](./mlx-uma-sched.md) · [ggml-uma-sched.md](./ggml-uma-sched.md) · [llama-server-uma-sched.md](./llama-server-uma-sched.md)
- Wishlist: `bmtl/.../uma_toolkit/WISHLIST_MLX.md`, `WISHLIST_GGML.md`
