# GGUF llama-server Metal gated through UMA broker (M22)

**Status:** PoC (Jul 2026). **Admission only** — same `uma_daemon` as [mlx-uma-sched.md](./mlx-uma-sched.md) / [ggml-uma-sched.md](./ggml-uma-sched.md).

```text
llama-server (Darwin Metal)
  Acquire (UMA_JOB_NAME=llama-server)
  llama_context::graph_compute:
    LeaseBegin(prefill|decode) → async compute → sched_synchronize → LeaseEnd
```

## Build

```bash
BUILD_UMA=auto ./scripts/build/build_llama_server.sh   # Darwin Metal; links x/mlxrunner/uma/libuma_llama.a
```

Requires vendor patch `llama/patches/0094-darwin-uma-hold-graph-compute.patch` (applied by `apply_llama_vendor_patches.sh`).

**Important:** UMA glue is linked **only into `libllama`**. Acquire runs in `llama_backend_init` so leases share one client (do not also static-link `libuma_llama.a` into the server executable — duplicate `g_client` breaks HOLD).

## Runtime

Same env as M20/M21: `ZEROLLAMA_UMA_SCHED=auto|require|degraded|off`, `ZEROLLAMA_UMA_SCHED_LOG=1`.

### Disabling

| Knob | Effect |
|------|--------|
| `ZEROLLAMA_UMA_SCHED=off` | No acquire in `llama_backend_init`; `graph_compute` runs ungated (no rebuild) |
| `BUILD_UMA=0 ./scripts/build/build_llama_server.sh` | No `ZEROLLAMA_UMA` / no `libuma_llama.a` in the binary |

See [mlx-uma-sched.md — Disabling UMA](./mlx-uma-sched.md#disabling-uma-build--runtime).

Lab smoke (port **18082**, never 11434/8081):

```bash
./scripts/phase/m22_llama_server_uma_signoff.sh
# or skip rebuild:
M22_SKIP_BUILD=1 ./scripts/phase/m22_llama_server_uma_signoff.sh
```

**Python runtime:** subprocess and **inprocess** (ctypes → same `libllama.dylib`) both inherit M22 when the dylib was built with `BUILD_UMA`. Set `ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess` and point `LLAMA_CPP_LIB` / vendor paths at that build. Optional: `UMA_JOB_NAME=inprocess` (auto when backend env is `inprocess`). Rebuild with `BUILD_UMA=auto` after patch **0094**.

## Wishlist

`bmtl/.../uma_toolkit/WISHLIST_GGML.md` — llama-server HOLD row.
