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

Lab smoke (port **18082**, never 11434/8081):

```bash
BUILD_UMA=1 ./scripts/build/build_llama_server.sh
ZEROLLAMA_UMA_SCHED=require ZEROLLAMA_UMA_SCHED_LOG=1 \
  ./vendor/llama-cpp-*/build/bin/llama-server --model "$GGUF" --port 18082 --host 127.0.0.1 -c 1024
# competitor HOLD_GPU ~5s → /completion should queue (wall ≥2s)
```

## Wishlist

`bmtl/.../uma_toolkit/WISHLIST_GGML.md` — llama-server HOLD row.
