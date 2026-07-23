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
./scripts/phase/m22_llama_server_uma_signoff.sh
# or skip rebuild:
M22_SKIP_BUILD=1 ./scripts/phase/m22_llama_server_uma_signoff.sh
```

**Python runtime:** if `LLAMA_SERVER_BIN` (or the vendor build path the runtime discovers) is this UMA-linked binary, the sidecar’s subprocess Metal path is admitted the same way — no separate Python wrap. Rebuild runtime’s llama-server with `BUILD_UMA=auto` after pulling patch **0094**.

## Wishlist

`bmtl/.../uma_toolkit/WISHLIST_GGML.md` — llama-server HOLD row.
