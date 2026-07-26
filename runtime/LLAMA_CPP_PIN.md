# llama.cpp pin (runtime)

The Python runtime shells out to **`llama-server`** from a pinned llama.cpp tree (GGUF forward, quant kernels). Phase 17 targets upstream-style **Go → llama-server** integration as well — see [docs/upstream-ollama-diff.md](../docs/upstream-ollama-diff.md).

## Unified runtime binary (one tree)

| Field | Value |
|-------|--------|
| **Recommended build tree** | `vendor/llama-cpp-86d86ed4/` (patched) — `./scripts/build/build_llama_server.sh` |
| **Optional sibling** | `../llama.cpp` @ `LLAMA_CPP_COMMIT` (unpatched; prefer vendor) |
| **Upstream repo** | `https://github.com/ggml-org/llama.cpp.git` |
| **Runtime commit** | **`LLAMA_CPP_COMMIT`** → `86d86ed4396b4130922f7b9af26e3d9fc11a591b` (master tip; past tag `b10064`) |
| **Binary** | `build/bin/llama-server` — `./scripts/build/build_llama_server.sh` |
| **Ollama patches** | `llama/patches/` via `Makefile.sync` + `./scripts/vendor/sync_vendor_llama.sh` (through **0098** DCA LSE tile+budget; **0097** native DCA graph; **0096** FA LSE export; **0095** dca hparams; **0094** GGUF dca keys; **0090** media-aware `/kv/seq-copy`; **0089** L3-R6b cell+tensor+pages COW; **0088** TBQ vec_dot; **0087** Bee loop-guard); container build: `./scripts/vendor/build_llama_server_container.sh`) |
| **Why ggml-org master** | Track upstream llama.cpp tip. Eliza QJL/Polar/TBQ applied as patches **0026–0030**; CUDA L2 completeness and Metal TBQ SET_ROWS follow in the mid series; native FP8 weights **0076–0079** (types 51/52 — see [native-fp8-gguf.md](../docs/native-fp8-gguf.md)); hardware PR ports **0080–0086**; Bee reasoning-loop guard **0087**; TBQ vec_dot dedupe **0088**; L3-R6b cell+tensor+pages COW **0089**; media-aware `/kv/seq-copy` **0090**. |

## In-process ggml (Go CGO) — unified with runtime

| Field | Value |
|-------|--------|
| **Vendor pin** | **`86d86ed4`** — `LLAMA_CPP_VERSION`, `LLAMA_CPP_COMMIT`, `vendor/llama-cpp-86d86ed4/` |
| **Upstream repo** | `https://github.com/ggml-org/llama.cpp.git` (same as runtime sibling) |
| **Ollama patches** | `llama/patches/` via `Makefile.sync` + `./scripts/vendor/sync_vendor_llama.sh` (through **0098** DCA LSE tile+budget; **0097** native DCA graph; **0096** FA LSE export; **0095** dca hparams; **0094** GGUF dca keys; **0090** media-aware `/kv/seq-copy`; **0089** L3-R6b cell+tensor+pages COW; **0088** TBQ vec_dot; **0087** Bee loop-guard); container build: `./scripts/vendor/build_llama_server_container.sh`) |
| **In-tree Metal dig** | E8_2 / TQ2 Metal kernels and concurrency guard in the mid series; Mac build embeds compiled metallib. Native FP8 weight types **51/52** (0076–0079). Metal FA-vec per-device (Q,NE) tables **0086** (ported onto monolithic `ggml-metal.metal`; keeps GQA2). |
| **Rebase helper** | `./scripts/vendor/rebase_vendor_unified.sh --sync` |

Runtime `llama-server` and in-process ggml share **one ggml-org `86d86ed4` base** + zerollama patches.

Upstream also ships **`llama/compat/`** — in-memory GGUF translation at CMake fetch time for **llama-server**. In-process **ggml** uses `llama/patches/` on a vendored tree synced via [docs/ggml-b9509-migration.md](../docs/ggml-b9509-migration.md).

## MLX pins (safetensors / mlxrunner)

| Field | Value |
|-------|--------|
| **MLX_VERSION** | `de7b4ed986b6d6f55b8ace5e73c24d1ca0bea89b` (upstream Ollama v0.31.2) |
| **MLX_C_VERSION** | `fba4470b89073180056c9ea46c443051375f7399` (upstream Ollama) |
| **Fetch** | `./scripts/mlx/ensure_mlx_sources.sh` (sibling `../mlx`, `../mlx-c`) |

**Why separate from llama.cpp:** MLX drives **safetensors** via `mlxrunner`, not GGUF ggml. Pin bumps require a **native dylib rebuild** — use `BUILD_MLX=1 ./scripts/build/build_zerollama_mac.sh` (dev) or `./scripts/build/build_production_mac.sh` (release).

**Rebuild (Darwin arm64):**

```bash
./scripts/mlx/ensure_mlx_sources.sh
git -C ../mlx checkout $(cat MLX_VERSION)
git -C ../mlx-c checkout $(cat MLX_C_VERSION)
BUILD_MLX=1 ./scripts/build/build_zerollama_mac.sh
./zerollama doctor   # mlx engine → build/metal-v*/lib/ollama/...
```

See [docs/apple-silicon-metal.md](../docs/apple-silicon-metal.md#mlx-engine-optional).

## Environment

| Variable | Purpose |
|----------|---------|
| `LLAMA_CPP_ROOT` | Root of llama.cpp checkout (default: `../llama.cpp` relative to repo) |
| `LLAMA_SERVER_BIN` | Override path to `llama-server` executable |
| `LLAMA_CPP_REPO` | Override clone URL (default: elizaOS/llama.cpp) |
| `ZEROLLAMA_LLAMA_FORK` | `0` = L1 q8_0 profiles; unset/`1` = auto-probe fork KV types |
| `OLLAMA_MLX_SOURCE` / `OLLAMA_MLX_C_SOURCE` | Override MLX sibling paths |

Rebuild llama.cpp when bumping `LLAMA_CPP_COMMIT`; run runtime integration tests on dual-GPU hosts.

## Bump checklist (runtime sibling)

1. Update `LLAMA_CPP_COMMIT` (and tag file `LLAMA_CPP_VERSION` if tagging)
2. `./scripts/build/build_llama_server.sh` — probes QJL + checkpoint flags in `--help`
3. `./scripts/phase/l2_fork_eval.sh` — profile argv smoke
4. `./scripts/phase/l2_full_gate.sh` or `./scripts/phase/l2_cuda_full_gate.sh` on GPU hosts

## Bump checklist (in-process vendor — when rebasing)

1. Update `LLAMA_CPP_VERSION` + `Makefile.sync` `FETCH_HEAD` / upstream URL
2. `make -f Makefile.sync clean apply-patches` (fix conflicts in vendor)
3. `./scripts/vendor/sync_vendor_llama.sh` → fix CGO breaks → `format-patch` if needed
4. Update `llama/build-info.cpp` BUILD_NUMBER/COMMIT
5. `./scripts/build/build_zerollama_mac.sh` && `./zerollama doctor`
