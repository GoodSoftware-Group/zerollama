# llama.cpp pin (runtime)

The Python runtime shells out to **`llama-server`** from a pinned llama.cpp tree (GGUF forward, quant kernels). Phase 17 targets upstream-style **Go → llama-server** integration as well — see [docs/upstream-ollama-diff.md](../docs/upstream-ollama-diff.md).

## Unified runtime binary (one tree)

| Field | Value |
|-------|--------|
| **Recommended build tree** | `vendor/llama-cpp-c84b3020/` (patched) — `./scripts/build_llama_server.sh` |
| **Optional sibling** | `../llama.cpp` @ `LLAMA_CPP_COMMIT` (unpatched eliza only; prefer vendor) |
| **Upstream repo** | `https://github.com/elizaOS/llama.cpp.git` |
| **Runtime commit** | **`LLAMA_CPP_COMMIT`** → `c84b30200c8d512c00c9d61c96bed078f1c0024d` |
| **Binary** | `build/bin/llama-server` — `./scripts/build_llama_server.sh` |
| **Why eliza base** | Superset of ggml-org: `dflash-draft`, QJL/Polar/TBQ KV, `--checkpoint-every-n-tokens`, upstream checkpoints. One binary; L1 vs fork GPU profiles are runtime flags (`ZEROLLAMA_LLAMA_FORK`), not separate builds. |

## In-process ggml (Go CGO) — unified with runtime

| Field | Value |
|-------|--------|
| **Vendor pin** | **`c84b3020`** — `LLAMA_CPP_VERSION`, `LLAMA_CPP_COMMIT`, `vendor/llama-cpp-c84b3020/` |
| **Upstream repo** | `https://github.com/elizaOS/llama.cpp.git` (same as runtime sibling) |
| **Ollama patches** | `llama/patches/0001–0016` via `Makefile.sync` + `./scripts/sync_vendor_llama.sh` |
| **Rebase helper** | `./scripts/rebase_vendor_unified.sh --sync` |

Runtime `llama-server` and in-process ggml now share **one elizaOS base commit** + zerollama patches.

Upstream also ships **`llama/compat/`** — in-memory GGUF translation at CMake fetch time for **llama-server**. In-process **ggml** uses `llama/patches/` on a vendored tree synced via [docs/ggml-b9509-migration.md](../docs/ggml-b9509-migration.md).

## MLX pins (safetensors / mlxrunner)

| Field | Value |
|-------|--------|
| **MLX_VERSION** | `2165dc08d7b33258260aa849d39f087d50e62962` (upstream Ollama) |
| **MLX_C_VERSION** | `fba4470b89073180056c9ea46c443051375f7399` (upstream Ollama) |
| **Fetch** | `./scripts/ensure_mlx_sources.sh` (sibling `../mlx`, `../mlx-c`) |

**Why separate from llama.cpp:** MLX drives **safetensors** via `mlxrunner`, not GGUF ggml. Pin bumps require a **native dylib rebuild** — use `BUILD_MLX=1 ./scripts/build_zerollama_mac.sh` (dev) or `./scripts/build_production_mac.sh` (release).

**Rebuild (Darwin arm64):**

```bash
./scripts/ensure_mlx_sources.sh
git -C ../mlx checkout $(cat MLX_VERSION)
git -C ../mlx-c checkout $(cat MLX_C_VERSION)
BUILD_MLX=1 ./scripts/build_zerollama_mac.sh
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
2. `./scripts/build_llama_server.sh` — probes QJL + checkpoint flags in `--help`
3. `./scripts/l2_fork_eval.sh` — profile argv smoke
4. `./scripts/l2_full_gate.sh` or `./scripts/l2_cuda_full_gate.sh` on GPU hosts

## Bump checklist (in-process vendor — when rebasing)

1. Update `LLAMA_CPP_VERSION` + `Makefile.sync` `FETCH_HEAD` / upstream URL
2. `make -f Makefile.sync clean apply-patches` (fix conflicts in vendor)
3. `./scripts/sync_vendor_llama.sh` → fix CGO breaks → `format-patch` if needed
4. Update `llama/build-info.cpp` BUILD_NUMBER/COMMIT
5. `./scripts/build_zerollama_mac.sh` && `./zerollama doctor`
