# llama.cpp pin (runtime)

The Python runtime shells out to **`llama-server`** from a pinned llama.cpp tree (GGUF forward, quant kernels). Phase 17 targets upstream-style **Go → llama-server** integration as well — see [docs/upstream-ollama-diff.md](../docs/upstream-ollama-diff.md).

| Field | Value |
|-------|--------|
| **Recommended tree** | `../../llama.cpp` (sibling of `zerollama` on the host) |
| **Zerollama in-process ggml pin** | **`b9611`** — repo-root `LLAMA_CPP_VERSION`, vendor `vendor/llama-cpp-b9611/` |
| **Sibling `../llama.cpp` (runtime)** | **`b9611`** @ `02182fc5` — rebuild with `./scripts/build_llama_server.sh` |
| **Upstream Ollama pin** | **`b9509`** (vanilla Ollama lags zerollama ggml pin) |
| **In-tree ggml commit (patched)** | `1aefee58` — see `llama/build-info.cpp` |
| **Binary** | `build/bin/llama-server` (use `./scripts/build_llama_server.sh` from zerollama) |
| **CUDA arch (4090)** | `CMAKE_CUDA_ARCHITECTURES=89-real` (script default; override if needed) |

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
| `LLAMA_CPP_ROOT` | Root of llama.cpp checkout (default: `../../llama.cpp` relative to repo) |
| `LLAMA_SERVER_BIN` | Override path to `llama-server` executable |
| `OLLAMA_MLX_SOURCE` / `OLLAMA_MLX_C_SOURCE` | Override MLX sibling paths |

Rebuild llama.cpp when bumping this commit; run runtime integration tests on dual-GPU hosts.

## Bump checklist

1. Update `LLAMA_CPP_VERSION` + `Makefile.sync` `FETCH_HEAD`
2. `make -f Makefile.sync clean checkout apply-patches` (fix conflicts in vendor)
3. `./scripts/sync_vendor_llama.sh` → fix CGO breaks → `format-patch` if needed
4. Update `llama/build-info.cpp` BUILD_NUMBER/COMMIT
5. `./scripts/build_zerollama_mac.sh` && `./zerollama doctor`
6. For runtime sibling: `LLAMA_CPP_ROOT=../llama.cpp ./scripts/build_llama_server.sh`
