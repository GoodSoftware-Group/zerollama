# llama.cpp pin (runtime)

The Python runtime shells out to **`llama-server`** from a pinned llama.cpp tree (GGUF forward, quant kernels).

| Field | Value |
|-------|--------|
| **Recommended tree** | `../../llama.cpp` (sibling of `zerollama` on the host) |
| **Recorded commit** | `4f13cb742476d81a6b42a2aa5996e82a478c2481` |
| **Binary** | `build/bin/llama-server` (use `./scripts/build_llama_server.sh` from zerollama) |
| **CUDA arch (4090)** | `CMAKE_CUDA_ARCHITECTURES=89-real` (script default; override if needed) |

## Environment

| Variable | Purpose |
|----------|---------|
| `LLAMA_CPP_ROOT` | Root of llama.cpp checkout (default: `../../llama.cpp` relative to repo) |
| `LLAMA_SERVER_BIN` | Override path to `llama-server` executable |

Rebuild llama.cpp when bumping this commit; run runtime integration tests on dual-GPU hosts.
