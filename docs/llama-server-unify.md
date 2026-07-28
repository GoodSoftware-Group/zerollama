# Unify llama-server (one binary)

**Goal:** retire `run/llama-server-tts/` — one vendor `llama-server` for Q2_0 ternary + Kokoro TTS.

## Patch set (on pin `86d86ed4` / HEAD after prior series)

| Patch | What |
|-------|------|
| **0082** (existing) | CUDA `Q2_0` g64 |
| **0099** | `GGML_CUDA_FORCE_CUBLAS` **getenv** runtime (5080 `serve.sh`) |
| **0100** | `tools/kokoro/` from Eliza (CPU iSTFT; `GGML_HAS_ISTFT` optional later) |
| **0101** | `LLAMA_BUILD_KOKORO` + `/v1/audio/speech` mount |

Generated from worktree commits on vendor HEAD `d967642b3` (2026-07-28). OmniVoice (~17k) is **not** in this series.

## Apply

```bash
# From a clean vendor checkout at the pin + patches through 0098:
git -C vendor/llama-cpp-86d86ed4 am llama/patches/0099-*.patch
git -C vendor/llama-cpp-86d86ed4 am llama/patches/0100-*.patch
git -C vendor/llama-cpp-86d86ed4 am llama/patches/0101-*.patch
# or: make -f Makefile.sync apply-patches / scripts/vendor/sync_vendor_llama.sh

cmake -S vendor/llama-cpp-86d86ed4 -B vendor/llama-cpp-86d86ed4/build \
  -DGGML_CUDA=ON -DLLAMA_BUILD_KOKORO=ON -DCMAKE_CUDA_ARCHITECTURES=120-real
cmake --build vendor/llama-cpp-86d86ed4/build -j --target llama-server
```

Then set `LLAMA_SERVER_BIN` to that binary and drop `ZEROLLAMA_KEEP_LLAMA_SERVER_BIN=1` (so the TTS copy is no longer pinned).

See [prism-ternary.md](./prism-ternary.md).
