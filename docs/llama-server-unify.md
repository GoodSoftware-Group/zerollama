# Unify llama-server (one binary)

**Goal:** retire `run/llama-server-tts/` — one vendor `llama-server` for Q2_0 ternary + Kokoro TTS.

## Patch set (on pin `86d86ed4` / HEAD after prior series)

| Patch | What |
|-------|------|
| **0082** (existing) | CUDA `Q2_0` g64 |
| **0099** | `GGML_CUDA_FORCE_CUBLAS` **getenv** runtime (5080 `serve.sh`) |
| **0100** | `tools/kokoro/` from Eliza (CPU iSTFT; `GGML_HAS_ISTFT` optional later) |
| **0101** | `LLAMA_BUILD_KOKORO` + `/v1/audio/speech` mount |
| **0102** | Bee B1 adaptive DFlash draft-max (`--spec-dm-adaptive`; default **off**) |

OmniVoice (~17k) is **not** in this series.

## Apply

```bash
# From a clean vendor checkout at the pin + patches through 0098:
git -C vendor/llama-cpp-86d86ed4 am llama/patches/0099-*.patch
git -C vendor/llama-cpp-86d86ed4 am llama/patches/0100-*.patch
git -C vendor/llama-cpp-86d86ed4 am llama/patches/0101-*.patch
git -C vendor/llama-cpp-86d86ed4 am llama/patches/0102-*.patch
# or: make -f Makefile.sync apply-patches / scripts/vendor/sync_vendor_llama.sh
```

### Chat / Metal (default)

```bash
./scripts/build/build_llama_server.sh   # LLAMA_BUILD_KOKORO=OFF
```

### TTS unify (5080 / prism)

```bash
LLAMA_BUILD_KOKORO=ON ./scripts/build/build_llama_server.sh
# CUDA example:
# LLAMA_BUILD_KOKORO=ON CMAKE_CUDA_ARCHITECTURES=120-real ./scripts/build/build_llama_server.sh
```

Then set `LLAMA_SERVER_BIN` to that binary and drop `ZEROLLAMA_KEEP_LLAMA_SERVER_BIN=1` (so the TTS copy is no longer pinned).

### B1 lab (DFlash only)

```bash
# Lab ports only — never 11434 / 8081
ZEROLLAMA_SPEC_DM_ADAPTIVE=profit \
  # … serve / llama-server with --spec-type draft-dflash …
```

Or pass `--spec-dm-adaptive profit` directly to `llama-server`.

See [prism-ternary.md](./prism-ternary.md), [llama-fork-watchlist.md](./llama-fork-watchlist.md).
