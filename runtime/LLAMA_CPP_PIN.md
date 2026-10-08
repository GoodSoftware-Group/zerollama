# llama.cpp pin (runtime)

The Python runtime shells out to **`llama-server`** from a pinned llama.cpp tree (GGUF forward, quant kernels). Phase 17 targets upstream-style **Go → llama-server** integration as well — see [docs/upstream-ollama-diff.md](../docs/upstream-ollama-diff.md).

## Unified runtime binary (one tree)

| Field | Value |
|-------|--------|
| **Recommended build tree** | `vendor/llama-cpp-b11351/` (patched) — `./scripts/build/build_llama_server.sh` |
| **Optional sibling** | `../llama.cpp` @ `LLAMA_CPP_COMMIT` (unpatched; prefer vendor) |
| **Upstream repo** | `https://github.com/ggml-org/llama.cpp.git` |
| **Runtime commit** | **`LLAMA_CPP_COMMIT`** → `631109b34da437a3c4a5ebd75091d677671392e3` (ggml-org tag **b11351**; L6 tip / Ollama v0.40.x) |
| **Binary** | `build/bin/llama-server` — `./scripts/build/build_llama_server.sh` |
| **Ollama patches** | `llama/patches/` via `Makefile.sync` + `./scripts/vendor/sync_vendor_llama.sh` (**137** on **b11351**, through **0137**; MultiDecode **0126–0128**; Laya **0124–0125**; Clef **0135** + ANE **0136** + Laya tip **0137**; pretok/BPE/unicode mid-series; FP8 / Kokoro / Bee / DCA / UMA earlier). Apply by filename — numbers shift on ladder rungs. Container: `./scripts/vendor/build_llama_server_container.sh`. BPE: [docs/faster-bpe-tokenize.md](../docs/faster-bpe-tokenize.md). MultiDecode: [docs/multidecode-llama-cpp.md](../docs/multidecode-llama-cpp.md). |
| **Why ggml-org master** | Track upstream llama.cpp tip. Eliza QJL/Polar/TBQ applied as patches **0026–0030**; CUDA L2 completeness and Metal TBQ SET_ROWS follow in the mid series; native FP8 weights **0076–0079** (types 51/52 — see [native-fp8-gguf.md](../docs/native-fp8-gguf.md)); hardware PR ports **0080–0086**; Bee reasoning-loop guard **0087**; TBQ vec_dot dedupe **0088**; L3-R6b cell+tensor+pages COW **0089**; media-aware `/kv/seq-copy` **0090**; llama-bench Eliza L2 KV type names **0093**; native DCA **0094–0098**; Metal recoverable nil pipeline + bf16 library gate + Polar/QJL SET_ROWS + fused QJL+Polar attn **0096–0098** (Lab D — `speed` runs on Mac, tok/s FAIL merge); Kokoro **0100–0101**; Bee B1 adaptive draft-max **0102**. **Hedge:** do not assume [elizaOS/llama.cpp](https://github.com/elizaOS/llama.cpp) keeps rebasing onto latest ggml-org — this pin remains source of truth. Sibling scout: `../eliza-llama.cpp` @ `ad56033` (OmniVoice/FFI stay out of our binary; Kokoro optional via `LLAMA_BUILD_KOKORO=ON`). |

## In-process ggml (Go CGO) — unified with runtime

| Field | Value |
|-------|--------|
| **Vendor pin** | **`b11351`** — `LLAMA_CPP_VERSION`, `LLAMA_CPP_COMMIT`, `vendor/llama-cpp-b11351/` (rollback: `vendor/llama-cpp-b11232/`, also b11081/b10969/…) |
| **Upstream repo** | `https://github.com/ggml-org/llama.cpp.git` (same as runtime sibling) |
| **Ollama patches** | Same series as runtime table (**137** format-patches on **b11351**; tip `common_batch` / `llama_process`; **0135** Clef + `ZEROLLAMA_CLEF_DIR`; **0136** ANE shim; **0137** Laya). Eliza QJL/Polar/TBQ shaders stay in `eliza-shipped/`. Pretok lives in **`src/unicode.*`**. Mac CGO: `llama/cgo_vendor_hash.cpp` + `llama_print_build_info(const char *, FILE *)`. Tip CGO also needs `ml/.../tiled/tiled.go` + `llama/.../parsers/parsers.go`. |
| **In-tree Metal dig** | Upstream split Metal kernels (`kernels/*.metal`). Eliza SET_ROWS + fused QJL+Polar attn remain as extra AIR objects. Native FP8 weight types **51/52**. FA-vec per-device tables are upstream. |
| **Rebase helper** | `./scripts/vendor/rebase_vendor_unified.sh --sync` |

Runtime `llama-server` and in-process ggml share **one ggml-org `b11351` base** + zerollama patches.

Upstream also ships **`llama/compat/`** — in-memory GGUF translation at CMake fetch time for **llama-server**. In-process **ggml** uses `llama/patches/` on a vendored tree synced via [docs/ggml-b9509-migration.md](../docs/ggml-b9509-migration.md).

## MLX pins (safetensors / mlxrunner)

| Field | Value |
|-------|--------|
| **MLX_VERSION** | `a59cc23190f1e26fabcba8693607bb4c34b2ac2b` (Ollama tip Oct 2026; Metal idle residency refresh upstream; sibling `../mlx`) |
| **MLX_C_VERSION** | `ebc88f10caa1b625e6b581437a8dea6df8a70085` (Ollama tip; carry patches in `mlx/compat/mlx-c/`) |
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
| `LLAMA_CPP_REPO` | Override clone URL (default historically elizaOS; **pin is ggml-org** via `Makefile.sync`) |
| `ZEROLLAMA_ALLOW_ELIZA_VOICE` | `1` = allow `tools/omnivoice` in build tree (default refuse). Kokoro is allowed; enable with `LLAMA_BUILD_KOKORO=ON` |
| `LLAMA_BUILD_KOKORO` | `ON` = mount `/v1/audio/speech` (patches **0100–0101**); default OFF |
| `ZEROLLAMA_SPEC_DM_ADAPTIVE` | `profit`/`1`/`on` = Bee B1 adaptive DFlash draft-max (**0102**); default off |
| `ZEROLLAMA_LLAMA_FORK` | `0` = L1 q8_0 profiles; unset/`1` = auto-probe fork KV types |
| `OLLAMA_MLX_SOURCE` / `OLLAMA_MLX_C_SOURCE` | Override MLX sibling paths |

Rebuild llama.cpp when bumping `LLAMA_CPP_COMMIT`; run runtime integration tests on dual-GPU hosts.

## L6 pin-debt ladder (toward Ollama tip **b11351**)

**Status:** **rung 5 tip on b11351** — **137** patches; Clef linked on tip `common_batch` API; matches Ollama v0.40.x pin. On RTX 5080 CT build with **CUDA 12.8** (not default CUDA 13.3).

| Item | Detail |
|------|--------|
| **Operator runbook** | [docs/llama-cpp-pin-ladder.md](../docs/llama-cpp-pin-ladder.md) — rungs `b10615 → b10864 → b10969 → b11081 → b11232 → b11351` |
| **Status script** | `./scripts/phase/l6_pin_ladder_status.sh` |
| **Clef staging** | `llama/clef/` — wire with `./scripts/phase/stage_clef_for_pin.sh --wire` |
| **Clef compile check** | `./scripts/phase/l6_clef_compile_check.sh` |

Clef **runtime wire + Cloudflare Flash product convert** are green on tip lab (`l6_clef_product_convert.sh`, `l6_clef_decisions_e2e.sh`). **Production CT** still uses `LLAMA_SERVER_BIN=…/llama-cpp-b10615` until operator runs `./scripts/phase/l6_promote_tip_env.sh --write` and restarts serve. Strands PointerRows on CUDA remain deferred (MLX on Mac). See [docs/system-one-score.md](../docs/system-one-score.md) · [docs/llama-cpp-pin-ladder.md](../docs/llama-cpp-pin-ladder.md).

## Bump checklist (runtime sibling)

1. Update `LLAMA_CPP_COMMIT` (and tag file `LLAMA_CPP_VERSION` if tagging)
2. `./scripts/build/build_llama_server.sh` — probes QJL + checkpoint flags in `--help`
3. `./scripts/phase/l2_fork_eval.sh` — profile argv smoke
4. `./scripts/phase/l2_full_gate.sh` or `./scripts/phase/l2_cuda_full_gate.sh` on GPU hosts

For a **pin-debt rung** (not tip chase), follow [docs/llama-cpp-pin-ladder.md](../docs/llama-cpp-pin-ladder.md) instead of jumping straight to **b11351**.

## Bump checklist (in-process vendor — when rebasing)

1. Update `LLAMA_CPP_VERSION` + `Makefile.sync` `FETCH_HEAD` / upstream URL
2. `make -f Makefile.sync clean apply-patches` (fix conflicts in vendor)
3. `./scripts/vendor/sync_vendor_llama.sh` → fix CGO breaks → `format-patch` if needed
4. Update `llama/build-info.cpp` BUILD_NUMBER/COMMIT
5. `./scripts/build/build_zerollama_mac.sh` && `./zerollama doctor`
6. At **b11232+**: `./scripts/phase/stage_clef_for_pin.sh --wire` and refresh Clef server hunks
