# llama.cpp pin (runtime)

The Python runtime shells out to **`llama-server`** from a pinned llama.cpp tree (GGUF forward, quant kernels). Phase 17 targets upstream-style **Go → llama-server** integration as well — see [docs/upstream-ollama-diff.md](../docs/upstream/ollama-diff.md).

## Unified runtime binary (one tree)

| Field | Value |
|-------|--------|
| **Recommended build tree** | `vendor/llama-cpp-b12119/` — `./scripts/build/build_llama_server.sh` |
| **Optional sibling** | `../llama.cpp` @ `LLAMA_CPP_COMMIT` (unpatched; the patched tree is `vendor/llama-cpp-b12119/`) |
| **Upstream repo** | `https://github.com/ggml-org/llama.cpp.git` |
| **Runtime commit** | **`LLAMA_CPP_COMMIT`** → `6f1e6543704bc5b080c7a98e7b6f707b31f6ed51` (ggml-org tag **b12119**, 2026-11-04). The pin stays on the release tag. Master’s remaining delta is CI `#31439` (`3de52ac`). `#30541` is cherry-picked as patch **0162**. |
| **Binary** | `build/bin/llama-server` — `./scripts/build/build_llama_server.sh` |
| **Ollama patches** | `llama/patches/` via `Makefile.sync` + `./scripts/vendor/sync_vendor_llama.sh` (**163** files, **0001–0163**, formatted on **b12119**, vendor tip `933b71f`). **0163** loads the full gemma4-assistant draft stack (`block_count`, not the MTP horizon). Do not edit `llama/llama.cpp` without a patch: `sync_vendor_llama.sh` refuses to rsync when vendor is missing that sentinel, and `apply-patches` will not stamp a new patch applied just because `LLAMA_CPP_VENDOR_HEAD` matches. Upstream unscaled `F8_E4M3` (type 43) stays beside block `FP8_E4M3` (type 51) and `FP8_E5M2` (type 52). Apply by filename — numbers shift on ladder rungs. Container: `./scripts/vendor/build_llama_server_container.sh`. BPE: [docs/faster-bpe-tokenize.md](../docs/upstream/faster-bpe-tokenize.md). MultiDecode: [docs/design/surfaces/multidecode-llama-cpp.md](../docs/design/surfaces/multidecode-llama-cpp.md). DeepInfer: [docs/deepinfer-ladder.md](../docs/upstream/deepinfer-ladder.md). |
| **Why ggml-org master** | Track upstream llama.cpp tip. Ranges below are the pre-b12119 catalog; the formatted series is **0001–0163**. Eliza QJL/Polar/TBQ applied as patches **0026–0030**; CUDA L2 completeness and Metal TBQ SET_ROWS follow in the mid series; block FP8 weights are types 51/52 beside upstream F8 type 43 (see [native-fp8-gguf.md](../docs/design/serving/native-fp8-gguf.md)); hardware PR ports **0080–0086**; Bee reasoning-loop guard **0087**; TBQ vec_dot dedupe **0088**; L3-R6b cell+tensor+pages COW **0089**; media-aware `/kv/seq-copy` **0090**; llama-bench Eliza L2 KV type names **0093**; native DCA **0094–0098**; Metal recoverable nil pipeline + bf16 library gate + Polar/QJL SET_ROWS + fused QJL+Polar attn **0096–0098** (Lab D — `speed` runs on Mac, tok/s FAIL merge); Kokoro **0100–0101**; Bee B1 adaptive draft-max **0102**. **Hedge:** do not assume [elizaOS/llama.cpp](https://github.com/elizaOS/llama.cpp) keeps rebasing onto latest ggml-org — this pin remains source of truth. Sibling scout: `../eliza-llama.cpp` @ `ad56033` (OmniVoice/FFI stay out of our binary; Kokoro optional via `LLAMA_BUILD_KOKORO=ON`). |

## In-process ggml (Go CGO) — unified with runtime

| Field | Value |
|-------|--------|
| **Vendor pin** | **`b12119`** — `LLAMA_CPP_VERSION`, `LLAMA_CPP_COMMIT`, `vendor/llama-cpp-b12119/` (rollback pin files: **b11870** in `LLAMA_CPP_*.prev`; older trees: `vendor/llama-cpp-b11351/`, also b11232/b11081/b10969/…) |
| **Upstream repo** | `https://github.com/ggml-org/llama.cpp.git` (same as runtime sibling) |
| **Ollama patches** | Same series as the runtime table (**163** format-patches, **0001–0163**, on **b12119**). Eliza QJL/Polar/TBQ shaders stay in `eliza-shipped/`. Pretok lives in **`src/unicode.*`**. Mac CGO: `llama/cgo_vendor_hash.cpp` + `llama_print_build_info(const char *, FILE *)`. Tip CGO also needs `ml/.../tiled/tiled.go` + `llama/.../parsers/parsers.go`. |
| **In-tree Metal dig** | Upstream split Metal kernels (`kernels/*.metal`). Eliza SET_ROWS + fused QJL+Polar attn remain as extra AIR objects. Native FP8 weight types **51/52**. FA-vec per-device tables are upstream. |
| **Rebase helper** | `./scripts/vendor/rebase_vendor_unified.sh --sync` |

Runtime `llama-server` and in-process ggml share **one ggml-org base** + zerollama patches. The pin base is **`b12119`**. After this rebase, `./scripts/vendor/sync_vendor_llama.sh` copies the patched tree into in-tree ggml (that rewrites `llama/build-info.cpp`).

Upstream also ships **`llama/compat/`** — in-memory GGUF translation at CMake fetch time for **llama-server**. In-process **ggml** uses `llama/patches/` on a vendored tree synced via [docs/design/serving/ggml-b9509-migration.md](../docs/design/serving/ggml-b9509-migration.md).

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

See [docs/design/apple/apple-silicon-metal.md](../docs/design/apple/apple-silicon-metal.md#mlx-engine-optional).

## Environment

| Variable | Purpose |
|----------|---------|
| `LLAMA_CPP_ROOT` | Root of llama.cpp checkout (default: `../llama.cpp` relative to repo) |
| `LLAMA_SERVER_BIN` | Override path to `llama-server` executable |
| `LLAMA_CPP_REPO` | Override clone URL (default historically elizaOS; **pin is ggml-org** via `Makefile.sync`) |
| `ZEROLLAMA_ALLOW_ELIZA_VOICE` | `1` = allow `tools/omnivoice` in build tree (default refuse). Kokoro is allowed; enable with `LLAMA_BUILD_KOKORO=ON` |
| `LLAMA_BUILD_KOKORO` | `ON` = mount `/v1/audio/speech` (Kokoro patches in the series); default OFF |
| `ZEROLLAMA_SPEC_DM_ADAPTIVE` | `profit`/`1`/`on` = Bee B1 adaptive DFlash draft-max (**0102**); timings/slots expose DI3 verified `accept_by_pos` (**0141**); default off |
| `ZEROLLAMA_LLAMA_FORK` | `0` = L1 q8_0 profiles; unset/`1` = auto-probe fork KV types |
| `OLLAMA_MLX_SOURCE` / `OLLAMA_MLX_C_SOURCE` | Override MLX sibling paths |

Rebuild llama.cpp when bumping `LLAMA_CPP_COMMIT`; run runtime integration tests on dual-GPU hosts.

## L6 pin-debt ladder (ggml-org tip **b12119**)

**Status:** pin is **b12119** (2026-11-04) with **163** patch files (**0001–0163**) rebased onto that tag. **0162** cherry-picks ggml-org `#30541` (quantized on-device KV state copy). **0163** keeps the gemma4-assistant draft stack across pin bumps. Production `llama-server` is built with CUDA 12.8 / sm_120. Rollback pin files: **b11870**.

| Item | Detail |
|------|--------|
| **Operator runbook** | [docs/design/serving/llama-cpp-pin-ladder.md](../docs/design/serving/llama-cpp-pin-ladder.md) — rungs `b10615 → … → b11351 → b11870 → b12119` |
| **Status script** | `./scripts/phase/l6_pin_ladder_status.sh` |
| **Clef staging** | `llama/clef/` — wire with `./scripts/phase/stage_clef_for_pin.sh --wire` |
| **Clef compile check** | `./scripts/phase/l6_clef_compile_check.sh` |

Clef **runtime wire + Cloudflare Flash product convert** were green on **b11351**. Production `llama-server` on this CT is the **b12119** binary. Rollback via `LLAMA_CPP_*.prev` (**b11870**) and `./scripts/phase/l6_promote_tip_env.sh --rollback` (refuses until that tree is built). See [docs/design/surfaces/system-one-score.md](../docs/design/surfaces/system-one-score.md) · [docs/design/serving/llama-cpp-pin-ladder.md](../docs/design/serving/llama-cpp-pin-ladder.md).

## Bump checklist (runtime sibling)

1. Update `LLAMA_CPP_COMMIT` (and tag file `LLAMA_CPP_VERSION` if tagging)
2. `./scripts/build/build_llama_server.sh` — probes QJL + checkpoint flags in `--help`
3. `./scripts/phase/l2_fork_eval.sh` — profile argv smoke
4. `./scripts/phase/l2_full_gate.sh` or `./scripts/phase/l2_cuda_full_gate.sh` on GPU hosts

For a **pin-debt rung** (not tip chase), follow [docs/design/serving/llama-cpp-pin-ladder.md](../docs/design/serving/llama-cpp-pin-ladder.md) instead of jumping straight to **b12119**. This pin's patch rebase is done; refresh in-tree ggml with `./scripts/vendor/sync_vendor_llama.sh` after the vendor tree changes.

## Bump checklist (in-process vendor — when rebasing)

1. Update `LLAMA_CPP_VERSION` + `Makefile.sync` `FETCH_HEAD` / upstream URL
2. `make -f Makefile.sync clean apply-patches` (fix conflicts in vendor)
3. `./scripts/vendor/sync_vendor_llama.sh` → fix CGO breaks → `format-patch` if needed
4. Update `llama/build-info.cpp` BUILD_NUMBER/COMMIT
5. `./scripts/build/build_zerollama_mac.sh` && `./zerollama doctor`
6. At **b11232+**: `./scripts/phase/stage_clef_for_pin.sh --wire` and refresh Clef server hunks
