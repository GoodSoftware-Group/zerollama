# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fleet LAN discovery (F4)

**Why:** Static `ZEROLLAMA_FLEET_PEERS` works for K8s and fixed IPs but homelab operators want zero-config LAN discovery without maintaining peer lists.

**What shipped:**

- **Nodes:** `ZEROLLAMA_MDNS=1` on `zerollama serve` advertises `_zerollama._tcp` (TXT: `role=node`, `version`).
- **Fleet manager:** `--mdns` / `ZEROLLAMA_FLEET_MDNS=1` browses LAN and merges with static peers; peers optional when browse enabled.
- **Fleet advertise:** `--mdns-advertise` / `ZEROLLAMA_FLEET_MDNS_ADVERTISE=1` registers `_zerollama-fleet._tcp` for agents.
- **Package:** `fleet/mdns/` (register, browse, peer URL helpers).

Doc: [docs/fleet-management.md](docs/fleet-management.md#lan-discovery-f4-mdns).

### Qwen 3.5 / 3.6 on Apple Silicon (Jun 2026)

**Why:** Library tags like `qwen3.6:latest` (`qwen35moe`) and LM Studio–style `qwen35` VL GGUFs became loadable after the b9611 pin, but Mac operators hit **three independent failures**: Go-engine Metal SIGSEGV during init, llama.cpp `rope.dimension_sections` mismatch on published blobs, and missing unary Metal kernels on first decode. Each looked like one “Metal bug”; fixes belong to different layers.

**What shipped:**

- **Engine routing (`llm/server.go`):** On **darwin**, `qwen35`, `qwen35moe`, and `qwen3next` use the **legacy llamarunner** (CGO llama.cpp + Metal) instead of the Go ollama engine. **Why:** `OllamaEngineRequired()` sends these archs through `ggml.New()`; a C Metal segfault does not surface as a Go error, so there is no fallback. Legacy path + mtmd handles qwen35 after llama.cpp arch support landed.
- **In-process compat (`llama/compat/`):** Wired the same Ollama GGUF translation layer used by **llama-server** into the **CGO llamarunner** (`compat.go`, hooks in `llama-model-loader.cpp` / `mtmd/clip.cpp`, import from `llama/llama.go`). **Why:** Published qwen35moe stores M-RoPE `rope.dimension_sections` as **3** elements; llama.cpp expects **4**. Compat existed but only ran on CMake-fetched builds—not the Mac default binary.
- **Metal shader embed:** `build_zerollama_mac.sh` runs `go generate ./ml/backend/ggml/ggml/src/ggml-metal/` before `go build`. **Why:** macOS JIT-compiles from embedded `ggml-metal-embed.metal`; when ggml adds kernels (e.g. `kernel_unary_f32_f32` for sigmoid in gated SSM) without regenerating the embed, **load succeeds** but **first token** crashes with `Function kernel_unary_f32_f32 was not found`.
- **Defensive ggml backend init:** Skip nil device backends in `ml/backend/ggml/ggml.go` scheduler setup (belt-and-suspenders for Metal device init edge cases).

**Operator notes:**

- Rebuild + **restart serve** after pulling.
- Cap **`num_ctx`** (2048–8192) for first tests; `n_ctx_seq < n_ctx_train` log is informational.
- `tensor API disabled for pre-M5` on M4 Max is **expected**, not the crash cause.

Doc: [docs/qwen35-apple-silicon.md](docs/qwen35-apple-silicon.md).

### Fleet management node (F3)

**Why:** Agents and integrations often see **many zerollama hosts**, not one. Per-node schedulers answer local FIFO and VRAM correctly but not “which box has model M warm?” Scatter-gather and long reservation quotes **waste GPU work** on constrained fleets. F3 adds a **thin management process** that polls F2 `/api/status`, builds a warm-model map, and returns `{url, node_id}` — it never loads or evicts remotely.

**What shipped:**

- **`zerollama fleet serve`** — static `ZEROLLAMA_FLEET_PEERS`; poll interval default 3s; listen default `0.0.0.0:11450`.
- **HTTP API:** `GET /health`, `GET /api/fleet/status`, `POST /api/fleet/assign` (warm-first, lowest-queue routing; `warm_only`, `exclude`).
- **Package:** `fleet/` (manager, assign logic, tests); env `ZEROLLAMA_FLEET_*` in `envconfig`.

Doc: [docs/fleet-management.md](docs/fleet-management.md), [docs/fleet-scheduling.md](docs/fleet-scheduling.md#shipped-f3-management-node-v0).

### macOS runtime serve logging

**Why:** `./scripts/serve_mac_runtime.sh` backgrounded sidecar and Go to log files with no terminal progress — operators thought the script hung. Startup now prints wait dots, log paths, and a ready banner with `tail -f` hints.

**What changed:** `scripts/macos_runtime_serve_lib.sh`, `scripts/serve_mac_runtime.sh` — `MACOS_RT_LOG`, `MACOS_GO_LOG` documented.

### llama.cpp b9611 + MLX pin bump

**Why:** Stay on current upstream llama.cpp (latest tag `b9611`, ahead of vanilla Ollama’s `b9509`) with a **reviewable 14-patch series** instead of in-tree-only deltas. MLX pins aligned with upstream Ollama for MTP/speculation parity. **Why not wait for vanilla Ollama:** zerollama’s in-process ggml runner needs Ollama deltas (no-alloc fit, device props, grammar) on a **clean vendor base**; lagging b9509 blocked Metal fixes and Phase 17 mergeability.

**What shipped:**

- **Pin:** `LLAMA_CPP_VERSION=b9611`, vendor `vendor/llama-cpp-b9611/`, **14 patches** (0013 mtmd C API, 0014 ollama_vocab grammar; 0011 rebased for b9611 CUDA struct layout).
- **Sync:** `./scripts/sync_vendor_llama.sh` (replaces `sync_vendor_b9509.sh` shim); strips mtmd CLI `main()` after rsync.
- **Go fixes:** `llama.go` mtmd `bitmap_wrapper` + `placeholder=false` (b9611 API); `build-info.cpp` @ `1aefee58`.
- **Build:** `./scripts/build_zerollama_mac.sh` uses `GOFLAGS=-mod=mod` so CGO builds succeed when `vendor/` is incomplete.
- **Sibling runtime tree:** `../llama.cpp` @ b9611 + `./scripts/build_llama_server.sh` for vanilla `llama-server` / `libllama.dylib` (Python runtime subprocess — separate from patched in-tree ggml).
- **MLX:** `MLX_VERSION` → `2165dc08`, `MLX_C_VERSION` → `fba4470b` (upstream Ollama pins). Rebuild dylibs after pin bump — see **MLX dylib rebuild** below.

Doc: [docs/ggml-b9509-migration.md](docs/ggml-b9509-migration.md) (filename kept; pin is b9611).

### MLX dylib rebuild (Jun 2026)

**Why:** `MLX_VERSION` / `MLX_C_VERSION` are independent of the ggml llama.cpp pin — safetensors inference uses **`libmlx.dylib` + `libmlxc.dylib`**, not CGO ggml. Bumping pins without rebuilding leaves `mlxrunner` on stale Metal code (wrong kernels, ABI drift vs regenerated Go/C shims). **`build_zerollama_mac.sh` does not rebuild MLX** — only ggml Metal embed + Go binary.

**What shipped:**

- Rebuilt **Metal v3** (macOS 14+) and **Metal v4** (macOS 26+ / NAX) at pins `2165dc08` / `fba4470b`.
- **Install layout:** `dist/darwin-arm64/lib/ollama/mlx_metal_v3/` and `mlx_metal_v4/` (`libmlx.dylib`, `libmlxc.dylib`, `mlx.metallib`).
- **Dev discovery:** `build/metal-v4/lib/ollama/libmlxc.dylib` (repo-root `./zerollama doctor` loads newest variant tree).
- **`GOFLAGS=-mod=mod`** in `build_production_mac.sh` — **why:** CMake runs `go generate ./x/...` during MLX configure; incomplete `vendor/` otherwise aborts configure.

**Operator commands:**

```bash
./scripts/ensure_mlx_sources.sh          # verify ../mlx ../mlx-c have pinned SHAs
export GOFLAGS=-mod=mod
./scripts/build_production_mac.sh      # MLX + release zerollama → dist/darwin-arm64/
# MLX dylibs only (no Go binary): cmake --build build/metal-v3 --target mlx mlxc && cmake --install …
```

Doc: [docs/apple-silicon-metal.md](docs/apple-silicon-metal.md#mlx-engine-optional), [docs/mac-dev-setup.md](docs/mac-dev-setup.md#dev-vs-production-mlx-layout).

### Apple Silicon sign-off (Jun 2026, M4 Max)

**Why:** Operators need a repeatable gate beyond `doctor` — runtime Metal inprocess is the **daily Mac path** (`apple_silicon.yaml`), not legacy ggml. Sign-off proves Phase 13–15 and tools without CUDA-centric `gpu_5080_session.sh`.

**Passed (GPU):**

- **Phase 13:** `/tmp/metal-session.json` — `metal-unified` probe, autotune factor ~0.98 for eliza-1-2b @ `num_ctx=512`.
- **Phase 14:** inprocess from YAML (`llama_backend_source=config`); generate/chat/stream; tokenize + render-chat; Go proxy with `RUN_E2E_PHASE14=1` / `X-Zerollama-Runtime: 1`.
- **Phase 15:** KV decode hook (`kv_decode_steps` via native `llama_decode`); multi-seq (`llama_parallel_slots=2`, `kv_inprocess_n_seq_max=2`).
- **Tools:** runtime + proxy `/api/chat` and `/v1/chat/completions` with tools (HTTP 200, not legacy 501).
- **CPU (no model load):** Phase 12 golden, Phase 15 KV native CI, `go test ./server/... -short`, coordination smoke.

**Known gaps (documented, not blockers for b9611):**

| Gap | Why it happened | Fix (Jun 2026) |
|-----|-----------------|----------------|
| **Proxy v1 stream hang** | SSE missing `[DONE]` on errors; curl waited for EOF | Runtime always emits `[DONE]`; Go proxy flushes SSE; e2e `--max-time` via `RUN_E2E_STREAM_MAX` |
| **Legacy ggml + runtime Metal** | Two stacks on one device | Scheduler blocks ggml when runtime `llama_server=true`; legacy smoke skips on darwin unless `RUN_E2E_LEGACY_FORCE=1` |
| **`num_gpu=0` init Metal** | Metal registered at first ggml init | First CPU-only load sets `GGML_DISABLE_METAL` before backend register |
| **`go test ./ml/backend/ggml/...`** | Dummy GGUF fixture segfault | `doctor` + `metal_signoff.sh` as gate |
| **MLX dylib** | Pin bump without rebuild | `./scripts/ensure_mlx_sources.sh` + `GOFLAGS=-mod=mod ./scripts/build_production_mac.sh` |

Scripts: `./scripts/metal_signoff.sh`, `./scripts/gpu_smoke_all.sh` with `RUN_E2E_PHASE14=1`. Guide: [docs/apple-silicon-metal.md](docs/apple-silicon-metal.md).

### ggml @ llama.cpp b9509 (real vendored tree)

**Why this release:** Zerollama’s **in-process ggml Metal runner** was pinned to an old llama.cpp base with **27/36 patches failing** on upstream’s current pin (`b9509`). Overlay-regenerating patches produced **multi‑MB fork snapshots**, not a clean b9509 ggml. Without rebasing, every upstream bump increased merge pain and blocked Phase 17 alignment with vanilla Ollama’s `LLAMA_CPP_VERSION`.

**What shipped:**

- **Clean b9509 base:** `vendor/llama-cpp-b9509/` + **12 rebased patches** in `llama/patches/` (backup: `llama/patches.pre-b9509-20260612/`).
- **Synced in-tree trees:** `ml/backend/ggml/ggml/` and `llama/llama.cpp/` via `./scripts/sync_vendor_b9509.sh`; `Makefile.sync` pins `FETCH_HEAD=b9509`.
- **Ollama API ports for b9509:** `ggml_backend_sched_new_ext` (fit/no-alloc sizing), extended `ggml_backend_dev_props` + NVML/ROCm mem helpers, `ollama_vocab` grammar, mtmd C API, LoRA plural API, device props in Go.
- **CGO build fixes for b9509 common/:** `jinja_wrap.cpp`, `httplib_wrap.cpp`, `llama/build-info.cpp`; exclude CLI `main()` from mtmd; `models.go` include path for `src/`.
- **Build verified:** `go build`, `zerollama doctor` on Apple Silicon.

**Not in this release:** full CUDA no-alloc pool overrides (`reserving_graph` stubs); automatic replacement of ggml with Go→llama-server (Phase 17 remains opt-in).

Doc: [docs/ggml-b9509-migration.md](docs/ggml-b9509-migration.md).

### Wan text-to-video (v1)

**Why this release:** OpenAI clients expect async video jobs (`POST /v1/videos` → poll → download). Wan runs in a **separate PyTorch stack** from GGUF chat; bolting it into the runtime or ggml runner would duplicate VRAM policy and job lifecycle. Reusing the **embedded training worker** (`run_script`) gives one GPU handoff story (Phase 8 broker, T6 defer queue) without a second public daemon.

**What shipped:**

- OpenAI-compatible **`POST /v1/videos`**, **`GET /v1/videos/:id`**, **`GET /v1/videos/:id/content`** for local Wan presets (`wan2.1-t2v:1.3b`, `wan2.2-ti2v-5b`, 16g manifests).
- **`video_gen`** capability + `video_generation` / `backend_paths` in model config; config-only registration (no GGUF blob).
- Go **`server/video_generate.go`**: payload build, frame caps on 16g, artifact sandbox under `$OLLAMA_MODELS/generated/`.
- Python **`scripts/wan_video_generate.py`** wrapper → upstream `generate.py` with `--save_file`, venv interpreter, progress lines.
- **`training.py`**: `python_bin` / `WAN_VENV` for wrapper; `{job_id}` in `output_path` and `WAN_OUTPUT_PATH`; merged stderr/stdout logging.
- **Defer queue**: `defer-*` ids pollable with `trainingworker.ErrJobNotFound` → HTTP 404; `videoModel` / `videoSize` / stable `submitted_at` on wire.
- **CLI:** `zerollama run <wan-model> "prompt"` via `x/videogen`.
- Install: `scripts/install_wan_video.sh`, `scripts/register_wan_models.sh`.

**Not in v1:** list/cancel on `/v1/videos`, TI2V image input, `:cloud` video, artifact TTL.

Doc: [docs/wan-t2v.md](docs/wan-t2v.md).

### Phase 15 — native KV block pool (v0)

**Why:** Continuous batching allocates KV block ids on every scheduler tick; pure Python competes for the GIL when training and runtime share one embedded interpreter.

**What shipped:**

- C extension `runtime.kv._kv_native` (`runtime/native/kv_block_pool.c`) — same API as Python `BlockPool`.
- Opt-in `ZEROLLAMA_RUNTIME_KV_NATIVE=1`; default remains Python; `/health` `kv` object reports `backend`, `native_requested`, `native_available`, and a `note` when the env is set but the extension is missing.
- CI: `setup.py build_ext --inplace` + `test_kv_native_parity.py` in regression workflow; `./scripts/phase15_kv_native_ci.sh`.

**Phase 15 v1 (scheduler KV):** `/health` `kv_scheduler`; block reserve for `max(prompt+max_tokens, num_ctx)`; subprocess `kv_slot` → llama-server `id_slot`.

**Phase 15 v2 (in-process multi-seq KV):** when `llama_parallel_slots`>1 and backend `inprocess`, one shared `llama_context` with per-sequence KV clear + `seq_id` batch decode; scheduler assigns `kv_slot` for in-process too.

**Phase 15 audit fixes:** `resolve_parallel_slots()` — `-np` in `llama_server_args()` wins over YAML; admission uses real tokenize when GGUF known; `/health` `kv_inprocess_n_seq_max` only when multi-seq; docs aligned.

**Phase 15 v3 (logical KV bind):** `runtime/runtime/kv/bind.py`; `/health` `kv_bind` + per-request `block_ids`; `assert_kv_capacity` at forward; in-process `kv_token_budget`; batch uses real tokenize + scheduler `kv_slot` in parallel decode.

**Phase 15 audit fixes:** clarify `block_ids[i]` = pool id for sequence page *i*; batch `kv_token_budgets`; in-process multi-seq `n_ctx` capped by pool `num_blocks * block_size`; doc two-KV-cap note for subprocess.

**Phase 15 v4:** `kv/physical.py` — llama `seq_pos` vs PA reserve; `/health` `kv_physical` + `kv_native_scheduler_tick`; native `scheduler_tick()`; `ZEROLLAMA_RUNTIME_KV_PHYSICAL_STRICT`.

**Phase 15 audit (health):** `physical_bind_level` when in-process weights loaded; `kv_physical` PA-only snapshot + note for single-seq; strict errors include `request_id` / `kv_slot`.

**Phase 15 v5:** `/health` `kv_scheduler_tick` `{value, source}` with Python fallback; `kv_physical_recent` ring buffer; expanded `phase15_kv_native_ci.sh`; [handoff-phase15-native-kv.md](docs/handoff-phase15-native-kv.md).

**Phase 15 v6:** native `decode_step(n)` on in-process `llama_decode`; `/health` `kv_decode_steps`; env `ZEROLLAMA_RUNTIME_KV_DECODE_HOOK`.

**Phase 15 audit (health/ops):** `kv_decode_steps` inactive reason for subprocess; per-completion `kv_decode_steps` on generate/stream; `kv_physical_recent` mismatches only; atomic C counters; scheduler tick health note.

**Not in v0:** GPU KV tensors, in-process llama KV bind, native decode loop. Doc: [docs/phase15-native-kv.md](docs/phase15-native-kv.md).

### Phase 14 — in-process llama forward (summary)

**Why this release:** The Python runtime already scheduled work and estimated VRAM, but every completion still crossed loopback HTTP to a `llama-server` child. That added latency, complicated GPU handoff with training/ggml, and left Go tools render-chat without a real tokenizer on the runtime path (heuristic truncation only).

**What shipped:**

- Three forward backends behind `ZEROLLAMA_RUNTIME_LLAMA_BACKEND`: **`subprocess`** (default), **`inprocess`** (ctypes + pinned `libllama.so`), **`llama-cpp-python`** (pip wheel, CPU-default GPU layers).
- **`POST /internal/tokenize`** (vocab-only, cached) + Go render path → `truncate_mode: tokenize` when no ggml runner.
- Ollama-shaped **sampling** on all backends (`runtime/runtime/worker/sampler_options.py`).
- Operator/smoke scripts: `phase14_serve_env.sh`, `phase14_backend_smoke.sh`, `phase14_both_backends.sh`; `RUN_E2E_PHASE14=1` in `e2e_runtime_smoke.sh`.
- **5080:** inprocess GPU + wheel CPU smokes pass; see [docs/gpu-5080-operator-guide.md](docs/gpu-5080-operator-guide.md).

**Not in scope (Phase 15+):** native KV scheduler, grammar/mirostat in-process, subprocess removal as default.

Full design: [docs/phase14-inprocess-llama.md](docs/phase14-inprocess-llama.md).

### Fixed

- **Phase 14 render tokenize (embed):** Go `tokenizeForRuntimeModel` uses `runtimeProxyConfigured()` (embedded loopback via `runtimeworker.BaseURL()`), not only `ZEROLLAMA_RUNTIME_URL`. **Why:** embed leaves URL unset; render-chat stayed `truncate_mode=heuristic`.
- **Phase 14 smoke proxy:** `RUN_E2E_PHASE14=1` sends `X-Zerollama-Runtime: 1` on Go proxy steps (smoke-only; `ZEROLLAMA_RUNTIME=1` is not `OLLAMA_RUNTIME_ALL`).
- **server sched tests:** `TestMain` unsets runtime env from operator shells so synthetic GGUF sched tests still exercise ggml load.
- **Phase 14 sign-off script:** `scripts/phase14_both_backends.sh` restarts serve for `inprocess` and `llama-cpp-python` smokes (embed-safe: does not export `ZEROLLAMA_RUNTIME_URL` to serve).
- **Phase 14 llama-cpp-python GPU:** wheel backend defaults to **CPU** (`n_gpu_layers=0`); GPU via `-ngl` or `ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS` (negative env values fall back to CPU with warning). **Why:** cu124 wheel can `free(): invalid pointer` on GPU decode on some hosts while ctypes inprocess works.
- **Phase 14 `phase14_both_backends.sh`:** fails if every backend skipped; clears stale `RUN_E2E_INPROCESS` between runs. **5080 guide:** Phase 14 sign-off checklist in [docs/gpu-5080-operator-guide.md](docs/gpu-5080-operator-guide.md).
- **Phase 14 in-process stream:** `llama_free` / sampler no longer run before stream chunks are consumed (segfault on streaming generate/chat). **Why:** `return generator` inside `try/finally` freed the context immediately.
- **Phase 14 lib path:** auto-detect `libllama.so` via zerollama repo root (`parents[3]`), not `/` + wrong sibling.
- **Phase 14 load parity:** apply `-mg` / `-sm` / `-ts` from `llama_server_args()`; reject speculative/draft on in-process backend.
- **Phase 14 backend config:** `resolve_llama_backend()` uses `RuntimeConfig.llama_backend` when env unset.
- **Phase 14 YAML `llama_backend`:** autoconfig files (e.g. `single_gpu.yaml`) load `llama_backend`; env still wins; invalid values fail at load (`canonical_llama_backend`). `/health` `llama_backend_source`: `env` | `config` (explicit YAML key) | `default` (packaged subprocess). **`llama_cpp`** on wheel backend reports `gpu_mode` / `n_gpu_layers` for operator GPU offload visibility.
- **Embed port conflict:** Go embed preflight refuses busy loopback `:8081` and matches `/health` `embed_boot` to this process (no silent attach to stale runtime). **Why:** `address already in use` while Go logged embed success on cudallama-style restarts.
- **Phase 14 in-process decode:** ctypes path uses heap `llama_batch_init` batches with explicit `pos[]` (fixes `llama_batch_get_one` UAF and uninitialized positions). **Why:** inprocess generate returned 502 `llama_decode failed` on multi-token prompts.
- **Phase 14 YAML smoke:** `phase14_yaml_config_smoke.sh` infers `RUN_E2E_*` backend flags from `/health` when `llama_backend_source=config` (rejects subprocess).
- **Phase 14 status:** ROADMAP exit criteria 3–4 signed off on 5080 dev host (`phase14_inprocess_smoke`, `phase14_wheel_cpu_smoke`).

- **Phase 13 VRAM audit:** IQ2_XS (`GGML_TYPE_IQ2_XS`) block size corrected to 74 bytes (was 138 — over-estimated weights). `VRAM_ESTIMATE_FACTOR` applies once on the outer estimate (speculative draft no longer scaled twice). Calibration uses fresh probe reads and precomputed estimates; multi-GPU `scope_warning` on `/health`. **Why:** audit found false rejections on IQ2_XS models and inflated draft VRAM.
- **Phase 11 dequeue fairness:** scheduler no longer stalls **`priority: normal`** when inference-first metrics are on (only **`low`** at queue head waits). **Why:** enqueue already allowed normal chat under defer/ggml/backlog; dequeue had been blocking all non-high work.
- **Phase 13 load `num_ctx` parity:** `generate` / `stream_generate` pass **admitted** `active.num_ctx` to `llama-server` (not pre-clamp request values). **Why:** VRAM clamp and precheck applied capped context to the queue but load still used the client’s larger `num_ctx` → estimate/load mismatch and possible OOM.
- **Phase 12 tools + clamp:** tools chat render and load share `InferenceEngine.resolve_num_ctx_for_request()` so Go `/internal/render-chat` truncation and `-c` agree when `VRAM_CLAMP_NUM_CTX` is on. **Why:** render used uncapped ctx while load used capped ctx.
- **`runtime_vram_estimate.sh`:** `--num-ctx` now works (`NUM_CTX` exported before JSON payload). **Why:** shell variable was not visible to Python builder.

### Added

- **Phase 14 5080 session:** `RUN_E2E_PHASE14_SIGNOFF=1` and `RUN_E2E_PHASE15=1` in `gpu_5080_session.sh` / `gpu_smoke_all.sh` (sign-off needs `LLAMA_CPP_LIB`).
- **Phase 14 5080 sign-off:** `phase14_5080_signoff.sh` — one-shot gate (`both_backends` + YAML config full + Phase 15 multi-seq). ROADMAP Phase 14 marked **Done**; subprocess remains packaged default.
- **Phase 14 smoke:** `phase14_backend_smoke.sh` and `RUN_E2E_PHASE14=1` in `e2e_runtime_smoke.sh`; sign-off wrappers `phase14_inprocess_smoke.sh`, `phase14_wheel_cpu_smoke.sh`; provenance via `phase14_yaml_config_smoke.sh`, `phase14_subprocess_default_smoke.sh`; optional `phase14_wheel_gpu_smoke.sh` and `phase14_enable_yaml_inprocess.sh`; optional `RUN_E2E_PHASE14=1` in `gpu_smoke_all.sh` / `gpu_5080_session.sh`.
- **Phase 15 inprocess KV smoke:** `phase15_inprocess_kv_smoke.sh` — self-contained inprocess serve + asserts `kv_decode_steps` on generate and `/health` (v6 decode hook); KV snapshot assertion now runs in a subshell so `exec` in the backend smoke does not swallow it.
- **Phase 15 in-process sign-off:** `phase15_inprocess_signoff.sh` — self-contained KV decode hook + multi-seq smokes; `phase15_inprocess_kv_smoke.sh` now starts its own serve.
- **Phase 15 KV snapshot helper:** `smoke_runtime_assert_kv_snapshot()` in `runtime_smoke_lib.sh`; used by Phase 15 GPU smokes.
- **Phase 14 5080 sign-off:** step 3 now runs full `phase15_inprocess_signoff.sh` (KV hook + multi-seq + snapshot).
- **CI:** regression workflow runs `go test ./x/runtimeworker/...` (embed `:8081` preflight tests).
- **Phase 14 YAML full smoke:** `phase14_yaml_config_full_smoke.sh` — optional #6 without editing packaged `single_gpu.yaml`.
- **Phase 14 render tokenize:** `llama-cpp-python` backend uses wheel `vocab_only` for `/internal/tokenize` (no `libllama.so`); subprocess can fall back to wheel vocab when ctypes lib is missing.
- **Phase 14 llama-cpp-python backend:** `ZEROLLAMA_RUNTIME_LLAMA_BACKEND=llama-cpp-python` uses the pip wheel (no `libllama.so` build); tokenize via loaded model when GGUF matches.
- **Phase 14 sampling:** `options.temperature`, `top_k`, `top_p`, penalties, and `seed` forwarded to subprocess `llama-server` and in-process libllama sampler chains; no sampling keys → greedy in-process default. `temperature: 0` sends `temperature: 0` on subprocess; render-chat tokenize falls back to heuristic only when runtime is unreachable (not on HTTP/model errors).
- **Phase 14 tokenize for render:** `POST /internal/tokenize` (libllama vocab-only, cached); Go `/internal/render-chat` uses `truncate_mode: tokenize` when no ggml runner and runtime URL is set. **Why:** tools chat truncation matched ggml only when a runner was loaded; runtime path had heuristic-only.
- **Phase 14 (v1) in-process llama:** `ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess` loads pinned `libllama.so` in the runtime process (ctypes); subprocess `llama-server` remains default. `/health` includes `llama_backend`. Doc: [docs/phase14-inprocess-llama.md](docs/phase14-inprocess-llama.md). **Why:** remove loopback HTTP and second process on the hot path; foundation for in-process tokenize (render) and Phase 15 native KV.
- **Phase 13 operator CLI:** `scripts/runtime_vram_estimate.sh` — `POST /internal/vram-estimate` for a GGUF (budget, `suggested_max_num_ctx`, host RAM). **Why:** tune context and quant choice before load on a tight GPU.
- **Phase 13 docs:** [docs/phase13-runtime-vram.md](docs/phase13-runtime-vram.md) — WHY-oriented estimate/clamp/autoconfig guide and 5080 workflow.
- **`InferenceEngine.resolve_num_ctx_for_request()`** — resolve + optional clamp without queuing; used by `_admit_one`, `/api/generate`, `/api/chat` (including tools render). **Why:** one code path for context policy; avoids render/load drift.
- **Phase 11 VRAM headroom env:** `ZEROLLAMA_RUNTIME_VRAM_MIN_FREE`, `ZEROLLAMA_RUNTIME_TRAINING_VRAM_RESERVE` (size strings; defaults 1 GiB / 2 GiB). `/health` exposes `vram_min_free_configured` and `vram_training_reserve_configured`. **Why:** 5080 operators can tune without editing Python constants.
- **Phase 13 `num_ctx` clamp (opt-in):** `ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX` default **off** (`auto`/`1` lowers request ctx to suggestion); API/stream include `vram_num_ctx` when clamped; tools stream forwards `vram_num_ctx`. **Why:** avoid silent context reduction; still available for single-GPU smoke.
- **GPU smokes:** `e2e_runtime_smoke.sh` calls `/internal/vram-estimate` and post-generate `vram_calibration`; `e2e_coordination_smoke.sh` prints Phase 13 health fields; `serve_gpu_example.sh` documents single-GPU VRAM env defaults. **Why:** CI/ops can validate estimate path without reading Python.
- **Phase 13 on `/v1/chat/completions`:** `prepare_v1_chat()` + `v1_request_options()` — resolve/clamp `num_ctx`, tools render parity, optional `options.gguf` on JSON body; non-stream `vram_num_ctx` in response. **Why:** OpenAI clients hitting :8081 directly should not bypass estimate/clamp policy.
- **Go v1 runtime proxy (Phase 9 + 13):** `runtimeV1ProxyOptions` injects manifest `options.gguf` and merges client `options` before forwarding to Python `/v1/chat/completions` (parity with `/api/chat` proxy). v1 legacy gate matches `/api/chat` for logprobs, vision, think, and OpenAI `reasoning` / `reasoning_effort`. **Why:** OpenAI-shaped clients via `:8080` had no GGUF path for VRAM precheck/clamp on the runtime.
- **GPU smoke:** `e2e_runtime_smoke.sh` exercises generate/chat/v1 (non-stream + stream) on runtime and Go proxy; `gpu_smoke_all.sh` runs coordination + full GPU/proxy smokes; `gpu_health_report.sh` prints calibration/autotune tuning hints from `/health`.
- **Go proxy tests:** chat tools + options forwarding; v1 SSE stream passthrough.
- **`gpu_health_report.sh`:** shared formatter `runtime.gpu_health_report` + tests; `/health` keys match calibration/autotune schema.
- **v1 `v1_request_options`:** top-level `max_tokens` promoted to `options.num_predict` for Phase 13 VRAM resolve (parity with Go proxy).
- **CI:** `scripts/check_gpu_scripts.sh` in regression workflow; `python -m runtime.gpu_health_report` CLI.
- **5080 operator guide:** [docs/gpu-5080-operator-guide.md](docs/gpu-5080-operator-guide.md) — WHY-oriented single-GPU workflow (session gate, API unload, snapshot, harmony/host-RAM limits).
- **Apple Silicon / Metal (M1–M2):** [docs/apple-silicon-metal.md](docs/apple-silicon-metal.md); `runtime/configs/apple_silicon.yaml` autoconfig on darwin; `metal-unified` VRAM probe (`vm_stat`); `read_host_memory()` for host budget on Mac; `macos_metal_smoke.sh`. **Why:** unified memory is not NVIDIA VRAM — CUDA `single_gpu.yaml` and `nvidia-smi` probes rejected valid Mac loads.
- **Apple Silicon fix:** `check_gguf_host_budget` and `/health` host mem now call `read_host_memory()` (was Linux-only); `vm.swapusage` parsed from real macOS `free = N.M` format. **Why:** audit found host pre-load checks silently skipped on Darwin and swap budget always zero.
- **MLX routing (M4):** [docs/mlx-routing-policy.md](docs/mlx-routing-policy.md); `modelUsesRuntimeInference` rejects `IsMLX()` even with mistaken Modelfile backend; Go tests. **Why:** safetensors must stay on mlxrunner, not Python GGUF runtime.
- **Mac session gate (M3):** `gpu_metal_session.sh` — smoke + Phase 13 snapshot + optional Phase 14. **Why:** Mac operators need the same repeatable gate as `gpu_5080_session.sh`.
- **Phase 13 Python:** `single_gpu.yaml` `vram:` block applied at runtime start (`vram_yaml_defaults.py`). **Why:** 16GB autoconfig installs get admission/autotune defaults without a long systemd env block; operator env still wins.
- **Phase 13 snapshot:** `python -m runtime.gpu_snapshot` reads session JSON and prints env hints (autotune `persist`, budget warnings, harmony skip). **Why:** portable tuning record after `gpu_5080_session.sh` without re-scraping `/health`.
- **GPU smoke unload:** `smoke_unload_ggml_runners` evicts stale ggml via `/api/ps` + `keep_alive:0` before Phase 8 broker; `mapfile` per model; `SMOKE_UNLOAD_MAX_WAIT` default 30s; 503+runner retry. **Why:** Go 503 before broker left runners loaded; `pkill` bypassed public unload and false-positive `pgrep` on shell lines.
- **GPU smoke:** `gpu_harmony_capture.sh` uses API unload instead of `pkill`; `gpu_5080_session.sh` runs snapshot + recommendations.
- **GPU smoke:** optional `RUN_E2E_TOOLS=1` for `/api/chat` with tools on `:8081` and `:8080` proxy (asserts 200 + not legacy 501); Go `TestRuntimeChatProxyStream`.
- **gpu_health_report:** export hint only when `suggested_estimate_factor` is in 0.1–3.
- **Phase 12 render:** `/internal/render-chat` uses ggml `Tokenize` when a runner for the model is already loaded (`truncate_mode`: `tokenize` | `heuristic` | `none`); `truncated` true only when prefix messages dropped; `has_tool_support` uses prepared messages; Python tools `meta.truncate_mode` / `meta.truncated` mirror Go; golden parity tests `runtime_render_golden_test.go`; single `prepareRenderMessages` in handler via `renderChatPromptPrepared`.
- **Phase 12 parse golden:** `runtime_parse_golden_test.go` — functiongemma one-shot + streaming chunks vs `model/parsers`; render→parse roundtrip.
- **Phase 12 truncation:** ggml `chatPrompt` uses `chatPromptTokenBudget` (same reserve as render-chat when `num_predict > 0`); single `prepareToolsForRender` in render handler.
- **Phase 12 parse golden:** harmony tool-call fixture in `runtime_parse_golden_test.go`; Python asserts `requires_go_tool_parser` for harmony parser meta.
- **GPU ops:** `gpu_harmony_capture.sh` + `--harmony` on `phase12_capture_tool_transcript.sh` (fix gguf override for pulled models); `gpu_5080_session.sh`; Phase 11 threshold env overrides; `phase12_golden_ci.sh` (`all|go|py`); clamp/snapshot smokes. Audit: legacy only via `RUN_E2E_LEGACY_ONLY` / legacy-only e2e; `smoke_prepare_vram` fails on 503+ggml runner; Phase 11 doc aligned with env overrides.
- **Go:** `TestRuntimeV1ChatCompletionsProxyForwardsTools`; CI `check_gpu_scripts` greps tools smoke markers; **fix** regression workflow runs `check_gpu_scripts` from repo root (not `runtime/` cwd).
- **GPU smoke:** `RUN_E2E_TOOLS=1` also exercises `/v1/chat/completions` with tools on `:8081` and `:8080` proxy.
- **Phase 12 render truncation:** `/internal/render-chat` heuristic reserves `num_predict` (or ~256 default) for completion headroom; Python tools path passes `n_predict` into Go render.
- **GPU smoke:** `RUN_E2E_VRAM_CLAMP=1` asserts `/health` `vram_num_ctx_policy.clamp_enabled` and probes high-`num_ctx` generate for `vram_num_ctx` or budget reject; default `OLLAMA_HOST` is `:8080`.
- **v1 parity:** `v1_max_tokens` accepts float JSON numbers and `options.num_predict`; `think:false` stays on runtime (Go + Python).
- **T6 training queue policy (Go):** optional `ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE` rejects training submit while ggml and/or runtime inference are busy; **`defer-*` job queue** when `ZEROLLAMA_TRAINING_QUEUE_ON_BUSY` (or `priority: low` / `queue_on_busy`) with auto-promote, tombstone TTL, retries, cancel, and list merge. Submit **`priority`**: `high` bypasses idle-wait, `low` prefers defer. **Why:** single-GPU hosts need inference-first scheduling without a second Python listener; batch training should wait for chat to finish instead of failing with opaque 409s.
- **`ZEROLLAMA_TRAINING_WAIT_GGML_LOADED`:** resident ggml runners (`OLLAMA_KEEP_ALIVE`) count as busy when idle-wait is on (opt-out for multi-GPU). **Why:** a loaded legacy model holds VRAM even with an empty queue.
- **Training blocks inference (monitor):** while training occupies GPU, pause ggml loads, evict runners, block runtime proxy (`ZEROLLAMA_BLOCK_INFERENCE_DURING_TRAINING`, default on). **Why:** training and chat must not fight for the same 16 GB without policy.
- **Phase 11 runtime admission (opinionated):** Python VRAM + inference-first policy with **two operator envs** only — `ZEROLLAMA_RUNTIME_INFERENCE_POLICY=off`, `ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM=0`. Thresholds, **2 GiB** training reserve, **1 GiB** min-free (1.5× for `low`), and backlog/defer/ggml constants live in code. **Why:** single-GPU operators should not tune a dozen `ADMISSION_*` flags; we measure and adjust constants instead. See [docs/phase11-runtime-admission.md](docs/phase11-runtime-admission.md).
- **Phase 11 enqueue GGUF precheck:** host + `check_gguf_vram_budget` **before** the waiting queue when the model path is known. **Why:** fail fast instead of filling the queue with work that cannot load.
- **Phase 11 inference priority:** `options.priority` (`high` / `normal` / `low` with aliases); high jumps queue and bypasses min-free gate (not model fit); `generate_batch` defaults to low. **Why:** align with Go training T6 without one global FIFO.
- **Scheduler VRAM re-check:** `SchedulerLoop.tick` re-runs admission before KV allocate (per-request `priority`). **`VRAM_MMAP_FACTOR`** scales weight estimate. **Why:** VRAM can drop while requests sit in queue; mmap’d GGUF may not need full tensor bytes on GPU.
- **Admit cleanup:** `cancel_waiting` on failed tick/batch; tick distinguishes `AdmissionRejected` (re-queue) vs misconfig (fail). **Why:** avoid stray queued requests after a failed admit.
- **Go → Python training GPU busy:** `POST /internal/training-gpu-busy` from training policy monitor; admission reserve follows Go `trainingOccupiesGPU`. **Why:** reserve headroom on direct `:8081` runtime traffic while training holds the card.
- **Go coordination on runtime `/health`:** `POST /internal/go-coordination` mirrors training defer queue counts and policy flags. **Why:** single-GPU operators see Go+Python queue state from the runtime sidecar.
- **Defer / ggml backlog (inference-first):** when Go mirror is fresh, **`priority: low`** is rejected at enqueue and stalled at dequeue; **normal** is not. Wired via `go-coordination` + `inference_policy.py` (no per-gate env). **Why:** inference-first SLO on a shared GPU without blocking default chat.
- **Ggml pause when runtime busy (Go):** `ZEROLLAMA_GGML_PAUSE_WHEN_RUNTIME_BUSY=auto` (on when runtime URL or embed is configured) pauses new ggml loads while `runtime_waiting + runtime_running` ≥ `ZEROLLAMA_GGML_PAUSE_RUNTIME_MIN_BACKLOG` (default 4); one `/health` probe per tick; mirror exposes `ggml_loads_paused`. **Why:** symmetric single-GPU policy — Python already blocks low-priority work when ggml is busy.
- **Phase 11 `/health` gates:** `admission.gates_active` uses explicit names (`low_would_wait`, `runtime_backlog_pressure`, …); legacy keys under `gates_active_compat`. **Why:** old `batch_backpressure: true` with `backlog: 0` confused operators — true means “low would wait,” not “everything blocked.”
- **Phase 13 startup env apply:** `ZEROLLAMA_RUNTIME_VRAM_APPLY_EXPORTED_ENV=1` loads `vram_estimate_factor.env` once at runtime start (skips when `VRAM_ESTIMATE_FACTOR` already set; autotune persist still wins per GGUF). **Why:** operators can `source` export without hand-copying into systemd/unit env.
- **Phase 13 KV head dims:** when GGUF omits `key_length`, derive from `attn_k` / `attn_v` tensor shapes + `embedding_length` / `head_count_kv`. **Why:** tighter KV VRAM estimates on sparse manifests.
- **Phase 13 quant KV bytes:** `VRAM_KV_BLOCK_LAYOUT=1` (default) uses ggml block layout for IQ/TQ/MXFP `key_type`/`value_type`; classic Q4/Q8 KV stays ≥2 bytes/element (conservative). **Why:** less KV over-estimate on IQ models without loosening Q4 admission.
- **T6 cross-queue FIFO:** global monotonic tickets (`POST /internal/cross-queue-seq`); Go mirror `fifo_go_oldest_*` / runtime `fifo_runtime_oldest` (waiting+running); Python blocks batch when **ggml** is ahead; ggml pending/loading yields to older runtime tickets; defer promotion waits for inference (runtime or ggml). **Why:** single-GPU ordering across ggml, runtime, and defer without operator knobs.
- **Coordination shutdown:** `finalizeInferenceCoordination` on daemon exit resumes ggml loads and pushes cleared `ggml_loads_paused` / `training_gpu_blocked` (avoids stale mirror until TTL). **Why:** rolling restarts should not leave Python thinking ggml is paused for 30s.
- **Phase 13 IQ/TQ block layouts:** VRAM weight estimates use ggml block sizes for IQ2/IQ3/IQ4/IQ1 and TQ1/TQ2 types (synced from ggml-common.h). `ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR` scales final estimates for operator calibration. **Why:** rare quants no longer over-count via fp16 fallback bytes/element.
- **Phase 13 probe calibration:** `ZEROLLAMA_RUNTIME_VRAM_PROBE_CALIBRATE=auto` records NVML/smi free VRAM before/after llama-server load; `/health` `vram_calibration` exposes `suggested_estimate_factor` (observed/raw). **Why:** operators can tune estimates on real hardware without automatic policy changes.
- **Phase 13 estimate autotune:** per-model factors in `STATE_DIR/vram_autotune.json` (v2); **`VRAM_AUTOTUNE_PERSIST=auto`**; **`VRAM_ESTIMATE_FACTOR_EXPORT=auto`** writes `vram_estimate_factor.env` + catalog after calibration. Pre-check uses `effective_vram_estimate_factor(gguf=...)`. **Why:** different models on one GPU; operators can `source` suggested env without hand-editing.
- **Phase 13 VRAM hardening:** per-layer `sliding_window` UINT32 arrays (hybrid SWA); unknown ggml KV types → fp16; Q4/Q8 sized conservatively; separate K/V dims; `ZEROLLAMA_RUNTIME_VRAM_SCRATCH_FACTOR`; cached `gguf_arch_hints` for `/health`; heuristic path uses resolved `num_ctx` consistently.
- **Phase 13 llama-server flag parity:** VRAM pre-check and `/health` `vram_estimate` parse config/`LLAMA_SERVER_EXTRA_ARGS` (`-c`, `-ngl`, `-np`) plus YAML `llama_parallel_slots`. **Why:** operators cap context with `-c 8192` but estimates used full GGUF `context_length` and ignored parallel slots.
- **Phase 13 speculative draft VRAM:** when `speculative.method` is a draft plugin, estimates add the draft GGUF (`--model-draft`, `--spec-draft-ngl`). **Why:** dual-model speculative decode needs both weight+KV budgets on a tight GPU.
- **Phase 13 ngram scratch:** ngram speculative methods add `ZEROLLAMA_RUNTIME_VRAM_NGRAM_SCRATCH_BYTES` (default 128MiB) to VRAM estimates. **Why:** ngram cache has no draft GGUF but still consumes GPU memory.
- **Phase 13 per-tensor GPU weights:** `VRAM_WEIGHT_TENSOR` sums GGUF tensors by layer for `-ngl`; `VRAM_WEIGHT_BLOCK_LAYOUT` uses ggml quant block sizes (Q4_0, Q4_K, …). Exact KV per GPU layer; `-ngl 0` skips GPU KV. **Why:** linear `ne×2` and full-layer KV over-estimated VRAM on partial offload and quants.
- **Phase 13 runtime VRAM (Python):** `ZEROLLAMA_RUNTIME_VRAM_PROBE` (`auto` / `nvml` / `nvidia-smi`), optional `nvidia-ml-py` via `pip install -e 'runtime/.[gpu]'`, unified-memory fallback (`host-unified`), KV scale from request `num_ctx` → env → GGUF `context_length`, layer scale from GGUF `block_count`. **Exact KV path:** `estimate_kv_cache_bytes` from GGUF layers, GQA heads, K/V dims, optional `attention.key_type`/`value_type` (quantized KV), `sliding_window` cap; `ZEROLLAMA_RUNTIME_VRAM_KV_EXACT` (default on); `/health` `vram_estimate`; request `options` passed into pre-check. **Why:** weights + explicit K+V avoids double-counting ctx/layer multipliers when metadata is complete.
- **T6 training night window (Go):** `ZEROLLAMA_TRAINING_ALLOWED_WINDOW` (e.g. `22:00-06:00`), `ZEROLLAMA_TRAINING_WINDOW_TZ`; `priority: high` bypasses; defer queue can hold jobs until the window opens when `ZEROLLAMA_TRAINING_QUEUE_ON_BUSY=1`. Invalid window env → **503** + warn-once log (fail closed). **Why:** first SLO hook for batch training on a shared GPU without a unified FIFO.
- **`gguf_model_hints`:** read block count / context length from GGUF metadata for VRAM estimates. **Why:** one file parse drives both layer and context scaling.
- **CI regression (Phase 10):** `.github/workflows/zerollama-regression.yaml` — Go `server`/`envconfig`/`trainingworker` + runtime pytest. **Why:** cross-language policy wiring regresses silently without a gate.
- **CPU training route smoke (T4 partial):** Go tests register `/api/train/*` and exercise `GET /api/train/status` without embedded Python. **Why:** catch route wiring breaks without CUDA on every PR.

- **Manifest → runtime (Phase 9):** Go proxy adds `options.gguf` from the Ollama manifest; Python runtime loads or swaps `llama-server` per request. **Why:** `ollama run <pulled-model>` on `:8080` without a global `LLAMA_MODEL` or `smoke` name.
- **Go VRAM broker (`server/vram`, Phase 8):** before ggml runner load → `training-handoff` on embedded runtime; before runtime proxy → unload all runners + `inference/resume`; before training job submit → both (OOM path unchanged). **Why:** single-GPU hosts no longer need manual curl between legacy and runtime stacks.
- **Runtime inference smoke:** [`docs/testing-smoke.md`](docs/testing-smoke.md) and [`scripts/e2e_runtime_smoke.sh`](scripts/e2e_runtime_smoke.sh) (`RUN_E2E_GPU`, `RUN_E2E_PROXY`). **Why:** two local stacks (Python runtime vs ggml runner) need an explicit checklist so operators do not confuse 404/503/OOM with API bugs.
- **`POST /internal/inference/resume`** on the Python runtime (internal only; Go broker calls it before runtime proxy). **Why:** `training-handoff` leaves inference `unloaded`; without resume, `:8081` generate returns 503 until process restart.
- **`runtime/configs/single_gpu.yaml`** and default `device_count: 1`. **Why:** `dual_4090.yaml` tensor split on one GPU makes `llama-server` fail fitting (`SPLIT_MODE_TENSOR`).
- **`LLAMA_SERVER_EXTRA_ARGS`** env (appended to `llama-server` argv). **Why:** operators need `-c` / other flags without editing YAML for every host.
- **Health `server_revision`** field (`fastapi-body-v3`). **Why:** confirm embedded Python reloaded after deploy without guessing from 422 bodies.

- **GPU training integration (Go + embedded Python):** when `OLLAMA_TRAINING` is true (default), the Go daemon embeds **CPython** via CGO (`x/trainingworker/pyembed`), loads repo-root **`training.py`**, exposes **`/api/train/*`** over HTTP, and optionally **TCP `:9500`** (newline JSON, legacy-compatible). **No** separate `python3` subprocess, **no** gRPC/`grpcio`, **no** UDS control plane. **Why:** one public process (Ollama), Python owns PyTorch/CUDA while Go owns ports, scheduler integration, and VRAM policy (inference-first OOM bridge: pause loads → evict runners → ack Python).
- **Zerollama → Eliza Cloud (default remote inference):** default upstream `https://www.elizacloud.ai`, `ELIZACLOUD_API_KEY` sent as `X-API-Key` on `/api/v1/...`; **Ed25519 request signing only** when `OLLAMA_CLOUD_BASE_URL` targets `ollama.com` (legacy cloud). Client paths `/v1/*` are rewritten to Eliza `/api/v1/*`; `/api/embed` and `/api/embeddings` map to `/api/v1/embeddings`. **Why:** OpenAI/Anthropic-compatible APIs and API-key auth match how agents integrate; legacy signing stays opt-in for ollama.com users.
- **Cloud model catalog merge:** `GET /api/v1/models` merged into local tag lists when cloud is enabled, with **singleflight** on fetch, **Cache-Control**–aware TTL (clamped), and dedupe by model name. **Why:** one combined list for operators; avoids stampedes and duplicate rows.

- **Native video sampling policy:** env `OLLAMA_VIDEO_SAMPLE_MODE` / `OLLAMA_VIDEO_STRIDE`, optional manifest `video_sampling` and `tokens_per_image`, centralized ffmpeg filter builder, structured **Info** logs after sampling, **`video_spans`** on `api.Message`, context **preflight** against `num_ctx` (messages with video only), and **[video-parity.md](docs/video-parity.md)** (Option 2 matrix).
- **Video understanding (VLM)** for OpenAI-compatible chat: `content` parts with `type: "video_url"` are merged into a single user message, decoded (data URI or remote HTTPS by default), sampled to frames via **ffmpeg**, and fed through the existing vision path as additional images (`docs/video-understanding.md`).
- **`api.Message.videos`** for raw video bytes on `POST /api/chat`; expansion runs before prompt rendering.
- **Manifest / capabilities:** `modality_backends.video_understanding` values `native` (default) or `sglang`; **`video`** capability alongside vision where applicable.
- **Optional SGLang proxy:** when `video_understanding=sglang` and `OLLAMA_SGLANG_URL` is set, `POST /v1/chat/completions` bodies that include `video_url` can be forwarded in full to SGLang’s `/v1/chat/completions`.
- **Environment variables** for limits and behavior: `OLLAMA_FFMPEG`, `OLLAMA_SGLANG_URL`, `OLLAMA_VIDEO_*` (see `docs/multimodal-backends.md`).
- **`FromChatRequestWithContext`** so remote `video_url` fetches respect request cancellation; `FromChatRequest` remains for callers without a context.

### Security

- Remote `video_url` fetches use **HTTPS by default**; `http://` requires `OLLAMA_VIDEO_ALLOW_INSECURE_HTTP=1`.
- **SSRF mitigation:** DNS resolution before GET with rejection of loopback/private/link-local targets (see `docs/video-understanding.md` for limitations).

### Changed

- **Phase 11 admission env removed:** per-gate `ZEROLLAMA_RUNTIME_ADMISSION_*`, `TRAINING_VRAM_RESERVE`, `ADMISSION_VRAM_BYPASS_PRIORITY`, etc. are **no longer read**. Behavior is fixed in `runtime/runtime/gpu/admission.py` and `inference_policy.py`. **Why:** product decision — opinionated defaults, tune in code after GPU measurement.
- **Phase 11 VRAM gate coupling:** `admission_vram_gate_enabled()` follows `CHECK_GPU_VRAM` only (not `INFERENCE_POLICY`). **Why:** disabling scheduling policy must not disable VRAM safety rails.
- **Phase 11 model budget:** `check_gguf_vram_budget` applies `max(model×margin, min_free×priority)` and training reserve on all load/enqueue/dequeue paths. **Why:** one coherent budget check instead of a separate 1 GiB probe when GGUF is known.
- **GPU training control plane:** subprocess `python3 -m trainingdaemon` + gRPC/UDS replaced by **embedded CPython** (`x/trainingworker/pyembed`). **`OLLAMA_TRAINING_PYTHONPATH`** now means the repo root containing **`training.py`**. **`grpcio`** is no longer required for training IPC.
- **Eliza outbound auth:** `X-API-Key` is applied to all proxied paths toward non-`ollama.com` upstreams (not only `/api/v1/...`); missing key logs **once** per process on first such request. **Path rewrite:** only `/v1` and `/v1/...` are mapped to `/api/v1/...` (avoids mangling paths like `/v1chat`). **Signing:** Ed25519 uses `isOllamaComUpstream()` instead of a redundant `signingHost` return value from `OLLAMA_CLOUD_BASE_URL` resolution.
- OpenAI multimodal `content` arrays are converted to **one** internal `api.Message` per assistant/user turn (text + images + videos) instead of multiple messages per part, preserving array order for vision inputs.
- **Native video:** invalid manifest `video_sampling.mode` logs a warning and falls back to **fps**; **`ExternalVideoDecodeHook`** runs only after empty/size checks (same as ffmpeg path).

### Fixed

- **Python runtime FastAPI `/api/generate`:** request body binding via `Body()`; removed `from __future__ import annotations` in `runtime/server/app.py`. **Why:** postponed annotations made FastAPI treat `req` as a **query** parameter (`422` “Field required” on `query.req`).
- **Runtime proxy `num_predict`:** no longer defaults to **128** when Ollama options omit it (`NumPredict: -1`). **Why:** answers looked “cut off” even though the legacy runner would run until stop/EOS.
- **`scripts/build_llama_server.sh`:** validate `nvcc` exists; search `cuda-12.8`; do not trust a broken `CUDA_HOME`. **Why:** `CUDACXX=/usr/local/cuda-13/bin/nvcc` failed when only CUDA 12.8 was installed.
- **Go tests:** `server/sched_test.go` `TestMain` sets `OLLAMA_NO_CLOUD=true` when unset. **Why:** Eliza catalog merge on dev machines broke list/tags tests expecting small fixture counts.
- **Runtime `llama_server` errors:** HTTP/JSON failures surface as `502` with detail, not empty 500 bodies.
- **Shared-interpreter `/health` hang (mitigation):** when training + embedded runtime share CPython, Go sets `ZEROLLAMA_RUNTIME_SHARED_PYTHON=1`; VRAM probe defaults to `nvidia-smi` (GIL-friendly), skips NVML when smi missing; `/health` TTL cache + single-flight with invalidation on handoff/resume/training-gpu-busy; `vram_probe_effective` on `/health`. **Why:** concurrent `/health` + `pynvml` + training threads could stall uvicorn (see `docs/bugs/shared-interpreter-health-hang.md`).
- **Runtime proxy `options.gguf`:** client-supplied `gguf` wins over manifest path (smoke `RUN_E2E_GGUF` on `:8080` proxy). **Why:** explicit override for VRAM smoke and ad-hoc weights.
- **Runtime dequeue VRAM precheck:** `check_gguf_vram_budget` at admit uses `resolve_vram_num_ctx` (options / env / `-c` / GGUF), not only `req.num_ctx`. **Why:** underestimated VRAM when `num_ctx` lived in `options` only.
- **GPU training OOM wait:** Python now keeps a **single** `threading.Event` from `_prepare_vram_relief_wait` through `_wait_vram_relief_after_oom` (stored on `BridgeState`). **Why:** re-registering a new event after Go had already ack’d caused a **lost wakeup** and up to a 120s stall.
- **GPU training shutdown:** `shutdown_ollama_training` signals `_pending_oom_event` before joining the job thread. **Why:** a thread blocked in the OOM wait could otherwise keep `join(30s)` from finishing cleanly.
- **GPU training repo path:** auto-detect walks cwd and `$HOME/zerollama`; explicit `OLLAMA_TRAINING_PYTHONPATH` / `ZEROLLAMA_REPO` must contain `training.py` or Start fails (no silent fallback). **Why:** typos must not load a different checkout.
- **`list_jobs` bridge:** `_job_to_dict` accepts `Job.to_dict()` output directly so the handler does not re-lock the queue per job. **Why:** less lock churn under load.

### Documentation

- **[scheduling-vram-policy.md](docs/scheduling-vram-policy.md)** — **why** inference and training are separate queues, VRAM broker, T6 defer queue, Phase 11–13 heuristics, tight-host checklist, code map.
- **[testing-smoke.md](docs/testing-smoke.md)** — dual-stack smoke, GPU handoff, 5080 build notes, troubleshooting (WHY-oriented).
- **GPU training (WHY-oriented):** expanded [`docs/gpu-training.md`](docs/gpu-training.md) (OOM event ordering, lost-wakeup prevention, progress polling vs push, init failure / restart, shutdown); [`docs/development.md`](docs/development.md) (embedded CPython build); [`x/trainingworker/README.md`](x/trainingworker/README.md); new [`x/trainingworker/pyembed/README.md`](x/trainingworker/pyembed/README.md); [`README.md`](README.md) in-repo link text; **code comments** in `x/trainingworker/client.go`, `server/training_api.go`, `x/trainingworker/pyembed/shim*.go`, `training_shim.c`, `bootstrap.py`, `training.py`, `envconfig/config.go`.
- **[ROADMAP.md](docs/ROADMAP.md)** — **GPU training (fine-tuning)** section; **why** the roadmap file exists; **Option 2** video phases; **[Zerollama remote cloud (Eliza)](docs/ROADMAP.md#zerollama-remote-cloud-eliza)** follow-ups and non-goals.
- **[eliza-cloud.md](docs/eliza-cloud.md)** — **why** Eliza is the default upstream, **why** `X-API-Key` vs Ed25519 signing, path rewrites, catalog merge/cache, raw upstream JSON on some routes, account stubs off ollama.com.
- **[video-understanding.md](docs/video-understanding.md)** — **why** merged OpenAI messages, ffmpeg→PNG, **why** preflight scopes to messages with video, **why** `video_spans`, logging at Info.
- **[multimodal-backends.md](docs/multimodal-backends.md)** — **why** env + manifest both apply to sampling.
- **[video-parity.md](docs/video-parity.md)** — **why** a parity matrix and reference workloads.
- Code comments in **`server/cloud_proxy.go`** / **`server/eliza_catalog.go`** (remote proxy defaults, path rewrite, singleflight) and **`server/modality`** (video policy, preflight, ffmpeg, expansion) plus **`types/model` / `api`** types where relevant — **why** decisions, not only **what**.
- Code comments in **`server/routes.go`**, **`server/sched.go`** — training route wiring and scheduler hooks for VRAM eviction.
