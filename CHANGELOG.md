# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### L1/L3 full gates + Proxmox build/serve ops (Jun 2026)

**Why:** L1/L3 exit criteria need one-shot orchestrators (not three manual scripts). Proxmox CT 1564 ships inference to remote Ruby clients but minimal checkouts cannot `go build` without vendored `cpp-httplib`; default `OLLAMA_HOST` binds localhost only.

- **`scripts/l1_cuda_full_gate.sh`** + **`l1_gate_report.sh`** — calibrate + concurrent bench → merged `gate.json` + PASS/REGRESS verdict. **Why:** single-stream can be flat (+0.7%); concurrent N=2 is the ship bar (+10.5% on eliza-1 9B).
- **`scripts/l1_full_gate.sh`** / **`l1_metal_gate.sh`** — platform wrappers (CUDA vs Metal).
- **`scripts/l3_cuda_full_gate.sh`** + **`l3_gate_report.sh`** — 8k smoke + 27k production gate → merged verdict. **Why:** wiring @ 8k ≠ agent-scale win @ 27k.
- **`scripts/l3_full_gate.sh`** — dispatches CUDA vs Mac smoke paths.
- **`gpu_5080_session.sh`** — optional `RUN_E2E_L1=1`, `RUN_E2E_L3=1` (need `CUDA_LLAMA_MODEL` or `LLAMA_MODEL`).
- **CGO build on minimal checkout** — root `.gitignore` `vendor/` excludes `llama/llama.cpp/vendor/cpp-httplib/`; copy from sibling `llama.cpp` or run `./scripts/sync_vendor_llama.sh` after full vendor clone. Doc: [gpu-5080-operator-guide.md](docs/gpu-5080-operator-guide.md#building-zerollama-cgo-on-proxmox-ct).
- **Production serve** — `OLLAMA_HOST=0.0.0.0:8080` for remote clients (Ruby `ZEROLLAMA_API_ENDPOINT`, `OLLAMA_HOST`); embedded runtime stays `127.0.0.1:8081`. Example: `scripts/serve_gpu_example.sh`; CT 1564 uses `~/bin/serve.sh` with log redirect to `/tmp/zerollama-serve.log`.
- **`linux_runtime_serve_lib.sh`** — `LINUX_RT_CURL_TIMEOUT=15` (cold `/health` ~9s on 5080); kill `llama-server` on `runtime_port+1` when stopping sidecar. **Why:** 2s curl timeout caused false “runtime failed to start”; orphan llama-server held VRAM across A/B legs.

### Phase 15 v32 — scheduler-driven auto-batch (Jun 2026)

**Why:** v27–v30 batch decode only ran on explicit ``generate_batch`` / ``/internal/generate-batch``. Concurrent ``/api/generate`` threads each called ``completion()`` separately — N ``llama_decode`` calls per token step. v32 opt-in coalesces admitted requests within a short window into ``completions_parallel``.

- **`runtime/runtime/kv/auto_batch.py`** — ``AutoBatchCoordinator``; flush on ``parallel_slots`` fill or ``ZEROLLAMA_KV_AUTO_BATCH_MS`` timeout; batch key includes sampler options hash.
- **`InferenceEngine.generate()``** — routes through coordinator when ``ZEROLLAMA_KV_AUTO_BATCH=1`` + in-process multiseq + linked batch decode.
- **`/health.kv_auto_batch`** — operator stats (``pending``, ``flush_count``, ``batched_requests``).
- **Tests:** ``tests/test_kv_auto_batch.py`` in ``phase15_kv_native_ci.sh``.
- **`runtime/runtime/server/app.py`** — ``Optional[dict[str, Any]]`` on ``InternalBatchGenerateBody.options`` (Python 3.9 + FastAPI compat).

**Env:** ``ZEROLLAMA_KV_AUTO_BATCH=1`` (default off); ``ZEROLLAMA_KV_AUTO_BATCH_MS=5`` (default). Streaming ``generate`` unchanged.

### L1 concurrent + L3 production gates — 5080 CT 1564 (Jun 2026)

**Why:** Single-stream L1 calibration showed only +0.5%/+0.7%; `n_parallel=2` must win under concurrent agent load. L3 @ 8k strict PASS did not prove cache at production ctx (27k).

- **`l1_cuda_concurrent_bench.sh` PASS** — eliza-1 9B, `L1C_N=2` @ 8k: profile ON **102.7** vs OFF **92.9** agg tok/s (**+10.5%**); ON leg 0 errors; OFF leg 1×502 (expected at `n_parallel=1`).
- **`l3_production_gate.sh` PASS** — eliza-1 9B @ `L3_NUM_CTX=26624`, `L3_PREFIX_REPEAT=150`: cached turn2 **0.72s** vs no-cache **1.48s**; `turn2/turn1=1.02` (strict ratio ≤0.75 not met — decode-bound after warm prefill).
- **`linux_runtime_serve_lib.sh`** — `curl -m` 15s on `/health` wait (WHY: cold health probe ~9s on 5080); kill llama-server on `runtime_port+1` on stop.
- **Docs:** [gpu-profiles-l1.md](docs/gpu-profiles-l1.md), [gpu-profiles-l3.md](docs/gpu-profiles-l3.md), [gpu-5080-operator-guide.md](docs/gpu-5080-operator-guide.md), [ROADMAP.md](docs/ROADMAP.md).

### Phase 15 v32b — writable bind upstream tracker (Jun 2026)

**Why:** Criterion #5 (writable PA→tensor page bind) is upstream-blocked; operators need a static probe and CI watch for when llama.cpp ships page-handle APIs — without requiring a live decode context.

- **`llama_memory_kv_ext_writable_bind_probe`** — staging C API in `llama-kv-ext.h`; returns available when `LLAMA_KV_EXT_WRITABLE_PAGE_MAP` is defined at libllama build time.
- **`page_bind_writable_probe()`** — native ext + Python facade; `/health.kv_page_bind` exposes `writable_bind_available`, `writable_bind_api`, `writable_bind_blocker`.
- **`scripts/phase15_llama_kv_ext_pin_check.sh`** — greps upstream `llama.h` for writable page-map symbol names and prints NOTICE when detected.

### L1 CUDA 5080 calibration — rtx-5080.json tuned on ship hardware (Jun 2026)

**Why:** Eliza-ported profile (`-np 4 -b 2048`) regressed single-stream on 1B Q8 (−12.5%); production 9B only −1% — slot overhead dominates on tiny models.

- **`scripts/l1_cuda_calibrate.sh`** — OFF vs ON (+ `L1_SWEEP_NP`) through `l2_cuda_bench.sh`; cleans `${L1_OUT_DIR}` each run.
- **`scripts/l2_cuda_bench.sh`** — `ZEROLLAMA_GPU_PROFILE` overridable (default `1`) for L1 OFF baseline.
- **`runtime/configs/gpu/rtx-5080.json`** — `n_parallel=2`, `batch_size=1024`, `ubatch_size=256` (half 4090 batch for 16 GiB). Measured: 1B **+0.5%**, 9B **+0.7%** vs OFF @ 8k.
- **Docs:** [gpu-profiles-l1.md](docs/gpu-profiles-l1.md), [gpu-5080-operator-guide.md](docs/gpu-5080-operator-guide.md), [ROADMAP.md](docs/ROADMAP.md).

### RTX 5080 CUDA gates — Phase 15 PASS, L2 FAIL merge, L3 STRICT PASS (Jun 2026)

**Why:** Metal sign-off (M5/M9) proved Phase 15 batch decode on Apple Silicon; CUDA 5080 (CT 1564, Proxmox) needed the same evidence before claiming cross-platform Phase 15 + borrowings L2/L3 status.

- **Phase 15 `phase15_inprocess_signoff.sh` PASS** — KV decode hook (`kv_decode_steps` native), multiseq `kv_inprocess_n_seq_max=2`, continuous batch decode (`batch_decode_in_c=true`) on RTX 5080 with patched b9611 `libllama.so` (`120-real`).
- **`scripts/phase15_inprocess_multiseq_smoke.sh`** — `ZEROLLAMA_GPU_PROFILE=0` on multiseq serve. **Why:** L1 `rtx-5080` sets `n_parallel=4`, overriding temp YAML `llama_parallel_slots: 2` (same pattern as `phase15_metal_signoff.sh`).
- **L2 `l2_cuda_full_gate.sh`** — stock **79.3** vs fork **56.9** tok/s @ 8192 ctx (OuteTTS 1B Q8; reruns ±1 tok/s); **FAIL merge** verdict (exit 1 = verdict fail, not broken run); compat smoke PASS.
- **L2 long-ctx 5080 (Jun 2026):** eliza-1 9B @ 8k/26624 — stock **~18.5** vs fork **~14.3** tok/s (~−22%); **no fork salvage at 27k**. 131k fork blocked: 9B needs ~31 GiB VRAM; 1B QJL `qjl1_256` incompatible with model head dim.
- **`gpu_profiles.py`** — emit `--checkpoint-every-n-tokens` (fork @ 96dd1a8); old `--ctx-checkpoint-interval` crashed llama-server on 131k leg.
- **`scripts/build_eliza_llama_server.sh`** + **`build_llama_server.sh`** — `LLAMA_BUILD_WEBUI=OFF` on Linux. **Why:** headless CT builds fail cmake `xxd.cmake` when HF WebUI download/npm build fails.
- **L3 `l3_cache_smoke.sh`** — **STRICT PASS** on eliza-1 9B (CT 1564): `L3_PREFIX_REPEAT=150`, cached turn2 **0.66s** vs no-cache **1.13s**; 1B Q8 remains SOFT PASS.
- **`scripts/l3_cache_smoke.sh`** — Linux/CUDA via `linux_runtime_serve_lib.sh`; `CUDA_LLAMA_MODEL` alias; `L3_PREFIX_REPEAT`; `ZEROLLAMA_GPU_PROFILE_CTX=1` on Linux (WHY: avoid n_ctx=1024 on long agent prefix).
- **`scripts/gpu_5080_session.sh`** — `RUN_E2E_PREFLIGHT` now respects env (default `1`). **Why:** Proxmox CT 1564 lacks vendored `cpp-httplib` for CGO; hardcoding preflight on blocked the official GPU gate — use `RUN_E2E_PREFLIGHT=0` on minimal trees; CI still runs `phase12_golden_ci.sh`.
- **Docs:** [gpu-5080-operator-guide.md](docs/gpu-5080-operator-guide.md) (Proxmox CT layout, gate sequence, preflight skip, WHYs), [gpu-profiles-l1.md](docs/gpu-profiles-l1.md), [gpu-profiles-l2.md](docs/gpu-profiles-l2.md), [gpu-profiles-l3.md](docs/gpu-profiles-l3.md), [ROADMAP.md](docs/ROADMAP.md).

### Phase 15 v31 — llama-kv-ext pin tracking + hybrid/iSWA resolve (Jun 2026)

**Why:** `llama-kv-ext.h` lived untracked in-tree — vendor sync could wipe it on pin bumps. Hybrid/iSWA models returned `unsupported_memory_type` even though attn KV is reachable via `get_base()` / `get_mem_attn()`.

- **`llama/patches/0015-ollama-llama-kv-ext-phase15.patch`** — formal patch on b9611 pin (cell map, tensor info, CMake entry).
- **`llama/llama.cpp/src/llama-memory-kv-ext.cpp`** — resolve `llama_kv_cache_iswa`, `llama_memory_hybrid`, `llama_memory_hybrid_iswa` to attn base cache; `llama_memory_kv_ext_classify`.
- **`scripts/phase15_llama_kv_ext_pin_check.sh`** — CI gate: patch + in-tree files + upstream `llama.h` deps at pin.
- **`runtime/native/kv_tensor_probe.c`** — exports `memory_kind` / `memory_kind_name` on probe.
- **Docs:** [phase15-llama-kv-ext-upstream.md](docs/phase15-llama-kv-ext-upstream.md).

**Still blocked:** writable cross-allocator PA→tensor page bind (needs upstream page-handle API); pure recurrent-only models.

### Phase 15 v30 — per-row C sampling in batch decode (Jun 2026)

**Why:** v29 batch steps sampled in Python via ctypes because v26 C path used one shared sampler (accept-state bleed) and `run_sample` always read the last logit row. v30 passes one sampler pointer per batch row so sampling stays in C with correct logit indices and isolated accept state.

- **`runtime/native/kv_decode_loop.c`** — `kv_decode_loop_run_batch_step` takes `smpl_ptrs[]` per row.
- **`runtime/native/kv_block_pool.c`** — `decode_loop_batch_step` accepts int (legacy) or list of smpl pointers.
- **`runtime/runtime/kv/native_decode_loop.py`** — `run_batch_step(..., smpl_ptrs=)`.
- **`runtime/runtime/worker/libllama_ctypes.py`** — `_decode_parallel_stream` uses C batch sampling when all row samplers are available.

### Phase 15 v29 — streaming batch decode + GPU sign-off (Jun 2026)

**Why:** v27 batched autoregressive steps for `generate_batch` but only returned full text at the end. v29 streams `seq_idx`-tagged chunks through the same C `run_batch_step` path so callers can consume interleaved tokens from N sequences. GPU sign-off needed a loopback hook because batch APIs are engine-internal (no public `/api/generate` batch yet).

- **`runtime/runtime/worker/libllama_ctypes.py`** — `_decode_parallel_stream()`; `complete_parallel_stream()`; shared `_parallel_jobs_and_smpls` / `_finalize_parallel_jobs`; non-stream collects from stream.
- **`runtime/runtime/worker/llama_inprocess.py`** — `completions_parallel_stream()` with batch path + sequential fallback.
- **`runtime/runtime/engine.py`** — `stream_generate_batch()` admits N requests and yields tagged NDJSON-shaped chunks.
- **`runtime/runtime/server/app.py`** — `POST /internal/generate-batch` (loopback-only) for GPU smokes and operator debug.
- **`scripts/phase15_batch_decode_smoke.sh`** — GPU batch + stream smoke; wired into `phase15_metal_signoff.sh` (step 3/5) and `phase15_inprocess_multiseq_smoke.sh`.
- **`scripts/phase15_runtime_kv_env.sh`** — **Why fix:** prefer sibling `../llama.cpp` (has `ggml.h`) over in-repo vendor stub; single venv Python build (avoids 3.9 universal overwrite that caused first-run Metal segfault).
- **Audit fixes:** post-prefill-only native sample (`batch_idx == -1`); sampler cleanup on `_parallel_jobs_and_smpls` failure; `RequestState.FINISHED` on stream batch stop; L3 disk-restore test binds `_prepare_seq_for_decode`.
- **GPU sign-off:** `./scripts/phase15_metal_signoff.sh` **PASS** (M4 Max, Jun 2026) — batch decode step reports `batch_decode_in_c=True`, non-stream + stream batch OK.
- **Tests:** `test_kv_decode_engine_batch.py`; `test_l3_inprocess_disk.py`.

### Phase 15 v28 — `/health` continuous batch plan export (Jun 2026)

**Why:** v27 wired batch decode but operators had no merged view of what `run_batch_step` would consume for N running sequences — only per-request `kv_forward_plans`. v28 adds `kv_continuous_batch` on `/health` for GPU sign-off.

- **`runtime/runtime/kv/forward_plan.py`** — `kv_continuous_batch_forward_plan()` merges running decode-phase rows into `kv_continuous_batch_step_plan`.
- **`runtime/runtime/engine.py`** — `kv_continuous_batch` health field + `kv_snapshot` export.
- **Tests:** `test_kv_forward_plan.py` — batch candidate filtering + `would_batch`.

### Phase 15 v27 — engine wiring for C continuous batch decode (Jun 2026)

**Why:** v26 shipped the C batch primitive but `generate_batch` still called `completion()` per row — N sequential `llama_decode` hot paths. v27 prefills each admitted sequence then merges autoregressive steps through `run_batch_step` when `n_seq_max>1` and the linked ext is available.

- **`runtime/runtime/worker/libllama_ctypes.py`** — `_prepare_seq_for_decode()` (extracted resume/clear); `complete_parallel()` + `_decode_parallel_non_stream()`; `n_batch` sized for `n_seq_max`; **one sampler chain per sequence** (audit fix: shared chain bled `llama_sampler_accept` state across rows).
- **`runtime/runtime/worker/llama_inprocess.py`** — `completions_parallel()` uses batch path when `native_batch_decode_available()`; sequential fallback when disabled or on error.
- **`runtime/runtime/kv/native_decode_loop.py`** — `native_batch_decode_available()`; env `ZEROLLAMA_KV_NATIVE_BATCH=0` disables.
- **Tests:** `test_kv_decode_engine_batch.py`; resume stub binds `_prepare_seq_for_decode`.

### Phase 15 v26 — continuous batch decode in C (Jun 2026)

**Why:** With `llama_parallel_slots>1`, each sequence previously called `llama_decode` separately from Python — N scheduler ticks, N GIL transitions. v26 batches N single-token rows into one C `llama_decode` (continuous batching scaffold).

- **`runtime/native/kv_decode_loop.c`** — `kv_decode_loop_run_batch_step()`; per-row page-bind validation; optional per-row C sampling.
- **`runtime/native/kv_block_pool.c`** — `decode_loop_batch_step`, `decode_batch_layout_multi` bindings; `batch_decode_in_c` on `decode_loop_status`.
- **`runtime/runtime/kv/native_decode_loop.py`** — `run_batch_step()`.
- **`runtime/runtime/kv/decode_plan.py`** — `kv_continuous_batch_step_plan()` for `/health` export.
- **Tests:** `test_kv_decode_batch_loop.py`.

### Phase 15 v25 — auto-link decode loop + 131k long-ctx validation (Jun 2026)

**Why:** C prefill/decode is the hot path but required `ZEROLLAMA_KV_DECODE_LOOP=1` on every build when libllama was present. The page-bind registry capped at 4096 pages (65536 tokens @ block_size=16), blocking 131072 ctx validation that L2 fork-only bench legs depend on.

- **`runtime/setup.py`** — auto-links libllama when found under `LLAMA_CPP_ROOT` / `LLAMA_CPP_LIB`; `ZEROLLAMA_KV_DECODE_LOOP=0` forces unlinked ext (CI); `=1` requires libllama and **exits non-zero** when missing (audit fix: was silently unlinked).
- **`runtime/native/kv_page_bind_internal.h`** — `KV_MAX_PAGES_PER_BIND` 4096 → 8192 (131072 ctx @ block_size=16).
- **`runtime/native/kv_decode_loop.c`** — post-prefill tensor probe moved to `kv_decode_loop_post_prefill_probe()` called after `Py_END_ALLOW_THREADS` (GIL-held registry write; fixes data race from v24 audit).
- **`scripts/phase15_kv_native_ci.sh`** — default unlinked build (`ZEROLLAMA_KV_DECODE_LOOP=0`); includes `test_kv_decode_long_ctx.py`.
- **Tests:** `test_kv_decode_long_ctx.py` — 8192-chunk prefill plan, page-bind boundary at 131072, C bind validation; `test_kv_native_build.py` — forced-link fail-fast.

### Phase 15 v24 — C decode loop page-bind validation + post-prefill tensor probe (Jun 2026)

**Why:** Native C prefill (`kv_decode_loop_run_prefill`) validated page tables only in Python before the GIL-released call; a ctypes bypass or future direct-C caller could write past PA-reserved pages. Tensor bind flags (`cell_pages_bound`, `tensor_pages_bound_slot`) only updated at `complete()` — too late for `/health` polling during long streaming prefills.

- **`runtime/native/kv_page_bind_internal.h`** — `kv_page_bind_validate_range()` (endpoint check, matches Python `validate_token_positions`).
- **`runtime/native/kv_decode_loop.c`** — validate each prefill chunk + decode step before `llama_decode`; post-prefill tensor probe (v25: moved to `kv_decode_loop_post_prefill_probe` in binding after GIL re-acquire).
- **`runtime/native/kv_block_pool.c`** — map bind validation failure (`-2`) to `ValueError` in Python bindings.
- **`runtime/runtime/kv/native_decode_loop.py`** — wrap C bind errors as `LlamaServerError`.
- **`scripts/phase15_metal_signoff.sh`** — step 4: `phase15_tensor_bind_probe.sh`; document why post-generate health cannot assert `tensor_pages_bound`.
- **Tests:** `test_decode_loop_prefill_c_page_bind_validation`.

### L3 — in-process disk cache audit fixes (Jun 2026)

**Why:** Audit found three correctness bugs in the initial disk parity implementation.

- **`runtime/worker/libllama_ctypes.py`** — `_save_slot_cache_disk` now derives token count from live `pos_max` via `sequence_kv_usage` instead of the caller's current-turn `prompt_tokens`; the token metadata written to the blob now matches the full KV history (prompt + all generated tokens). Removed `prompt_tokens` parameter from the API.
- **`runtime/worker/libllama_ctypes.py`** — disk restore guard no longer requires `decode_pos == 0`; any `not is_resume` + pinned slot attempts restore, fixing the case where a running sidecar has a stale owner and non-zero decode_pos.
- **`runtime/cache_bridge.py`** — `prepare_slot_cache_dir` takes `evict: bool = False`; eviction now runs at most once per session (on worker `start()`), not on every save turn.
- **`runtime/worker/llama_inprocess.py`** — calls `prepare_slot_cache_dir(evict=True)` at session start.
- **`runtime/tests/test_l3_inprocess_disk.py`** — updated test stubs to match new `_save_slot_cache_disk` signature (uses `sequence_kv_usage` mock).

### L3 — in-process disk cache parity (Jun 2026)

**Why:** subprocess L3 used llama-server `--slot-save-path`; in-process only had RAM resume (v17). Agent threads lost prefix KV on sidecar restart.

- **`runtime/cache_bridge.py`** — `inprocess_disk_cache_enabled`, `slot_cache_filename`, `slot_cache_file_path`, `prepare_slot_cache_dir`; `/health.llama_cache.inprocess_disk_cache`.
- **`runtime/worker/libllama_ctypes.py`** — `llama_state_seq_{save,load}_file` ctypes; save after pinned decode; restore before clear when slot cold.
- **`runtime/worker/llama_inprocess.py`** — `slot_cache_model_hash` from L1 argv cache types.
- **Env:** `ZEROLLAMA_LLAMA_CACHE_DISK=0` disables disk only.
- **Scripts:** `l3_inprocess_smoke.sh`, `l3_agent_bench.sh`.
- **Tests:** `test_l3_inprocess_disk.py`, cache_bridge disk helpers.

### L2 — CUDA gate audit fixes (Jun 2026)

**Why:** Post-creation audit found four bugs before the CUDA gate scripts were run in CI.

- **`scripts/l2_cuda_full_gate.sh`** — called `l2_runtime_compat_smoke.sh` (Darwin/Metal only: `macos_runtime_serve_lib.sh`, `lsof`, `.dylib`); replaced with `l2_cuda_runtime_compat_smoke.sh`.
- **`scripts/linux_runtime_serve_lib.sh`** — had `set -euo pipefail` at the top; removed. Sourced library scripts must not set shell options — the caller's `set -euo pipefail` must govern (a sourced `-e` would override caller error handling and cause unexpected exits).
- **`scripts/l2_cuda_bench.sh`** — redundantly sourced `runtime_uv_venv.sh` and `runtime_smoke_lib.sh` before sourcing `linux_runtime_serve_lib.sh`, which already sources both; removed the duplicate `source` calls.
- **`scripts/l2_cuda_bench.sh`, `scripts/l2_metal_bench.sh`** — Python bench core read `llama_server_args` for reporting from a static YAML file, which may not match the arguments chosen by `ZEROLLAMA_AUTO_CONFIG=1` at runtime. Fixed: now prefer `gpu_profile.llama_server_args` from the live `/health` response; fall back to YAML only when that field is absent.
- **`scripts/check_gpu_scripts.sh`** — added new CUDA gate scripts to the syntax-check array and added `grep` assertions for their key content.
- **`docs/gpu-profiles-l2.md`** — step 5 in the CUDA build/run section incorrectly called `l2_runtime_compat_smoke.sh` (Mac-only); updated to `l2_cuda_runtime_compat_smoke.sh`.

### L2 — CUDA bench gate scripts (Jun 2026)

**Why:** `l2_metal_bench.sh` is Darwin/Metal only (dylib, apple_silicon.yaml, lsof). CUDA sign-off needs a parallel bench path for RTX 5080-class hosts before the vendor-merge decision.

- **`scripts/linux_runtime_serve_lib.sh`** — shared sidecar start/stop for Linux; mirrors `macos_runtime_serve_lib.sh` (fuser, .so, single_gpu.yaml).
- **`scripts/l2_cuda_bench.sh`** — Linux A/B: stock vs fork decode tok/s + VRAM JSON at configurable `num_ctx`; same `L2_HIGH_CTX_WARMUPS` logic as Metal.
- **`scripts/l2_cuda_full_gate.sh`** — CUDA gate orchestrator: eval + compat + 8k/27k/131k bench legs + `l2_gate_report.sh` verdict.
- **`docs/gpu-profiles-l2.md`** — CUDA build/run section; updated runtime integration table; gate status entry.

### L2 — 131k decode bench warmups (Jun 2026)

**Why:** fork-only 131k leg measured first-touch KV alloc, not steady-state decode tok/s.

- **`scripts/l2_metal_bench.sh`** — `L2_HIGH_CTX_WARMUPS` (default 2 when `num_ctx >= 65536`); reports warmup count in JSON.
- **`scripts/l2_full_gate.sh`** — 131k leg uses `L2_BENCH_RUNS=2`, `L2_NUM_PREDICT=64`, warmups.

### Phase 15 v23 — unified prefill chunker + sign-off C pool defaults (Jun 2026)

**Why:** `kv_decode_prefill_plan` and `_prefill_prompt` duplicated chunk boundaries; exported `logits_last` did not match execution (v15 requires final prefill chunk True). Sign-off left C block pool off despite built ext.

- **`runtime/runtime/kv/decode_plan.py`** — `iter_prefill_execute_chunks()`; plan export uses it; final chunk `logits_last=True`.
- **`runtime/runtime/worker/libllama_ctypes.py`** — ctypes prefill calls shared chunker.
- **`scripts/phase15_runtime_kv_env.sh`** — shared env (`ZEROLLAMA_RUNTIME_KV_NATIVE=1`, native decode/sample); `phase15_runtime_kv_ext_build`.
- **`scripts/phase15_metal_signoff.sh`**, **`phase15_inprocess_signoff.sh`** — source env; build ext when `PHASE15_BUILD_KV_EXT=1` (default).
- **Tests:** updated `test_kv_decode_plan.py` for logits_last + execute parity.

### Phase 15 v22 — fix stale decode_pos after sequence clear; re-enable native sampling (Jun 2026)

**Root cause:** In `LlamaLoadedSession._complete_locked` (multiseq / shared-ctx path), `decode_pos` was read from the live KV sequence *before* the `is_resume` check. On the non-resume path `_clear_sequence` wiped the slot, but `decode_pos` still held the previous sequence's final position (7–13 in practice) and was forwarded unchanged as `current_pos` into `_decode_stream` / `_decode_non_stream`. The native C prefill skipped entirely (`start_pos >= n_prompt`) and `llama_sampler_sample` was called with no valid logits → intermittent segfault on Metal. Repro: `ZEROLLAMA_GPU_PROFILE=1` (n_seq_max=8), 5 sequential generates without resume — crashed on loop 4 reliably.

**Fix (one line):** reset `decode_pos = 0` immediately after `_clear_sequence`. The slot is empty; position 0 is the only correct value.

- **`runtime/runtime/worker/libllama_ctypes.py`** — `decode_pos = 0` after `_clear_sequence` on non-resume path; `infer_trace complete.clear stale_decode_pos=N` emitted for observability.
- **`runtime/runtime/infer_trace.py`** — new opt-in trace module (`ZEROLLAMA_INFER_TRACE=1`); wired into `engine.py` and `libllama_ctypes.py` for reload/reuse/prefill/sample phase logging.
- **`scripts/phase15_metal_signoff.sh`** — `ZEROLLAMA_KV_NATIVE_SAMPLE` default changed back to `1` (workaround removed now root cause is fixed).
- **`scripts/e2e_runtime_smoke.sh`** — removed Darwin `sleep` workaround before `/api/chat`.
- **`scripts/phase15_metal_crash_repro.sh`** — new repro bisect harness (runtime_loop / broker_gguf / phase14_full scenarios).
- **`runtime/tests/test_infer_trace.py`** — unit tests for `infer_trace` enable/disable.

**Verified:** `phase15_metal_signoff.sh` PASS with `ZEROLLAMA_KV_NATIVE_SAMPLE=1`; bisect 10/10 invocations × 5 generates = 50 calls with no crash.

### Phase 15 v21b — tensor bind probe correctness fixes (Jun 2026)

**Why:** v21 audit found five correctness issues in `kv_tensor_bind_attempt`: wrong early-exit blocker for `lctx==NULL`, stale-flag clear happened before `llama_get_memory` (obscuring failure path), `seq_max+1` could overflow `int32_t`, `blocker` on `/health` used a static fallback string even when a probe had run, and `accounting_ok` was compared as raw int.

- **`runtime/native/kv_tensor_probe.c`** — restructured guard order: `lctx==NULL` sets `KV_TENSOR_BLOCKER_NO_PAGE_API`; stale-flag clear is now inside the same block that precedes `llama_get_memory`; overflow guard replaces `seq_max+1` with `base + llama_token_cells` (avoids `INT32_MAX + 1` wrap).
- **`runtime/runtime/kv/page_bind.py`** — `blocker` field now uses the probe's own blocker string whenever a probe ran (cell_bound or not); `accounting_ok` normalised via `bool()` before `None`-guard; `accounting_aligned` in output is `None` when no probe ran.
- **`runtime/tests/test_kv_tensor_probe.py`** — new: `test_page_bind_health_blocker_from_probe_when_cell_bound`, `test_page_bind_health_blocker_fallback_when_no_probe`.
- **`runtime/tests/test_kv_page_bind.py`** — `test_page_bind_health_without_native_ext` now asserts `slots == []` and `bind_level is None`.
- **`docs/phase15-native-kv.md`** — `kv_page_bind` health field row expanded with `status`/`bind_level`/`blocker`/`slots` value catalogue.

### Phase 15 v21 — per-slot bind registry + post-decode warnings (Jun 2026)

**Why:** v20 bind state lived only inside ephemeral probe results; operators could not see per-slot `cell_pages_bound` on `/health` without a running request + linked probe.

- **`page_bind_slots()`** — C export of active registry rows; `/health.kv_page_bind.slots`.
- **`libllama_ctypes.py`** — post-decode warns on incomplete bind (`cell_map_gap`, etc.) when accounting is ok.
- **Scripts** — health smoke + metal signoff assert `slots`/`bind_level`; decode loop build prefers vendored fork.

### Phase 15 v20b — tensor bind audit fixes (Jun 2026)

**Why:** v20 audit found compile bug (C++ `cell_index_for` in `.c` file), wrong tensor for multi-stream, shifted-position cell map, stale bind flags, and health state machine inconsistency.

- **`kv_tensor_probe.c`** — use `llama_memory_kv_cell_for_pos` for stream; probe only live token pages + partial last page; clear stale bind flags before attempt.
- **`llama-kv-cache.cpp`** — `kv_tensor_k/v(kv_layer, stream)` returns per-stream 2D view when `n_stream>1`.
- **`llama-kv-ext.h`** — `llama_memory_kv_tensor_info(..., stream, ...)`.
- **`runtime/kv/page_bind.py`** — misaligned does not override `status=bound`.
- **Tests** — bound-not-overridden-by-misaligned health case.

### Phase 15 v20 — cell + tensor bind via llama-kv-ext (Jun 2026)

**Why:** v19 accounting bind could not resolve PA pages to llama KV storage. v20 adds a staging API in the pinned llama.cpp fork and wires zerollama's tensor probe to cell-index + K/V tensor verification after decode.

- **`llama/llama.cpp/include/llama-kv-ext.h`** — `llama_memory_kv_cell_for_pos`, `llama_memory_kv_cell_map_range`, `llama_memory_kv_tensor_info`.
- **`llama/llama.cpp/src/llama-memory-kv-ext.cpp`** — implementation for standard `llama_kv_cache`.
- **`llama/llama.cpp/src/llama-kv-cache.{h,cpp}`** — `cell_index_for`, `kv_tensor_k/v`.
- **`runtime/native/kv_tensor_probe.c`** — v20 bind attempt; `cell_pages_bound`, `tensor_pages_bound`.
- **`runtime/kv/page_bind.py`** — `status=bound`, `bind_level=tensor` when probe succeeds.
- **`runtime/setup.py`** — prefer `llama/llama.cpp` vendored root for linked builds.

**Requires:** rebuild libllama from fork before `ZEROLLAMA_KV_DECODE_LOOP=1` link.

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v20-ops--cell--tensor-bind-via-llama-kv-ext-jun-2026).

### Phase 15 v20a — native page table on forward plans (Jun 2026)

**Why:** v19 `page_bind_table` was script-only; operators comparing `kv_forward_plans.pages[]` to the C registry had no single JSON view. v20a mirrors the native registry on admitted plans with a parity flag.

- **`runtime/kv/tensor_probe.py`** — `page_table_native_parity()`.
- **`runtime/kv/forward_plan.py`** — `native_page_table`, `page_table_native_parity` when C registry populated.
- **`scripts/phase15_kv_native_ci.sh`** — adds `test_kv_tensor_probe.py`, `test_kv_decode_engine_resume.py`.
- **Tests** — forward plan native mirror; misaligned health status.

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v20a-ops--native-page-table-in-forward-plans-jun-2026).

### Phase 15 v19 — tensor bind scaffold (Jun 2026)

**Why:** v8–v18 seq-position bind could validate token ranges but could not map PA `block_ids` onto llama KV tensor pages — blocked on missing public llama.cpp page-handle API. v19 unblocks the path with accounting-level verify + operator probes so full tensor bind is a thin layer when upstream ships handles.

- **`native/kv_tensor_probe.c`** — `kv_tensor_probe_run`: `llama_get_memory`, seq positions, PA page fit vs live cells.
- **`native/kv_page_bind_internal.h`** — shared page bind registry for pool + probe.
- **`native/kv_block_pool.c`** — `page_bind_table(kv_slot)` export; `page_bind_tensor_probe(ctx_ptr, seq_id, kv_slot)` when linked.
- **`runtime/kv/tensor_probe.py`** — Python facade.
- **`runtime/kv/page_bind.py`** — health includes `tensor_probe`, `tensor_bind_ready`, `blocker`, `accounting_aligned`.
- **`runtime/worker/libllama_ctypes.py`** — post-decode tensor probe warn/strict.
- **`scripts/phase15_tensor_bind_probe.sh`** — build + table export smoke.
- **Tests** — `tests/test_kv_tensor_probe.py`.

**Still blocked for full tensor bind:** public llama.cpp API to attach external page ids to KV tensor storage.

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v19-ops--tensor-bind-scaffold-jun-2026).

### Phase 15 v18 — kv_resume health + L3 two-turn gate (Jun 2026)

**Why:** v16–v17 resume state lived only inside `LlamaLoadedSession._seq_last_owner` with no operator visibility. v18 exposes `/health.kv_resume` and adds a Metal sign-off step for two-turn L3 `prompt_cache_key` traffic.

- **`runtime/worker/libllama_ctypes.py`** — `resume_owner_snapshot()` for health export.
- **`runtime/engine.py`** — `_kv_resume_health()` on `/health` and `kv_snapshot`.
- **`scripts/phase15_metal_signoff.sh`** — step 3: two-turn L3 generate + `kv_resume` assert.
- **`scripts/phase15_health_smoke.sh`** — asserts `kv_resume` key.
- **Tests** — `test_kv_resume_health_*`, `test_generate_l3_second_turn_passes_current_pos`.

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v18-ops--kv_resume-health--l3-gate-jun-2026).

### Phase 15 v17 — L3 session resume owner (Jun 2026)

**Why:** v16b keyed slot ownership on `request_id`, but L3 pinned sessions (`slot_pinned=True`) allocate a **new** `request_id` every HTTP turn while reusing `prompt_cache_key` and `kv_slot`. Multi-turn agent chat therefore always failed the owner check, cleared good prefix KV, and re-prefilled from scratch — defeating L3 cache for in-process backends.

- **`runtime/cache_bridge.py`** — `slot_resume_owner_key(kv_bind_req)`: pinned → `cache:{prompt_cache_key}`; otherwise → `request_id`.
- **`runtime/worker/libllama_ctypes.py`** — `_seq_last_owner` (renamed from v16b `_seq_last_request_id`); resume guard uses `slot_resume_owner_key`; owner cleared on sequence clear and on `close()` (model teardown).
- **Tests** — `test_slot_resume_owner_key_*`, `test_complete_skips_clear_l3_second_turn`, `test_complete_clears_sequence_different_pinned_session`, `test_close_clears_seq_last_owner`.

**Still open:** tensor page bind (blocked on llama.cpp API).

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v17-ops--l3-session-resume-owner-jun-2026).

### Phase 15 v16b — slot-ownership guard for resume (Jun 2026)

**Why:** v16 added `decode_pos > 0` as the guard for skipping `_clear_sequence`, but that condition alone is insufficient: a *different* request can land on the same slot after the first one completes.  Without an ownership check, the second request would resume into stale KV from the first, producing corrupted generations.  v16b adds `_seq_last_owner` on `LlamaLoadedSession` (shipped as `_seq_last_request_id`, renamed in v17) so `complete()` only skips the clear when the incoming owner matches the last writer of that slot.

- **`runtime/worker/libllama_ctypes.py`**
  - `LlamaLoadedSession._seq_last_owner: dict[int, str]` — tracks last owning key per seq slot (v16b: `request_id` only).
  - `complete()` — skip `_clear_sequence` only when `decode_pos > 0` **and** owner matches; writes owner back after decode (stream and non-stream paths).
  - `_resolve_decode_current_pos` — `WHY` docstring explaining the no-op on the single-seq path.
- **`runtime/engine.py`** — `_decode_current_pos_for_request` gets a `WHY` docstring documenting the read-outside-lock pattern, why it is safe, and how `_seq_last_owner` re-validates under the lock.
- **Tests** — `tests/test_kv_decode_engine_resume.py`:
  - `test_complete_skips_clear_same_request_id` — same request resumes; no clear.
  - `test_complete_clears_sequence_different_request_id` — different request on same slot; clears.
  - `test_complete_clears_sequence_no_req_id_with_decode_pos` — `kv_bind_req=None`; conservative clear.
  - `test_complete_clears_sequence_when_current_pos_zero` — refactored onto shared helper.
  - `test_engine_decode_current_pos_for_request` — asserts exact `(lib, ctx, seq_id)` args to `current_pos_for_seq`.

**Still open:** tensor page bind (blocked on llama.cpp API).

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v16b-ops--slot-ownership-guard-jun-2026).

### Phase 15 v16 — engine resume wiring (Jun 2026)

**Why:** v15 added `_decode_stream(current_pos=)` but generate always started from position 0 and cleared the seq. v16 reads live llama seq positions at completion time and skips `_clear_sequence` when resuming.

- **`runtime/kv/physical.py`** — `current_pos_for_seq(lib, ctx, seq_id)`.
- **`runtime/engine.py`** — `_decode_current_pos_for_request()`; passed to `completion` / `completion_stream` / batch.
- **`runtime/worker/libllama_ctypes.py`** — `complete(..., current_pos=)`; skip clear when `decode_pos > 0`.
- **`runtime/worker/llama_inprocess.py`** — forwards `current_pos` / `current_positions`.
- **Tests** — `tests/test_kv_decode_engine_resume.py`.

**Still open:** tensor page bind (blocked on llama.cpp API). **Hardened by v16b** (slot-ownership guard).

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v16-ops--engine-resume-wiring-jun-2026).

### Phase 15 v15 — sampling in C + resume prefill wiring (Jun 2026)

**Why:** v14 hardened the C decode loop but sampling still round-tripped through ctypes each token, and `_decode_stream` always prefilled from position 0 despite `decode_work` exporting live `current_pos`.

- **`native/kv_decode_loop.c`** — `kv_decode_loop_sample`; optional `smpl` on `kv_decode_loop_run_step` (decode + sample in one GIL-released block).
- **`native/kv_block_pool.c`** — `decode_loop_sample(smpl_ptr, ctx_ptr)`; `decode_loop_step(..., smpl_ptr=0)` returns `(steps, token)` when sampling; `/health.kv_decode_loop.sampling_in_c`.
- **`runtime/kv/native_decode_loop.py`** — `run_sample`, `run_step(..., smpl_ptr=)`; `greedy_decode_tokens` uses C sampling when linked.
- **`runtime/worker/libllama_ctypes.py`** — `_decode_stream(..., current_pos=0)` wires remaining prefill + decode resume; C sampling on native fast path.
- **Tests** — `tests/test_kv_decode_stream_resume.py`; E2E patches `run_sample` for ctypes baseline.

**Still open:** tensor page bind; engine passes `current_pos` from `kv_physical` into generate (API ready on `_decode_stream`).

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v15-ops--sampling-in-c--resume-prefill-jun-2026).

### Phase 15 v14 — harden C decode loop (Jun 2026)

**Why:** v13 moved `llama_decode` into C but still held the GIL and lacked resume prefill + operator E2E confidence. v14 releases the GIL, validates page bind before C calls, supports `pos_start` for remaining prefill, and adds optional linked-build parity smoke.

- **`native/kv_block_pool.c`** — `Py_BEGIN_ALLOW_THREADS` around `kv_decode_loop_run_prefill` / `run_step`; `/health.kv_decode_loop.gil_released`; `decode_loop_prefill(..., pos_start=0)`.
- **`native/kv_decode_loop.c`** — `pos_start` on prefill (llama positions = `pos_start + tok_off`).
- **`runtime/kv/native_decode_loop.py`** — `pos_start`, `kv_slot` + `validate_token_positions`; `greedy_decode_tokens()` for E2E parity.
- **`runtime/worker/libllama_ctypes.py`** — passes `kv_slot` into native prefill/step calls.
- **Tests** — `tests/test_kv_decode_loop_e2e.py` (gated: `RUN_E2E_DECODE_LOOP=1`, `LLAMA_MODEL`, linked ext).
- **CI** — `scripts/phase15_kv_decode_loop_build.sh` checks `gil_released`; runs E2E when env set.

**Still open (v15+):** tensor page bind.

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v14-ops--harden-c-decode-loop-jun-2026).

### Phase 15 v13 — native C decode loop via llama_decode (Jun 2026)

**Why:** v12 proved `libllama` links; v13 moves the hot `llama_decode` call into C, reducing GIL contention. Sampling (`llama_sampler_sample`) stays in Python — it's negligible relative to the forward pass and allows reuse of the existing ctypes sampler chain.

- **`native/kv_decode_loop.c`** — `kv_decode_loop_run_prefill` (page-aligned chunking + repeated `llama_decode`); `kv_decode_loop_run_step` (single-token decode step). Manual heap `llama_batch` — **why:** `llama_batch_get_one` is stack-unsafe for chunked prefill; `llama_batch_init` leaves arrays uninitialized unless we fill every field.
- **`native/kv_decode_loop.h`** — declarations for both entry points (gated by `ZEROLLAMA_KV_DECODE_LOOP`).
- **`native/kv_block_pool.c`** — Python bindings `decode_loop_prefill(ctx_ptr, tokens, seq_id, block_size)` and `decode_loop_step(ctx_ptr, token, seq_id, current_pos)`, `#ifdef` gated.
- **`runtime/kv/native_decode_loop.py`** — `run_prefill()` / `run_step()` with `ctx_ptr: int` (ctypes `c_void_p` value); returns `None` when not linked.
- **`runtime/worker/libllama_ctypes.py`** (`_decode_stream`) — v13 fast path: C prefill → sample (ctypes) → C step loop; ctypes fallback when ext not linked or encoder model.
- **Tests** — `test_kv_decode_work_plan.py` covers `run_prefill` / `run_step` no-op safety when not linked.
- **CI** — `scripts/phase15_kv_decode_loop_build.sh` verifies `decode_loop_prefill` + `decode_loop_step` symbols.

**Audit (v13):** documented reserved `n_seq_max` in `kv_batch_alloc` (inner seq arrays are length 1 today); removed stale `sampled_out` from header comment; conftest only resets `vram_yaml_defaults._APPLIED` when a test actually applied YAML defaults.

**Runtime test hermeticity (Jun 2026):** **why** full pytest was failing on Python 3.9 / macOS from env leaks and syntax — `runtime/server/app.py` uses `Optional[]` (FastAPI needs live annotations); `tests/conftest.py` clears native `page_bind` slots and restores VRAM YAML env keys between tests; `engine.py` drops `zip(strict=True)`.

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v13-ops--native-c-decode-loop-jun-2026). **Next:** [v14 — harden C decode loop](docs/phase15-native-kv.md#v14-ops--harden-c-decode-loop-jun-2026) (shipped Jun 2026).

### Phase 15 v12 — libllama link build + probe (Jun 2026)

**Why:** v11 shipped `decode_loop_status` but always reported `ctypes`. v12 wires optional libllama linking at build time so operators can verify the extension links before a full C decode loop lands.

- **`runtime/setup.py`** — `ZEROLLAMA_KV_DECODE_LOOP=1` + `LLAMA_CPP_LIB` / `LLAMA_CPP_ROOT` → `-DZEROLLAMA_KV_DECODE_LOOP`, link `-lllama`, rpath.
- **`native/kv_decode_loop.c`** — calls `llama_max_devices()` as link probe; exposed on `/health.kv_decode_loop.llama_max_devices`.
- **`scripts/phase15_kv_decode_loop_build.sh`** — optional smoke (skips when libllama not built).
- **Audit (v11):** removed duplicate top-level `decode_steps` on forward plans (`decode_work.decode` is canonical); `_empty_prefill_plan` sets `prefill_complete: true`.

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v12-ops--libllama-link-build--probe-jun-2026).

### Phase 15 v11 — unified decode work plan + libllama link scaffold (Jun 2026)

**Why:** v9–v10 export separate `decode_prefill` and `decode_steps`; operators and a future C loop need one phase indicator. Linking libllama into the native ext is a separate build — v11 adds the probe and contract before the loop ships.

- **`kv_decode_work_plan()`** — unified `{phase, prefill, decode}` (`admit` / `prefill` / `decode` / `done`).
- **`kv_forward_plans[].decode_work`** — always present when admitted; includes live phase when `current_pos` known.
- **`current_pos_by_request_from_physical()`** — testable helper; engine uses it for forward-plan wiring.
- **`/health.kv_decode_loop`** — `{available, reason, link}`; C `decode_loop_status()` (linked loop blocked until `ZEROLLAMA_KV_DECODE_LOOP=1` + `LLAMA_CPP_LIB` at build time).
- **Tests / CI** — `tests/test_kv_decode_work_plan.py` in `phase15_kv_native_ci.sh`.

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v11-ops--unified-decode-work-plan--libllama-link-scaffold-jun-2026).

### Phase 15 v10 — in-progress decode plans (Jun 2026)

**Why:** v9 exported admit-time prefill plans (`pos_start=0`). Running requests need the **current** llama write position so operators see remaining prefill + planned decode steps on `/health` without guessing.

- **`kv_decode_prefill_plan(..., pos_start=)`** — plan remaining prompt from `current_pos`; `prefill_complete` when prefill done.
- **`kv_decode_step_plan()`** — single-token decode steps with `logits_last=True` (matches `_decode_stream`); `pending_prefill` while `current_pos < n_prompt`.
- **`kv_forward_plans[].decode_steps`** + **`plan_current_pos`** — when live seq positions available from `kv_physical`.
- **`next_pos_from_llama()`** — `llama_pos_max + 1` → next write position.
- **Engine** — `_kv_forward_plans_health()` wires `kv_physical.running[].llama_pos_max` into forward plans.

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v10-ops--in-progress-decode-plans-jun-2026).

### Phase 15 v9 — decode prefill plan on forward plans (Jun 2026)

**Why:** exit criterion #6 requires C batch layout wired to `kv_forward_plans`. v8 page-chunks at decode time; v9 exports the same plan on `/health` and `/internal/kv-snapshot` so operators and a future native decode loop share one contract without running inference.

**What shipped:**

- **`runtime/kv/decode_plan.py`** — `kv_decode_prefill_plan()`; page-aligned chunks + optional native `batch_layout` summary. **Why shared chunker:** calls the same `iter_prefill_decode_chunks` as `libllama_ctypes._prefill_prompt` — plan boundaries cannot drift from real decode.
- **`kv_forward_plans[].decode_prefill`** — present when request has admitted `block_ids` and `prompt_tokens`. **Why both guards:** waiting requests have no page table yet; exporting a plan without reserved blocks would mislead operators.
- **`logits_last: false` on every prefill chunk** — matches ctypes prefill (`_prefill_prompt` never sets logits on prefill batches). **Why not True on the last chunk:** the first sampled token’s logit comes from the decode loop’s separate single-token batch, not the final prefill batch.
- **`pos_start=0`** — plan covers the full prompt at admit time; continuation positions (`n_pos` after generation or multi-turn) stay a decode-time concern (v10+).
- **Tests / CI** — `tests/test_kv_decode_plan.py` (10 tests) in `phase15_kv_native_ci.sh`.

**Audit fixes (same release):**

- **`logits_last` on final prefill chunk** — earlier draft marked the last chunk `True`; corrected after tracing `_prefill_prompt`. **Why:** llama prefill batches do not emit sampling logits; marking the last chunk True would mislead a future C decode loop.
- **Test structure** — split `test_forward_plan_omits_decode_prefill_without_block_table` out of the page-boundary test; removed unused `pytest` import. **Why:** unrelated assertions in one test name hide failures.

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v9-ops--decode-prefill-plan-export-jun-2026).

### Phase 15 v8 — seq-position page bind + C decode batch layout (Jun 2026)

**Why:** ROADMAP exit criteria 5–6 need llama.cpp **tensor** APIs that do not exist on the public surface. Operators still need (a) PA page tables enforced before decode so generation cannot silently exceed reserved blocks, and (b) batch metadata built off the GIL hot path so a future native decode loop can wire to `kv_forward_plans` without another refactor. v8 ships the bookkeeping + layout layer; tensor mapping and `llama_decode` in C remain blocked upstream.

**What shipped:**

- **`runtime/native/kv_block_pool.c`** — `page_bind_set/clear/resolve/stats`, `decode_batch_layout`, `decode_prefill_chunks`. **Why C:** admission and decode share the same page table; keeping resolve + batch layout in the extension avoids Python list churn on every token batch.
- **`runtime/kv/page_bind.py`** — register PA page tables on scheduler admit, clear on `complete()`; `/health` `kv_page_bind.status=partial`, `bind_level=seq_position` when native ext built. **Why admit-time register:** decode validates against the table that was reserved at admission — not a stale export from `/health`.
- **`runtime/kv/native_decode_batch.py`** — C-built `llama_batch` field lists; page-aligned prefill chunks for prompts longer than one PA page. **Why page-aligned chunks:** matches PA page boundaries so bind validation and future tensor bind share the same token ranges.
- **`runtime/runtime/worker/libllama_ctypes.py`** — ctypes decode uses native batch builder; overrun raises `LlamaServerError` (not raw C exceptions). **Why:** stream/generate paths already catch `LlamaServerError`; operators see a clear KV bind failure instead of a traceback.
- **`runtime/runtime/scheduler/loop.py`** — `register_request_bind` on admit, `unregister_request_bind` on complete; block size from `pools[0].block_size`. **Why pool block_size:** `SchedulerLoop` has no separate config field — the pool is the source of truth for page size.

**M14 — `zerollama doctor --fix` clones llama.cpp:**

- **`cmd/doctor.go`** — runs `ensure_llama_cpp_sibling.sh` (with `ZEROLLAMA_REPO`) before `build_llama_server.sh`. **Why:** fresh clones failed opaquely at Metal build time; `mac_setup` already cloned first — doctor should match that order so `--fix` is a one-command bootstrap.

**Audit fixes (same release):**

- **`self.block_size` crash** — admission called `register_request_bind(..., block_size=self.block_size)` but `SchedulerLoop` has no such field → `AttributeError` on first admit. **Why:** use `pools[0].block_size` like every other KV callsite.
- **`tensor_pages_bound` type** — C `page_bind_stats` now returns Python `False` (not `0`); Python normalizes to `bool`. **Why:** `/health` JSON should not mix int/bool for the same semantic flag.
- **Duplicate bind validation** — validate once in `build_batch_from_tokens` / `_make_batch`, not again in chunk iterator. **Why:** hot path; same check twice bought nothing.
- **`n_predict=0` prefill** — decode loop now decodes the prompt when `limit=0`. **Why:** old condition `n_pos + batch.n_tokens < n_prompt + limit` skipped the only prefill decode when `limit=0`.

**Still blocked:** PA `block_ids` → llama **tensor** KV pages; full decode loop in C without ctypes.

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v8-ops--seq-position-page-bind--c-decode-batch-jun-2026) · ROADMAP Phase 15 criteria 5–6 · [mac-dev-setup.md](docs/mac-dev-setup.md) (M14 doctor).

### Ggml manifest `num_ctx` suggest + opt-in clamp (M12, Jun 2026)

**Why:** High-VRAM tier sets server default `num_ctx=262144`; merged manifest defaults pre-allocate full KV at ggml load and can hang qwen35/recurrent models before the first token. Phase 13 runtime already exposes `suggested_max_num_ctx` + opt-in clamp — the Go ggml scheduler had docs only, so operators on the legacy/Metal path had no API signal when manifest defaults exceeded VRAM.

**What shipped:**

- **`server/ggml_num_ctx.go`** — binary-search `suggested_max_num_ctx` from `fs/ggml.GraphSize` + file size vs **free** VRAM (same overhead floor as `sched.go`); optional clamp in `scheduleRunner` before `GetRunner`.
- **`GET /api/show`** — `ggml_num_ctx.suggested_max_num_ctx`; `merged_num_ctx` when the merged default exceeds the suggestion (separate fields so clients are not confused about clamped vs requested context).
- **`/api/chat` / `/api/generate`** — `ggml_num_ctx` only when clamp applied (mirrors runtime `vram_num_ctx`; default off preserves operator trust).
- **Env (default off):** `ZEROLLAMA_GGML_CLAMP_NUM_CTX`, `ZEROLLAMA_GGML_SUGGEST_CTX_MAX`, `ZEROLLAMA_GGML_VRAM_MARGIN`.

**Audit fixes (same release):**

- **No total-VRAM fallback** — early code used startup `totalVRAM` when free was unknown; that over-suggested context (total ≠ free). **Why:** fail-open suggest is safer than pretending all installed VRAM is available.
- **Free-VRAM cache (~2s) for `/api/show`** — avoids live `GPUDevices` probe on every show (CLI startup calls show often). Load path refreshes via `LoadedRunnersForDiscovery()`.
- **Removed silent `recover()` in suggest hi-bound** — `llm.LoadModel` always returns parsed GGUF; panics should not be swallowed.

Doc: [scheduling-vram-policy.md](docs/scheduling-vram-policy.md#ggml-vram-suggest-and-opt-in-clamp-m12-jun-2026).

### Go ollama-engine Metal stability — qwen35moe on Apple Silicon (Jun 2026)

**Why:** `qwen35*` is in `OllamaEngineRequired()` — the Go ollama-engine path is the long-term default on every OS. On M-series Macs, load aborted in `ggml_backend_sched_reserve` with `GGML_ASSERT(tensor->buffer == NULL)` during worst-case graph reserve (after Metal init succeeded). C aborts do not return Go errors, so a darwin-only legacy gate had blocked the Go engine for qwen35. Operators also saw stale Metal free-memory during scheduling and per-load `/health` latency on the training submit path.

**Root cause (sched_reserve):** `newTensor` eagerly allocated backing buffers for **every** graph intermediate while `LoadOperationFit` also called `sched_reserve`, which assigns the same tensors via `ggml_backend_tensor_alloc`. Double assignment tripped the assert on qwen35moe’s large MoE + SSM graphs on Metal.

**What shipped:**

- **`Context.Persistent()`** — `ml/backend.go`, `ml/backend/ggml/ggml.go`: KV/recurrent buffer contexts mark tensors for eager alloc; transient graph intermediates defer to `sched_reserve` / `sched_alloc_graph`. **Why:** KV cells must exist before forward; graph scratch must not pre-claim buffers the scheduler owns.
- **kvcache** — `causal.go`, `encoder.go`, `recurrent.go`, `recurrent_checkpoints.go`: buffer contexts call `.Persistent()`.
- **Darwin routing** — `llm/server.go` `pickOllamaEngine`: removed darwin-only legacy fallback for `qwen35*`/`qwen3next`. **Why:** root allocator fix makes Go engine viable; legacy llamarunner + compat remains for llama-server path and investigation.
- **Worst-case reserve** — removed qwen35 arch blocklist from `runner/ollamarunner/runner.go` `reserveWorstCaseGraph` (was masking the assert instead of fixing allocation).
- **Metal unified free memory** — `discover/runner.go` no longer skips free-memory refresh on darwin/arm64; `discover/metal_unified.go` + `capMetalUnifiedFree()` fill Metal device free bytes from host `vm_stat`. **Why:** bootstrap subprocess free memory is process-local; scheduler layer fit needs unified pool headroom, not stale zeros.
- **Runtime health cache** — `server/inference_workload.go`: 500ms TTL on `runtimeInferenceHealth()`. **Why:** training submit idle-wait hit `:8081/health` on every ggml load attempt.
- **Smoke** — `scripts/qwen35_mac_smoke.sh`: accept `thinking` when `response` is empty (qwen3.6 thinking models); `scripts/runtime_smoke_lib.sh` `smoke_unload_ggml_runners`: single-quoted Python for unload payload (shell brace expansion broke `{...}` dicts).

**Sign-off (M4 Max):** full gate `./scripts/metal_signoff.sh` with `RUN_E2E_QWEN35=1 RUN_E2E_QWEN35_MODEL=qwen3.6:latest` — Phase 13–15 + qwen35 Go ollama-engine generate/unload PASS (Jun 2026).

Doc: [qwen35-apple-silicon.md](docs/qwen35-apple-silicon.md), [apple-silicon-metal.md](docs/apple-silicon-metal.md#go-ollama-engine-sched_reserve-jun-2026). ROADMAP: [M10](docs/ROADMAP.md#apple-silicon--metal-track).

### Metal sign-off gate repairs (Jun 2026)

**Why:** After the sched_reserve fix, `./scripts/metal_signoff.sh` still failed on unrelated smoke wiring — not Metal regressions. Each failure blocked validating the full M3 + Phase 15 + qwen35 chain on one command.

**What shipped:**

- **Sign-off order** — `scripts/m3_metal_signoff.sh` runs **qwen35 before Phase 15**. **Why:** Phase 15’s exit trap stops the `:8081` sidecar; qwen35 handoff/resume needs runtime `/health` after ggml unload.
- **Phase 15 multiseq** — `scripts/phase15_metal_signoff.sh` sets `ZEROLLAMA_GPU_PROFILE=0` for the `llama_parallel_slots=2` temp YAML. **Why:** L1 `apple-silicon-128g` sets `n_parallel=8`, overriding yaml `2` and breaking `kv_inprocess_n_seq_max` assertions.
- **L3 + inprocess** — `llama_inprocess.py` / `llama_cpp_python.py` accept `cache_prompt` (ignored on inprocess). **Why:** L3 cache bridge passes the kwarg from `engine.py`; Phase 14 generate returned HTTP 500 without it.
- **Unload payload** — `smoke_unload_ggml_runners` uses single-quoted Python `-c` (see Metal stability smoke bullet above).

**Full gate (M4 Max, Jun 2026):**

```bash
LLAMA_CPP_ROOT=../llama.cpp ./scripts/build_llama_server.sh
RUN_E2E_QWEN35=1 RUN_E2E_QWEN35_MODEL=qwen3.6:latest ./scripts/metal_signoff.sh
```

### L2 elizaOS/llama.cpp fork evaluation spike (Jun 2026)

**Why:** L1 tuned flags on stock `q8_0` cache types cap VRAM and tok/s vs eliza-v3’s QJL/Polar/TurboQuant kernels. Replacing `vendor/llama-cpp-b9611/` blindly would break Ollama patches and qwen35 compat — L2 evaluates the fork in an isolated sibling tree first.

**What shipped:**

- **`runtime/llama_fork.py`** — fork detection via `ZEROLLAMA_LLAMA_FORK` or `llama-server --help` probe for `--ctx-checkpoints`.
- **`gpu_profiles.py`** — merges `_eliza_fork_llama_server_flags` (QJL/Polar cache types) when fork enabled; emits `--ctx-checkpoints` on fork builds only.
- **`scripts/build_eliza_llama_server.sh`** — clone/build `elizaOS/llama.cpp` @ `96dd1a8` into `../eliza-llama.cpp`.
- **`scripts/l2_fork_eval.sh`** — probe + profile argv smoke.
- **GPU JSON** — `_eliza_fork_llama_server_flags` on 3090/4090/5080/5090/H200 profiles.
- **`/health.llama_fork`** — observability for operators.
- **`scripts/l2_metal_bench.sh`**, **`l2_runtime_compat_smoke.sh`**, **`l2_gate_report.sh`**, **`l2_full_gate.sh`** — Metal A/B + verdict orchestrator.
- **`m3_metal_signoff.sh`** — `RUN_E2E_L2=1` runs full gate.
- Doc: [gpu-profiles-l2.md](docs/gpu-profiles-l2.md).

**Gate (open):** measured tok/s + VRAM win on 5080 + M-series before vendor merge.

**Metal gate run (M4 Max 128 GiB, Jun 2026):**

| Model | ctx | Stock | Fork |
|-------|-----|-------|------|
| eliza-1-2b | 8192 | 37.6 tok/s, q8_0 | 20.5 tok/s, tbq4_0/tbq3_0 |
| eliza-1-27b | 26624 | 13.2 tok/s | 12.7 tok/s |
| eliza-1-27b | 131072 | admission fail (est.) | 5.0 tok/s (fork-only leg) |

**Runtime compat smoke:** PASS (`l2_runtime_compat_smoke.sh`). **Verdict:** stock wins decode on measured A/B legs; fork enables 131k ctx where stock path rejects. **Vendor merge blocked** pending CUDA 5080 bench + qwen35 ggml smoke. Scripts: `l2_full_gate.sh`, `l2_gate_report.sh`.

### L3 prompt cache key → llama-server slot bridge (Jun 2026)

**Why:** L1 raises peak tok/s via batch/parallel flags; L2 may shrink KV footprint. Neither fixes **repeat prefill** — agent threads and fixed system prompts re-run the full prompt every turn when Phase 15 assigns a fresh dynamic `id_slot` and releases it on `complete()`. Eliza-v3 solves this with stable cache keys hashed into llama-server slots plus optional on-disk slot save. L3 ports that bridge without pulling in Eliza’s device UI or bundle catalog.

**What shipped:**

- **`runtime/cache_bridge.py`** — key resolution (`conversationId` → segments → prefix → `promptCacheKey`), `derive_slot_id(key, parallel)`, `--slot-save-path` argv, mtime TTL eviction, `/health` snapshot. **Why separate module from `gpu_profiles`:** cache keys are per-request/session; GPU JSON is per-hardware.
- **Admission** — `_admit_one()` sets `prompt_cache_key`, pinned `kv_slot`, `slot_pinned` before scheduler tick. **Why at admit:** slot must be reserved before `/completion` so llama-server sees a stable `id_slot`.
- **Phase 15 loop** — `SlotAllocator.try_acquire()` for pinned slots; concurrent same-slot requests re-queue; allocator releases tracking on complete but llama-server keeps KV (next turn re-derives same slot from key hash).
- **Subprocess completions** — `cache_prompt: true` when a cache key is present. **Why:** tells llama-server to persist prefix KV into the slot (RAM + optional disk under `--slot-save-path`).
- **Inprocess / wheel workers** — `cache_prompt` kwarg accepted and ignored on `LlamaInprocessWorker` and `llama-cpp-python` backend. **Why:** `engine.py` always passes the flag after L3 admit; Phase 14 Metal sign-off hit HTTP 500 without a matching worker signature.
- **Disk cache** — `~/.cache/zerollama/llama-cache/<modelHash>/`; hash includes GGUF path, draft model, `--cache-type-k/v` so fork vs stock profiles do not collide. TTL via `ZEROLLAMA_LLAMA_CACHE_TTL_MS` (default 1h) for llama-server `slot_*.bin` names.
- **`/health.llama_cache`** — enabled, root, `model_hash`, `slot_save_path`, file stats; `model_loaded` false when GGUF path configured but not on disk yet.
- **Batch** — `generate_batch` + `completions_parallel` pass per-request `cache_prompt`; `options.prompt_cache_keys[]` for per-row keys.
- **`scripts/l3_cache_smoke.sh`**, **`l3_gate_report.sh`** — two-turn cache latency gate; `RUN_E2E_L3=1` on `m3_metal_signoff.sh`.
- **Fixes:** gpu profile emits `-fa on` (stock llama-server rejected bare `-fa`); `macos_runtime_serve_lib` preserves `ZEROLLAMA_RUNTIME_LLAMA_BACKEND=subprocess` for L2/L3 smokes.

**Audit fixes (Jun 2026, second pass):**

- **Canonical GGUF path in `model_hash`** — `_canonical_model_path()` resolves symlinks before hashing. **Why:** same weights via LM Studio symlink vs absolute path must share one `--slot-save-path` directory; fragmented hashes miss disk-restored prefix KV on restart.
- **Orphan model-hash dir sweep** — `evict_orphaned_cache_dirs()` on llama-server cold start. **Why:** L2 profile switches change cache-type fields in the hash; stale sibling dirs accumulate expired `slot_*.bin` files and waste disk without touching the active model hash.
- **Batch `prompt_cache_keys` semantics** — when the list is present, out-of-range indices get no key (no flat-key fallback). **Why:** unrelated batch rows must not accidentally share one pinned slot when only some rows specify keys.
- **`complete()` ordering** — document and enforce `unregister_request_bind` before `SlotAllocator.release`. **Why:** releasing first lets the next tick `try_acquire` the slot while native page bind is still being torn down — stale block ids on decode.

**Operator env:** `ZEROLLAMA_LLAMA_CACHE=0` (disable), `ZEROLLAMA_LLAMA_CACHE_ROOT`, `ZEROLLAMA_LLAMA_CACHE_TTL_MS`. Requires `-np > 1` (L1 profile) for multi-session pinning; subprocess backend for full disk save. Doc: [gpu-profiles-l3.md](docs/gpu-profiles-l3.md).

**Remaining:** in-process disk parity; agent-scale bench on CUDA 5080.

### L1 GPU profiles — per-hardware llama-server autotune (Jun 2026)

**Why:** Phase 13 prevents OOM and suggests `num_ctx`; it does not pick batch size, parallel slots, flash-attn, or MTP draft depth. Operators on 5080 vs 4090 vs M4 Max were either using conservative one-size YAML defaults or hand-copying flags from eliza. Wrong `-b`/`-np` wastes VRAM headroom or silently caps throughput.

**What shipped:**

- **`runtime/gpu_profiles.py`** — loads `runtime/configs/gpu/*.json`, merges flags into `RuntimeConfig.llama_server_args()`. **Why at config load:** runtime inprocess/subprocess paths already consume `llama_server_args()`; Go ggml is a separate track (Phase 17).
- **NVIDIA** — match by `nvidia-smi` name or VRAM bucket (`index.json` `fallback_buckets`). Applied only when loading `single_gpu.yaml` / `dual_4090.yaml`.
- **Apple Silicon** — RAM tiers from `hw.memsize` (`apple_silicon_16g` … `128g`). Applied only when loading `apple_silicon.yaml`. **Why tiers not chip ID:** same M-series SKU ships multiple unified-memory sizes.
- **Stock safety** — fork-only cache types → `q8_0`; fork-only argv (`ctx_checkpoints`) never emitted; `mlock` default **off** in JSON (opt-in `LLAMA_SERVER_EXTRA_ARGS=--mlock`).
- **`runtime/nvidia_probe.py`** — shared cached `nvidia-smi` probes for autoconfig + profiles. **Why split module:** avoids circular import between `autoconfig` and `gpu_profiles`.
- **Observability** — `/health.gpu_profile` (`id`, `bucket_label`, `unified_memory_gb`, `n_parallel`, emit flags). `macos_metal_smoke.sh` prints profile fields.
- **M4 Max sign-off** — `apple-silicon-128g`: `-np 8`, `-c 131072` aligned with Phase 13 `suggested_max_num_ctx`.

**Operator env:** `ZEROLLAMA_GPU_PROFILE=0` (disable), `ZEROLLAMA_GPU_PROFILE_CTX=0` (skip profile `-c`), `LLAMA_SERVER_EXTRA_ARGS` (override). Doc: [gpu-profiles-l1.md](docs/gpu-profiles-l1.md).

**Remaining:** CUDA 5080 benchmark gate to validate/tune `rtx-5080.json` (Apple side marked done in ROADMAP L1).

### Mac GPU bootstrap discovery — Metal reported as CPU (Jun 2026)

**Why:** Operators on M-series Macs saw `inference compute library=cpu`, `total_vram="0 B"`, and `offloaded 0/N layers to GPU` even though llama.cpp init logged `Apple M4 Max`. The scheduler uses bootstrap GPU discovery to pick layer layout and VRAM-tiered default `num_ctx`; empty discovery forced CPU-only scheduling and capped context at 4096 on 128 GiB machines.

**Root cause:** Bootstrap discovery asks the ollama-engine runner `GET /info`. The handler loaded a dummy model with **zero GPU layers**, which calls `ensureDevices(true)` on first init and sets `GGML_DISABLE_METAL=1` permanently (`sync.Once`). Metal never registered in the discovery subprocess, so `/info` returned no GPUs. Inference runners are separate processes and still init Metal, but the **main server** believed there was no GPU.

**What shipped:**

- **`DiscoverBackendDevices()`** — `ml/backend/ggml/ggml.go`: probe path that calls `ensureDevices(false)` so Metal registers during bootstrap only. **Why:** discovery must not reuse the `num_gpu=0` CPU-only gate meant to avoid runtime sidecar contention on actual loads.
- **Ollama-engine `/info`** — `runner/ollamarunner/runner.go`: use `DiscoverBackendDevices()` when no model is loaded instead of dummy zero-layer `model.New`. **Why:** faster (no temp GGUF) and correct on darwin unified memory.
- **Docs** — [apple-silicon-metal.md](docs/apple-silicon-metal.md#gpu-bootstrap-discovery-jun-2026): symptoms, fix, startup speed env vars.

**Verify after rebuild:** `./scripts/build_zerollama_mac.sh && ./zerollama serve` → log shows `inference compute library=Metal`, `total_vram` ~100+ GiB, model load logs `offloaded N/N layers to GPU`.

**Related (harmless load/chat warnings):** qwen35 GGUFs may log `control-looking token … was not control-type` (bad `token_type` in blob; llama.cpp overrides) and `embeddings required but some input tokens were not marked as outputs` (llamarunner embedding mode vs chat batch). See [qwen35-apple-silicon.md — Token warnings](docs/qwen35-apple-silicon.md#token-warnings-jun-2026).

### Prompt truncation surfaced in API responses (Jun 2026)

**Why:** When input exceeded `num_ctx`, runners logged `truncating input prompt` but clients got a normal 200 with no indication that most of the prompt was dropped.

**What shipped:** Final `/api/chat` and `/api/generate` responses (and streams' last chunk) now include:

- `prompt_truncated: true` and `original_prompt_tokens` when token-level truncation occurred in the runner
- `messages_truncated: true` and `messages_dropped` when `chatPrompt` removed older messages

Set `"truncate": false` on the request to get HTTP 400 instead of silent truncation.

### Model unload after create/stop + manifest `num_ctx` vs load-time KV (Jun 2026)

**Why:** Operators updated `num_ctx` via `/api/create` (manifest showed 262144) but `/api/ps` still reported `context_length: 4096` — the in-memory runner was never evicted. `zerollama stop` returned success while the model stayed loaded when `refCount > 0` or the scheduler key did not match the loaded runner. Separately, persisting **`num_ctx: 262144` as the model default** caused generation to hang: llamarunner pre-allocates KV for the manifest context at **load** time, not per request.

**Root causes:**

- **`/api/create`** updated manifest blobs only; the ggml scheduler reused the warm runner (`needsReload` never ran until the next inference request with different merged options).
- **`expireRunner`** only queued unload when `refCount <= 0`; active or leaked references left the runner resident while HTTP returned `done_reason: "unload"`.
- **Manifest `parameters.num_ctx`** is merged in `modelOptions()` and passed to `llama.Load` — KV/recurrent buffers are sized for that value immediately. Large defaults (262K on qwen35moe) can block or hang before first token; **request `options.num_ctx`** on an already-loaded smaller context is a different path (Hermes / runtime may grow per request; ggml may still `needsReload` when runner options differ).

**What shipped:**

- **`expireRunner`** — always schedules unload; `processExpiredRunner` retries while `refCount > 0`. **`findLoadedRunner`** — match by `ModelPath`, then `ShortName` / `Name`. **Why:** stop and empty-prompt unload must not silently no-op.
- **`/api/create`** — after successful manifest write, evicts any loaded runner for that model. **Why:** parameter changes (`num_ctx`, `num_gpu`, …) must apply on next load without manual stop.
- **Docs** — [qwen35-apple-silicon.md](docs/qwen35-apple-silicon.md#manifest-num_ctx-vs-request-options), [scheduling-vram-policy.md](docs/scheduling-vram-policy.md#go-ggml-scheduler-keep_alive-unload-and-num_ctx-at-load).

**Operator guidance:** Keep manifest default **`num_ctx` modest (e.g. 4096)** for fast reliable loads on ggml; pass large context via **`options.num_ctx` per request** when Hermes or the runtime detects need. Verify unload: `curl …/api/generate -d '{"model":"…","prompt":"","keep_alive":0}'` then empty `/api/ps`. `/api/ps` `context_length` reflects the **loaded** runner, not the manifest or a single request's options.

### Mac dev bootstrap tiers (Jun 2026)

**Why:** Another developer cloning the repo could not run `./scripts/mac_setup.sh` successfully without your exact layout: **`../llama.cpp` had to exist manually**, **metal sign-off ran by default** (needs a pulled text GGUF in `~/.ollama/models`), and **CI smokes default `OLLAMA_HOST=:8080`** while daily **`zerollama serve` uses `:11434`** — copy-pasting smoke curl against the wrong port looked like a broken server.

**What shipped:**

| Tier | Goal | Command |
|------|------|---------|
| **0** | Build + serve | `./scripts/dev_bootstrap.sh` |
| **1** | Chat | `./zerollama pull llama3.2:3b` |
| **2** | Metal sign-off | `MAC_SETUP_SIGNOFF=1 MAC_SETUP_GO=0 MAC_SETUP_BUILD=0 ./scripts/mac_setup.sh` |
| **3** | qwen35 smoke | `RUN_E2E_QWEN35_MODEL=tag ./scripts/qwen35_mac_smoke.sh` |

- **`dev_bootstrap.sh`** — thin entry; sets `MAC_SETUP_SIGNOFF=0`, `MAC_SETUP_LLAMA_CLONE=1`. **Why:** one command name for “fresh clone, any checkout path.”
- **`ensure_llama_cpp_sibling.sh`** — shallow-clone `LLAMA_CPP_VERSION` pin to `${REPO}/../llama.cpp`. **Why:** runtime inprocess and sign-off need `libllama.dylib`; failing late in `build_llama_server.sh` was opaque.
- **`zerollama doctor --fix`** — runs `ensure_llama_cpp_sibling.sh` before Metal `build_llama_server.sh` (same order as `mac_setup`). **Why:** tier-0 bootstrap should not require knowing about sibling clone scripts — `doctor --fix` is the self-service path on fresh checkouts.
- **`mac_setup.sh`** — sign-off **off** by default; `mac_setup_has_signoff_model()` skips sign-off with pull instructions when no GGUF; tier hints at end. **`MAC_SETUP_LLAMA_OPTIONAL=1`** for ggml-only dev without llama build.
- **`build_llama_server.sh`** — default `LLAMA_CPP_ROOT=${REPO}/../llama.cpp` (was `${REPO}/../../llama.cpp`). **Why:** sibling path must not depend on repo nesting depth.
- **`qwen35_mac_smoke.sh`** — documents `OLLAMA_HOST=:11434` override for daily serve.
- **Docs** — [mac-dev-setup.md](docs/mac-dev-setup.md) (tiers, port table), [development.md](docs/development.md), [README.md](README.md).

**Ports (why two Go ports):** upstream Ollama convention is `:11434`; CI/sign-off scripts historically used `:8080` to avoid clashing with a system Ollama. Smokes set `OLLAMA_HOST` internally — daily dev does not.

ROADMAP: [M14](docs/ROADMAP.md#apple-silicon--metal-track).

### Unified Mac build (Jun 2026)

**Why:** Operators had to remember two scripts — `build_zerollama_mac.sh` (ggml) and `build_production_mac.sh` (MLX dylibs) — to run safetensors from repo-root `./zerollama`.

**What shipped:**

- **`build_mlx_dylibs_mac.sh`** — shared CMake install for MLX Metal v3/v4 (dev `build/metal-v*/` or production `dist/darwin-arm64/` via `INSTALL_PREFIX`).
- **`build_zerollama_mac.sh`** — `BUILD_MLX=auto` (default): builds MLX dylibs when `../mlx` exists and `build/metal-v3/.../libmlxc.dylib` is missing; `BUILD_MLX=0` for fast ggml-only; `BUILD_MLX=1` / `MLX_FORCE=1` to force rebuild.
- **`build_production_mac.sh`** — regenerates `ggml-metal-embed.metal` before `build_darwin.sh`; arm64 MLX cmake delegated to `build_mlx_dylibs_mac.sh`.
- **`zerollama doctor`** — MLX fix hint points at `BUILD_MLX=1 ./scripts/build_zerollama_mac.sh` for dev.

Doc: [mac-dev-setup.md](docs/mac-dev-setup.md#dev-vs-production-mlx-layout).

### Go ollama engine on darwin (qwen35) — superseded (Jun 2026)

**Superseded by:** [Go ollama-engine Metal stability — qwen35moe on Apple Silicon](#go-ollama-engine-metal-stability--qwen35moe-on-apple-silicon-jun-2026) above. The darwin legacy gate and qwen35 worst-case reserve blocklist are removed; sched_reserve root fix + M4 Max sign-off landed.

### Mac smoke gaps (Jun 2026)

**Why:** M4 Max sign-off exposed three Mac-only failure modes that looked like one “broken Metal” bug but were independent: SSE proxies hung without terminal frames, runtime + legacy ggml both touched Metal on one device, and `num_gpu=0` still registered the Metal backend at first ggml init.

**What shipped:**

- **Proxy v1 SSE hang** — Python runtime always yields `data: [DONE]` on error mid-stream; Go proxy uses `copyRuntimeResponseBody` (flush after each chunk). **Why:** Gin buffered `io.Copy` until EOF; partial SSE streams hung curl/CI for 20+ minutes.
- **Darwin Metal contention** — `server/darwin_ggml_policy.go`: block ggml when runtime `llama_server=true`; contention checks run **before** `PrepareForLegacyRunner` so the sidecar is not evicted for a load that will be skipped. **Why:** dual Metal residency wedged the GPU silently.
- **`num_gpu=0` Metal init** — `GGML_DISABLE_METAL` gates Metal backend registration in C++; Go sets it before first `OnceLoad` on CPU-only loads. **Why:** embed/CPU-only smokes still initialized Metal and contended with the runtime sidecar.
- **Scheduler HTTP status** — `ErrRuntimeInferenceModel` → 400, `ErrDarwinMetalContention` → 503 (was generic 500). **Why:** operators need actionable routing errors, not “internal server error.”
- **e2e smokes** — `RUN_E2E_STREAM_MAX` cap; legacy ggml skipped on darwin when runtime holds Metal unless `RUN_E2E_LEGACY_FORCE=1`.

Doc: [docs/apple-silicon-metal.md](docs/apple-silicon-metal.md#scheduler-errors-http-status).

### Apple Silicon polish (Jun 2026)

**Why:** M10 qwen35 VL manifests could pick `clip` as primary family; LM Studio MLX imports need full disk copy but listed models anyway; qwen35 parser lost thinking text when streams ended without `</think>`; catalog hid MLX models silently on `statfs` errors; GGUF+config.json dirs failed pull; Windows CI broke on `syscall.Statfs`.

**What shipped:**

- **`PrimaryFamily()`** — routes renderers/parsers/thinking for VL manifests where projector (`clip`) was stored first; returns `""` for projector-only manifests (`server/model_family.go`). **Why:** create-time layer order stored `ModelFamily=clip` on qwen35 VL blobs.
- **Qwen35 parser** — `flushDoneEvents` on stream end (thinking, whitespace-after-close, trailing content). **Why:** truncated streams dropped reasoning text that never received a close tag.
- **LM Studio integration (v0.0.1)** — discover `~/.lmstudio/models`, merge into `list`/`/api/tags`, pull-from-cache for GGUF and MLX safetensors. **Why:** avoid re-downloading weights LM Studio already fetched.
  - **Native MLX import** — `ImportSafetensorsFromDirectory` when `config.json` + `.safetensors` present. **Why:** MLX→GGUF conversion fails on dtypes like `U32`.
  - **Disk checks** — `ImportCopyBytes` / `HasDiskForImport`; catalog hides MLX when `OLLAMA_MODELS` volume lacks ~model size + 512 MiB; pull fails early with readable error. **Why:** repack doubles disk use; mid-import ENOSPC wastes operator time.
  - **`OLLAMA_LMSTUDIO_LIST_ALL=1`** — list all discoverable models anyway (pull still enforces). **Why:** hidden models confused operators on tight disks.
  - **`dirIsMLXSafetensors`** — only MLX-layout dirs count toward copy bytes. **Why:** legacy safetensors without `config.json` symlink like GGUF and must stay visible in catalog.
  - **`weightFilesOnly`** — strip `config.json` before GGUF convert. **Why:** multi-file dirs (GGUF + HF metadata) sent JSON to the GGUF parser.
  - **Portable free space** — `diskspace_unix.go` / `diskspace_windows.go`. **Why:** `syscall.Statfs` is not available on Windows CI.
- **Opt-in qwen35 smoke** — `./scripts/qwen35_mac_smoke.sh` (runtime handoff → Go ollama-engine generate); `RUN_E2E_QWEN35=1` on `metal_signoff.sh` / `m3_metal_signoff.sh` (**runs before Phase 15** — sidecar teardown order). **Why:** qwen35 uses ggml on `:8080`, not the runtime inprocess path covered by Phase 14 alone.
- **Mac build** — `build_zerollama_mac.sh` passes `-ldflags` version (`VERSION` env, default `0.0.1`).

Docs: [lmstudio-import.md](docs/lmstudio-import.md), [apple-silicon-metal.md](docs/apple-silicon-metal.md), [qwen35-apple-silicon.md](docs/qwen35-apple-silicon.md), [mlx-routing-policy.md](docs/mlx-routing-policy.md).

## [0.0.1] — 2026-06-12

First tagged zerollama build with embedded version string. Includes LM Studio cache import, MLX disk policy, and Mac polish items above. Operators: `./scripts/build_zerollama_mac.sh && ./zerollama serve`.

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

**Why:** `MLX_VERSION` / `MLX_C_VERSION` are independent of the ggml llama.cpp pin — safetensors inference uses **`libmlx.dylib` + `libmlxc.dylib`**, not CGO ggml. Bumping pins without rebuilding leaves `mlxrunner` on stale Metal code (wrong kernels, ABI drift vs regenerated Go/C shims). Use **`BUILD_MLX=1 ./scripts/build_zerollama_mac.sh`** (dev) or **`./scripts/build_production_mac.sh`** (release layout) after pin bumps.

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

**Why:** Operators need a repeatable gate beyond `doctor` — runtime Metal inprocess is the **daily Mac path** (`apple_silicon.yaml`), while qwen35 validates the **Go ollama-engine ggml** path on the same Metal device. Sign-off proves Phase 13–15, optional qwen35, and tools without CUDA-centric `gpu_5080_session.sh`.

**One command (recommended):**

```bash
LLAMA_CPP_ROOT=../llama.cpp ./scripts/build_llama_server.sh
RUN_E2E_QWEN35=1 RUN_E2E_QWEN35_MODEL=qwen3.6:latest ./scripts/metal_signoff.sh
```

**Why qwen35 before Phase 15 in the script:** Phase 15 stops the runtime sidecar on exit; qwen35 needs `:8081` for training-handoff and resume after ggml unload.

**Passed (GPU, Jun 2026):**

- **Phase 13:** `/tmp/metal-session.json` — `metal-unified` probe, L1 GPU profile, autotune catalog.
- **Phase 14:** inprocess from YAML (`llama_backend_source=config`); generate/chat/stream; tokenize + render-chat; Go proxy.
- **Qwen35 (opt-in):** `qwen3.6:latest` via Go `--ollama-engine`; generate + API unload; thinking field OK.
- **Phase 15:** KV decode hook (`kv_decode_steps`); multi-seq (`llama_parallel_slots=2`, `kv_inprocess_n_seq_max=2` with `ZEROLLAMA_GPU_PROFILE=0`).
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
| **Qwen35 in full gate** | Phase 15 killed sidecar before qwen35 resume | qwen35 runs **before** Phase 15 in `m3_metal_signoff.sh`; opt-in `RUN_E2E_QWEN35=1` |
| **Phase 15 multiseq vs L1** | `apple-silicon-128g` forced `n_parallel=8` | Multiseq step uses `ZEROLLAMA_GPU_PROFILE=0` + temp yaml `llama_parallel_slots: 2` |
| **Phase 14 + L3 cache_prompt** | Inprocess worker lacked kwarg | Accept/ignore `cache_prompt` on inprocess and wheel backends |
| **Scheduler 400/503** | Contention returned generic HTTP 500 | `handleScheduleError` maps runtime-routed → 400, Metal contention → 503 |

Scripts: `./scripts/metal_signoff.sh`, `./scripts/qwen35_mac_smoke.sh`. Guide: [docs/apple-silicon-metal.md](docs/apple-silicon-metal.md).

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
