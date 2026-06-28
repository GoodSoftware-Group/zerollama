# Inference smoke testing

Operator checklist for validating **local inference** on a GPU host (e.g. RTX 5080/4090). **Why this doc exists:** Zerollama runs **two** local inference stacks today—the embedded **Python runtime** (`llama-server` subprocess on loopback `:8081`) and the **legacy ggml runner** (`zerollama runner` on the main HTTP port). They share one GPU but do not coordinate VRAM automatically; smoke tests prove each path works and document how to switch between them without mystery 503/404/OOM errors.

**Third reference arm (optional):** clone upstream Ollama and serve on another port for A/B — no Python sidecar, Go→llama-server only. See [upstream-ollama-diff.md](./upstream-ollama-diff.md) and [llama-cpp-backend.md](./llama-cpp-backend.md).

---

## What you are proving

| Layer | Command / artifact | Why it matters |
|-------|-------------------|----------------|
| Go unit tests | `go test ./server/... ./envconfig/...` | Handlers, cloud catalog merge, runtime proxy logic—no GPU, deterministic CI. **Why `OLLAMA_NO_CLOUD` in `TestMain`:** machines with `ELIZACLOUD_API_KEY` merge hundreds of cloud models and break list/tags tests that expect a small fixture set. |
| Runtime health | `GET :8081/health` | Sidecar is up; check `llama_model`, `inference_state`, `vram_budget`, `autoconfig`. |
| VRAM pre-flight | `./scripts/runtime_vram_estimate.sh <gguf>` | **Why:** same as `/internal/vram-estimate` — budget before load ([phase13-runtime-vram.md](./phase13-runtime-vram.md)). |
| Runtime generate | `POST :8081/api/generate` or `RUN_E2E_GPU=1` script | Python → `llama-server` → GGUF (`LLAMA_MODEL`). |
| Runtime chat | `POST :8081/api/chat` or `RUN_E2E_GPU=1` script | Plain chat; tools use Go `render-chat` when model has a parser (generic JSON path otherwise). |
| Runtime v1 chat | `POST :8081/v1/chat/completions` or `RUN_E2E_GPU=1` script | Phase 13 `prepare_v1_chat` + optional `vram_num_ctx` when clamp on. |
| Go proxy | `RUN_E2E_PROXY=1` script | `/api/generate`, `/api/chat`, `/v1/chat/completions` via `:8080` (manifest `options.gguf`). |
| Legacy runner | `POST :8080/api/generate` with a **pulled** model name | Manifest-backed models (e.g. `llama3:latest`) via ggml—what most users expect from `ollama run`. |

**Smoke complete** when health + `RUN_E2E_GPU=1` + `RUN_E2E_PROXY=1` + at least one legacy model request succeed.

**One-shot wrapper** (coordination + GPU + proxy, including stream paths):

```bash
./scripts/gpu_smoke_all.sh
./scripts/gpu_health_report.sh   # post-load calibration / autotune hints
```

| Script | Role |
|--------|------|
| `gpu_smoke_all.sh` | Coordination + `RUN_E2E_GPU=1` + `RUN_E2E_PROXY=1` + health report |
| `gpu_health_report.sh` | `/health` tuning summary (Python `runtime.gpu_health_report`) |
| `runtime_vram_estimate.sh` | Pre-load VRAM budget for a GGUF + `num_ctx` |
| `e2e_coordination_smoke.sh` | Go↔runtime mirror fields only |
| `serve_gpu_example.sh` | Example env for 5080-class single-GPU serve |
| `check_gpu_scripts.sh` | `bash -n` GPU scripts + import `runtime.gpu_health_report` (no GPU) |
| `phase12_golden_ci.sh` | `check_gpu_scripts` + `go test -run Golden` + tools meta pytest (no GPU); also run in CI |
| `gpu_phase13_snapshot.sh` | JSON snapshot of `/health` + optional `/internal/vram-estimate` for 5080 calibration |
| `gpu_clamp_smoke.sh` | `RUN_E2E_VRAM_CLAMP=1` runtime generate (serve must set `VRAM_CLAMP_NUM_CTX=auto\|1`) |
| `phase12_capture_tool_transcript.sh` | Capture real tools-chat output for Harmony/parser golden updates (GPU) |
| `gpu_5080_session.sh` | `RUN_E2E_PREFLIGHT=1` (default) + `gpu_smoke_all` + Phase 13 snapshot + recommendations; **`RUN_E2E_PREFLIGHT=0`** skips Go golden when CGO httplib vendoring missing (Proxmox CT) — **why:** GPU smokes should not fail on parser compile in minimal trees; CI still runs `phase12_golden_ci.sh`. Optional `RUN_E2E_PHASE14=1` … — **official 16GB gate** — see [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md) |
| `macos_metal_smoke.sh` | Darwin: Phase 12 go golden + metal probe pytest + coordination + `/health` autoconfig/probe — see [apple-silicon-metal.md](./apple-silicon-metal.md) |
| `gpu_metal_session.sh` | Darwin one-shot: `macos_metal_smoke` + Phase 13 snapshot + optional Phase 14 inprocess — Mac counterpart to `gpu_5080_session.sh` |
| `m3_metal_signoff.sh` / `metal_signoff.sh` | Full Mac Metal gate (Phase 13–15). **`RUN_E2E_QWEN35=1`** adds qwen35 **before Phase 15** — **why:** Phase 15 stops the sidecar; qwen35 needs runtime handoff/resume. **`RUN_E2E_L3=1`** runs prefix-cache smoke — **why:** verify stable `prompt_cache_key` wiring + `/health.llama_cache` (latency win optional on tiny models). M4 Max PASS Jun 2026. |
| `phase17_llama_server_smoke.sh` | Go → llama-server E2E (`--llama-server-backend`); uses pulled tag via `P17_MODEL` / `RUN_E2E_PROXY_MODEL`; needs `LLAMA_SERVER_BIN`. Doc: [phase17-llama-server.md](./phase17-llama-server.md) |
| `phase17_linux_auto_smoke.sh` | Linux-only: plain `zerollama serve` (`P17_LINUX_AUTO=1`); asserts `/api/status` `backend.llama_server=auto` |
| `phase16_edge_smoke.sh` | Phase 16 edge (`serve --edge`, runtime off); wraps Phase 17 smoke. Doc: [phase16-thin-edge.md](./phase16-thin-edge.md) |
| `phase11_metal_admission_smoke.sh` | **Darwin** Phase 11 admission pytest + live coordination/`/health` admission fields (`apple_silicon.yaml` defaults); JSON via `HEALTH_JSON` env (not heredoc argv) |
| `phase13_metal_vram_smoke.sh` | **Darwin** Phase 13 VRAM pytest + live `/internal/vram-estimate` + `gpu_phase13_snapshot.sh`; estimate JSON via `ESTIMATE_JSON` env |
| `phase11_13_15_metal_signoff.sh` | **Darwin** ordered gate: Phase 11 → 13 → 15 (`phase15_kv_native_ci` + optional `phase15_metal_signoff` + upstream KV watch); `METAL_SELF_START=1` bootstraps sidecar+Go; **`macos_export_llama_cpp_paths`** prefers vendor `llama-cpp-<pin>` for linked `_kv_native` (M4 Max PASS Jun 2026) |
| `phase16_edge_build_smoke.sh` | No GPU: `-tags edge` unit tests + edge binary build + `zerollama -v` marker + runner stub message (CI regression); **why:** proves v1 subprocess stub and v2 CGO exclusion compile cleanly. Captures CLI/runner output before `grep` — **why:** `grep -q` in pipes under `pipefail` false-fails with SIGPIPE even when output matches |
| `RUN_E2E_UPSTREAM_GGUF=1` | 5080 bundle: sets `RUN_E2E_P17`, `RUN_E2E_P17_LINUX_AUTO`, `RUN_E2E_EDGE` in `gpu_5080_session.sh` — **why:** one flag for upstream-shaped ship-hardware sign-off |
| `serve_linux_auto.sh` | Linux operator helper: plain `zerollama serve` with auto llama-server when binary discoverable |
| `serve_edge.sh` | Phase 16 edge serve (`--edge`, runtime chat off) |
| `phase15_upstream_kv_watch.sh` | No GPU: scan in-tree + ollama-upstream `llama.h` for writable page-handle symbols (Phase 15 criterion 5) |
| `phase17_l2_pin_status.sh` | No GPU: stock vs eliza L2 pin report + merge gate pointers (Phase 17 criterion 7) |
| `phase17_llama_server_vision_smoke.sh` | Opt-in vision chat+image on llama-server (`RUN_E2E_P17_VISION=1`, `P17_VISION_MODEL`); verifies serve log routes through llama-server subprocess. Doc: [phase17-llama-server.md](./phase17-llama-server.md) |
| `flash_moe_smoke.sh` | **Darwin opt-in** Flash-MoE: tier 0 = unit tests + `--moe-sidecar` binary check; tier 1 = direct llama-server startup; tier 2 = `zerollama serve` E2E (`RUN_E2E_FLASH_MOE=1`). **Why tiered:** sidecar + MoE GGUF are operator-local, not in git. Doc: [flash-moe.md](./flash-moe.md) |
| `ane_probe_smoke.sh` | **Darwin opt-in** ANE bridge smoke via `zerollama ane-probe`. **Why subprocess:** private ANE APIs isolated from main binary. Doc: [ane-probe.md](./ane-probe.md) |
| `video_expand_cache_smoke.sh` | Native video xfer unit gate — expansion/session/URL LRU + preflight spans — **why:** SGLang Tier 1 caches without GPU/VLM; runs `go test` on `server/modality` + `openai`. Doc: [sglang-multimodal-borrowings.md](./sglang-multimodal-borrowings.md) |
| `video_agent_cache_smoke.sh` | Agent two-turn session cache — raw video + **pre-expanded** layout restore; modality + OpenAI `video_url` + Qwen3-VL render + runner stub tests; **`RUN_E2E_VIDEO_AGENT=1`** live `/api/chat` + `/v1/chat/completions` + preprocessed turn 2 + ffmpeg lavfi + log grep — **why:** proves resend-clip agent loop across API shapes. Needs `VIDEO_SMOKE_MODEL` for live. |
| `video_l3_agent_gate.sh` | Combined operator gate — runs `video_agent_cache_smoke.sh`; **`RUN_E2E_L3=1`** adds `l3_cache_smoke.sh` + `l3_gate_report.sh` — **why:** video session cache and L3 prefix KV use the same `prompt_cache_key` discipline but different layers. |
| `video_agent_infer_smoke.sh` | Live VLM inference gate — two-turn `/api/chat`; strict pass on turn-2 `cached_prompt_tokens` (L3 subprocess or **ollama-engine input cache**); v1 `cached_tokens` advisory — **`RUN_E2E_VIDEO_AGENT_INFER=1`** + `VIDEO_SMOKE_MODEL`; **`VIDEO_AGENT_INFER_SOFT=1`** when MLX/cache off; agent turns send **`enable_prefix_mm_cache: true`** with `prompt_cache_key`; **`VIDEO_AGENT_INFER_PREPROC=1`** optional padded+`grid_thw` infer leg (requires `VIDEO_AGENT_GO_LOG` for `preprocessed layout session cache hit`); **`VIDEO_AGENT_INFER_PREFIX_MM_WARN=1`** optional prefix-mm hint leg (no session key); **`VIDEO_AGENT_INFER_VIT_SESSION=1`** strict ViT session overlay leg (requires `VIDEO_AGENT_GO_LOG`); grep `padded_input_ids runner inject`, `precomputed_embedding runner inject`, `processor_output runner inject`, `vision grid hints`, `vision embed session cache hit`, `vision embed global cache hit`, `precomputed_embedding global cache hit`, `precomputed_embedding session cache hit`, `grid_thw hint resize`. |
| `video_agent_infer_gate_report.sh` | Verdict printer for infer JSON report — **why:** operator sign-off like `l3_gate_report.sh`. |
| `gen_video_testdata.sh` | Writes `server/modality/testdata/lavfi_1s_64x64.mp4` via ffmpeg lavfi — **why:** optional Phase D fixture without committing binary blobs to git. |
| `l3_cache_smoke.sh` | Two-turn same cache key; JSON timing report — **why:** L3 is prefill-bound; gate checks bridge wiring before agent-scale bench. Needs subprocess backend + L1 `-np > 1`. Doc: [gpu-profiles-l3.md](./gpu-profiles-l3.md) |
| `l3_spec_cache_smoke.sh` | Spec decode × prefix cache policy — **why:** eagle3/mtp/dflash must disable `cache_prompt` + disk persist; checks `/health.llama_cache.policy`. Default `L3_SPEC_METHOD=ngram` (no draft GGUF). Doc: [gpu-profiles-l3.md](./gpu-profiles-l3.md) |
| `l3_prefix_cache_trace_replay.sh` | Offline golden trace replay — **why:** SWA/draft-spec regressions without GPU; replays `tests/fixtures/prefix_cache_golden.jsonl` against `KVCacheSpec`. |
| `l3_prefix_block_pool_smoke.sh` | Prefix block pool policy — **why:** hash-chain verification before `cache_prompt`; offline pytest + optional live. |
| `l3_radix_prefix_smoke.sh` | Cross-slot Radix prefix share — **why:** same system prompt, different cache keys; offline plan replay + optional `L3_RADIX_LIVE=1` (forces vendor llama-server, probes `/kv/seq-copy`, checks `radix_seed` trace). Doc: [radix-prefix-share.md](./radix-prefix-share.md) |
| `l3_gate_report.sh` | PASS/FAIL from `l3_cache_smoke.sh` JSON (strict latency or soft wiring pass) |
| `l1_cuda_calibrate.sh` | L1 profile OFF vs ON (+ `L1_SWEEP_NP`) on production GGUF — **why:** tune `rtx-5080.json` from ship hardware, not eliza port |
| `l1_cuda_concurrent_bench.sh` | N parallel `/api/generate` (barrier-synced) OFF vs ON — **why:** validate `n_parallel=2` wins under agent concurrency, not just single-stream |
| `l1_cuda_full_gate.sh` / `l1_gate_report.sh` | L1 production gate — calibrate + concurrent + merged `gate.json` verdict |
| `l1_full_gate.sh` | Platform dispatch (CUDA full gate vs Mac metal gate) |
| `l3_production_gate.sh` | L3 strict gate @ 27k ctx on 9B+ — **why:** 8k smoke can pass wiring while production agent prefix at 26k+ is where cache pays |
| `l3_cuda_full_gate.sh` / `l3_gate_report.sh` | L3 production gate — 8k smoke + 27k production + merged verdict |
| `l3_full_gate.sh` | Platform dispatch (CUDA full gate vs Mac smoke) |
| `l2_full_gate.sh` / `l2_gate_report.sh` | Fork vs stock A/B + runtime compat — **why:** vendor merge blocked until measured wins; see [gpu-profiles-l2.md](./gpu-profiles-l2.md) |
| `qwen35_mac_smoke.sh` | Opt-in qwen35/qwen3.6 generate on darwin via Go ollama-engine; handoffs runtime Metal first; accepts `thinking` or `response` — see [qwen35-apple-silicon.md](./qwen35-apple-silicon.md) |
| `e2e_training_ops_smoke.sh` | `GET /api/train/status` + jobs; optional TCP ping (no train job submit) |
| `repro_shared_interpreter_health_hang.sh` | Training + embedded runtime on `19180`/`19181`; 5× `/health` must not hang |
| `phase14_backend_smoke.sh` | Phase 14: one backend on running serve (`RUN_E2E_PHASE14=1`, `/internal/tokenize`, render-chat). Preflight prints `llama_backend` + `llama_backend_source`. **Rebuild + restart serve** — [phase14-inprocess-llama.md](./phase14-inprocess-llama.md) |
| `phase14_inprocess_smoke.sh` | 5080 ctypes GPU sign-off: `RUN_E2E_INPROCESS=1` + `llama_backend_source=env` (ROADMAP exit #3) |
| `phase14_yaml_config_smoke.sh` | Backend smoke with `llama_backend_source=config`; infers backend flags from `/health` (YAML key, no env override) |
| `phase14_subprocess_default_smoke.sh` | Backend smoke with `llama_backend_source=default` (packaged subprocess; no env override, no YAML `llama_backend` key) |
| `phase14_wheel_cpu_smoke.sh` | Wheel CPU sign-off (`RUN_E2E_LLAMA_CPP_PYTHON=1`, `llama_cpp.gpu_mode=cpu` after generate) |
| `phase14_wheel_gpu_smoke.sh` | Optional wheel GPU offload smoke (`llama_cpp.gpu_mode=gpu` after generate) |
| `phase14_enable_yaml_inprocess.sh` | Enable `llama_backend: inprocess` in `single_gpu.yaml` after ctypes sign-off |
| `phase14_both_backends.sh` | Phase 14: restarts serve for `inprocess` then `llama-cpp-python` (embed-safe; wheel CPU ~10 min). Sets `RUN_E2E_LLAMA_BACKEND_SOURCE=env` per backend |
| `phase14_5080_signoff.sh` | One-shot 5080 gate: `phase14_both_backends` + YAML config full + Phase 15 multi-seq (self-contained restarts) |
| `phase14_yaml_config_full_smoke.sh` | Temp YAML with `llama_backend: inprocess`; asserts `llama_backend_source=config` without editing repo YAML |
| `phase14_serve_env.sh` | Source before `zerollama serve` — **why:** unset `ZEROLLAMA_RUNTIME_URL` so Go embeds `:8081` (exporting URL forces external sidecar mode) |
| `phase15_kv_native_ci.sh` | Build C `BlockPool` + KV pytest bundle + `phase15_health_smoke.sh` (no GPU); [phase15-native-kv.md](./phase15-native-kv.md) |
| `phase15_health_smoke.sh` | Assert `/health` KV keys (`kv_forward_plans`, `kv_page_bind`, `kv_live_physical`, …) via `InferenceEngine` only; `kv_page_bind.status=partial` when native ext built |
| `phase15_inprocess_signoff.sh` | One-shot Phase 15 GPU gate: KV decode hook + multi-seq (self-contained restarts). |
| `phase15_inprocess_kv_smoke.sh` | Self-contained: starts inprocess serve, asserts `kv_decode_steps` on generate and post-generate `/health` (GPU host) |
| `phase15_inprocess_multiseq_smoke.sh` | Temp YAML `llama_parallel_slots: 2` + inprocess; asserts `kv_inprocess_n_seq_max` and generate; ends with `phase15_batch_decode_smoke.sh` when linked ext available |
| `phase15_batch_decode_smoke.sh` | GPU: continuous batch decode via `POST /internal/generate-batch` (non-stream + stream); needs multiseq sidecar (`kv_inprocess_n_seq_max≥2`, `batch_decode_in_c`); [phase15-native-kv.md](./phase15-native-kv.md#continuous-batch-decode-v26v30) |
| `phase15_metal_signoff.sh` | Mac Metal Phase 15 gate (5 steps): KV hook → multiseq → **batch decode** → L3 two-turn → tensor bind; sources `phase15_runtime_kv_env.sh` |
| `phase15_runtime_kv_env.sh` | Shared env for Phase 15 GPU smokes — **why:** one place to enable C pool + native decode + build linked ext. **`phase15_runtime_kv_ext_build`** `rm -rf build` before link (CPU CI unlinked ext must not reuse); prefers **vendor** `llama-cpp-<pin>` when present (`macos_export_llama_cpp_paths` on Mac) |
| `gpu_harmony_capture.sh` | Optional real-weight harmony capture — **needs ~40+ GiB host RAM** for `gpt-oss:20b` MXFP4 on runtime path; **not** required on 5080 (~19 GiB); CI uses Go golden |

CI (`.github/workflows/zerollama-regression.yaml`): Phase 12 is covered by `go test ./server/...` (Golden tests) and runtime pytest (`test_go_render_chat.py`), plus `check_gpu_scripts.sh`. Optional self-hosted: `.github/workflows/zerollama-gpu-smoke.yaml` (`workflow_dispatch`; repo vars `GPU_SMOKE_*`, serve must be up).

On a GPU host: `RUN_E2E_PREFLIGHT=1 ./scripts/gpu_smoke_all.sh` or `./scripts/phase12_golden_ci.sh` (all parts).

**Phase 12 golden (no GPU):** `go test ./server/... -run Golden` or `./scripts/phase12_golden_ci.sh go`; Python: `./scripts/phase12_golden_ci.sh py` (needs `pip install -e runtime/.[serve]`).

Optional tools smoke (runtime `/api/chat` with tools — must return HTTP 200 and `done`; 501 means vision/think/logprobs routed to legacy):

```bash
RUN_E2E_GPU=1 RUN_E2E_TOOLS=1 ./scripts/e2e_runtime_smoke.sh
```

Asserts assistant `content` or `tool_calls` on `/api/chat` and `/v1/chat/completions` (does not golden-match a specific tool invocation).

With proxy smoke (`RUN_E2E_PROXY=1`), the same tools checks run on `:8080` when `RUN_E2E_TOOLS=1` and `RUN_E2E_GPU=1`.

```bash
RUN_E2E_GPU=1 RUN_E2E_PROXY=1 RUN_E2E_TOOLS=1 ./scripts/e2e_runtime_smoke.sh
# or:
RUN_E2E_TOOLS=1 ./scripts/gpu_smoke_all.sh
```

**5080 full stack** (runtime + proxy + optional legacy ggml + tools on a pulled parser model):

```bash
CGO_ENABLED=1 go build -o zerollama .
export LLAMA_MODEL LLAMA_SERVER_BIN
RUN_E2E_PREFLIGHT=1 \
RUN_E2E_TOOLS=1 \
RUN_E2E_PROXY_MODEL=your-local-tag \
RUN_E2E_LEGACY=1 RUN_E2E_LEGACY_MODEL=your-local-tag \
./scripts/gpu_smoke_all.sh
```

`gpu_smoke_all` calls `POST :8081/internal/inference/resume` when training handoff or pause blocks loads (after coordination, before proxy/tools, and before health report). Legacy ggml runs **after** runtime/proxy steps (`RUN_E2E_LEGACY_ONLY=1`). `RUN_E2E_LEGACY_MODEL` defaults to `RUN_E2E_PROXY_MODEL` or `llama3.2:3B`. Smokes pass `num_ctx` via `RUN_E2E_NUM_CTX` (default `4096`).

---

## Prerequisites

**Go build** (embed + training need CGO):

```bash
cd ~/zerollama
export CGO_ENABLED=1
go build -o zerollama .
```

**Python runtime** (embedded or sidecar):

```bash
cd runtime && pip install -e ".[serve]"
```

**`llama-server`** built for **your** GPU arch. **Why:** a 4090 build (`sm_89`) on a 5080 (`sm_120`) loads the wrong CUDA kernels; CMake must find a real `nvcc` (not a missing `cuda-13` path).

```bash
# RTX 5080 (Blackwell) example — use the toolkit that has bin/nvcc
export CUDA_HOME=/usr/local/cuda-12.8   # adjust after: ls /usr/local/cuda*/bin/nvcc
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"

cd ~/zerollama
CMAKE_CUDA_ARCHITECTURES=120-real ./scripts/build_llama_server.sh
export LLAMA_SERVER_BIN=~/llama.cpp/build/bin/llama-server
```

**Model on 16GB VRAM:** prefer **quantized** GGUF (Q4/IQ). **Why:** FP16 weights for multi‑billion‑parameter models often exceed 15GB once KV cache is reserved.

**Single-GPU config** (avoid dual-4090 tensor split on one card):

```bash
export ZEROLLAMA_RUNTIME_CONFIG=~/zerollama/runtime/configs/single_gpu.yaml
# or: ZEROLLAMA_DEVICE_COUNT=1 ZEROLLAMA_TENSOR_PARALLEL=1
```

---

## Start the daemon

```bash
export LLAMA_MODEL=/absolute/path/to/model.gguf
export LLAMA_SERVER_BIN=/absolute/path/to/llama-server
export OLLAMA_TRAINING=false          # unless torch deps installed
export OLLAMA_HOST=http://127.0.0.1:8080   # if not default 11434

./zerollama serve
```

**Why `LLAMA_MODEL` / `LLAMA_SERVER_BIN` on the serve process:** the embedded runtime reads env at **startup**, not from the shell that runs the smoke script later.

---

## Scripted smoke

```bash
cd ~/zerollama

# Health only (no GPU generate)
./scripts/e2e_runtime_smoke.sh

# Direct runtime (:8081)
export LLAMA_MODEL LLAMA_SERVER_BIN
RUN_E2E_GPU=1 ./scripts/e2e_runtime_smoke.sh

# When LLAMA_MODEL is too large for free VRAM (VRAM pre-check 502), pass a smaller GGUF:
export RUN_E2E_GGUF=/path/to/small.q8_0.gguf
RUN_E2E_GPU=1 ./scripts/e2e_runtime_smoke.sh

# VRAM estimate only (no generate) — loopback :8081
./scripts/runtime_vram_estimate.sh /path/to/model.gguf --num-ctx 8192

# Via Go proxy (:8080) — needs zerollama serve running
export OLLAMA_HOST=http://127.0.0.1:8080
RUN_E2E_PROXY=1 ./scripts/e2e_runtime_smoke.sh

# Proxy + small GGUF (uses model smoke + X-Zerollama-Runtime, or override on pulled name via options):
export RUN_E2E_GGUF=/path/to/small.q8_0.gguf
RUN_E2E_PROXY=1 ./scripts/e2e_runtime_smoke.sh
# Or a pulled tag (manifest gguf; RUN_E2E_GGUF overrides options.gguf when set):
# RUN_E2E_PROXY_MODEL=llama3.2:3B RUN_E2E_GGUF=... RUN_E2E_PROXY=1 ./scripts/e2e_runtime_smoke.sh

# Optional: verify Phase 13 clamp is enabled on the running daemon:
# ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX=auto RUN_E2E_GPU=1 RUN_E2E_VRAM_CLAMP=1 ./scripts/e2e_runtime_smoke.sh
```

**Proxy without `X-Zerollama-Runtime`:** use a **pulled** local model that is runtime-default eligible (Phase 12) or has `modality_backends.inference: zerollama-runtime`; Go sends `options.gguf` from the manifest (Phase 9). The header is for ad-hoc names (e.g. `smoke`) or **`RUN_E2E_PHASE14=1`** (forces runtime for Phase 14 sign-off). **`RUN_E2E_GGUF`** sets `options.gguf` on the proxied request (same as direct `:8081` smoke). **`OLLAMA_RUNTIME_ALL=1`** proxies every local tag without the header.

**Why the script still calls `/internal/inference/resume`:** belt-and-suspenders if an older daemon is running; with Phase 8 the Go broker resumes runtime inference before proxying and evicts runners before legacy loads.

**Training ops (no job submit):** with `OLLAMA_TRAINING=true` and `zerollama serve` running:

```bash
OLLAMA_HOST=http://127.0.0.1:8080 ./scripts/e2e_training_ops_smoke.sh
# Optional legacy TCP listener (default :9500):
RUN_E2E_TRAINING_TCP=1 OLLAMA_TRAINING_TCP=:9500 ./scripts/e2e_training_ops_smoke.sh
```

Repro ports (`19180` / `:19650`): see `scripts/repro_shared_interpreter_health_hang.sh`.

**Go↔runtime coordination mirror** (embedded or sidecar runtime):

```bash
ZEROLLAMA_RUNTIME_URL=http://127.0.0.1:8081 ./scripts/e2e_coordination_smoke.sh
```

---

## GPU sharing (runtime ↔ legacy)

Both stacks use the same GPU. **Phase 8 (Go `server/vram`):** before a ggml runner load, the daemon calls `training-handoff` on the embedded runtime; before runtime proxy requests, it unloads all ggml runners and calls `inference/resume`. Training job submit does both proactively. **No public unload API** — smokes use the same contract as operators (`keep_alive:0`), not `pkill`.

**Why API unload first (`smoke_unload_ggml_runners`):** A **503** from Go (training busy, admission) can occur **before** the broker runs, leaving a stale ggml runner loaded. Unload via `/api/ps` + empty `generate` matches production and avoids false `pgrep` matches on shell lines. See [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md).

**Smoke VRAM prep (`smoke_prepare_vram_for_runtime`):** optional API unload, then one `X-Zerollama-Runtime` proxy generate triggers the broker. On **503** with a runner still loaded: retry unload, resume, broker once. `gpu_smoke_all` runs legacy **after** runtime via `RUN_E2E_LEGACY_ONLY=1` — do not pass `RUN_E2E_LEGACY=1` together with `RUN_E2E_GPU=1` on a single `e2e_runtime_smoke.sh` (e2e errors with a hint).

Manual hooks (debugging only):

```bash
curl -sS -X POST http://127.0.0.1:8081/internal/training-handoff
curl -sS -X POST http://127.0.0.1:8081/internal/inference/resume
```

**Unload one legacy model:**

```bash
curl -sS "$OLLAMA_HOST/api/generate" -H 'Content-Type: application/json' \
  -d '{"model":"llama3:latest","prompt":"","keep_alive":0}'
```

GPU smokes call `smoke_unload_ggml_runners` (reads `/api/ps`, or `RUN_E2E_UNLOAD_MODEL`) before the runtime broker when `smoke_ggml_runner_running` sees a ggml child (`/zerollama runner --`). Waits up to `SMOKE_UNLOAD_MAX_WAIT` seconds (default 30). On Go **503** with a runner still loaded, `smoke_prepare_vram_for_runtime` retries unload then the broker once.

---

## Troubleshooting

| Symptom | Likely cause | What to do |
|---------|----------------|------------|
| `502` GPU memory on `RUN_E2E_GPU=1` | VRAM pre-check: `LLAMA_MODEL` too large for free VRAM | Use a smaller quant, or `RUN_E2E_GGUF=/path/to/small.gguf` in the smoke script |
| `422` / `Field required` on `query.req` | Old FastAPI app without `Body()` fix | Restart `zerollama`; health should show `server_revision: fastapi-body-v3` |
| `503 inference paused for training` | Stale runtime state / Go training gate | `POST :8081/internal/inference/resume`; smokes call this automatically; proxy may 503 before broker evicts ggml |
| `502` could not admit (KV pool) | Stale handoff, ggml holding VRAM, or huge `num_ctx` | Resume + `smoke_prepare_vram`; use `RUN_E2E_GGUF` small quant; `RUN_E2E_NUM_CTX=4096` |
| `502` host memory on large pull | Runtime `check_gguf_host_budget` (e.g. gpt-oss:20b MXFP4 ~44 GiB mmap) | Use smaller quant/GGUF, more host RAM, or legacy ggml path; `gpu_harmony_capture` needs RAM not just VRAM |
| `404 model 'smoke' not found` on proxy | Legacy handler, no runtime proxy | `X-Zerollama-Runtime: 1`, `OLLAMA_RUNTIME_ALL=1`, or runtime-default model |
| `truncate_mode=heuristic` on Phase 14 render-chat | Stale serve or embed without runtime URL | Rebuild `zerollama`; use `phase14_serve_env.sh` (embed); needs current Go + `/internal/tokenize` |
| `/health missing llama_backend_source` | Stale serve binary | Rebuild `zerollama` from current tree; restart serve |
| `RUN_E2E_LLAMA_BACKEND_SOURCE=config` fails | Env still set on serve | Unset `ZEROLLAMA_RUNTIME_LLAMA_BACKEND`; uncomment `llama_backend` in YAML; restart serve |
| `llama_backend_source=default` on subprocess serve | Expected when autoconfig YAML has no `llama_backend` key | Set env or uncomment YAML key; use `RUN_E2E_LLAMA_BACKEND_SOURCE=default` only to assert packaged default |
| `address already in use` on `:8081` / embed warns then stale `/health` | Previous `zerollama serve` or `zerollama-runtime` sidecar still listening | `ss -tlnp \| grep 8081`; `pkill -f 'zerollama serve'`; unset `ZEROLLAMA_RUNTIME_URL`; use `scripts/serve_gpu_example.sh` or `phase14_serve_env.sh` |
| `embedded runtime not started` (port in use) | Go preflight blocked embed (fixed vs silent stale attach) | Free `:8081` before `./serve.sh`; only one embed listener per host |
| Remote client cannot reach API | `OLLAMA_HOST` default `127.0.0.1:11434` (localhost only) | Set `OLLAMA_HOST=0.0.0.0:8080`; verify `ss -tlnp \| grep 8080`; clients use Go `:8080`, not embedded `:8081` |
| `go build` fails `cpp-httplib/httplib.h` | httplib not vendored in minimal checkout | `rsync -a ~/llama.cpp/vendor/cpp-httplib/ llama/llama.cpp/vendor/cpp-httplib/` — [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md#building-zerollama-cgo-on-proxmox-ct) |
| Screen shows no serve output | `exec zerollama serve >> /tmp/zerollama-serve.log` | `tail -f /tmp/zerollama-serve.log` — **why:** log redirect keeps screen quiet |
| L1 bench "runtime failed to start" | `/health` cold probe ~9s; curl `-m 2` too short | Fixed in `linux_runtime_serve_lib.sh` (`LINUX_RT_CURL_TIMEOUT=15`); kill orphan `llama-server` on `:8082` |
| ggml `runner terminated` during Phase 14 proxy | Pulled tag hit legacy path | `RUN_E2E_PHASE14=1` adds header; or `OLLAMA_RUNTIME_ALL=1` on serve |
| `llama-server exited` / CUDA OOM | Model too large, wrong arch, tensor split on 1 GPU | Quantized GGUF, `single_gpu.yaml`, rebuild `120-real` |
| Legacy `runner terminated` | GPU held by runtime | Retry (broker should hand off); or manual `training-handoff` |
| Empty JSON from `curl -sf` | HTTP error hidden | Use smoke script or `curl -w '%{http_code}'` |

---

## Video agent + padded multimodal (operator)

**Why three smokes:** `video_expand_cache_smoke` proves unit caches without GPU; `video_agent_cache_smoke` proves session expansion with `_debug_render_only` (no real VLM); `video_agent_infer_smoke` proves turn-2 prefix hits on **real** vision prefill — the layer agents actually care about for latency.

**Unit (no GPU):**

```bash
./scripts/video_expand_cache_smoke.sh
go test ./server/modality/... -run 'PaddedLayoutConsume|Deepseek|qwen25vl'
```

**Expand-only live** (`RUN_E2E_VIDEO_AGENT=1`): session VIDEO cache + preprocessed layout restore — needs `VIDEO_SMOKE_MODEL`, ffmpeg, running serve, optional `VIDEO_AGENT_GO_LOG`.

**Full infer live** (`RUN_E2E_VIDEO_AGENT_INFER=1`): real VLM prefill + turn-2 `cached_prompt_tokens`. Mac Metal ollama-engine: input-cache hits count even when `llama_cache.enabled=false`. Set `VIDEO_AGENT_INFER_SOFT=1` only when KV cache is off or model is MLX-only.

```bash
RUN_E2E_VIDEO_AGENT_INFER=1 \
  VIDEO_SMOKE_MODEL=qwen3-vl:latest \
  VIDEO_AGENT_GO_LOG=/tmp/zerollama-go.log \
  ./scripts/video_agent_infer_smoke.sh
./scripts/video_agent_infer_gate_report.sh /tmp/video-agent-infer-smoke.json
```

**Preprocessed padded leg** (`VIDEO_AGENT_INFER_PREPROC=1`, requires `VIDEO_AGENT_GO_LOG`): turn-1 sends `padded_input_ids` + `images` + `video_spans` + `grid_thw`; turn-2 resends frames only — strict layout-cache grep + turn-2 `cached_prompt_tokens` (use `VIDEO_AGENT_INFER_SOFT=1` when cache off):

| Log line | Meaning |
|----------|---------|
| `preprocessed layout session cache hit` | Session layout LRU restored `padded_input_ids` |
| `padded_input_ids runner inject` | Runner consumed pretokenized ids (ollama-engine / llamarunner) |
| `vision embed session cache hit` | Per-agent ViT overlay on turn 2+ |
| `vision grid hints` | Client `grid_thw` vs embed-count compare (Info) |

**Ollama-engine padded families** (native Go VLM inject — grep `layout_consume=` on access log):

| Consume mode | Families |
|--------------|----------|
| `qwen3vl_hf_runner_inject` | Qwen3-VL, qwen25vl, qwen2vl |
| `gemma4_img_runner_inject` | Gemma4 |
| `mllama_img_runner_inject` | mllama |
| `gemma3_img_runner_inject` | Gemma3 |
| `llama4_img_runner_inject` | Llama4 |
| `lfm2_img_runner_inject` | LFM2 |
| `glmocr_img_runner_inject` | GLM-OCR |
| `mistral3_img_runner_inject` | Mistral3 |
| `deepseekocr_img_runner_inject` | DeepSeek-OCR |

Still `deferred_non_qwen3vl`: text-only archs (gemma3n, glm4moelite). Doc: [sglang-multimodal-borrowings.md](./sglang-multimodal-borrowings.md), [phase17-llama-server.md](./phase17-llama-server.md#padded-multimodal-inject-llama-server).

---

## Related docs

- [runtime-embed.md](./runtime-embed.md) — single-process embed
- [runtime/docs/OPERATIONS.md](../runtime/docs/OPERATIONS.md) — day-two ops
- [python-migration.md](./python-migration.md) — phased migration
- [gpu-training.md](./gpu-training.md) — training vs inference VRAM policy
