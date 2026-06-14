# Roadmap

This file tracks **directional** plans. It is not a commitment schedule.

**Why this file exists:** Large features (video, remote cloud, GPU training) touch **API compatibility**, **security**, and **optional subprocesses / upstreams**. A short roadmap keeps **intent** and **non-goals** visible so contributors do not assume every deployment wants the same tradeoffs.

---

## Product model: queues, stakeholders, and GPU time

Directional (not a shipped scheduler contract): **GPU-backed batching system** fed by many **stakeholders** at once: local CLI and library pulls, OpenAI-compatible HTTP, agents and integrations, optional **Eliza cloud** merge, and (when enabled) the embedded **Python runtime** path. Those surfaces funnel into **inference work** that must be **admitted, queued, and executed** as efficiently as the hardware allows—today split between the **Go scheduler** (ggml runners, eviction, public routing) and the **Python runtime scheduler** (PagedAttention bookkeeping, `llama-server` orchestration).

**Training** is intentionally a **separate job queue** (`/api/train/*`, embedded `training.py`, optional TCP `:9500`): jobs are submitted, listed, and cancelled independently of a single chat FIFO. **VRAM** is shared: Go scaffolding (Phase 8) and Python coordination ensure inference and training do not corrupt each other’s memory; **Phase 11** moves more of that **policy** into Python.

**What is not automatic yet:** a single product-level **orchestrator on one machine** that says “maximize inference throughput until the backlog is idle, then drain training jobs” or “night window only.” That is **scheduling policy on top** of the existing queues—priority classes, SLOs, and optional idle-time training—aligned with the training track below and **Phase 11+**.

**What is also not automatic yet:** a **fleet management node** when many zerollama instances serve one agent population. Each node keeps its local scheduler; a separate layer handles **discovery** (directional: mDNS), **warm-model routing**, and **short assignment tokens**—not long reservations or scatter-gather across GPUs. See [fleet-scheduling.md](./fleet-scheduling.md) and the [Fleet scheduling track](#fleet-scheduling-multi-node) below.

Documenting both shapes here avoids mistaking today’s **two schedulers + one VRAM broker per node** for a finished global or multi-node optimizer.

---

## Technology ladder (north star)

Directional stack—not a promise to delete Go or Python on a fixed date.

```text
Layer 0 (today)     Go daemon — registry, HTTP, Eliza proxy, embed lifecycle, ggml scheduler
        ↓ shrink    Why: shipped fast; still owns processes Python cannot see (runners)
Layer 1 (now→)      Python runtime — PA bookkeeping, batching policy, llama-server orchestration, training
        ↓ shrink    Why: best place for scheduler experiments before freezing native APIs
Layer 2 (later)     C / Rust hot path — GGUF decode, KV blocks, in-process llama.cpp (or forked kernels)
        ↓           Why: throughput and memory layout; no GIL/subprocess HTTP on the critical path
                    Candidate fork: elizaOS/llama.cpp (TurboQuant, QJL, Polar) — see borrowings **L2**
Edge (long-lived)   Thin API + pull + cloud — may stay Go *or* move to Rust; “Go gone” means inference
                    control plane gone from Go, not necessarily zero Go in the repo
```

**Upstream Ollama note:** Vanilla [ollama/ollama](https://github.com/ollama/ollama) already routes **default GGUF** as **Go → llama-server** (no Python sidecar). That is **Layer 2 at the llama-server boundary**, not our Python runtime. Zerollama should **converge default GGUF chat** toward that shape ([Phase 17](#phase-17--upstream-gguf-path-alignment-directional)) while keeping Python for admission, training, and Phase 15 experiments. See [upstream-ollama-diff.md](./upstream-ollama-diff.md).

**VRAM policy** follows the same ladder: **Go broker (scaffolding)** → **Python `InferenceGpuCoordinator`** → **native allocator** when Python shrinks.

**Detailed migration log (Phases 0–7, done):** [python-migration.md](./python-migration.md). **Operator smoke:** [testing-smoke.md](./testing-smoke.md).

**Operator guide (WHY-oriented, phases 8–13 + T6):** [scheduling-vram-policy.md](./scheduling-vram-policy.md) — VRAM broker, training defer queue, runtime pre-checks, env tables, code map.

---

## Local inference — actionable phases

Phases **0–7** are **done** (sidecar, embed, Go proxy, spec decode plugins). Work below is **8+**.

| Phase | Goal | Owner | Exit criteria |
|-------|------|--------|----------------|
| **8** | **Automatic VRAM handoff (scaffolding)** | Go | **Done** — `server/vram`: legacy load → `training-handoff`; runtime proxy → `UnloadAllRunners` + `inference/resume`; training submit → both. No public unload API. |
| **9** | **Manifest → runtime** | Go + Python | **Done** — Go proxy sets `options.gguf` from manifest and forwards client `options` (e.g. `num_ctx`); runtime loads/swap per request. `LLAMA_MODEL` remains optional fallback for direct `:8081` / smoke. |
| **10** | **CI regression gate** | Repo | **Done** — `.github/workflows/zerollama-regression.yaml`: `go test` (incl. Golden) + runtime pytest (incl. tools meta) + `check_gpu_scripts.sh`. Local/GPU preflight: `./scripts/phase12_golden_ci.sh`. Optional: `zerollama-gpu-smoke.yaml` (`workflow_dispatch`, self-hosted). |
| **11** | **VRAM + admission policy in Python** | Python | **Partial** — inference-first + VRAM checks; **low** throttling; min-free + training reserve via env or `single_gpu.yaml` `vram:` block. Backpressure thresholds overridable (`ZEROLLAMA_RUNTIME_BACKLOG_BATCH_MIN`, …). **5080:** `gpu_5080_session.sh` PASS with `RUN_E2E_PREFLIGHT=0` on CT 1564 (Go golden needs full vendored tree); defaults unchanged (gates active, admission fits). |
| **12** | **Runtime default for text local models** | Go + Python | **Done** (tools path) — default-on; streaming proxies; tools via Go render + stateful `parse-tool-output` sessions. Render ctx aligned with load via `resolve_num_ctx_for_request`. v1 proxy injects manifest `options.gguf`. CI goldens: `./scripts/phase12_golden_ci.sh`. **Harmony real-weight:** CI synthetic only; `gpt-oss:20b` needs ~40+ GiB host RAM on runtime path (not required on 5080). |
| **13** | **Single-GPU + host autoconfig** | Python | **Partial** — estimates, autotune catalog + `estimate_factor_source`, `suggested_max_num_ctx`, clamp default **off** in YAML; `python -m runtime.gpu_snapshot` after session JSON; `vram:` defaults in `single_gpu.yaml`. **5080 gate:** [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md). Doc: [phase13-runtime-vram.md](./phase13-runtime-vram.md). **L1 profiles:** [gpu-profiles-l1.md](./gpu-profiles-l1.md) (Metal tiers shipped; CUDA tune pending). |
| **14** | **In-process llama forward** | Python → C/Rust | **Done** — see [exit criteria](#phase-14--exit-criteria-done). Shipped: ctypes `inprocess`, wheel (CPU default), tokenize, sampling, YAML `llama_backend`, `llama_backend_source`, `llama_cpp` `/health`, heap-batch decode fix. Smokes: `phase14_inprocess_smoke`, `phase14_5080_signoff`, optional `phase14_wheel_gpu_smoke` (failed on 5080). Doc: [phase14-inprocess-llama.md](./phase14-inprocess-llama.md). |
| **15** | **Native scheduler + KV** | C/Rust | **Partial (v0–v30 ops)** — see [exit criteria](#phase-15--exit-criteria-partial). C pool + tick/decode hooks; **v9–v16** decode plan export, libllama link, C decode loop (GIL + sampling in C), engine resume via `current_pos`; **v24–v30** page-bind validation, auto-link build, 131k ctx bind cap, continuous batch decode + engine wiring + `/health` batch plan + streaming batch decode + per-row C batch sampling. Go KV snapshot; GPU `phase15_inprocess_signoff`. **Blocked:** tensor page bind (hybrid/iSWA). Docs: [phase15-native-kv.md](./phase15-native-kv.md), [handoff-phase15-native-kv.md](./handoff-phase15-native-kv.md). |
| **16** | **Thin edge daemon** | Rust or minimal Go | Pull/registry/cloud only; all local generate/chat through native runtime. **Why:** complete “Go gone” for inference control plane. |
| **17** | **Upstream GGUF path alignment** | Go + llama.cpp | **Directional** — port `llm/llama_server.go`, adopt `llama/compat/`, bump `LLAMA_CPP_VERSION` to upstream pin; deprecate ggml runner for plain text GGUF. Python runtime stays for PA/training/admission—not permanent chat middleman. Doc: [upstream-ollama-diff.md](./upstream-ollama-diff.md). Test harness today: [llama-cpp-backend.md](./llama-cpp-backend.md). |

**Deprioritized:** public `POST /api/runtime/unload` or `/resume` — automatic eviction only ([Phase 8](#local-inference--actionable-phases) → [Phase 11](#local-inference--actionable-phases)).

**Non-goals (this ladder):** RadixAttention v1; required vLLM/SGLang servers; rewriting training in C++ (`llama-finetune` WIP); bit-for-bit SGLang parity.

### Phase 14 — exit criteria (Done)

| # | Criterion | Owner |
|---|-----------|--------|
| 1 | Three backends, sampling, `/internal/tokenize`, Go render `truncate_mode=tokenize` | **Done** (code) |
| 2 | YAML `llama_backend` + `/health` `llama_backend_source` + provenance smokes | **Done** (code) |
| 3 | **5080 ctypes GPU:** `phase14_inprocess_smoke.sh` (or `RUN_E2E_INPROCESS=1 phase14_backend_smoke`) | **Done** (5080 dev host) |
| 4 | **Wheel CPU:** `phase14_wheel_cpu_smoke.sh` (or `phase14_both_backends.sh`) | **Done** (5080 dev host) |
| 5 | **Wheel GPU (optional):** `phase14_wheel_gpu_smoke.sh` after `ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS` on serve | **Failed** on 5080 dev host (`free(): invalid pointer`); use inprocess for GPU |
| 6 | **Packaged default (optional):** `phase14_enable_yaml_inprocess.sh` or `phase14_yaml_config_full_smoke.sh` | **Done** (5080 dev host, temp YAML) |

Mark **Done** when 1–2 and **3–4** pass on ship hardware. **5** failed on 5080 dev host (wheel GPU); **6** optional YAML packaged default — use `phase14_yaml_config_full_smoke.sh` without editing repo YAML. Subprocess stays the repo default until operators opt in. **One-shot:** `./scripts/phase14_5080_signoff.sh`. See [phase14-inprocess-llama.md](./phase14-inprocess-llama.md).

### Phase 15 — exit criteria (Partial)

| # | Criterion | Owner |
|---|-----------|--------|
| 1 | C block pool + Python facade; `phase15_kv_native_ci.sh` | **Done** (code) |
| 2 | Logical bind, forward plans, `/internal/kv-snapshot`, Go loopback proxy | **Done** (code) |
| 3 | In-process decode hook (`kv_decode_steps`) + multi-seq (`llama_parallel_slots`>1) | **Done** (code) |
| 4 | **5080 GPU:** `phase15_inprocess_signoff.sh` (KV hook + multi-seq + batch decode + `kv_page_bind` snapshot) | **Done** (RTX 5080 CT 1564, Jun 2026) |
| 5 | **Tensor page bind** — PA `block_ids` → llama KV tensor pages | **Partial (Jun 2026)** — **v8:** seq-position bind; **v19:** accounting probe; **v20:** cell index + K/V tensor verify via fork `llama-kv-ext.h` when linked (`status=bound` on standard kv_cache); **v31:** hybrid/iSWA resolve to attn base cache; patch **0015** + pin check. **Why partial:** writable cross-allocator bind still needs upstream-stable page-handle API; pure recurrent unsupported |
| 6 | **Native decode batch** in C wired to `kv_forward_plans` | **Partial (Jun 2026)** — C batch layout + page-aligned chunks; **v9–v11:** plan export; **v12:** libllama link; **v13:** `llama_decode` in C; **v14–v16:** GIL release, sampling in C, `_decode_stream` + engine resume via `current_pos`; **v16b–v18:** resume owner + `/health.kv_resume`; **v19–v20:** tensor bind scaffold + `llama-kv-ext` cell/tensor bind; **v20a:** `native_page_table` on forward plans; **v26–v30:** `kv_decode_loop_run_batch_step`, engine `generate_batch` / `stream_generate_batch`, per-row `smpl_ptrs[]`, `/internal/generate-batch`, **Metal batch sign-off PASS (M4 Max Jun 2026)** |

Mark **Done** when 1–3 and **4** pass on ship hardware. **5–6** partial until upstream llama.cpp ships stable writable KV page API for all memory types. CPU gate: `./scripts/phase15_kv_native_ci.sh`. GPU gate: `./scripts/phase15_inprocess_signoff.sh` (Linux embed) + `./scripts/phase15_metal_signoff.sh` (Mac sidecar, includes batch decode step 3/5). Linked tensor bind + batch decode: rebuild libllama from fork + `ZEROLLAMA_KV_DECODE_LOOP=1`; sign-off scripts source `phase15_runtime_kv_env.sh`. See [phase15-native-kv.md](./phase15-native-kv.md).

### Phase 17 — upstream GGUF path alignment (directional)

**Why:** Upstream Ollama removed `runner/ollamarunner` for text GGUF and integrated **`llama-server` from Go** (`llm/llama_server.go`). Zerollama still defaults to **ggml runner** on Mac and uses **Python runtime** as the bridge for `--llama-cpp-backend`. Aligning default GGUF with upstream reduces merge pain, modernizes the llama.cpp pin, and removes an extra hop (Go → Python → llama) from the hot path—without dropping training or Phase 15 work.

| # | Criterion | Owner |
|---|-----------|--------|
| 1 | Document deltas vs upstream checkout | **Done** — [upstream-ollama-diff.md](./upstream-ollama-diff.md) |
| 2 | Bump sibling llama.cpp + pin toward upstream `b9509`; rebuild `llama-server` | **Done** — root `LLAMA_CPP_VERSION`, [runtime/LLAMA_CPP_PIN.md](../runtime/LLAMA_CPP_PIN.md) |
| 2b | **Rebase in-tree ggml/llama.cpp to real b9509** (not overlay snapshot) | **Done** — 12 patches, vendor sync script, build+doctor; [ggml-b9509-migration.md](./ggml-b9509-migration.md) |
| 3 | Port `llama/compat/` overlay; reduce overlapping `llama/patches/` | **Partial** — compat imported; ggml patches rebased to b9509; dedup open |
| 4 | Port `llm/llama_server.go` + discovery probe; eligible GGUF uses Go → llama-server | **Scaffold** — `--llama-server-backend`; see [phase17-llama-server.md](./phase17-llama-server.md) |
| 5 | Benchmark ggml vs Go-llama-server vs Python runtime on ship hardware | **Done (M7)** — ggml ~164 vs upstream ~158 tok/s @ 4k ctx; keep ggml Mac default |
| 6 | Deprecate `OLLAMA_NEW_ENGINE` / ollamarunner for plain text GGUF (keep vision/thinking until parity) | Open |
| 7 | Coordinate llama.cpp pin with borrowings **L2** (eliza fork kernels vs upstream merge) | **Partial** — L2 spike infra shipped; vendor merge gated on bench |

**Non-goals:** full rebase onto upstream; deleting `runtime/` or training; replacing Eliza with ollama.com.

**Compare workflow:** `./scripts/clone_upstream_ollama.sh` → build upstream on `:11435`, zerollama on `:11434`. See [upstream-ollama-diff.md](./upstream-ollama-diff.md#compare--benchmark-workflow).

### Phase 8 — shipped

See `server/vram/broker.go` and `server/runtime_manifest.go`. Phase 14 in-process forward is **Done**; next: **Phase 11** admission tuning and **Phase 15** native KV tensor bind.

---

## Zerollama remote cloud (Eliza)

**Shipped direction:** Default upstream [Eliza Cloud](https://www.elizacloud.ai); API key via `ELIZACLOUD_API_KEY`; path rewrite to Eliza `/api/v1/...`; catalog merge with singleflight + cache TTL; `:cloud` model suffix for routing. See [eliza-cloud.md](./eliza-cloud.md) for full rationale.

### Possible follow-ups

- **Stricter response mapping:** Optional adapters from raw Eliza JSON to OpenAI/Ollama-shaped responses where schemas stabilize, without losing fields today’s clients rely on.
- **Catalog hardening:** Explicit policy when local and remote names collide beyond simple dedupe (e.g. precedence rules in UI).
- **Multipart / image routes:** Clearer behavior when `model` is not in JSON (document or extend passthrough detection).

### Non-goals (for this track)

- **Replacing** Eliza with another host transparently without configuration — different providers have different auth and routes; `OLLAMA_CLOUD_BASE_URL` exists precisely to make that an explicit operator choice.
- **Guaranteeing** bit-for-bit parity with every Eliza API revision — we proxy and merge where Zerollama needs; upstream drift is handled case by case.

**Why a separate subsection from video:** Remote HTTP inference and local ffmpeg/video pipelines share almost no code paths; mixing them in one bullet list would blur ownership and testing strategy.

---

## Python inference runtime (GGUF + PagedAttention)

**Status:** Phases **0–7** complete — see [Local inference — actionable phases](#local-inference--actionable-phases) for **8+**.

**Direction:** GGUF-first **`runtime/`** with PagedAttention block pools; not HTTP to vLLM/SGLang. **Embed:** [runtime-embed.md](./runtime-embed.md). **Spec decode:** [runtime/docs/SPECULATIVE.md](../runtime/docs/SPECULATIVE.md). **Smoke validated:** [testing-smoke.md](./testing-smoke.md) on RTX 5080-class hosts. **Operator ladder (WHY):** [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md) — one session script proves Phase 10–13 on 16 GB; harmony real-weight is **not** required (CI Go golden + host-RAM limits for `gpt-oss:20b`).

**Still on ggml runner:** vision, logprobs, think, MLX safetensors, and models without `zerollama-runtime` backend. **Tools** on runtime-routed text models use Go render/parse (Phase 12 — done). Handoff: [handoff-phase12-runtime-tools.md](./handoff-phase12-runtime-tools.md).

**Upstream tension:** Vanilla Ollama has **no** Python runtime; default GGUF is **Go → llama-server** ([upstream-ollama-diff.md](./upstream-ollama-diff.md)). Zerollama’s Python layer stays for PA, admission, training, and Phase 15—but **Phase 17** targets upstream-style Go integration for default text GGUF. **`--llama-cpp-backend`** is the current test harness (Go → Python → llama), not the long-term default shape.

---

## Apple Silicon & Metal track

**Why a separate track:** CUDA Phase 11/13 work (`single_gpu.yaml`, `nvidia-smi`, `gpu_5080_session.sh`) does not apply to unified memory. Mac users need **Metal ggml** (default), **runtime admission** that probes `vm_stat`, and clear **MLX vs GGUF** routing—not a copy of the 5080 playbook.

**Guide:** [apple-silicon-metal.md](./apple-silicon-metal.md)

| Milestone | Goal | Owner | Exit criteria |
|-----------|------|--------|----------------|
| **M1** | **Unified memory admission** | Python | **Shipped (audit)** — `metal-unified` probe; `read_host_memory()` on darwin (load + `/health`); `apple_silicon.yaml` autoconfig; `check_gguf_host_budget` no longer Linux-only; `vm.swapusage` parser fixed. |
| **M2** | **Operator smoke + docs** | Repo | **Shipped** — `macos_metal_smoke.sh`; guide + ROADMAP; pytest for darwin probe + snapshot hints; `check_gpu_scripts` greps. |
| **M3** | **Runtime Metal parity** | Python | **Shipped** — `m3_metal_signoff.sh` / `gpu_metal_session.sh`; Phase 13 snapshot + Phase 14 inprocess on Metal (M4 Max sign-off, Jun 2026). `apple_silicon.yaml` sets **`llama_backend: inprocess`**; M3 validates `llama_backend_source=config`. Use a **text-only** GGUF with pinned llama.cpp (not vision gemma3 on old pin). Mac daily serve: **`zerollama serve`** (auto sidecar `:8081`); `./scripts/serve_mac_runtime.sh` for CI (prints log paths — see [fleet-management.md](./fleet-management.md#macos-runtime-stack-related)). Optional Phase 15: `RUN_E2E_PHASE15=1 ./scripts/m3_metal_signoff.sh` or `./scripts/phase15_metal_signoff.sh`. |
| **M4** | **MLX policy** | Go + docs | **Shipped** — [mlx-routing-policy.md](./mlx-routing-policy.md); `IsMLX()` excluded from runtime default **and** explicit Modelfile backend; Go tests. **Dylibs:** rebuild at `MLX_VERSION` / `MLX_C_VERSION` via `build_production_mac.sh` (Jun 2026 sign-off @ `2165dc08` / `fba4470b`). |
| **M5** | **Phase 15 Metal KV sign-off** | Python | **Shipped (Jun 2026, M4 Max PASS)** — `phase15_metal_signoff.sh` (5 steps: KV hook, multiseq, **continuous batch decode** via `phase15_batch_decode_smoke.sh`, L3 two-turn, tensor bind); `metal_signoff.sh` optional `RUN_E2E_PHASE15=1`. **Why batch step:** v27–v30 engine batch path must run on real Metal multiseq sidecar, not CPU pytest mocks. |
| **M6** | **MPS LoRA training + Mac operator polish** | Python + Go + CI | **Shipped** — PyTorch MPS + PEFT in `training.py`; QLoRA rejected on Darwin; `training_uv_venv.sh`; **`zerollama serve` Darwin bootstrap** (uv venvs, sidecar `:8081`, autoconfig); `zerollama doctor --json --fix`; Darwin CI (`macos-darwin-smoke`). **Extended (M14):** tiered clone bootstrap — see **M14**. |
| **M7** | **Upstream-shape GGUF benchmark (Metal)** | Repo | **Done** — ggml Metal ~164 tok/s vs upstream Go→llama-server ~158 tok/s (`llama3.2:3b`, `num_ctx=4096`, 6 epochs, idle GPU). Keep ggml default; Phase 17 for mergeability. [phase17-llama-server.md](./phase17-llama-server.md) |
| **M8** | **ggml @ b9611 (real vendored tree)** | Repo | **Done** — 14 patches on `vendor/llama-cpp-b9611/` (0011 GPU discovery rebased, 0012 no-alloc, 0013 mtmd C API, 0014 ollama_vocab); `sync_vendor_llama.sh`; Mac `build_zerollama_mac.sh` + `doctor`. Vanilla Ollama still on **b9509**. [ggml-b9509-migration.md](./ggml-b9509-migration.md) |
| **M9** | **Metal operator sign-off (Jun 2026)** | Repo | **Done (M4 Max)** — `./scripts/metal_signoff.sh` (Phase 13–15 + optional qwen35). **Jun 2026 full gate PASS** with `RUN_E2E_QWEN35=1 RUN_E2E_QWEN35_MODEL=qwen3.6:latest`. **Gaps fixed:** v1 SSE + proxy flush; darwin ggml policy; `num_gpu=0` Metal gate; bootstrap discovery; sched_reserve; sign-off order (qwen35 before Phase 15); Phase 15 multiseq + `ZEROLLAMA_GPU_PROFILE=0`; L3 `cache_prompt` on inprocess workers. |
| **M10** | **Qwen 3.5/3.6 GGUF on Mac** | Go + llama/compat + ggml Metal | **Done (Jun 2026, M4 Max)** — Go **ollama-engine** on Metal for `qwen35moe` after `sched_reserve` fix (graph intermediates defer to scheduler; KV buffer contexts use `Persistent()`); darwin no longer forces legacy llamarunner for `qwen35*`; in-process compat CGO link for llama-server/legacy; Metal embed regen in `build_zerollama_mac.sh`; `PrimaryFamily()` for VL manifests (projector-only → `""`); qwen35 `flushDoneEvents`; LM Studio MLX disk checks; opt-in `./scripts/qwen35_mac_smoke.sh` (thinking models: accept `thinking` when `response` empty); **unified Mac build** — `build_zerollama_mac.sh` + `build_mlx_dylibs_mac.sh` with `BUILD_MLX=auto`. **Sign-off:** full `metal_signoff.sh` + qwen35; `qwen3.6:latest` 41/41 GPU layers via ollama-engine. **Why M10 exists:** published GGUF metadata differs from llama.cpp-native; stale `ggml-metal-embed.metal` broke first decode (sigmoid/unary); runtime/ggml dual-Metal contention on Darwin. **Not done:** full 27B Q8 VL on unified memory; qwen35 in default CI (opt-in smoke only). Doc: [qwen35-apple-silicon.md](./qwen35-apple-silicon.md). |
| **M11** | **GPU bootstrap discovery on Mac** | Go | **Done (Jun 2026)** — `DiscoverBackendDevices()` + ollama-engine `/info` no longer uses zero-layer dummy load (which set `GGML_DISABLE_METAL` via `sync.Once`). **Why:** operators saw `total_vram=0` and CPU-only layer layout while inference subprocesses still logged Metal; scheduler trusted empty discovery. Doc: [apple-silicon-metal.md](./apple-silicon-metal.md#gpu-bootstrap-discovery-jun-2026). |
| **M12** | **Scheduler unload + manifest `num_ctx` clarity** | Go + docs | **Done (Jun 2026)** — `expireRunner` always queues unload + `findLoadedRunner` name fallback; `/api/create` evicts warm runners; API surfaces prompt truncation; docs for **load-time KV vs request `options.num_ctx`**; **ggml VRAM suggest + opt-in clamp** (`ZEROLLAMA_GGML_CLAMP_NUM_CTX`, `/api/show` `ggml_num_ctx`, load responses when clamped). **Why:** create updated manifest but `/api/ps` stayed at old ctx; stop returned success while model stayed loaded; manifest `num_ctx: 262144` hung ggml load; high-VRAM tier default (262144) needs operator guidance without silent clamp; total VRAM must not stand in for free VRAM in suggest. **Audit:** `merged_num_ctx` vs `num_ctx` field split on show; 2s free-VRAM cache for show; load path refreshes via loaded runners. Doc: [qwen35-apple-silicon.md](./qwen35-apple-silicon.md#manifest-num_ctx-vs-request-optionsnum_ctx-jun-2026), [scheduling-vram-policy.md](./scheduling-vram-policy.md#ggml-vram-suggest-and-opt-in-clamp-m12-jun-2026). |
| **M13** | **L1 GPU profiles (Metal tiers)** | Python | **Done (Jun 2026, M4 Max)** — RAM-tier JSON (`apple_silicon_16g` … `128g`), `gpu_profiles.py`, `/health.gpu_profile`. **Why:** unified memory needs different `-np`/batch than CUDA discrete VRAM; one conservative profile left 128 GiB machines under-utilized. Sign-off: `./scripts/m3_metal_signoff.sh`. Doc: [gpu-profiles-l1.md](./gpu-profiles-l1.md). |
| **M14** | **Portable Mac dev bootstrap (any checkout)** | Repo + docs | **Done (Jun 2026)** — `dev_bootstrap.sh`, `ensure_llama_cpp_sibling.sh`, `mac_setup.sh` tier 0 defaults (sign-off off, auto-clone `../llama.cpp`), `build_llama_server.sh` sibling path fix, port table (`:11434` daily vs `:8080` CI). **Why:** fresh clones failed without operator-specific `Sites/inference` layout, manual llama.cpp clone, or pre-pulled models; sign-off in default `mac_setup` blocked onboarding. Doc: [mac-dev-setup.md](./mac-dev-setup.md). **`doctor --fix`** runs `ensure_llama_cpp_sibling.sh` before Metal build. |

**Already optimized (Go, shipped):** Metal ggml runner, scheduler unified-memory behavior, Phase 8 broker with runtime embed.

**Mac onboarding (tiers 0–3):** **`./scripts/dev_bootstrap.sh`** → `./zerollama serve` → `pull` → optional `MAC_SETUP_SIGNOFF=1` — [mac-dev-setup.md](./mac-dev-setup.md). **Why separate from M9 sign-off:** tier 0 must succeed with zero pulled models and zero manual sibling clones.

**Mac operator default (why not legacy ggml):** `apple_silicon.yaml` → **`llama_backend: inprocess`**; Go proxy pulled tags need **`X-Zerollama-Runtime: 1`** or **`RUN_E2E_PHASE14=1`** in smokes — otherwise manifest names route to ggml and contend with the runtime sidecar on one Metal device.

**Not goals:** Replacing ggml Metal with MLX for all GGUF; NVML on Mac; duplicating `gpu_5080_session` on Darwin.

**Borrowings track:** Inference speed first — [Local voice & llama borrowings](#local-voice--llama-borrowings-eliza-v3) **L1–L3** (GPU profiles, fork kernels, KV prefix cache); voice **L5–L8** deferred.

---

## Local voice & llama borrowings (eliza-v3)

**Why a separate track:** [eliza-v3](https://github.com/elizaos/eliza) (`plugin-local-inference`, `elizaOS/llama.cpp`) ships a **fused on-device stack**—custom llama.cpp kernels, per-GPU autotune, and a duplex voice graph (ASR → MTP LLM → chunked TTS with barge-in). Zerollama already covers **OpenAI `/v1/audio/*`** via **Piper + Whisper subprocesses** ([multimodal-backends.md](./multimodal-backends.md)) and **MTP/ngram** in the Python runtime ([SPECULATIVE.md](../runtime/docs/SPECULATIVE.md)). This track ports **patterns and data** that fit our Go + Python shape—**not** the Eliza-1 bundle catalog, Capacitor/AOSP mobile loaders, or device-bridge WebSocket layer.

**Priority:** **Inference speed first** (tok/s, long-ctx VRAM, prefix-cache hit rate, MTP acceptance). Voice UX milestones (**L5+**) follow once **L1–L3** are measured on ship hardware (5080 + M-series).

**Reference tree (local):** `~/Sites/eliza-v3/plugins/plugin-local-inference/` (runtime + voice), `native/configs/gpu/` (autotune JSON), `packages/shared/src/local-inference/gpu-profiles.ts`, submodule `native/llama.cpp` → **`elizaOS/llama.cpp`**.

**Relationship to other tracks:** Distinct from [Zerollama remote cloud (Eliza)](#zerollama-remote-cloud-eliza) (HTTP proxy to Eliza Cloud). Complements [Phase 13](#local-inference--actionable-phases) (autotune), [Phase 15](#phase-15--exit-criteria-partial) (slot/KV), [Phase 17](#phase-17--upstream-gguf-path-alignment-directional) (llama.cpp pin). [Apple Silicon](#apple-silicon--metal-track): **L1** adds Metal/unified-memory profile variants; **L2** may add Metal KV kernels.

### Tier A — inference speed (do first)

| Milestone | Goal | Owner | Exit criteria |
|-----------|------|--------|----------------|
| **L1** | **Per-GPU llama profiles (CUDA + Metal)** | Python | **Partial (Jun 2026)** — `runtime/configs/gpu/*.json` + `runtime/gpu_profiles.py` → `RuntimeConfig.llama_server_args()` (`-b`, `-ub`, `-fa`, `--cache-type-k/v`, `-np`, MTP `draft_*`). **Apple: done** — RAM tiers (`apple_silicon_16g` … `128g`) from `hw.memsize`; M4 Max 128g sign-off (`-np 8`, `-c 131072`). **NVIDIA: partial** — **5080 measured (CT 1564):** detection PASS (`rtx-5080`, `n_parallel=4`); A/B @ 8k on 1B Q8 — profile ON **37.9** vs OFF **43.3** tok/s (single-stream regression; tune `rtx-5080.json` on production GGUF). Fork KV → `q8_0`; fork argv suppressed; `mlock` off by default. Disable: `ZEROLLAMA_GPU_PROFILE=0`. Doc: [gpu-profiles-l1.md](./gpu-profiles-l1.md). |
| **L2** | **`elizaOS/llama.cpp` fork evaluation** | Repo + C | **Partial (Jun 2026)** — sibling build, fork probe, profile unlock, `l2_fork_eval.sh`, `l2_metal_bench.sh`, `l2_cuda_bench.sh`, `l2_full_gate.sh`, `l2_cuda_full_gate.sh`, `linux_runtime_serve_lib.sh`. **Metal:** 8k stock wins decode; 27k/131k warmup fix in. **CUDA 5080 (Jun 2026):** 8k bench — stock **79.3** vs fork **56.9** tok/s (1B Q8); **FAIL merge**; compat PASS; long-ctx legs pending. **Gate open:** 27k/131k on 5080 + qwen35 ggml smoke. Pin `96dd1a8`. Doc: [gpu-profiles-l2.md](./gpu-profiles-l2.md). |
| **L3** | **Prompt cache key → slot bridge** | Go + Python | **Partial (Jun 2026)** — pinned slots, subprocess + in-process RAM resume, **in-process disk parity** (`llama_state_seq_*`), batch keys, `/health.llama_cache`, smokes. **5080 (Jun 2026):** `l3_cache_smoke.sh` **SOFT PASS** (bridge wired; 1B @ 8k no latency win). **Open:** strict PASS on larger model / agent bench. Doc: [gpu-profiles-l3.md](./gpu-profiles-l3.md). |

**Suggested order (Tier A):** **L1** → **L3** (low friction, immediate wins) → **L2** (fork spike in parallel with L1 measurement; merge when gated).

### Tier B — voice (after Tier A baseline)

| Milestone | Goal | Owner | Exit criteria |
|-----------|------|--------|----------------|
| **L5** | **Voice phrase cache (Piper)** | Go | Pre-synthesize short fillers at serve/warm; masks TTS cold-start—not LLM tok/s. Ref: eliza `phrase-cache.ts`. |
| **L6** | **Qwen3-ASR in-process** | Go + Python | `modality_backends.transcribe: qwen3-asr`; Whisper subprocess fallback. Removes ASR subprocess overhead only. |
| **L7** | **Duplex voice pipeline** | Go + Python | Streaming ASR → LLM (MTP) → chunked TTS; VAD barge-in; MTP rollback on TTS. Ref: eliza `engine-bridge.ts`. |
| **L8** | **Kokoro TTS backend (optional)** | Go | Lightweight ONNX TTS alongside Piper. Defer OmniVoice until **L7**. |

**Non-goals (this track):**

- **iOS / Capacitor / AOSP** — not a zerollama deployment target today.
- **Apple Foundation Models** fast-path (iOS 26+ only).
- **Eliza-1 curated bundles** — zerollama keeps Modelfile/manifest + registry pull.
- **Standalone OmniVoice / libelizainference** monolith unless **L7** proves subprocess voice insufficient.
- **Replacing** Eliza Cloud proxy or LM Studio import with eliza local-inference UI.

**Cross-links:** Spec decode — [runtime/docs/SPECULATIVE.md](../runtime/docs/SPECULATIVE.md). CUDA — [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md). Mac — [apple-silicon-metal.md](./apple-silicon-metal.md). Voice today — [multimodal-backends.md](./multimodal-backends.md).

---

## Fleet scheduling (multi-node)

**Why a separate track:** Agents and integrations often see **many zerollama hosts**, not one. Per-node schedulers (Go + Python) are correct locally but do not answer “**which box has this model warm?**” or “**may I cancel and try another node without wasting VRAM?**” Scatter-gather and long quote/reservation windows **waste GPU work** on constrained fleets; the target is a **thin management node**, **warm-model routing**, and **honest status** over the wire.

**Guide:** [fleet-scheduling.md](./fleet-scheduling.md) · **Operator:** [fleet-management.md](./fleet-management.md)

| Milestone | Goal | Owner | Exit criteria |
|-----------|------|--------|----------------|
| **F1** | **Stream progress contract** | Go + Python | **Shipped** — `/api/chat` and `/api/generate` emit `status` (`accepted` / `queued` / `loading` / `generating`), `position`, `queue_depth` on streaming paths; runtime proxy flushes `accepted` before forward. Agents can implement cancel-while-queued. Doc: [fleet-scheduling.md](./fleet-scheduling.md#status-contract-node--agent). |
| **F2** | **Node status for fleet polling** | Go + Python | **Shipped** — `GET /api/status` includes `inference.ggml` (pending/active/loaded, `loaded_models`, `loading`) and `inference.runtime` (waiting/running, `llama_loaded`, probe `available`). Doc: [fleet-scheduling.md](./fleet-scheduling.md#shipped-fleet-polling). |
| **F3** | **Management node v0** | Go | **Shipped** — `zerollama fleet serve`; static peers; warm-model map; `POST /api/fleet/assign` returns `{url, node_id}`. Doc: [fleet-management.md](./fleet-management.md). |
| **F4** | **LAN discovery (mDNS)** | Go (+ mgmt) | **Shipped** — nodes advertise `_zerollama._tcp` (`ZEROLLAMA_MDNS=1`); fleet browses (`--mdns`) and optionally advertises `_zerollama-fleet._tcp`. Static `ZEROLLAMA_FLEET_PEERS` still supported. |
| **F5** | **Short-TTL assignment token** | Go + mgmt | Optional header; ~5–10s hold one queue slot; validate on node; expire without long quote window. |
| **F6** | **Operator + agent playbooks** | Docs | Sticky model shards, warm-only SLA, cancel policy (`queued` yes, `loading` no); explicit non-goals (scatter-gather, 60s quotes). |

**Routing policy (directional):** Prefer **loaded model + lowest queue**; cold route only when SLA allows; management **assigns** node, never starts loads remotely.

**Relationship to Phase 11 / T6:** Single-GPU admission and training defer remain on each node. Fleet layer only **chooses** which node receives the request.

**Not goals:** Global preemption across nodes; Redis pull-queue as v1 requirement; reservation market with frequent cancel (penalty/backoff deferred).

---

## GPU training (fine-tuning)

**Shipped direction:** Go embeds **CPython** (`x/trainingworker/pyembed`), serves **`/api/train/*`** and legacy **TCP `:9500`** (newline JSON), and coordinates VRAM with the scheduler on CUDA OOM (pause new inference loads → unload runners → ack Python). Training logic stays in repo-root **`training.py`**, loaded in-process—not a second public listener on 9500.

**Why this track exists:** Fine-tuning stacks (Transformers, PEFT, bitsandbytes) are Python-native; Ollama’s control plane is Go. Embedding Python keeps **public wire** in Go while avoiding a second process and gRPC for every deployment.

### Phases (training track)

| Phase | Goal | Exit criteria |
|-------|------|----------------|
| **T1** | **Proactive VRAM (with inference Phase 8)** | **Done** — broker before `load_model` and on inference/runtime paths; OOM path remains safety net. |
| **T2** | **Auth on `/api/train` and `:9500`** | Same threat model as main HTTP API. |
| **T3** | **Progress over HTTP** | SSE or WebSocket; reduce poll-only UX. |
| **T4** | **CPU CI smoke** | **Partial** — Go tests register `/api/train/*` and exercise `GET /api/train/status` (502 without embed is OK in CI). Full embedded-python health still needs integration. |
| **T5** | **Native training (optional)** | Rust/libtorch only if Python embed becomes a bottleneck—default stay on PyTorch. |
| **T6** | **Unified queue policy (directional)** | **Partial** — idle-wait; `priority`; defer queue; **allowed window**; **cross-queue FIFO** (global tickets, Go↔Python mirror). **Next:** richer SLO classes, stricter training/inference class separation. |

### Non-goals (for this track)

- **Guaranteeing** mid-training automatic resume after OOM without checkpoints—unsafe for arbitrary `Trainer` loops; v1 notifies Go and may retry **model load** only.
- **Replacing** `training.py` with an empty stub—the library is the reference implementation until a deliberate migration plan exists.

**Why a separate subsection from video / Eliza:** Training touches **subprocess lifecycle**, **GPU memory shared with llama runners**, and **optional TCP**—different failure modes and operators than multimodal decode or remote HTTP.

---

## Video understanding (VLM) — shipped direction

- **Native path:** `video_url` / `videos` → ffmpeg frame sampling → same vision pipeline as images (**frame-list semantics**).
- **Optional SGLang:** Full-body proxy of `POST /v1/chat/completions` when `video_understanding=sglang` and `OLLAMA_SGLANG_URL` is set.

**Why** this split: local users should not need another server; advanced users can delegate decoding and model-specific video handling to SGLang when they already run it.

## Video generation — Wan T2V v1 (shipped)

**Why a separate track from VLM:** Video **understanding** (ffmpeg → vision encoder) and video **generation** (diffusion on GPU for minutes) share almost no code. Mixing them in one roadmap bullet hid ownership, testing, and API limits.

**Shipped (v1):**

- OpenAI async Videos API on Go `:8080` — `POST /v1/videos`, `GET /v1/videos/:id`, `GET /v1/videos/:id/content`.
- Local **Wan** only (`wan2.1-t2v:1.3b`, `wan2.2-ti2v-5b`, 16g manifests); weights via `install_wan_video.sh`, not Ollama blobs.
- Jobs run as training **`run_script`** (wrapper `scripts/wan_video_generate.py` → upstream `generate.py`).
- **VRAM / queue:** training handoff, optional `defer-*` when inference busy; artifacts under `$OLLAMA_MODELS/generated/<job_id>.mp4`.

**Operator guide:** [wan-t2v.md](./wan-t2v.md) (architecture, status mapping, env, troubleshooting).

| Milestone | Goal | Why |
|-----------|------|-----|
| **v1** | Wan T2V via training queue | **Done** — reuse embed + broker instead of new subprocess daemon. |
| **v1.1** | Cancel + TTL + kill running Wan | Operators need to abort long jobs and reclaim disk. |
| **v1.2** | TI2V `input_reference`, 24g/32g tiers | Product parity with Wan2.2 TI2V; headroom for longer clips. |
| **v1.3** | Optional **Wan CPU worker** sidecar | Remote T5 encode / VAE on high-RAM host; 16 GB GPU CT. Plan: [wan-cpu-worker.md](./wan-cpu-worker.md). |
| **Later** | Diffusers / other `runner` values | Other stacks without forking Wan wrapper for each upstream. |

**Not in v1:** list videos, `DELETE /v1/videos/:id`, Eliza `:cloud` passthrough, in-process Diffusers runner.

**Future tracks:** GGUF Wan2.2 if 16 GB OOMs; upstream proxy for non-Wan stacks; CogVideoX / LTX via `runner: diffusers`.

## Option 2 — Narrow the gap without SGLang (in-tree over time)

**Execution checklist:** [video-parity.md](./video-parity.md) (reference workloads + parity matrix).

**Goal:** Keep inference **inside** Ollama/zerollama (ffmpeg + vision runner) and **deliberately port or reimplement** the behaviors that matter, instead of depending on an HTTP proxy to SGLang.

**Why a separate “Option 2” track:** Many users want **native** inference only (no second server). Option 2 spells out **policy** (how frames are chosen), **representation** (how templates see images vs video), and **limits** (context, mllama) as separate layers—so we do not conflate “better ffmpeg” with “SGLang-style scheduling.”

**Reality check:** SGLang is a large Python serving stack; **100% behavioral parity on every model** is not a single milestone. This roadmap is about **closing the gap** where it matters for **your** models and workloads.

### Phase A — Decode and sampling policy

- Align **fps / stride / max frames** with reference behavior for target models (env + per-model options where needed).
- Expand **container support** (what ffmpeg accepts) and document **failure modes** (corrupt input, no keyframes).
- Optional: **deterministic** sampling (fixed seeds / fixed frame indices) for reproducible evals.

**Why first:** Native path quality is dominated by **how** you turn video into frames, not the HTTP boundary.

### Phase B — Renderer and template semantics

- Where a model distinguishes **video spans** vs **unrelated images**, extend templates/renderers with **explicit placeholders** (not only a flat `[img-*]` list).
- Per **model family** (e.g. Qwen3-VL, others): document **expected** ordering and token layout.

**Why:** Frame-list semantics are a **ceiling** until templates express “N frames from one clip.”

### Phase C — Context and limits

- **Token-aware** budgeting: relate frame count to **effective vision tokens** and `num_ctx` so users get **actionable errors** before runtime blowups.
- Tune **mllama** / single-image constraints vs multi-frame video (clear errors or automatic downsample).

**Why:** SGLang’s stack does scheduling/budgeting; native path must encode **policy** explicitly.

### Phase D — Validation and regression

- **Golden tests:** small fixtures (short MP4/WebM) with **expected frame counts** / hashes after sampling.
- **Per-model smoke:** optional CI jobs when ffmpeg + GPU are available.

**Why:** Parity is proven by **tests**, not by matching another repo’s README.

### Phase E — Optional subprocess bridge (still no SGLang HTTP proxy)

- If a **specific** decode or preprocessing step must stay in Python, a **narrow subprocess** contract (stdin/stdout or temp files) can wrap **only that step**, with Ollama owning scheduling and limits—different from proxying full chat.

**Why:** Sometimes parity needs **one** binary without adopting a second full server.

### What this does *not* promise

- Automatic parity with **every** SGLang model and feature.
- Replacing **SGLang’s** distributed scheduler or custom kernels without equivalent work in Ollama.

## Cross-cutting / hardening

| Item | Phase hint | Why |
|------|------------|-----|
| **Per-GPU llama flags + MTP autoconfig** | Borrowings **L1**; Phase **13** | Tuned batch/parallel/draft on stock cache types—first measurable tok/s win |
| **Long-ctx KV quant kernels** | Borrowings **L2**; Phase **17** | TurboQuant / QJL / Polar—largest VRAM + decode win when fork merges |
| **Prefix cache → llama slots** | Borrowings **L3**; Phase **15** v1b | **Why:** dynamic slots discard KV each turn; stable keys → pinned `id_slot` + `cache_prompt` skip repeat prefill on agent threads. Audit (Jun 2026): canonical disk paths, orphan hash-dir sweep, strict batch keys, native bind before slot release. Doc: [gpu-profiles-l3.md](./gpu-profiles-l3.md). |
| Inference vs training **priority / idle policy** | Training **T6**, inference **Phase 11** | One GPU, many clients—documented target is queued work + policy, not “implicitly fair” |
| **Fleet management + warm routing** | Fleet **F2–F5** | Many nodes, many agents—thin orchestrator, status, mDNS; avoid scatter-gather on constrained GPUs |
| **Local voice latency + duplex** | Borrowings **L5**, **L7** (Tier B) | Phrase cache + streaming pipeline after inference baseline |
| **In-process ASR (Qwen3-ASR)** | Borrowings **L6** (Tier B) | ASR subprocess removal—not LLM tok/s |
| Eliza catalog / response mapping | Eliza follow-ups | Operator UX when local + cloud lists collide |
| Video Option 2 A–D | Video track | Native VLM quality without SGLang dependency |
| SSRF hardening | Security | High-assurance deployments |
| ffmpeg / SGLang E2E | Video + hardening | Regression beyond unit tests |

## How to contribute

Open an issue or PR with a concrete use case (API shape, model family, deployment constraints). **Why** matters as much as **what** for multimodal features—resource limits and API compatibility affect everyone.

---

## LM Studio integration

**Why this track exists:** Many Mac operators run LM Studio for discovery/download and zerollama for API/agents. Sharing the same on-disk cache avoids duplicate downloads and registry bandwidth. The hard part is **layout mismatch** (LM Studio tree vs zerollama blob store) and **format split** (GGUF symlinks vs MLX repack).

### Shipped (v0.0.1)

| Item | Why |
|------|-----|
| Discover `~/.lmstudio/models` (+ optional roots) | Default LM Studio layout; no config file to maintain |
| Merge into `list` / `/api/tags` with `remote_host=lmstudio` | Agents see one catalog; local pulls win on name collision |
| Pull-from-cache (GGUF symlink, MLX native import) | Skip registry when files already exist |
| Disk checks + `OLLAMA_LMSTUDIO_LIST_ALL` | MLX repack needs ~full model free; hide or list-all avoids confusion |
| `weightFilesOnly` / `dirIsMLXSafetensors` | Correct path selection; GGUF+config dirs must not hit MLX or JSON-as-GGUF |

Doc: [lmstudio-import.md](./lmstudio-import.md).

### Directional follow-ups

| Item | Why | Exit criteria |
|------|-----|---------------|
| **In-place MLX load** | Repack doubles disk for large models on tight volumes | Manifest references external LM Studio path; mlxrunner reads without full copy; disk check optional or advisory only |
| **Cross-volume symlinks** | `OLLAMA_MODELS` on small disk, LM Studio cache on large external drive | Document + test `createLink` across volumes; clear errors when symlink unsupported |
| **Quant tag parity** | LM Studio folder names ≠ Ollama tags | Expand fuzzy match tests; document naming table per publisher |
| **CI fixture cache** | Regression without real 70 GB weights | Temp-dir fixtures in `internal/lmstudio` + `server/lmstudio_*_test.go` (partial today) |
| **GLM / Hermes MLX runtime bugs** | Some MLX families panic in `mlxrunner` | Per-model compatibility matrix in docs; upstream MLX fixes |

**Non-goals:** Proxying LM Studio’s own HTTP server; modifying LM Studio cache files in place; automatic two-way sync with LM Studio’s internal DB.
