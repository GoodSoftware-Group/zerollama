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
| **11** | **VRAM + admission policy in Python** | Python | **Partial** — inference-first + VRAM checks; **low** throttling; min-free + training reserve via env or `single_gpu.yaml` `vram:` block. Backpressure thresholds overridable (`ZEROLLAMA_RUNTIME_BACKLOG_BATCH_MIN`, …). **5080:** `gpu_5080_session.sh` PASS with `RUN_E2E_PREFLIGHT=0` on CT 1564. **Mac (Jun 2026):** `./scripts/phase11_metal_admission_smoke.sh` or ordered `./scripts/phase11_13_15_metal_signoff.sh` — metal-unified probe + `apple_silicon.yaml` admission fields (M4 Max PASS). |
| **12** | **Runtime default for text local models** | Go + Python | **Done** (tools path) — default-on; streaming proxies; tools via Go render + stateful `parse-tool-output` sessions. Render ctx aligned with load via `resolve_num_ctx_for_request`. v1 proxy injects manifest `options.gguf`. CI goldens: `./scripts/phase12_golden_ci.sh`. **Harmony real-weight:** CI synthetic only; `gpt-oss:20b` needs ~40+ GiB host RAM on runtime path (not required on 5080). |
| **13** | **Single-GPU + host autoconfig** | Python | **Partial** — estimates, autotune catalog + `estimate_factor_source`, `suggested_max_num_ctx`, clamp default **off** in YAML; `python -m runtime.gpu_snapshot` after session JSON; `vram:` defaults in `single_gpu.yaml`. **5080 gate:** [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md). **Mac (Jun 2026):** `./scripts/phase13_metal_vram_smoke.sh` — live estimate + snapshot (M4 Max PASS). Doc: [phase13-runtime-vram.md](./phase13-runtime-vram.md). **L1 profiles:** [gpu-profiles-l1.md](./gpu-profiles-l1.md) (**Done** — Metal tiers + 5080 CUDA gate). |
| **14** | **In-process llama forward** | Python → C/Rust | **Done** — see [exit criteria](#phase-14--exit-criteria-done). Shipped: ctypes `inprocess`, wheel (CPU default), tokenize, sampling, YAML `llama_backend`, `llama_backend_source`, `llama_cpp` `/health`, heap-batch decode fix. Smokes: `phase14_inprocess_smoke`, `phase14_5080_signoff`, optional `phase14_wheel_gpu_smoke` (failed on 5080). Doc: [phase14-inprocess-llama.md](./phase14-inprocess-llama.md). |
| **15** | **Native scheduler + KV** | C/Rust | **Partial (v0–v47 ops)** — see [exit criteria](#phase-15--exit-criteria-partial). C pool + tick/decode hooks; **v9–v16** decode plan export, libllama link, C decode loop (GIL + sampling in C), engine resume via `current_pos`; **v24–v30** page-bind validation, auto-link build, 131k ctx bind cap, continuous batch decode + engine wiring + `/health` batch plan + streaming batch decode + per-row C batch sampling; **v33** fork writable page-map (`llama_memory_kv_page_map`); **v35** transposed-V layout API + last-probe health; **v36** GGUF layer-group enrichment (`kv_full_layers`, `kv_swa_layers`, `tensor_layers_expected`); **v37** stream auto-batch (`ZEROLLAMA_KV_AUTO_BATCH_STREAM=1`); **v38** copy descriptors + `tensor_layers_bind_complete`; **v39** `kv_page_migration` on `/internal/kv-snapshot`; **v40** `page_migration_summary` on forward plans + snapshot pointer redaction; **v41** operator sign-off smokes (v40 redaction + stream auto-batch GPU gate); **v42** `page_migration_summary` on `/health.kv_page_bind` + `migration_summary` on snapshot branches; **v43** migration summary GPU sign-off smokes; **v44** non-stream auto-batch GPU gate (`phase15_auto_batch_smoke.sh`); **v45** auto-batch env wiring + combined sign-off; **v46** Linux embed auto-batch parity + `RUN_E2E_PHASE15_AUTO_BATCH`; **v47** external-buffer alias probe + validate (patch 0019, no tensor mutation). Go KV snapshot; GPU sign-off **PASS** on Metal (M4 Max) and CUDA 5080 (CT 1564). **Open:** ggml allocator overlay bind (`HOST_REBASE` / device alias). Docs: [phase15-native-kv.md](./phase15-native-kv.md), [handoff-phase15-native-kv.md](./handoff-phase15-native-kv.md). |
| **16** | **Thin edge daemon** | Rust or minimal Go | **Partial (v0 ops, v1 runner stub, v2 CGO drop)** — `-tags edge` excludes in-process ggml (`server.go`); llama-server-only `NewLlamaServer`. Doc: [phase16-thin-edge.md](./phase16-thin-edge.md). |
| **17** | **Upstream GGUF path alignment** | Go + llama.cpp | **Partial** — criteria 1–6 done; Linux `auto` + `/api/status` `backend` + `/api/version` `edge_build`; only L2 pin merge open. Doc: [phase17-llama-server.md](./phase17-llama-server.md). |

**Deprioritized:** public `POST /api/runtime/unload` or `/resume` — automatic eviction only ([Phase 8](#local-inference--actionable-phases) → [Phase 11](#local-inference--actionable-phases)).

**Non-goals (this ladder):** full RadixAttention ref-count DAG (v1 donor seed shipped — [radix-prefix-share.md](./radix-prefix-share.md)); required vLLM/SGLang servers; rewriting training in C++ (`llama-finetune` WIP); bit-for-bit SGLang parity.

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
| 4 | **5080 GPU:** `phase15_inprocess_signoff.sh` (KV hook + multi-seq + batch decode + `kv_page_bind` snapshot) | **Done (RTX 5080 CT 1564, Jun 2026)** — OuteTTS 1B Q8; `kv_decode_steps=56`; `batch_decode_in_c=True`; multiseq + `/internal/generate-batch` PASS |
| 5 | **Tensor page bind** — PA `block_ids` → llama KV tensor pages | **Partial (Jul 2026)** — **v8:** seq-position bind; **v19–v20:** cell + tensor verify via `llama-kv-ext.h`; **v33:** fork `llama_memory_kv_page_map` writable spans + `physical_pages_bound` on `/health`; **v34:** multi-layer tensor verify + writable page-map fan-out (`llama_memory_kv_n_layers`); **v35:** transposed-V layout (`llama_memory_kv_cache_layout`, `v_transposed` on page_map, last-probe `/health`); **v36:** GGUF layer-group enrichment (`kv_full_layers`, `kv_swa_layers`, `tensor_layers_expected` for hybrid model bind-success criterion); **v38:** copy descriptors for migration; **v47:** external-buffer alias **validate** (patch 0019 — classifies SAME_POINTER / HOST_REBASE / BLOCKED_* without mutating tensors). **Open:** ggml allocator overlay bind (v48+) |
| 6 | **Native decode batch** in C wired to `kv_forward_plans` | **Partial (Jun 2026)** — C batch layout + page-aligned chunks; **v9–v11:** plan export; **v12:** libllama link; **v13:** `llama_decode` in C; **v14–v16:** GIL release, sampling in C, `_decode_stream` + engine resume via `current_pos`; **v16b–v18:** resume owner + `/health.kv_resume`; **v19–v20:** tensor bind scaffold + `llama-kv-ext` cell/tensor bind; **v20a:** `native_page_table` on forward plans; **v26–v30:** `kv_decode_loop_run_batch_step`, engine `generate_batch` / `stream_generate_batch`, per-row `smpl_ptrs[]`, `/internal/generate-batch`, **Metal batch sign-off PASS (M4 Max Jun 2026)**, **CUDA 5080 batch sign-off PASS (CT 1564 Jun 2026)** |

Mark **Done** when 1–3 and **4** pass on ship hardware. **5–6** partial until upstream llama.cpp ships stable writable KV page API for all memory types. CPU gate: `./scripts/phase15_kv_native_ci.sh`. GPU gate: `./scripts/phase15_inprocess_signoff.sh` (Linux embed) + `./scripts/phase15_metal_signoff.sh` (Mac sidecar, includes batch decode step 3/5). **Mac ordered gate (Jun 2026):** `./scripts/phase11_13_15_metal_signoff.sh` (`METAL_SELF_START=1`, vendor libllama via `macos_export_llama_cpp_paths`). **Mac full operator gate:** `./scripts/metal_signoff.sh` (+ optional `RUN_E2E_QWEN35_MODEL=eliza-1-2b:latest`). Linked tensor bind + batch decode: rebuild libllama from vendor pin + clean `_kv_native` build; sign-off scripts source `phase15_runtime_kv_env.sh`; **`smoke_runtime_assert_kv_snapshot`** accepts **`bound`+`tensor`** when kv-ext linked. See [phase15-native-kv.md](./phase15-native-kv.md).

### Phase 17 — upstream GGUF path alignment (directional)

**Why:** Upstream Ollama removed `runner/ollamarunner` for text GGUF and integrated **`llama-server` from Go** (`llm/llama_server.go`). Zerollama still defaults to **ggml runner** on Mac and uses **Python runtime** as the bridge for `--llama-cpp-backend`. Aligning default GGUF with upstream reduces merge pain, modernizes the llama.cpp pin, and removes an extra hop (Go → Python → llama) from the hot path—without dropping training or Phase 15 work.

| # | Criterion | Owner |
|---|-----------|--------|
| 1 | Document deltas vs upstream checkout | **Done** — [upstream-ollama-diff.md](./upstream-ollama-diff.md) |
| 2 | Bump sibling llama.cpp + pin toward upstream **`b9781`**; rebuild `llama-server` | **Done (Jun 2026)** — vendor `llama-cpp-b9781`, 16 patches, in-tree sync + Metal embed regen |
| 2b | **Rebase in-tree ggml/llama.cpp to real b9509** (not overlay snapshot) | **Done** — 12 patches, vendor sync script, build+doctor; [ggml-b9509-migration.md](./ggml-b9509-migration.md) |
| 3 | Port `llama/compat/` overlay; reduce overlapping `llama/patches/` | **Done (Jun 2026)** — compat imported; 0007 retired (BakLLaVA → compat); 0016 hooks + 0017 ggml deltas in vendor series; 0015 header patch fixed |
| 4 | Port `llm/llama_server.go` + discovery probe; eligible GGUF uses Go → llama-server | **Done (Jun 2026)** — `--llama-server-backend`; **Linux auto-default**; `discover/llama_server.go`; `LeadingBOSForRenderer`; **`phase17_llama_server_smoke.sh` PASS** — [phase17-llama-server.md](./phase17-llama-server.md) |
| 5 | Benchmark ggml vs Go-llama-server vs Python runtime on ship hardware | **Done (M7)** — ggml ~164 vs upstream ~158 tok/s @ 4k ctx; keep ggml Mac default |
| 6 | Deprecate `OLLAMA_NEW_ENGINE` / ollamarunner for plain text GGUF | **Done (Jun 2026)** — env ignored for routing; Linux `auto` + explicit llama-server; Mac ggml default |
| 7 | Coordinate llama.cpp pin with borrowings **L2** (fork KV profiles vs L1 q8_0) | **Partial** — QJL/Polar/TBQ on ggml-org `8f114a9b` (patches **0026–0030** + CUDA follow-ups **0067–0070**); **FAIL default profiles** (5080 @ 8k/27k; 4090 TBQ @ 65k–131k quiet host −20…−21% decode / −27…−34% VRAM). Profiles stay opt-in. `./scripts/phase17_l2_pin_status.sh` |

**Non-goals:** full rebase onto upstream; deleting `runtime/` or training; replacing Eliza with ollama.com.

**Jun 2026 ports — why each landed:**

| Port | Why zerollama needed it |
|------|-------------------------|
| **llama-server discovery** | Upstream schedulers expect CUDA `ARCHS=` + ROCm gfx from llama-server stderr; ggml `/info` never had parity. Hybrid bootstrap keeps Mac ggml default. |
| **LeadingBOS** | DisableJinja means Go owns the full prompt; without BOS dedup, Gemma4/LFM2 double-count the first token. |
| **PromptTokens from chatPrompt** | Tail-truncate by token ID; re-tokenizing truncated text diverges on edge cases and breaks MLX MTP budget checks. |
| **MLX M15 / M15a (agent prompts)** | Context cap, tokenize LRU, keep-alive, SSE keepalive, live-session + rotating-KV restore, prompt-chain on truncate — [mlx-agent-prompts.md](./mlx-agent-prompts.md). **Why:** 131k megaprompts every turn; turn-2 trie hits without `fast_path` still slow on OptiQ. |
| **M15b QoS + project tracking** | Session gate TOCTOU fix, `zerollama ps` PROJECT/SESSION, inference-path branching, client Tier 2 ladder — [agent-qos-and-project-tracking.md](./agent-qos-and-project-tracking.md). **Why:** concurrent streams raced MLX runner; operators couldn't attribute GPU; Tier 2 options must not hit vanilla Ollama/CUDA proxies. |
| **PreservedTokens** | Harmony/tool parsers register vocab IDs llama-server must not shift during context operations. |
| **Launch drift guard** | `zerollama launch` writes inline config; stale files caused silent wrong-model agent sessions. |
| **Pin `b9781` (file)** | Single source of truth for sibling `../llama.cpp`; vendor sync is gated — **why:** pin file documents intent before every operator runs full vendor re-apply |
| **v0.30.11 Go port** | Native chat on generate, CUDA/Vulkan discovery fixes, MLX speculate refactor — **why:** merge parity without Claude/OpenCode auto-install or Mac engine swap |
| **`-lc++` in `llama.go`** | CGO tests (`go test ./discover/`) link jinja C++ without requiring full production `CGO_LDFLAGS` from shell scripts. |
| **Native `gpu-discover`** | llama-server stderr lacks PCI IDs; subprocess probe merges compute capability + gfx without loading a model. |
| **Integrated GPU / gfx1151** | Strix Halo iGPU dropped by default iGPU filter; upstream allowlists `gfx1151`. |
| **Metal discovery retry** | Tensor API probe failures retry with `GGML_METAL_TENSOR_DISABLE=1` and persist on device env. |
| **Cohere2 MoE MLX (#16670)** | Command A / North safetensors need `cohere` parser/renderer + `x/models/cohere2_moe`. |
| **OMP launch (#16410)** | oh-my-pi agent integration; ManagedSingleModel + web search plugin gate. |
| **Cline providers.json (#16402)** | Cline CLI reads `providers.json`; dual-write with legacy `globalState.json`. |
| **Qwen Code launch** | Upstream `zerollama launch qwen` — settings.json OpenAI provider → `/v1`. |
| **Pool launch** | Upstream `zerollama launch pool` — enterprise agent CLI env wiring. |
| **Phase 17 smoke** | E2E Go→llama-server generate via pulled tag (`P17_MODEL`); accepts `thinking` field. |
| **Phase 17 vision smoke** | Opt-in `phase17_llama_server_vision_smoke.sh` — chat+image on `--llama-server-backend`; auto-picks smallest projector manifest. |
| **Launch model inventory** | One `/api/tags` load per run → `LaunchModel[]`; drops N× `/api/show` at config time. [launch-model-inventory.md](./launch-model-inventory.md) |

**Compare workflow:** `./scripts/clone_upstream_ollama.sh` → build upstream on `:11435`, zerollama on `:11434`. See [upstream-ollama-diff.md](./upstream-ollama-diff.md#compare--benchmark-workflow).

### Phase 16 — exit criteria (Partial)

| # | Criterion | Owner | State |
|---|-----------|--------|--------|
| 1 | `--edge` routes GGUF via llama-server; runtime chat off | Go | **Done (v0)** |
| 2 | Linux serve `auto` routes all GGUF | Go | **Done (Jun 2026)** |
| 3 | Operator doc + env table | Docs | **Done** — [phase16-thin-edge.md](./phase16-thin-edge.md) |
| 4 | Edge smoke (`phase16_edge_smoke.sh`) | Repo | **Done (Mac + CUDA 5080 Jun 2026)** — `RUN_E2E_UPSTREAM_GGUF=1` bundle PASS on CT 1564 (serve profile-off restart before base smokes) — **runbook:** [5080-runbook.md](./5080-runbook.md#tier-4--phase-16--17-upstream-gguf-path) |
| 5 | `/api/status` `inference.backend` policy snapshot | Go | **Done (Jun 2026)** — fleet + operator visibility |
| 6 | Edge compile marker (`-tags edge`) | Go | **Done (v1)** — `build_zerollama_edge.sh`; subprocess runner stub; `ggml_linked=false` in `/api/status` |
| 7 | Drop in-process ggml from edge binary | Go | **Partial (v2)** — `server.go` excluded with `//go:build !edge`; edge dep tree has no `llama`/`model` CGO; Python embed/MLX remain |
| 8 | Inference control plane “Go gone” | Go | **Not started** — sched loads llama-server + MLX only |

Mark **v0 Done** when 1–5 pass and criterion 4 smoke passes on ship hardware (Mac **Done** Jun 2026; Linux CUDA **Done** Jun 28 2026 on CT 1564 via individual P17/edge smokes).

### Phase 8 — shipped

See `server/vram/broker.go` and `server/runtime_manifest.go`. **Next (ship hardware):** Phase **11** admission tuning; Phase **15** writable tensor bind (upstream-blocked); Phase **17** L2 pin merge. **Done on 5080 (Jun 2026):** [5080-runbook.md](./5080-runbook.md) tiers 1–4 + Radix live + `RUN_E2E_UPSTREAM_GGUF` bundle. **Production serve:** [`serve_production_wrapper.sh`](../scripts/serve_production_wrapper.sh) → `~/bin/serve.sh` (WHY: in-repo `serve_gpu_example.sh` must not be copied verbatim to `~/bin`).

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

**Direction:** GGUF-first **`runtime/`** with PagedAttention block pools; not HTTP to vLLM/SGLang. **Embed:** [runtime-embed.md](./runtime-embed.md). **Spec decode:** [runtime/docs/SPECULATIVE.md](../runtime/docs/SPECULATIVE.md). **Smoke validated:** [testing-smoke.md](./testing-smoke.md) on RTX 5080-class hosts. **Operator ladder (WHY):** [5080-runbook.md](./5080-runbook.md) — ordered tiers (base → L1/L3 → Phase 15 → upstream GGUF); [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md) — build/serve/troubleshooting. One session script proves Phase 10–13 on 16 GB; harmony real-weight is **not** required (CI Go golden + host-RAM limits for `gpt-oss:20b`).

**Still on ggml runner:** vision, logprobs, think, MLX safetensors, and models without `zerollama-runtime` backend. **Tools** on runtime-routed text models use Go render/parse (Phase 12 — done). Handoff: [handoff-phase12-runtime-tools.md](./handoff-phase12-runtime-tools.md).

**Upstream tension:** Vanilla Ollama has **no** Python runtime; default GGUF is **Go → llama-server** ([upstream-ollama-diff.md](./upstream-ollama-diff.md)). Zerollama’s Python layer stays for PA, admission, training, and Phase 15—but **Phase 17** targets upstream-style Go integration for default text GGUF. **`--llama-cpp-backend`** is the current test harness (Go → Python → llama), not the long-term default shape.

---

## Apple Silicon & Metal track

**Why a separate track:** CUDA Phase 11/13 work (`single_gpu.yaml`, `nvidia-smi`, `gpu_5080_session.sh`) does not apply to unified memory. Mac users need **Metal ggml** (default), **runtime admission** that probes `vm_stat`, and clear **MLX vs GGUF** routing—not a copy of the 5080 playbook.

**Guide:** [apple-silicon-metal.md](./apple-silicon-metal.md)

| Milestone | Goal | Owner | Exit criteria |
|-----------|------|--------|----------------|
| **M1** | **Unified memory admission** | Python | **Shipped (audit)** — `metal-unified` probe; `read_host_memory()` on darwin (load + `/health`); `apple_silicon.yaml` autoconfig; `check_gguf_host_budget` no longer Linux-only; `vm.swapusage` parser fixed. |
| **M2** | **Operator smoke + docs** | Repo | **Shipped** — `macos_metal_smoke.sh`; guide + ROADMAP; pytest for darwin probe + snapshot hints; `check_gpu_scripts` greps. **Jun 2026:** ordered Phase 11→13→15 Mac gate — `./scripts/phase11_13_15_metal_signoff.sh` (M4 Max PASS; `METAL_SELF_START=1` self-contained). |
| **M3** | **Runtime Metal parity** | Python | **Shipped** — `m3_metal_signoff.sh` / `gpu_metal_session.sh`; Phase 13 snapshot + Phase 14 inprocess on Metal (M4 Max sign-off, Jun 2026). `apple_silicon.yaml` sets **`llama_backend: inprocess`**; M3 validates `llama_backend_source=config`. Use a **text-only** GGUF with pinned llama.cpp (not vision gemma3 on old pin). Mac daily serve: **`zerollama serve`** (auto sidecar `:8081`); `./scripts/serve_mac_runtime.sh` for CI (prints log paths — see [fleet-management.md](./fleet-management.md#macos-runtime-stack-related)). Optional Phase 15: `RUN_E2E_PHASE15=1 ./scripts/m3_metal_signoff.sh` or `./scripts/phase15_metal_signoff.sh`. |
| **M4** | **MLX policy** | Go + docs | **Shipped** — [mlx-routing-policy.md](./mlx-routing-policy.md); `IsMLX()` excluded from runtime default **and** explicit Modelfile backend; Go tests. **Dylibs:** rebuild at `MLX_VERSION` / `MLX_C_VERSION` via `build_production_mac.sh` (Jun 2026 sign-off @ `2165dc08` / `fba4470b`). **Optional build path:** [mlx-cgo](#optional-mac-mlx-build-path-mlx-cgo) (not default). |
| **M5** | **Phase 15 Metal KV sign-off** | Python | **Shipped (Jun 2026, M4 Max PASS)** — `phase15_metal_signoff.sh` (5 steps: KV hook, multiseq, **continuous batch decode** via `phase15_batch_decode_smoke.sh`, L3 two-turn, tensor bind); `metal_signoff.sh` optional `RUN_E2E_PHASE15=1`. **Why batch step:** v27–v30 engine batch path must run on real Metal multiseq sidecar, not CPU pytest mocks. |
| **M6** | **MPS LoRA training + Mac operator polish** | Python + Go + CI | **Shipped** — PyTorch MPS + PEFT in `training.py`; QLoRA rejected on Darwin; `training_uv_venv.sh`; **`zerollama serve` Darwin bootstrap** (uv venvs, sidecar `:8081`, autoconfig); `zerollama doctor --json --fix`; Darwin CI (`macos-darwin-smoke`). **Extended (M14):** tiered clone bootstrap — see **M14**. |
| **M7** | **Upstream-shape GGUF benchmark (Metal)** | Repo | **Done** — ggml Metal ~164 tok/s vs upstream Go→llama-server ~158 tok/s (`llama3.2:3b`, `num_ctx=4096`, 6 epochs, idle GPU). Keep ggml default; Phase 17 for mergeability. [phase17-llama-server.md](./phase17-llama-server.md) |
| **M8** | **ggml @ b9781 (real vendored tree)** | Repo | **Done (Jun 2026)** — 16 patches on `vendor/llama-cpp-b9781/`; `sync_vendor_llama.sh`; matches upstream Ollama v0.30.11 pin. [ggml-b9509-migration.md](./ggml-b9509-migration.md) |
| **M9** | **Metal operator sign-off (Jun 2026)** | Repo | **Done (M4 Max)** — `./scripts/metal_signoff.sh` (Phase 13–15 + optional qwen35). **Jun 2026 full gate PASS** with `RUN_E2E_QWEN35=1 RUN_E2E_QWEN35_MODEL=eliza-1-2b:latest` (canonical ship tag; `qwen3.6:latest` also valid). **Why eliza-1 for the gate:** same qwen35 family as production Eliza bundles — 2B is fast enough for handoff/resume without pulling a separate upstream tag. **Gaps fixed:** v1 SSE + proxy flush; darwin ggml policy; `num_gpu=0` Metal gate; bootstrap discovery; sched_reserve; sign-off order (qwen35 before Phase 15); Phase 15 multiseq + `ZEROLLAMA_GPU_PROFILE=0`; L3 `cache_prompt` on inprocess workers; **`smoke_runtime_assert_kv_snapshot`** accepts linked **`bound`+`tensor`** (vendor kv-ext), not only `partial`+`seq_position`. |
| **M10** | **Qwen 3.5/3.6 GGUF on Mac** | Go + llama/compat + ggml Metal | **Done (Jun 2026, M4 Max)** — Go **ollama-engine** on Metal for `qwen35moe` after `sched_reserve` fix (graph intermediates defer to scheduler; KV buffer contexts use `Persistent()`); darwin no longer forces legacy llamarunner for `qwen35*`; in-process compat CGO link for llama-server/legacy; Metal embed regen in `build_zerollama_mac.sh`; `PrimaryFamily()` for VL manifests (projector-only → `""`); qwen35 `flushDoneEvents`; LM Studio MLX disk checks; opt-in `./scripts/qwen35_mac_smoke.sh` (thinking models: accept `thinking` when `response` empty); **unified Mac build** — `build_zerollama_mac.sh` + `build_mlx_dylibs_mac.sh` with `BUILD_MLX=auto`. **Sign-off:** full `metal_signoff.sh` + qwen35 via **`eliza-1-2b:latest`** or **`qwen3.6:latest`** (41/41 GPU layers on 3.6). **Why M10 exists:** published GGUF metadata differs from llama.cpp-native; stale `ggml-metal-embed.metal` broke first decode (sigmoid/unary); runtime/ggml dual-Metal contention on Darwin. **Not done:** full 27B Q8 VL on unified memory; qwen35 in default CI (opt-in smoke only). Doc: [qwen35-apple-silicon.md](./qwen35-apple-silicon.md). |
| **M11** | **GPU bootstrap discovery on Mac** | Go | **Done (Jun 2026)** — `DiscoverBackendDevices()` + ollama-engine `/info` no longer uses zero-layer dummy load (which set `GGML_DISABLE_METAL` via `sync.Once`). **Phase 17 add-on:** `discover/llama_server.go` for Linux auto / `ZEROLLAMA_LLAMA_SERVER=1`; Mac default stays ggml runner fallback. **Why:** operators saw `total_vram=0` and CPU-only layer layout while inference subprocesses still logged Metal; scheduler trusted empty discovery. Doc: [apple-silicon-metal.md](./apple-silicon-metal.md#gpu-bootstrap-discovery-jun-2026). |
| **M12** | **Scheduler unload + manifest `num_ctx` clarity** | Go + docs | **Done (Jun 2026)** — `expireRunner` always queues unload + `findLoadedRunner` name fallback; `/api/create` evicts warm runners; API surfaces prompt truncation; docs for **load-time KV vs request `options.num_ctx`**; **ggml VRAM suggest + opt-in clamp** (`ZEROLLAMA_GGML_CLAMP_NUM_CTX`, `/api/show` `ggml_num_ctx`, load responses when clamped). **Why:** create updated manifest but `/api/ps` stayed at old ctx; stop returned success while model stayed loaded; manifest `num_ctx: 262144` hung ggml load; high-VRAM tier default (262144) needs operator guidance without silent clamp; total VRAM must not stand in for free VRAM in suggest. **Audit:** `merged_num_ctx` vs `num_ctx` field split on show; 2s free-VRAM cache for show; load path refreshes via loaded runners. Doc: [qwen35-apple-silicon.md](./qwen35-apple-silicon.md#manifest-num_ctx-vs-request-optionsnum_ctx-jun-2026), [scheduling-vram-policy.md](./scheduling-vram-policy.md#ggml-vram-suggest-and-opt-in-clamp-m12-jun-2026). |
| **M13** | **L1 GPU profiles (Metal tiers)** | Python | **Done (Jun 2026, M4 Max)** — RAM-tier JSON (`apple_silicon_16g` … `128g`), `gpu_profiles.py`, `/health.gpu_profile`. **Why:** unified memory needs different `-np`/batch than CUDA discrete VRAM; one conservative profile left 128 GiB machines under-utilized. Sign-off: `./scripts/m3_metal_signoff.sh`. Doc: [gpu-profiles-l1.md](./gpu-profiles-l1.md). |
| **M14** | **Portable Mac dev bootstrap (any checkout)** | Repo + docs | **Done (Jun 2026)** — `dev_bootstrap.sh`, `ensure_llama_cpp_sibling.sh`, `mac_setup.sh` tier 0 defaults (sign-off off, auto-clone `../llama.cpp`), `build_llama_server.sh` sibling path fix, port table (`:11434` daily vs `:8080` CI). **Why:** fresh clones failed without operator-specific `Sites/inference` layout, manual llama.cpp clone, or pre-pulled models; sign-off in default `mac_setup` blocked onboarding. Doc: [mac-dev-setup.md](./mac-dev-setup.md). **`doctor --fix`** runs `ensure_llama_cpp_sibling.sh` before Metal build. |
| **M15** | **MLX agent prompt hardening** | Go + mlxrunner + docs | **Done (Jun 2026)** — bogus HF `text_config.max_position_embeddings` fix; `capMLXScheduleOptions` + tail truncate; `PromptTokens` single-tokenize passthrough; tokenize LRU; 30m MLX keep-alive floor; SSE keepalive during prefill; reload/prefill/prompt-size operator logs. **Why:** agent megaprompts (131k tokens) caused multi-minute prefill, cold reload every 5m, double tokenize, empty client streams; Gemma4 config exported vocab_size as ctx. Doc: [mlx-agent-prompts.md](./mlx-agent-prompts.md). |
| **M15a** | **MLX agent live-session + restore** | mlxrunner + server | **Done (Jul 2026)** — `tryExtendLiveSession` LCP + gen-token rewind; **`rewindCachesViaSnapshots`** + **`bestRestorableOffset`** for rotating KV; same-branch trie restore + snapshot fallback; **`tunePrefillConfig`** hot tails; **1× sliding_window** trie snapshots; prompt-chain invalidate on **`messages_dropped`**; message fingerprint on equal count; **`fast_path` / `same_branch`** agentstats; **MLX sidecar defer** (unkeyed `/api/generate` waits while agent hot). **Why:** turn 2 hit 99% trie cached but ~75s page-in; `fast_path` absent without snapshot rewind on wrapped OptiQ; turn 3+ collapsed to ~16k when message drops rewrote prefix; warn noise on sidecar `/api/generate`. **Requires:** stable `prompt_cache_key` every turn (Hermes `extra_body`). Doc: [mlx-agent-prompts.md](./mlx-agent-prompts.md#m15a-live-session--restore-jul-2026). |
| **M15b** | **Agent QoS + project tracking + cross-backend safety** | Go + clients + docs | **Done (Jul 2026)** — **`reserveScheduleQoS`** claims gate before runner wait (TOCTOU fix); **`project_id` / `project_name`** on `/api/ps` + **`zerollama ps`**; **`inference_path.go`** backend branching (GGUF preserves client keys; unkeyed GGUF no-op; eliza gated); MLX subprocess exit logging; **`GET /api/version`** `runner_paths` + `session_qos_gate`; Hermes/ruby-trivia/simpleagent Tier 2 ladder. **Why:** Jul 4 concurrent streams crashed MLX; operators couldn't see harness ownership; optimizations must not slow vanilla Ollama/vLLM/CUDA stacks. Doc: [agent-qos-and-project-tracking.md](./agent-qos-and-project-tracking.md). |
| **M16** | **Flash-MoE (anemll) via llama-server** | Go + fork build | **Partial (Jun 2026)** — `--moe-*` passthrough, Modelfile options, `build_flash_moe_llama_server.sh`, **`flash_moe_smoke.sh` tier 0–2**, doctor check, envconfig; **why:** Qwen3.5-class MoE exceeds unified RAM; ggml Metal path cannot slot-bank stream from SSD. **Open:** `pull` sidecar extract, vendor pin merge. Doc: [flash-moe.md](./flash-moe.md). |
| **M17** | **ANE probe (maderix/ANE)** | Repo + subprocess | **Partial (Jun 2026)** — `tools/ane-probe`, `libane_bridge` smoke, doctor + hidden CLI; **why:** validate ANE reachability before hybrid inference research without linking private APIs into Go. Doc: [ane-probe.md](./ane-probe.md). |
| **M18** | **ANE in-process dflash draft (B1–B7, P1–P11 matmul)** | llama-common + discover | **Partial (Jul 2026, lab)** — P1–**P3** blk.0 FFN on ANE (`shadow_hidden_cos=1.0`). **P7–P9** blk.1 gate/up/down use **host fp32**. **P9** chain 10 — lab `golden_cosine=1.0`. **P7b/B8** chain 8 — **2b proxy** `golden_cosine=1.0`; **27b native** chain 8 **`golden_cosine=0.9854`** (real `dflash_fc` + per-layer export handoff). **P10** chain 11 — **2b** `golden_cosine=1.0`; **27b native** chain 11 **`golden_cosine=1.0`** (`dflash_fc` + `dflash_hidden_norm` + `blk.0.attn_q`). **Open:** attn/KV, lm_head for token parity (`shadow_match_pct`). Doc: [ane-draft-inprocess.md](./ane-draft-inprocess.md). |
| **M19** | **Optional Mac MLX build (mlx-cgo)** | Repo + build | **Not started (eval only)** — optional **`OLLAMA_MLX_C_SOURCE`** fork [WhoseBiasDoYallSeek/mlx-cgo](https://github.com/WhoseBiasDoYallSeek/mlx-cgo) instead of upstream `ml-explore/mlx-c`. **Why consider:** Go `mlxgen` replaces Python mlx-c codegen (no `pip`); committed `mlx/c/` for normal builds; may avoid `go generate ./x/...` failures during cmake on newer Go. **Not default:** ship path stays upstream pins + `build_mlx_dylibs_mac.sh`; **5080 CUDA / imagegen MLX unchanged** (mlx-cgo is Apple Silicon–only). **Runtime unchanged:** zerollama still uses `libmlx`/`libmlxc` dylibs + mlxrunner subprocess — not static CGo embed. **Open:** pin + smoke vs upstream; `libjaccl` install in cmake bundle; fork drift tracking. See [Optional Mac MLX build path](#optional-mac-mlx-build-path-mlx-cgo). |

**Already optimized (Go, shipped):** Metal ggml runner, scheduler unified-memory behavior, Phase 8 broker with runtime embed.

**Mac onboarding (tiers 0–3):** **`./scripts/dev_bootstrap.sh`** → `./zerollama serve` → `pull` → optional `MAC_SETUP_SIGNOFF=1` — [mac-dev-setup.md](./mac-dev-setup.md). **Why separate from M9 sign-off:** tier 0 must succeed with zero pulled models and zero manual sibling clones.

**Mac operator default (why not legacy ggml):** `apple_silicon.yaml` → **`llama_backend: inprocess`**; Go proxy pulled tags need **`X-Zerollama-Runtime: 1`** or **`RUN_E2E_PHASE14=1`** in smokes — otherwise manifest names route to ggml and contend with the runtime sidecar on one Metal device.

**Not goals:** Replacing ggml Metal with MLX for all GGUF; NVML on Mac; duplicating `gpu_5080_session` on Darwin.

### Optional Mac MLX build path (mlx-cgo)

**Status:** evaluation only — **not** wired into CI, production Mac builds, or default `ensure_mlx_sources.sh`.

**What it is:** [mlx-cgo](https://github.com/WhoseBiasDoYallSeek/mlx-cgo) is a fork of upstream [mlx-c](https://github.com/ml-explore/mlx-c) that replaces the Python code-generator with a Go binary (`cmd/mlxgen`) and ships committed generated C headers under `mlx/c/`. Normal fork builds do not require Python or mlx-c regen.

**Why it might matter here:**

| Pain today (upstream mlx-c) | mlx-cgo claim |
|-----------------------------|---------------|
| cmake configure runs `go generate ./x/...` (can fail on newer Go, e.g. import-cycle noise) | Committed `mlx/c/`; regen optional via `./scripts/regenerate.sh` |
| Python mlx-c generator in the build loop | Go-only `mlxgen` |
| Stale `dist/.../mlx_metal_v*/` missing deps (e.g. `libjaccl.dylib`) | Same cmake install path — still need `jaccl` in `RUNTIME_DEPENDENCIES` bundle |

**What does *not* change if we adopt it as `OLLAMA_MLX_C_SOURCE`:**

- **Runtime architecture** — safetensors still go through **mlxrunner subprocess** + dynamic `libmlxc.dylib` (`x/mlxrunner/mlx/dynamic.go`), not static CGo link into `zerollama`.
- **Linux CUDA MLX** — imagegen and 5080 MLX stay on upstream `ml-explore/mlx` + `mlx-c` pins; mlx-cgo README is Apple Silicon only.
- **GGUF default** — Metal ggml remains default for pulled GGUF; MLX is still `ModelFormat: safetensors` only ([mlx-routing-policy.md](./mlx-routing-policy.md)).

**How to try (operator / dev, unsupported):**

```bash
git clone https://github.com/WhoseBiasDoYallSeek/mlx-cgo.git ../mlx-cgo
# align fork with MLX_VERSION / MLX_C_VERSION pins as needed
export OLLAMA_MLX_C_SOURCE="$PWD/../mlx-cgo"
INSTALL_PREFIX=dist/darwin-arm64 BUILD_MLX_V4=0 ./scripts/build_mlx_dylibs_mac.sh
./zerollama doctor   # expect [ok] mlx engine
```

**Exit criteria before promoting from “optional” to “supported”:**

1. Metal v3 dylib smoke (`doctor`, `mlxrunner` load, one safetensors model e.g. `ornith-9b-optiq`) **PASS** vs upstream mlx-c at the same pin.
2. Documented pin policy (fork commit tracked beside `MLX_C_VERSION`).
3. No regression on 5080 CUDA MLX build (still upstream sources only).
4. cmake configure no longer requires fragile `go generate ./x/...` when using the fork as C source **or** upstream path is fixed independently.

**Borrowings track:** Inference speed first — [Local voice & llama borrowings](#local-voice--llama-borrowings-eliza-v3) **L1–L3** (GPU profiles, fork kernels, KV prefix cache); voice **L5–L8** deferred.

---

## Intel Arc A380 track (Vulkan / 6 GB)

**Why a separate track:** CUDA 5080 and Mac Metal playbooks do not apply to **Mesa ANV + 6 GB GDDR6**. Research lane [`~/bmtl/asm_lab/lanes/arc-a380`](../../bmtl/asm_lab/lanes/arc-a380) documented per-request **`load_ms` ~580 ms**, partial **`num_gpu` cliff**, and integer-dot instability — operators need env pins, honest **`total_duration_eval_tok_s`**, and **local image gen** that fits VRAM without MLX.

**Guide:** [a380-runbook.md](./a380-runbook.md) · **Image:** [sd-vulkan-a380.md](./sd-vulkan-a380.md), [sd-openvino-a380.md](./sd-openvino-a380.md)

| Milestone | Goal | Owner | Exit criteria |
|-----------|------|--------|----------------|
| **A1** | **Vulkan GGUF chat** | Repo + vendor | **Done** — `OLLAMA_VULKAN=1`, `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`, vendor llama-server, `arc-a380` L1 profile; sign-off `a380_signoff.sh` |
| **A2** | **SD 1.5 via sd.cpp (Vulkan)** | Repo | **Done** — `external-image` hook, `sd15-vulkan` / Q8 / turbo / SDXL manifests, `install_stable_diffusion.sh`; **`diffusion_fa`** for ANV |
| **A3** | **SD via OpenVINO GenAI** | Repo | **Done** — `openvino-image` backend, `sd15-openvino` / `sdxl-openvino`, per-manifest `external_image_bin`; venv under `/usr/share/zerollama/openvino-genai/` |
| **A4** | **Image in `ls` + bench PERF** | Go | **Done** — `zerollama ls image`; PERF seconds for image tags; `bench` image cap 2 epochs + min-epochs clamp; `ModelsSearchDirs()` for service model roots |
| **A5** | **Production systemd stack** | Repo | **Done** — `build_zerollama_a380.sh`, `install_a380_llama_server.sh`, `zerollama-a380.service`, `/etc/zerollama/a380-llama.env` |

**Not goals:** CUDA training on Arc; MLX imagegen; SYCL/Level Zero as zerollama default backend; cloud image billing in local-only deployments.

---

## Local voice & llama borrowings (eliza-v3)

**Why a separate track:** [eliza-v3](https://github.com/elizaos/eliza) (`plugin-local-inference`, `elizaOS/llama.cpp`) ships a **fused on-device stack**—custom llama.cpp kernels, per-GPU autotune, and a duplex voice graph (ASR → MTP LLM → chunked TTS with barge-in). Zerollama already covers **OpenAI `/v1/audio/*`** via **Piper + Whisper subprocesses** ([multimodal-backends.md](./multimodal-backends.md)) and **MTP/ngram** in the Python runtime ([SPECULATIVE.md](../runtime/docs/SPECULATIVE.md)). This track ports **patterns and data** that fit our Go + Python shape—**not** the Eliza-1 bundle catalog, Capacitor/AOSP mobile loaders, or device-bridge WebSocket layer.

**Priority:** **Inference speed first** (tok/s, long-ctx VRAM, prefix-cache hit rate, MTP acceptance). Voice UX milestones (**L5+**) follow once **L1–L3** are measured on ship hardware (5080 + M-series).

**Reference tree (local):** `~/Sites/eliza-v3/plugins/plugin-local-inference/` (runtime + voice), `native/configs/gpu/` (autotune JSON), `packages/shared/src/local-inference/gpu-profiles.ts`, submodule `native/llama.cpp` → **`elizaOS/llama.cpp`**.

**Relationship to other tracks:** Distinct from [Zerollama remote cloud (Eliza)](#zerollama-remote-cloud-eliza) (HTTP proxy to Eliza Cloud). Complements [Phase 13](#local-inference--actionable-phases) (autotune), [Phase 15](#phase-15--exit-criteria-partial) (slot/KV), [Phase 17](#phase-17--upstream-gguf-path-alignment-directional) (llama.cpp pin). [Apple Silicon](#apple-silicon--metal-track): **L1** adds Metal/unified-memory profile variants; **L2** may add Metal KV kernels.

### Tier A — inference speed (do first)

| Milestone | Goal | Owner | Exit criteria |
|-----------|------|--------|----------------|
| **L1** | **Per-GPU llama profiles (CUDA + Metal)** | Python | **Done (Jun 2026)** — **Apple:** RAM tiers; M4 Max 128g; `l1_metal_gate.sh`. **NVIDIA 5080:** `rtx-5080.json` (`n_parallel=2`, `batch_size=1024`, `ubatch_size=256`); **concurrent N=2 +~16–20%** on eliza-1 9B (`l1_cuda_full_gate.sh`, Jun 2026); single-stream **−5%** @ 8k (np=2 overhead — informational). Optional supernova GGUF re-run. Disable: `ZEROLLAMA_GPU_PROFILE=0`. Doc: [gpu-profiles-l1.md](./gpu-profiles-l1.md). |
| **L2** | **Fork KV profiles on unified `llama-server`** | Repo + C | **Partial (Jul 2026)** — kernels on ggml-org via **0026–0030** + **0067–0071**; pin `8f114a9b`. **Ship gate FAIL** for default tok/s. **VRAM opt-in** = TBQ (`FORK_PROFILE=vram`). **Speed** (QJL/Polar) runs on `8f` but **−48…−85%** decode @ 8k/27k — keep experimental. **0072:** `POST /cuda-graph/invalidate` + `llama_context_cuda_graph_invalidate` on vendor (was sibling-only). Doc: [gpu-profiles-l2.md](./gpu-profiles-l2.md). |
| **L3** | **Prompt cache key → slot bridge** | Go + Python | **Done (Jun 2026)** — pinned slots, subprocess + in-process RAM/disk, batch keys, `/health.llama_cache`. **vLLM spike (Jun 2026):** selective-retention policy (`prefix_cache_policy.py`) — SWA/hybrid GGUF classification, draft-spec disables `cache_prompt`+disk, subprocess `seq_pos` from timings + `GET /slots` fallback. **Decode graph invalidation (Jun 2026):** epoch + `llama_context_cuda_graph_invalidate` (in-process) + `POST /cuda-graph/invalidate` (subprocess llama-server); doc [decode-graph-invalidation.md](./decode-graph-invalidation.md). Smokes: `l3_cache_smoke.sh`, `l3_spec_cache_smoke.sh` (`L3_RUN_SPEC_CACHE=1` on full gate). **5080:** `l3_cuda_full_gate.sh` PASS on eliza-1 9B — 8k cached **−42%** vs no-cache; 27k cached **0.72s** vs **1.48s**. Disable: `ZEROLLAMA_LLAMA_CACHE=0`. Doc: [gpu-profiles-l3.md](./gpu-profiles-l3.md). |

**vLLM borrowings — spike closed (Jun 2026).** GGUF-first runtime only; no HTTP-to-vLLM.

| Taken | Deferred (non-goals) |
|-------|---------------------|
| SWA/hybrid prefix retention policy | Full RadixAttention ref-count block DAG across requests |
| Draft-spec × prefix-cache guards (eagle3/mtp/dflash) | Required vLLM/SGLang server |
| Subprocess slot `seq_pos` + `/slots` backfill | Model Runner V2 / PyTorch engine |
| Streaming parser delta tests (`<` in tool JSON) | Rust frontend |
| `l3_spec_cache_smoke.sh` policy gate | Remote LMCache / Mooncake / NIXL connectors |
| Pluggable `KVCacheSpec` (`kv_cache_spec.py`) | Per-slot CUDA graph **capture** (`DecodeGraphCache.lookup`) |
| Prefix cache trace replay (`ZEROLLAMA_PREFIX_CACHE_TRACE=1`) | Required vLLM/SGLang server |
| Spec × page bind validation (`kv/spec_bind.py`) | Required vLLM/SGLang server |
| In-process spec bind at decode (`_prepare_seq_for_decode`) | Per-slot CUDA graph **capture** |
| Decode graph epoch scaffold (`decode_graph_policy.py`) | Per-slot CUDA graph **capture** |
| `DecodeGraphCache` stub + global epoch in `graph_capture_key()` | Per-slot CUDA graph **capture** |
| Prefill/decode trace epochs + `graph_capture_key()` | Per-slot CUDA graph **capture** |
| `llama_cpp_probe.py` — sibling `../llama.cpp` CUDA graph flags | Remote object-store KV (LMCache/Mooncake) |
| **`llama_context_cuda_graph_invalidate` + epoch bump wiring** — ggml graph clear on KV slot invalidation (CUDA); in-process native/ctypes + subprocess `POST /cuda-graph/invalidate`; doc [decode-graph-invalidation.md](./decode-graph-invalidation.md) | Per-slot CUDA graph **capture** |
| vLLM Jun 2026: `drop_eagle_block` (draft RAM cache + drop-last-block) | |
| **Cross-slot Radix prefix share (v1 + v2)** — `radix_prefix_share.py`, L3-R2…R5, `POST /kv/seq-copy`, `l3_radix_prefix_smoke.sh`; doc [radix-prefix-share.md](./radix-prefix-share.md) | llama-level shared KV pages + NIXL/Mooncake blob pull |
| `cache_salt` tenant slot isolation | Cross-node KV VRAM donor (same process only today) |
| `ZEROLLAMA_PREFIX_CACHE_RETENTION_INTERVAL` (SWA sparse retention) | |
| **Hybrid KV coordinator** (`kv/hybrid_kv_coordinator.py`) + **hybrid Radix gate** (`radix_seq_copy_policy.py`, L3-R5) | attn+recurrent hybrid `seq_cp` live probe (LFM2) |
| **Hash-chained prefix block pool** + LMCache tier (`file://` + `redis://`, L3-R4) | |
| `l3_prefix_cache_trace_replay.sh` golden replay | |

**Suggested order (Tier A):** **L1** → **L3** (low friction, immediate wins) → **L2** (fork spike in parallel with L1 measurement; merge when gated).

### Radix v2 (L3-R) — product gaps

**Why a separate track:** [Cross-slot Radix v1](./radix-prefix-share.md) shipped donor→target KV seed for agent fleets (shared system prompt, different cache keys). v1 intentionally stops short of vLLM RadixAttention — operators need a published gap list and ordered milestones, not “Radix” marketing without scope.

| Milestone | Goal | WHY | Exit criteria |
|-----------|------|-----|---------------|
| **L3-R0** | **Radix v1 shipped** | L3 slot-per-key leaves duplicate prefills across keys | `l3_radix_prefix_smoke.sh` offline PASS; Mac `L3_RADIX_LIVE=1` PASS; doc [radix-prefix-share.md](./radix-prefix-share.md) |
| **L3-R1** | **5080 live Radix gate** | Cross-slot seed on ship CUDA hardware | **Done (Jun 2026)** — `L3_RADIX_LIVE=1` on eliza-1 9B @ CT 1564: donor **10.6s** → target **0.66s**, `radix_seed` 128 tok; row in [gpu-profiles-l3.md](./gpu-profiles-l3.md) |
| **L3-R2** | **Warm-target catch-up** | Agents sometimes warm a slot then extend prefix; v1 skipped when `seq_pos > 0` | **Done (Jun 2026)** — `verify_target_slot_prefix` + donor search past target-owned blocks; full seq-copy when donor matched > target `seq_pos` |
| **L3-R3** | **Ref-count block DAG** | v1 = one contiguous donor chain; vLLM shares blocks across arbitrary overlap | **Done (Jun 2026)** — multi-holder block entries, `release_slot_holders`, best-donor selection across overlapping slots |
| **L3-R4** | **Remote LMCache tier** | Local `file://` metadata only; fleet restarts / cross-node need blob paths | **Done (Jun 2026)** — `redis://` metadata backend (stdlib RESP); `ZEROLLAMA_LMCACHE_TTL_SEC`; hydrates block pool cross-node (blobs still local) |
| **L3-R5** | **Hybrid-memory Radix** | SWA/hybrid GGUF skipped all `seq_cp` in v1 | **Done (Jun 2026)** — `radix_seq_copy_policy`: Gemma-style hybrid allowed when copy ≤ SWA window; env `ZEROLLAMA_RADIX_HYBRID_SEQ_COPY=0`; attn+recurrent still operator-gated |

**Non-goals (Radix track):** HTTP-to-vLLM; required SGLang sidecar; replacing L3 slot pinning with global Radix-only scheduling.

**Track status (Jun 2026):** L3-R0…L3-R5 **complete** in-tree. Next Radix work is **physical** KV sharing (llama.cpp pages), **blob** federation (NIXL/Mooncake), and Go scheduler visibility — not another admission-layer milestone.

**Doc:** [radix-prefix-share.md](./radix-prefix-share.md) — architecture, env, product gaps, validation status.

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

**Cross-links:** Spec decode — [runtime/docs/SPECULATIVE.md](../runtime/docs/SPECULATIVE.md). CUDA — [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md) (build, serve, gates). Mac — [apple-silicon-metal.md](./apple-silicon-metal.md). Voice today — [multimodal-backends.md](./multimodal-backends.md). Remote clients — [runtime-embed.md](./runtime-embed.md#remote-clients-go-api-vs-embedded-runtime).

---

## LocalAI control-plane borrowings (Jun 2026)

**Why a separate subsection:** [eliza-v3 L1–L3](./ROADMAP.md#local-voice--llama-borrowings-eliza-v3) optimizes **throughput and KV reuse** in the runtime. [LocalAI](https://github.com/mudler/LocalAI) patterns optimize **daemon lifecycle and metadata**—orthogonal, both agent-facing. We adopt LocalAI’s control-plane ideas without gRPC backends or a model gallery.

**Guide:** [localai-borrowings.md](./localai-borrowings.md)

| Milestone | Goal | Owner | Exit criteria |
|-----------|------|--------|----------------|
| **LA1** | **Fast GGUF metadata** | Go | **Done** — `DecodeMetadata` / `LoadModelMetadata`; scheduler + show paths |
| **LA2** | **GGUF guess hooks** | Go | **Done** — create/show/load; capped `num_ctx`; parser guess; env kill-switch |
| **LA3** | **Scheduler watchdog** | Go | **Done** — LRU, VRAM reclaim, busy timeout, load coalescing, pull `singleflight` |
| **LA4** | **Concurrency groups** | Go | **Done** — manifest field + scheduler eviction before load |
| **LA5** | **Post-load metadata probe** | Go | **Done** — `/api/ps`, fleet `loaded_model_details` |
| **LA6** | **Fleet score + affinity** | Go | **Done** — `ScoreCandidates`, prefix cache, probe cache, `/internal/score` |
| **LA7** | **Manifest repair command** | Go | **Done** — `zerollama repair` rewrites params/config/template without re-download; [manifest hygiene](./localai-borrowings.md#manifest-hygiene-existing-tags) |
| **LA8** | **HF importer v0** | Go | **Done** — `huggingface://` / `hf://` pull → local manifest; optional `source` + local name |
| **LA9** | **Logprob score API** | Go + runtime | **Done** — `POST /api/score`; llamarunner, ollamarunner, llama-server backends |
| **LA10** | **Model bench cache** | Go | **Done** — `zerollama bench` + **PERF** in `ls` (tok/s or seconds); `~/.ollama/bench.json` keyed by digest; image/video_gen kinds; [bench-cache.md](./bench-cache.md) |

**Not goals:** `backend.proto` plugin zoo, NATS cluster, full gallery parity, replacing Phase 15/17 engines.

**Next candidates (upstream watch):** [localai-borrowings.md — LA11+](./localai-borrowings.md#candidates-la11--suggested-priority) — intelligent score router (LA11), fleet radix routing (LA13), resumable peer transfers (LA14).

**Relationship to Fleet F-track:** LA6 extends [F3 management node](./fleet-management.md) routing; LA5 feeds future capacity-aware scores; LA13 would extend F7 when cross-node prefix visibility lands (L3-R4).

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
| **F7** | **Filter-then-score routing** | Go | **Done (Jun 2026)** — `ScoreCandidates`, prefix-cache affinity, health probe cache, `POST /internal/score`, capacity weights from `loaded_model_details`; doc [localai-borrowings.md](./localai-borrowings.md). |

**Routing policy (directional):** Prefer **loaded model + lowest queue**; cold route only when SLA allows; management **assigns** node, never starts loads remotely.

**Relationship to Phase 11 / T6:** Single-GPU admission and training defer remain on each node. Fleet layer only **chooses** which node receives the request.

**Not goals:** Global preemption across nodes; Redis pull-queue as v1 requirement; reservation market with frequent cancel (penalty/backoff deferred).

---

## GPU training (fine-tuning)

**Shipped direction:** Go embeds **CPython** (`x/trainingworker/pyembed`), serves **`/api/train/*`** and legacy **TCP `:9500`** (newline JSON), and coordinates VRAM with the scheduler on CUDA OOM (pause new inference loads → unload runners → ack Python). Training logic stays in repo-root **`training.py`**, loaded in-process—not a second public listener on 9500.

**5080 / CUDA operator default (Jun 2026):** link **`zerollama` against `libpython3.11`** and one **`.venv-training`** on 3.11 (matches `runtime/.venv`). **Why:** single Python generation per host; duplicate 3.10 `venv-training/` trees wasted ~15 GiB and caused ABI footguns. Build: [`scripts/training_embed_build_env.sh`](../scripts/training_embed_build_env.sh). **Production serve:** [`scripts/serve_production_wrapper.sh`](../scripts/serve_production_wrapper.sh) → `~/bin/serve.sh` — **WHY not copy `serve_gpu_example.sh`:** `~/bin` breaks `_ROOT` resolution.

**Why this track exists:** Fine-tuning stacks (Transformers, PEFT, bitsandbytes) are Python-native; Ollama’s control plane is Go. Embedding Python keeps **public wire** in Go while avoiding a second process and gRPC for every deployment.

### Phases (training track)

| Phase | Goal | Exit criteria |
|-------|------|----------------|
| **T1** | **Proactive VRAM (with inference Phase 8)** | **Done** — broker before `load_model` and on inference/runtime paths; OOM path remains safety net. |
| **T2** | **Auth on `/api/train` and `:9500`** | Same threat model as main HTTP API. |
| **T3** | **Progress over HTTP** | SSE or WebSocket; reduce poll-only UX. |
| **T4** | **CPU CI smoke** | **Partial** — Go tests register `/api/train/*` and exercise `GET /api/train/status` (502 without embed is OK in CI). Full embedded-python health still needs integration. **Operator hygiene (Jun 2026):** `.venv-training` ABI must match `ldd zerollama` libpython; 5080 ships **3.11 embed + venv** via [`training_embed_build_env.sh`](../scripts/training_embed_build_env.sh) — see [gpu-training.md](./gpu-training.md#installing-python-deps-embedded-interpreter). |
| **T5** | **Native training (optional)** | Rust/libtorch only if Python embed becomes a bottleneck—default stay on PyTorch. |
| **T6** | **Unified queue policy (directional)** | **Partial** — idle-wait; `priority`; defer queue; **allowed window**; **cross-queue FIFO** (global tickets, Go↔Python mirror). Operator guide: [t6-unified-queue.md](./t6-unified-queue.md); smoke: `./scripts/e2e_t6_queue_smoke.sh`. **Next:** richer SLO classes, stricter training/inference class separation. |

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

## Image generation — MLX fast path + ComfyUI utility (partial)

**Why a separate track from Wan / VLM / MLX-only imagegen:** Text understanding (vision encoders), Wan T2V (minute-scale diffusion via training `run_script`), and **still-image generation** share almost no code. Inside still-image generation there are two jobs:

1. **Interactive T2I** on a small set of models → keep in-tree MLX (`x/imagegen`).
2. **Agent utility** (edit, ControlNet, LoRA, many HF DiTs) → orchestrate ComfyUI rather than reimplement DiTs in Go.

Mixing “port Qwen-Image to MLX” with “agent needs ControlNet” hid the real product decision: **maximum agent utility comes from Comfy graphs**, not from another MLX model family.

**Shipped (v0 — Jul 2026):**

- `modality_backends.image=comfyui` + `server/modality/comfyui` HTTP bridge; OpenAI `/v1/images/*` via existing generate middleware.
- Named workflows + `GET /api/image/workflows`; config-only `comfy/*` manifests; `PrepareForImageGen` before Comfy jobs.
- Worked-example workflow JSON for Qwen-Image(+edit), FLUX.1/2-dev, GLM-Image, Klein 9B — **operator-calibrated**, not bit-for-bit verified against every Comfy install.

**Operator guide:** [comfyui-image-backend.md](./comfyui-image-backend.md). Fast path stays [imagegen-zimage-turbo.md](./imagegen-zimage-turbo.md).

| Milestone | Goal | Why |
|-----------|------|-----|
| **v0** | Comfy bridge + agent options + discovery | **Done** — unlock utility without MLX DiT ports. |
| **v0.1** | Calibrate one golden workflow (e.g. Qwen-Image GGUF) on 5080 / ship GPU | Prove templates end-to-end; document exact custom nodes + filenames. |
| **v0.2** | Cancel /interrupt Comfy on client disconnect | Stop orphaned GPU jobs when agents abort polls. |
| **v0.3** | Optional `upscale` template + checkpoint override in `backend_paths` | Plan leftover; agents ask for upscale often. |
| **Later** | Async job shape (Wan-like) for multi-minute GLM/FLUX.2 | Sync HTTP is fine for demos; agents need job ids for minutes-long gens. |

**Non-goals:** shipping ComfyUI inside the Go binary; dropping MLX Z-Image/Klein; guaranteeing interactive latency for GLM-Image / FLUX.2-dev on 16 GB; replacing `external-image` (escape hatch remains).

## Option 2 — Narrow the gap without SGLang (in-tree over time)

**Execution checklist:** [video-parity.md](./video-parity.md) (reference workloads + parity matrix). **Shipped borrowings:** [sglang-multimodal-borrowings.md](./sglang-multimodal-borrowings.md).

**Goal:** Keep inference **inside** Ollama/zerollama (ffmpeg + vision runner) and **deliberately port or reimplement** the behaviors that matter, instead of depending on an HTTP proxy to SGLang.

**Why a separate “Option 2” track:** Many users want **native** inference only (no second server). Option 2 spells out **policy** (how frames are chosen), **representation** (how templates see images vs video), and **limits** (context, mllama) as separate layers—so we do not conflate “better ffmpeg” with “SGLang-style scheduling.”

**Reality check:** SGLang is a large Python serving stack; **100% behavioral parity on every model** is not a single milestone. This roadmap is about **closing the gap** where it matters for **your** models and workloads.

### Shipped (Jun 2026 — SGLang Tier 1)

| Item | Why |
|------|-----|
| Pooled `video_url` HTTP + remote body LRU | Repeat agent clips should not re-open TLS or re-download containers |
| Global + **session** ffmpeg expansion cache | ffmpeg cost dominates; session cache survives global LRU eviction when `prompt_cache_key` is set |
| Preprocessed `video_spans` skip path | Already-expanded clients must not decode twice |
| OpenAI `prompt_tokens_details` (image/video/audio + `cached_tokens`) | Operators and agents need modality + prefix-cache visibility |
| OpenAI `prompt_cache_key` on `/v1/chat/completions` | OpenAI-shaped clients must pin session expansion + L3 like `/api/chat` |
| Preflight: latest-user scoping for pre-expanded spans | Echoed multimodal history must not false-reject agent follow-ups |
| Capability checks on `video_spans` (not only raw `videos`) | Pre-expanded SGLang clients must not bypass vision model gates |
| Gemma4 span placeholders (HF path) | Models that group clips vs flat `[img-N]` frames |
| Inference access log `cached_prompt_tokens` | Fleet ops correlate L3 hits without parsing every stream chunk |
| Inference access log `image_tokens` / `video_tokens` / `audio_tokens` | Post-expand modality heuristics on `inference response out` (mirrors OpenAI `prompt_tokens_details`) |
| Smoke: `video_expand_cache_smoke.sh`, `video_agent_cache_smoke.sh`, `video_l3_agent_gate.sh`, `video_agent_infer_smoke.sh` | CI + agent-loop proof without GPU/VLM; optional L3 text + live VLM infer gates; preproc leg needs `VIDEO_AGENT_GO_LOG` |
| Optional `grid_thw` on `video_spans` | Preprocessed + native ffmpeg; layout cached in expansion LRU; **runner hints** on `llm.ImageData` when **client explicit**; **`mtmd_bitmap_set_grid_hint`** forward on M-RoPE (Qwen-VL) |
| `padded_input_ids` layout cache | Accept + cache in **session** expansion LRU; restore on agent turn 2 |
| `padded_input_ids` runner stub | Log + access log + `_debug_render_only`; Qwen3-VL HF partial consume |
| Pre-expanded layout session cache | Fingerprint `images`+`video_spans`; restore `padded_input_ids` on turn 2 |
| Qwen3-VL padded inject audit (tool turns) | Tool pseudo-user blocks excluded from splice; warn on mismatch; `deferred_multimodal_history` fallback |
| llama-server pretokenized truncate + media | Vision blocks kept intact when `num_ctx` forces middle discard |
| OpenAI SSE `finish_reason: cancelled` | Disconnect prefill abort visible to agents (not mapped to `stop`) |
| `precomputed_embedding` ingest (ollama-engine + llamarunner) | SGLang clients skip ViT when feature rows + `padded_input_ids` supplied; all native VLMs on ollama-engine |
| `processor_output` ingest (ollama-engine) | HF `pixel_values` + grid; skip PNG decode; single-tile for llama4/lfm2 |
| `enable_prefix_mm_cache` + session ViT overlay | SGLang flag + `prompt_cache_key` pin per-thread ViT embeds; warn if flag without key |
| Infer smoke: preproc + prefix-mm + vit-session + grid_thw legs | Operator proof for padded layout restore, prefix-mm hint, ViT session overlay, mtmd grid forward (`VIDEO_AGENT_INFER_*`) |

### Phase A — Decode and sampling policy

- Align **fps / stride / max frames** with reference behavior for target models (env + per-model options where needed).
- Expand **container support** (what ffmpeg accepts) and document **failure modes** (corrupt input, no keyframes).
- Optional: **deterministic** sampling (fixed seeds / fixed frame indices) for reproducible evals.

**Why first:** Native path quality is dominated by **how** you turn video into frames, not the HTTP boundary.

**Partial:** env + manifest `video_sampling`, structured Info logs, global/session caches (see borrowings doc).

### Phase B — Renderer and template semantics

- Where a model distinguishes **video spans** vs **unrelated images**, extend templates/renderers with **explicit placeholders** (not only a flat `[img-*]` list).
- Per **model family** (e.g. Qwen3-VL, others): document **expected** ordering and token layout.

**Why:** Frame-list semantics are a **ceiling** until templates express “N frames from one clip.”

**Partial:** `video_spans` metadata + Gemma4 HF placeholders; Qwen3-VL per-frame `<|vision_start|>…` when `!RenderImgTags` (SGLang parity); production `[img-N]` per frame for flat runner — **documented + tested** (`qwen3vl_video_test.go`).

### Phase C — Context and limits

- **Token-aware** budgeting: relate frame count to **effective vision tokens** and `num_ctx` so users get **actionable errors** before runtime blowups.
- Tune **mllama** / single-image constraints vs multi-frame video (clear errors or automatic downsample).

**Why:** SGLang’s stack does scheduling/budgeting; native path must encode **policy** explicitly.

**Partial:** `EstimateMultimodalTokens` + preflight for raw `videos` and pre-expanded `video_spans` (latest user only); capability on `video_spans`; mllama single-image preflight before ffmpeg; OpenAI `prompt_cache_key` on `/v1/chat/completions`; usage breakdown (heuristic, latest-user scoped).

### Phase D — Validation and regression

- **Golden tests:** small fixtures (short MP4/WebM) with **expected frame counts** / hashes after sampling.
- **Per-model smoke:** optional CI jobs when ffmpeg + GPU are available.

**Why:** Parity is proven by **tests**, not by matching another repo’s README.

**Partial:** unit tests for caches, token budget, preprocessed spans, policy golden tests (`video_policy_golden_test.go`); optional ffmpeg golden (`video_ffmpeg_golden_test.go`, skips without ffmpeg); agent two-turn test (`TestExpandVideosInChatRequest_agentSecondTurn`); OpenAI session test (`openai/video_agent_session_test.go`); Qwen3-VL span render tests; `./scripts/video_expand_cache_smoke.sh` + `./scripts/video_agent_cache_smoke.sh` (live E2E: `/api/chat` + `/v1/chat/completions`); `./scripts/video_agent_infer_smoke.sh` for live VLM + `cached_prompt_tokens` (`RUN_E2E_VIDEO_AGENT_INFER=1`; Mac ollama-engine uses input-cache hits; optional `VIDEO_AGENT_INFER_PREPROC=1` + `VIDEO_AGENT_GO_LOG` for padded layout restore; optional `VIDEO_AGENT_INFER_PREFIX_MM_WARN=1` for prefix-mm hint).

### Phase E — Optional subprocess bridge (still no SGLang HTTP proxy)

- If a **specific** decode or preprocessing step must stay in Python, a **narrow subprocess** contract (stdin/stdout or temp files) can wrap **only that step**, with Ollama owning scheduling and limits—different from proxying full chat.

**Why:** Sometimes parity needs **one** binary without adopting a second full server.

**Partial:** `ExternalVideoDecodeHook` for ffmpeg replacement only.

### Next steals (from SGLang, not scheduled)

- **Preprocessed metadata** — pretokenized layout **cache** keyed by session + video digest or pre-expanded fingerprint (runner consume). **Shipped (Jun 2026):** session LRU; Qwen3-VL (+ qwen25vl) + Gemma4 + mllama + Gemma3 + Llama4 + LFM2 + GLM-OCR + Mistral3 + **DeepSeek-OCR** runner inject on **ollama-engine** (Mac default), ggml llamarunner, and llama-server; multi-turn padded splice; tool-turn span fix (Qwen3-VL); `deferred_multimodal_history` on splice failure; llama-server vision-aware truncate; **`precomputed_embedding`** on all native ollama-engine VLMs + llamarunner embed chunks; **`processor_output`** on ollama-engine (deepseekocr SAM deferred).
- **Chunked prefill abort** — **Shipped (Jun 2026):** C `kv_decode_loop_abort_set/clear`; streaming + non-stream disconnect cancel (ctypes + llama-server HTTP close + wheel internal stream).
- **ViT / encoder cache** — **Partial (Jun 2026):** configurable initial slots + **auto-grow** to `OLLAMA_IMAGE_EMBED_CACHE_MAX` + **session overlay** per `prompt_cache_key` on llamarunner and **ollama-engine**; **`enable_prefix_mm_cache`** OpenAI/options compat + warn without session key; cross-request radix sharing deferred.
- **`grid_thw` → mtmd forward** — **Partial (Jun 2026):** client explicit `[1,H,W]` on `llm.ImageData`; **`mtmd_bitmap_set_grid_hint`** resize on M-RoPE; server ffmpeg estimates stay preflight-only (`GridTHWExplicit`); [mtmd-grid-thw-handoff.md](./mtmd-grid-thw-handoff.md).
- **Video agent infer E2E** — **Partial (Jun 2026):** `video_agent_infer_smoke.sh` proves turn-2 `cached_prompt_tokens`; optional preproc / prefix-mm / **vit-session** / **grid_thw** legs (`VIDEO_AGENT_INFER_*` + `VIDEO_AGENT_GO_LOG`); GPU host sign-off still operator-run.

### What this does *not* promise

- Automatic parity with **every** SGLang model and feature.
- Replacing **SGLang’s** distributed scheduler or custom kernels without equivalent work in Ollama.

## Cross-cutting / hardening

| Item | Phase hint | Why |
|------|------------|-----|
| **Per-GPU llama flags + MTP autoconfig** | Borrowings **L1**; Phase **13** | Tuned batch/parallel/draft on stock cache types—first measurable tok/s win |
| **Long-ctx KV quant kernels** | Borrowings **L2**; Phase **17** | TurboQuant / QJL / Polar—largest VRAM + decode win when fork merges |
| **Prefix cache → llama slots** | Borrowings **L3**; Phase **15** v1b | **Why:** dynamic slots discard KV each turn; stable keys → pinned `id_slot` + `cache_prompt` skip repeat prefill on agent threads. **Jun 2026:** vLLM-inspired SWA/draft-spec policy; subprocess `seq_pos` tracking; decode graph invalidate via in-process native API + subprocess `POST /cuda-graph/invalidate`. Doc: [gpu-profiles-l3.md](./gpu-profiles-l3.md), [decode-graph-invalidation.md](./decode-graph-invalidation.md). |
| Inference vs training **priority / idle policy** | Training **T6**, inference **Phase 11** | One GPU, many clients—documented target is queued work + policy, not “implicitly fair” |
| **Fleet management + warm routing** | Fleet **F2–F5** | Many nodes, many agents—thin orchestrator, status, mDNS; avoid scatter-gather on constrained GPUs |
| **Local voice latency + duplex** | Borrowings **L5**, **L7** (Tier B) | Phrase cache + streaming pipeline after inference baseline |
| **LocalAI control-plane borrowings** | **LA1–LA10** | Fast GGUF read, guess hooks, watchdog, concurrency groups, fleet score, repair, HF pull, logprob score API, **bench cache** — [localai-borrowings.md](./localai-borrowings.md) |
| **In-process ASR (Qwen3-ASR)** | Borrowings **L6** (Tier B) | ASR subprocess removal—not LLM tok/s |
| Eliza catalog / response mapping | Eliza follow-ups | Operator UX when local + cloud lists collide |
| Video Option 2 A–D | Video track | Native VLM quality without SGLang dependency; **Jun 2026:** Tier 1 + audit fixes shipped — [sglang-multimodal-borrowings.md](./sglang-multimodal-borrowings.md) |
| SSRF hardening | Security | High-assurance deployments |
| ffmpeg / video agent E2E | Video + hardening | `video_agent_cache_smoke.sh` expand-only live (`RUN_E2E_VIDEO_AGENT=1`); `video_agent_infer_smoke.sh` full infer + `cached_prompt_tokens` (`RUN_E2E_VIDEO_AGENT_INFER=1`; `VIDEO_AGENT_INFER_PREPROC=1` / `VIDEO_AGENT_INFER_PREFIX_MM_WARN=1` need `VIDEO_AGENT_GO_LOG`); `video_l3_agent_gate.sh` bundles unit + optional L3/infer |

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
