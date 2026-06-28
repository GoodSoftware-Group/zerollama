# RTX 5080 runbook — what to run (Jun 2026)

**Status (CT 1564, Jun 28 2026):** **Full re-sign-off PASS** — tiers 1–4 (Phase 11–13 base, L1/L3, Phase 15, Phase 16/17 individual smokes). Only **Radix cross-slot live** (tier 2b) and **L2 fork merge** remain open / informational.

**Audience:** Operators on **16 GiB CUDA** hosts (RTX 5080-class, e.g. Proxmox CT 1564) validating zerollama after pull or before release.

**Mac counterpart:** [apple-silicon-metal.md](./apple-silicon-metal.md) — `./scripts/metal_signoff.sh`.

---

## Start here (three commands)

**Why one path:** env, build, and tier order were split across five docs. Use **`5080_env.sh`** + **`5080_resignoff.sh`** — everything else is optional deep dive.

```bash
# Inside CT 1564 (pct exec 1564 -- bash -lc '…')
cd ~/zerollama
git pull

source ./scripts/5080_env.sh          # PYTHONPATH, paths, RUN_E2E_PREFLIGHT=0
./scripts/5080_resignoff.sh --build   # rebuild + tiers 0→4 (stop on first FAIL)

# Radix cross-slot (after L3 same-key PASS) — needs vendor llama-server:
source ./scripts/5080_env.sh
./scripts/5080_resignoff.sh --tier 2 --radix --vendor --no-serve
```

| Script | Role |
|--------|------|
| [`scripts/5080_env.sh`](../scripts/5080_env.sh) | Source once — CT paths, GGUF, `PYTHONPATH`, helpers: `5080_start_serve`, `5080_build_patched_libllama`, `5080_build_vendor_llama_server`, `5080_setup_venvs` |
| [`scripts/5080_resignoff.sh`](../scripts/5080_resignoff.sh) | Full re-sign-off driver (`--tier N`, `--radix`, `--vendor`, `--build`) |
| [`scripts/gpu_5080_session.sh`](../scripts/gpu_5080_session.sh) | Tier 1 only (Phase 11–13 + snapshot) when serve already up |
| [`scripts/llama_patch_doctor.sh`](../scripts/llama_patch_doctor.sh) | Vendor / `/kv/seq-copy` before Radix live |
| [`scripts/serve_gpu_example.sh`](../scripts/serve_gpu_example.sh) | Production `OLLAMA_HOST=0.0.0.0:8080` template |

**Deep dives (only when debugging):** [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md) (VRAM, MLX, code map) · [testing-smoke.md](./testing-smoke.md) (full script catalog) · [gpu-profiles-l3.md](./gpu-profiles-l3.md) (L3/Radix policy)

---

## Before you run

### Proxmox / CT layout

**Why:** GPU passthrough and CUDA toolchains live in an **LXC** (e.g. CT **1564**, `cudallama`), not on the Proxmox host. Host `nvcc` 12.3 **cannot** compile `sm_120` for RTX 5080 — run all gates **inside** the CT.

```bash
# From Proxmox host
pct exec 1564 -- bash -lc 'cd /root/zerollama && …'
```

| Path (inside CT) | Role |
|------------------|------|
| `/root/zerollama` | Repo — use in all scripts below |
| `/root/llama.cpp` | Sibling llama.cpp (pin **b9781** + kv-ext patch **0014**) |
| `/root/Llama-OuteTTS-1.0-1B-Q8_0.gguf` | Tier 1 smoke GGUF (~1B Q8) |
| `/root/eliza-1-9b-256k.gguf` | Tier 2 L1/L3 production proxy |

### One-time host setup

| Step | Command / check | Why |
|------|-----------------|-----|
| **CGO build** | `CGO_ENABLED=1 go build -o zerollama .` | Go ggml + embed need CGO. |
| **cpp-httplib (minimal CT)** | `rsync -a ~/llama.cpp/vendor/cpp-httplib/ llama/llama.cpp/vendor/cpp-httplib/` then rebuild | **Symptom:** `fatal error: cpp-httplib/httplib.h`. Root `.gitignore` excludes `vendor/`; CT often lacks httplib. **Why `RUN_E2E_PREFLIGHT=0`:** skip Go golden when httplib missing; CI still runs `phase12_golden_ci.sh`. |
| **Linux link** | `-lstdc++` in `llama/llama.go` (not `-lc++`) | **Symptom:** `cannot find -lc++` on Debian CTs. |
| **llama-server** | `LLAMA_CPP_ROOT=../llama.cpp ./scripts/build_llama_server.sh` | Runtime subprocess + Phase 17 smokes need CUDA `llama-server`. |
| **Patched libllama (Tier 3)** | See [Build patched libllama](#build-patched-libllama-phase-15--sm_120) | Phase 15 linked `_kv_native` needs kv-ext symbols in `libllama.so`. |
| **Smoke GGUF** | 1B Q8 (e.g. OuteTTS) for base session | Fits 16 GiB; calibration evidence only until production re-run. |
| **Production GGUF (L1/L3)** | eliza-1 9B @ 8k/27k | L1 concurrent + L3 strict/production gates — not 1B smoke. |
| **Pulled tag** | `zerollama pull llama3.2:3b` | Phase 17 / edge smokes need a local manifest name. |
| **Runtime venv** | `RUNTIME_UV_SYNC=1 ./scripts/runtime_uv_venv.sh` | Embed needs `uvicorn` on `PYTHONPATH`; Phase 15 `_kv_native` build needs `setuptools>=75`. |
| **Training venv (optional)** | `./scripts/training_uv_venv.sh` | Embedded training when `OLLAMA_TRAINING=true`. |
| **NVML (Proxmox passthrough)** | `libnvidia-ml1` must match host kernel module | CT 1564: host **590.48.01** — if `nvidia-smi` reports driver/library mismatch: `nvidia-driver-pinning-590.48.01` + `libnvidia-ml1=590.48.01-1` (`--allow-downgrades`). |

**CT 1564 build + serve (after pull):**

```bash
source ./scripts/5080_env.sh
5080_setup_venvs          # first time only — runtime (+ training) venv
5080_build_zerollama      # or: ./scripts/5080_resignoff.sh --build
5080_build_sibling_llama_server
5080_pull_proxy_model     # llama3.2:3b for Phase 17 smokes
```

**Port cleanup:** `5080_stop_serve` (prefer over `pkill -f 'zerollama serve'` — can match the wrapping shell).

### Session env

**Do not copy export blocks from older docs.** Source once:

```bash
source ./scripts/5080_env.sh   # PYTHONPATH, GGUF paths, RUN_E2E_PREFLIGHT=0, serve helpers
5080_print_env                 # sanity check
```

Override any variable after sourcing (e.g. `export CUDA_LLAMA_MODEL=/path/other.gguf`).

**Remote clients:** production serve binds `OLLAMA_HOST=0.0.0.0:8080` — embedded runtime stays **`127.0.0.1:8081`**; remote clients use **`:8080` only**.

---

## Build patched libllama (Phase 15 / sm_120)

**Why:** Phase 15 in-process sign-off links `_kv_native` against `libllama.so` with **kv-ext** symbols. Stock sibling `llama.cpp` without patch **0014** fails `nm … llama_memory_kv_`. Host CUDA **12.3** cannot compile **sm_120** — install **cuda-nvcc-12-8** in the CT.

```bash
source ./scripts/5080_env.sh
5080_build_patched_libllama    # checkout b9781 + copy kv-ext from in-tree; cmake sm_120
```

**Pin check (no GPU):** `./scripts/phase15_llama_kv_ext_pin_check.sh` — run before a long CUDA rebuild.

**Vendor alternative:** `5080_build_vendor_llama_server` for Radix `/kv/seq-copy` (patch 0017); Phase 15 KV-ext on sibling is still the documented path for tier 3. See [phase15-llama-kv-ext-upstream.md](./phase15-llama-kv-ext-upstream.md).

---

## Start serve (required before Tier 1)

**Why:** `gpu_5080_session.sh` runs smokes against **already running** embed on `:8081` + Go on `:8080`. Stale listeners or missing `PYTHONPATH` cause false FAILs before any gate runs.

```bash
source ./scripts/5080_env.sh
5080_start_serve    # 5080_stop_serve + zerollama serve + /health wait
```

Production template (bind `0.0.0.0:8080`): [`scripts/serve_gpu_example.sh`](../scripts/serve_gpu_example.sh).

**VRAM prep during smokes:** scripts call `smoke_unload_ggml_runners` (API `keep_alive:0`) before runtime load — **why:** 503 before the broker can leave stale ggml runners holding VRAM. Do not `pkill` runners during gates.

---

## Tier 0 — sanity (no GPU)

**Why:** catch script drift and parser regressions before tying up the card.

```bash
./scripts/check_gpu_scripts.sh
./scripts/phase12_golden_ci.sh          # full dev host only; skipped when RUN_E2E_PREFLIGHT=0
./scripts/phase15_kv_native_ci.sh     # CPU Phase 15 pytest bundle
./scripts/phase15_upstream_kv_watch.sh  # upstream writable-bind symbol watch (no GPU)
./scripts/phase17_l2_pin_status.sh      # L2 pin report (no GPU)
```

---

## Tier 1 — official base gate (Phase 11–13)

**Why:** same role as Mac `phase11_13_15_metal_signoff.sh` for admission + VRAM + coordination — discrete NVML path, not unified memory.

```bash
kill $(pgrep -xo zerollama) 2>/dev/null || fuser -k 8080/tcp 8081/tcp 2>/dev/null || true
./scripts/gpu_5080_session.sh
```

**Pass:** `PASS: gpu_5080_session` + `/tmp/5080-session.json` (or `GPU_PHASE13_SNAPSHOT_OUT`).

**What runs inside:** Phase 12 preflight (unless `RUN_E2E_PREFLIGHT=0`) → `gpu_smoke_all` (coordination, VRAM prep, runtime e2e, health report) → Phase 13 snapshot → `python -m runtime.gpu_snapshot` hints.

**Requires:** serve running — see [Start serve](#start-serve-required-before-tier-1).

**Re-sign-off PASS (CT 1564, Jun 28 2026):** base session with `RUN_E2E_PREFLIGHT=0` after rebuild + NVML 590.48.01 fix + embed `PYTHONPATH`.

---

## Tier 2 — L1 + L3 production gates (borrowings)

**Why:** Phase 13 estimates *fit*; **L1** picks throughput knobs (`rtx-5080.json`: `n_parallel=2`, `batch_size=1024`); **L3** proves prompt-cache → slot bridge on agent-scale GGUF.

```bash
export CUDA_LLAMA_MODEL=/root/eliza-1-9b-256k.gguf   # 7B–9B production proxy

# Combined in one session:
RUN_E2E_PREFLIGHT=0 RUN_E2E_L1=1 RUN_E2E_L3=1 \
  CUDA_LLAMA_MODEL="$CUDA_LLAMA_MODEL" \
  ./scripts/gpu_5080_session.sh

# Or standalone:
./scripts/l1_cuda_full_gate.sh
./scripts/l3_cuda_full_gate.sh
./scripts/l3_production_gate.sh    # 27k ctx production verdict
```

| Gate | Status (CT 1564, Jun 28 2026) | Notes |
|------|-------------------------------|--------|
| **L1** single-stream | **PASS** | eliza-1 9B @ 8k: profile ON **+58%** vs OFF (88.7 vs 56.1 tok/s) |
| **L1** concurrent N=2 | **PASS** | +10.0% agg tok/s (63.5 vs 57.7) — **why `np=2`:** L3 needs ≥2 slots |
| **L3** 8k strict | **PASS** | cached turn2 faster than turn1 and no-cache control |
| **L3** 27k production | **PASS** | cached faster than no-cache; strict turn2/turn1 ratio may exceed 0.75 on 9B |
| **Radix cross-slot live** | **Pending (5080)** / **PASS (Mac)** | `RUN_E2E_L3_RADIX=1` or `L3_RADIX_LIVE=1 ./scripts/l3_radix_prefix_smoke.sh` — needs **vendor** llama-server (patch 0017). Mac Jun 2026: donor **8.2s** → target **0.58s**, `radix_seed` 128 tokens. 5080: run after L3 same-key PASS on eliza-1 9B. |

Optional spec-decode × L3 policy leg:

```bash
RUN_E2E_L3_SPEC=1 RUN_E2E_PREFLIGHT=0 RUN_E2E_L3=1 ./scripts/gpu_5080_session.sh

# Radix cross-slot (optional — after same-key L3 PASS; vendor llama-server required):
RUN_E2E_PREFLIGHT=0 RUN_E2E_L3_RADIX=1 CUDA_LLAMA_MODEL="$CUDA_LLAMA_MODEL" ./scripts/gpu_5080_session.sh
# Or folded into L3 gate:
RUN_E2E_PREFLIGHT=0 RUN_E2E_L3=1 RUN_E2E_L3_RADIX=1 CUDA_LLAMA_MODEL="$CUDA_LLAMA_MODEL" ./scripts/gpu_5080_session.sh
# Standalone:
CUDA_LLAMA_MODEL="$CUDA_LLAMA_MODEL" L3_RADIX_LIVE=1 ./scripts/l3_radix_prefix_smoke.sh
```

---

## Tier 3 — Phase 14 + 15 (in-process KV)

**Why:** Mac closes this with `./scripts/metal_signoff.sh`; CUDA uses **embed** path (`phase15_inprocess_signoff.sh`), not uv sidecar. Phase 15 needs **patched** `libllama.so` + `LLAMA_CPP_LIB`; multiseq smokes set **`ZEROLLAMA_GPU_PROFILE=0`** — **why:** L1 `-np 4` breaks `kv_inprocess_n_seq_max=2`.

### Phase 15 only (~10 min)

```bash
export LLAMA_CPP_LIB="$HOME/llama.cpp/build/bin/libllama.so"   # or vendor build with kv-ext
kill $(pgrep -xo zerollama) 2>/dev/null || fuser -k 8080/tcp 8081/tcp 2>/dev/null || true
./scripts/phase15_inprocess_signoff.sh
```

**Pass:** KV hook (`kv_decode_steps>0`, `batch_decode_in_c=True`), multiseq, batch decode via `/internal/generate-batch`, `smoke_runtime_assert_kv_snapshot` accepts **`bound`+`tensor`** when kv-ext linked.

**Pitfalls (CT 1564):**

- **Port 8081:** multiseq restarts serve — stale embed on `:8081` makes the new serve fail embed and the multiseq curl hangs. Free ports before sign-off.
- **`_kv_native` build:** if `pip install -e` fails on `tool.setuptools.ext-modules`, run `uv pip install --python runtime/.venv/bin/python 'setuptools>=75'` then `phase15_runtime_kv_ext_build` (or `PHASE15_BUILD_KV_EXT=0` when `.so` already built for your Python).

**Re-sign-off PASS (CT 1564, Jun 28 2026):** OuteTTS 1B Q8 — KV hook + multiseq + batch decode.

### Phase 14 + 15 full sign-off (~15–20 min)

```bash
export LLAMA_CPP_LIB="$HOME/llama.cpp/build/bin/libllama.so"
RUN_E2E_PREFLIGHT=0 RUN_E2E_PHASE14_SIGNOFF=1 ./scripts/gpu_5080_session.sh
```

**Why separate from Tier 1:** sign-off **restarts serve** per backend; folded into session only when explicitly flagged.

**Phase 14 note:** use **inprocess** for CUDA GPU — wheel GPU aborts on 5080 (`free(): invalid pointer`); see [phase14-inprocess-llama.md](./phase14-inprocess-llama.md).

---

## Tier 4 — Phase 16 + 17 upstream GGUF path

**Why:** Mac edge binary smoke PASS; **Linux CUDA** Phase 16/17 operator sign-off on ship hardware ([ROADMAP Phase 16 #4](./ROADMAP.md#phase-16--exit-criteria-partial)).

**Re-sign-off (CT 1564, Jun 28 2026):** individual P17 / Linux auto / edge smokes **PASS** (`/tmp/phase17-llama-server-smoke.json`, `/tmp/phase17-linux-auto-smoke.json`, `/tmp/phase16-edge-smoke.json`). Full `RUN_E2E_UPSTREAM_GGUF=1` bundle **may FAIL** on the bundled base runtime leg if fork cache types (`qjl1_256`) reach stock `llama-server` after L1/L2 — run individual smokes below, or set `ZEROLLAMA_GPU_PROFILE=0` before the bundle.

```bash
export LLAMA_SERVER_BIN="$HOME/llama.cpp/build/bin/llama-server"
export LD_LIBRARY_PATH="$(dirname "$LLAMA_SERVER_BIN"):${LD_LIBRARY_PATH:-}"
export RUN_E2E_PROXY_MODEL=llama3.2:3b
kill $(pgrep -xo zerollama) 2>/dev/null || fuser -k 8080/tcp 8081/tcp 2>/dev/null || true

# One flag — sets RUN_E2E_P17 + RUN_E2E_P17_LINUX_AUTO + RUN_E2E_EDGE (needs serve up for base leg):
RUN_E2E_PREFLIGHT=0 RUN_E2E_UPSTREAM_GGUF=1 ./scripts/gpu_5080_session.sh
```

**Recommended on 5080 (avoids fork-cache × stock llama-server clash):**

```bash
LLAMA_SERVER_BIN="$LLAMA_SERVER_BIN" P17_MODEL=llama3.2:3b ./scripts/phase17_llama_server_smoke.sh
LLAMA_SERVER_BIN="$LLAMA_SERVER_BIN" P17_MODEL=llama3.2:3b ./scripts/phase17_linux_auto_smoke.sh
P17_NUM_PREDICT=32 LLAMA_SERVER_BIN="$LLAMA_SERVER_BIN" P16_MODEL=llama3.2:3b ./scripts/phase16_edge_smoke.sh
```

**Individual smokes via session flags** (also valid; each restarts serve):

```bash
RUN_E2E_P17=1 ./scripts/gpu_5080_session.sh              # Go → llama-server generate
RUN_E2E_P17_LINUX_AUTO=1 ./scripts/gpu_5080_session.sh   # plain serve, backend.llama_server=auto
RUN_E2E_EDGE=1 ./scripts/gpu_5080_session.sh             # serve --edge, runtime chat off
./scripts/phase16_edge_build_smoke.sh                    # no GPU: -tags edge compile
```

**Pass criteria:** `phase17_llama_server_smoke` generate OK; `/api/status` shows `inference.backend` with `llama_server`; edge smoke: no `:8081` runtime chat; `GET /api/version` `edge_build` when using `-tags edge` binary. Edge smoke: if `empty response` at default `num_predict=8`, retry with `P17_NUM_PREDICT=32`.

**L2 pin merge (Phase 17 #7):** still **FAIL @ 8k** stock vs fork on CT 1564 — informational only; run `./scripts/l2_full_gate.sh` when evaluating fork profiles.

---

## Tier 5 — optional / periodic

| Goal | Command | Why |
|------|---------|-----|
| **Training ops** | `RUN_E2E_TRAINING_OPS=1` (serve needs `OLLAMA_TRAINING=true`) | Embedded training HTTP/TCP without blocking inference smokes |
| **Tools chat** | `RUN_E2E_TOOLS=1 ./scripts/gpu_smoke_all.sh` | Runtime `/api/chat` with tools — 501 means wrong route |
| **VRAM clamp** | `RUN_E2E_VRAM_CLAMP=1` + `ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX=auto` on serve | Opt-in clamp policy — default off in YAML |
| **L2 fork eval** | `./scripts/l2_full_gate.sh` | Compare stock vs eliza fork @ 8k; long-ctx fork-only legs optional |
| **Radix live (5080)** | `L3_RADIX_LIVE=1 ./scripts/l3_radix_prefix_smoke.sh` | Cross-slot prefix share — **WHY vendor:** patch 0017 `/kv/seq-copy`; fold via `RUN_E2E_L3_RADIX=1` on `gpu_5080_session.sh` |
| **Phase 17 vision** | `RUN_E2E_P17_VISION=1 P17_VISION_MODEL=llava:latest` | Heavy; opt-in projector smoke |
| **Decode graph** | Rebuild llama-server with `GGML_CUDA_GRAPHS=ON`; L3 smokes | CUDA graph invalidation on slot clear — [decode-graph-invalidation.md](./decode-graph-invalidation.md) |
| **Writable KV bind watch** | `./scripts/phase15_upstream_kv_watch.sh` | Phase 15 criterion #5 blocked upstream — no live decode required |

---

## Recommended full re-sign-off sequence

**Preferred:** one driver (after `git pull` inside CT 1564):

```bash
cd ~/zerollama
source ./scripts/5080_env.sh
./scripts/5080_resignoff.sh --build
# Optional Radix (vendor llama-server):
./scripts/5080_resignoff.sh --tier 2 --radix --vendor --no-serve
```

**Manual equivalent** (stop on first FAIL):

```bash
# 0. Env + one-time build (see sections above)
source ./scripts/5080_env.sh
export RUN_E2E_PREFLIGHT=0
5080_start_serve

# 1. Base Phase 11–13
./scripts/gpu_5080_session.sh

# 2. L1 + L3 production
RUN_E2E_L1=1 RUN_E2E_L3=1 ./scripts/gpu_5080_session.sh

# 3. Phase 15 KV (restarts serve inside script)
5080_stop_serve
./scripts/phase15_inprocess_signoff.sh

# 4. Phase 16/17 upstream (individual smokes)
LLAMA_SERVER_BIN="$LLAMA_SERVER_BIN" P17_MODEL=llama3.2:3b ./scripts/phase17_llama_server_smoke.sh
LLAMA_SERVER_BIN="$LLAMA_SERVER_BIN" P17_MODEL=llama3.2:3b ./scripts/phase17_linux_auto_smoke.sh
P17_NUM_PREDICT=32 LLAMA_SERVER_BIN="$LLAMA_SERVER_BIN" P16_MODEL=llama3.2:3b ./scripts/phase16_edge_smoke.sh
```

**Artifacts to keep:**

| File | Contents |
|------|----------|
| `/tmp/5080-session.json` | Phase 13 snapshot (`GPU_PHASE13_SNAPSHOT_OUT`) |
| `/tmp/l1-production-gate/` | L1 calibrate + concurrent bench (`gate.json`) |
| `/tmp/l3-cuda-full-gate/` | L3 8k smoke + 27k production (`gate.json`, `production-27k.json`) |
| `/tmp/phase17-llama-server-smoke.json` | Phase 17 `--llama-server-backend` smoke |
| `/tmp/phase17-linux-auto-smoke.json` | Phase 17 Linux auto routing smoke |
| `/tmp/l3-radix-prefix-smoke-live.json` | Radix cross-slot live (`L3_RADIX_OUT`) |

---

## `RUN_E2E_*` quick reference

| Flag | Script / effect |
|------|-----------------|
| `RUN_E2E_PREFLIGHT=0` | Skip `phase12_golden_ci` in session — **CT default** |
| `RUN_E2E_L1=1` | `l1_cuda_full_gate.sh` |
| `RUN_E2E_L3=1` | `l3_cuda_full_gate.sh` |
| `RUN_E2E_L3_RADIX=1` | `l3_radix_prefix_smoke.sh` live (vendor `/kv/seq-copy`; 9B+ GGUF) |
| `L3_RUN_RADIX=1` | Same, folded into `l3_cuda_full_gate.sh` |
| `RUN_E2E_PHASE14=1` | Phase 14 backend smoke (serve must match backend) |
| `RUN_E2E_PHASE14_SIGNOFF=1` | `phase14_5080_signoff.sh` — needs `LLAMA_CPP_LIB` |
| `RUN_E2E_PHASE15=1` | `phase15_inprocess_signoff.sh` — needs `LLAMA_CPP_LIB` |
| `RUN_E2E_P17=1` | Go → llama-server smoke |
| `RUN_E2E_P17_LINUX_AUTO=1` | Linux auto routing smoke |
| `RUN_E2E_EDGE=1` | Phase 16 edge smoke |
| **`RUN_E2E_UPSTREAM_GGUF=1`** | **Bundles P17 + LINUX_AUTO + EDGE** |
| `RUN_E2E_P17_VISION=1` | Vision llama-server smoke (heavy) |
| `RUN_E2E_TRAINING_OPS=1` | Training HTTP smoke |
| `RUN_E2E_TOOLS=1` | Tools chat smoke |

Full table: [testing-smoke.md](./testing-smoke.md).

---

## Status matrix (Jun 28 2026 re-sign-off, CT 1564)

| Track | Gate | Status |
|-------|------|--------|
| Phase 11–13 | `gpu_5080_session.sh` | **PASS** |
| L1 | `l1_cuda_full_gate.sh` | **PASS** (+58% single-stream, +10% concurrent @ 8k) |
| L3 | `l3_cuda_full_gate.sh` / production @ 27k | **PASS** |
| Phase 15 | `phase15_inprocess_signoff.sh` | **PASS** |
| Phase 14 | `phase14_5080_signoff.sh` | **PASS** (historical) |
| L2 @ 8k | `l2_full_gate.sh` | **FAIL merge** (stock wins — expected) |
| Radix live | `l3_radix_prefix_smoke.sh` | **Mac PASS** (Jun 2026); **5080 pending** — use `RUN_E2E_L3_RADIX=1` |
| Phase 17 P17 | `phase17_llama_server_smoke.sh` | **PASS** |
| Phase 17 Linux auto | `phase17_linux_auto_smoke.sh` | **PASS** |
| Phase 16 edge CUDA | `phase16_edge_smoke.sh` | **PASS** (`P17_NUM_PREDICT=32`) |
| `RUN_E2E_UPSTREAM_GGUF=1` bundle | full session wrapper | **Partial** — base runtime leg may fail after L1/L2 fork cache on stock `llama-server`; use individual smokes |
| Phase 17 L2 pin merge | criterion #7 | **Partial** |

**Not required on 5080:** `gpt-oss:20b` harmony real-weight (~40+ GiB host RAM); Mac `metal_signoff.sh`.

---

## Troubleshooting (CT 1564)

| Symptom | Likely cause | Fix |
|---------|----------------|-----|
| `cpp-httplib/httplib.h: No such file` | Minimal checkout, no vendor httplib | `rsync` from sibling `llama.cpp`; or `RUN_E2E_PREFLIGHT=0` for GPU-only gate |
| `cannot find -lc++` | Debian uses libstdc++ | Rebuild with `-lstdc++` in `llama/llama.go` LDFLAGS |
| `nvidia-smi` driver/library mismatch | CT userspace ≠ host kernel module | Pin `libnvidia-ml1=590.48.01-1` (CT 1564) |
| `ModuleNotFoundError: uvicorn` | Embed without runtime venv on `PYTHONPATH` | [Start serve](#start-serve-required-before-tier-1) env block |
| `/health` timeout / serve “failed to start” | Cold CUDA health ~9s; old 2s curl | Wait with `curl -m 15`; see `LINUX_RT_CURL_TIMEOUT` |
| 502/503 runtime smokes, ggml still in `/api/ps` | Broker never ran; stale runner | Smokes retry API unload (`keep_alive:0`); free ports and restart serve |
| Phase 15 multiseq hang | Stale `:8081` after serve restart | `kill $(pgrep -xo zerollama)` + `fuser -k 8080/tcp 8081/tcp` before sign-off |
| `_kv_native` / setuptools ext-modules fail | Old setuptools in runtime venv | `uv pip install --python runtime/.venv/bin/python 'setuptools>=75'` |
| `nm … llama_memory_kv_` empty | Unpatched libllama | [Build patched libllama](#build-patched-libllama-phase-15--sm_120) |
| Phase 15 `kv_inprocess_n_seq_max=4` not 2 | L1 profile overrides yaml slots | Multiseq smoke sets `ZEROLLAMA_GPU_PROFILE=0` — if manual serve, export it |
| `free(): invalid pointer` wheel GPU | pip wheel vs pinned libllama skew | Use **inprocess** for CUDA GPU, not wheel GPU |
| Edge smoke `empty response` | Default `num_predict=8` too short | `P17_NUM_PREDICT=32` |
| `RUN_E2E_UPSTREAM_GGUF=1` bundle FAIL | L1 fork cache types on stock `llama-server` | Individual P17/edge smokes; or `ZEROLLAMA_GPU_PROFILE=0` before bundle |
| `/health` hang with training + embed | Shared CPython GIL | `ZEROLLAMA_RUNTIME_SHARED_PYTHON=1`; see [shared-interpreter-health-hang.md](./bugs/shared-interpreter-health-hang.md) |

---

## After a green re-sign-off

1. **Production quant:** one real load so autotune persists under `~/.cache/zerollama/vram_autotune.json`.
2. **Optional clamp:** `ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX=auto` only if you accept automatic `num_ctx` lowering in API responses.
3. **Do not** copy smoke-only global `VRAM_ESTIMATE_FACTOR` when autotune persist is on.
4. **Phase 11 thresholds:** tune backlog env only under measured chat+training load — idle smoke proves fit, not contention policy.

---

## Related docs

- [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md) — build, serve, VRAM unload, troubleshooting
- [ROADMAP.md](./ROADMAP.md) — phase exit criteria + open items
- [phase16-thin-edge.md](./phase16-thin-edge.md) — `--edge` semantics
- [phase17-llama-server.md](./phase17-llama-server.md) — upstream GGUF alignment
- [apple-silicon-metal.md](./apple-silicon-metal.md) — Mac `metal_signoff.sh` counterpart
