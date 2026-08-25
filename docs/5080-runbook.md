# RTX 5080 runbook — complete operator guide (Jun 2026; **continue Jul 2026**)

**CUDA lane map (common vs 5080 vs dual 4090):** [cuda-lanes.md](./cuda-lanes.md)

**Status (CT 1564):** **Full re-sign-off PASS** (Jun 2026) — tiers 0–4 + Radix live + `RUN_E2E_UPSTREAM_GGUF=1` bundle. **L2 fork merge** remains informational (stock wins @ 8k — expected).

**Continue here (Aug 2026):** L1 + Phase 17 `auto` + embed attach shipped on CT 1564. **Spec A/B (eliza-1-2b @ 8k, L1 q8_0/−b 1024/−np 2):** baseline **~188–189 tok/s**; `ngram-simple` wires / **0 accepts**; DFlash/MTP need matching target (Qwen3.5-2B) or newer pin. **Radix agent lane (lab `:18081`):** `ZEROLLAMA_INFERENCE_PROFILE=agent` → `l3_agent_subprocess.yaml` (not Phase 17 Go alone — Radix is Python runtime L3). Live PASS beside prod: OuteTTS 1B @ 4k, donor **5.36s** → target **0.52s**, `radix_seed` **64**, `kv_unified` metadata (`/tmp/l3-radix-prefix-smoke-live.json`). Do **not** flip daily prod to `agent` (multi-slot / kv_unified tradeoffs); keep `auto`→throughput + `single_gpu.yaml`. Lab LD must include `/root/nvidia-host` (fixed in `linux_runtime_export_llama_ld_path`) or ctypes/`llama-server` abort `munmap_chunk`. Next: Phase 15 polish / RotorQuant no-merge.

**This is the only doc you need on CT 1564.** Build, serve, env, every gate, pass criteria, artifacts, and troubleshooting live here. Do not switch to [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md) for daily ops — it is a legacy appendix. Mac counterpart: [apple-silicon-metal.md](./apple-silicon-metal.md) + `./scripts/gpu/metal_signoff.sh`. Fork labs: [llama-fork-watchlist.md](./llama-fork-watchlist.md).

---

## Contents

1. [Start here (three commands)](#start-here-three-commands)
2. [Scripts and helpers](#scripts-and-helpers)
3. [`5080_resignoff.sh` tiers](#5080_resignoffsh-tiers)
4. [Before you run](#before-you-run)
5. [Build patched libllama (Phase 15)](#build-patched-libllama-phase-15--sm_120)
6. [Vendor llama-server (Radix)](#vendor-llama-server-radix--patch-0017)
7. [Start serve (sign-off)](#start-serve-required-before-tier-1)
8. [Production serve](#production-serve-binserve)
9. [Training embed + venvs](#training-embed--venvs)
10. [Tier 0–5 gates](#tier-0--sanity-no-gpu)
11. [`gpu_5080_session.sh` internals](#gpu_5080_sessionsh-internals)
12. [Phase 13 VRAM / snapshot](#phase-13-vram--snapshot)
13. [VRAM prep during smokes](#vram-prep-during-smokes)
14. [L1 profile (`rtx-5080.json`)](#l1-profile-rtx-5080json)
15. [L2 fork eval (informational)](#l2-fork-eval-informational)
16. [**Tier F — RotorQuant / post-L2 labs (Jul 2026)**](#tier-f--rotorquant--post-l2-labs-jul-2026)
17. [L3 + Radix criteria](#l3--radix-criteria)
18. [Phase 14 sign-off](#phase-14-sign-off)
19. [Harmony / host RAM](#harmony--host-ram-not-vram)
20. [Environment reference](#environment-reference)
21. [Full re-sign-off sequence](#recommended-full-re-sign-off-sequence)
22. [Status matrix](#status-matrix-jun-28-2026-re-sign-off-ct-1564)
23. [Troubleshooting](#troubleshooting-ct-1564)
24. [After green](#after-a-green-re-sign-off)
25. [Code map](#code-map)
26. [Optional: MLX imagegen](#optional-mlx-imagegen)

---

## Start here (three commands)

**Why one path:** env, build, and tier order were split across five docs. Use **`5080_env.sh`** + **`5080_resignoff.sh`** — everything else is optional deep dive.

```bash
# Inside CT 1564 (pct exec 1564 -- bash -lc '…')
cd ~/zerollama
git pull

source ./scripts/gpu/5080_env.sh          # PYTHONPATH, paths, RUN_E2E_PREFLIGHT=0
./scripts/gpu/5080_resignoff.sh --build   # rebuild + tiers 0→4 (stop on first FAIL)

# Radix cross-slot (after L3 same-key PASS) — needs vendor llama-server:
source ./scripts/gpu/5080_env.sh
./scripts/gpu/5080_resignoff.sh --tier 2 --radix --vendor --no-serve
```

| Script | Role |
|--------|------|
| [`scripts/gpu/5080_env.sh`](../scripts/gpu/5080_env.sh) | Source once — CT paths, GGUF, `PYTHONPATH`, helpers: `5080_start_serve`, `5080_build_patched_libllama`, `5080_build_vendor_llama_server`, `5080_setup_venvs` |
| [`scripts/gpu/5080_resignoff.sh`](../scripts/gpu/5080_resignoff.sh) | Full re-sign-off driver (`--tier N`, `--radix`, `--vendor`, `--build`) |
| [`scripts/gpu/gpu_5080_session.sh`](../scripts/gpu/gpu_5080_session.sh) | Tier 1 only (Phase 11–13 + snapshot) when serve already up |
| [`scripts/vendor/llama_patch_doctor.sh`](../scripts/vendor/llama_patch_doctor.sh) | Vendor / `/kv/seq-copy` before Radix live |
| [`scripts/serve/serve_gpu_example.sh`](../scripts/serve/serve_gpu_example.sh) | In-repo production env (`OLLAMA_HOST=0.0.0.0:8080`, PYTHONPATH, vendor llama-server) |
| [`scripts/serve/serve_production_wrapper.sh`](../scripts/serve/serve_production_wrapper.sh) | Install as `~/bin/serve.sh` — **WHY:** copying the example to `~/bin` breaks repo-root detection |

### `5080_env.sh` helpers (after `source`)

| Function | Purpose |
|----------|---------|
| `5080_print_env` | Sanity dump of paths, `PYTHONPATH`, `RUN_E2E_PREFLIGHT` |
| `5080_setup_venvs` | `runtime_uv_venv.sh` + optional `.venv-training` |
| `5080_setup_cuda` | Require `cuda-nvcc-12-8` / `CUDACXX` for sm_120 builds |
| `5080_build_zerollama` | rsync httplib if needed + `training_embed_build_env.sh` + `go build` |
| `5080_build_sibling_llama_server` | `build_llama_server.sh` on `~/llama.cpp` |
| `5080_build_patched_libllama` | b9781 + kv-ext copy + cmake sm_120 → `LLAMA_CPP_LIB` |
| `5080_build_vendor_llama_server` | Vendor pin build + `llama_patch_doctor.sh` (Radix `/kv/seq-copy`) |
| `5080_patch_doctor` | Probe patch 0017 on `:8082` |
| `5080_pull_proxy_model` | `zerollama pull` for Phase 17 tag if missing |
| `5080_stop_serve` | SIGKILL zerollama + `fuser` `:8080`/`:8081`/`:8082` |
| `5080_wait_health` | curl `:8081/health` up to 30×15s |
| `5080_start_serve` | stop + background serve + health wait |
| `5080_cd_repo` | `cd $Z5080_REPO` |

**Defaults set by sourcing:** `RUN_E2E_PREFLIGHT=0`, `OLLAMA_HOST=http://127.0.0.1:8080`, `unset ZEROLLAMA_RUNTIME_URL`, `PYTHONPATH=runtime/.venv + .venv-training`, prefer vendor `llama-server` when built, `Z5080_VENDOR_PIN=86d86ed4`.

### Daily ops (no full re-sign-off)

| Goal | Command |
|------|---------|
| Pull + smoke base only | `git pull && source ./scripts/gpu/5080_env.sh && 5080_start_serve && ./scripts/gpu/gpu_5080_session.sh` |
| Production serve | `~/bin/serve.sh` (wrapper → in-repo example); `tail -f /tmp/zerollama-serve.log` |
| Stop everything | `5080_stop_serve` |
| Health | `curl -s http://127.0.0.1:8081/health \| jq .` |
| Training status | `curl -s http://127.0.0.1:8080/api/train/status \| jq .` |
| Single tier | `./scripts/gpu/5080_resignoff.sh --tier 2 --no-serve` (serve must already be healthy) |

---

## `5080_resignoff.sh` tiers

| Tier | What | Serve profile |
|------|------|----------------|
| **0** | `check_gpu_scripts`, Phase 15 CI pytest, upstream KV watch, L2 pin report | — |
| **1** | `gpu_5080_session.sh` (Phase 11–13) | **`ZEROLLAMA_GPU_PROFILE=0`** — **why:** rtx-5080 fork QJL breaks 1B OuteTTS smoke |
| **2** | L1 + L3 via session; optional **`--radix`** | **L1 profile ON** (`n_parallel=2`) |
| **3** | `phase15_inprocess_signoff.sh` | restarts serve inside script |
| **4** | P17 + Linux auto + edge individual smokes | stopped/restarted per smoke |
| **5** | Printed hints only (L2, training ops) | — |

**Flags:** `--build` (zerollama + sibling llama-server + pull tag), `--vendor` (vendor llama-server + patch doctor), `--radix` (after tier 2), `--no-serve` (tiers 1–2 only, serve must be healthy), `--tier N` (single tier).

**WHY profile-off on tier 1:** embedded runtime reads L1 profile at serve start; `qjl1_256` on stock path breaks 1B smoke; tier 2 needs real `rtx-5080` profile for L3 `-np 2`.

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
| **llama-server** | `LLAMA_CPP_ROOT=../llama.cpp ./scripts/build/build_llama_server.sh` | Runtime subprocess + Phase 17 smokes need CUDA `llama-server`. |
| **Patched libllama (Tier 3)** | See [Build patched libllama](#build-patched-libllama-phase-15--sm_120) | Phase 15 linked `_kv_native` needs kv-ext symbols in `libllama.so`. |
| **Smoke GGUF** | 1B Q8 (e.g. OuteTTS) for base session | Fits 16 GiB; calibration evidence only until production re-run. |
| **Production GGUF (L1/L3)** | eliza-1 9B @ 8k/27k | L1 concurrent + L3 strict/production gates — not 1B smoke. |
| **Pulled tag** | `zerollama pull llama3.2:3b` | Phase 17 / edge smokes need a local manifest name. |
| **Runtime venv** | `RUNTIME_UV_SYNC=1 ./scripts/runtime/runtime_uv_venv.sh` | Embed needs `uvicorn` on `PYTHONPATH`; Phase 15 `_kv_native` build needs `setuptools>=75`. |
| **Training venv + embed** | `sudo apt install python3.11-dev`; `source ./scripts/training/training_embed_build_env.sh 3.11 && CGO_ENABLED=1 go build -o zerollama .`; `TRAINING_UV_PYTHON_VER=3.11 ./scripts/training/training_uv_venv.sh --verify` | **WHY 3.11 on 5080/CT 1564:** runtime `.venv` is already 3.11; default `python3-embed` on Ubuntu 22.04 is 3.10 — without `training_embed_build_env.sh` the binary and venv diverge. **Production:** `cp scripts/serve/serve_production_wrapper.sh ~/bin/serve.sh` (not a verbatim copy of `serve_gpu_example.sh`). |
| **Legacy 3.10 cleanup** | After both binary and `.venv-training` are 3.11: `rm -rf venv-training/ .venv-training-py310.bak` | **WHY:** duplicate torch stacks ~7 GiB each; legacy `venv-training/` is ignored by scripts. See [gpu-training.md](./gpu-training.md#installing-python-deps-embedded-interpreter). |
| **NVML (Proxmox passthrough)** | `libnvidia-ml1` must match host kernel module | CT 1564: host **590.48.01** — if `nvidia-smi` reports driver/library mismatch: `nvidia-driver-pinning-590.48.01` + `libnvidia-ml1=590.48.01-1` (`--allow-downgrades`). |

**CT 1564 build + serve (after pull):**

```bash
source ./scripts/gpu/5080_env.sh
5080_setup_venvs          # first time only — runtime (+ training) venv
5080_build_zerollama      # or: ./scripts/gpu/5080_resignoff.sh --build
5080_build_sibling_llama_server
5080_pull_proxy_model     # llama3.2:3b for Phase 17 smokes
```

**Port cleanup:** `5080_stop_serve` (prefer over `pkill -f 'zerollama serve'` — can match the wrapping shell).

### Session env

**Do not copy export blocks from older docs.** Source once:

```bash
source ./scripts/gpu/5080_env.sh   # PYTHONPATH, GGUF paths, RUN_E2E_PREFLIGHT=0, serve helpers
5080_print_env                 # sanity check
```

Override any variable after sourcing (e.g. `export CUDA_LLAMA_MODEL=/path/other.gguf`).

**Remote clients:** production serve binds `OLLAMA_HOST=0.0.0.0:8080` — embedded runtime stays **`127.0.0.1:8081`**; remote clients use **`:8080` only**.

---

## Build patched libllama (Phase 15 / sm_120)

**Why:** Phase 15 in-process sign-off links `_kv_native` against `libllama.so` with **kv-ext** symbols. Stock sibling `llama.cpp` without patch **0014** fails `nm … llama_memory_kv_`. Host CUDA **12.3** cannot compile **sm_120** — install **cuda-nvcc-12-8** in the CT.

```bash
source ./scripts/gpu/5080_env.sh
5080_build_patched_libllama    # checkout b9781 + copy kv-ext from in-tree; cmake sm_120
```

**Pin check (no GPU):** `./scripts/phase/phase15_llama_kv_ext_pin_check.sh` — run before a long CUDA rebuild.

**Vendor alternative:** `5080_build_vendor_llama_server` for Radix `/kv/seq-copy` (patch 0017); Phase 15 KV-ext on sibling is still the documented path for tier 3. See [phase15-llama-kv-ext-upstream.md](./phase15-llama-kv-ext-upstream.md).

---

## Vendor llama-server (Radix / patch 0017)

**Why not sibling alone:** bare `~/llama.cpp` often lacks `POST /kv/seq-copy` even when in-tree carries patch **0017**. Radix cross-slot live **requires** vendor build.

```bash
source ./scripts/gpu/5080_env.sh
# First time / after pin bump: vendor tree @ 86d86ed4
cd ~/zerollama && make -f Makefile.sync vendor   # → vendor/llama-cpp-86d86ed4

5080_build_vendor_llama_server    # build + llama_patch_doctor.sh
5080_patch_doctor                 # optional live probe on :8082
```

**Fold into resignoff:** `./scripts/gpu/5080_resignoff.sh --tier 2 --radix --vendor --no-serve`

**Patch doctor (offline):** `./scripts/vendor/llama_patch_doctor.sh` — greps vendor HEAD, binary `/kv/seq-copy`, resolved paths.

---

## Start serve (required before Tier 1)

**Why:** `gpu_5080_session.sh` runs smokes against **already running** embed on `:8081` + Go on `:8080`. Stale listeners or missing `PYTHONPATH` cause false FAILs before any gate runs.

**Gate / dev (loopback, from `5080_env.sh`):**

```bash
source ./scripts/gpu/5080_env.sh
5080_start_serve    # 5080_stop_serve + zerollama serve + /health wait
```

---

## Production serve (`~/bin/serve.sh`)

**Never start serve on the Proxmox hypervisor** (ryzen7950x0). Host `/root/zerollama` is a symlink into this CT; `~/bin/serve.sh` on PVE binds host IPs, not `192.168.255.164`. Always `pct exec 1564 -- ~/bin/serve.sh`. The wrapper exits if `systemd-detect-virt` is not `lxc`.

**Why a separate path from `5080_start_serve`:** sign-off uses loopback `:8080` and may set `ZEROLLAMA_GPU_PROFILE=0` for 1B smoke. Production binds **`0.0.0.0:8080`** for remote clients (Ruby `ZEROLLAMA_API_ENDPOINT`, Open WebUI, etc.) and keeps the **L1 `rtx-5080` profile** on (`n_parallel=2`, vendor fork KV). Embedded runtime stays **`127.0.0.1:8081`** — remote clients must not point at `:8081`.

**WHY not copy `serve_gpu_example.sh` to `~/bin`:** that script lives in `scripts/` and resolves repo root as `dirname/..`. In `~/bin/serve.sh`, `..` is **`$HOME`**, not `~/zerollama` — helper scripts and `PYTHONPATH` never load; serve exits or embed fails silently when stdout is redirected.

**Install (once after pull):**

```bash
cd ~/zerollama
cp scripts/serve/serve_production_wrapper.sh ~/bin/serve.sh
chmod +x ~/bin/serve.sh
```

**Start + verify:**

```bash
~/bin/serve.sh                    # blocks; logs → /tmp/zerollama-serve.log
# Install / refresh the binary the CT actually runs (not the Proxmox host /usr/bin):
#   cp -a ~/zerollama/zerollama /usr/bin/zerollama   # inside CT 1564 / cudallama
# Sign-off scripts that start a temp serve on 127.0.0.1:8080 must tear it down —
# a second `zerollama serve` now refuses with a clear occupied-bind error.
tail -f /tmp/zerollama-serve.log

curl -s http://127.0.0.1:8080/api/version | jq .
curl -s http://127.0.0.1:8081/health | jq '{status, profile: .gpu_profile.id, backend: .llama_backend}'
curl -s http://127.0.0.1:8080/api/train/status | jq .
```

**Remote smoke (CT 1564 example):** `curl -s http://192.168.255.164:8080/api/tags`

In-repo reference (same env, for debugging): [`scripts/serve/serve_gpu_example.sh`](../scripts/serve/serve_gpu_example.sh) sets:

| Env | Production value | Why |
|-----|------------------|-----|
| `OLLAMA_HOST` | `0.0.0.0:8080` | Remote clients (not `127.0.0.1:11434`) |
| `OLLAMA_TRAINING` | `true` | Embedded training on `:9500` / `/api/train/*` |
| `ZEROLLAMA_RUNTIME_CONFIG` | `runtime/configs/single_gpu.yaml` | 16GB CUDA defaults + L1 profile |
| `ZEROLLAMA_RUNTIME_VRAM_*` | min-free 1GiB, training reserve 2GiB, autotune auto | Phase 11/13 headroom |
| `OLLAMA_LIBRARY_PATH` | `/usr/lib/ollama`, `cuda_v12`, `mlx_cuda_v12` | ggml + optional MLX imagegen |
| `GGML_CUDA_USE_GRAPHS` | `0` on 5080 | Stability on sm_120 until validated |
| `SERVE_LOG` | `/tmp/zerollama-serve.log` | Quiet screen; `tail -f` |

---

## Training embed + venvs

**Why 3.11 on CT 1564:** `runtime/.venv` is Python 3.11; Ubuntu 22.04 default `python3-embed` is 3.10 — mismatch breaks embedded training (`ModuleNotFoundError` or wrong `.venv-training` path in logs).

```bash
sudo apt install python3.11-dev python3.11-embed
cd ~/zerollama
source ./scripts/gpu/5080_env.sh
5080_setup_venvs

# Rebuild binary pinned to 3.11 embed:
source ./scripts/training/training_embed_build_env.sh 3.11
CGO_ENABLED=1 go build -o zerollama .
cp zerollama /usr/bin/zerollama

TRAINING_UV_PYTHON_VER=3.11 ./scripts/training/training_uv_venv.sh --verify
```

**Cleanup after migration:** `rm -rf venv-training/ .venv-training-py310.bak` (~7 GiB each if duplicate torch stacks).

**Verify training after `~/bin/serve.sh`:**

```bash
curl -s http://127.0.0.1:8080/api/train/status | jq .
```

---

## Tier 0 — sanity (no GPU)

**Why:** catch script drift and parser regressions before tying up the card.

```bash
./scripts/check_gpu_scripts.sh
./scripts/phase/phase12_golden_ci.sh          # full dev host only; skipped when RUN_E2E_PREFLIGHT=0
./scripts/phase/phase15_kv_native_ci.sh     # CPU Phase 15 pytest bundle
./scripts/phase/phase15_upstream_kv_watch.sh  # upstream writable-bind symbol watch (no GPU)
./scripts/phase/phase17_l2_pin_status.sh      # L2 pin report (no GPU)
```

---

## Tier 1 — official base gate (Phase 11–13)

**Why:** same role as Mac `phase11_13_15_metal_signoff.sh` for admission + VRAM + coordination — discrete NVML path, not unified memory.

```bash
kill $(pgrep -xo zerollama) 2>/dev/null || fuser -k 8080/tcp 8081/tcp 2>/dev/null || true
./scripts/gpu/gpu_5080_session.sh
```

**Pass:** `PASS: gpu_5080_session` + `/tmp/5080-session.json` (or `GPU_PHASE13_SNAPSHOT_OUT`).

**Requires:** serve running — see [Start serve](#start-serve-required-before-tier-1). Step order: [`gpu_5080_session.sh` internals](#gpu_5080_sessionsh-internals).

**Re-sign-off PASS (CT 1564, Jun 28 2026):** base session with `RUN_E2E_PREFLIGHT=0` after rebuild + NVML 590.48.01 fix + embed `PYTHONPATH`.

---

## Tier 2 — L1 + L3 production gates (borrowings)

**Why:** Phase 13 estimates *fit*; **L1** picks throughput knobs (`rtx-5080.json`: `n_parallel=2`, `batch_size=1024`); **L3** proves prompt-cache → slot bridge on agent-scale GGUF.

```bash
export CUDA_LLAMA_MODEL=/root/eliza-1-9b-256k.gguf   # 7B–9B production proxy

# Combined in one session:
RUN_E2E_PREFLIGHT=0 RUN_E2E_L1=1 RUN_E2E_L3=1 \
  CUDA_LLAMA_MODEL="$CUDA_LLAMA_MODEL" \
  ./scripts/gpu/gpu_5080_session.sh

# Or standalone:
./scripts/phase/l1_cuda_full_gate.sh
./scripts/phase/l3_cuda_full_gate.sh
./scripts/phase/l3_production_gate.sh    # 27k ctx production verdict
```

| Gate | Status (CT 1564, Jun 28 2026) | Notes |
|------|-------------------------------|--------|
| **L1** single-stream | **−5% @ 8k** (informational) | eliza-1 9B: OFF **~90** tok/s (`np=1`) vs ON **~85** (`rtx-5080`, `np=2`, stock `q8_0`) — expected single-request cost of `n_parallel=2`; calibrate sets `ZEROLLAMA_GPU_PROFILE_CTX=0` (no profile `-c` during 8k bench) |
| **L1** concurrent N=2 | **PASS** | **+~16–20%** agg tok/s (ON **~65** vs OFF **~55** @ 8k) — **why `np=2`:** L3 needs ≥2 slots; OFF leg may show 1×502 on 2nd thread (expected at `np=1`) |
| **L3** 8k strict | **PASS** | cached turn2 faster than turn1 and no-cache control |
| **L3** 27k production | **PASS** | cached faster than no-cache; strict turn2/turn1 ratio may exceed 0.75 on 9B |
| **Radix cross-slot live** | **PASS** | eliza-1 9B @ 8k, vendor `llama-server` (patch 0017): donor **10.6s** → target **0.66s**; `radix_seed` **128** tokens; `/tmp/l3-radix-prefix-smoke-live.json`. Mac Jun 2026: donor **8.2s** → target **0.58s**. |

Optional spec-decode × L3 policy leg:

```bash
RUN_E2E_L3_SPEC=1 RUN_E2E_PREFLIGHT=0 RUN_E2E_L3=1 ./scripts/gpu/gpu_5080_session.sh

# Radix cross-slot (optional — after same-key L3 PASS; vendor llama-server required):
RUN_E2E_PREFLIGHT=0 RUN_E2E_L3_RADIX=1 CUDA_LLAMA_MODEL="$CUDA_LLAMA_MODEL" ./scripts/gpu/gpu_5080_session.sh
# Or folded into L3 gate:
RUN_E2E_PREFLIGHT=0 RUN_E2E_L3=1 RUN_E2E_L3_RADIX=1 CUDA_LLAMA_MODEL="$CUDA_LLAMA_MODEL" ./scripts/gpu/gpu_5080_session.sh
# Standalone:
CUDA_LLAMA_MODEL="$CUDA_LLAMA_MODEL" L3_RADIX_LIVE=1 ./scripts/phase/l3_radix_prefix_smoke.sh
```

---

## Tier 3 — Phase 14 + 15 (in-process KV)

**Why:** Mac closes this with `./scripts/gpu/metal_signoff.sh`; CUDA uses **embed** path (`phase15_inprocess_signoff.sh`), not uv sidecar. Phase 15 needs **patched** `libllama.so` + `LLAMA_CPP_LIB`; multiseq smokes set **`ZEROLLAMA_GPU_PROFILE=0`** — **why:** L1 `-np 4` breaks `kv_inprocess_n_seq_max=2`.

### Phase 15 only (~10 min)

```bash
export LLAMA_CPP_LIB="$HOME/llama.cpp/build/bin/libllama.so"   # or vendor build with kv-ext
kill $(pgrep -xo zerollama) 2>/dev/null || fuser -k 8080/tcp 8081/tcp 2>/dev/null || true
./scripts/phase/phase15_inprocess_signoff.sh
```

**Pass:** KV hook (`kv_decode_steps>0`, `batch_decode_in_c=True`), multiseq, batch decode via `/internal/generate-batch`, `smoke_runtime_assert_kv_snapshot` accepts **`bound`+`tensor`** when kv-ext linked.

**Pitfalls (CT 1564):**

- **Port 8081:** multiseq restarts serve — stale embed on `:8081` makes the new serve fail embed and the multiseq curl hangs. Free ports before sign-off.
- **`_kv_native` build:** if `pip install -e` fails on `tool.setuptools.ext-modules`, run `uv pip install --python runtime/.venv/bin/python 'setuptools>=75'` then `phase15_runtime_kv_ext_build` (or `PHASE15_BUILD_KV_EXT=0` when `.so` already built for your Python).

**Re-sign-off PASS (CT 1564, Jun 28 2026):** OuteTTS 1B Q8 — KV hook + multiseq + batch decode.

**Optional tensor bind spot-check** (after multiseq PASS):

```bash
source ./scripts/phase/phase15_runtime_kv_env.sh
phase15_runtime_kv_ext_build
./scripts/phase/phase15_tensor_bind_probe.sh
curl -s :8081/health | jq '.kv_page_bind, .kv_decode_loop'
```

Expect `batch_decode_in_c: true`; after generate with active bind: `status: "bound"`, `bind_level: "tensor"` when kv-ext linked.

### Phase 14 + 15 full sign-off (~15–20 min)

```bash
export LLAMA_CPP_LIB="$HOME/llama.cpp/build/bin/libllama.so"
RUN_E2E_PREFLIGHT=0 RUN_E2E_PHASE14_SIGNOFF=1 ./scripts/gpu/gpu_5080_session.sh
```

**Why separate from Tier 1:** sign-off **restarts serve** per backend; folded into session only when explicitly flagged.

**Phase 14 note:** use **inprocess** for CUDA GPU — wheel GPU aborts on 5080 (`free(): invalid pointer`). Full checklist: [Phase 14 sign-off](#phase-14-sign-off).

---

## Tier 4 — Phase 16 + 17 upstream GGUF path

**Why:** Mac edge binary smoke PASS; **Linux CUDA** Phase 16/17 operator sign-off on ship hardware ([ROADMAP Phase 16 #4](./ROADMAP.md#phase-16--exit-criteria-partial)).

**Re-sign-off (CT 1564, Jun 2026):** `RUN_E2E_UPSTREAM_GGUF=1` bundle **PASS** after `gpu_5080_session.sh` auto-restarts serve with `ZEROLLAMA_GPU_PROFILE=0` (1B smoke + P17 + Linux auto + edge). Individual smokes still valid standalone.

```bash
export LLAMA_SERVER_BIN="$HOME/llama.cpp/build/bin/llama-server"
export LD_LIBRARY_PATH="$(dirname "$LLAMA_SERVER_BIN"):${LD_LIBRARY_PATH:-}"
export RUN_E2E_PROXY_MODEL=llama3.2:3b
kill $(pgrep -xo zerollama) 2>/dev/null || fuser -k 8080/tcp 8081/tcp 2>/dev/null || true

# One flag — sets RUN_E2E_P17 + RUN_E2E_P17_LINUX_AUTO + RUN_E2E_EDGE (needs serve up for base leg):
RUN_E2E_PREFLIGHT=0 RUN_E2E_UPSTREAM_GGUF=1 ./scripts/gpu/gpu_5080_session.sh
```

**Recommended on 5080 (avoids fork-cache × stock llama-server clash):**

```bash
LLAMA_SERVER_BIN="$LLAMA_SERVER_BIN" P17_MODEL=llama3.2:3b ./scripts/phase/phase17_llama_server_smoke.sh
LLAMA_SERVER_BIN="$LLAMA_SERVER_BIN" P17_MODEL=llama3.2:3b ./scripts/phase/phase17_linux_auto_smoke.sh
P17_NUM_PREDICT=32 LLAMA_SERVER_BIN="$LLAMA_SERVER_BIN" P16_MODEL=llama3.2:3b ./scripts/phase/phase16_edge_smoke.sh
```

**Individual smokes via session flags** (also valid; each restarts serve):

```bash
RUN_E2E_P17=1 ./scripts/gpu/gpu_5080_session.sh              # Go → llama-server generate
RUN_E2E_P17_LINUX_AUTO=1 ./scripts/gpu/gpu_5080_session.sh   # plain serve, backend.llama_server=auto
RUN_E2E_EDGE=1 ./scripts/gpu/gpu_5080_session.sh             # serve --edge, runtime chat off
./scripts/phase/phase16_edge_build_smoke.sh                    # no GPU: -tags edge compile
```

**Pass criteria:** `phase17_llama_server_smoke` generate OK; `/api/status` shows `inference.backend` with `llama_server`; edge smoke: no `:8081` runtime chat; `GET /api/version` `edge_build` when using `-tags edge` binary. Edge smoke: if `empty response` at default `num_predict=8`, retry with `P17_NUM_PREDICT=32`.

**L2 pin merge (Phase 17 #7):** still **FAIL @ 8k** stock vs fork on CT 1564 — informational only; see [L2 fork eval](#l2-fork-eval-informational).

---

## `gpu_5080_session.sh` internals

**Why one wrapper:** CI runs without GPU; CT needs preflight → smoke → snapshot → optional flags in one place.

| Step (order) | When | Why |
|--------------|------|-----|
| `phase12_golden_ci.sh` | `RUN_E2E_PREFLIGHT=1` (default off on CT via `5080_env.sh`) | Go Harmony/parser golden |
| `gpu_smoke_all.sh` | always | Coordination, VRAM prep, runtime e2e, health report |
| `gpu_phase13_snapshot.sh` | always when `RUN_E2E_PHASE13_SNAPSHOT=1` | JSON → `/tmp/5080-session.json` |
| `python -m runtime.gpu_snapshot` | after snapshot | Copy-paste env hints |
| Profile-off serve restart | `RUN_E2E_UPSTREAM_GGUF=1` | **Why:** fork KV cache types break 1B base leg after L1/L2 |
| `phase14_*` / `phase15_*` | optional flags | See [Environment reference](#environment-reference) |
| `l1_cuda_full_gate.sh` | `RUN_E2E_L1=1` | Needs `CUDA_LLAMA_MODEL` |
| `l3_cuda_full_gate.sh` | `RUN_E2E_L3=1` | 9B+ production GGUF |
| `l3_radix_prefix_smoke.sh` | `RUN_E2E_L3_RADIX=1` | Vendor `/kv/seq-copy` |
| P17 / edge smokes | `RUN_E2E_P17*` / `RUN_E2E_EDGE` / bundle | Upstream GGUF path |

**Pass:** `PASS: gpu_5080_session` + snapshot file.

---

## Phase 13 VRAM / snapshot

### `single_gpu.yaml` `vram:` defaults (applied at runtime start)

| YAML key | Default | Why |
|----------|---------|-----|
| `min_free` | `1GiB` | Admission floor |
| `training_reserve` | `2GiB` | Headroom during training handoff |
| `estimate_factor_autotune` | `auto` | Per-GGUF calibration |
| `probe_calibrate` | `auto` | Post-load VRAM observation on `/health` |
| `clamp_num_ctx` | `"0"` | **Off** by default — enable `auto` only if you accept API `vram_num_ctx` |

### `gpu_snapshot` rules after session

- Autotune **persist** wins — do not set global `VRAM_ESTIMATE_FACTOR` when persist exists.
- Warn when budget `fits_with_margin=false`.
- Harmony real-weight **skipped** on ~19 GiB CT — use `phase12_golden_ci.sh` on dev host instead.

```bash
python -m runtime.gpu_snapshot /tmp/5080-session.json
```

---

## VRAM prep during smokes

**Problem:** Go may return **503** before Phase 8 broker runs; stale ggml runner holds VRAM → runtime 502/503.

**Solution:** `smoke_unload_ggml_runners` in `scripts/runtime/runtime_smoke_lib.sh`:

1. `pgrep -f '/zerollama runner --'` only (not shell lines).
2. `/api/ps` → each model `keep_alive:0`.
3. Wait up to **`SMOKE_UNLOAD_MAX_WAIT`** (default **30s**).
4. On 503 + runner still up: retry unload + `runtime_resume_if_needed`.

**Do not `pkill` runners during gates** — breaks operator contract and metrics.

---

## L1 profile (`rtx-5080.json`)

**Path:** `runtime/configs/gpu/rtx-5080.json` — loaded when `ZEROLLAMA_GPU_PROFILE=1` (default in production `serve_gpu_example.sh`).

| Flag | Value | Why |
|------|-------|-----|
| `n_parallel` | **2** | L3 needs ≥2 slots; `np=4` regressed 1B single-stream (−12%) on CT 1564 |
| `batch_size` / `ubatch_size` | **1024** / **256** | Half 4090 defaults for 16 GiB |
| `ctx_size` | **32768** | Agent-scale window; L1 calibrate sets `ZEROLLAMA_GPU_PROFILE_CTX=0` during 8k bench |
| `cache_type_k` / `v` | **q8_0** (stock path) | Fork-only keys `_eliza_fork_*`: `qjl1_256` / `q4_polar` when L2 lands |
| `flash_attn` | **true** | Throughput on sm_120 |

**Profile OFF (`ZEROLLAMA_GPU_PROFILE=0`):** tier 1 base smoke, Phase 15 multiseq, `RUN_E2E_UPSTREAM_GGUF=1` bundle — **why:** fork QJL breaks 1B OuteTTS and stock llama-server path.

---

## L2 fork eval (informational)

**Status:** **FAIL merge @ 8k** — stock wins decode; **expected**; does not block re-sign-off. **L2 Done (Jul 2026)** on pin `86d86ed4` — TBQ is **VRAM opt-in** (`ZEROLLAMA_LLAMA_FORK=1` / `AUTO_VRAM`), not default.

```bash
export CUDA_LLAMA_MODEL="${CUDA_LLAMA_MODEL:-/root/eliza-1-9b-256k.gguf}"
export LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-$HOME/zerollama/vendor/llama-cpp-86d86ed4}"
# First time / after pull: rebuild vendor CUDA llama-server
5080_build_vendor_llama_server
# Gate:
L2_BUILD_FORK=0 ./scripts/phase/l2_cuda_full_gate.sh   # uses existing bin when present
./scripts/phase/l2_gate_report.sh /tmp/l2-cuda-gate/bench-*.json
# Or cross-platform pin report (no GPU):
./scripts/phase/phase17_l2_pin_status.sh
```

| Model @ 8192 ctx | Stock tok/s | Fork tok/s | Verdict |
|------------------|-------------|------------|---------|
| 1B Q8 | **79.3** | 56.9 | Stock wins |
| 9B eliza-1 | **18.6** | 14.4 | Stock wins |
| 27k | ~−22% fork | same pattern | Informational |
| 131k fork | blocked | 9B VRAM; 1B QJL head | Optional `L2_RUN_131K_FORK=1` |
| **Mac M4 Max (Jul 2026, eliza-1-2b)** | **68.8** | **48.9** TBQ (−29%) | Same FAIL-merge pattern; `/tmp/l2-metal-bench-0088.json` |

**Artifacts:** `/tmp/l2-cuda-gate/`. **WHY headless:** `L2_BUILD_FORK=1` sets `LLAMA_BUILD_WEBUI=OFF` on Linux — WebUI build fails on CT.

**Phase 17 #7:** vendor merge still blocked for **tok/s defaults** — fork profiles remain **opt-in**.

---

## Tier F — RotorQuant / post-L2 labs (Jul 2026)

**Why this tier:** Mac finished Bee **B0** + Metal L2 smoke + **Metal Lab A** (Jul 2026 quiet GPU). RotorQuant `planar3`/`iso3` vs TBQ measured on **both** CUDA FA and Metal — **no-merge** everywhere.

**Hand-off from Mac (already done):**

| Item | Status |
|------|--------|
| Pin `86d86ed4` + patches **0080–0088** (+ **0093** llama-bench names) | Committed |
| Metal `llama-server` rebuild | OK |
| B0 stop / force-close smoke | PASS on lab `:18082` |
| Metal L2 stock vs TBQ @ 8k | Stock wins decode (−29% fork) |
| **Metal Lab A** Llama-3.2-3B planar/iso/TBQ/QJL | **No-merge** planar/iso; stock **f16**; `speed` unblocked by **0097+0098** (tg ~37, still FAIL merge); `tmp/metal-ab/v2/` + `d1b/` |
| RotorQuant sibling scout | `../llama-cpp-rotorquant` @ `feature/planarquant-kv-cache` (Metal build OK) |

**Do not** touch production Go `:8080` / embed `:8081` for these labs. Use **`:18082`** (harness default) or another free lab port.

### F0 — sync + rebuild vendor CUDA binary

```bash
cd ~/zerollama && git pull
source ./scripts/gpu/5080_env.sh    # Z5080_VENDOR_PIN=86d86ed4
5080_setup_cuda
make -f Makefile.sync vendor        # if vendor/llama-cpp-86d86ed4 missing/stale
5080_build_vendor_llama_server      # sm_120 llama-server + patch doctor
./vendor/llama-cpp-86d86ed4/build/bin/llama-server --help | grep -E 'reasoning-loop|cache-type'
```

### F1 — RotorQuant A/B (primary)

```bash
# Clone / update fork (CUDA CT)
git clone -b feature/planarquant-kv-cache \
  https://github.com/johndpope/llama-cpp-turboquant.git ~/llama-cpp-rotorquant \
  || (cd ~/llama-cpp-rotorquant && git fetch && git checkout feature/planarquant-kv-cache && git pull)

cmake -S ~/llama-cpp-rotorquant -B ~/llama-cpp-rotorquant/build \
  -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_WEBUI=OFF
cmake --build ~/llama-cpp-rotorquant/build -j --target llama-server llama-bench

# Lab ports only — never :8080/:8081
export CUDA_LLAMA_MODEL="${CUDA_LLAMA_MODEL:-/root/eliza-1-9b-256k.gguf}"
export LLAMA_SERVER_BIN=~/zerollama/vendor/llama-cpp-86d86ed4/build/bin/llama-server
export ROTORQUANT_LLAMA_SERVER_BIN=~/llama-cpp-rotorquant/build/bin/llama-server
export L2_PORT=18082
export L2_NUM_CTX=8192
export L2_RQ_ALSO_LLAMA_BENCH=1
export L2_RQ_OUT=/tmp/l2-rotorquant-ab.json

cd ~/zerollama
./scripts/phase/l2_rotorquant_ab.sh
# Optional long-ctx leg:
L2_NUM_CTX=26624 L2_RQ_OUT=/tmp/l2-rotorquant-ab-27k.json ./scripts/phase/l2_rotorquant_ab.sh
```

**Exit before cherry-pick** (from [llama-fork-watchlist.md](./llama-fork-watchlist.md)):

1. `planar3` or `iso3` decode ≥ TBQ at same ctx, **or** clear VRAM win with acceptable PPL / prefill.
2. Prefill (`llama-bench -p`) not much worse than TBQ.
3. FA path stable on 5080 (then dual-4090).
4. **Type IDs collide** — RotorQuant `PLANAR3=44` / `ISO3=45` / `PLANAR4=46` / `ISO4=47` overlap our `TBQ3`/`TBQ4`/`QJL`. Remap to free IDs (≥53 or unused gaps) before any vendor patch.

**If it fails:** leave as external lab binary; keep TBQ opt-in.

### F2 — Bee B0 on CUDA (optional sanity)

Mac already PASSed. Optional on 5080 with a think-capable GGUF (lab port):

```bash
# Lab only
./vendor/llama-cpp-86d86ed4/build/bin/llama-server \
  -m "$CUDA_LLAMA_MODEL" -c 8192 -ngl 99 \
  --host 127.0.0.1 --port 18082 \
  --reasoning-loop-guard force-close
# POST /completion with reasoning_budget_* tags + forced loop → expect loop_guard / stop_detail
```

### F3 — after RotorQuant result

| Outcome | Next |
|---------|------|
| RotorQuant wins | New patches after **0088** (types + CUDA FA/SET_ROWS); remap IDs; dual-4090 A/B |
| TBQ holds | No codec merge; defer **B1 adaptive draft-max**; optional TQ3 FP4 (Lab C) only with TQ3 weights |
| Dual-4090 | Same harness; `CUDA_VISIBLE_DEVICES=1` sidecar style — see [cuda-lanes.md](./cuda-lanes.md) |

Doc: [llama-fork-watchlist.md](./llama-fork-watchlist.md) · harness: `scripts/phase/l2_rotorquant_ab.sh`

---

## L3 + Radix criteria

### L3 subprocess cache (same-key prefix)

**Strict PASS:** turn 2 faster than turn 1 **or** `cached_faster_than_no_cache` with `L3_COMPARE_NO_CACHE=1`.

**Jun 2026 eliza-1 9B @ 8k:** cached **0.66s** vs no-cache **1.13s** (`L3_PREFIX_REPEAT=150`).

**27k production:** cached must beat no-cache control — strict `turn2/turn1 ≤ 0.75` may fail when turn-1 already warmed the slot; alternate PASS is cached vs no-cache.

```bash
CUDA_LLAMA_MODEL="$CUDA_LLAMA_MODEL" ./scripts/phase/l3_production_gate.sh
```

### Agent lane vs Phase 17 (ops)

| Lane | Knob | YAML | Where Radix lives |
|------|------|------|-------------------|
| Daily throughput (prod default) | `ZEROLLAMA_INFERENCE_PROFILE=auto` | `single_gpu.yaml` (1 slot) | Off — L1 np=2 is Go/llama-server only |
| Agent / shared-prefix | `ZEROLLAMA_INFERENCE_PROFILE=agent` | `l3_agent_subprocess.yaml` (slots + `radix_share`) | Python runtime (`:8081` embed or sidecar) |

**WHY not `RADIX=1` alone:** with `single_gpu.yaml`, `llama_parallel_slots=1` so cross-slot seq-copy never fires. Explicit `ZEROLLAMA_RUNTIME_CONFIG` still wins over the agent default.

**Lab beside prod:** `ZEROLLAMA_RUNTIME_URL=http://127.0.0.1:18081 L3_RADIX_LIVE=1 …` — never restart/kill `:8081`/`:11434`. Prefer a small GGUF when prod already holds VRAM; eliza-1 9B @ 4 slots aborted beside ~6 GiB prod load (use OuteTTS 1B or free GPU).

### Radix cross-slot live (vendor only)

**Requires:** vendor `llama-server` with patch **0017** (`POST /kv/seq-copy`). Sibling-only build often lacks route even when source has patch.

**PASS (CT 1564):** donor prefill **10.6s** → target **0.66s**; `radix_seed` **128** tokens; artifact `/tmp/l3-radix-prefix-smoke-live.json`.

```bash
CUDA_LLAMA_MODEL="$CUDA_LLAMA_MODEL" L3_RADIX_LIVE=1 ./scripts/phase/l3_radix_prefix_smoke.sh
# Or: RUN_E2E_L3_RADIX=1 ./scripts/gpu/gpu_5080_session.sh
```

**Optional spec-decode × L3:** `RUN_E2E_L3_SPEC=1` — policy leg after same-key L3 PASS.

---

## Phase 14 sign-off

**Why separate from Tier 3:** Phase 14 validates ctypes **`libllama.so`** in the embedded runtime (no loopback `llama-server`). Phase 15 adds KV-ext on top of the same inprocess path.

**5080 rule:** use **inprocess** for CUDA GPU — pip wheel GPU aborts (`free(): invalid pointer`).

### Checklist

1. Rebuild: `5080_build_zerollama` or `CGO_ENABLED=1 go build -o zerollama .`
2. Patched `LLAMA_CPP_LIB` on sibling or vendor build
3. Serve with inprocess backend:

```bash
source ./scripts/phase/phase14_serve_env.sh
export LLAMA_MODEL="${LLAMA_MODEL:-/root/Llama-OuteTTS-1.0-1B-Q8_0.gguf}"
export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-$HOME/llama.cpp/build/bin/libllama.so}"
export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess
5080_stop_serve && ./zerollama serve &
5080_wait_health
```

4. Smoke + render:

```bash
export RUN_E2E_PROXY_MODEL="${RUN_E2E_PROXY_MODEL:-llama3.2:3b}"
./scripts/phase/phase14_inprocess_smoke.sh
# Or: RUN_E2E_INPROCESS=1 ./scripts/phase/phase14_backend_smoke.sh
```

**Pass:** `PASS: phase14_backend_smoke`; `/health`: `llama_backend=inprocess`, `llama_backend_source=env`, `llama_server=false`; render-chat `truncate_mode=tokenize`.

### Fold into session

```bash
RUN_E2E_PREFLIGHT=0 RUN_E2E_PHASE14_SIGNOFF=1 ./scripts/gpu/gpu_5080_session.sh
# Runs phase14_5080_signoff.sh — restarts serve per backend (~15–20 min with Phase 15)
```

### Optional: YAML inprocess (after env smoke PASS)

```bash
./scripts/phase/phase14_enable_yaml_inprocess.sh
# restart serve without ZEROLLAMA_RUNTIME_LLAMA_BACKEND
RUN_E2E_LLAMA_BACKEND_SOURCE=config ./scripts/phase/phase14_yaml_config_smoke.sh
```

Uncomment `llama_backend: inprocess` in `runtime/configs/single_gpu.yaml` — `/health` shows `llama_backend_source=config`.

---

## Harmony / host RAM (not VRAM)

| Path | `gpt-oss:20b` MXFP4 |
|------|---------------------|
| Runtime load | Often **~44 GiB host RAM** for mmap budget — fails on ~19 GiB CT |
| CI | `TestGoldenHarmonyParseToolOutput` in `phase12_golden_ci.sh` |
| 5080 GPU gate | **`gpu_5080_session.sh` does not require** `gpu_harmony_capture.sh` |

**WHY document here:** operators chase harmony real-weight on hardware that cannot succeed; Harmony ships via Go parser tests. Do **not** block 5080 re-sign-off on `gpt-oss:20b`.

---

## Tier 5 — optional / periodic

| Goal | Command | Why |
|------|---------|-----|
| **Training ops** | `RUN_E2E_TRAINING_OPS=1` (serve needs `OLLAMA_TRAINING=true`) | Embedded training HTTP/TCP without blocking inference smokes |
| **Phase 11 contention** | `./scripts/phase/phase11_5080_contention_smoke.sh` or `RUN_E2E_PHASE11_CONTENTION=1` | Concurrent normal + low generates; JSON `/tmp/phase11-5080-contention.json` — [phase11-runtime-admission.md](./phase11-runtime-admission.md) |
| **Tools chat** | `RUN_E2E_TOOLS=1 ./scripts/gpu/gpu_smoke_all.sh` | Runtime `/api/chat` with tools — 501 means wrong route |
| **VRAM clamp** | `RUN_E2E_VRAM_CLAMP=1` + `ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX=auto` on serve | Opt-in clamp policy — default off in YAML |
| **L2 fork eval** | `./scripts/phase/l2_full_gate.sh` | Compare stock vs eliza fork @ 8k; long-ctx fork-only legs optional |
| **Radix live (5080)** | `L3_RADIX_LIVE=1 ./scripts/phase/l3_radix_prefix_smoke.sh` | Cross-slot prefix share — **WHY vendor:** patch 0017 `/kv/seq-copy`; fold via `RUN_E2E_L3_RADIX=1` on `gpu_5080_session.sh` |
| **Phase 17 vision** | `RUN_E2E_P17_VISION=1 P17_VISION_MODEL=llava:latest` | Heavy; opt-in projector smoke |
| **Decode graph** | Rebuild llama-server with `GGML_CUDA_GRAPHS=ON`; L3 smokes | CUDA graph invalidation on slot clear — [decode-graph-invalidation.md](./decode-graph-invalidation.md) |
| **Writable KV bind watch** | `./scripts/phase/phase15_upstream_kv_watch.sh` | Phase 15 criterion #5 blocked upstream — no live decode required |

---

## Recommended full re-sign-off sequence

**Preferred:** one driver (after `git pull` inside CT 1564):

```bash
cd ~/zerollama
source ./scripts/gpu/5080_env.sh
./scripts/gpu/5080_resignoff.sh --build
# Optional Radix (vendor llama-server):
./scripts/gpu/5080_resignoff.sh --tier 2 --radix --vendor --no-serve
```

**Manual equivalent** (stop on first FAIL):

```bash
# 0. Env + one-time build (see sections above)
source ./scripts/gpu/5080_env.sh
export RUN_E2E_PREFLIGHT=0
5080_start_serve

# 1. Base Phase 11–13
./scripts/gpu/gpu_5080_session.sh

# 2. L1 + L3 production
RUN_E2E_L1=1 RUN_E2E_L3=1 ./scripts/gpu/gpu_5080_session.sh

# 3. Phase 15 KV (restarts serve inside script)
5080_stop_serve
./scripts/phase/phase15_inprocess_signoff.sh

# 4. Phase 16/17 upstream (individual smokes)
LLAMA_SERVER_BIN="$LLAMA_SERVER_BIN" P17_MODEL=llama3.2:3b ./scripts/phase/phase17_llama_server_smoke.sh
LLAMA_SERVER_BIN="$LLAMA_SERVER_BIN" P17_MODEL=llama3.2:3b ./scripts/phase/phase17_linux_auto_smoke.sh
P17_NUM_PREDICT=32 LLAMA_SERVER_BIN="$LLAMA_SERVER_BIN" P16_MODEL=llama3.2:3b ./scripts/phase/phase16_edge_smoke.sh
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

## Environment reference

### `5080_env.sh` variables (set on `source`)

| Variable | Default (CT 1564) | Role |
|----------|-------------------|------|
| `Z5080_REPO` | `~/zerollama` | Repo root |
| `Z5080_LLAMA_CPP` | `~/llama.cpp` | Sibling pin b9781 + kv-ext |
| `Z5080_VENDOR_PIN` | `86d86ed4` | Vendor tree (Radix + L2 patches through **0088**) |
| `LLAMA_MODEL` | `/root/Llama-OuteTTS-1.0-1B-Q8_0.gguf` | Tier 1 smoke |
| `CUDA_LLAMA_MODEL` | `/root/eliza-1-9b-256k.gguf` | L1/L3 production |
| `RUN_E2E_GGUF` | `$LLAMA_MODEL` | Runtime load path in smokes |
| `RUN_E2E_PROXY_MODEL` | `llama3.2:3b` | Phase 17 manifest name |
| `LLAMA_CPP_LIB` / `LLAMA_SERVER_BIN` | sibling or vendor `build/bin` | Subprocess + ctypes |
| `OLLAMA_HOST` | `http://127.0.0.1:8080` | Go API (sign-off loopback) |
| `ZEROLLAMA_RUNTIME_CONFIG` | `runtime/configs/single_gpu.yaml` | Embed + VRAM YAML |
| `PYTHONPATH` | `runtime/.venv` + `.venv-training` | **Required** for embed |
| `RUN_E2E_PREFLIGHT` | **`0`** on CT | Skip Go golden when httplib missing |
| `GPU_PHASE13_SNAPSHOT_OUT` | `/tmp/5080-session.json` | Phase 13 JSON |
| `P17_MODEL` | `$RUN_E2E_PROXY_MODEL` | Phase 17 smokes |

### Serve / production overrides

| Variable | Sign-off | Production (`~/bin/serve.sh`) |
|----------|----------|-------------------------------|
| `OLLAMA_HOST` | `127.0.0.1:8080` | **`0.0.0.0:8080`** |
| `ZEROLLAMA_GPU_PROFILE` | `0` tier 1 / upstream bundle | **`1`** (rtx-5080) |
| `OLLAMA_TRAINING` | optional | **`true`** |
| `GGML_CUDA_USE_GRAPHS` | — | **`0`** on 5080 until validated |

### Gate / smoke variables

| Variable | Role |
|----------|------|
| `ZEROLLAMA_RUNTIME_URL` | Sidecar URL when **not** embedded — **must be unset** for embed sign-off |
| `ZEROLLAMA_RUNTIME_EMBED` | `on` for CT 1564 |
| `ZEROLLAMA_GPU_PROFILE_CTX` | `0` during L1 8k calibrate — **why:** profile `-c 32768` false-fails single-stream |
| `ZEROLLAMA_RUNTIME_VRAM_*` | See [Phase 13 VRAM](#phase-13-vram--snapshot) |
| `RUN_E2E_UNLOAD_MODEL` | Fallback when `/api/ps` empty but runner exists |
| `SMOKE_UNLOAD_MAX_WAIT` | ggml teardown wait (default **30s**) |
| `LINUX_RT_CURL_TIMEOUT` | `/health` per attempt (default **15s**) — cold CUDA ~9s |
| `GPU_SNAPSHOT_RECOMMEND` | `0` skips inline snapshot hints |
| `L3_RADIX_OUT` | Radix live JSON path |
| `PHASE15_BUILD_KV_EXT` | `0` when `.so` already built for embed Python |

### `RUN_E2E_*` session flags

| Flag | Script / effect |
|------|-----------------|
| `RUN_E2E_PREFLIGHT=0` | Skip `phase12_golden_ci` — **CT default** |
| `RUN_E2E_L1=1` | `l1_cuda_full_gate.sh` — needs `CUDA_LLAMA_MODEL` |
| `RUN_E2E_L3=1` | `l3_cuda_full_gate.sh` — 9B+ GGUF |
| `RUN_E2E_L3_RADIX=1` | `l3_radix_prefix_smoke.sh` — vendor `/kv/seq-copy` |
| `L3_RUN_RADIX=1` | Same, folded into `l3_cuda_full_gate.sh` |
| `RUN_E2E_L3_SPEC=1` | Spec-decode × L3 policy leg |
| `RUN_E2E_PHASE14=1` | Phase 14 backend smoke |
| `RUN_E2E_PHASE14_SIGNOFF=1` | `phase14_5080_signoff.sh` |
| `RUN_E2E_PHASE15=1` | `phase15_inprocess_signoff.sh` |
| `RUN_E2E_P17=1` | Go → llama-server smoke |
| `RUN_E2E_P17_LINUX_AUTO=1` | Linux `backend.llama_server=auto` |
| `RUN_E2E_EDGE=1` | Phase 16 edge smoke |
| **`RUN_E2E_UPSTREAM_GGUF=1`** | **Bundles P17 + LINUX_AUTO + EDGE**; profile-off restart |
| `RUN_E2E_P17_VISION=1` | Vision smoke (heavy) |
| `RUN_E2E_TRAINING_OPS=1` | Training HTTP — needs `OLLAMA_TRAINING=true` |
| `RUN_E2E_TRAINING_TCP` | Legacy TCP `:9500` ping |
| `RUN_E2E_TOOLS=1` | Tools chat smoke |
| `RUN_E2E_VRAM_CLAMP=1` | Opt-in clamp policy smoke |
| `RUN_E2E_PHASE13_SNAPSHOT=1` | Default on in session |
| `RUN_E2E_PHASE11_CONTENTION=1` | `phase11_5080_contention_smoke.sh` — chat+low admission under load |

---

## Status matrix (Jun 28 2026 re-sign-off, CT 1564)

| Track | Gate | Status |
|-------|------|--------|
| Phase 11–13 | `gpu_5080_session.sh` | **PASS** |
| Phase 11 contention | `phase11_5080_contention_smoke.sh` | **PASS (Jul 2026)** — normal 2/2, low 6/6, 0×503; min-free 1 GiB + reserve 2 GiB; **keep code defaults** |
| Phase 13 calibrate | load `eliza-1-9b-256k` @ 8k + `gpu_phase13_snapshot.sh` | **PASS (Jul 2026)** — observed ~5.44 GiB, factor **0.739** → autotune persist; **no** global `VRAM_ESTIMATE_FACTOR`; clamp stays off |
| L1 | `l1_cuda_full_gate.sh` | **PASS (concurrent)** — single-stream **−5%** @ 8k (np=2 overhead); concurrent **+~16–20%** @ 8k |
| L3 | `l3_cuda_full_gate.sh` / production @ 27k | **PASS** |
| Phase 15 | `phase15_inprocess_signoff.sh` | **PASS** |
| Phase 14 | `phase14_5080_signoff.sh` | **PASS** (historical) |
| L2 @ 8k | `l2_cuda_full_gate.sh` / Metal A/B | **FAIL merge** (stock wins — expected); **Done** as VRAM opt-in on pin `86d86ed4` |
| Radix live | `l3_radix_prefix_smoke.sh` | **PASS** (Mac + 5080 Jun 2026) — donor **10.6s** → target **0.66s** on CT 1564; `RUN_E2E_L3_RADIX=1` |
| Phase 17 P17 | `phase17_llama_server_smoke.sh` | **PASS** |
| Phase 17 Linux auto | `phase17_linux_auto_smoke.sh` | **PASS** |
| Phase 16 edge CUDA | `phase16_edge_smoke.sh` | **PASS** (`P17_NUM_PREDICT=32`) |
| `RUN_E2E_UPSTREAM_GGUF=1` bundle | full session wrapper | **PASS** — auto-restarts serve profile-off before base smokes (fixes qjl1_256 × 1B after L1/L2) |
| Phase 17 L2 pin merge | criterion #7 | **Partial** (defaults stay L1) |
| **Tier F RotorQuant** | `l2_rotorquant_ab.sh` | **No-merge (Jul 2026)** — planar/iso abort on SET_ROWS; TBQ holds as VRAM opt-in |
| Bee B0 loop-guard | patch **0087** | **Landed** — Mac smoke PASS; optional CUDA sanity |

**Not required on 5080:** `gpt-oss:20b` harmony real-weight (~40+ GiB host RAM); Mac `metal_signoff.sh`.

---

## Troubleshooting (CT 1564)

| Symptom | Likely cause | Fix |
|---------|----------------|-----|
| `cpp-httplib/httplib.h: No such file` | Minimal checkout, no vendor httplib | `rsync` from sibling `llama.cpp`; or `RUN_E2E_PREFLIGHT=0` for GPU-only gate |
| `cannot find -lc++` | Debian uses libstdc++ | Rebuild with `-lstdc++` in `llama/llama.go` LDFLAGS |
| `nvidia-smi` driver/library mismatch | CT userspace ≠ host kernel module | Pin `libnvidia-ml1=590.48.01-1` (CT 1564) |
| Serve exits immediately / no `:8080` listener | Copied `serve_gpu_example.sh` to `~/bin` — `_ROOT=$HOME` | `cp scripts/serve/serve_production_wrapper.sh ~/bin/serve.sh`; tail `/tmp/zerollama-serve.log` |
| `ModuleNotFoundError: uvicorn` | Embed without runtime venv on `PYTHONPATH` | Use `~/bin/serve.sh` (wrapper → in-repo example sets `runtime/.venv/...` before training site-packages). `RUNTIME_UV_SYNC=1 ./scripts/runtime/runtime_uv_venv.sh` |
| `training worker not started` / `.venv-training/lib/python3.10/…` | Training venv ABI ≠ embedded libpython | Rebuild with [`training_embed_build_env.sh`](../scripts/training/training_embed_build_env.sh) **or** recreate venv for `$(./scripts/training/training_uv_venv.sh --embed-py)`; restart `~/bin/serve.sh`. Doc: [gpu-training.md](./gpu-training.md#installing-python-deps-embedded-interpreter) |
| Duplicate training venvs eating disk | Legacy `venv-training/` + `.venv-training-py310.bak` after 3.11 migration | `rm -rf venv-training/ .venv-training-py310.bak`; keep only `.venv-training/` (~7 GiB saved per removed tree) |
| `/health` timeout / serve “failed to start” | Cold CUDA health ~9s; old 2s curl | Wait with `curl -m 15`; see `LINUX_RT_CURL_TIMEOUT` |
| 502/503 runtime smokes, ggml still in `/api/ps` | Broker never ran; stale runner | Smokes retry API unload (`keep_alive:0`); free ports and restart serve |
| Phase 15 multiseq hang | Stale `:8081` after serve restart | `kill $(pgrep -xo zerollama)` + `fuser -k 8080/tcp 8081/tcp` before sign-off |
| `_kv_native` / setuptools ext-modules fail | Old setuptools in runtime venv | `uv pip install --python runtime/.venv/bin/python 'setuptools>=75'` |
| `nm … llama_memory_kv_` empty | Unpatched libllama | [Build patched libllama](#build-patched-libllama-phase-15--sm_120) |
| Phase 15 `kv_inprocess_n_seq_max=4` not 2 | L1 profile overrides yaml slots | Multiseq smoke sets `ZEROLLAMA_GPU_PROFILE=0` — if manual serve, export it |
| `free(): invalid pointer` wheel GPU | pip wheel vs pinned libllama skew | Use **inprocess** for CUDA GPU, not wheel GPU |
| Edge smoke `empty response` | Default `num_predict=8` too short | `P17_NUM_PREDICT=32` |
| `RUN_E2E_UPSTREAM_GGUF=1` bundle FAIL | Stale serve still on rtx-5080 fork KV | `gpu_5080_session.sh` restarts profile-off when bundled; or manual `ZEROLLAMA_GPU_PROFILE=0 ZEROLLAMA_LLAMA_FORK=0 5080_start_serve` |
| L1 single-stream **−39%** false FAIL | Profile ON emitted `-c 32768` during 8k calibrate | `l1_cuda_calibrate.sh` sets `ZEROLLAMA_GPU_PROFILE_CTX=0`; rerun gate |
| L1 concurrent **0 tok/s** / `llama-server exited early` | `LLAMA_CPP_LIB` sibling `.so` with vendor `llama-server` | Scripts call `l1_export_llama_binary_env` — vendor bin + matching `libllama.so`; `5080_stop_serve` before L1 |
| L1 sidecar 502 + `go-coordination` spam | Embedded `zerollama serve` still on `:8081` | `5080_stop_serve` (SIGKILL + `:8082`) before `l1_cuda_full_gate.sh` |
| `/health` hang with training + embed | Shared CPython GIL | `ZEROLLAMA_RUNTIME_SHARED_PYTHON=1`; see [shared-interpreter-health-hang.md](./bugs/shared-interpreter-health-hang.md) |

---

## After a green re-sign-off

1. **Production quant:** ~~one real load so autotune persists~~ **Done (Jul 2026)** — `eliza-1-9b-256k` factor **0.739** in `~/.cache/zerollama/vram_autotune.json`; snapshot `/tmp/phase13-5080-eliza9b.json`.
2. **Optional clamp:** `ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX=auto` only if you accept automatic `num_ctx` lowering in API responses.
3. **Do not** copy smoke-only global `VRAM_ESTIMATE_FACTOR` when autotune persist is on (health report agrees).
4. **Phase 11 under load:** ~~contention smoke~~ **Done** — keep backlog/reserve defaults.
5. **Jul 2026 continue:** Tier F RotorQuant **no-merge** (SET_ROWS abort); Bee **B1** (**0102**) opt-in A/B vs static draft-n-max; optional TQ3 FP4 only with TQ3 weights; kill orphan `llama/llama.cpp` servers if free VRAM looks ~1 GiB with `llama_server=false`.

---

## Code map

| Piece | Path |
|-------|------|
| Session env | `scripts/gpu/5080_env.sh` |
| Re-sign-off driver | `scripts/gpu/5080_resignoff.sh` |
| Base gate wrapper | `scripts/gpu/gpu_5080_session.sh` |
| VRAM unload | `scripts/runtime/runtime_smoke_lib.sh` |
| Phase 13 snapshot | `scripts/gpu/gpu_phase13_snapshot.sh` |
| Snapshot hints | `runtime/runtime/gpu_snapshot.py` |
| VRAM YAML defaults | `runtime/runtime/vram_yaml_defaults.py`, `runtime/configs/single_gpu.yaml` |
| L1 profile | `runtime/configs/gpu/rtx-5080.json` |
| Phase 14 smokes | `scripts/phase/phase14_serve_env.sh`, `phase14_backend_smoke.sh`, `phase14_5080_signoff.sh` |
| Phase 15 smokes | `scripts/phase/phase15_inprocess_signoff.sh`, `phase15_runtime_kv_env.sh` |
| L2 / L3 gates | `scripts/phase/l2_cuda_full_gate.sh`, `l3_cuda_full_gate.sh`, `l3_radix_prefix_smoke.sh` |
| RotorQuant A/B | `scripts/phase/l2_rotorquant_ab.sh` |
| Fork labs | `docs/llama-fork-watchlist.md`, `docs/gpu-profiles-l2.md` |
| Eliza fork build | `scripts/build/build_eliza_llama_server.sh` |
| Phase 17 / edge | `scripts/phase/phase17_llama_server_smoke.sh`, `phase17_linux_auto_smoke.sh`, `phase16_edge_smoke.sh` |
| Production serve | `scripts/serve/serve_production_wrapper.sh` → `scripts/serve/serve_gpu_example.sh` |
| Phase 8 broker | `server/vram/broker.go` |
| Vendor sync | `Makefile.sync` → `vendor/llama-cpp-<pin>` |
| Patch doctor | `scripts/vendor/llama_patch_doctor.sh` |

---

## Optional: MLX imagegen

**Model:** `x/z-image-turbo` (~12 GB) — **separate stack** from ggml/runtime (MLX subprocess `libmlxc.so`). Not part of re-sign-off tiers 0–4.

**One-time sm_120 build (inside CT):**

```bash
apt install -y libopenblas-dev liblapacke-dev
export PATH=/usr/local/cuda-12.8/bin:$PATH
cmake -B build-mlx --preset "MLX CUDA 12" \
  -DMLX_CUDA_ARCHITECTURES=120-real \
  -DBLAS_INCLUDE_DIRS=/usr/include/x86_64-linux-gnu \
  -DLAPACK_INCLUDE_DIRS=/usr/include
./scripts/mlx/patch_mlx_c_array.sh
./scripts/mlx/patch_mlx_c_debug_cleanup.sh
./scripts/mlx/patch_mlx_cuda_vram.sh
cmake --build build-mlx --target mlx --target mlxc --parallel
sudo mkdir -p /usr/lib/ollama/mlx_cuda_v12
sudo cp -a dist/lib/ollama/mlx_cuda_v12/* /usr/lib/ollama/mlx_cuda_v12/
```

**Serve:** `OLLAMA_LIBRARY_PATH` includes `/usr/lib/ollama/mlx_cuda_v12` (already in `serve_gpu_example.sh`).

**Generate:**

```bash
OLLAMA_HOST=127.0.0.1:8080 zerollama pull x/z-image-turbo
OLLAMA_HOST=127.0.0.1:8080 zerollama stop <other-model>   # needs ~12 GiB alone
OLLAMA_HOST=127.0.0.1:8080 zerollama run x/z-image-turbo "a sunset over mountains"
```

**16 GiB default:** 384×384 long edge — 1024² OOMs; override `ZEROLLAMA_IMAGE_MAX_SIDE` only with measured headroom.

| Symptom | Fix |
|---------|-----|
| `mlx eval failed (ret=1)` | Stop other models; rerun `patch_mlx_cuda_vram.sh` rebuild |
| `completed without image data` | Serve log / NDJSON `error:`; ensure no stale runner kill |
| Wrong resolution | Rebuild — dimensions resolve in MLX subprocess only |

Deep dive (architecture only): [imagegen-zimage-turbo.md](./imagegen-zimage-turbo.md).

---

## Appendix (optional reads)

| Doc | When |
|-----|------|
| [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md) | Historical duplicate — use this runbook instead |
| [ROADMAP.md](./ROADMAP.md) | Phase exit criteria, product backlog |
| [testing-smoke.md](./testing-smoke.md) | Full script catalog beyond 5080 |
| [apple-silicon-metal.md](./apple-silicon-metal.md) | Mac `metal_signoff.sh` |
| [phase16-thin-edge.md](./phase16-thin-edge.md) | `--edge` semantics |
| [phase17-llama-server.md](./phase17-llama-server.md) | Upstream GGUF routing detail |
