# GPU operator guide — 5080-class single-GPU hosts

> **Daily re-sign-off:** `source scripts/5080_env.sh` → `./scripts/5080_resignoff.sh` — see **[5080-runbook.md](./5080-runbook.md)**.  
> **This file:** extended reference only (VRAM, MLX, production serve, code map).

**Quick checklist (primary):** [5080-runbook.md](./5080-runbook.md) — build, serve, tiers 0–5, troubleshooting.

**Related:** [testing-smoke.md](./testing-smoke.md) (script reference), [phase11-runtime-admission.md](./phase11-runtime-admission.md) (who gets the GPU when busy), [phase13-runtime-vram.md](./phase13-runtime-vram.md) (estimate/clamp/autotune), [scheduling-vram-policy.md](./scheduling-vram-policy.md) (full stack), [upstream-ollama-diff.md](./upstream-ollama-diff.md) (upstream default is Go→llama-server, no Python runtime).

---

## Why this guide exists

Zerollama runs **three VRAM consumers** on one card:

1. **Go ggml runners** (`zerollama runner …`) — legacy path, tools families still on ggml for some models.
2. **Python runtime** (`llama-server` subprocess) — default text path when runtime proxy/embed is on.
3. **Embedded training** — same process, separate job queue.

Phase 8 added **automatic handoff** in Go (`server/vram`), but smokes and operators still hit confusing failures when:

- A **stale ggml runner** blocks runtime load while Go returns **503** before the broker runs.
- **VRAM estimates** are wrong until probe calibration runs per GGUF.
- Operators chase **`gpt-oss:20b` harmony** on a host that lacks **host RAM** for MXFP4 mmap (~40+ GiB), not VRAM.

This guide documents the **5080 session** workflow and the **why** behind each script and Python module added for measurement—not a second scheduling system.

---

## Official gate: `gpu_5080_session.sh`

**On CT 1564:** `source scripts/5080_env.sh` first (or `./scripts/5080_resignoff.sh` for the full driver). Env, build, and tier order live in [`5080-runbook.md`](./5080-runbook.md) — not here.

```bash
source ./scripts/5080_env.sh
5080_start_serve
./scripts/gpu_5080_session.sh
```

**Why one script:** CI covers Go Golden + pytest without a GPU; a single GPU host needs a repeatable preflight → smoke → snapshot → recommendations loop so Phase 11/13 tuning is evidence-based, not guesswork.

Override paths after `source ./scripts/5080_env.sh` if your GGUF or `llama-server` differ from CT 1564 defaults.

**What it runs (in order):**

| Step | Why |
|------|-----|
| `phase12_golden_ci.sh` | Parser/render correctness without GPU (Harmony included synthetically in Go tests). |
| `gpu_smoke_all.sh` | Coordination mirror, VRAM prep, runtime + proxy e2e, health report. |
| `gpu_phase13_snapshot.sh` | JSON artifact for before/after tuning (`/tmp/5080-session.json`). |
| `python -m runtime.gpu_snapshot` | Human-readable env hints from that JSON (autotune persist, harmony skip). |
| `phase14_backend_smoke.sh` (optional) | When `RUN_E2E_PHASE14=1` — serve must already use the target backend. With `RUN_E2E_INPROCESS=1` alone, runs `phase14_inprocess_smoke.sh` (env provenance). |
| `phase14_5080_signoff.sh` (optional) | When `RUN_E2E_PHASE14_SIGNOFF=1` — both backends + YAML config + Phase 15 multi-seq (`LLAMA_CPP_LIB` required). |
| `phase15_inprocess_signoff.sh` (optional) | When `RUN_E2E_PHASE15=1` — KV decode hook + multi-seq (`LLAMA_CPP_LIB` required). |
| `phase15_inprocess_multiseq_smoke.sh` (optional) | Multi-seq only (included in signoff). |
| `phase17_llama_server_smoke.sh` (optional) | When `RUN_E2E_P17=1` — Go → llama-server generate (`LLAMA_SERVER_BIN` + pulled tag). |
| `phase16_edge_smoke.sh` (optional) | When `RUN_E2E_EDGE=1` — `serve --edge`, runtime chat off (`LLAMA_SERVER_BIN` + pulled tag). |

**Pass criteria:** `PASS: gpu_5080_session` and snapshot file written. Smoke GGUF calibration (e.g. ~1.20× for OuteTTS Q8) is **smoke evidence only** until you run the same flow on your **production** GGUF (e.g. supernova fp16).

Optional L1/L3 production gates (defaults from `5080_env.sh` — override `CUDA_LLAMA_MODEL` if needed):

```bash
source ./scripts/5080_env.sh
RUN_E2E_L1=1 RUN_E2E_L3=1 ./scripts/gpu_5080_session.sh
# Radix cross-slot (vendor llama-server — 5080_build_vendor_llama_server first):
RUN_E2E_L3_RADIX=1 ./scripts/gpu_5080_session.sh
# Or standalone:
./scripts/l1_cuda_full_gate.sh
./scripts/l3_cuda_full_gate.sh
L3_RADIX_LIVE=1 ./scripts/l3_radix_prefix_smoke.sh
```

**Proxmox CT / minimal checkout — skip Go preflight:** already set by `5080_env.sh` (`RUN_E2E_PREFLIGHT=0`). On non-CT hosts without sourcing, export it explicitly.

**Optional Phase 14 in one session** (after ctypes smoke passes on serve):

```bash
RUN_E2E_PHASE14=1 RUN_E2E_INPROCESS=1 ./scripts/gpu_5080_session.sh
```

**Full Phase 14/15 sign-off** (self-contained serve restarts; ~15–20 min):

```bash
source ./scripts/5080_env.sh
5080_build_patched_libllama   # if nm lacks kv-ext symbols
RUN_E2E_PHASE14_SIGNOFF=1 ./scripts/gpu_5080_session.sh
```
```

**Phase 15 in-process sign-off** (KV decode hook + multi-seq; self-contained restarts):

```bash
export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so
RUN_E2E_PHASE15=1 ./scripts/gpu_5080_session.sh
# or standalone:
./scripts/phase15_inprocess_signoff.sh
```

**Optional Phase 17 / Phase 16 edge (Go → llama-server, upstream shape):**

```bash
export LLAMA_SERVER_BIN=/path/to/llama-server
export RUN_E2E_PROXY_MODEL=llama3.2:3b
RUN_E2E_P17=1 ./scripts/gpu_5080_session.sh
RUN_E2E_EDGE=1 ./scripts/gpu_5080_session.sh
RUN_E2E_P17_LINUX_AUTO=1 ./scripts/gpu_5080_session.sh   # plain serve, asserts backend.llama_server=auto
RUN_E2E_UPSTREAM_GGUF=1 ./scripts/gpu_5080_session.sh    # bundle P17 + P17_LINUX_AUTO + EDGE
RUN_E2E_P17_VISION=1 P17_VISION_MODEL=llava:latest ./scripts/gpu_5080_session.sh  # opt-in vision (heavy)
```

---

## Phase 15 CUDA libllama + sign-off

**Status:** **PASS (Jun 2026, CT 1564 / cudallama)** on RTX 5080 — OuteTTS 1B Q8, sibling `libllama.so` with kv-ext + decode loop.

**Why separate from L1 gate:** Phase 15 needs a **patched** libllama (kv-ext symbols + optional `ZEROLLAMA_KV_DECODE_LOOP=1` ext build). L1 profile autotune (`rtx-5080.json`) is orthogonal; multiseq sign-off must **disable** L1 override on serve (`ZEROLLAMA_GPU_PROFILE=0`) so yaml `llama_parallel_slots: 2` maps to `kv_inprocess_n_seq_max=2`, not L1 `-np 4`.

### Build patched libllama (5080 / sm_120)

Host CUDA **12.3** cannot compile **sm_120** (Blackwell). Install **cuda-nvcc-12-8** in the container and point CMake at it:

```bash
# Inside CT (example: pct exec 1564 -- bash)
apt install cuda-nvcc-12-8   # or equivalent for your image
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc

cd ~/llama.cpp && git checkout b9781   # zerollama pin
# Patch 0014 may not apply cleanly to stock b9781 alone — copy from zerollama tree:
#   include/llama-kv-ext.h, src/llama-memory-kv-ext.cpp, kv-cache cell_index changes, CMakeLists
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120-real
cmake --build build -j
nm -D build/bin/libllama.so | grep llama_memory_kv_   # expect four symbols
```

### Sign-off (embed path)

```bash
export LLAMA_MODEL=/root/Llama-OuteTTS-1.0-1B-Q8_0.gguf
export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so
pkill -f 'zerollama serve' || true   # stale :8080/:8081 blocks embed startup
./scripts/phase15_inprocess_signoff.sh
```

**Pass criteria:** `phase15_inprocess_kv_smoke` (`kv_decode_steps>0`, `batch_decode_in_c=True`), `phase15_inprocess_multiseq_smoke` (`kv_inprocess_n_seq_max=2`), `phase15_batch_decode_smoke` (batch + stream via `/internal/generate-batch`).

Multiseq smoke sets `ZEROLLAMA_GPU_PROFILE=0` automatically (same rationale as `phase15_metal_signoff.sh` step 2).

See also: [phase15-native-kv.md](./phase15-native-kv.md), [handoff-phase15-native-kv.md](./handoff-phase15-native-kv.md).

---

## 5080 gate summary (CT 1564, Jun 28 2026 re-sign-off)

| Gate | Result | Notes |
|------|--------|-------|
| Phase 11–13 base | **PASS** | `gpu_5080_session.sh`, `RUN_E2E_PREFLIGHT=0` |
| Phase 15 in-process | **PASS** | KV hook + multiseq + batch decode; OuteTTS 1B Q8 |
| Phase 17 llama-server | **PASS** | `phase17_llama_server_smoke.sh` |
| Phase 17 Linux auto | **PASS** | `phase17_linux_auto_smoke.sh` |
| Phase 16 edge CUDA | **PASS** | `phase16_edge_smoke.sh` (`P17_NUM_PREDICT=32`) |
| L1 autotune | **PASS (concurrent)** | eliza-1 9B @ 8k: concurrent **+~16–20%**; single-stream **−5%** (np=2 overhead) |
| L3 cache (subprocess) | **PASS** | 8k strict + 27k production on eliza-1 9B |
| L2 CUDA (8k) | **FAIL merge** | Stock wins decode — expected; fork profiles opt-in |
| Radix cross-slot live | **PASS** | `L3_RADIX_LIVE=1` on eliza-1 9B — donor **10.6s** → target **0.66s**; vendor `/kv/seq-copy` |
| `RUN_E2E_UPSTREAM_GGUF=1` bundle | **PASS** | Auto-restarts serve profile-off before base smokes; then P17 + Linux auto + edge |

Full checklist: [5080-runbook.md](./5080-runbook.md). Individual L2/L3: [gpu-profiles-l2.md](./gpu-profiles-l2.md), [gpu-profiles-l3.md](./gpu-profiles-l3.md).

---

## Building zerollama (CGO) on Proxmox CT

**Why this section exists:** CT 1564 and other minimal checkouts sync `llama/llama.cpp/vendor/{miniaudio,nlohmann,stb}` into git but **not** `cpp-httplib` — root `.gitignore` matches `vendor/`. CGO compiles `download.cpp` / `httplib_wrap.cpp`, which `#include <cpp-httplib/httplib.h>`. Without the header, `go build` fails before any GPU work runs.

**Symptom:**

```text
fatal error: cpp-httplib/httplib.h: No such file or directory
```

**Fix (fast — sibling llama.cpp checkout):**

```bash
cd ~/zerollama
rsync -a ~/llama.cpp/vendor/cpp-httplib/ llama/llama.cpp/vendor/cpp-httplib/
CGO_ENABLED=1 go build -o zerollama .
sudo cp zerollama /usr/bin/zerollama   # if serve uses /usr/bin/zerollama
```

**Fix (full vendor sync):** clone `vendor/llama-cpp-b9781`, `make -f Makefile.sync apply-patches`, `./scripts/sync_vendor_llama.sh` — see [ggml-b9509-migration.md](./ggml-b9509-migration.md).

**Why `RUN_E2E_PREFLIGHT=0` on GPU gate:** `gpu_5080_session.sh` can skip `phase12_golden_ci.sh` when httplib is missing; CI and full dev hosts still run Go golden tests. GPU smokes should not fail on parser compile in a tree that only ships inference.

**Linux link (`cannot find -lc++`):** Debian CTs link libstdc++, not libc++. Use `-lstdc++` in `llama/llama.go` `#cgo linux LDFLAGS` (not `-lc++`).

**NVML mismatch (Proxmox GPU passthrough):** if `nvidia-smi` reports driver/library version mismatch, align userspace `libnvidia-ml1` with the host kernel module (CT 1564: **590.48.01**). See [5080-runbook.md](./5080-runbook.md#one-time-host-setup).

**Embedded runtime (`ModuleNotFoundError: uvicorn`):** export `PYTHONPATH` to include `runtime/.venv/.../site-packages` before `zerollama serve`. Full build + re-sign-off checklist: [5080-runbook.md](./5080-runbook.md).

---

## Production serve (`~/bin/serve.sh`)

**Why not default `127.0.0.1:11434`:** upstream Ollama binds localhost on port 11434. Remote clients (Ruby `ZEROLLAMA_API_ENDPOINT`, ruby-trivia `OLLAMA_HOST`, Open WebUI, etc.) cannot reach localhost on the GPU box. CT 1564 listens on **`192.168.255.164:8080`** (`eth0` on `vmbr253`).

**Why a wrapper instead of copying `serve_gpu_example.sh`:** the example script assumes it lives in `scripts/` and sets `_ROOT=$(dirname "$0")/..`. Installed as `~/bin/serve.sh`, `..` resolves to **`$HOME`**, not the zerollama checkout — `source scripts/training_uv_venv.sh` fails, `PYTHONPATH` is empty, and serve dies before binding `:8080` (worse when logs redirect to `/tmp/zerollama-serve.log` and the screen looks idle).

**Install (once):**

```bash
cd ~/zerollama
cp scripts/serve_production_wrapper.sh ~/bin/serve.sh
chmod +x ~/bin/serve.sh
```

**Start:**

```bash
~/bin/serve.sh
tail -f /tmp/zerollama-serve.log
```

The wrapper sets `ZEROLLAMA_REPO=${HOME}/zerollama`, `SERVE_LOG=/tmp/zerollama-serve.log`, and `exec`s [`scripts/serve_gpu_example.sh`](../scripts/serve_gpu_example.sh) in-repo. That example sets:

- `OLLAMA_HOST=0.0.0.0:8080` — remote clients
- `ZEROLLAMA_GO_URL=http://127.0.0.1:8080` — embed → Go `/internal/*` on loopback
- **`PYTHONPATH`** — `runtime/.venv/.../site-packages` **before** `.venv-training` (uvicorn + torch ABI)
- vendor **`LLAMA_SERVER_BIN`** when `vendor/llama-cpp-*` is built (fork QJL + Radix seq-copy)

**Required for remote inference:**

```bash
export OLLAMA_HOST=0.0.0.0:8080   # set by serve_gpu_example.sh
zerollama serve
```

Log line should show `Listening on [::]:8080` or `0.0.0.0:8080` — **not** `127.0.0.1:11434`.

**Embedded runtime stays loopback:** Go embeds Python runtime on `127.0.0.1:8081` (`ZEROLLAMA_RUNTIME_EMBED_PORT`). Remote clients talk to **`:8080` only**; they must not point at `:8081`.

**Why log redirect:** screen/tmux sessions stay quiet; GIN + runner logs accumulate in one file. **Watch live:** `tail -f /tmp/zerollama-serve.log`. **Trade-off:** no stdout in the screen — set `SERVE_LOG=` empty and use `tee` if you want both (see comment in `serve_gpu_example.sh`).

**After `git pull` + rebuild:** restart `~/bin/serve.sh` so `/usr/bin/zerollama` picks up the new binary.

**Smoke from another host:**

```bash
curl -s http://192.168.255.164:8080/api/tags
curl -s http://192.168.255.164:8080/v1/models
```

**Production sanity (on the GPU box):**

```bash
curl -s http://127.0.0.1:8080/api/version | jq .
curl -s http://127.0.0.1:8081/health | jq '{status, profile: .gpu_profile.id, backend: .llama_backend}'
curl -s http://127.0.0.1:8080/api/train/status | jq .
```

---

## L1 GPU profiles (CUDA autotune)

**Why:** Phase 13 (below) estimates fit and suggests `num_ctx`. **L1** merges eliza-derived llama-server flags — `-b`, `-ub`, `-np`, `-fa`, cache types, MTP `draft_*` — when `single_gpu.yaml` loads on a CUDA host.

**Detection:** `nvidia-smi` GPU name → JSON `match_names` (e.g. RTX 4090 → `4090.json`); else VRAM bucket in `runtime/configs/gpu/index.json` (16 GiB → `rtx-5080` profile).

**Verify:**

```bash
curl -s http://127.0.0.1:8081/health | jq '.gpu_profile, .llama_args'
```

**Disable / override:** `ZEROLLAMA_GPU_PROFILE=0`; `ZEROLLAMA_GPU_PROFILE_CTX=0` to skip profile `-c`; `LLAMA_SERVER_EXTRA_ARGS` appended last.

**5080 L1 gate (Jun 2026, CT 1564 — calibrated):**

| Check | Result |
|-------|--------|
| Profile detection | **PASS** — `rtx-5080`, `n_parallel=2`, `source=match` |
| `test_gpu_profiles.py` | **PASS** |
| `gpu_smoke_all` + snapshot | **PASS** |
| Single-stream A/B @ 8k | **−5%** (informational) — eliza-1 9B OFF **~90** vs ON **~85** tok/s (`np=1` vs `np=2`; calibrate uses `ZEROLLAMA_GPU_PROFILE_CTX=0`) |
| Concurrent A/B @ 8k (`L1C_N=2`) | **PASS** — ON **~65** vs OFF **~55** agg tok/s (**+~16–20%**) |

**Tune workflow:** `./scripts/l1_cuda_full_gate.sh` (calibrate + concurrent + verdict), or run `./scripts/l1_cuda_calibrate.sh` and `./scripts/l1_cuda_concurrent_bench.sh` separately; edit `runtime/configs/gpu/rtx-5080.json`; rerun. **Session wrapper:** `RUN_E2E_L1=1 ./scripts/gpu_5080_session.sh`. **Why `np=2`:** L3 agent cache needs ≥2 slots; `np=4` wasted KV on single-stream / 1B smoke.

Full doc: [gpu-profiles-l1.md](./gpu-profiles-l1.md).

Individual smokes (optional):

```bash
./scripts/phase15_inprocess_kv_smoke.sh
./scripts/phase15_inprocess_multiseq_smoke.sh   # num_ctx must fit PA block pool (4096 in smoke)
```

---

## VRAM prep: why API unload before the broker

**Problem:** `smoke_prepare_vram_for_runtime` triggers Phase 8 by posting `X-Zerollama-Runtime` to `/api/generate`. If Go returns **503** (training busy, admission) **before** the broker path, ggml runners may **stay loaded** and runtime smokes fail with 502/503.

**Why not `pkill`:** Killing runners bypasses the public contract, races with in-flight loads, and does not appear in scheduler metrics operators already use.

**Solution:** `smoke_unload_ggml_runners` in `scripts/runtime_smoke_lib.sh`:

1. Detect child only: `pgrep -f '/zerollama runner --'` — **why:** avoid matching shell lines that contain the words `zerollama runner`.
2. List models from `/api/ps`; unload each with `prompt:""` + `keep_alive:0` — **why:** same unload path as documented operator API.
3. Wait up to `SMOKE_UNLOAD_MAX_WAIT` (default **30s**) — **why:** large models tear down slowly; 15s was too tight on real hosts.
4. On 503 + runner still up: retry unload, `runtime_resume_if_needed`, broker once more.

`gpu_harmony_capture.sh` uses the same unload — **why:** harmony capture needs a clean card without manual `pkill`.

---

## Phase 13: YAML defaults + snapshot recommendations

### `single_gpu.yaml` `vram:` block

**Why YAML, not only env:** Autoconfig already picks `single_gpu.yaml` when one GPU is visible. Putting Phase 11/13 **defaults in-repo** gives new installs sensible headroom without copying a long env block into systemd—while **env still wins** when set.

Applied at runtime start by `vram_yaml_defaults.py` (before optional `VRAM_APPLY_EXPORTED_ENV` file load):

| YAML key | Default | Why |
|----------|---------|-----|
| `min_free` | `1GiB` | Admission floor when GPU checks on. |
| `training_reserve` | `2GiB` | Headroom while training/handoff holds the card. |
| `estimate_factor_autotune` | `auto` | Per-GGUF calibration beats one global factor. |
| `probe_calibrate` | `auto` | Record observed VRAM after load for `/health`. |
| `clamp_num_ctx` | `"0"` | **Off** by default — **why:** silent context reduction surprised operators; enable `auto` only when you accept `vram_num_ctx` in API responses. |

### `python -m runtime.gpu_snapshot`

**Why:** `/health` and `gpu_health_report.sh` are rich but ephemeral. The snapshot JSON is a **portable record**; `gpu_snapshot` turns it into copy-paste hints without re-querying a live server.

**Rules (WHY-oriented):**

- If autotune has `effective_factor` + `persist` → **do not** set a global `VRAM_ESTIMATE_FACTOR` (per-GGUF persist wins at load).
- If autotune off but `suggested_estimate_factor` in range → export one global factor (smoke hosts without persist).
- Warn if either budget says `fits_with_margin=false`.
- Always note: **harmony real-weight** skipped on ~19 GiB host — use `./scripts/phase12_golden_ci.sh` instead.

`gpu_phase13_snapshot.sh` includes `vram_autotune.persist` in JSON and sets `ZEROLLAMA_REPO_ROOT` so recommendations work from repo root without manual `PYTHONPATH`.

---

## Harmony and host RAM (not VRAM)

| Path | `gpt-oss:20b` MXFP4 |
|------|---------------------|
| Runtime load | Often **~44 GiB host RAM** for mmap budget check — fails on ~19 GiB CT. |
| CI | `TestGoldenHarmonyParseToolOutput` — parser correctness without weights. |
| 5080 GPU gate | `gpu_5080_session.sh` — **does not require** harmony capture. |

**Why document explicitly:** VRAM-only thinking sends operators to `gpu_harmony_capture.sh` on hardware that cannot succeed; the product still ships Harmony via Go parser tests.

---

## Environment quick reference

| Variable | Role |
|----------|------|
| `OLLAMA_HOST` | Go daemon (`:8080`) for proxy smokes and unload API. |
| `ZEROLLAMA_RUNTIME_URL` | Python runtime (`:8081`) when not embedded. |
| `LLAMA_MODEL` / `RUN_E2E_GGUF` | Smoke GGUF path. |
| `RUN_E2E_PROXY_MODEL` | Pulled name for manifest proxy checks. |
| `RUN_E2E_TRAINING_OPS` | With `gpu_smoke_all` / `gpu_5080_session`: run `e2e_training_ops_smoke.sh` after coordination. |
| `RUN_E2E_TRAINING_TCP` | Also ping legacy TCP (`OLLAMA_TRAINING_TCP`, default `:9500`). |
| `RUN_E2E_UNLOAD_MODEL` | Fallback when `/api/ps` empty but runner process exists. |
| `SMOKE_UNLOAD_MAX_WAIT` | Seconds to wait for ggml teardown (default 30). |
| `GPU_PHASE13_SNAPSHOT_OUT` | Snapshot JSON path (default `/tmp/5080-session.json`). |
| `RUN_E2E_PREFLIGHT` | `1` (default in `gpu_5080_session.sh`) runs `phase12_golden_ci.sh` first; `0` skips Go CGO golden — **why:** Proxmox CT often lacks vendored `cpp-httplib`; GPU smokes should not fail on parser compile. |
| `RUN_E2E_L1` | `1` runs `l1_cuda_full_gate.sh` (needs `CUDA_LLAMA_MODEL` / `LLAMA_MODEL`). |
| `RUN_E2E_L3` | `1` runs `l3_cuda_full_gate.sh` (needs 9B+ production GGUF). |
| `OLLAMA_HOST` | **Bind address for Go API.** Default upstream `127.0.0.1:11434` — use `0.0.0.0:8080` for remote clients. |
| `LINUX_RT_CURL_TIMEOUT` | Sidecar `/health` wait per attempt (default **15s**). **Why:** cold CUDA health probe ~9s; old 2s curl caused false startup failure. |
| `GPU_SNAPSHOT_RECOMMEND` | `0` to skip inline recommendations in snapshot script. |

---

## Phase 14 sign-off (in-process llama)

**Why often run separately:** Phase 14 validates ctypes `libllama.so` in the embedded runtime (no loopback `llama-server`). Serve must be restarted with `ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess` (or YAML). You can fold it into `gpu_5080_session.sh` with `RUN_E2E_PHASE14=1` once serve is on that backend. Use a **small Q8 GGUF** on the card you ship.

### Checklist (5080)

1. Rebuild: `CGO_ENABLED=1 go build -o zerollama .` from this repo.
2. Source embed env (do **not** leave `ZEROLLAMA_RUNTIME_URL` set):

```bash
source ./scripts/phase14_serve_env.sh
export LLAMA_MODEL=/path/to/small.q8_0.gguf
export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so
export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess
./zerollama serve
```

3. Other terminal — GPU inprocess smoke + render tokenize:

```bash
export LLAMA_MODEL=/path/to/same.gguf
export RUN_E2E_PROXY_MODEL=<pulled-local-tag>   # for /internal/render-chat
./scripts/phase14_inprocess_smoke.sh
```

**Pass:** `PASS: phase14_backend_smoke`, `/health` shows `llama_backend=inprocess`, `llama_backend_source=env`, `llama_server=false`, render-chat `truncate_mode=tokenize`.

Equivalent manual flags: `RUN_E2E_INPROCESS=1 ./scripts/phase14_backend_smoke.sh`.

### Optional: inprocess via YAML (no env override)

After ctypes GPU smoke passes with env, you can pack the default into autoconfig:

1. Uncomment `llama_backend: inprocess` in `runtime/configs/single_gpu.yaml`.
2. Restart serve **without** `ZEROLLAMA_RUNTIME_LLAMA_BACKEND`.
3. Confirm provenance:

```bash
RUN_E2E_LLAMA_BACKEND_SOURCE=config ./scripts/phase14_yaml_config_smoke.sh
```

`/health` should show `llama_backend_source=config`. Invalid YAML values fail at config load with a clear error (`canonical_llama_backend`).

Enable YAML in one step after env-based inprocess smoke passes:

```bash
./scripts/phase14_enable_yaml_inprocess.sh
# restart serve without ZEROLLAMA_RUNTIME_LLAMA_BACKEND
RUN_E2E_LLAMA_BACKEND_SOURCE=config ./scripts/phase14_yaml_config_smoke.sh
```

### Optional: wheel CPU sign-off

After inprocess passes, or standalone if you only need the pip wheel path:

```bash
# serve: ZEROLLAMA_RUNTIME_LLAMA_BACKEND=llama-cpp-python
export LLAMA_MODEL=/path/to/same.gguf
./scripts/phase14_wheel_cpu_smoke.sh
```

Or run both backends with restarts: `./scripts/phase14_both_backends.sh`.

**One-shot 5080 gate** (both backends + YAML config + Phase 15 multi-seq; self-contained serve restarts):

```bash
export LLAMA_MODEL=/path/to/small.q8_0.gguf
export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so
./scripts/phase14_5080_signoff.sh
```

### Optional: wheel GPU smoke

After CPU wheel smoke passes (`phase14_both_backends`), optional GPU offload on hosts where the pip wheel is stable:

```bash
# serve: ZEROLLAMA_RUNTIME_LLAMA_BACKEND=llama-cpp-python
#         ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS=99
export LLAMA_MODEL=/path/to/same.gguf
./scripts/phase14_wheel_gpu_smoke.sh
```

### Optional: both in-process backends

```bash
export LLAMA_MODEL=... RUN_E2E_PROXY_MODEL=...
./scripts/phase14_both_backends.sh
```

| Backend | 5080 expectation |
|---------|------------------|
| **inprocess** (ctypes) | GPU — primary sign-off |
| **llama-cpp-python** (wheel) | CPU default (~10 min smoke); GPU only if `ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS` set and stable on your wheel |

**Do not** use wheel GPU on 5080 for production until the pip wheel passes `create_completion` with offload on your host (some cu124 builds abort with `free(): invalid pointer`).

Doc: [phase14-inprocess-llama.md](./phase14-inprocess-llama.md).

---

## Proxmox / CT layout (why run gates inside the container)

**Why document this:** Operators often SSH to the Proxmox **host** (`pct status` works) while GPU passthrough and CUDA toolchains live in an **LXC** (e.g. CT **1564**, hostname `cudallama`). The host may have an older `nvcc` (12.3) that **cannot** compile `sm_120` for RTX 5080; the CT needs **CUDA 12.8+** `nvcc` and all sign-off commands should run **inside** the CT.

```bash
# From Proxmox host — all GPU gates below use this wrapper
pct exec 1564 -- bash -lc 'cd /root/zerollama && …'
```

| Path | Role |
|------|------|
| Host mount | `/var/lib/vz/private/1564/root/zerollama` (optional) |
| Inside CT | `/root/zerollama` — **use this in scripts** |
| Sibling llama.cpp | `/root/llama.cpp` (pin **b9781** + patch **0014**) |
| Smoke GGUF | e.g. `/root/Llama-OuteTTS-1.0-1B-Q8_0.gguf` (~1B Q8 fits 16 GB) |

**One-time CT setup (WHY each step):**

1. **`cuda-nvcc-12-8`** in the CT — host CUDA 12.3 rejects `compute_120`; 5080 Blackwell needs 12.8+.
2. **Reset sibling `llama.cpp` to tag `b9781`** — copying kv-ext onto `master` breaks `llama-kv-cache.h` vs the rest of the tree.
3. **Apply fork files from zerollama** (patch 0014 may not apply cleanly to stock b9781 alone):

```bash
ZROOT=/root/zerollama
L=/root/llama.cpp
git -C "$L" checkout -f b9781
cp "$ZROOT/llama/llama.cpp/include/llama-kv-ext.h" "$L/include/"
cp "$ZROOT/llama/llama.cpp/src/llama-memory-kv-ext.cpp" "$L/src/"
cp "$ZROOT/llama/llama.cpp/src/llama-kv-cache.{h,cpp}" "$L/src/"
grep -q llama-memory-kv-ext.cpp "$L/src/CMakeLists.txt" || \
  sed -i '/llama-memory-recurrent.cpp/a\            llama-memory-kv-ext.cpp' "$L/src/CMakeLists.txt"
```

4. **Rebuild libllama** with `CMAKE_CUDA_ARCHITECTURES=120-real`, `BUILD_SHARED_LIBS=ON` (see [phase15-llama-kv-ext-upstream.md](./phase15-llama-kv-ext-upstream.md)).

**Stale serve:** kill `zerollama serve` on `:8080`/`:8081` before Phase 15 embed smokes — a leftover process blocks embed and health checks hit the wrong backend.

---

## Phase 15 + borrowings gates (5080 sign-off sequence)

**Why this order:** pin check proves fork files before a long CUDA build; Phase 15 needs patched `libllama.so`; L2/L3 use **subprocess** `llama-server` (different path than Phase 15 **inprocess**).

| Gate | Script | Jun 2026 (CT 1564, RTX 5080) |
|------|--------|------------------------------|
| 0 Pin | `phase15_llama_kv_ext_pin_check.sh` | PASS in-tree; sibling symbols after rebuild |
| 1 Phase 15 | `phase15_inprocess_signoff.sh` | **PASS** — KV hook, multiseq `n_seq_max=2`, batch decode |
| 2 L2 CUDA | `l2_cuda_full_gate.sh` | **FAIL merge** — 8k stock wins (1B **79.3** / fork **56.9**; 9B **18.6** / **14.4**); **27k** same (~−22%); **131k fork** blocked (9B VRAM; 1B QJL head) |
| 3 L3 cache | `l3_cache_smoke.sh` + `l3_gate_report.sh` | **STRICT PASS** on eliza-1 9B @ 8k (`L3_PREFIX_REPEAT=150`, cached turn2 **0.66s** vs no-cache **1.13s**); 1B Q8 SOFT PASS |
| 3b L3 production | `l3_production_gate.sh` | **PASS** on eliza-1 9B @ 27k — cached **0.72s** vs no-cache **1.48s**; strict ratio **1.02** (open on supernova) |
| 3c L3 spec × cache | `l3_spec_cache_smoke.sh` | Policy leg — `L3_SPEC_METHOD=ngram` → `allow_cache_prompt=true`; eagle3 + draft → disabled. Optional: `L3_RUN_SPEC_CACHE=1` on full gate |
| 4 L1 concurrent | `l1_cuda_concurrent_bench.sh` | **PASS** — `L1C_N=2`: ON **~65** vs OFF **~55** agg tok/s (**+~16–20%** @ 8k) |

### Gate 1 — Phase 15 in-process (CUDA)

```bash
export LLAMA_MODEL=/root/Llama-OuteTTS-1.0-1B-Q8_0.gguf
export LLAMA_CPP_LIB=/root/llama.cpp/build/bin/libllama.so
export LLAMA_CPP_ROOT=/root/llama.cpp
export OLLAMA_HOST=http://127.0.0.1:8080
./scripts/phase15_inprocess_signoff.sh
# PASS: phase15_inprocess_signoff
```

**What it checks:** `kv_decode_steps` increment (native), `kv_inprocess_n_seq_max=2`, `batch_decode_in_c=true`, batch + stream via `/internal/generate-batch`.

**WHY `ZEROLLAMA_GPU_PROFILE=0` in multiseq smoke:** L1 `rtx-5080` sets `n_parallel=2` (tuned); multiseq smoke YAML uses `llama_parallel_slots: 2` — disable profile when testing explicit slot count.

### Gate 3 — L3 agent cache bench

```bash
# Production gate (recommended — 8k + 27k):
export CUDA_LLAMA_MODEL=/root/eliza-1-9b-256k.gguf
./scripts/l3_cuda_full_gate.sh
# or: RUN_E2E_L3=1 ./scripts/gpu_5080_session.sh

# Individual legs:
export L3_PREFIX_REPEAT=150 L3_COMPARE_NO_CACHE=1
L3_OUT=/tmp/l3-cache-smoke-9b.json ./scripts/l3_cache_smoke.sh
./scripts/l3_production_gate.sh
./scripts/l3_gate_report.sh /tmp/l3-cuda-full-gate/gate.json

# 1B wiring smoke (SOFT PASS expected):
export CUDA_LLAMA_MODEL=/root/Llama-OuteTTS-1.0-1B-Q8_0.gguf
./scripts/l3_cache_smoke.sh
```

**Strict gate:** turn 2 faster than turn 1 **or** `cached_faster_than_no_cache` with `L3_COMPARE_NO_CACHE=1`. **Jun 2026 9B @ 8k:** cached **0.66s** vs no-cache **1.13s** on same turn-2 prompt. Doc: [gpu-profiles-l3.md](./gpu-profiles-l3.md).

### Gate 3b — L3 production gate (27k ctx)

Folded into `./scripts/l3_cuda_full_gate.sh`. Individual leg:

```bash
export CUDA_LLAMA_MODEL=/root/eliza-1-9b-256k.gguf
./scripts/l3_production_gate.sh
# PASS: cached faster than no-cache (Jun 2026: 0.72s vs 1.48s)
```

**Why 27k:** agent threads use long `num_ctx`; cache must skip prefill at production window, not just 8k smoke. Strict ratio `turn2/turn1 ≤ 0.75` may fail when turn-1 already warmed the slot — alternate PASS is cached vs no-cache control.

### Gate 4 — L1 concurrent bench

```bash
export CUDA_LLAMA_MODEL=/root/eliza-1-9b-256k.gguf
./scripts/l1_cuda_concurrent_bench.sh
# PASS: profile ON +~16–20% aggregate tok/s vs OFF at L1C_N=2 (Jun 2026 re-measure)
```

Folded into session: `RUN_E2E_PHASE15=1` or `RUN_E2E_PHASE14_SIGNOFF=1 ./scripts/gpu_5080_session.sh`.

### Gate 2 — L2 fork eval (CUDA A/B)

```bash
export CUDA_LLAMA_MODEL=$LLAMA_MODEL
# First time: L2_BUILD_FORK=1 (sets LLAMA_BUILD_WEBUI=OFF on Linux — headless WebUI build fails)
./scripts/l2_cuda_full_gate.sh
./scripts/l2_gate_report.sh /tmp/l2-cuda-gate/bench-*.json
```

Artifacts: `/tmp/l2-cuda-gate/`. **Verdict (Jun 2026):** stock wins decode @ 8192 ctx on 1B Q8; vendor merge still blocked. Optional long-ctx legs: `L2_RUN_27K=1 L2_RUN_131K_FORK=1`. Doc: [gpu-profiles-l2.md](./gpu-profiles-l2.md).

### Optional — tensor bind spot-check

After multiseq sidecar (`kv_inprocess_n_seq_max≥2`):

```bash
source ./scripts/phase15_runtime_kv_env.sh
phase15_runtime_kv_ext_build
./scripts/phase15_tensor_bind_probe.sh
curl -s :8081/health | jq '.kv_page_bind, .kv_decode_loop'
```

Expect `batch_decode_in_c: true`; after generate with active bind: `status: "bound"`, `bind_level: "tensor"` when linked ext is present.

---

## MLX image generation (experimental)

**Model:** `x/z-image-turbo` (~12 GB tensor manifest). **Why separate from ggml:** diffusion runs in the MLX imagegen subprocess (`libmlxc.so`), not the Python runtime or ggml runner. **Why a fourth stack:** manifest safetensors + staged VRAM (encoder → transformer → VAE) do not map onto llama.cpp KV or runtime `llama-server`.

**Full guide:** [imagegen-zimage-turbo.md](./imagegen-zimage-turbo.md) — architecture, troubleshooting, code map.

### One-time MLX build (5080 / sm_120)

```bash
cd /root/zerollama
apt install -y libopenblas-dev liblapacke-dev
export CUDNN_INCLUDE_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/include
export CUDNN_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib
export PATH=/usr/local/cuda-12.8/bin:$PATH

cmake -B build-mlx --preset "MLX CUDA 12" \
  -DMLX_CUDA_ARCHITECTURES=120-real \
  -DBLAS_INCLUDE_DIRS=/usr/include/x86_64-linux-gnu \
  -DLAPACK_INCLUDE_DIRS=/usr/include
cmake --build build-mlx --target mlx --target mlxc --parallel
cmake --install build-mlx --component MLX --strip
sudo mkdir -p /usr/lib/ollama/mlx_cuda_v12
sudo cp -a dist/lib/ollama/mlx_cuda_v12/* /usr/lib/ollama/mlx_cuda_v12/
```

**Why patch before rebuild on 16 GB:**

```bash
# mlx-c/array.cpp: mlx_array_detach + cudaMemcpy D2H latent export
./scripts/patch_mlx_c_array.sh
# mlx-c/array.cpp: remove debug fprintf noise from OOM diagnosis
./scripts/patch_mlx_c_debug_cleanup.sh
# mlx/backend/cuda/allocator.cpp: cudaMalloc, 90% limit, disable recycle
./scripts/patch_mlx_cuda_vram.sh

cmake --build build-mlx --target mlx --target mlxc --parallel
sudo cp -a dist/lib/ollama/mlx_cuda_v12/* /usr/lib/ollama/mlx_cuda_v12/
```

**Why three patches:** they touch two source packages (`mlx-src` vs `mlx-c-src`) and have distinct roles. All are idempotent — safe to re-run after a clean cmake configure.

### Serve env (included in `serve_gpu_example.sh` via `~/bin/serve.sh` wrapper)

**WHY wrapper:** production installs `serve_production_wrapper.sh` as `~/bin/serve.sh`; it `exec`s the in-repo example below. Do not copy the example file itself to `~/bin`.

```bash
export OLLAMA_LIBRARY_PATH=/usr/lib/ollama:/usr/lib/ollama/mlx_cuda_v12
export LD_LIBRARY_PATH=/usr/lib/ollama/mlx_cuda_v12:$LD_LIBRARY_PATH
```

Restart `zerollama serve` after installing libs — the MLX subprocess inherits these from serve.

### Pull and generate

```bash
OLLAMA_HOST=127.0.0.1:8080 zerollama pull x/z-image-turbo   # not library/z-image
OLLAMA_HOST=127.0.0.1:8080 zerollama stop <other-model>       # needs ~12 GB VRAM alone
OLLAMA_HOST=127.0.0.1:8080 zerollama run x/z-image-turbo "a sunset over mountains"
```

**VRAM:** Image model needs the full GPU (~12 GB weights + activations). Stop other loaded runners first (`zerollama ps` / `zerollama stop`). Set `OLLAMA_MAX_LOADED_MODELS=1` on tight hosts.

**Default size on 16 GB CUDA:** 384×384 long edge — **why:** denoise activations scale with pixel count; 1024² OOMs even when weights fit. Override with `ZEROLLAMA_IMAGE_MAX_SIDE` only if you have measured headroom.

### Common failures

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| `mlx eval failed (ret=1)` | OOM during transformer load | Stop other models; run `patch_mlx_cuda_vram.sh` rebuild; lower `ZEROLLAMA_IMAGE_MAX_SIDE` |
| `completed without image data` | Subprocess error after stream started | Check serve log / NDJSON `error:` line; ensure scheduler did not kill in-use runner (fixed: defer unload) |
| Scheduler panic on load | Stale `activeLoading` handle | Rebuild with `clearActiveLoading` fix |
| CPU-only / `/info` panic | ggml props ABI | Rebuild with `ggml_backend_dev_name` fix |
| Wrong resolution | Go pre-clamped before runner | Rebuild — dimensions resolve only in MLX subprocess |

---

## Code map

| Piece | Path |
|-------|------|
| Phase 14 smokes | `scripts/phase14_serve_env.sh`, `phase14_backend_smoke.sh`, `phase14_both_backends.sh` |
| Phase 15 smokes | `scripts/phase15_inprocess_signoff.sh`, `phase15_runtime_kv_env.sh`, `phase15_llama_kv_ext_pin_check.sh` |
| L2 / L3 gates | `scripts/l2_cuda_full_gate.sh`, `l3_cuda_full_gate.sh`, `l3_gate_report.sh` |
| Eliza fork build | `scripts/build_eliza_llama_server.sh` (`LLAMA_BUILD_WEBUI=OFF` on Linux) |
| Unload + broker prep | `scripts/runtime_smoke_lib.sh` |
| 5080 one-liner | `scripts/gpu_5080_session.sh` |
| Snapshot JSON | `scripts/gpu_phase13_snapshot.sh` |
| Recommendations | `runtime/runtime/gpu_snapshot.py` |
| YAML → env | `runtime/runtime/vram_yaml_defaults.py` |
| 16GB defaults | `runtime/configs/single_gpu.yaml` |
| Phase 8 broker | `server/vram/broker.go` |
| MLX imagegen guide | `docs/imagegen-zimage-turbo.md`, `x/imagegen/README.md` |
| MLX VRAM patch | `scripts/patch_mlx_cuda_vram.sh` |

---

## What to do after a green session

1. **Production quant:** Run one real load (or full session) with your serve GGUF so autotune persists under `~/.cache/zerollama/vram_autotune.json`.
2. **Optional clamp:** Enable `ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX=auto` on serve only if you want automatic `num_ctx` lowering.
3. **Do not** copy smoke-only `VRAM_ESTIMATE_FACTOR=1.2` globally unless autotune is off and you have no persist for that file.
4. **Phase 11 thresholds:** Change backlog env overrides only if you measured contention under real chat+training load (5080 session alone only proves “fits at idle smoke”).
