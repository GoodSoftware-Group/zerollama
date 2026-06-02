# GPU operator guide — 5080-class single-GPU hosts

**Audience:** Operators on one consumer GPU (e.g. RTX 5080 ~16 GB VRAM, ~19 GiB host RAM) running embedded or sidecar Python runtime + optional ggml runners + training.

**Related:** [testing-smoke.md](./testing-smoke.md) (script reference), [phase11-runtime-admission.md](./phase11-runtime-admission.md) (who gets the GPU when busy), [phase13-runtime-vram.md](./phase13-runtime-vram.md) (estimate/clamp/autotune), [scheduling-vram-policy.md](./scheduling-vram-policy.md) (full stack).

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

**Why one script:** CI covers Go Golden + pytest without a GPU; a single GPU host needs a repeatable preflight → smoke → snapshot → recommendations loop so Phase 11/13 tuning is evidence-based, not guesswork.

```bash
export OLLAMA_HOST=http://127.0.0.1:8080
export LLAMA_SERVER_BIN=/usr/bin/llama-server
export LLAMA_MODEL=/path/to/small-smoke.gguf    # e.g. 1B Q8 for smoke
export RUN_E2E_GGUF=$LLAMA_MODEL
export RUN_E2E_PROXY_MODEL=llama3.2:3B          # optional manifest proxy checks
# Optional: embedded training HTTP/TCP (serve must have OLLAMA_TRAINING=true)
# RUN_E2E_TRAINING_OPS=1 RUN_E2E_TRAINING_TCP=1
cd /root/zerollama && ./scripts/gpu_5080_session.sh
```

**What it runs (in order):**

| Step | Why |
|------|-----|
| `phase12_golden_ci.sh` | Parser/render correctness without GPU (Harmony included synthetically in Go tests). |
| `gpu_smoke_all.sh` | Coordination mirror, VRAM prep, runtime + proxy e2e, health report. |
| `gpu_phase13_snapshot.sh` | JSON artifact for before/after tuning (`/tmp/5080-session.json`). |
| `python -m runtime.gpu_snapshot` | Human-readable env hints from that JSON (autotune persist, harmony skip). |

**Pass criteria:** `PASS: gpu_5080_session` and snapshot file written. Smoke GGUF calibration (e.g. ~1.20× for OuteTTS Q8) is **smoke evidence only** until you run the same flow on your **production** GGUF (e.g. supernova fp16).

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
| `GPU_SNAPSHOT_RECOMMEND` | `0` to skip inline recommendations in snapshot script. |

---

## Phase 14 sign-off (in-process llama)

**Why separate from `gpu_5080_session`:** Phase 14 validates ctypes `libllama.so` in the embedded runtime (no loopback `llama-server`). Use a **small Q8 GGUF** on the card you ship.

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
RUN_E2E_INPROCESS=1 ./scripts/phase14_backend_smoke.sh
```

**Pass:** `PASS: phase14_backend_smoke`, `/health` shows `llama_backend=inprocess`, `llama_server=false`, render-chat `truncate_mode=tokenize`.

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

## Code map

| Piece | Path |
|-------|------|
| Phase 14 smokes | `scripts/phase14_serve_env.sh`, `phase14_backend_smoke.sh`, `phase14_both_backends.sh` |
| Unload + broker prep | `scripts/runtime_smoke_lib.sh` |
| 5080 one-liner | `scripts/gpu_5080_session.sh` |
| Snapshot JSON | `scripts/gpu_phase13_snapshot.sh` |
| Recommendations | `runtime/runtime/gpu_snapshot.py` |
| YAML → env | `runtime/runtime/vram_yaml_defaults.py` |
| 16GB defaults | `runtime/configs/single_gpu.yaml` |
| Phase 8 broker | `server/vram/broker.go` |

---

## What to do after a green session

1. **Production quant:** Run one real load (or full session) with your serve GGUF so autotune persists under `~/.cache/zerollama/vram_autotune.json`.
2. **Optional clamp:** Enable `ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX=auto` on serve only if you want automatic `num_ctx` lowering.
3. **Do not** copy smoke-only `VRAM_ESTIMATE_FACTOR=1.2` globally unless autotune is off and you have no persist for that file.
4. **Phase 11 thresholds:** Change backlog env overrides only if you measured contention under real chat+training load (5080 session alone only proves “fits at idle smoke”).
