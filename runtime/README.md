# zerollama-runtime

GGUF-first Python inference runtime (PagedAttention KV). See [../docs/python-migration.md](../docs/python-migration.md).

**Scheduling & VRAM (why this exists next to Go):** [../docs/scheduling-vram-policy.md](../docs/scheduling-vram-policy.md) — full stack. **Phase 11 admission (who gets the GPU):** [../docs/phase11-runtime-admission.md](../docs/phase11-runtime-admission.md). **Phase 13 estimates (how much VRAM):** [../docs/phase13-runtime-vram.md](../docs/phase13-runtime-vram.md). **Phase 14 forward (subprocess vs in-process):** [../docs/phase14-inprocess-llama.md](../docs/phase14-inprocess-llama.md). **Operations:** [docs/OPERATIONS.md](./docs/OPERATIONS.md).

**Pre-flight VRAM (no load):** `../scripts/runtime_vram_estimate.sh <gguf> [--num-ctx N]` — same path as `/internal/vram-estimate`. **Why:** pick quant and context before starting `llama-server` on a 16 GB card.

**GPU smokes (host with model loaded):** `../scripts/gpu_smoke_all.sh` — coordination + inference paths; optional `RUN_E2E_TOOLS=1` for tools chat on `:8081` and Go proxy `:8080`. `../scripts/gpu_health_report.sh` uses `runtime.gpu_health_report` for `/health` tuning output (export hint only when factor is in 0.1–3).

## Single command (Phase 7)

```bash
export LLAMA_SERVER_BIN=.../llama-server
export LLAMA_MODEL=.../model.gguf
./scripts/serve_with_runtime.sh
# or: zerollama-runtime up
```

See [docs/OPERATIONS.md](./docs/OPERATIONS.md).

## Install (dev)

Requires [uv](https://docs.astral.sh/uv/). Create the venv **on the machine where you run tests** (do not copy `.venv` from another host — scripts embed absolute paths).

```bash
./scripts/runtime_uv_venv.sh
# or manually:
cd runtime
uv venv .venv --python 3.11
uv pip install -e ".[dev,serve]"
source .venv/bin/activate
```

**Note:** Go **embedded** runtime still uses system `libpython3` (see `docs/gpu-training.md`). The uv venv is for **sidecar** runtime (`serve_with_runtime.sh`, smokes, M3 sign-off).

## Tests (Phase 0 — no GPU)

```bash
./scripts/runtime_uv_venv.sh
cd runtime
source .venv/bin/activate
pytest
# or: .venv/bin/pytest
```

## Build llama-server

```bash
./scripts/build_llama_server.sh
export LLAMA_SERVER_BIN=/path/to/llama.cpp/build/bin/llama-server
```

### Phase 14 — in-process forward (optional)

**Why:** The default path shells out to `llama-server` (loopback HTTP + second process). Phase 14 loads the same `libllama` inside this process for lower latency, simpler VRAM handoff, and vocab tokenize for Go tools render.

| Backend | When to use |
|---------|-------------|
| `subprocess` (default) | Production unless you have smokes green on your card |
| `inprocess` | **5080 GPU sign-off** — needs `libllama.so` next to `LLAMA_SERVER_BIN` |
| `llama-cpp-python` | No local build; **CPU default** on many CUDA wheels |

```bash
# On the Go serve process (not only this shell):
export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess
export LLAMA_CPP_LIB=/path/to/llama.cpp/build/bin/libllama.so
# wheel: export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=llama-cpp-python
# wheel GPU (if stable): export ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS=99
# YAML: uncomment llama_backend: inprocess in runtime/configs/single_gpu.yaml (env wins)
```

`/health` reports `llama_backend_source`: `env` (override set), `config` (explicit YAML key), or `default` (packaged subprocess). When inprocess load fails on Darwin, the sidecar may fall back to subprocess `llama-server` (`llama_backend_fallback: true`, `llama_backend_requested: inprocess`); control via `ZEROLLAMA_RUNTIME_INPROCESS_FALLBACK` (`auto` on Mac when `LLAMA_SERVER_BIN` is set).

Sign-off: `../scripts/phase14_inprocess_smoke.sh` (5080 GPU), `../scripts/phase14_wheel_cpu_smoke.sh` (wheel CPU), `../scripts/phase14_yaml_config_smoke.sh`, or `../scripts/phase14_both_backends.sh` — see [../docs/phase14-inprocess-llama.md](../docs/phase14-inprocess-llama.md).

## Sidecar server (Phase 1–3)

**Single GPU:** `configs/single_gpu.yaml` (`tensor_parallel: 1`, layer split). **Why:** tensor split on one GPU makes `llama-server` fail model fitting.

Dual RTX 4090 example (`tensor_parallel: 2`, `-sm layer -ts 1,1 -fit off`):

```bash
export LLAMA_SERVER_BIN=/var/lib/vz/private/1564/root/llama.cpp/build/bin/llama-server
export LLAMA_MODEL=/path/to/model.gguf
zerollama-runtime serve --port 8081
# or: zerollama-runtime serve --config configs/dual_4090.yaml
```

Override via env: `ZEROLLAMA_RUNTIME_CONFIG`, `ZEROLLAMA_TENSOR_PARALLEL`, `ZEROLLAMA_KV_NUM_BLOCKS`.

Endpoints: `GET /health`, `GET /ready` (503 when not accepting loads), `POST /api/generate`, `POST /api/chat`, `POST /v1/completions`, `POST /internal/vram-estimate` (loopback), `POST /internal/training-handoff`, `POST /internal/inference/resume`.

Smoke / GPU handoff: [../docs/testing-smoke.md](../docs/testing-smoke.md). **Why two internal endpoints:** handoff unloads GPU for training or legacy runners; resume clears `inference_state` without full process restart.

Pin: [LLAMA_CPP_PIN.md](./LLAMA_CPP_PIN.md)

Speculative decoding: [docs/SPECULATIVE.md](./docs/SPECULATIVE.md) — e.g. `zerollama-runtime serve --config configs/dual_4090_ngram.yaml`

## Go daemon integration (Phase 1)

Terminal 1 — Python sidecar:

```bash
export ZEROLLAMA_RUNTIME_URL=http://127.0.0.1:8081   # set on Go side too
export LLAMA_SERVER_BIN=.../llama-server
export LLAMA_MODEL=.../model.gguf
zerollama-runtime serve --port 8081
```

Terminal 2 — zerollama with proxy:

```bash
export ZEROLLAMA_RUNTIME_URL=http://127.0.0.1:8081
# ZEROLLAMA_RUNTIME defaults to on when URL is set (set ZEROLLAMA_RUNTIME=0 to disable)
zerollama serve
```

Per-model opt-in: in `config.json` set `"modality_backends": { "inference": "zerollama-runtime" }`.
Streaming generate/chat are supported on the runtime; tools, vision, and logprobs still use the Go runner.

Training OOM path calls `POST /internal/training-handoff` on the runtime URL when configured.

## Admission & inference-first (Phase 11)

**Why:** on one GPU, chat, batch work, and training share VRAM. Phase 11 rejects or stalls work **before** the queue grows and **before** `llama-server` starts.

| Knob | Role |
|------|------|
| `ZEROLLAMA_RUNTIME_INFERENCE_POLICY=off` | Disable defer/ggml/backlog throttling for **`priority: low` only** |
| `ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM=0` | Disable host + GPU budget checks and 1 GiB headroom floor |
| `ZEROLLAMA_RUNTIME_MAX_QUEUE` | Optional waiting-queue cap (default **512** in code) |

**Priority** (`options.priority`): `high` (interactive) jumps the queue and bypasses min-free VRAM; `low` / `batch` is throttled under pressure; **`normal` default chat is not blocked** by Go mirror gates.

**Constants** (2 GiB training reserve, 1 GiB min free, backlog thresholds): `runtime/runtime/gpu/admission.py`, `inference_policy.py` — tune after measurement, not new env vars.

Details: [../docs/phase11-runtime-admission.md](../docs/phase11-runtime-admission.md). Smoke: `../scripts/e2e_coordination_smoke.sh`.

## VRAM pre-check (Phase 13)

Before `llama-server` starts, the runtime estimates weights + KV headroom vs free GPU memory. **Why:** failing after subprocess start wastes time and returns opaque CUDA OOMs. Phase 11 runs the same budget at **enqueue** when the GGUF path is known.

| Concern | Env / behavior | Why |
|---------|----------------|-----|
| Enable check | `ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM` (auto-on when probe available) | Opt-out on hosts without NVIDIA tools |
| Probe | `ZEROLLAMA_RUNTIME_VRAM_PROBE=auto\|nvml\|nvidia-smi` | NVML avoids `nvidia-smi` subprocess; optional `pip install -e '.[gpu]'` |
| Unified GPU | `ZEROLLAMA_RUNTIME_VRAM_UNIFIED_FALLBACK` (default on) | iGPU: NVML may report NOT_SUPPORTED; use Linux `MemAvailable` |
| Context scale | request `num_ctx` → `ZEROLLAMA_RUNTIME_VRAM_NUM_CTX` → GGUF `context_length` | Match operator intent to KV budget |
| Layer scale | GGUF `block_count` vs `ZEROLLAMA_RUNTIME_VRAM_LAYER_BASE` (32) | Deeper models need more KV; heuristic not exact |

Full table: [docs/OPERATIONS.md](./docs/OPERATIONS.md). Tests: `pytest tests/test_gpu_vram.py` (no GPU).
