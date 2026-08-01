---
name: gpu-capability-discovery
description: "Determine which GPU backend, autoconfig profile, and inference path a zerollama server picked (Metal/CUDA/MLX/inprocess llama-server) before choosing models or debugging performance."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, gpu, autoconfig, metal, cuda, mlx, backend, health]
    category: mlops
    related_skills: [zerollama-integration, diagnose-server-health, configure-zerollama-env, fleet-vram-admission, model-suggester]
---

# GPU Capability Discovery Skill

Determine what GPU backend a running [zerollama](https://github.com/GoodSoftware-Group/zerollama)
server actually picked — Metal/ggml, CUDA, MLX, or in-process llama-server —
before choosing a model size/quant or debugging why inference is slower
than expected. Zerollama probes hardware once at startup and exposes the
result via the runtime `/health` endpoint and `zerollama doctor`; there is
no separate agent-facing "list my GPUs" HTTP API (native probing is an
internal, hidden CLI subcommand used by the server itself).

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/api/can-load -d '{}'   # 400/422 = route exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

## When to Use

- Before picking a model quant/size, to know whether the box is Apple
  Silicon (MLX/Metal, unified memory) or CUDA (discrete VRAM)
- Debugging unexpectedly slow inference — confirm the backend isn't a
  CPU/inprocess fallback
- Understanding why an autoconfig profile chose a particular default
  (`single_gpu.yaml`, `apple_silicon.yaml`, etc.)

## How to Run

```bash
# Runtime sidecar health (autoconfig pick, backend, VRAM probe)
curl -s http://127.0.0.1:8081/health | python3 -m json.tool

# Same info surfaced with pass/fail interpretation
zerollama doctor
```

Key `/health` fields:

| Field | Meaning |
|---|---|
| `autoconfig.pick` | Which profile was selected (`apple_silicon`, `single_gpu`, `custom`, ...) |
| `llama_backend` | Actual backend in use (`inprocess`, subprocess llama-server, etc.) |
| `llama_backend_source` | Where that choice came from (`default`, `env`, `yaml`) |
| `llama_backend_requested` | What was explicitly asked for, if anything |
| `llama_backend_fallback` | `true` means the preferred backend failed to load and it fell back — a real problem, not just informational |
| `vram_probe_effective` | Which VRAM-estimation path is active |

## Interpreting the result

| Symptom | Likely meaning |
|---|---|
| `pick` is not `apple_silicon`/`custom` on a Mac | Env/YAML override is steering away from the expected profile — check `ZEROLLAMA_RUNTIME_CONFIG` |
| `source: env`, `backend != inprocess` | `ZEROLLAMA_RUNTIME_LLAMA_BACKEND` is forcing a non-default backend | 
| `backend: inprocess`, `source: default` | Running in-process without an explicit yaml/env choice — usually fine, but means no one deliberately picked it |
| `llama_backend_fallback: true` | The requested backend failed to load (often a missing/broken `libllama`); check for a text-only GGUF fallback or fix `libllama` per `diagnose-server-health` |

## Choosing model size/quant from hardware class

| Platform | Typical guidance |
|---|---|
| Apple Silicon (unified memory, MLX/Metal) | Larger models fit thanks to shared RAM/VRAM; MLX path for safetensors, ggml Metal for GGUF |
| Single discrete GPU (16GB class) | Favor Q4/Q5 GGUF quants; `runtime/configs/single_gpu.yaml` autoconfig applies |
| Multi-GPU / tensor-parallel | Check `device_count`, `tensor_parallel`, `split_mode`, `tensor_split`, `main_gpu` surfaced by `POST /api/can-load` (see `fleet-vram-admission`) before assuming a big model fits |

## Pitfalls

- **There's no public "list GPUs" HTTP endpoint** — `gpu-discover` is a
  hidden, internal CLI subcommand the server invokes on itself for
  discovery; agents should read `/health` and `zerollama doctor` output
  instead of trying to shell out to it directly.
- **`/health` is the runtime sidecar (`:8081`), not the Go API (`:11434`)**
  — hitting the Go port for this data returns nothing useful; use the
  sidecar port (or `ZEROLLAMA_RUNTIME_URL` if it's been overridden).
- **A "warn" doesn't always mean broken** — e.g. `inprocess` with
  `source: default` is a normal, working configuration on many hosts; only
  `llama_backend_fallback: true` or missing `libllama` are hard failures.
- **Hardware tiers documented per platform, not universal** — CUDA arch
  flags (5080 = `120-real`, 4090 = `89`) and Apple tiers are host-specific;
  check `docs/apple-silicon-metal.md` / `docs/cuda-lanes.md` rather than
  assuming one guidance fits all GPUs.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `diagnose-server-health` — full `zerollama doctor` report, including this same backend/health data plus fix hints
- `configure-zerollama-env` — changing which backend/profile gets picked
- `fleet-vram-admission` — capacity dry-runs that surface host topology (`device_count`, `tensor_parallel`, etc.)
- `model-suggester` — picking a model that actually fits the discovered GPU
