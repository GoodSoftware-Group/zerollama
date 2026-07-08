# CUDA hardware lanes — common vs topology-specific

**Why this doc:** Zerollama CUDA work was documented primarily around the **RTX 5080 16 GiB** sign-off (CT 1564). Other hosts run **different topologies** (dual RTX 4090, single 3090, …). Operators need one place to see what applies to **any CUDA box** vs what is **lane-specific** setup and testing.

**Related:** [5080-runbook.md](./5080-runbook.md), [testing-smoke.md](./testing-smoke.md), [gpu-profiles-l1.md](./gpu-profiles-l1.md), [runtime/README.md](../runtime/README.md).

---

## Lane model

| Layer | Question it answers |
|-------|---------------------|
| **CUDA-common** | Does the Go + Python runtime + llama-server stack work on *this* CUDA driver? |
| **Hardware lane** | Is tensor/layer split, batch size, and sign-off GGUF appropriate for *this* GPU count and VRAM? |

---

## Hardware lanes (today)

| Lane ID | Detect | Runtime YAML | L1 profile | Typical serve |
|---------|--------|--------------|------------|---------------|
| **`rtx_5080`** | 1 GPU, ≤18 GiB VRAM | `single_gpu.yaml` | `rtx-5080.json` | Embedded runtime @ `:8081`, Go @ `:8080` |
| **`single_4090`** | 1 GPU, ~24 GiB | `single_gpu.yaml` | `rtx-4090.json` | Same as 5080 lane shape |
| **`dual_4090`** | 2+ GPUs | `dual_4090.yaml` | `rtx-4090.json` (per card) | External sidecar + Go (`ZEROLLAMA_RUNTIME_URL`) |
| **`single_cuda`** | 1 GPU, other VRAM | `single_gpu.yaml` | bucket from `index.json` | Same as 5080 unless overridden |

Autodetect: `source scripts/cuda_common_env.sh && cuda_lane_detect`  
Override: `CUDA_LANE=dual_4090`

---

## CUDA-common (any lane)

These do **not** assume 16 GiB, sm_120, or tensor parallel.

### Env / scripts

| Artifact | Role |
|----------|------|
| [`scripts/cuda_common_env.sh`](../scripts/cuda_common_env.sh) | Shared libs, CGO flags, llama-server discovery, lane detect |
| [`scripts/cuda_common_gate.sh`](../scripts/cuda_common_gate.sh) | Tier-0 + optional GPU smokes — **run on every CUDA host** |

### Tests (no topology tuning)

| Gate | Script | GPU load |
|------|--------|----------|
| Script syntax + imports | `check_gpu_scripts.sh` | No |
| KV native CI | `phase15_kv_native_ci.sh` | No |
| L2 pin status | `phase17_l2_pin_status.sh` | No |
| Coordination + runtime generate/chat | `gpu_smoke_all.sh` | Yes (small GGUF) |
| Phase 12 golden | `phase12_golden_ci.sh` | No |
| L2 fork A/B | `l2_cuda_full_gate.sh` | Yes |
| Phase 14/15 sign-off | `phase14_*`, `phase15_inprocess_signoff.sh` | Yes |

### Runtime / policy (topology-agnostic logic)

- Phase 11 admission, Phase 13 VRAM — [phase11-runtime-admission.md](./phase11-runtime-admission.md), [phase13-runtime-vram.md](./phase13-runtime-vram.md)
- L3 prefix cache policy — [gpu-profiles-l3.md](./gpu-profiles-l3.md)
- Decode graph invalidation — [decode-graph-invalidation.md](./decode-graph-invalidation.md)

---

## Lane-specific: RTX 5080 (16 GiB single-GPU)

**Doc:** [5080-runbook.md](./5080-runbook.md)  
**Env:** `source scripts/5080_env.sh`  
**Session:** `./scripts/gpu_5080_session.sh` or `./scripts/5080_resignoff.sh --build`

| Concern | 5080-specific |
|---------|---------------|
| YAML | `single_gpu.yaml` — **layer** split, `tensor_parallel: 1` |
| L1 profile | `rtx-5080.json` — `n_parallel=2`, smaller batch |
| CUDA build | **sm_120** — CUDA 12.8+ nvcc |
| VRAM fixtures | 1B Q8 smoke, 9B for L1/L3 |
| Sign-off | `5080_resignoff.sh` tiers 0–4 |

**Not for dual 4090:** Do not rely on `single_gpu.yaml` for 2×4090 large models — use `dual_4090.yaml`.

---

## Lane-specific: dual RTX 4090

**Env:** `source scripts/dual_4090_env.sh`  
**Session:** `./scripts/gpu_lane_session.sh`

| Concern | dual-4090-specific |
|---------|---------------------|
| YAML | `dual_4090.yaml` — **layer** split `-sm layer -ts 1,1 -fit off` |
| L1 profile | `rtx-4090.json` per GPU (`n_parallel=8`) |
| Serve | Edge build → external `zerollama-runtime.service` @ `:8081` |
| Scheduler | `OLLAMA_SCHED_SPREAD=1` on Go daemon |
| Install | `/opt/zerollama/runtime` |

### Production (this host)

```bash
systemctl status zerollama-runtime ollama
curl -s http://127.0.0.1:8081/health | jq '{status, autoconfig, gpu_profile}'
curl -s http://127.0.0.1:2083/api/status | jq .inference
```

### dual-4090 gates (beyond CUDA-common)

| Gate | Notes |
|------|-------|
| `dual_4090_health_check` | `autoconfig.pick == dual_4090`, 2+ GPUs |
| `l1_cuda_full_gate.sh` | 9B+ GGUF, validate `rtx-4090` profile |
| `l3_cuda_full_gate.sh` | 27k ctx production leg |

---

## Decision tree

```
1. source scripts/cuda_common_env.sh && cuda_print_lane_summary
2. ./scripts/cuda_common_gate.sh
     RUN_E2E_GPU=1 RUN_E2E_PROXY=1 LLAMA_MODEL=... LLAMA_SERVER_BIN=...

If rtx_5080:
  source scripts/5080_env.sh
  ./scripts/5080_resignoff.sh  OR  gpu_5080_session.sh

If dual_4090:
  source scripts/dual_4090_env.sh
  dual_4090_health_check
  optional: l1_cuda_full_gate.sh / l3_cuda_full_gate.sh
```

One-shot: `./scripts/gpu_lane_session.sh`

---

## Config reference

| File | `device_count` | `split_mode` | When |
|------|----------------|--------------|------|
| `single_gpu.yaml` | 1 | layer | 5080, single 4090 |
| `dual_4090.yaml` | 2 | layer | 2×4090 |
| `dual_4090_ngram.yaml` | 2 | layer | + ngram speculative |

---

## What is CUDA-common vs was labeled 5080-only

| Topic | Was documented as | Actually |
|-------|-------------------|----------|
| `gpu_smoke_all.sh` | 5080 session | **CUDA-common** |
| `l2_cuda_full_gate.sh` | 5080 sign-off | **CUDA-common** fork eval |
| `phase14/15 signoff` | 5080 | **CUDA-common** (needs libllama on host) |
| `l1_cuda_full_gate.sh` | 5080 | Shared script; **profile JSON + YAML** are lane-specific |
| `5080_env.sh` | 5080 only | 5080 paths; sources **`cuda_common_env.sh`** for shared vars |
