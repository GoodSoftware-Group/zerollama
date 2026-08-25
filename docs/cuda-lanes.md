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

Autodetect: `source scripts/gpu/cuda_common_env.sh && cuda_lane_detect`  
Override: `CUDA_LANE=dual_4090`

---

## CUDA-common (any lane)

These do **not** assume 16 GiB, sm_120, or tensor parallel.

### Env / scripts

| Artifact | Role |
|----------|------|
| [`scripts/gpu/cuda_common_env.sh`](../scripts/gpu/cuda_common_env.sh) | Shared libs, CGO flags, llama-server discovery, lane detect |
| [`scripts/gpu/cuda_common_gate.sh`](../scripts/gpu/cuda_common_gate.sh) | Tier-0 + optional GPU smokes — **run on every CUDA host** |

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
**Env:** `source scripts/gpu/5080_env.sh`  
**Session:** `./scripts/gpu/gpu_5080_session.sh` or `./scripts/gpu/5080_resignoff.sh --build`

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

**Env:** `source scripts/gpu/dual_4090_env.sh`  
**Session:** `./scripts/gpu/gpu_lane_session.sh`

| Concern | dual-4090-specific |
|---------|---------------------|
| YAML | `dual_4090.yaml` — **layer** split `-sm layer -ts 1,1 -fit off` |
| L1 profile | `rtx-4090.json` per GPU (`n_parallel=8`) |
| Serve | Edge build → external `zerollama-runtime.service` @ `:8081` |
| Scheduler | `OLLAMA_SCHED_SPREAD=1` on Go daemon |
| Install | `/opt/zerollama/runtime` |

### Production (this host)

```bash
systemctl status zerollama zerollama-runtime ollama
systemctl start|stop|restart zerollama          # Go API :2083 (this fork)
journalctl -u zerollama -f
systemctl start|stop|restart ollama             # upstream Ollama :11434 (fallback for other users)
journalctl -u ollama -f
systemctl start|stop|restart zerollama-runtime  # Python sidecar :8081
systemctl start|stop zerollama-stack.target     # zerollama API + runtime (not upstream ollama)
curl -s http://127.0.0.1:8081/health | jq '{status, autoconfig, gpu_profile}'
curl -s http://127.0.0.1:2083/api/version       # zerollama
curl -s http://127.0.0.1:11434/api/version      # ollama
```

CLI: `zerollama` talks to `:2083` if `OLLAMA_HOST` is unset in that binary's defaults; `ollama` defaults to `:11434`. To hit zerollama with the ollama CLI: `OLLAMA_HOST=127.0.0.1:2083 ollama list`.

### dual-4090 gates (beyond CUDA-common)

| Gate | Notes |
|------|-------|
| `dual_4090_health_check` | `autoconfig.pick == dual_4090`, 2+ GPUs |
| `l1_cuda_full_gate.sh` | 9B+ GGUF, validate `rtx-4090` profile |
| `l3_cuda_full_gate.sh` | 27k ctx production leg |

---

## Decision tree

```
1. source scripts/gpu/cuda_common_env.sh && cuda_print_lane_summary
2. ./scripts/gpu/cuda_common_gate.sh
     RUN_E2E_GPU=1 RUN_E2E_PROXY=1 LLAMA_MODEL=... LLAMA_SERVER_BIN=...

If rtx_5080:
  source scripts/gpu/5080_env.sh
  ./scripts/gpu/5080_resignoff.sh  OR  gpu_5080_session.sh

If dual_4090:
  source scripts/gpu/dual_4090_env.sh
  dual_4090_health_check
  optional: l1_cuda_full_gate.sh / l3_cuda_full_gate.sh
```

One-shot: `./scripts/gpu/gpu_lane_session.sh`

---

## Quantization roadmap (CUDA / llama-server)

Weight formats the **Go → Python runtime → patched llama-server** path can load from GGUF. Distinct from the **MLX `x/create`** path (mxfp8, etc.).

### Today

| Format | Load / infer (GGUF) | Dual 4090 (sm_89) | Notes |
|--------|---------------------|-------------------|-------|
| **Q4 / Q5 / Q8 / K-quants** | Yes | Yes | Primary production quants |
| **Fork KV** (`qjl1_256`, `q4_polar`, `tbq3_0`, `tbq4_0`) | Yes | Yes | Patch 0020 + fork profiles |
| **MXFP4** | Yes (ggml) | Yes (generic MMQ) | e.g. gpt-oss family |
| **NVFP4** | Yes (ggml) | Yes (generic MMQ) | **Not** Blackwell-native path on 4090 — uses Q8_1 activation quant in `mmq.cu`; slower than sm_120 but functional |
| **HF FP8 safetensors** | Via convert | N/A until GGUF | Default convert **dequants** → F16/BF16; **`--fp8-native`** writes types 51/52 — see [native-fp8-gguf.md](./native-fp8-gguf.md) |
| **Native FP8 GGUF weights** | Yes | Yes (MMVQ float + MMQ amax→int8) | `GGML_TYPE_FP8_E4M3=51` / `GGML_TYPE_FP8_E5M2=52` (F16 scale + 32×FP8); CUDA convert/get_rows + MMVQ/MMQ; convert `--fp8-native` (ModelOpt/per-row + compressed-tensors `strategy:tensor` + **128×128** `weight_scale_inv`). Probe: `./scripts/fp8_cuda_probe.sh`; load: `./scripts/fp8_cuda_load_smoke.sh` |

**NVFP4 on this stack:** eliza pin + vendor tree include `GGML_TYPE_NVFP4`, CUDA `mmq` / `mmvq` / `convert` / `quantize` kernels, and llama.cpp NVFP4 scale-tensor loading. Expect **correctness on 4090**, not the Blackwell `blackwell_mma_available()` fast path (5080 / sm_120). Binary probe (no GGUF): `./scripts/gpu/nvfp4_cuda_probe.sh`.

**Host fixture (dual-4090, Jul 2026):**
- NVFP4: [FreedomAISVR/gpt-oss-20B-NVFP4-GGUF](https://huggingface.co/FreedomAISVR/gpt-oss-20B-NVFP4-GGUF) → `/mnt/ssd2/models/nvfp4/gpt-oss-20b/gpt-oss-20b-nvfp4.gguf` (~12 GiB; NVFP4 experts + Q8_0 non-experts)
- MXFP4 (A/B): [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) → `/mnt/ssd2/models/nvfp4/gpt-oss-20b-mxfp4/gpt-oss-20b-mxfp4.gguf`

**Sign-off:** `./scripts/gpu/nvfp4_cuda_signoff.sh` (probe + direct `/completion` @ 8k ctx). Artifact: `/tmp/nvfp4-cuda-signoff.json`. Runtime doctor: `/health.llama_patches.cuda_weight_formats` reports `{nvfp4, mxfp4, fp8_e4m3, fp8_e5m2, libggml_cuda}` from the resolved CUDA backend.

**Native FP8 (no GGUF required):** `./scripts/fp8_cuda_probe.sh` — full operator doc [native-fp8-gguf.md](./native-fp8-gguf.md). Load smoke: `./scripts/fp8_cuda_load_smoke.sh`.

| Format | Decode (8k ctx, GPU1) | Peak VRAM | Notes |
|--------|------------------------|-----------|-------|
| **NVFP4** | **211.9 tok/s** | 12128 MiB | Generic MMQ on sm_89 |
| **MXFP4** | **220.2 tok/s** | 11558 MiB | Same MoE 20B family |

On dual-4090, NVFP4 is within ~4% decode of MXFP4 and uses ~5% more VRAM — **correctness PASS**; do not expect Blackwell-native wins here. Q4_K A/B skipped (no matching gpt-oss Q4_K blob; MXFP4 is the fair sibling format).

### Roadmap

| Priority | Item | Why |
|----------|------|-----|
| **P1** | **Native FP8 weights** — `GGML_TYPE_FP8_E4M3` + CUDA MMVQ/MMQ + `--fp8-native` | **Built (Jul 2026)** — E4M3 type 51 + E5M2 type 52; CPU/CUDA dequant + MMVQ/MMQ + ModelOpt/`weight_scale_inv`/compressed-tensors tensor native pack. Probe: `./scripts/fp8_cuda_probe.sh`; load: `./scripts/fp8_cuda_load_smoke.sh` |
| **P1** | **NVFP4 dual-4090 sign-off** — smoke + L1 gate fixture, document expected perf vs Q4_K / vs 5080 Blackwell path | **Done (Jul 2026)** — `nvfp4_cuda_probe.sh` + `nvfp4_cuda_signoff.sh`; NVFP4≈MXFP4 tok/s on sm_89 (`/tmp/nvfp4-cuda-signoff.json`). Blackwell path remains P2 (5080) |
| **P2** | **NVFP4 / MXFP4 on Blackwell** — validate `blackwell_mma_available()` path in container builds (sm_120), bench vs generic MMQ | 5080 lane; NVIDIA gpt-oss collaboration path |
| **P2** | **Ollama compat + runtime probes** for NVFP4/MXFP4/FP8 in `llama_patch_health` / `/ready` | **Done (Jul 2026)** — `cuda_weight_formats` on `/health.llama_patches` + `/ready` warnings when `libggml-cuda` lacks NVFP4/MXFP4 markers |
| **P3** | **FP8 KV cache** (distinct from weight FP8) | Optional; fork KV uses QJL/Polar/TBQ today |

**Out of scope for CUDA roadmap:** MLX **mxfp8** (`x/create`, imagegen) — Apple/MLX only unless a separate MLX-CUDA bridge lands.

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
