# Wan CPU worker (optional accelerator) — plan

Directional design for offloading **CPU- and RAM-heavy** Wan stages to a separate host. Not shipped today. Complements single-node mitigations in [wan-t2v.md](./wan-t2v.md) (`WAN_UNLOAD_T5`, `WAN_VAE_CPU`).

---

## Problem

On **16g** presets (`wan2.1-t2v:1.3b`, etc.), the GPU box must run:

| Stage | Device | Typical host RAM |
|-------|--------|------------------|
| T5-XXL text encode | CPU | **~11 GB** (weights resident until encode completes) |
| DiT diffusion (1.3B) | GPU | Low host RAM if weights stay on GPU |
| VAE decode | GPU or CPU | **+0–2 GB** host if forced to CPU |

Even with **`WAN_UNLOAD_T5=1`** (release T5 after encode), a **16 GB CT** still sees a **startup spike** (T5 + DiT load) and end-of-job DiT-on-CPU footprint. Operators with a **large CPU/RAM box** (or spare Proxmox host) should be able to keep the **GPU CT thin** and push T5/VAE work elsewhere.

**Goal:** optional **`wan-cpu-worker`** sidecar — same deployment pattern as external runtime (`ZEROLLAMA_RUNTIME_URL`) and external training TCP (`:9500`).

---

## Non-goals (v1)

- Remote **DiT diffusion** (stays on GPU CT; network tensor size and latency are unacceptable).
- Replacing the embedded training worker or `/v1/videos` job queue.
- Multi-tenant SaaS auth, billing, or cross-tenant GPU sharing.
- Automatic checkpoint sync from Hugging Face inside the worker (operators install weights once, like today).

---

## Architecture

### Today (single node)

```text
GPU CT                         
zerollama serve :8080         
  run_script → wan_video_generate.py → wan_generate_entry.py → generate.py
    T5 (CPU) → DiT (GPU) → VAE (GPU/CPU) → mp4
```

### Target (optional CPU worker)

```text
GPU CT (16 GB RAM, 5080)              CPU RAM box (64–256 GB, many cores)
────────────────────────              ────────────────────────────────────
zerollama serve :8080
  run_script → wan_video_generate.py
    │ encode RPC ─────────────────────► wan-cpu-worker :9510
    │◄── context, context_null           (T5 resident, no reload per job if warm)
    │ DiT sample (local GPU)
    │ decode RPC ─────────────────────► wan-cpu-worker :9510  [optional phase B]
    │◄── frames or mp4 chunk
    └── write $OLLAMA_MODELS/generated/<job_id>.mp4
```

**Why split here:** RPC payloads are **megabytes** (embeddings, latents). T5 **checkpoints** (~11 GB) stay on the CPU host only.

---

## What to offload (phased)

| Phase | Offload | GPU CT RAM win | Complexity |
|-------|---------|----------------|------------|
| **A** | T5 encode only | **~11 GB** after startup path removed | Low — one RPC, patch before diffusion loop |
| **B** | VAE decode (optional) | **~2 GB** + faster GPU CT if decode was CPU | Medium — large latent tensors, streaming |
| **C** | Health, queue, metrics | Ops | Low–medium |
| **D** | Shared NFS ckpt path | Avoid duplicate weight copies | Ops / storage |

**Recommended first ship:** Phase **A** only.

---

## Service: `wan-cpu-worker`

### Process layout

| Item | Choice |
|------|--------|
| Language | Python (reuse Wan venv + `install_wan_video.sh` deps) |
| HTTP | FastAPI + uvicorn on **`9510`** (default), or Unix socket on same host |
| Weights | Same paths as GPU box: `WAN_CKPT_DIR`, `WAN_REPO`, or shared mount |
| Concurrency | Thread pool or job queue; **one T5 model resident**; cap parallel encodes by RAM |

### API (v0 sketch)

All requests include **`profile`** (`wan2.1-t2v-1.3b`, …) and **`ckpt_hash`** or **`ckpt_dir`** so worker refuses mismatched weights.

#### `GET /health`

```json
{ "ok": true, "profiles_loaded": ["wan2.1-t2v-1.3b"], "t5_ready": true }
```

#### `POST /v1/wan/encode`

**Request**

```json
{
  "profile": "wan2.1-t2v-1.3b",
  "ckpt_dir": "/data/wan/Wan2.1-T2V-1.3B",
  "prompt": "A cat on a stage",
  "neg_prompt": ""
}
```

`neg_prompt` empty → worker uses profile default (`sample_neg_prompt` from Wan config).

**Response**

```json
{
  "context": { "storage": "inline_b64", "dtype": "bfloat16", "shape": [512, 4096], "data": "..." },
  "context_null": { "storage": "inline_b64", "dtype": "bfloat16", "shape": [512, 4096], "data": "..." }
}
```

**v0.1 alternative:** write `.pt` or `.safetensors` to a shared **`WAN_CPU_SCRATCH`** directory keyed by `job_id`; response returns paths only (better for large tensors / Phase B).

#### `POST /v1/wan/decode` (Phase B)

**Request:** latents + `profile` + `frame_num` + `size`.  
**Response:** raw RGB tensor metadata or **mp4 bytes** (prefer file path on shared storage for big payloads).

### Auth

| Mode | Use |
|------|-----|
| None | Lab LAN only |
| `WAN_CPU_WORKER_TOKEN` | Shared bearer on GPU CT + worker |
| mTLS | Production between Proxmox hosts |

Worker must **not** be exposed on the public internet without auth.

---

## GPU-side integration

### Environment variables (proposed)

| Variable | Role |
|----------|------|
| `WAN_CPU_WORKER_URL` | Base URL, e.g. `http://192.168.1.50:9510`. Unset → local T5/VAE (today). |
| `WAN_CPU_WORKER_TOKEN` | Optional bearer token |
| `WAN_CPU_ENCODE_ONLY` | `1` (default when URL set): skip local T5 load entirely |
| `WAN_CPU_DECODE` | `0`/`1` — use worker for VAE decode (Phase B) |
| `WAN_CPU_SCRATCH` | Shared temp dir for path-based tensor exchange |
| `WAN_CPU_TIMEOUT_SEC` | RPC timeout per stage (default 600 encode, 1800 decode) |

Go **`buildWanVideoPayload`** (`server/video_generate.go`) would pass these through `run_script` env when set globally or per-manifest `backend_paths.wan_cpu_worker_url`.

### Code touch points

| Layer | Change |
|-------|--------|
| **`scripts/wan_cpu_client.py`** (new) | HTTP client, retry, tensor (de)serialize |
| **`scripts/wan_memory_hooks.py`** | If `WAN_CPU_WORKER_URL` set, skip local T5 load / unload hooks |
| **`scripts/wan_generate_entry.py`** | Patch `WanT2V.generate` to inject remote `context` / `context_null` before diffusion |
| **`scripts/wan_video_generate.py`** | Log `WAN_CPU_WORKER_URL`; progress lines `PROGRESS:10:encoded via cpu worker` |
| **`server/video_generate.go`** | Optional manifest/backend_paths + env passthrough |
| **`modelfiles/wan2.1-t2v/config.json`** | Optional `"wan_cpu_worker_url"` in `backend_paths` |
| **`docs/wan-t2v.md`** | Link here; troubleshooting row for worker down → fallback |

### Fallback policy

| Condition | Behavior |
|-----------|----------|
| URL unset | Current local path (no behavior change) |
| Worker health fail at job start | Fail fast with clear error **or** fallback to local T5 if `WAN_CPU_FALLBACK_LOCAL=1` |
| Encode OK, decode fail (Phase B) | Fall back to local GPU VAE if VRAM allows |

Default: **fail fast** (avoid silent 16 GB OOM on GPU CT).

---

## Wan pipeline patch (Phase A detail)

Upstream `WanT2V.generate` (simplified):

```text
load T5 → encode prompt → encode neg → [diffusion loop] → VAE decode → return video
```

**Remote encode injection:**

1. GPU subprocess starts **without** constructing `T5EncoderModel` when `WAN_CPU_ENCODE_ONLY=1`.
2. Wrapper calls `wan_cpu_client.encode(...)` → tensors on GPU (`context`, `context_null`).
3. Call into Wan with prebuilt contexts (monkeypatch or forked `generate` helper) → diffusion unchanged.
4. Rest of job identical.

**Progress mapping for `/v1/videos`:**

| PROGRESS | Stage |
|----------|--------|
| 0–5 | Job start |
| 10 | Remote encode done |
| 5–95 | Diffusion (tqdm) |
| 96–100 | Save mp4 |

---

## Deployment models

### Model 1 — Two Proxmox CTs (recommended)

| CT | RAM | Role |
|----|-----|------|
| `cudallama` | 16 GB | GPU, zerollama serve, thin Wan venv (DiT + VAE only) |
| `wan-cpu` | 64–128 GB | `wan-cpu-worker`, T5 (+ VAE) weights, no GPU |

Private network between CTs; firewall **9510** to GPU CT only.

### Model 2 — Same host, two processes

CPU worker on host with bind-mounted RAM disk; GPU CT uses `http://host.lan:9510`. Less isolation, simpler lab setup.

### Model 3 — Shared NFS

`ckpt_dir` and `WAN_CPU_SCRATCH` on NFS; both nodes mount read-only checkpoints, read-write scratch. Avoids duplicating 17 GB Wan 1.3B tree.

---

## Observability

| Signal | Where |
|--------|--------|
| Encode latency / queue depth | Worker `GET /metrics` or structured logs |
| RPC errors | GPU wrapper `SCRIPT:` lines + job `error` in GET `/v1/videos/:id` |
| Weight version | Log `ckpt_dir` + sha256 of `models_t5_*.pth` at worker start |

---

## Testing plan

| Test | Command / gate |
|------|----------------|
| Worker unit | `pytest runtime/tests/` or `scripts/tests/test_wan_cpu_client.py` — mock HTTP, tensor roundtrip |
| Worker smoke | `curl :9510/health`; encode fixed prompt, compare hash to local encode |
| E2E GPU+CPU | `WAN_CPU_WORKER_URL=... RUN_E2E_WAN_CPU=1 scripts/e2e_wan_cpu_worker.sh` |
| CI | CPU-only encode test on GitHub runner **without** GPU; skip E2E in default regression |
| Fallback | Kill worker mid-job → expect failed job with actionable message |

---

## Implementation roadmap

### Milestone 1 — Design freeze (this doc)

- [x] Problem, phases, API sketch, env vars
- [ ] Review: tensor transport (inline b64 vs scratch files)
- [ ] Review: default port `9510` vs reuse training `:9500` multiplex (prefer **separate** service)

### Milestone 2 — Phase A (T5 remote encode)

- [ ] `scripts/wan_cpu_worker.py` — FastAPI app, load T5 once, `/health`, `/v1/wan/encode`
- [ ] `scripts/wan_cpu_client.py` — client used by `wan_generate_entry.py`
- [ ] Patch diffusion entry to skip local T5 when URL set
- [ ] `buildWanVideoPayload` env passthrough
- [ ] Doc + operator install snippet
- [ ] Smoke script

**Exit criteria:** Full video on GPU CT with **≤8 GB host RSS during diffusion**; T5 never loaded on GPU CT.

### Milestone 3 — Phase B (optional remote VAE)

- [ ] `/v1/wan/decode` + scratch file protocol
- [ ] `WAN_CPU_DECODE=1` path in wrapper
- [ ] E2E latency benchmarks (network vs local CPU VAE)

### Milestone 4 — Production hardening

- [ ] Token auth, request size limits, job scratch TTL
- [ ] Prometheus metrics, graceful shutdown
- [ ] ROADMAP.md cross-link; optional `systemd` unit example

---

## Open questions

1. **Tensor format:** inline base64 (simple) vs safetensors on NFS (fast for Phase B latents)?
2. **Worker venv:** same `~/.zerollama/third_party/wan/venv` as GPU box vs dedicated slim venv (T5 + tokenizers only)?
3. **Manifest:** global env `WAN_CPU_WORKER_URL` vs per-model `backend_paths.wan_cpu_worker_url`?
4. **Queueing:** one encode at a time vs pool sized by RAM (T5 is ~11 GB; 128 GB box → ~10 parallel?)?
5. **Version skew:** reject encode if worker `torch`/Wan patch level differs from GPU subprocess?

---

## Related docs

| Doc | Link |
|-----|------|
| Wan T2V operator guide | [wan-t2v.md](./wan-t2v.md) |
| Training / run_script queue | [gpu-training.md](./gpu-training.md) |
| External runtime sidecar pattern | [runtime-embed.md](./runtime-embed.md) |
| Product queues / VRAM | [ROADMAP.md](./ROADMAP.md) |
| 5080 operator constraints | [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md) |
