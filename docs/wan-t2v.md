# Wan text-to-video (zerollama)

Local **text-to-video (T2V)** uses an OpenAI-compatible **async** Videos API. Generation runs as a **`run_script` job** on the embedded training worker—not as a GGUF chat runner and not as weights inside Ollama blobs.

## Why this design

| Choice | Why |
|--------|-----|
| **`POST /v1/videos` (OpenAI shape)** | Agents and SDKs already poll `id` → `status` → download URL. Sync HTTP would block for 30–60+ minutes and tie up load balancers. |
| **Training `run_script` queue** | Wan needs a long-lived GPU subprocess, `PROGRESS:` lines, and VRAM handoff with chat. Reusing the existing embedded Python worker avoids a second public port and duplicate job state. |
| **Thin Go layer + `wan_video_generate.py` wrapper** | Upstream Wan ships `generate.py` with its own venv and CLI. Go owns registry, auth, defer queue, and artifact paths; Python runs the subprocess we control. |
| **Config-only manifests (`video_gen`)** | Checkpoints live under `~/.zerollama/third_party/wan/` (multi‑GB). Blobs are for GGUF chat models; T2V presets are JSON + `backend_paths` only. |
| **`$OLLAMA_MODELS/generated/<job_id>.mp4`** | Predictable, per-job artifacts; `safeVideoArtifactPath` blocks path traversal. Content handler only serves files under that tree. |
| **VRAM broker before submit** | Video and training share one GPU with ggml/runtime inference. Unloading runners before Wan avoids opaque CUDA OOM mid-generation. |
| **`defer-*` job ids when busy** | Same policy as training T6: inference-first hosts queue video behind chat instead of failing with 409 spam. Clients keep polling the **same** id after promotion. |

## Architecture

```text
Client  POST /v1/videos
   │
   ▼
Go :8080  VideoCreateHandler
   │  validate model (video_gen + modality video_generation=wan)
   │  buildWanVideoPayload → run_script JSON (env, python_bin, {job_id} paths)
   │  submitTrainingJob (queue_on_busy) → Python queue OR defer-* queue
   ▼
training.py  run_local_script
   │  venv python  scripts/video/wan_video_generate.py
   ▼
wan_video_generate.py  →  Wan repo generate.py  (--save_file → artifact mp4)
   │
   ▼
GET /v1/videos/:id        (status, progress, model, created_at)
GET /v1/videos/:id/content (completed only; video/mp4)
```

**Why two Python interpreters:** `run_local_script` launches the **wrapper** with the Wan **venv** (`python_bin` / `WAN_PYTHON`). The wrapper launches **`generate.py`** with the same venv. The embedded training interpreter only orchestrates; it does not need Wan’s PyTorch stack.

## Requirements

- **`OLLAMA_TRAINING=true`** (default when PyTorch embed is available). If training is disabled, `POST /v1/videos` returns **503**.
- **GPU ~16 GB** for shipped **16g** presets (`wan2.1-t2v:1.3b`, `wan2.2-ti2v-5b`).
- **Host RAM ~16 GB** for the Wan subprocess on 16g presets (T5 released after encode; see below). A **24 GB** CT is comfortable; **16 GB** works with `WAN_UNLOAD_T5=1` (default on 16g).
- Wan repo + checkpoints (install script below).

## Install

```bash
./scripts/video/install_wan_video.sh --profile all   # or 1.3b | 2.2
./scripts/video/register_wan_models.sh
```

**`flash_attn`:** Off by default (`WAN_INSTALL_FLASH_ATTN=1` to compile). Upstream honors **`MAX_JOBS`** (ninja `-j`) and **`NVCC_THREADS`** (`nvcc --threads`); if `MAX_JOBS` is unset, setup.py sets it from **container CPU count** (not `WAN_FLASH_ATTN_MAX_JOBS`). Our script exports both, e.g. `WAN_INSTALL_FLASH_ATTN=1 WAN_FLASH_ATTN_MAX_JOBS=1 WAN_NVCC_THREADS=1`. For 5080 only: `FLASH_ATTN_CUDA_ARCHS=120`. Wan works without it (PyTorch SDPA).

**Why a separate install:** Wan weights and CUDA deps are large and version-pinned to upstream repos; pulling them into every `zerollama serve` image would bloat all users.

Register uses **config-only** manifests (`scripts/register_wan_manifest`)—no `FROM` GGUF layer.

## API (v1)

| Method | Path | Notes |
|--------|------|--------|
| POST | `/v1/videos` | JSON: `model`, `prompt`, optional `size`, `seed`, `options.frames`, `options.steps` → **202** + job |
| GET | `/v1/videos/:id` | `status`, `progress`, `model`, `size`, `created_at`, optional `error` |
| GET | `/v1/videos/:id/content` | `video/mp4` when job completed; **202** while running; **410** if cancelled |

### Status mapping

| Internal (training / defer) | OpenAI `status` | Why |
|-----------------------------|-----------------|-----|
| defer waiting | `pending` | Job not yet in Python queue (inference busy). |
| pending / promoted | `queued` | Accepted; waiting for GPU slot or promotion. |
| running | `in_progress` | `generate.py` active. |
| completed | `completed` | Artifact path recorded in `resultJson`. |
| failed | `failed` | Subprocess or Wan error. |
| cancelled | `cancelled` | Distinct from failed (defer cancel or train cancel). |

**Why `created_at` is stable:** Go stamps `submitted_at` on the payload at submit time; Python stores it on the job; polls parse the same timestamp—so the 202 response and later GETs agree.

**Not in v1:** list videos, `DELETE /v1/videos/:id` (use `/api/train/jobs/:id` for pending defer/train jobs only), multipart image-to-video (`input_reference`), `:cloud` models.

**Cloud:** models with `:cloud` suffix are rejected with **400**—Wan is local-only.

## Example

```bash
JOB=$(curl -s http://127.0.0.1:8080/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{"model":"wan2.1-t2v:1.3b","prompt":"A cat on a stage","size":"832x480"}' \
  | jq -r .id)

curl -s "http://127.0.0.1:8080/v1/videos/${JOB}" | jq .
curl -o out.mp4 "http://127.0.0.1:8080/v1/videos/${JOB}/content"
```

CLI (polls the same API):

```bash
zerollama run wan2.1-t2v:1.3b "A cat on a stage"
```

## VRAM, queues, and inference

- Submit path participates in **VRAM broker** / training handoff (unloads ggml runners when configured).
- **`ZEROLLAMA_BLOCK_INFERENCE_DURING_TRAINING`** may block chat/runtime proxy while a video job holds the training slot—**why:** one GPU cannot fairly run chat + Wan without policy.
- **One running training-queue job** at a time; extra submits wait or get **`defer-*`** ids when `queue_on_busy` applies.
- **Poll the id returned from POST** (`defer-…` or Python job id). For defer jobs, status merges into the promoted Python job while keeping the defer id in responses when you poll `defer-*`.

## Artifacts and timeouts

- Output: `$OLLAMA_MODELS/generated/<job_id>.mp4` (`{job_id}` expanded when the job starts).
- Override timeout: `ZEROLLAMA_WAN_VIDEO_TIMEOUT` (seconds) replaces manifest `video_generation.timeout_sec`.
- Wrapper passes `WAN_SUBPROCESS_TIMEOUT` to bound inner `generate.py` wait (aligned with job timeout).

**Why no “find latest mp4 in repo” fallback:** A scan could return a **stale** file from an earlier run. We require `--save_file` output at the expected path.

## Presets (16g)

Manifests: `modelfiles/wan2.1-t2v/config.json`, `modelfiles/wan2.2-ti2v-5b/config.json`.

| Profile | Notes |
|---------|--------|
| **wan2.1-t2v-1.3b** | ~49 frames default on 16g; manifest sets `offload_model` + `t5_cpu` (T5-XXL does not fit on GPU with DiT on 16GB). |
| **wan2.2-ti2v-5b** | Up to **81** frames on 16g (cap); manifest enables `offload_model`, `t5_cpu`; Go sets `WAN_CONVERT_MODEL_DTYPE` because upstream README recommends it for consumer GPUs. |

`backend_paths`: `wan_repo`, `wan_ckpt_dir`, `wan_venv` (default `~/.zerollama/third_party/wan/venv`).

**`vae_tiling` in manifest:** Documented for operators; upstream `generate.py` has no `--vae_tiling` flag—VRAM tuning for 2.2 uses `--offload_model`, `--t5_cpu`, `--convert_model_dtype` instead.

On `vram_tier: 16g`, request `options.frames` is capped unless manifest `video_generation.frames` sets a higher ceiling.

## Environment variables

| Variable | Role |
|----------|------|
| `OLLAMA_TRAINING` | Must be enabled for `/v1/videos`. |
| `OLLAMA_MODELS` | Artifact root `$OLLAMA_MODELS/generated/`. |
| `ZEROLLAMA_WAN_VIDEO_TIMEOUT` | Global timeout override (seconds). |
| `ZEROLLAMA_TRAINING_QUEUE_ON_BUSY` | Defer video submit when inference busy (with submit `queue_on_busy`). |
| `WAN_PYTHON` / `WAN_VENV` | Override interpreter for Wan (see install script). |
| `WAN_DISABLE_CUDNN` | `1` force off, `0` force on, unset = probe cache (`~/.zerollama/third_party/wan/.wan_torch_probe.json`). |
| `WAN_UNLOAD_T5` | `1` (default on 16g) drop T5-XXL from host RAM after prompt encode (~11G freed). |
| `WAN_VAE_CPU` | `1` (default on 16g) CPU VAE decode — GPU VAE needs ~15G contiguous VRAM on 5080; `0` if you have headroom. |
| `ZEROLLAMA_WAN_UNLOAD_T5` / `ZEROLLAMA_WAN_VAE_CPU` | Wrapper defaults when `WAN_*` unset. |

## Code map

| Area | Path |
|------|------|
| HTTP handlers | `server/video_generate.go` |
| OpenAI types / status | `openai/openai.go` (`Video`, `VideoFromTrainingJob`) |
| Defer queue metadata | `server/training_submit.go` |
| Wrapper script | `scripts/video/wan_video_generate.py` |
| SM120 torch probe | `scripts/video/wan_torch_compat.py` |
| Host RAM hooks (T5 unload) | `scripts/video/wan_memory_hooks.py` |
| Generate entry (cuDNN probe) | `scripts/video/wan_generate_entry.py` |
| Job execution | `training.py` (`run_local_script`, `{job_id}` substitution) |
| Wire format | `x/trainingworker/pyembed/bootstrap.py` (`_job_to_dict`) |
| CLI | `x/videogen/cli.go` |
| Install / register | `scripts/video/install_wan_video.sh`, `scripts/video/register_wan_models.sh` |

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| **503** on POST | `OLLAMA_TRAINING=false` or embed failed to start. |
| **400** cloud model | Local Wan only. |
| **502** on GET/content | Training worker error; check daemon logs (`SCRIPT:` / `SCRIPT ERROR:`). |
| **404** defer id | Expired tombstone or wrong id; use id from POST response. |
| **202** on content | Job still running—poll until `status=completed`. |
| Missing mp4 after “success” | `generate.py` did not write `--save_file` path—check Wan logs in job stdout. |
| **`CUDA out of memory`** (T5 / `text_encoder.model.to`) | On 16g use `t5_cpu` + `offload_model` (set in `wan2.1-t2v` manifest; re-run `./scripts/video/register_wan_models.sh`). Unload other GPU models before video. |
| **Host RAM ~24G** / CT OOM | T5-XXL (~11G) stays loaded upstream until encode finishes. Default **`WAN_UNLOAD_T5=1`** on 16g frees it before diffusion. Avoid **`WAN_VAE_CPU=1`** unless GPU decode OOMs (CPU VAE adds host RAM). Brief startup spike may need **swap** on a 16G CT while T5+DiT load. |
| **`cuDNN version incompatibility`** (PyTorch 9.19 vs runtime 9.10) at GPU check | **`LD_LIBRARY_PATH`** from `zerollama serve` (ggml `/usr/hostlibs`, CUDA toolkit) shadows PyTorch's bundled cuDNN. Fixed in-tree: wrapper prepends `venv/.../torch/lib` and drops cudnn/hostlibs entries (`wan_torch_compat.sanitize_ld_library_path_for_pytorch`). Retry the job; or unset hostlibs from serve env if you manage LD manually. |
| **`free(): invalid pointer`** / exit **250** at `0/25` sampling | **cuDNN conv bug on SM120 (5080)** with torch 2.11+cu128: cuDNN-backed `Conv2d`/`Conv3d` SIGABRT. Not flash_attn. Run `./scripts/video/install_wan_video.sh` (runs `wan_torch_compat.py` probe); entry auto-disables cuDNN and uses native CUDA conv on GPU. Override: `WAN_DISABLE_CUDNN=0` only after verifying a fixed torch/cuDNN drop. Nightly 2.12+cuDNN 9.20 still fails as of 2026-04. |
| **`free(): invalid pointer`** (legacy flash_attn) | Source-built `flash_attn` on SM120 can also SIGABRT. Keep `pip uninstall flash-attn`; **`WAN_FORCE_SDPA=1`** uses PyTorch SDPA. |
| **`CUDA unknown error`** in generate.py | Broken GPU in CT after OOM/crash: check `ls -l /dev/nvidia-uvm` (should be `crw*`, not `----------`). Restart the GPU CT or re-attach passthrough; verify `wan venv/bin/python3 -c "import torch; print(torch.cuda.is_available())"`. |
| Wrong torch / import from legacy `venv-training/` | Obsolete repo-root venv (3.10 era). Training embed reads **`.venv-training/lib/pythonX.Y/site-packages`** only. Remove `venv-training/` after migrating to 3.11; see [gpu-training.md](./gpu-training.md#installing-python-deps-embedded-interpreter). Wan subprocesses strip training `PYTHONPATH` — rebuild/restart serve after pull. |
| Wrong Python / import errors | Set `backend_paths.wan_venv` or `WAN_PYTHON` to venv from `install_wan_video.sh`. |
| `flash_attn` / `No module named torch` during pip | Use `install_wan_video.sh` (torch first). Do not bare `pip install -r requirements.txt`. |
| Host OOM during `flash_attn` | Many `cicc` = ninja + nvcc threads. Set **`MAX_JOBS=1 NVCC_THREADS=1`** on the same line as pip, or use our script with `WAN_INSTALL_FLASH_ATTN=1 WAN_FLASH_ATTN_MAX_JOBS=1 WAN_NVCC_THREADS=1`. Or skip compile entirely. |

## Future (see [ROADMAP.md](./ROADMAP.md))

- v1.1: artifact TTL, `DELETE /v1/videos/:id`, kill running subprocess
- 24g/32g manifests, TI2V `input_reference`, optional GGUF Wan2.2 path
- **Optional [Wan CPU worker](./wan-cpu-worker.md)** — remote T5 encode / VAE decode on a high-RAM host (16 GB GPU CT target)
- E2E smoke: `RUN_E2E_WAN_T2V=1` when `scripts/e2e_wan_t2v.sh` lands

Not part of default CI today.
