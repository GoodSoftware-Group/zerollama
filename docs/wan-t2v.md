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

### Apple Silicon (Darwin)

- Install uses **Python ≥3.10** (prefer 3.11/3.12) and **MPS torch**; system 3.9 + recent wheels often **SIGSEGV** on `import torch`.
- Default device is **MPS** with **float32 DiT** + CPU VAE (`WAN_VAE_CPU=1`). Force CPU with `WAN_FORCE_CPU=1`.
- Serve must not leak `.venv-training` into Wan’s `DYLD_LIBRARY_PATH` (handled by `wan_torch_compat` / training `run_script` env).
- Free disk before `--profile 2.2` (~20 GB weights).

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

**Not in v1:** list videos, `DELETE /v1/videos/:id` (use `/api/train/jobs/:id` for pending defer/train jobs only), `:cloud` models.

**Shipped (keyframes):** session media uploads + TI2V multi-keyframe (see below). OpenAI multipart `input_reference` is still not a separate field — use `/v1/media/{session}/{label}` + `options.keyframes`.

**Cloud:** models with `:cloud` suffix are rejected with **400**—Wan is local-only.

## Media uploads (`/v1/media`)

Full design (problem→choice→**why**, lifecycle, errors): **[media-uploads.md](./media-uploads.md)**.

Agents upload stills (and later clips) **before** `POST /v1/videos` so large base64 bodies do not blow JSON limits.

**Why a separate API:** Create stays under an 8 MiB JSON cap; frames stream via PUT; retries re-PUT labels instead of resending megabytes inside the job body. **Why not model `blobs/`:** soft TTL state vs permanent layers — see media-uploads.

| Method | Path | Notes |
|--------|------|--------|
| PUT | `/v1/media/{session}/{label}` | Stream raw bytes; server SHA-256 CAS-dedupes under `$OLLAMA_MODELS/media/cas/` |
| HEAD | `/v1/media/{session}/{label}` | Exists? |
| GET | `/v1/media/{session}` | List labels `{label, digest, bytes, content_type, kind}` |
| GET | `/v1/media/{session}/{label}` | Download bytes |
| DELETE | `/v1/media/{session}/{label}` | Drop pointer (CAS may linger until LRU) |

- **Session / label:** `[A-Za-z0-9._-]{1,128}` — agent-chosen (e.g. `anim-$(uuidgen)` / `kf0`).
- **No refcounts:** TTL + byte-cap LRU; if CAS is gone, `POST /v1/videos` returns **400** `code=media_missing` with `missing_labels` — re-PUT and retry. **Why no refcounts:** every training/job path would need pin/unpin; miss→re-upload is enough for soft animation state.
- **Kinds:** `image` / `video` / `other` from Content-Type + sniff. Wan keyframes must be **images**.

## Keyframe → inbetweens (TI2V)

Requires `wan2.2-ti2v-5b` (not `wan2.1-t2v`). N image labels → **N−1** start-conditioned Wan segments (start = label *i*), then ffmpeg appends a short **still of the final keyframe** so the clip ends on that frame.

**Why N−1 start-conditioned segments (not true FLF):** shipped Wan TI2V conditions on a *start* image per call; full first→last diffusion needs a later FLF runner. **Why append a final still:** without it the last keyframe was unused and the clip did not land on the agent’s intended end pose — the still enforces the end frame visually until FLF lands.

**Why materialize to `generated/keyframes/kf-*`:** CAS may be LRU-evicted while Wan runs for tens of minutes; staging (hardlink/copy) freezes paths for the job. Staging is removed on submit failure and (when `VIDEO_CLEANUP_KEYFRAME_DIR=1`) after the wrapper finishes.

**Why ffmpeg `-c copy` then re-encode:** stream-copy is fast when segment codecs match; mismatched SPS/PPS or timebases from per-segment Wan outputs often break concat — libx264 fallback keeps the job from failing after expensive generation.

```bash
SID=anim-$(uuidgen)
for i in 0 1 2 3; do
  curl -X PUT "http://127.0.0.1:8080/v1/media/${SID}/kf${i}" \
    -H 'Content-Type: image/png' --data-binary @frame${i}.png
done
curl -X PUT "http://127.0.0.1:8080/v1/media/${SID}/final" \
  -H 'Content-Type: image/png' --data-binary @final.png

curl -s "http://127.0.0.1:8080/v1/media/${SID}" | jq .   # optional verify

JOB=$(curl -s http://127.0.0.1:8080/v1/videos \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"wan2.2-ti2v-5b\",
    \"prompt\": \"smooth natural motion between poses\",
    \"size\": \"832x480\",
    \"options\": {
      \"frames\": 49,
      \"media_session\": \"${SID}\",
      \"keyframes\": [\"kf0\", \"kf1\", \"kf2\", \"kf3\", \"final\"]
    }
  }" | jq -r .id)
```

Alternate refs: `options.keyframes: ["${SID}/kf0", …]` without `media_session`.

**Backends:** `modality_backends.video_generation=wan` (shipped). `rife` is reserved for classical optical-flow inbetweens (same media API; not implemented yet). **Why reserve `rife` now:** agents and OpenAPI can name the backend without a second upload protocol when classical inbetweens ship.

**Limits:** image PUT ≤ 25 MiB; video PUT ≤ 256 MiB (for future clip inputs). Staging keyframes under `$OLLAMA_MODELS/generated/keyframes/` are removed after the job (or if submit fails).

## Example (text-only T2V)

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
- **Job-scoped exclusive GPU** (default): after submit, `/v1/videos` holds `fulfillment=exclusive` + `ZEROLLAMA_BLOCK_INFERENCE_DURING_TRAINING` until the job is `completed`/`failed`/`cancelled`. **Why:** chat `fulfillment` is request-scoped and ends when POST returns; Wan needs the card for tens of minutes. Opt out: `"options":{"zerollama":{"fulfillment":"none"}}`.
- **`ZEROLLAMA_BLOCK_INFERENCE_DURING_TRAINING`** also blocks chat/runtime proxy while a video job holds the training slot—**why:** one GPU cannot fairly run chat + Wan without policy.
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
| **wan2.2-ti2v-5b** | Up to **81** frames on 16g (cap); manifest enables `offload_model`, `t5_cpu`; Go sets `WAN_CONVERT_MODEL_DTYPE` because upstream README recommends it for consumer GPUs. Go also defaults **`WAN_MMGP=1`** + profile **5** so DiT is budget-paged (stock `offload_model` alone still full-loads ~10 GiB). |

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
| `WAN_VAE_CPU` | Under **mmgp** (16g TI2V default): **`0`** — GPU VAE folded into mmgp pipe. CPU VAE + mmgp thrash ≤24 GiB hosts and can OOM-kill serve. Without mmgp on 16g: `1`. Override: `ZEROLLAMA_WAN_VAE_CPU`. |
| `ZEROLLAMA_WAN_UNLOAD_T5` / `ZEROLLAMA_WAN_VAE_CPU` | Wrapper/Go defaults when `WAN_*` unset. |
| `ZEROLLAMA_WAN_MIN_HOST_RAM_GIB` | **Raise-only** admission floor (default profile floors: mmgp+GPU VAE ≥12 GiB, mmgp+CPU VAE ≥24 GiB, CPU VAE ≥14 GiB). Undercutting requires `ZEROLLAMA_WAN_MIN_HOST_RAM_FORCE=1`. Wired in `server/video_admit.go` — reject with **503** before queue. |
| `ZEROLLAMA_WAN_OMP_NUM_THREADS` | Cap Wan child OpenMP/torch threads (default **2**) so serve keeps CPU. |
| `ZEROLLAMA_WAN_HOST_RESERVE_GIB` / `WAN_RLIMIT_AS_GIB` | Reserve is documentation/admission context. **`RLIMIT_AS` default off** — CUDA maps GPU VRAM into process VA; a tight AS cap breaks `cudaGetDeviceCount`. Set `ZEROLLAMA_WAN_RLIMIT_AS_GIB` only with a large value (e.g. 96+) if you need a hard virtual cap. |
| `WAN_MMGP` / `ZEROLLAMA_WAN_MMGP` | `1` (default on 16g **TI2V**) enables [mmgp](https://github.com/deepbeepmeep/mmgp) layer/budget offload so stock Wan cannot full-load DiT via `model.to(cuda)`. Opt out: `0`. See [wangp-borrowings.md](./wangp-borrowings.md). |
| `WAN_MMGP_PROFILE` / `ZEROLLAMA_WAN_MMGP_PROFILE` | mmgp profile **5** default (VerylowRAM_LowVRAM). With GPU VAE under mmgp, host floor is ~12 GiB; CPU VAE still wants ~24 GiB. |
| `WAN_MMGP_QUANTIZE` / `ZEROLLAMA_WAN_MMGP_QUANTIZE` | `1` → on-the-fly int8 transformer quant if profile 5 still OOMs (default `0` for bf16 quality). |

## Code map

| Area | Path |
|------|------|
| HTTP handlers | `server/video_generate.go`, `server/media_handlers.go` |
| Media CAS index | `server/media/` (`$OLLAMA_MODELS/media/`) — design [media-uploads.md](./media-uploads.md) |
| OpenAI types / status | `openai/openai.go` (`Video`, `VideoFromTrainingJob`, `NewMediaMissingError`) |
| Defer queue metadata | `server/training_submit.go` |
| Wrapper script | `scripts/video/wan_video_generate.py` |
| SM120 torch probe | `scripts/video/wan_torch_compat.py` |
| Host RAM hooks (T5 unload / mmgp) | `scripts/video/wan_memory_hooks.py` — [wangp-borrowings.md](./wangp-borrowings.md) |
| Host admission / containment | `server/video_admit.go` (MemAvailable gate, OMP caps, `RLIMIT_AS`) |
| Generate entry (cuDNN probe) | `scripts/video/wan_generate_entry.py` |
| Job execution | `training.py` (`run_local_script`, `{job_id}` substitution) |
| Wire format | `x/trainingworker/pyembed/bootstrap.py` (`_job_to_dict`) |
| CLI | `x/videogen/cli.go` |
| Install / register | `scripts/video/install_wan_video.sh`, `scripts/video/register_wan_models.sh` |

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| **503** on POST | `OLLAMA_TRAINING=false`, embed failed, **or host-RAM admission** (`server/video_admit.go`) — box total/free below plan floor. |
| **400** cloud model | Local Wan only. |
| **502** on GET/content | Training worker error; check daemon logs (`SCRIPT:` / `SCRIPT ERROR:`). |
| **404** defer id | Expired tombstone or wrong id; use id from POST response. |
| **400** `media_missing` | Label never uploaded, TTL expired, or CAS LRU evicted — re-PUT `missing_labels`. |
| **400** `media_type_mismatch` | Wan keyframe not an image. |
| **413** on PUT / POST | Over media or create-body cap — use `/v1/media` for frames ([media-uploads.md](./media-uploads.md)). |
| **202** on content | Job still running—poll until `status=completed`. |
| Missing mp4 after “success” | `generate.py` did not write `--save_file` path—check Wan logs in job stdout. |
| **`CUDA out of memory`** (T5 / `text_encoder.model.to`) | On 16g use `t5_cpu` + `offload_model` (set in `wan2.1-t2v` manifest; re-run `./scripts/video/register_wan_models.sh`). Unload other GPU models before video. |
| **`CUDA out of memory`** mid TI2V diffusion (~10 GiB DiT) | Ensure **exclusive** GPU + **`WAN_MMGP=1`** (default on 16g TI2V). Logs: `WAN: mmgp profile=5` and `skipped full DiT .to(cuda)`. Default under mmgp is **GPU VAE** (not CPU). If still OOM: `WAN_MMGP_QUANTIZE=1`. |
| **Host RAM thrash / serve dies / 8 CPUs pegged** | CPU VAE + mmgp on ≤16–24 GiB CT — refused by admission when CPU VAE forced without ≥24 GiB. Prefer mmgp+GPU VAE; OMP capped; child `RLIMIT_AS` leaves reserve for serve. Do **not** Ctrl-C serve mid-job. |
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
- 24g/32g manifests, optional GGUF Wan2.2 path
- Wan FLF / first+last on TI2V-5B; **RIFE** backend on the same `/v1/media` labels
- Video-clip morph (`options.clips`) using the same media store (`kind=video`)
- **Optional [Wan CPU worker](./wan-cpu-worker.md)** — remote T5 encode / VAE decode on a high-RAM host (16 GB GPU CT target)
- E2E smoke: `RUN_E2E_WAN_T2V=1` / `RUN_E2E_WAN_TI2V=1` when scripts land

Not part of default CI today.
