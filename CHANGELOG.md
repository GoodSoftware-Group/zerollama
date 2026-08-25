# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### CUDA plugin discovery from `run/zerollama` — Aug 2026

**Why:** `nvidia-smi` and `OLLAMA_LLM_LIBRARY=cuda_v13` were fine, but serve still started CPU-only (`gpu_found=false` in ~1 ms) when the binary lived in `run/zerollama`. `ml.LibOllamaPath` only checked `../lib/ollama` / `build/lib/ollama` / `dirname(exe)` and never used `OLLAMA_LIBRARY_PATH` or `/usr/lib/ollama`.

**Shipped:** honor `OLLAMA_LIBRARY_PATH` and Linux package dirs (`/usr/lib/ollama`, `/usr/local/lib/ollama`); prefer a root that actually has `cuda_v*` (etc.); glob those roots in GPU bootstrap; INFO when `OLLAMA_LLM_LIBRARY` is set but that plugin dir was not searched. The `~/zerollama/lib/ollama → /usr/lib/ollama` symlink is no longer required.


### OpenAPI LocalAI control plane — Aug 2026

**Why:** `/docs` lagged the shipped LA7/LA9/LA11/LA15/LA17 routes so agents could not discover aliases, score, or routers.

**Shipped:** `server/openapi/openapi.yaml` documents `/api/aliases`, `/api/score`, `/api/router/decide`, `/api/router/corpus`, `/api/repair`, experimental `web_search`/`web_fetch` (SSRF 400), plus `X-Zerollama-*` rewrite headers and LA18 cooldown on generate/chat 503.

### LocalAI LA15 outbound SSRF — Aug 2026

**Why:** Gallery-style URL validation so user-supplied fetches cannot hit loopback, RFC1918, or cloud metadata.

**Shipped:** `internal/ssrf.ValidateExternalURL` (LocalAI `IsPublicIP`); video_url, HF pull/redirects, registry blob `Location`, experimental `web_fetch` `url`. `http://huggingface.co` upgrades to HTTPS.

### SGLang multimodal ports (#32914 / #34892 / #31957 / #33898) — Aug 2026

**Why:** Weekly SGLang scan (`4e5a05148a..896acc8860`) flagged portable agent-facing MM guards; CUDA-IPC / HiCache / Radix skipped.

**Shipped:**
- Reject still images / padded multimodal on text-only models with **400** (chat + generate) — #32914
- Opt-in `OLLAMA_MEDIA_ALLOWED_HOSTS` + redirect host re-check + `Content-Length` early reject for `video_url` — #34892
- Reject zero/`frame_count` mismatches on pre-expanded `video_spans` (no silent grid remount) — #31957
- Qwen3-VL tool-result media uses `renderContent`; OpenAI multipart tool messages keep `tool_call_id` — #33898

Doc: [sglang-multimodal-borrowings.md](docs/sglang-multimodal-borrowings.md).

### LocalAI LA17 model aliases — Aug 2026

**Why:** Clients send `gpt-4` without a copied manifest; `cp` is too heavy for a name redirect.

**Shipped:** `~/.ollama/aliases.yaml` one-hop map; `GET`/`POST /api/aliases`; rewrite on chat/generate/embed/score/show. Chains rejected.

### LocalAI LA11b KNN router — Aug 2026

**Why:** Score-LM routing needs a small classifier GGUF; many operators already have an embed model and labelled examples.

**Shipped:** `classifier: knn` + `embedder` + YAML corpus; cosine KNN (k=3, sim 0.80, vote 0.5); undecidable → fallback; `GET`/`POST /api/router/corpus` (counts / session overlay).

### LocalAI LA11 score router — Aug 2026

**Why:** One client-facing model name should pick a specialist from the prompt without a generate round-trip.

**Shipped:** `~/.ollama/router.yaml` policies; `POST /api/router/decide`; softmax over LA9 scores; in-band `/api/chat` and `/api/generate` rewrite. KNN corpus is **LA11b** (not in this slice).

### LocalAI LA18–LA20 scheduler lifecycle — Aug 2026

**Why:** Polling agents respawn crash-loop loads; hard-killed `zerollama` can orphan `llama-server`; operators need a hard VRAM ceiling below physical.

**Shipped:** Failed-load geometric cooldown (`ZEROLLAMA_LOAD_COOLDOWN`, `503` + `Retry-After`). Linux `Pdeathsig` SIGKILL on ggml / llama-server / MLX runner subprocesses (`ZEROLLAMA_BACKEND_PARENT_WATCH`). `ZEROLLAMA_VRAM_BUDGET` (`80%` / `12GiB`) caps Go GPU discovery and Python free-VRAM probes (`min(detected, budget)`).

### vLLM skip covered MM payload (#52041) — Aug 2026

**Why:** vLLM drops multimodal tensor bytes for items whose placeholder span is fully inside a prefix-cache-covered region — workers never consume them.

**Shipped:** ollama-engine and ggml llamarunner `deferVisionEncode` stub GridTHW vision spans for input-cache lookup, then hydrate ViT/mtmd only on the uncached tail; llama-server subprocess strips covered `multimodal_data` on agent turn N+1 via session prefix tracker for Qwen3-VL plus Gemma3/4, mllama, Llama4, LFM2/GLM-OCR, Mistral3, and DeepSeek-OCR padded layouts.

### vLLM retention default (#52216) — Aug 2026

**Why:** vLLM promoted `prefix_cache_retention_interval` and changed unset default from dense to `0` (block-aligned SWA checkpoints only).

**Shipped:** `ZEROLLAMA_PREFIX_CACHE_RETENTION_INTERVAL` unset + no YAML `l3.retention_interval` → `0`. Explicit `N>0` or YAML override unchanged.

### vLLM L3 pattern ports (#50321 / #48668) — Aug 2026

**Why:** Aug 20 vLLM rescan (`118bcde44` → `f8e0602713`) flagged partial LMCache tier hits and zero-output steps dropping prefix-cache metrics.

**Shipped:**
- Partial secondary-tier load: resume from longest LMCache prefix when tail blocks are absent remotely (not all-or-nothing `prefix_block_hash_mismatch`)
- Zero-output prefix-cache metrics: stream tail + non-stream `/api/generate` keep `cached_prompt_tokens` / `cache_creation_tokens` when `eval_count=0`

Doc: [vllm-borrowings.md](docs/vllm-borrowings.md), [upstream-siblings.md](docs/upstream-siblings.md).

### MiniMax H3 video-c — 24→50 DiT layer audio fix (Aug 2026)

**Why:** H3 T2VA output was a ~93%-clipped audio waveform (`a_rms=45.34`,
11895/12800 samples clipped). The `--generate` path silently ran only the first
**24 of the 50** DiT blocks (`H3_DIT_DEFAULT_GENERATE_LAYERS = 24`), truncating
the residual stack so the final AdaLN/RMSNorm saw a wrong hidden state — the
audio velocity was ~20–70× too large and Euler integrated it off the VAE's
~1.3 latent manifold. The "50L rank-1 cliff" reasoning that justified the cap
was a misdiagnosis of correct-model behavior at small canvases; ComfyUI's own
`_forward` on this exact pruned int8 export produces the photoreal fox at
1344×768×**50L**.

**What:**
- Default `H3_DIT_DEFAULT_GENERATE_LAYERS` **24 → 50** (full model) in `x/video-c/include/h3_dit_host.h`, with a WHY comment; `main.c --layers` help updated (generate default 50, `--dit-denoise` stays 1-layer smoke test)
- Verified stage-by-stage against ComfyUI `_forward` (MPS bf16, dequantized int8 export): host raw hidden `h_audio_rms`~6–8e3, final RMSNorm ~0.37, curve-table `scale_rms=0.852`/`shift_rms=0.010` (bit-identical to Comfy), audio velocity rms ~0.35–0.7 vs host 50L `vel_audio≈1.0`; 24L was ~68
- New regression gate: `latent_rms=1.18298 a_rms=0.504888`, `clipped=0/12800` (seed=1, "A red fox walking through snow"), replaces `latent_rms=17.2124 a_rms=45.3436`
- Env-gated `H3_DUMP_STAGES=1` stage diagnostic in `family_h3/h3_dit_forward.c` (raw h / norm / scale / shift / ha / velocity per step; zero overhead when unset)
- Docs corrected: [video-c.md](./docs/video-c.md) (removed the wrong "50L rank-1/gray / 24-layer" theory and old "science — closed" conclusion; 24L artifacts recontextualized as truncation), [ROADMAP.md](./docs/ROADMAP.md) v1.4b, `x/video-c/README.md`, `x/video-c/AGENTS.md`

### MiniMax Music 3 — hear on Mac, then C (Aug 2026)

**Why:** Local song generation is not Piper TTS and not MiniMax cloud `/v1/music_generation`. ComfyUI’s Music 3 port is GPL (never a runtime). Omni CUDA `sgl-omni serve` cannot be the first listen on Apple Silicon. H3 AudioVAE is the wrong VAE.

**What:**
- Lab hear: `scripts/audio/music3_mlx_generate.py` (mlx-audio pin `784b29e`) + `mlx-community/MiniMax-Music3-8bit`; venv `.venv-music`
- C11 `x/music-c` — Omni prompt/chunk/DAV geometry; `--tokenize` is prompt pack; `--decode-audio` synthetic until `dav.pth`
- HTTP: `POST /v1/audio/generations` (202) + poll/content; `speech=music3` aliases the same **async** job (not OpenAI WAV bytes)
- `training.py` expands `{job_id}` in **all** run_script env strings (not only `WAN_OUTPUT_PATH`); default python `.venv-music`
- Explicit `duration` wins over `max_new_tokens`; exclusive GPU hold like Wan
- Docs: [music-c.md](./docs/music-c.md), [music-c-findings.md](./docs/music-c-findings.md)

### Training T9 (Partial) — stock Trainer efficiency (unified backend)

- **No** second backend / Unsloth Core fork — polish existing `training.py` + PEFT + Transformers `Trainer`
- [`training_optim.py`](./training_optim.py): gradient checkpointing (CUDA default on); `adamw_torch_fused` / `adamw_bnb_8bit` (QLoRA); pin_memory; `use_rslora` default on; optional LoftQ; richer knobs (`gradient_accumulation_steps`, `seed`, `max_steps`, `torch_compile`)
- [`training_labels.py`](./training_labels.py): **completion-only loss** (default on; skipped when `packing=true`)
- Auto `padding_free_flash_attn` when flash-attn is installed
- Tests: `python3 -m unittest tests.test_training_optim`

### Training T7 (Done) — train → serve export

- [`training_export.py`](./training_export.py): `register_model` writes `FROM`+`ADAPTER` Modelfile; create via CLI or **HTTP** blob upload + `POST /api/create` (`register_via=auto|cli|http`)
- `export_gguf`: merge LoRA → `convert_hf_to_gguf` → `llama-quantize` (`export_quant`); **memory-cap** unload/empty-cache after merge before convert (`export_unload`)
- Wired from `training.py` after `lora_adapter/` save; result includes `export`
- Smoke: `./scripts/training/t7_train_export_smoke.sh` (`RUN_E2E_T7=1` on lab `:11435`)
- Tests: `python3 -m unittest tests.test_training_export`

### Training T8 (Done) — padding-free + packing + chat templates + Modelfile render

- [`training_collate.py`](./training_collate.py): **`padding_free` default on** → `DataCollatorWithFlattening`; `padding_free=false` → longest-pad; opt-in `padding_free_flash_attn` → FA2 + `cu_seq_lens_*` (`collate=flattening_flash`)
- [`training_format.py`](./training_format.py): `format=auto|chatml|llama3|hf|alpaca|modelfile`; `messages[]`; `max_length` default 2048
- [`training_modelfile.py`](./training_modelfile.py) + [`zerollama template render --train`](./cmd/template_render.go): serve-parity Go TEMPLATE for SFT (strips trailing generation priming)
- [`training_pack.py`](./training_pack.py): opt-in `packing: true`
- Loss-curve fixture: [`training_loss_fixture.py`](./training_loss_fixture.py); FA/cu_seqlens smoke: [`scripts/training/t8_flash_attn_5080_smoke.sh`](./scripts/training/t8_flash_attn_5080_smoke.sh) (5080; `flash-attn` optional package)
- Job result includes `format`, `max_length`, `packing`, `padding_free`, `padding_free_flash_attn`, `attn_implementation`, `collate`
- Tests: `python3 -m unittest tests.test_training_format tests.test_training_pack tests.test_training_collate tests.test_training_loss_fixture tests.test_training_modelfile`


### Doctor template hygiene (`--repair-models`)

- New recipes: `chatml_missing_stops` (any ChatML family — stop tokens only), `missing_response_placeholder` (append `{{ .Response }}`), `empty_template`, `think_parser_mismatch` (Qwen3 invasive TEMPLATE rewrites remain family-gated)
- `--all-local` scans `/api/tags` (explicit cold-load opt-in; default still warm `/api/ps` only)
- Docs: [doctor-model-repair.md](./docs/doctor-model-repair.md), minefield §3.1


### Embed start no longer blocks :8080 for 120s (Aug 2026)

**Why:** Restarts looked down for ~2 minutes while `runtimeworker.Start` waited for uvicorn `/health` (or hit the full 120s timeout) before Gin accepted traffic. Phase 17 Go→llama-server does not need embed for text GGUF.

**What:** sync-wait `ZEROLLAMA_RUNTIME_EMBED_SYNC_WAIT` (default **3s**), then finish health poll in the background (total `ZEROLLAMA_RUNTIME_EMBED_READY_WAIT`, default 120s) and publish `BaseURL` when ready.

**Follow-up (CT 1564):** training `Py_Initialize` before Go `setenv(ZEROLLAMA_RUNTIME_EMBED_BOOT)` left Python `os.environ` stale — `/health` omitted `embed_boot` and Go never published `BaseURL`. Fixed via libc `getenv` in `runtime/engine.py` + embed bootstrap. Go `/health` client timeout **2s → 15s** (cold CUDA). **Split llama-server** (~18 KiB + `libllama-server-impl.so`) is discoverable again (1 MiB size gate had forced ggml / `llama_server=off`).

### Inference profile auto + L1 on Phase 17 Go path (Aug 2026)

**Why:** Linux prod defaults to Go → llama-server, which ignored calibrated `runtime/configs/gpu/*.json` (live 5080 loads used f16 KV / `-b 512`). Operators also stacked L1/L3/FORK/graphs env vars instead of one workload lane.

**What:**
- `llm/gpu_profile.go` — Phase 17 launches apply L1 JSON (`rtx-5080`: q8_0, `-b 1024`/`-ub 256`, FA on, np cap); VRAM fit uses the same KV/batch; **nvidia-smi fallback** when DeviceInfo is empty; avoid slog key `source` (reserved)
- `effectiveGgmlFreeVRAM` — nvidia-smi free-VRAM fallback when discover returns **no** devices (fixes `free_vram=0 B` → forced `-np 1`); does not override non-empty zero FreeMemory from loaded-runner accounting
- `ZEROLLAMA_INFERENCE_PROFILE=auto|throughput|agent|vram|off` — soft-defaults when unset; `serve_gpu_example.sh` defaults `auto`
- `/api/status` → `inference.config.inference_profile*` + `gpu_profile_id`; OpenAPI + `runtime_env_doctor.sh` updated
- MTP/spec draft_max from L1 only when MTP/spec is actually enabled
- **Verified on CT 1564:** tinyllama load → `--cache-type-k q8_0 -b 1024 -ub 256 --flash-attn on`, `gpu_profile_id=rtx-5080`
### Media uploads + Wan TI2V keyframe inbetweens (Aug 2026)

**Why:** Agents need N keyframes → short clips without stuffing megabytes of base64 into `POST /v1/videos`. Soft animation state must not share lifecycle with permanent model `blobs/`, and missing frames after TTL/LRU must be recoverable by re-upload (no client digests, no refcount pin/unpin across the training queue).

**What:**
- `PUT/HEAD/GET/DELETE /v1/media/{session}/{label}` + `GET /v1/media/{session}` — server SHA-256 CAS under `$OLLAMA_MODELS/media/`, session pointers, kind sniff (`image`/`video`/`other`), TTL + CAS byte-cap LRU (no refcounts)
- `POST /v1/videos` accepts `options.media_session` + `options.keyframes` (or `session/label` refs); materializes staging under `generated/keyframes/`; **400** `media_missing` / `media_type_mismatch`
- Wan wrapper: N−1 start-conditioned TI2V segments + **final keyframe still**; ffmpeg concat `-c copy` then libx264 fallback; staging cleanup
- Limits: image PUT 25 MiB, video PUT 256 MiB, video-create JSON 8 MiB; `rife` backend reserved
- Docs: [media-uploads.md](docs/media-uploads.md), [wan-t2v.md](docs/wan-t2v.md); skill `generate-video`; OpenAPI media routes

### GPT-OSS mxfp4-q8 MoE router quant (Aug 2026)

**Why:** `zerollama bench gpt-oss-120b:mxfp4-q8` panicked in `SparseMoE.route` (`index out of range [0] with length 0`) — empty router logits.

**What:** Per-tensor quantization overrides that omit `mode` no longer inherit global `mxfp4` when bits are not 4 (mlx-lm mixed exports: routers `bits: 8` → affine/int8). Clearer MoE route shape errors; `ls image_gen` alias for `CapabilityImage`.

### `zerollama ls image_gen` alias (Aug 2026)

**Why:** Capability wire value stays `image` (vision took the short name for understanding), but operators expect a `_gen` pair with `video_gen`.

**What:** `zerollama ls image_gen` / `image-gen` filter the same as `ls image` (`CapabilityImage`). Wire capability unchanged.

### Remote storage RDMA throughput (mlx4 bounce path) (Aug 2026)

**Why:** First remote RDMA READ was only ~30% above 10 GbE TCP (~182 vs ~137 MiB/s) despite 40 Gb/s QDR — serial READ depth 1, per-window `ibv_reg_mr`, and bounce copies left the link idle.

**What:**
- Pipeline outstanding RDMA READs (`max_rd_atomic`/`max_dest_rd_atomic` 16); session advertises `max_rd_atomic` so older peers stay at depth 1 (avoids WC status 9)
- Pin-prefetch next window while reading current; reuse local + server bounce MRs; skip multi-hundred-MiB `memset`
- mmap+mlock MR attempted then bounce fallback (mlx4 `ibv_reg_mr` on file mmap → EFAULT)
- Docs: honest mlx4 bounce throughput expectations

### Doctor model repair (`--repair-models`) + generate think Init order (Aug 2026)

**Why:** Community GGUFs (e.g. milkey Kalomaze Qwen3 MoE, moophlo Qwen3-Coder) scored 0 on harness traps while weights were fine — default `/api/generate` parked answers in `thinking`, or ChatML system role triggered `/` loops. Operators blamed the model; the fix is Modelfile + serve Init order. `doctor --fix` is host bootstrap and must not silently rewrite tags.

**What:**

- `zerollama doctor --repair-models [MODEL...]` — dry-run diagnose (warm `/api/ps` if no args); `--apply` recreates the same tag `FROM` itself with patched `TEMPLATE`/`PARSER`/`stop`
- Recipes (Qwen3 family only): `think_generate_empty`, `slash_system_collapse`; non-qwen3 symptoms → `manual_review` (no auto-patch)
- Live trap-12/64 probes **default `/api/generate`** and points FixHint at `--repair-models`; unload before that arm to avoid prefix-cache poison
- GenerateHandler: default `think=false` **before** thinking-parser `Init` (was `nil` → `defaultThinking` → empty `response`)
- Package [`internal/modelrepair`](internal/modelrepair); guide [docs/doctor-model-repair.md](docs/doctor-model-repair.md)

### Remote model storage daemon (v1) (Aug 2026)

**Why:** Inference disks fill with hundreds of GB of models long before VRAM is the bottleneck. Operators need one canonical `$OLLAMA_MODELS` tree on a bigger box and on-demand fetch into a capped local cache — without NFS-only semantics, without a separate sync tool for every `run`/`chat`, and with room for InfiniBand and later tensor-addressed streaming.

**What:**
- CLI: `zerollama storage serve` (lab `:18090`) and `zerollama storage push [--reclaim]`
- HMAC-SHA256 shared-secret auth; **RDMA READ** data plane (`-tags rdma`, `POST /v1/rdma/session` + MR lease + `IBV_WR_RDMA_READ`) with TCP HTTP Range-GET fallback
- `GetModel` / `ensureBlob` fetch-on-miss; persist LRU + ephemeral scratch
- Scheduler **refcount** pin on load / `ReleaseModelBlobs` on unload (shared layers + auto ephemeral delete)
- Safe reclaim (delete only after all referencing manifests pushed); verify-before-rename; singleflight downloads; hex-only digest paths
- GGUF catalog + `GET /v1/tensor/…`; `tensorproto` spec for future stream/runtime paging

See [docs/remote-model-storage.md](docs/remote-model-storage.md).

### `zerollama ls` CTX column (host-aware) (Aug 2026)

**Why:** Operators hit OOM with dual MLX loads + 80k ctx; `ls` showed PARAMS/PERF but not what context this host can hold *now*.

**What:** `/api/tags` adds `host_max_context` (GGUF GraphSize binary-search vs free VRAM + loaded credit; MLX size heuristic). CLI **CTX** column: train max when it fits, else `16k–80k`-style range. Train ceiling remains `details.context_length`. Narrow TTYs (&lt;100 cols) render `ls`/`ps` as **2-line rows** (`term.GetSize` / `$COLUMNS`) so tables fit ~80 columns; pipes stay wide. Regenerated `docs/assets/demo-operator-cli.gif` (v3) — `ls` shows **CTX** (incl. host–train ranges from live).

### Vendor rebase to ggml-org `5f55650a` (b10199+1) — Jul 2026

**Why:** Track ggml-org master tip (~41 commits past b10159) while keeping the zerollama quilt (MTP think path, pretok, DCA, Eliza KV, server Radix).

**What:**

- `LLAMA_CPP_VERSION=5f55650a`, `LLAMA_CPP_COMMIT=5f55650a78f9…`, `Makefile.sync` → `vendor/llama-cpp-5f55650a` (`BUILD_NUMBER=10199`)
- Rebased **131** Ollama/zerollama patches (`make -f Makefile.sync clean apply-patches` → **fail=0**); **dropped** absorbed CUDA Q2_0 **#25707** (old 0082); fixed corrupt DCA **0097** hunk header; resolved server/speculative/Metal/DCA/unicode path drift; **0127–0131** compile fixups (FP8 brace, Q5_K get_rows, COW/MSA kv-cache, pretok_blob decl, seq-copy/`load_mode` for #26221)
- `./scripts/vendor/sync_vendor_llama.sh` → in-tree `llama/llama.cpp` + `ml/backend/ggml`
- Lab binary: `/opt/zerollama/llama-server-5f55650a` (container CUDA `89-real`); MTP smoke on `:18082` (qwen3.6:27b, `--spec-type draft-mtp`, no mmproj) — content `"4"` with thinking off; thinking-on coherent, draft accept ~38–50%
- Production `:2083` cut over to `/opt/zerollama/llama-server-5f55650a` (`scripts/systemd/dual_4090-ollama.conf`); `f95de977` kept for rollback
- `ZEROLLAMA_KEEP_BUILD=1` (default in container script) keeps failed CUDA build trees for incremental fixups; seq-copy verify glob fixed for `libllama-server-impl*`

### README demo GIF (Jul 2026)

**Why:** Marketing backlog wanted a turn-1 vs turn-2 visual; CLI DX fields needed to show up outside a text table.

**What:** `docs/assets/demo-operator-cli.gif` + `scripts/marketing/make_readme_demo_gif.py` — terminal-style `ls` (PARAMS/**CTX**/PERF), `ps` (PROJECT/SESSION), harness curl, measured tokenize cards (389→81 ms), L3 story card; optional `--ttft-json` from `capture_ttft_for_gif.py` (lab `:11435`) when GPU free. Embedded under §1.5. (Aug 2026: v3 refresh for CTX column.)

### README operator CLI DX (Jul 2026)

**Why:** Marketing tour sold harness APIs but not the day-to-day `ls` / `ps` fields operators actually stare at.

**What:** README §4.6 — `PARAMS` / `PERF` on `ls`, `PROJECT` / `SESSION` (+ PROCESSOR / CONTEXT / UNTIL) on `ps`, with example tables from a live Mac lab.

### Open-source shoutouts (Jul 2026)

**Why:** Marketing pass asked for partner nods beyond a one-line Credits bullet.

**What:** README **Open-source shoutouts** table + [docs/open-source-shoutouts.md](docs/open-source-shoutouts.md). Thank-you issues: [Gigatoken #1](https://github.com/chynggi/gigatoken-llama.cpp/issues/1), [minefield #18](https://github.com/Blackwellboy/model-serving-minefield/issues/18), [Hermes #75009](https://github.com/NousResearch/hermes-agent/issues/75009). Nod to [X](https://x.com) / [@spaceodili](https://x.com/spaceodili) for connecting projects.

### README marketing benches (Jul 2026)

**Why:** README claimed megaprompt tokenize / L3 / decode wins; needed fresh evidence without killing production `:11434`.

**What:** Ran offline `run_tokenize_bpe_identity_bench.sh --bench` (identity green). Qwen2/GPT-2 legacy still **~270–390 ms/MiB**; fast path **~3–7×**. Canonical write-up: [docs/readme-marketing-benches.md](docs/readme-marketing-benches.md). README hero + lab card + §4.1 updated. Skipped L3 + m4 decode A/B while ornith held the GPU / script stops `:11434`.

**Why:** Changelog already shipped tokenizer (0106–0126), minefield doctor, Hermes/QoS, image/video — the top of README still sold “Ollama + training/fleet,” buried agent diffs, and dumped a duplicate community-integrations wall.

**What:** Progressive README (quick start → why → compatibility → tour → API → platforms → docs). Hero = **megaprompts** (Gigatoken-inspired tokenize + SGLang/vLLM-inspired L3 prompt cache ELI5) + visuals; who-for / not-for; lab numbers card; 30-second win; harness-shaped `/api/chat` example; platforms as Apple Silicon / CUDA / Arc. Drop in-repo integrations catalog → link [ollama § Community Integrations](https://github.com/ollama/ollama#community-integrations). Operator prose moved to linked docs.

### Mac setup docs + doctor paths (Jul 2026)

**Why:** Fresh-clone docs said “Xcode CLI + Go 1.22 + uv,” but `go.mod` needs **1.24.1+**, CGO needs full **Xcode.app** (or Homebrew Python) for `python3-embed`, and default bootstrap needs **cmake**. `doctor --fix` still called pre-reorg flat script paths and failed immediately.

**What:**

- [mac-dev-setup.md](docs/mac-dev-setup.md) + [AGENTS.md](AGENTS.md) — current prereqs, post-reorg script map, pin `f95de977` / b10159, public `LLAMA_CPP_REPO=ggml-org` note
- `cmd/doctor.go` + `server/darwin_sidecar.go` — `scripts/runtime/`, `scripts/build/`, `scripts/vendor/` paths
- README / apple-silicon / phase17 — drop stale b9781/b9611 Mac build claims where they mislead onboarding

### SWAR/NEON ASCII pretok consume (patch 0126)

**Why:** Letter/digit runs on ASCII byte pretok still walked an 8-wide LUT loop; BMTL T01/T05 show SWAR/NEON consume wins without rewriting the whole scanner.

**What:**

- **0126** — borrow-safe SWAR + aarch64 NEON letter/digit consume (SWAR-first; NEON after an 8-hit); `LLAMA_BPE_NO_SIMD_PRETOK=1` A/B
- **Measured (Qwen2 medians):** ASCII ~**1.3×** vs `NO_SIMD`; chat/mixed ~**1.1–1.25×**; identity green (`simd-pretok` gate)

Docs: [faster-bpe-tokenize.md](docs/faster-bpe-tokenize.md), [findings](docs/faster-bpe-tokenize-findings.md).

### Space + printable byte-encode fast path (patch 0125)

**Why:** Most English pretok words are ` ?\p{L}+` (leading space). Space remaps under GPT-2 byte-encode, so the 0117 printable skip never fired and every letter paid a LUT append.

**What:**

- **0125** — detect space + printable ASCII; remap space once, `memcpy` the rest (`LLAMA_BPE_NO_BYTE_ENC_FAST` unchanged)
- **Measured:** Qwen2 ASCII ~**7.5 ms/MiB** (was ~8.5); GPT-2 ASCII ~**6.7 ms**; identity green

Docs: [faster-bpe-tokenize.md](docs/faster-bpe-tokenize.md), [findings](docs/faster-bpe-tokenize-findings.md).

### Byte-level mixed pretok islands (patch 0124)

**Why:** After 0122/0123, mixed text still paid a full megaprompt uint32 decode before cpt islands ran.

**What:**

- **0124** — `ascii_bytes_seg` on ASCII gaps; decode only non-ASCII islands; `LLAMA_BPE_NO_BYTE_MIXED=1` A/B
- **Measured (Qwen2):** dense mixed ~**20.4 ms** vs ~**21.8 ms** `NO_BYTE_MIXED` (~1.07×); identity green (byte-mixed gate)

Docs: [faster-bpe-tokenize.md](docs/faster-bpe-tokenize.md), [findings](docs/faster-bpe-tokenize-findings.md).

### ASCII islands for GPT-2 / Llama3 / Qwen3.5 (patch 0123)

**Why:** 0122 only covered Qwen2; mixed GPT-2/Llama3/Qwen3.5 megaprompts still paid the full Unicode pretok scanner after one non-ASCII byte.

**What:**

- **0123** — cpt `ascii_seg` + mixed islands for GPT-2 / Llama3 / Qwen3.5 (family-specific letter/number/punct rules)
- **Measured:** identity green on Qwen2/35/Llama3/GPT-2 (incl. ascii-islands gate); dense mixed ~noise–1.05× vs `NO_ASCII_PRETOK`

Docs: [faster-bpe-tokenize.md](docs/faster-bpe-tokenize.md), [findings](docs/faster-bpe-tokenize-findings.md).

### ASCII islands in mixed Qwen2 pretok (patch 0122)

**Why:** One non-ASCII byte (CJK / café / emoji) made the whole megaprompt miss `unicode_seg_is_ascii`, so Qwen2 paid the full Unicode pretok scanner on ASCII-majority text.

**What:**

- **0122** — ASCII gaps → `ascii_seg`; letter/punct non-ASCII islands → `unicode_seg`; punctuation keeps optional leading space (` ·`, ` 🚀`)
- **Measured (Qwen2):** mixed `mega_1mib` ~**19.5 ms** (~1.1× vs `NO_ASCII_PRETOK`); ascii ~8.5 ms; identity green (incl. ascii-islands + café/`hello ·世界` snippets)

Docs: [faster-bpe-tokenize.md](docs/faster-bpe-tokenize.md), [findings](docs/faster-bpe-tokenize-findings.md).

### Pretok blob for mixed path + ASCII bulk cpt decode (patch 0121)

**Why:** After 0119/0120, mixed Unicode megaprompts still built ~N `std::string` pretok words via `unicode_regex_split`, and the general path decoded UTF-8 one codepoint at a time even on ASCII-majority text.

**What:**

- **0121** — `try_blob` fills for the general cpt path; 8-wide ASCII cpt fill before `unicode_cpt_from_utf8`
- **Measured (Qwen2):** mixed `mega_1mib` ~**19.5 ms** (~1.1× vs `NO_PRETOK_BLOB`); ascii ~8.5 ms; identity green (incl. pretok-blob on mixed)

Docs: [faster-bpe-tokenize.md](docs/faster-bpe-tokenize.md), [findings](docs/faster-bpe-tokenize-findings.md).

### Pretok cache once per session (patch 0120)

**Why:** After 0107, `pretok_cache.init()` ran at the start of every `session.tokenize()` call. Chat/mixed megaprompts with specials are split into many fragments — each re-zeroed a 4096-slot table and started cold, making cache ON **~2.6× slower** than `NO_PRETOK_CACHE`.

**What:**

- **0120** — init pretok→ids cache once per BPE session (one `llama_tokenize`); fragments reuse the warmed table
- **Measured (Qwen2):** mixed `mega_1mib` ~**21 ms** (was ~91); chat ~**15 ms** (was ~94); ascii ~9 ms unchanged; identity green

Docs: [faster-bpe-tokenize.md](docs/faster-bpe-tokenize.md), [findings](docs/faster-bpe-tokenize-findings.md).

### Pretok blob — skip vector&lt;string&gt; on ASCII BPE (patch 0119)

**Why:** After 0118, ASCII custom pretok still built ~370k `std::string`/MiB for the BPE session to consume.

**What:**

- **0119** — `unicode_pretok_blob` (storage+lens, or view into text when all words printable); BPE walks `(ptr,len)`; `LLAMA_BPE_NO_PRETOK_BLOB=1` A/B
- **Measured (Qwen2, `mega_1mib_ascii`):** ~**8.7 ms** blob vs ~**9.5 ms** `NO_PRETOK_BLOB` (~1.09×); identity green across vocab harness

Docs: [faster-bpe-tokenize.md](docs/faster-bpe-tokenize.md), [findings](docs/faster-bpe-tokenize-findings.md).

### Specials trie + memchr skip (patch 0112)

**Why:** After 0111, each candidate first byte still memcmp'd every special sharing that byte (Qwen ~200 starting with `<`). Dense chat markers paid that tax repeatedly.

**What:**

- **0112** — load-time `naive_trie` over `cache_special_tokens`; LTR walk keeps longest eligible match; `memchr` when only one interesting first byte
- **Measured (Qwen2, dense `<|im_start|>` every ~150 B, 1 MiB):** ~**65 ms** trie vs ~**217 ms** `FORCE_LEGACY_SPECIALS` (~3.3×); identity green. English-between-markers identity-bench seed stays ~100 ms (BPE dominates over specials match).

Docs: [faster-bpe-tokenize.md](docs/faster-bpe-tokenize.md), [findings](docs/faster-bpe-tokenize-findings.md).

### Fuse ASCII pretok materialize + byte-encode (patch 0118)

**Why:** ASCII byte pretok still built N substrings then ran byte-encode (printable words copied twice).

**What:**

- **0118** — one-pass offsets → remapped words on the ASCII custom path; shared `unicode_byte_enc_table`
- **Measured (Qwen2 1 MiB ASCII):** ~**8.8 ms** vs ~**12.6 ms** `NO_BYTE_ENC_FAST` (~1.42×); identity green

Docs: [faster-bpe-tokenize.md](docs/faster-bpe-tokenize.md), [findings](docs/faster-bpe-tokenize-findings.md).

### GPT-2 byte-encode flatten + printable skip (patch 0117)

**Why:** After ASCII pretok (0114–0116), Qwen/GPT-2 still remapped every pretok byte through `unordered_map` + `string +=`. Printable `0x21..0x7E` is identity under GPT-2 bytes↔unicode.

**What:**

- **0117** — flat `enc[256]` LUT with `append(len)`; skip remap for printable-only words; `LLAMA_BPE_NO_BYTE_ENC_FAST=1` for A/B
- **Measured (1 MiB ASCII):** Qwen2/GPT-2 ~**1.33×** vs `NO_BYTE_ENC_FAST` (~9.5 ms vs ~12.8 ms); identity green. Stack ~**9–10 ms/MiB** (~105 MiB/s).

Docs: [faster-bpe-tokenize.md](docs/faster-bpe-tokenize.md), [findings](docs/faster-bpe-tokenize-findings.md).

### Qwen3.5 all-ASCII byte pretok (patch 0116)

**Why:** Qwen3.5 pretok adds `\p{M}` combining marks. No ASCII codepoint is `\p{M}`, so English megaprompts still paid full uint32 decode without needing a separate scanner.

**What:**

- **0116** — route Qwen3.5 all-ASCII (and ASCII segments) to the Qwen2 byte/ascii_seg scanners; hot flags LUT on the Unicode fallback
- **Measured (Qwen3.5 mega_1mib_ascii):** ~**11.8 ms** vs ~**14.6 ms** `NO_ASCII_PRETOK` (~1.24×); identity green

Docs: [faster-bpe-tokenize.md](docs/faster-bpe-tokenize.md), [findings](docs/faster-bpe-tokenize-findings.md).

### GPT-2 + Llama3 all-ASCII byte pretok (patch 0115)

**Why:** 0114 only covered Qwen2. GPT-2 / Llama3 English megaprompts paid the same uint32 decode tax.

**What:**

- **0115** — all-ASCII byte pretok for GPT-2 and Llama3 custom regexes (shared dispatch with 0114); `LLAMA_BPE_NO_ASCII_PRETOK=1` still disables
- **Measured (1 MiB ASCII):** GPT-2 ~**11.2 ms** (~1.34× vs `NO_ASCII_PRETOK`); Llama3 ~**12.2 ms** (~1.19×); Qwen2 unchanged ~1.26×; identity green

Docs: [faster-bpe-tokenize.md](docs/faster-bpe-tokenize.md), [findings](docs/faster-bpe-tokenize-findings.md).

### Qwen2 all-ASCII byte pretok (patch 0114)

**Why:** Even after 0110, pure-ASCII Qwen2 still decoded every byte to `uint32` (+ `cpt_byte_off`) before the LUT scanner — ~4× RAM and a full pass for nothing.

**What:**

- **0114** — if text is all `< 0x80` and the sole regex is Qwen2 custom, pretok on bytes and `substr` words; 8-wide letter consume; still `LLAMA_BPE_NO_ASCII_PRETOK=1` for A/B
- **Measured (Qwen2 mega_1mib_ascii):** ~**11.4 ms** vs ~**15 ms** with `NO_ASCII_PRETOK` (~1.3×); identity green. Identity-bench ~**87 MiB/s**.

Docs: [faster-bpe-tokenize.md](docs/faster-bpe-tokenize.md), [findings](docs/faster-bpe-tokenize-findings.md).

### Byte-indexed specials trie + load-time gates (patch 0113)

**Why:** 0112 still used `naive_trie` (`std::map` per byte) and rebuilt the first-byte interesting mask every `tokenize()`.

**What:**

- **0113** — `llm_specials_byte_trie` with `child[256]` indices; load-time `specials_interesting` / `_ud`; harness identity vs `FORCE_LEGACY_SPECIALS` on mixed/chat seeds
- **Measured:** dense-chat ≈**noise vs 0112** (~65 ms) — remaining time is BPE between markers. Ships for O(1) walk + per-call gate removal + harness coverage.

Docs: [faster-bpe-tokenize.md](docs/faster-bpe-tokenize.md), [findings](docs/faster-bpe-tokenize-findings.md).

### LTR special-token partition (patch 0111, gigatoken T47)

**Why:** After fixing the bench seed, chat megaprompts with repeated `<|im_start|>` / `<|im_end|>` were still **~500–850 ms/MiB** on Qwen. Cost was `tokenizer_st_partition`: O(|specials| × |fragments| × find) with ~294 specials — each absent special still scanned the whole text, and matches exploded the fragment list.

**What:**

- **0111** — first-byte-gated left-to-right longest match over `cache_special_tokens` (already longest-first); `LLAMA_BPE_FORCE_LEGACY_SPECIALS=1` keeps the old nested find path for identity A/B
- Bench adds `mega_1mib_chat` (ASCII + chat markers)
- **Measured (Qwen2, 1 MiB):** chat seed **~850 ms → ~99 ms (~8.5×)**; mixed Unicode seed similarly ~8×; pure ASCII ~14–19 ms (already fine)

Docs: [faster-bpe-tokenize.md](docs/faster-bpe-tokenize.md), [findings](docs/faster-bpe-tokenize-findings.md).

### ASCII pretok LUT + Qwen2 fast path (patch 0110)

**Why:** After 0109, we still thought Qwen “English” was pretok-scanner-bound at ~700 ms/MiB. That was a **measurement trap**: the identity-bench seed embeds CJK/emoji (byte-encode merge cost) and chat specials (`<|im_start|>` every ~150 B — special-token scan). Pure ASCII English is already ~18 ms/MiB.

**What:**

- **0110** — 128-entry ASCII flags/class LUT for `c < 128` in GPT-2/Llama3/Qwen2 scanners; dedicated Qwen2 all-ASCII segment loop; `LLAMA_BPE_NO_ASCII_PRETOK=1` disables the all-ASCII path for A/B
- Bench adds `mega_1mib_ascii` (no specials) so English pretok is not confused with specials/Unicode
- **Measured:** pure ASCII Qwen2 ~**18 ms/MiB** (~54 MiB/s); 0110 vs 0109 on that shape ~**1.0–1.07×**. Mixed Unicode / specials-heavy seeds stay hundreds of ms (different levers).

Docs: [faster-bpe-tokenize.md](docs/faster-bpe-tokenize.md), [findings](docs/faster-bpe-tokenize-findings.md).

### Faster pretok scanner hygiene (patch 0109)

**Why:** After 0108, Qwen English megaprompts were still ~650–700 ms and spent that time in the custom pretok scanner. Full SIMD remains deferred; three safe cleanups remove wasted work around it.

**What:**

- **0109** — `src/unicode.cpp`: lazy `text_collapsed` (only when `std::regex` fallback needs it — GPT-2/Llama/Qwen custom paths never did); ASCII letter/digit consume (gigatoken T01); materialize valid UTF-8 pretoks via original byte spans (FFFD path unchanged)
- Bit-identical vs 0108 token dumps + `FORCE_LEGACY` identity gate
- **Measured (Mac aarch64, mega_1mib interleaved vs 0108):** ≈ **noise / ~1.0–1.05×** — scanner still dominates Qwen English; ship for wasted-pass removal, not headline speedup

Docs: [faster-bpe-tokenize.md](docs/faster-bpe-tokenize.md), [findings](docs/faster-bpe-tokenize-findings.md).

### Faster pretok materialize (patch 0108)

**Why:** After 0106+0107, English Qwen megaprompts were still pretok-bound in `unicode_regex_split`. Full pretok SIMD is a separate identity project; safer wins were double UTF-8 decode, a noop roundtrip before GPT-2 byte remap, and per-codepoint temporary `std::string`s when rebuilding words.

**What:**

- **0108** — `src/unicode.cpp`: share one `cpts` vector with custom splitters; `unicode_cpt_append_utf8` (no temp string per cpt); skip identity `cpts_from_utf8`↔`cpt_to_utf8` in `unicode_byte_encoding_process`
- Bit-identical vs pristine unicode (A/B token dumps) and vs `FORCE_LEGACY` merge path
- **Measured (Mac aarch64, mega_1mib fast path, 0108 vs pristine unicode):** GPT-2 ~**1.9×**, Llama3 BPE ~**1.4×**, Gemma4 ~**1.2×**; Qwen2 English ≈ wash (regex scanner still dominates)

Docs: [faster-bpe-tokenize.md](docs/faster-bpe-tokenize.md), [findings](docs/faster-bpe-tokenize-findings.md).

### Per-tokenize pretok→ids cache (patch 0107, gigatoken T25/T26)

**Why:** After id-pair merge (0106), repeated short identifiers in agent/code megaprompts still re-ran BPE merge from scratch. Gigatoken’s pretok cache (T25/T26) memoizes `word bytes → ≤4 token ids`; miss path is unchanged so identity stays bit-identical vs `LLAMA_BPE_FORCE_LEGACY`.

**What:**

- **0107** — per-`llama_tokenize()` open-addressed cache; keys **4–15** bytes; values ≤4 ids; skip &lt;3 initial symbols (probe tax &gt; savings on spaces/`the`); `LLAMA_BPE_NO_PRETOK_CACHE=1` opt-out for A/B
- Apply after **0106**; sync vendor → `llama/llama.cpp/` (CGO)
- **Measured (Mac aarch64, 1 MiB):** English repeating seed ≈ wash vs 0106-only (pretok-bound); **code-like** repeated identifiers Qwen2 **~1.25×** vs 0106-only (~3.1× vs stock legacy). Gemma4 still ~2.8× from 0106 (cache neutral there).

Docs: [faster-bpe-tokenize.md](docs/faster-bpe-tokenize.md), [findings](docs/faster-bpe-tokenize-findings.md).

### Faster BPE tokenize — gigatoken lessons (M15d, patch 0106)

**Why:** Agent megaprompts re-run `llama_tokenize` every turn; stock BPE merge did `std::string` ×2 + string-pair hash lookups and showed up as hundreds of ms on `/v1/tokenize` before any forward. Rust gigatoken is the wrong product (bulk corpus, non-Darwin); its **id-pair table** and **tiered short merge** ideas still apply inside vendored `llama-vocab`.

**What:**

- **0106** — load-time `(id,id)→(rank,merged_id)` open-addressed table; id-based soft-delete merge; ≤64-symbol linear min-scan (T20); **piece4** packed UTF-8 LUT for char→id (avoids per-char `std::string` map — without it Qwen2 was a wash/regress); `LLAMA_BPE_FORCE_LEGACY=1` for same-binary identity A/B
- Trees: apply on `vendor/llama-cpp-*`, then `./scripts/vendor/sync_vendor_llama.sh` so **`llama/llama.cpp/` (CGO)** gets the patch — sibling-only apply leaves Mac ggml tokenize unpatched
- Bench/identity: `./scripts/bench/run_tokenize_bpe_identity_bench.sh` [`--bench`]
- Docs: [faster-bpe-tokenize.md](docs/faster-bpe-tokenize.md), [findings](docs/faster-bpe-tokenize-findings.md); ROADMAP **M15d**

**Measured (Mac aarch64, 1 MiB vocab-only, fair A/B):** Gemma4 ~**2.8×**, Llama3 BPE ~**1.27×**, Qwen2 ~**1.07×** (pretok still dominates Qwen). Do not cite earlier unreproducible 6–22× `/tmp` numbers.

### Batch chat wire format documented (Hermes, Jul 2026)

**Why:** Hermes deferred `/v1/chat/completions/batch` adoption because OpenAPI only said “OpenAI-shaped list / wrapper,” and the aux client assumed it could POST mixed models in one body. Without a stable response schema, client grouping work could not start.

**What:**

- OpenAPI: `ChatCompletionsBatchRequest` / `RequestItem` / `Response` / `BatchItem` — response is a **wrapper** `{object:"chat.completion.batch", model, count, completions[]}` with `completions[i]` ↔ `requests[i]` (not a bare list, not OpenAI async `/v1/batches`).
- Docs: [hermes-zerollama-gap.md](docs/hermes-zerollama-gap.md) §8 wire format + client notes; [hermes-gap-closure-findings.md](docs/hermes-gap-closure-findings.md) Finding 6 (group-by-model is **client** work).
- **Why same-model only:** one GGUF / one `generate_batch` runner; server-side mixed-model grouping would invent a second scheduler. Cap remains `min(8, llama_parallel_slots)`.
- **Why document before multi-model API:** honest same-model contract unblocks Hermes; mixed-model fan-out stays a non-goal until a client needs it after adopting this schema.

**Docs:** OpenAPI `server/openapi/openapi.yaml`; ROADMAP **M15e** (batch contract polish); README Hermes section.

### Mid-stream preempt signal + grammar (M15f, Jul 2026)

**Why:** Hermes still could not tell natural `done_reason=stop` from scheduler eviction mid-decode, and runtime-proxied GGUF silently dropped `response_format`/`format` (unconstrained JSON despite client schemas).

**What:**

1. **`done_reason: "preempted"` + `preempted_reason`** on the terminal NDJSON/SSE chunk (and OpenAI `finish_reason: "preempted"`).
2. **MLX soft mid-stream preempt** — when interactive admits and a lower-class session is inflight on the same model key, cancel that request context (`interactive_wait_inflight_lower` → victim sees `lower_wait_interactive`). **Not** ggml/Python hard kill (still defers on `refCount>0`).
3. **Python runtime format forwarding** — `format` / OpenAI `response_format` → llama-server `json_schema` / `grammar` on `/completion`.
4. **GBNF** via `format: {"type":"gbnf","grammar":"..."}` (native `/api/*`; `/v1` via `extra_body.format`).
5. **tools + grammar → HTTP 400** (`grammar is not supported together with tools`).

**Docs:** [hermes-gap-closure-findings.md](docs/hermes-gap-closure-findings.md); [hermes-zerollama-gap.md](docs/hermes-zerollama-gap.md); ROADMAP **M15f**; OpenAPI.

### Hermes gap closure (M15e, Jul 2026)

**Why:** Hermes wishlist mixed “already shipped under another name,” native-only `/api/*`, and six **real** product gaps (accept-and-drop `think`, no server timeout, no wait-abort reason, no can-load topology, no prefix-cache lease, no public batch). Closing them without inventing a second TP planner or mid-stream hard preemption.

**What:**

1. **`think` on `/v1`** — bind + map in `FromChatRequest` (precedence over aliases). **Why bind not allowlist:** passthrough alone silently dropped Hermes reasoning.
2. **Per-call `timeout`** — chat/generate/OpenAI + all runtime proxies; **504** on deadline (≠ **499** cancel). **Why:** client disconnect alone left stuck gens holding slots.
3. **`preempted_reason`** — QoS defer abort / busy bodies. **Why:** wait-abort signal, not mid-stream kill.
4. **`/api/can-load` topology** — `device_count` / `tensor_parallel` / `split_mode` / `tensor_split` / `main_gpu`. **Why report not plan:** surface runtime config; don’t invent Go TP.
5. **`POST/DELETE /api/cache/pin`** — prefix-cache lease (MLX trie + L3 TTL; not idle slot retention). **Why separate from `/api/pin`:** model residency ≠ `prompt_cache_key` residency.
6. **`POST /v1/chat/completions/batch`** — public batch via `generate_batch` (same-model, no tools/vision/think). **Why thin Go proxy:** decode batching already lives in Python. **Wire (documented Jul 2026):** `{object:"chat.completion.batch", model, count, completions[]}` with `completions[i]` ↔ `requests[i]`; clients group mixed models themselves. **Why document the wrapper:** underspecified “OpenAI-shaped list / wrapper” blocked Hermes aux-client work.

**Docs:** [hermes-zerollama-gap.md](docs/hermes-zerollama-gap.md); [hermes-gap-closure-findings.md](docs/hermes-gap-closure-findings.md); OpenAPI `server/openapi/openapi.yaml`; ROADMAP **M15e**.

### Bee B1 + llama-server unify (0100–0103) — Jul 2026

**Why:** One vendor `llama-server` for chat + Kokoro TTS; adaptive DFlash draft-max (Bee profit) behind opt-in.

**Shipped:**
- **0100** — `GGML_CUDA_FORCE_CUBLAS` getenv (5080 serve contract)
- **0101** / **0103** — `tools/kokoro` + optional `LLAMA_BUILD_KOKORO` `/v1/audio/speech` (replaces dual TTS binary)
- **0102** — Bee B1 `--spec-dm-adaptive profit` (default off); Go `ZEROLLAMA_SPEC_DM_ADAPTIVE`
- Build: kokoro tree allowed; `LLAMA_BUILD_KOKORO=ON` opt-in; OmniVoice still refused
- Patch **0099** is native DCA (see Dual Chunk Attention below); darwin/Metal occupy **0094–0098**

Doc: [llama-server-unify.md](docs/llama-server-unify.md), [llama-fork-watchlist.md](docs/llama-fork-watchlist.md).

### SGLang multimodal ports (#31417 / #31438 / #31832) — Jul 2026

**Why:** Weekly SGLang scan (`4a76699dfc..4e5a05148a`) flagged portable agent-facing MM patterns; CUDA-IPC / breakable CG / HiCache infra skipped.

**Shipped:**
- `ClientMediaError` / `ServerMediaError` — corrupt/unfetchable media → **400**; missing ffmpeg / host IO → **500** (`MediaHTTPStatus` on chat expand)
- `OLLAMA_MM_IO_WORKERS` (default 4) — parallel multi-clip ffmpeg expand with ordered spans
- `ExpandAudioClipsInChatRequest` — demux WebM/MP4/AVI `input_audio` containers to WAV via ffmpeg
- OpenAI / options `session_id` → `prompt_cache_key` alias (SGLang #29436)

Doc: [sglang-multimodal-borrowings.md](docs/sglang-multimodal-borrowings.md).

### vLLM L3 pattern ports (#48123 / #48596 / #48535 / #48911) — Jul 2026

**Why:** Weekly vLLM scan flagged high-ROI KV/cache patterns that map cleanly onto LMCache + prefix block pool without OffloadingConnector.

**Shipped:**
- Per-request `options.zerollama.kv_load_tiers` → `TierFilter` (skip LMCache/blob secondary lookup; host disk restore gated)
- Defer blob finalize until slot `.bin` exists; `finalize_slot_blob` + reuse-race flush + post-disk-save finalize
- `cache_creation_tokens` / OpenAI `created_cache_tokens` / Anthropic `cache_creation_input_tokens` (`creation = newly_cached − hit_at_admit`)
- SWA reachable-tail store mask on `register_prefix`

Doc: [vllm-borrowings.md](docs/vllm-borrowings.md), [upstream-siblings.md](docs/upstream-siblings.md).

### Mac Metal Lab D1b fused QJL+Polar attn (0098) — Jul 2026

**Why:** After **0097** SET_ROWS, `qjl1_256/q4_polar` still aborted in CPU fused attn (Metal shaders existed but were not wired; embed omitted them).

**Shipped:** Patch `0098` — Metal `supports_op` + encode for `GGML_OP_FUSED_ATTN_QJL_TBQ` (Polar V), `kernel_qjl_project_q_f32`, embed append of fused polar. Smoke: pp512 ~961, tg128 ~37 on Llama-3.2-3B (still ~0.24× stock f16).

### Mac Metal Lab D1 Polar + QJL SET_ROWS (0097) — Jul 2026

**Why:** `FORK_PROFILE=speed` aborted on Metal because `SET_ROWS` lacked `q4_polar` encode, and the embed path never included eliza-shipped QJL SET_ROWS.

**Shipped:** Patch `0097` — `quantize_q4_polar` + SET_ROWS templates in `ggml-metal.metal`, device allowlist, QJL host-name fix, embed append of QJL encode. Smoke: `f16/q4_polar` PASS (tg ~33 on 3B). Full speed profile completed by **0098**.
### Vendor rebase to ggml-org b10159 (`f95de977`) — Jul 2026

**Why:** Nanbeige4.2 needs `LLM_ARCH_NANBEIGE` (upstream #25994). Pin `86d86ed4` could not load those GGUFs; Q8/bf16 need a current llama-server.

**Shipped:**
- `LLAMA_CPP_VERSION=f95de977`, `LLAMA_CPP_COMMIT=f95de9776b5b…`, `Makefile.sync` → `vendor/llama-cpp-f95de977` (`BUILD_NUMBER=10159` / tag **b10159**)
- Rebased **103** Ollama/zerollama patches onto the new base (verify: `make -f Makefile.sync clean apply-patches`); skipped absorbed Q6_K get_rows + NVFP4 #25730; resolved Metal FA↔FWHT, nanbeige+dflash arch, FP8/TBQ, Bee budget API
- `serve.sh` / `5080_env.sh` / `serve_gpu_example.sh` resolve vendor pin from `Makefile.sync`

### Native Dual Chunk Attention in llama.cpp (Qwen2/2.5) — Jul 2026

**Why:** Qwen 1M / official long-ctx needs DCA; YaRN alone is not the algorithm. Product path is patched ggml CUDA, not a separate engine.

**Shipped:** Vendor pin + patches **0099** (DCA series on b10159; after darwin/Metal 0094–0098) — load `*.attention.dca.*`, DualChunk RoPE + `s(L)`, FA LSE export, Qwen2 graph `build_attn_dca` (3× FA + masks + LSE merge; CUDA graphs off). Prefill uses the same three stage masks. Serve via stock `llama-server` / zerollama-runtime. Oracle: `scripts/dca_oracle_logits.py` (n=0 ≈ stock FA; n≥1 ≈ SGLang dense); helper `scripts/dca_unit_ref.py`. Doc [dca-dual-chunk-attention.md](docs/dca-dual-chunk-attention.md).

**5080 validation:** n=0 PASS on stamped Qwen2.5-3B (`chunk_len=192`, max \|Δlogprob\| ≈ 0.05 vs stock FA). Graph LSE path uses GPU-backed buffer + FA→consumer barrier. n≥1 vs SGLang dense still needs a local HF Instruct-1M tree.

### Dual Chunk Attention SGLang sidecar (lab / legacy) — Jul 2026

**Why:** Earlier interim path; retained for oracle / experiments. **Not** the Qwen long-ctx product path.

**Shipped:** `/v1/chat/completions` proxies when `modality_backends.inference=sglang` + `OLLAMA_SGLANG_URL`. Launch helper `scripts/serve/sglang_dca_example.sh`; patch **0094** GGUF `*.attention.dca.*` key names.

### 5080 serve pins stock q8_0 KV; CUDA 12.9 + 32k/65k depth — Jul 2026

**Why:** L1 `rtx-5080.json` already had `q8_0`, but production shells could still inherit `ZEROLLAMA_LLAMA_FORK=1` (QJL/TBQ) and skip the stock path. Long-ctx A/B needed confirmation past 16k.

**Shipped:** `serve_gpu_example.sh` defaults `ZEROLLAMA_GPU_PROFILE=1` + `ZEROLLAMA_LLAMA_FORK=0`; `single_gpu.yaml` / runbook note the Jul 2026 bench. Host: CUDA **12.9** nvcc + cublas installed. Depth on Llama-3.1-8B: q8_0 ≈ f16 at 32k/65k (**85/59** vs **88/58** tg) — keep stock q8_0.

### 5080 KV alpha A/B + llama-bench fork types (0093) — Jul 2026

**Why:** Lab RotorQuant/planar/iso and vendor TBQ/QJL needed apples-to-apples `llama-bench -ctk`, but `llama-bench` hardcodes type names separately from `common/arg.cpp` and rejected Eliza L2 names.

**Shipped:** Patch `0093` — accept `tbq3_0` / `tbq4_0` / `tbq3_k` / `tbq4_k` / `qjl1_256` / `q4_polar` / `tbq3_tcq` / `e8_2` in `tools/llama-bench`. Watchlist updated with CT 1564 results: **no-merge** planar/iso; stock **q8_0** confirmed fastest on Llama-3.1-8B; TBQ remains VRAM opt-in.

### Mac Metal Lab A RotorQuant A/B — Jul 2026

**Why:** 5080 no-merge was CUDA-only; Apple Silicon needed its own quiet-GPU measurement before closing planar/iso.

**Measured (M4 Max, Llama-3.2-3B Q4_K_M, `tmp/metal-ab/v2/`):** stock **f16** best (tg ~151); q8_0 ~0.91×; planar/iso **no-merge** (pp2048 collapse); TBQ/turbo3 VRAM-only. QJL/Polar later unblocked by **0097–0098** (still tok/s FAIL merge).

**Docs:** `docs/llama-fork-watchlist.md` (Metal table), `tmp/metal-ab/RESULTS.md`.


### Session → cache great loop (M15c) — Jul 2026

**Why:** Agent harnesses on one connection need to declare session/cache intent (`parent` / `reset` / `level`), have the server schedule and retain on that intent, then hit MLX/L3/Radix when safe — without `cold:` key prefixes or client TTL floors that fight `keep_alive`.

**Shipped (contract + enforcement):**

| Surface | What | Why |
|---------|------|-----|
| `options.zerollama.cache_reset` | Force miss under the **same** `prompt_cache_key` this turn | Soft “new branch” without a second key namespace |
| `options.zerollama.cache_level` | `auto`\|`gpu`\|`dram`\|`disk` (default auto) | Retention hint; auto = heuristics (no surprise) |
| `session_parent` / `session_group` | Multiplex **key hot-map** `wait_parent`; Radix prefer on equal-length ties | Many agents on one runner; donors stay hash-verified |
| Gate primary | Derived from keyHot on begin/end | Stops primary `inflight` leak when concurrent begins overwrite the slot |
| Parent match | Expand raw parent via `injectMLXSessionKey` candidates | Children send raw ids; MLX may have rewritten aux/bg branches |
| GGUF reset | Deny resume **and** bump decode-graph + `seed_seq_pos(0)` / `seq_rm`; **skip Radix** | Soft deny alone left residual warm state; Radix seed undid reset |
| Version / OpenAPI / ps | Caps + `ProcessSessionInfo` parent/level/fulfillment | Progressive probe + fleet visibility |

**Learnings (audit):** primary-slot match-end leaks; soft GGUF reset is not invalidate; Radix-after-deny undoes reset; parent must resolve to gate keys; `gpu`≈`dram` is honest until spill exists. Detail: [agent-qos-and-project-tracking.md](./docs/agent-qos-and-project-tracking.md#findings--learnings-session--cache).

Doc: [agent-qos-and-project-tracking.md](./docs/agent-qos-and-project-tracking.md), [mlx-agent-prompts.md](./docs/mlx-agent-prompts.md), [radix-prefix-share.md](./docs/radix-prefix-share.md).

### Host wishlist Phase B — pin, propose, thrash dampen (Jul 2026)

**Why:** Phase A gave dry-run admit but Decide still lacked session pins, multi-model plans, and less ggml↔runtime unload thrash — without lying that Python can hold two GGUFs.

**Why an audit pass after first ship:** the first Phase B cut wired pin into `findRunnerToUnload` only. The runtime VRAM broker still called `UnloadAllRunners`, so pins were wiped every chat turn on runtime-default hosts — clients trusted a lease that did nothing. B0 also skipped unload on GGUF match alone while leftover ggml could still hold VRAM (OOM). Per-lease pin caps allowed N leases to stack past `MAX_LOADED`. Soft-pin of runtime GGUFs was missing, so Python could still swap under a “pinned” model.

**Shipped:**

| Surface | What | Why |
|---------|------|-----|
| Broker B0 | Skip `UnloadAllRunners` when request GGUF matches `/health.model_swap.loaded_gguf` **and** ggml scheduler is empty | Same-GGUF chat should not thrash; leftover ggml must still be cleared |
| Sched B1 | `ZEROLLAMA_WARM_HYSTERESIS` (default 3m) prefers keep recently used ggml victims | Reduce ping-pong eviction among warm ggml runners |
| `POST /api/pin` + `DELETE /api/pin/:id` | TTL eviction leases; fail closed on 2+ distinct runtime GGUFs (per request **and** cross-lease); **global distinct-key budget**; stores `RuntimeGGUFs`; status `inference.pins` | Session residency without loading; honest single-resident Python |
| Pin vs broker | `UnloadAllRunners` / `UnloadOtherRunners` keep pin+fulfillment keys; training / exclusive bench use `UnloadAllRunnersForced` | Soft pin must survive the broker; training/bench must reclaim GPU |
| Pin vs runtime | Residual pinned/in-use ggml → 503 (`runtime_pin_ggml`) **before** `ResumeInference`; other GGUF while runtime pin active → 503 (`runtime_pin_gguf`) | Fail closed beats dual-stack OOM or silent GGUF swap |
| `POST /api/propose-load` | Batch can-load + `co_resident` / `serialize_required` plan | Decide needs multi-model honesty without a calibrator |
| Caps | `pin_reserve` + `propose_sidecar` true; **`stable_multi_model_swap` false** | Advertise what works; do not claim multi-GGUF swap |
| Single-serve | Stronger occupied-bind error before Listen | Orphan loopback `:8080` stole production APIs |

**Learnings (audit):** pin without broker respect is worse than no pin; B0 needs ggml-empty; global pin budget; 503 before resume when pins block clear; Go must soft-pin runtime GGUF paths. Detail: [inference-wishlist-host.md](./docs/inference-wishlist-host.md#findings--learnings-phase-a--b).

Doc: [inference-wishlist-host.md](./docs/inference-wishlist-host.md).

### Host wishlist Phase A — capacity & admission APIs (Jul 2026)

**Why:** Orient Inventory / Decide / hire-map clients could detect `distribution=zerollama` but still guessed capacity from env folklore. They had no public dry-run admit, no Prometheus scrape, no Retry-After on busy 503s, and empty generations looked like semantic refusals. Stable multi-model swap / pin / propose stay deferred (scheduler redesign).

**Shipped:**

| Surface | What |
|---------|------|
| `GET /api/status` → `inference.config` | `NUM_PARALLEL`, effective/configured `MAX_LOADED_MODELS`, `MAX_QUEUE`, keep-alive, residency owner, slots-vs-models |
| `POST /api/can-load` | Dry-run; never `GetRunner`. Runtime `confidence=exact` via `ProbeVramEstimate`; ggml `heuristic`. Fail closed if estimate/GGUF missing. `already_loaded` requires GGUF path match |
| `GET /api/metrics` | Hand-rolled Prometheus text (queue gauges + result counters); runtime proxy paths instrumented |
| Busy `503` | `Retry-After: 2` + JSON `retry_after` (ggml queue, Metal contention, runtime admit) |
| Errors / empty | Partial timings on error JSON; `cause=host_unstable`; `done_reason=empty_generation` |
| `GET /api/version` | Progressive-probe caps (`can_load`, `metrics`, …) + honest `false` for swap/pin/propose |
| OpenAPI | `/api/status`, `/api/can-load`, `/api/metrics`, `InferenceError`, 503 docs |

**Learnings (audit fixes):** do not set `already_loaded` from “any llama loaded”; fail closed when VRAM estimate is unavailable; keep `host_unstable` matchers tight; instrument runtime proxy metrics.

Doc: [inference-wishlist-host.md](./docs/inference-wishlist-host.md).

### Fulfillment QoS — complete vs benchmark (Jul 2026)

**Why:** Agent QoS could defer background work but had no request-scoped way to lock in a model for a clean bench or guarantee a critical turn finishes without eviction / peer VRAM pressure (SQL-transaction-like begin→release).

**Shipped:** `options.zerollama.fulfillment`:
- **`complete`** (`guarantee`, `reliable`) — interactive elevation, wait for clear slot, protect model from eviction, 30m keep-alive floor when unset; other interactive on other models allowed
- **`benchmark`** (`bench`, `speed`, `exclusive`) — exclusive GPU: wait for idle, unload peer runners, block all other traffic until release, 2h keep-alive floor when unset

Injects `prompt_cache_key` when omitted (`fulfill:{mode}:{project_id}`). Advertised on `GET /api/version` (`capabilities.fulfillment`). Docs: [agent-qos-and-project-tracking.md](./docs/agent-qos-and-project-tracking.md).

### Fix Metal bootstrap discovery abort (total_vram=0) — Jul 2026

**Why:** `build_zerollama_mac.sh` embeds compiled `default.metallib` bytes in `ggml-metal-embed.metal`, but `ggml_metal_library_init` still treated the embed as UTF-8 source for `newLibraryWithSource`. MTLB magic fails UTF-8 → nil NSString → Metal assert abort → ollama-engine `/info` dies → serve logs `library=cpu` / `total_vram="0 B"` / `default_num_ctx=4096`.

- **Patch 0104 / `ggml-metal-device.m`** — detect `MTLB` magic and load via `newLibraryWithData`; keep source JIT path for text embeds; refuse nil source instead of aborting.
- **Mac build guard** — fail the build if vendor sync drops the compiled-metallib loader, instead of shipping a binary that silently discovers CPU only.

**Verify:** `./zerollama runner --ollama-engine --port 65432` then `curl -s localhost:65432/info` → `Apple M4 Max` with non-zero `total_memory`.

### Phase L3 — media-aware llama-server seq-copy (Jul 2026)

**Why:** `SLOT_SEQ_COPY` used `check_no_mtmd`, which rejected **all** multimodal models even for pure-text prefixes. Per-token `push_back` also threw on `LLAMA_TOKEN_NULL` and dropped `map_idx_to_media`, so media KV could never be cloned safely.

**Shipped:** patch **0090** — `server_tokens::{has_media,first_media_index,safe_prefix_len}`; SEQ_COPY uses `clone()+keep_first` (media-safe); body `allow_media` (default true). `allow_media=false` clamps to text before first media. Python: `ZEROLLAMA_RADIX_MEDIA_SEQ_COPY` (default on) → POST `allow_media`. Rebuild vendor `llama-server` to pick up.

### Phase MM — KV pad_value radix (Jul 2026)

**Why:** Vision soft/pad token IDs are identical across images, so token-id prefix keys cannot tell clips apart (SGLang stamps `pad_value = 1e6 + hash%2^30` on every vision pad). Zerollama already had `MultimodalHash` on the first slot; trailing PostTokenize pads still shared Token=image_pad with hash 0.

**Shipped:** `model/mmradix` — `PadValueFromHash` / `ApplyToInputs` / `ClampForEmbed`. ollama-engine applies after multimodal build (`OLLAMA_KV_MM_PAD_RADIX`, default **on**); stamps Token + MultimodalHash across the SameBatch vision span; clamps to vocab before TokenEmbedding. Log: `kv mm pad_value radix applied`. Kill-switch: `OLLAMA_KV_MM_PAD_RADIX=0`.

### Phase MM — ViT radix cross-request embed pool (Jul 2026)

**Why:** Session overlay pins one `prompt_cache_key`; the small global slot LRU thrashed under fleet load so a second agent with the same clip re-ran ViT. SGLang’s `MultiModalStaticCache` is a server-wide byte-budget pool keyed by content hash.

**Shipped:** `OLLAMA_VIT_RADIX` (default **on**) enables a content-addressed pool with **100 MiB** byte budget (`OLLAMA_IMAGE_EMBED_CACHE_BYTES` overrides). Pool **grows beyond** `OLLAMA_IMAGE_EMBED_CACHE_MAX` under budget (hard cap 4096 slots). Hit log: `vision embed radix cache hit` (plus legacy `global` for greps). ollama-engine + ggml llamarunner. Smoke: `VIDEO_AGENT_INFER_VIT_RADIX=1`. Kill-switch: `OLLAMA_VIT_RADIX=0`.

### Vendor — fix duplicate TBQ vec_dot (0088) (Jul 2026)

**Why:** Patches 0071+0073 both added `ggml_vec_dot_tbq{3,4}_0_f32`, breaking Metal/CPU `llama-server` builds.

**Shipped:** `llama/patches/0088-ggml-cpu-drop-duplicate-TBQ3-TBQ4-vec_dot-defs-0071-.patch`

### Vendor — Bee reasoning-loop guard Lab B0 (0087) (Jul 2026)

**Why:** Qwen think models can loop in hidden reasoning; BeeLlama’s detector is the smallest useful server UX port.

**Shipped:**

- `llama/patches/0087-server-Bee-inspired-reasoning-loop-guard-Lab-B0.patch` on pin `86d86ed4`
- Opt-in `--reasoning-loop-guard {off,force-close,stop}` (default **off**) + request JSON fields
- Wired via `process_token` + `reasoning_budget_tracking` (no Bee accept-callbacks on our pin)
- Native `/completion` JSON: `stop_detail` + `loop_guard.{triggered,action,reason}` when triggered
- Bonus: `SLOT_SEQ_COPY` uses `check_slot_no_media` (fixes stale `check_no_mtmd`)
- Lab A: shallow sibling `../llama-cpp-rotorquant` @ `feature/planarquant-kv-cache` (CUDA A/B on 5080 host)

### Phase MM — skip-ViT precomputed infer smoke (Jul 2026)

**Why:** Linux auto→ggml unlocked `precomputed_embedding`, but infer smoke only grepped inject lines as advisory and never POSTed feature rows.

**Shipped:** `VIDEO_AGENT_INFER_PRECOMPUTED=1` (+ `VIDEO_AGENT_GO_LOG`) posts synthetic Qwen vision-block `padded_input_ids` + `precomputed_embedding` (embd from `/api/show`); strict fail without `precomputed_embedding runner inject`; gate report enforces the flag.

### Phase L3 — HiCache host/storage token tiers (Jul 2026)

**Why:** `sglext.cached_tokens_details` and access-log fields existed but always stayed at device-only (`cache_n`). Operators could not see disk-slot vs federated blob restores.

**Shipped:**

- Runtime stamps `cached_tokens_host` on in-process disk restore; `cached_tokens_storage` (+ backend scheme) on L3 blob restore
- `llm.CompletionResponse` + `/api/chat`/`generate` metrics + access log populate the tiers

### Phase MM — forward ffmpeg `grid_thw` estimates to M-RoPE ViT (Jul 2026)

**Why:** After `mtmd_bitmap_set_grid_hint` shipped, keeping server estimates preflight-only left native video on a second smart_resize path. Forwarding the Qwen-style estimate forces ViT to the same `[H,W]` as preflight/usage.

**Shipped:** `GridTHWPerRaster` forwards any valid span `GridTHW` (client or ffmpeg); `GridTHWExplicit` stays observability-only.

### Phase 17 — Linux auto vision stays on ggml (Jul 2026)

**Why:** Linux `ZEROLLAMA_LLAMA_SERVER=auto` previously routed mmproj GGUF through llama-server, which rejects `precomputed_embedding` / `processor_output` (base64 rasters only). SGLang skip-ViT clients hit 400 on the Linux default path even though llamarunner already splices embed rows.

**Shipped:** Linux auto = plain-text → llama-server; vision (mmproj) → ggml llamarunner (same as Mac). Explicit `ZEROLLAMA_LLAMA_SERVER=1` still forces vision through llama-server for Phase 17 vision smoke.

### Docs/lab — llama.cpp fork watchlist + RotorQuant A/B (Jul 2026)

**Why:** GitHub fork scout found RotorQuant/IsoQuant and BeeLlama server controls beyond ik/anemll; measure before pin patches.

**Shipped:**

- [docs/llama-fork-watchlist.md](docs/llama-fork-watchlist.md) — Lab A RotorQuant, Lab B Bee (B0 **0087** + B1 **0102** landed), Lab C TQ3 FP4
- `./scripts/phase/l2_rotorquant_ab.sh` — multi-leg decode/VRAM A/B on lab port `:18082`

### Phase MM — real M-RoPE `grid_thw` forward (Jul 2026)

**Why:** Docs claimed `mtmd_bitmap_set_grid_hint` shipped while Go still no-op'd; same PNG + client grid drifted from ViT embed count → padded inject misalign.

**Shipped:**

- **`mtmd_bitmap_set_grid_hint`** + dyn_size honor (`W*patch × H*patch`, log `grid_thw hint resize`)
- **ollama-engine** Qwen3-VL / Qwen2.5-VL / glmocr / qwen3next `EncodeMultimodalWithGrid`
- ViT embed cache hash includes grid when set; Go bind in `llama/llama.go`

**Non-goals:** non–M-RoPE / llava-uhd / fixed-size families.

### Vendor — hardware PR ports 0080–0086 (Jul 2026)

**Why:** Bring open ggml-org/llama.cpp PRs that matter on M4 Max Metal, RTX 5080 Blackwell, and Ada (dual-4090) onto pin `86d86ed4`.

**Shipped (format-patches):**

- **0080** `#25648` — Metal F16 `mul_mat` null-pipeline crash
- **0081** `#24565` — CUDA Blackwell fattn MMA config
- **0082** `#25707` — CUDA Q2_0
- **0083** `#25788` — Metal `gated_delta_net` cache fusion
- **0084** `#22587` — CUDA GDN row-per-warp
- **0085** `#25730` — NVFP4 W4A4 activation quant
- **0086** `#25750` — Metal FA-vec per-device (Q,NE) tuning (monolithic metal port; GQA2 kept)

**Skipped:** `#20831` (RDNA/AMD MMVQ). Drafts/lab `#25635` `#23440` `#14570` still deferred.

**Vendor HEAD:** `5f7b7879` (`LLAMA_CPP_VENDOR_HEAD`). Rebuild `llama-server` / Mac CGO to pick up kernels.

### Build — dedupe TBQ CPU vec_dot + donor stub (Jul 2026)

**Why:** Vendor rebase left duplicate `ggml_vec_dot_tbq3_0_f32` / `tbq4_0_f32` in `ggml-cpu/quants.c`, and the non-`LLAMA_KV_EXT_DONOR_BUFFER` path omitted `llama_kv_ext_donor_try_consume` — both broke CGO `go test` link.

**Shipped:** remove the second TBQ pair; add nullptr stub for `llama_kv_ext_donor_try_consume` when donor buffer is off.

### Phase MM — deepseekocr processor_output (Jul 2026)

**Why:** Last deferred ollama-engine `processor_output` family — SGLang OCR clients already send HF pixels; falling back to PNG re-ran ProcessImage (aspect tiling + normalize) before SAM+CLIP.

**Shipped:**

- **`processor_output`** — `image_grid_thw [1,rows,cols]` (2–9 tiles); `pixel_values` = row-major `640²×3` locals + `1024²×3` global
- Shared **`multimodalFromPatches`** with PNG encode

**Non-goals:** llamarunner/mtmd processor_output; non–M-RoPE `grid_thw`.

### Phase MM — llamarunner ViT byte budget (Jul 2026)

**Why:** Close SGLang #28441 on ggml llamarunner — ollama-engine already had `OLLAMA_IMAGE_EMBED_CACHE_BYTES`; slot-only LRU still thrashed mixed tiny embeds vs video frames on the mtmd path.

**Shipped:**

- Shared float32-byte budget across PNG `MtmdChunk` + precomputed global LRUs (`0` = slot-only)
- Cross-pool oldest eviction under pressure; session overlay hash caps unchanged

**Non-goals:** Mooncake page arena; cross-runner radix ViT share.

### Phase MM — llama4 multi-tile processor_output (Jul 2026)

**Why:** Multi-tile Llama4 clients already pack a local canvas + global tile; rejecting that forced PNG re-encode on ollama-engine.

**Shipped:**

- **`processor_output`** — `image_grid_thw [1,H,W]` local canvas (divisible by `image_size`); multi-tile requires global `image_size²` floats appended (EncodeMultimodal packing)
- Shared **`multimodalFromPixels`** path with PNG encode (tile separators + global chunk unchanged)

**Non-goals:** llamarunner processor_output.

### Phase MM — LFM2 multi-tile preprocessed ingest (Jul 2026)

**Why:** SGLang high-res LFM2 clients already run the HF processor / ViT; single-tile-only ingest forced PNG decode + server-side split.

**Shipped:**

- **`processor_output`** — `image_grid_thw [1,rows,cols]` packs row-major `tile_size²` tiles (+ optional thumbnail remainder); single-canvas `[1,H,W]` pixels unchanged
- **`precomputed_embedding`** — same tile grid; equal-sized chunks (+ optional equal-sized thumbnail when rows divisible by tiles+1)
- **`PostTokenize`** — row/col + thumbnail markers via existing multi-tile layout

**Non-goals:** llamarunner processor_output.

### Vendor rebase to ggml-org master `86d86ed4` (Jul 2026)

**Why:** Pull latest [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) tip and rebase zerollama patches.

- **Pin:** `LLAMA_CPP_VERSION=86d86ed4`, `LLAMA_CPP_COMMIT=86d86ed4396b…`, `Makefile.sync` → `vendor/llama-cpp-86d86ed4` (`BUILD_NUMBER=10065`, past tag `b10064`)
- **Patches:** **79** format-patches (was 80 on `8f114a9b`); dropped merged upstream Metal ports **snake #25459** and **Q2_0 #25419**
- **Ports:** CUDA GPU discovery / NVML, fused QJL SET_ROWS, Metal 64×8 mul_mm, FP8 E4M3/E5M2 MMQ load-tiles for the post-rewrite `mmq.cuh`
- **Backup:** `llama/patches.pre-8f114a9b-20260717/`
- **Verify:** clean `make -f Makefile.sync clean apply-patches`; `llama_patch_health` **pass**

**Rebuild:** `./scripts/build/build_llama_server.sh` (and Mac CGO via `./scripts/build/build_zerollama_mac.sh` when needed). Do not auto-start on production `:11434`.

### Phase MM — capacity-aware ViT embed cache (Jul 2026)

**Why:** SGLang #28441 moved multimodal caches to capacity-aware pools. Slot-only LRU treats tiny embeds like 32-frame video; unbounded per-session overlay pins can grow without bound under agent churn.

**Shipped:**

- **`OLLAMA_IMAGE_EMBED_CACHE_BYTES`** — optional float32-byte budget on ollama-engine global ViT LRU (0 = slot-only)
- **Per-session hash cap** — overlay hashes LRU-capped at `OLLAMA_IMAGE_EMBED_CACHE_MAX` (ollamarunner + llamarunner PNG/precomputed)
- **Shared session refs** — session store pins the same embed value as global (clone on restore/return only)

**Non-goals:** Mooncake page arena; cross-runner radix ViT share.

### F6 — Operator + agent playbooks (Jul 2026)

**Why:** F1–F5 / F7 shipped the control plane; operators and agents still needed one place for sticky shards, warm-only SLA, and cancel-while-queued rules (and explicit non-goals).

**Shipped:** [docs/fleet-playbooks.md](./docs/fleet-playbooks.md) — sticky model shards, `warm_only` / `prefer_warm`, cancel policy (`queued` yes / `loading` no), F5 token usage, anti-patterns (scatter-gather, 60s quotes).

**Non-goals:** new fleet APIs; remote load from the manager.

### F5 — Short-TTL assignment tokens (Jul 2026)

**Why:** Warm-queue races between assign and first chat chunk; long quotes waste GPU.

**Shipped:**

- HMAC `assignment_token` / `expires_at` / `expires_in` on `POST /api/fleet/assign` when `ZEROLLAMA_FLEET_ASSIGN_SECRET` is set
- Node `POST /api/fleet/assign-hold` registers soft hold; `X-Zerollama-Assignment-Token` on `/api/chat` + `/api/generate` consumes it
- `/api/status.inference.ggml.assign_holds` (+ folded into `pending`) so fleet scoring sees pressure
- TTL default **8s** (2–30s clamp); optional push kill-switch `ZEROLLAMA_FLEET_ASSIGN_PUSH=0`

**Non-goals:** multi-minute reservations; remote load (sticky shards / cancel policy → [F6 playbooks](./docs/fleet-playbooks.md)).

### L3-R11 — Go prefix hashes + auto blob peers (Jul 2026)

**Why:** L3-R9 needs agents to send `prefix_block_hashes` without reimplementing the Python SHA-256 chain; L3-R10 peers were easy to leave unset on nodes that already have `ZEROLLAMA_FLEET_PEERS`.

**Shipped:**

- Go package `prefixblock` — `ModelScope` / `Hash` / `Iter` / `Hashes` golden-matched to `runtime/kv/prefix_block_hash.py`
- Blob peer resolution: `ZEROLLAMA_LMCACHE_BLOB_PEERS` → Go coordination `lmcache_blob_peers` → `ZEROLLAMA_FLEET_PEERS`
- Go coordination push of fleet/blob peers to Python
- `blob_digests` on `/health` + `/api/status.inference.runtime.radix`
- Fleet hash score prefers peers with `blob_digest_blocks` / `blob_digests` (reason `radix_blob`)

### L3-R10 — HTTP peer blob pull (Jul 2026)

**Why:** L3-R7 federated digests still required a shared filesystem (`ZEROLLAMA_LMCACHE_BLOB_ROOT` / NFS). Cold fleet nodes without that mount full-prefilled.

**Shipped:**

- Runtime `GET /kv/blob/{digest}` — raw slot blob octets
- Go proxy `GET /api/kv/blob/{digest}` → runtime (LAN peer reachable)
- `materialize_blob` local miss → `ZEROLLAMA_LMCACHE_BLOB_PEERS` pull (`/api/kv/blob` then `/kv/blob`)
- Optional `ZEROLLAMA_LMCACHE_BLOB_HTTP_TOKEN`; kill-switch `ZEROLLAMA_LMCACHE_BLOB_HTTP=0`
- Health: `prefix_block_pool.lmcache_blobs.http`

**Non-goals:** NIXL RDMA / Mooncake page arena (upgrade transport later); authenticated mTLS mesh.

### L3-R9 — Fleet content-hash routing / LA13 (Jul 2026)

**Why:** L3-R8 residency score only saw `entry_count`; agents sharing a system prompt need the peer that holds *those* prefix blocks.

**Shipped:**

- Python `/health.kv_resume.prefix_block_pool.block_hashes` — newest-first capped sample (`ZEROLLAMA_RADIX_HEALTH_HASH_CAP`, default 128)
- `api.RadixMirrorStatus.BlockHashes` on `GET /api/status` → fleet poll
- `prefix_block_hashes` on `POST /api/fleet/assign` and `/internal/score`
- Longest leading-hash match soft score (`ZEROLLAMA_FLEET_RADIX_HASH_SCORE`, default on); reason `radix_hash`

**Non-goals:** Go-side hash computation from raw prompts; NIXL RDMA; guaranteed restore (still soft routing).

### L3-R8 — Go Radix mirror + fleet soft score (Jul 2026)

**Why:** Fleet assign only had session-key affinity; operators/fleet could not see Python prefix-block / Radix health without curling the sidecar.

**Shipped:**

- **`api.RadixMirrorStatus`** on `GET /api/status` → `inference.runtime.radix` (parsed from runtime `/health.kv_resume`)
- Fleet poll already carries `Inference` on `NodeSnapshot` — peers now expose radix residency
- Soft score bonus when `session_key`/`prompt_cache_key` set and peer has `radix_share` + `entry_count>0` (`ZEROLLAMA_FLEET_RADIX_SCORE`, default on)

**Non-goals:** Go-side `seq_cp`; decode rewrite. Content-hash routing → **L3-R9**.

### L3-R7 — federated LMCache slot blobs (Jul 2026)

**Why:** L3-R4 Redis shared prefix *hashes* but `blob_path` stayed local to the donor node — cold nodes still full-prefilled. Content-addressed blobs close that without NIXL.

**Shipped:**

- **`lmcache_blob.py`** — SHA-256 blob tree under `file://…/blobs` or `ZEROLLAMA_LMCACHE_BLOB_ROOT`
- **`blob_digest`** on `LMCacheBlockRecord` / prefix pool; publish on `register_prefix`
- **`find_blob_prefix` / `radix_blob_restore`** — cold restore when live Radix donor absent; in-process `llama_state_seq_load_file`, subprocess materialize to `--slot-save-path`
- Health: `prefix_block_pool.lmcache_blobs`; env `ZEROLLAMA_LMCACHE_BLOBS` (default on with tier)

**Non-goals:** NIXL RDMA / Mooncake; S3 object store; Go Radix mirror.

### L3-R6b Done — cell + tensor + used-cell pages COW (Jul 2026)

**Why:** Shared `TAG_KV_CACHE_SHARE_CELLS` early-returns blocked diverge writes; Gemma4 share-cb aliases K/V tensors so metadata-only fork was a no-op; full-tensor copy wasted bandwidth when occupancy ≪ `kv_size`.

**Shipped:**

- **`ZEROLLAMA_KV_COW=1`** — `ensure_unique_cells()` deep-copies `llama_kv_cells_vec`
- **`ZEROLLAMA_KV_COW_TENSORS=1`** — `ensure_unique_tensors()` allocates private K/V + copy from donor
- **`ZEROLLAMA_KV_COW_PAGES=1`** — copy only used cell ranges (K contiguous / V `v_trans` row-scatter); ≥80% density falls back to full copy; VRAM still full-size
- Agent YAML `l3.kv_cow` + `kv_cow_tensors` + `kv_cow_pages`; patch **0089**; health `l3_r6b: done`

**Non-goals:** non-agent global default-on; NIXL; sparse/sub-capacity tensor allocation (VRAM reduction).

### Phase 15 v59 — L3-R6 metadata-path readiness (Jul 2026)

**Why:** Close the ops loop on practical shared-prefix KV (v50–v58) with an explicit Done vs deferred split — metadata path shippable; true COW is upstream.

**Shipped:**

- **`l3_r6_metadata_readiness`** — `/health.kv_resume.l3_r6_metadata` (`complete`, checks, deferred)
- ROADMAP **L3-R6a Done** / **L3-R6b** later Partial (see above)

**Non-goals (at v59):** implementing COW; non-agent global default-on.

### Phase 15 v58 — Radix→unified couple (Jul 2026)

**Why:** Full COW still needs llama cell-allocator work. Radix without unified still buffer-copied — couple closes that footgun.

**Shipped:**

- **`ZEROLLAMA_KV_UNIFIED_WITH_RADIX`** — default on; when Radix enabled, enable unified unless `ZEROLLAMA_KV_UNIFIED=0`
- **`kv_unified_source`** — `env` | `yaml` | `radix_couple` | `off` on health

**Non-goals:** true COW on diverge; non-agent global default-on.

### Phase 15 v57 — in-process idle-slot purge (Jul 2026)

**Why:** Subprocess already purges idle slots when unified KV is full; in-process multi-seq had no equivalent and could starve the shared cell pool under L3 resume.

**Shipped:**

- **`idle_slot_purge.py`** — on ctypes `llama_decode` fail under `kv_unified`, clear one idle seq and retry
- **`ZEROLLAMA_KV_UNIFIED_IDLE_PURGE=0`** kill-switch (default on when unified)
- Health `/health.kv_resume.kv_unified_idle_purge`

**Non-goals:** proactive purge; native decode-loop purge; COW.

### Phase 15 v56 — opt-in strict unified sizing (Jul 2026)

**Why:** v55 made undersize visible; CI/agent fleets need an explicit fail-closed load gate without bricking default agent profile hosts.

**Shipped:**

- **`ZEROLLAMA_KV_UNIFIED_STRICT=1`** — `assert_kv_unified_sizing` / `KvUnifiedSizingError` on subprocess argv + in-process init
- Health `kv_unified_sizing.strict` + `runtime_env.kv_unified_strict`

**Non-goals:** default-on strict; auto-bump `-c`; COW.

### Phase 15 v55 — unified sizing probe + default-on criteria (Jul 2026)

**Why:** After v54, operators still had only a prose note for shared-pool sizing. Numeric health probe is required before any global default-on.

**Shipped:**

- **`kv_unified_sizing_status`** — `/health.kv_resume.kv_unified_sizing` (`ok` / `recommended_min_ctx` / floor)
- **`ZEROLLAMA_KV_UNIFIED_MIN_TOKENS_PER_SLOT`** — soft floor (default 512); advisory only
- **Default-on criteria** documented in phase15 (agent smoke + sizing ok + kill-switch; idle purge/COW still block non-agent default)

**Non-goals:** global default-on; hard admit reject on undersize; COW.

### Phase 15 v54 — agent YAML `l3.kv_unified` (Jul 2026)

**Why:** After v53, agent fleets still needed a second env for metadata Radix share. Profile YAML closes that without global default-on.

**Shipped:**

- **`l3.kv_unified`** — same env-wins pattern as `radix_share` (`ZEROLLAMA_KV_UNIFIED=0/1` overrides)
- **`l3_agent_subprocess.yaml`** — `kv_unified: true` with sizing WHY comment
- **`/health.kv_resume.kv_unified_note`** — shared cell-pool sizing advisory when on

**Non-goals:** global default-on; coupling unified to every `radix_share`; numeric undersize fail; COW.

### Phase 15 v53 — subprocess `--kv-unified` argv (Jul 2026)

**Why:** v52 only wired in-process; L3 agent / Radix live path is subprocess. `ZEROLLAMA_KV_UNIFIED=1` was a no-op for that path while health claimed metadata share.

**Shipped:**

- **`with_llama_kv_unified`** — inject `--kv-unified` into llama-server argv when env on; `--no-kv-unified` / `-no-kvu` wins
- **`_llama_server_start_args`** — calls it after cache argv

**Non-goals:** default-on; agent-profile auto-enable; in-process idle purge; COW.

### Phase 15 v52 — opt-in unified KV stream (Jul 2026)

**Why:** Real L3-R6 share without rewriting llama's cell allocator: `kv_unified` → `n_stream=1` → same-stream `seq_cp` is metadata-only. Default stays off (shared cell pool contention).

**Shipped:**

- **`ZEROLLAMA_KV_UNIFIED=1`** — sets `cparams.kv_unified` on in-process multi-seq load
- Overlay donor estimate uses `streams=1` when unified
- In-process Radix `seq_cp` hardened to full-range (`-1,-1`) + trim (server parity)
- Health/trace: `kv_unified`, `seq_cp_mode` (`metadata` | `buffer_copy`); metadata mode records 0 approx copy bytes

**Non-goals:** default-on; subprocess `--kv-unified`; idle-slot purge; COW.

### Phase 15 v51 — overlay donor page-offset catalog (Jul 2026)

**Why:** L3-R6 needs proof that PA pages are addressable ranges inside the v50-owned donor before any cell-share / skip-`seq_cp` work.

**Shipped:**

- **`overlay_page_catalog.py`** — `span_in_donor` / `page_donor_offsets` / live `map_page` catalog
- **`/health.kv_page_bind.overlay_page_catalog`** — summary (`all_in_donor`, pages checked; no per-page rows)
- **`/internal/kv-snapshot.overlay_page_catalog`** — full `pages[]` with `k_offset`/`v_offset`/`block_id`
- Engine mirrors donor `ptr`/`size` from auto-wire or manual register

**Non-goals:** skip `seq_cp`, COW, `kv_unified`, CUDA donor.

### Phase 15 v50 — overlay donor auto-wire + L3-R6 start (Jul 2026)

**Why:** Physical shared KV pages (true RadixAttention) need the runtime to own the KV byte region before llama can share cells. v48/v49 required manual register-before-load; Radix still `seq_cp`s.

**Shipped:**

- **v50 auto-wire** — with `ZEROLLAMA_KV_OVERLAY_BIND=1`, in-process load allocates a page-aligned host donor (GGUF estimate × streams × 2 + 32 MiB, or `ZEROLLAMA_KV_OVERLAY_DONOR_BYTES`) and registers it before context construction. Kill-switch: `ZEROLLAMA_KV_OVERLAY_AUTO=0`.
- **`/health.kv_page_bind.overlay_bind_auto`** — operator visibility.
- **Radix approx copy cost** — `approx_copy_bytes` on share traces + `approx_copy_bytes_total` on `radix_share` health (observability until physical share).
- **L3-R6 Partial** — ROADMAP milestone opened; multi-seq shared cells + COW still deferred.

**Doc:** [phase15-native-kv.md](docs/phase15-native-kv.md) v50, [radix-prefix-share.md](docs/radix-prefix-share.md).

### L2 auto-VRAM fork flag + track close (Jul 2026)

**Why:** Measured L2 gates FAIL flipping defaults for tok/s (stock faster). TBQ still wins VRAM at long ctx (−27…−35%). Operators needed either manual `ZEROLLAMA_LLAMA_FORK=1` or a topology YAML — not a ctx-aware opt-in.

**Shipped:**

- **`ZEROLLAMA_LLAMA_FORK_AUTO_VRAM=1`** — when `FORK` unset, enable fork (TBQ via `FORK_PROFILE=vram`) only if configured ctx ≥ threshold (default **32768** via `ZEROLLAMA_LLAMA_FORK_AUTO_VRAM_CTX`). Ctx hint: `ZEROLLAMA_RUNTIME_VRAM_NUM_CTX` / `ZEROLLAMA_LLAMA_CTX` / `LLAMA_ARG_CTX_SIZE`. Explicit `FORK=0/1` still wins.
- **YAML** — `serve.llama_fork_auto_vram` / `llama_fork_auto_vram_ctx` in `vram_yaml_defaults`.
- **L2 Done** — ROADMAP / [gpu-profiles-l2.md](docs/gpu-profiles-l2.md): infrastructure + VRAM opt-in complete; defaults stay L1 until tok/s ≥2/3 gate passes.

### Marconi × retention preservation for Radix (Jul 2026)

**Why:** vLLM #47782 fixed selective hybrid retention killing Marconi-style shared-system-prompt cache hits. Zerollama cold seed was already OK at `seq_pos=0`, but (1) Radix applied full `swa_allows_cache_prompt` (including retention) so warm catch-up at non-aligned positions got `hybrid_swa_denied`, and (2) admission skipped Radix entirely when full-prompt `cache_prompt` was denied for SWA window — even when a shorter matched prefix would fit.

**Shipped:**

- **`radix_seq_copy_policy`** — window-only hybrid gate (no retention); retention stays for same-slot resume.
- **`_prefix_cache_admission`** — try Radix even when `cache_prompt` was denied; successful seed flips `allow` True. Draft/`allow_cache_prompt=false` still skips Radix.
- **Docs** — [vllm-borrowings.md](docs/vllm-borrowings.md) records the borrow + explicit non-goals (KV watermark, partial hybrid hash hits).

### Native FP8 GGUF weights (E4M3 / E5M2) — Jul 2026

**Why:** HF FP8 checkpoints were only usable after full F16/BF16 dequant (or re-quant). That wastes convert time, disk, and VRAM, and leaves no first-class CUDA matmul path for native E4M3/E5M2 GGUF weights. This is **weight** FP8 — not FP8 KV (fork KV stays QJL/Polar/TBQ).

**Shipped:**

- **`GGML_TYPE_FP8_E4M3=51` / `FP8_E5M2=52`** — F16 block scale + 32× IEEE FP8 (34 B / block, Q8_0-shaped)
- **Patches 0073–0076** — type + CPU quant/dot; CUDA convert/get_rows; MMVQ (float×Q8_1) + MMQ (amax→int8 tiles); convert `--fp8-native` including **128×128** `weight_scale_inv` when `block_size[-1]%32==0`; E5M2 twin
- **Runtime** — `gguf_estimate` layouts; `/health` / `/ready` `cuda_weight_formats.{fp8_e4m3,fp8_e5m2}` needles
- **Probes** — `./scripts/fp8_cuda_probe.sh`, `fp8_e4m3_gguf_roundtrip.py`, `fp8_e5m2_gguf_roundtrip.py`
- **Build** — container SET_ROWS verify uses `grep -aF` first (**why:** `strings|grep` false-negatives on freshly linked bind-mounted libs); build lock `mkdir -p` parent (**why:** failed builds `rm -rf` the build dir and mis-report “lock held”)
- Packaged `/usr/local/lib/ollama` refreshed from vendor tip in `LLAMA_CPP_VENDOR_HEAD` (restart `zerollama-runtime` to load)

**Doc:** [native-fp8-gguf.md](docs/native-fp8-gguf.md), [cuda-lanes.md](docs/cuda-lanes.md).

**Non-goals:** Blackwell-native FP8/NVFP4 MMA (5080 P2); MLX mxfp8; flipping production `llama_fork` off stock.

### Explicit context-overflow fields (Jul 2026)


**Why:** A ~44k-token prompt at `num_ctx=8192` returned HTTP 200 with no `prompt_truncated`. Clients had to infer overflow from `prompt_eval_count` pinned near the window (and sometimes `done_reason: "length"`). Two gaps caused that:

1. **Go `chatPrompt`** — `tailTruncatePrompt` dropped tokens but discarded the pre-truncation count; `recordInferencePromptSize` hardcoded `originalTokens=0`. Runner trim then reported `original_prompt_tokens` as the already-trimmed size (~8192), not ~44k.
2. **Runtime proxy** — `/api/generate` and `/api/chat` forwarded the raw prompt to the Python sidecar without Go truncation. llama-server context-shifted silently; responses had no truncation metadata.

**Shipped:**

- **`chatPrompt` → `originalPromptTokens`** — propagate pre-drop size through routes into `applyPromptTruncation` / `applyGenerateTruncation` (prefer chatPrompt count over smaller runner count).
- **Runtime `detect_context_overflow`** — on stream done chunks (and non-stream generate), set `prompt_truncated` / `original_prompt_tokens` when admit-time tokens exceed `num_ctx` or `prompt_eval_count` is pinned near the window.
- **OpenAPI + docs** — fields documented on Generate/Chat response schemas; see [scheduling-vram-policy.md](docs/scheduling-vram-policy.md#prompt-truncation-in-responses).

**Client check:** `prompt_truncated == true` (or compare `original_prompt_tokens` to `num_ctx`). Soft signals remain: `prompt_eval_count ≈ num_ctx`, `done_reason: "length"`. Set `"truncate": false` for HTTP 400 instead of silent drop on Go paths that honor it.

### L2 CUDA on ggml-org `8f114a9b` (patches 0067–0070) — Jul 2026

**Why:** Rebase onto ggml-org pin broke TBQ load (missing CPU `type_traits` / CUDA SET_ROWS + fattn vec routing). QJL/Polar as the default fork pairing also lost badly on 4090 tok/s.

**Shipped:**

- **0067–0070** — TBQ flash-attn vec helpers, CLI KV types + `-cpent`, CUDA SET_ROWS/GET_ROWS + fattn routing, CPU type_traits/dispatch for TBQ/QJL/Polar
- **`ZEROLLAMA_LLAMA_FORK_PROFILE` default `vram`** (TBQ) instead of `speed` (QJL/Polar) when `ZEROLLAMA_LLAMA_FORK=1`
- 4090 gates: TBQ long-ctx **−27…−35% VRAM**; QJL/Polar **−48…−85% decode** @ 8k/27k — speed stays experimental
- Docs / pin status / `l2_cuda_bench.sh` health alignment; vendor → in-tree sync
- **`LLAMA_CPP_VENDOR_HEAD`** → `95f753fd` (post-0067–0070 tip); patch doctor / build scripts probe `libllama-server-impl*` for `/kv/seq-copy` (thin wrapper no longer embeds the string)
- Force `-fa on` when fork KV types are TBQ/QJL/Polar (llama.cpp hard-requires FA for quantized V)
- `l3_radix_prefix_smoke.sh`: derive llama-server port from `ZEROLLAMA_RUNTIME_URL` (never hard-kill prod `:8081`/`:8082`)
- **0071** — fix `/kv/seq-copy`: do not `prompt_clear` after copy (it `seq_rm`s the KV just written); match `copy_state_to` `-1,-1` ranges; L3 radix live PASS on 4090
- Packaged `/usr/local/lib/ollama` refreshed from vendor `5a4d99f0` (0071); L3 cache smoke PASS on 4090 (`/tmp/l3-cache-smoke-8f.json`) — restart `zerollama-runtime` to load new libs

**Non-goals:** flipping production `llama_fork` off stock; merging fork profiles as default-on for tok/s.

### ComfyUI image backend (agent-max utility) — Jul 2026

**Why:** Agents need edit / img2img / ControlNet / LoRA on Qwen-Image, FLUX.1/2-dev, GLM-Image, and Klein 9B — not only MLX Z-Image / Klein 4B. Porting each DiT into `x/imagegen` would take months per family; ComfyUI already packages those graphs. Zerollama **orchestrates** a running ComfyUI (`modality_backends.image=comfyui`) instead of embedding Diffusers or expanding the raw `external-image` hook.

**Shipped:**

- **`BackendComfyUI`** + `handleComfyImageGenerate` — routes `/api/generate` and OpenAI `/v1/images/generations|edits`; calls `vram.PrepareForImageGen` (**why:** exclusive GPU like MLX, unlike historical `external-image`).
- **`server/modality/comfyui`** — upload → inject bindings → `POST /prompt` → poll `/history` → `/view`; surfaces Comfy `execution_error` details (**why:** `/prompt` often returns 200 before node OOM/missing-checkpoint fails).
- **Agent options** — `options.workflow`, `negative_prompt`, `lora` / `lora_strength`, `control_image` / `control_strength`; discovery via `GET /api/image/workflows?model=`.
- **Config-only presets** — `comfy/qwen-image` (+ img2img/controlnet), `comfy/qwen-image-edit`, `comfy/flux1-dev`, `comfy/flux2-dev`, `comfy/glm-image`, `comfy/flux2-klein-9b`; register with `./scripts/register_comfy_models.sh`.
- **Env** — `OLLAMA_COMFYUI_URL`, `OLLAMA_COMFYUI_TIMEOUT`, `OLLAMA_COMFYUI_WORKFLOWS_ROOT` (**why root:** manifests ship relative `scripts/comfyui/...` paths that otherwise depend on daemon cwd).
- **Workflow JSON** — worked examples under `scripts/comfyui/`; **not** verified drop-in against every Comfy+custom-node install (calibrate filenames/node types first). No fake default LoRA (`none.safetensors`) — Comfy rejects missing files.
- **Tests** — unit (render, mock HTTP, path resolve, execution errors); opt-in `RUN_E2E_COMFY=1`.
- **Doc:** [comfyui-image-backend.md](docs/comfyui-image-backend.md), [multimodal-backends.md](docs/multimodal-backends.md), roadmap image-generation track.

**Non-goals:** bundling ComfyUI in the Go binary; MLX ports of Qwen/GLM/FLUX.1; interactive-speed guarantees for GLM-Image / FLUX.2-dev on 16 GB; cancelling in-flight Comfy jobs on HTTP disconnect (follow-up: `/interrupt`).

### Qwen3-Next MTP (patch 0065)

**What:** Port [#25589](https://github.com/ggml-org/llama.cpp/pull/25589) for our hybrid MoE stack (qwen3next / eliza-class).

- **0065** — NextN/MTP draft graph + `graph_mtp` decl; hybrid KV filters include `LLM_ARCH_QWEN3NEXT`
- In-tree CGO uses `nextn_predict_layers` (qwen35moe parity); vendor pin keeps upstream `n_layer_nextn`
- MTP/NextN GGUFs needed to exercise the draft head; trunk load path unchanged for non-MTP weights
- Conversion scripts from the PR deferred (use upstream convert when building MTP GGUFs)

**Vendor HEAD:** `2dfb59d30` (`LLAMA_CPP_VENDOR_HEAD`).

### In-tree Metal dig re-sync (0062–0064)

**What:** Re-apply vendor Metal dig into CGO `ml/backend/ggml/` after the MTP cleanup wipe.

- **0063** — already present on in-tree `ggml-metal-context.m` (no-op re-apply)
- **0064** — TQ2_0 Metal kernels applied to in-tree `ggml-metal.metal` / device / impl
- **0062 E8_2** — in-tree at type id **`51`** / ftype **`29`** (keeps eliza `Q1_0_g32/g128` at 42/43; vendor dig still uses id 43)
- `build_zerollama_mac.sh` now compiles + embeds `default.metallib` (+ eliza-shipped) instead of Metal source

### Metal TQ2_0 (patch 0064)

**What:** Rewrite [#12485](https://github.com/ggml-org/llama.cpp/pull/12485) Metal TQ2_0 onto modern pipeline API (PR still targets old `ggml-metal.m`).

- **0064** — dequant + mul_mv/get_rows/mul_mm/mul_mm_id/mul_mv_id for `GGML_TYPE_TQ2_0` (type already on pin)
- `N_R0_TQ2_0=4` / `N_SG_TQ2_0=2` (matches old PR 8-row TG dispatch)

**Vendor HEAD:** `4d637f2e` (`LLAMA_CPP_VENDOR_HEAD`). Still defer MoE disk #23440 / CE WIP #18121 / RWKV fuse #25206.

### Metal non-Apple concurrency guard (patch 0063)

**What:** Extract correctness fix from [#19527](https://github.com/ggml-org/llama.cpp/pull/19527) (AMD/Intel Metal).

- **0063** — auto-disable `MTLDispatchTypeConcurrent` when `!supports_gpu_family_apple7`
- Skipped PR managed-buffer hunks (invalid `BytesNoCopy`+Managed / buffer `synchronizeResource`)

**Vendor HEAD:** `73e05a20` (`LLAMA_CPP_VENDOR_HEAD`). Still defer MoE disk #23440 / TQ2_0 / full #19527 managed path.

### GGML_TYPE_E8_2 KV quant (patch 0062)

**What:** Port [#25352](https://github.com/ggml-org/llama.cpp/pull/25352) E8 lattice 2-bit KV (`2.125` bpe, `QK_E8_2=128`).

- **0062** — CPU quant/dequant/vec_dot + Metal dequant/quantize, cpy, get_rows, set_rows, FA (dk128/256)
- Type id **`43`** / ftype **`29`** (upstream used 42/28; those are `Q2_0` on this pin)
- Metal: use `const float lut[]` (not `constant`) — address-space on automatic locals fails MSL compile
- CUDA hunks deferred (Mac Metal path)

**Vendor HEAD:** `9267b8bc` (`LLAMA_CPP_VENDOR_HEAD`). Still defer MoE disk #23440 / TQ2_0.

### Metal async 2D tensor copy (patch 0061)

**What:** Fill backend `set/get_tensor_2d_async` (were NULL) from [#22515](https://github.com/ggml-org/llama.cpp/pull/22515).

- **0061** — Metal blit-based 2D strided async copies (multi-GPU / batched host↔device)
- Fixed upstream PR wiring bug (set/get were swapped in the iface table)

**Vendor HEAD:** `9267b8bc` (`LLAMA_CPP_VENDOR_HEAD`). Still defer MoE disk #23440 / TQ2_0.

### Metal IM2COL_3D (patches 0058–0060)

**What:** Extract remaining useful piece of [#16669](https://github.com/ggml-org/llama.cpp/pull/16669) (DIAG_MASK already via 0044).

- **0058** — Metal `IM2COL_3D` f32/f16 kernels + dispatch (3D conv im2col on GPU)
- **0059** — wire `src0`/`src1`/`dst` for `GGML_TENSOR_BINARY_OP_LOCALS`
- **0060** — explicit f32/f16 kernels (Metal rejects decltype template instantiation here)

**Vendor HEAD:** `cc1364ff` (`LLAMA_CPP_VENDOR_HEAD`). Still defer MoE disk #23440 / TQ2_0; async 2D landed in 0061.

### Metal ARGMAX ties + FA V-skip (patches 0056–0057)

**What:** Small correctness/opt-in Metal dig after OUT_PROD.

- **0056** — [#25032](https://github.com/ggml-org/llama.cpp/pull/25032): ARGMAX first-index-on-ties (Metal simd + CPU `vec_argmax`)
- **0057** — [#21119](https://github.com/ggml-org/llama.cpp/pull/21119): opt-in FA V-skip (`GGML_METAL_FA_SKIP_V=1`)

**Vendor HEAD:** was `aa1f50cb`; superseded by 0058 above.

### Metal OUT_PROD (patch 0050)

**What:** Dig of remaining open Metal PRs; port the clean missing op.

- **0050** — [#23724](https://github.com/ggml-org/llama.cpp/pull/23724): partial Metal `OUT_PROD` (f32 + q4_0/q4_1/q8_0/mxfp4 × f32)
- Already on pin (no port): #21782 `ROLL`, #18878 `FLOOR`/ceil/round (unified unary), #22595 rsets_rm (ANE path)
- Still defer: #23440 MoE disk (+920), #12485 TQ2_0 (old metal.m); IM2COL_3D landed in 0058–0060; async 2D in 0061

**Vendor HEAD:** was `c551f738`; superseded by 0056–0057 above.

### Metal GLA + NVFP4 (patches 0048–0049)

**What:** Remaining high-value Metal niche ports after 0046–0047.

- **0048** — [#21452](https://github.com/ggml-org/llama.cpp/pull/21452): Metal `GATED_LINEAR_ATTN` (head 64/128)
- **0049** — [#20456](https://github.com/ggml-org/llama.cpp/pull/20456): Metal NVFP4 mul_mat/get_rows (type already on pin)

**Vendor HEAD:** was `67426a65`; superseded by 0050 above.

### Metal snake fuse + Q2_0 (patches 0046–0047)

**What:** Next deferred Metal niche cluster after 0044–0045.

- **0046** — [#25459](https://github.com/ggml-org/llama.cpp/pull/25459): fuse snake activation (`mul→sin→sqr→mul→add`) on Metal
- **0047** — [#25419](https://github.com/ggml-org/llama.cpp/pull/25419): Q2_0 Metal backend (dequant / mul_mv / mul_mm / cpy / get_rows)

**Vendor HEAD:** was `d0621e6d`; superseded by 0048–0049 above.

### Metal DIAG_MASK_INF + pad_reflect optimize (patches 0044–0045)

**What:** Small correctness/perf Metal ports after 0043.

- **0044** — [#24844](https://github.com/ggml-org/llama.cpp/pull/24844): Metal `DIAG_MASK_INF` (f32) so causal mask stays on GPU
- **0045** — [#23992](https://github.com/ggml-org/llama.cpp/pull/23992): `PAD_REFLECT_1D` float4 path when p0/row strides are 16-byte aligned

**Vendor HEAD:** was `cc61c3f8`; superseded by 0046–0047 above.

### Metal q8_0 KV flash-attn + QJL build fix (patches 0042–0043)

**What:** Next Metal cluster after 0032–0041, plus rebuild-time ggml correctness.

- **0042** — fix QJL `GGML_OP_COUNT` (97→101) / `OP_SYMBOL` lag and corrupted TBQ/`TQ` wrappers in `ggml-quants.c`
- **0043** — [#25556](https://github.com/ggml-org/llama.cpp/pull/25556): Q8_0→F16 materialize for head256 high-GQA FA, packed q8_0 loads, `vec_gqa2` (rebased onto `nqptg` from #21443)

**Vendor HEAD:** was `f245612e`; superseded by 0044–0045 above.

### Metal correctness ports from ggml-org (patches 0032–0034)

**What:** Cherry-picked three open upstream Metal PRs onto pin `8f114a9b` (still open upstream; ported as format-patches).

- **0032** — [#25371](https://github.com/ggml-org/llama.cpp/pull/25371): null-check Metal buffer alloc on OOM (avoid `is_shared(NULL)` crash)
- **0033** — [#24368](https://github.com/ggml-org/llama.cpp/pull/24368): wind down residency sets at teardown instead of `GGML_ASSERT` abort
- **0034** — [#25442](https://github.com/ggml-org/llama.cpp/pull/25442): MoE down-proj L2 rescale so Metal `MUL_MAT_ID` f16 cast does not NaN

**Vendor HEAD:** was `e9de09c6`; superseded by 0035–0036 below.

### Metal small-batch mul_mat (patches 0035–0036)

**What:** Speculative / small-batch Metal matmul path from open upstream PRs (issue [#25250](https://github.com/ggml-org/llama.cpp/issues/25250)).

- **0035** — [#25453](https://github.com/ggml-org/llama.cpp/pull/25453): extend small-batch mat-vec dispatch to `ne11 <= 16` (+ `r1ptg` for 9..16)
- **0036** — [#25377](https://github.com/ggml-org/llama.cpp/pull/25377): 64×8 `mul_mm` tile for `q4_0×f32` at bs 5..16; Q4_0 hands off at `ne11 > 4` (merged with 0035 so non-Q4_0 still use mat-vec through 16)

**Vendor HEAD:** `7abf6c43` (`LLAMA_CPP_VENDOR_HEAD`). Synced Metal + regenerated `ggml-metal-embed.metal`.

### Metal Qwen3-VL flash-attn (patch 0037)

**What:** [#21443](https://github.com/ggml-org/llama.cpp/pull/21443) — ~11% faster large-image encode on Metal for f16 head72 (Qwen3-VL).

- **0037** — `nqptg=16` + `nsg=8` for f16 dk72; generalize FA kernel Q loops (`QB = Q/8`); new `kernel_flash_attn_ext_f16_dk72_dv72_q16`

**Vendor HEAD:** `6e5cb627` (`LLAMA_CPP_VENDOR_HEAD`).

### Metal Qwen3-Next / Mamba / deployment (patches 0038–0040)

- **0038** — [#23401](https://github.com/ggml-org/llama.cpp/pull/23401): `@available` guard for `waitUntilSignaledValue`
- **0039** — [#25533](https://github.com/ggml-org/llama.cpp/pull/25533): Mamba-2 SSM scan group kernel (Nemotron-H)
- **0040** — [#16143](https://github.com/ggml-org/llama.cpp/pull/16143): fused `RMS_NORM+MUL+SWIGLU` for Qwen3-Next (rebased onto `get_pipeline_norm` / float4 kernels)

**Vendor HEAD:** `9024286f` (`LLAMA_CPP_VENDOR_HEAD`). Dig high-value Metal ports complete (0032–0040).

### Build fix: GGML_ALIGN / GGML_THREAD_LOCAL (patch 0041)

**What:** QJL/Polar CPU kernels (0027) used `GGML_ALIGN` / `GGML_THREAD_LOCAL` without defining them — Mac CGO failed. **0041** adds the macros to `ggml.h`.

**Vendor HEAD:** `3b8af95a8`.

### ElizaOS QJL/Polar/TBQ extraction (patches 0026–0030)

**What:** Ported QJL (Quantized Johnson-Lindenstrauss), PolarQuant Q4, and TurboQuant 3/4-bit KV-cache compression from the elizaOS llama.cpp fork into the ggml-org vendor tree as formal patches.

- **0026** — types + ops: 7 new GGML_TYPE_* enums, 4 new GGML_OP_* ops, graph builders, type_traits
- **0027** — CPU kernels: ~32 files (qjl/, polarquant/, fused-attn, quants), full quantize/dequantize
- **0028** — CUDA kernels: QJL score, PolarQuant decode, TBQ3_TCQ, fused attention, SET_ROWS
- **0029** — Metal shaders: standalone .metal files compiled into metallib
- **0030** — Fused QJL-K attn dispatch + SET_ROWS wiring in ggml-cuda.cu + llama-graph.cpp

**Deferred:** Vulkan dispatch (shaders exist in elizaOS but not wired to backend).

### MLX bump to `main` tip `4367c73b` (Jul 2026)

**Why:** Local sibling `../mlx` and `dist/.../libmlx.dylib` were still on the May pin (`2165dc08`) while the repo had tracked Ollama’s Jul 3 pin (`de7b4ed9`). Operators asked for latest MLX; mlx `main` was 18 commits ahead of that pin (NAX configure warnings, GGUF metadata int64 fix, etc.).

**What shipped:**

- **`MLX_VERSION`** → `4367c73b` (2026-07-10 — Warn when NAX kernels disabled). **`MLX_C_VERSION`** unchanged at `fba4470b` (already mlx-c `main` tip / Ollama 0.31.2 bindings).
- Rebuilt Metal **v3** + **v4** dylibs into `build/metal-v*/` and `dist/darwin-arm64/lib/ollama/mlx_metal_v*/` (`libmlx` embeds `4367c73`).
- Sibling checkouts: `../mlx` @ pin, `../mlx-c` @ pin.

**Operator:** restart `zerollama serve` so mlxrunner reloads dylibs (already-loaded processes keep the old mapping). Verify: `./zerollama doctor` → mlx engine ok; `strings …/libmlx.dylib | grep 4367c73`.

### Vendor rebase to ggml-org master `8f114a9b` (Jul 2026)

**Why:** Operators asked for latest [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (not elizaOS). Pin is master tip `8f114a9b` (past tag `b9951`).

- **Pin:** `LLAMA_CPP_VERSION=8f114a9b`, `LLAMA_CPP_COMMIT=8f114a9b573b…`, `Makefile.sync` → `vendor/llama-cpp-8f114a9b` (`UPSTREAM=ggml-org/llama.cpp`, `BUILD_NUMBER=9952`)
- **Patches:** **41** format-patches (incl. 0032–0040 Metal ports); clean re-apply **`fail=0`**
- **Deferred:** eliza fused QJL/Polar/TBQ CUDA (**old 0021**) — types not in ggml-org; restore via eliza overlay or extraction series if L2 fork KV is required
- **Includes:** kv-ext + alias probe, seq-copy, ANE lab, SWA/mmproj fit, `n_ubatch` cap, GPU discovery, compat hooks

### Vendor rebase to elizaOS `ad56033f` / b10064 (Jul 2026) — superseded

Brief elizaOS pin for in-tree QJL; replaced by ggml-org master above when operators requested upstream tip.
### Upstream Ollama v0.31.2 cherry-picks (Jul 2026)

**Why:** Upstream shipped inference fixes and MLX bumps between v0.30.11 and v0.31.2; zerollama ports additive Go/compat fixes. Vendor pin is elizaOS **`ad56033f`** (see above) — not required to match Ollama’s ggml-org `b9888` for these cherry-picks.

- **#16996** — iGPU mmproj offload with `LLAMA_ARG_FIT_TARGET` padding (`llm/llama_server.go`)
- **#15901** — apply format constraint for all thinking parsers when `think=false` (`server/routes.go`)
- **#16994** — Flash Attention on CUDA CC 6.x (`ml/device.go`)
- **#16949** — JetPack runner fallback to standard CUDA when jetpack libs absent (`discover/runner.go`)
- **#16999** — UTF-8-safe compat tensor reads via `ggml_fopen` (`llama/compat/llama-ollama-compat-util.cpp`)
- **#16964** — Gemma 4 MoE fused expert loading for quantized `.experts.*` tensors (`x/models/gemma4/gemma4.go`)
- **#17056** — MLX bump to `de7b4ed9` (`MLX_VERSION`; `MLX_C_VERSION` unchanged)
- **main `d47859ce`** — Qwen3.5/Next parser/renderer selection before broad `qwen3` match (`x/create/client/create.go`)

**Not ported:** agent UI/harness (#17017, #16963), wholesale `x/create` rewrite (#16919), Mac-default llama-server runner removal, GGUF create hardening (#17062 — evaluate separately).

### Phase 15 v33 writable KV page bind + Darwin sidecar compatibility (Jul 2026)

**Why:** PA block tables needed writable K/V tensor spans via fork `llama_memory_kv_page_map`; Mac builds had to rebuild `_kv_native` when kv-ext changed; stale `:8081` sidecars survived `zerollama serve` restarts and reported `native_ext_not_built` despite a fresh `.so` on disk.

- **Fork kv-ext v33** — `llama_memory_kv_page_map` in `llama-kv-ext.h` / `llama-memory-kv-ext.cpp`; `LLAMA_KV_EXT_WRITABLE_PAGE_MAP=1` on libllama build.
- **Native ext** — `kv_tensor_probe.c`, `page_bind.py`, `/health.kv_page_bind.writable_bind_*` and `physical_pages_bound` after decode probe.
- **Mac build** — `scripts/vendor/stage_llama_kv_ext_for_vendor.sh`; `BUILD_RUNTIME_KV_EXT=auto` in `build_zerollama_mac.sh` with `.build-stamps/runtime-kv-native.sha` fingerprint cache.
- **Darwin sidecar** — `BootstrapDarwinSidecar` compares `kv_native_build_sha` on `/health` vs on-disk stamp; stops and respawns stale sidecar on mismatch (default persist policy no longer traps old Python processes).
- **v34 multi-layer bind** — `llama_memory_kv_n_layers`; tensor probe loops all KV layers, not just layer 0; writable `llama_memory_kv_page_map` fans out per layer so every attention layer's K/V backing is verified; `/health.kv_page_bind.kv_n_layers` + `tensor_layers_verified`. **Why layer 0 was insufficient:** LLMs with per-layer k/v scaling (MLA, GQA variants) can allocate different tensor shapes per layer; verifying only layer 0 would silently pass the probe while deeper layers were unmapped.
- **v35 transposed-V layout + last-probe health** — `llama_memory_kv_cache_layout` exposes `kv_size`, `n_stream`, `v_transposed`; `llama_kv_page_map.v_transposed` flags non-FA V layout on each page result; multi-stream cell stream-consistency guard added to `llama_kv_ext_page_map_contiguous`; MLA null-V allowed through the arg guard (was incorrectly rejected); `page_bind_last_tensor_probe` persists the last successful decode probe indexed by bind-slot position (not kv_slot value), so `/health` can show `kv_v_transposed`, `kv_cache_kv_size`, `kv_cache_n_stream`, `tensor_layers_verified` **even after `page_bind_clear`** on generation complete. **Why transposed-V matters:** default (non-FA) llama KV caches write V tokens along dim 1 but each token's embedding is scattered across `n_embd_v_gqa` rows; a page-map caller reading `v_span_bytes` as a flat buffer gets interleaved data from multiple embedding dimensions — the `v_transposed` flag is required to select the correct scatter/gather access pattern.
- **v36 GGUF layer-group enrichment** — `page_bind_health(kv_coordinator=)` accepts a `HybridKVCacheCoordinator`; engine resolves coordinator from `_health_gguf_path()` at health time; `/health.kv_page_bind` adds `kv_coordinator_kind`, `kv_full_layers`, `kv_swa_layers`, `tensor_layers_expected`. **WHY:** hybrid models (Gemma 3/4, Falcon H1) have full-attention + SWA layers; the llama attn cache only backs full-attention layers so `tensor_layers_verified < total_layers` is expected — `tensor_layers_expected = kv_full_layers` gives operators the correct bind-success comparison without inferring a failure from the SWA gap.
- **v37 stream auto-batch** — `StreamAutoBatchCoordinator` coalesces concurrent streaming `/api/generate` when `ZEROLLAMA_KV_AUTO_BATCH_STREAM=1`; demuxes `completions_parallel_stream` chunks by `request_id`/`seq_idx`; `/health.kv_auto_batch` splits into `non_stream` + `stream` stats. **WHY:** v32 only batched non-stream `generate()`; concurrent streaming HTTP still ran one `llama_decode` per row per token step.
- **v38 external-buffer copy descriptors** — `page_copy_descriptor()` + `map_page(..., kv_layer=)` / `map_page_all_layers()`; `/health.kv_page_bind.tensor_layers_bind_complete` boolean; `external_buffer_alias_ready=false` on descriptors (true ggml alias still upstream-blocked). **WHY:** raw `v_span_bytes` is misleading when `v_transposed=1`; migration code needs structured copy plans with row-stride scatter/gather metadata.
- **v39 migration plan on kv-snapshot** — `build_page_migration_plan()` + `kv_page_migration` on `GET /internal/kv-snapshot` when tensor/physical bind is complete; live running request preferred, last-probe fallback. **WHY:** v38 descriptors required manual `map_page` calls — snapshot gives operators the full pages×layers plan without a script.
- **v40 migration summary + pointer redaction** — `page_migration_summary` on running `kv_forward_plans`; snapshot plans redact `src_ptr` by default (`ZEROLLAMA_KV_MIGRATION_INCLUDE_PTRS=1` for raw pointers). **WHY:** /health is polled frequently — summary avoids expensive map_page fan-out and leaking process-local addresses into logs.
- **v41 operator sign-off wiring** — health/snapshot smokes assert v40 redaction; `phase15_stream_auto_batch_smoke.sh` for concurrent streaming auto-batch (`RUN_P15_STREAM_AUTO_BATCH=1` in metal sign-off).
- **v42 migration summary on kv_page_bind** — `page_migration_summary` on `/health.kv_page_bind` after bind; `migration_summary` on all `kv_page_migration` snapshot branches; last-probe full plan when ctx still loaded. **WHY:** post-decode operators poll kv_page_bind, not forward plans — summary must survive bind clear via last-probe snapshot.
- **v43 migration summary GPU sign-off** — `smoke_runtime_assert_migration_summary()` + `phase15_migration_summary_smoke.sh`; metal sign-off step 2b + inprocess multiseq post-generate gate.
- **v44 non-stream auto-batch GPU smoke** — `phase15_auto_batch_smoke.sh`; opt-in `RUN_P15_AUTO_BATCH=1` in metal sign-off (`ZEROLLAMA_KV_AUTO_BATCH=1` on sidecar).
- **v45 auto-batch sign-off wiring** — `phase15_runtime_auto_batch_env_apply()` exports env before multiseq sidecar restart; `phase15_auto_batch_signoff.sh` + `RUN_P15_AUTO_BATCH_ALL=1`; `smoke_runtime_assert_kv_auto_batch()` shape check.
- **v46 Linux embed auto-batch parity** — `phase15_inprocess_multiseq_smoke.sh` exports kv + auto-batch env to embed serve; `RUN_E2E_PHASE15_AUTO_BATCH=1` on 5080 session.
- **v47 external-buffer alias validate (staging)** — patch **0019** on `llama-kv-ext.h`: `llama_memory_kv_ext_external_alias_probe`, `llama_memory_kv_page_alias_validate` (`LLAMA_KV_EXT_EXTERNAL_ALIAS=1`); alias modes classify feasibility without tensor mutation. **WHY:** v38 copy descriptors know *how* to migrate bytes, but operators still need a machine-readable answer to “can this external PA pointer zero-copy alias llama’s page_map span?” before v48 overlay bind ships. Runtime: `external_alias_probe()`, `alias_validate()`, `/health.kv_page_bind.external_alias_*`; `page_copy_descriptor(..., alias_plan=)` sets `external_buffer_alias_ready` when `alias_ready` (SAME_POINTER only). Metal stacks report `BLOCKED_DEVICE` until device alias path exists.
- **SIGBUS fix (post-decode `/health`)** — `kv_native_probe_result_dict()` `Py_BuildValue` format aligned to 20 integer probe fields (v35 `kv_cache_kv_size` / `kv_cache_n_stream`); short format treated GPU K/V pointers as C strings → SIGBUS after multiseq generate + health poll.
- **`smoke_runtime_assert_kv_snapshot()`** — accepts `bound`+`physical`; allows `kv_bind.physical_pages_bound` when v33 writable page-map is linked.

### Agent QoS hardening, project tracking, and cross-backend safety (M15b, Jul 2026)

**Why:** Production Jul 4 logs showed two concurrent MLX streams slipping through QoS defer checks (TOCTOU), MLX subprocess death with no stderr in logs, and operators unable to see which agent harness owned a loaded model. Separately, Tier 2 client options (eliza, keep_alive floor, prefix-mm-cache) risked affecting vanilla Ollama / vLLM / CUDA proxies if sent or injected unconditionally.

- **`reserveScheduleQoS`** — claims session gate slot **before** `GetRunner` wait; handlers `defer releaseQoS()` on all paths. **Why:** two streams passed defer checks within ~13ms and raced into the same MLX runner.
- **`project_id` / `project_name`** — parsed from `options.zerollama` (aliases: `client_id`, `project`, `client_name`); surfaced on `GET /api/ps` and **`zerollama ps`** PROJECT / SESSION columns. **Why:** fleet operators need “who owns this GPU?” without log grep.
- **`server/inference_path.go`** — backend detection (`mlx`, `gguf_ggml`, `gguf_llama_server`, `runtime`); `gateSessionKey` preserves GGUF client keys; unkeyed GGUF skips MLX interactive wait; `agentSessionMetadataEnabled` gates eliza injection. **Why:** MLX trie branch rewriting must not break llama-server L3 keys; CUDA batch jobs without session keys must not stall behind Mac agent cooldown.
- **MLX exit logging** — `slog.Error("mlx runner subprocess exited", …)` with exit code + stderr tail. **Why:** Jul 4 crash had no subprocess death line in `log`.
- **Version API** — `zerollama.capabilities.session_qos_gate`, `runner_paths`. **Why:** clients probe once and send Tier 2 hints only on zerollama nodes.
- **Clients** — Hermes (`hermes-lean`), ruby-trivia, simpleagent: detect zerollama via `/api/version`; send `project_id` / `qos_class` / `prompt_cache_key` progressively.

**Docs:** [docs/agent-qos-and-project-tracking.md](docs/agent-qos-and-project-tracking.md); [ROADMAP M15b](docs/ROADMAP.md#apple-silicon--metal-track).

### MLX agent live-session hardening (M15a, Jul 2026)

**Why:** Hermes on `gemma4:26b-optiq` showed turn-2 **99% `cached_tokens`** but **60–90s** TTFT (`fast_path` never logged), turn-3+ **~16k cached** when `messages_dropped` increased, and noisy `mlx_cache_warn` on short `/api/generate` sidecar calls. Trie restore worked; live KV rewind and prompt-chain alignment did not.

- **`tryExtendLiveSession`** — LCP vs prior prompt-only IDs; `trimPathToOffset` excludes gen past LCP; **`bestRestorableOffset`** picks largest snapshot boundary ≤ LCP; **`rewindCachesViaSnapshots`** page-in on active branch when rotating KV rejects `Restore(nil, …)`. **Why:** wrapped OptiQ windows cannot live-rewind 65k offsets; gen tokens in KV broke naive compare; blind mid-edge cap left 2k-token restore gaps.
- **`trySameBranchRestore`** — snapshot fallback after failed live rewind. **`same_branch`** on trie `cache hit` logs + `tunePrefillConfig` hot-tail path. **Why:** same as live session but trie-keyed — skips leaf page-out/in when branch unchanged.
- **`trieSnapshotInterval`** — agent + rotating KV: **1× sliding_window** (1024 OptiQ), was 2× (2048). **Why:** coarser snapshots explained ~16,384-token partial restores after message-level truncation changed the prefix.
- **Prompt chain** — invalidate on `messages_dropped > 0`; fingerprint check when message count unchanged; `prompt_chain_miss` `reason=messages_truncated`. **Why:** truncated history invalidates cached stable tokens; in-place message edits falsely spliced.
- **Observability** — `mlx_cache` agentstats: `same_branch`, `rewound_to`; KV-restore warns → Debug unless long prompt + session key. **Why:** operators could not separate live-session wins from trie hits or correlate drops with cache collapse.
- **MLX sidecar defer** — unkeyed `/api/generate` on MLX waits while a different keyed session is in-flight or within 90s post-turn cooldown (no max wait; bounded only by request context). Chat/OpenAI paths skip unkeyed defer so concurrent `stream=false` calls are not blocked behind agent cooldown. **Why:** background sidecar `begin()` clobbered live KV between agent turns; the old 2m cap forced unsafe proceed and unkeyed chat hit client 120s timeouts while waiting.

**Docs:** [docs/mlx-agent-prompts.md](docs/mlx-agent-prompts.md#m15a-live-session--restore-jul-2026); [ROADMAP M15a](docs/ROADMAP.md).

### Local image generation on Intel Arc A380 (Vulkan + OpenVINO) — Jul 2026

**Why:** MLX imagegen (Z-Image, Flux) needs **16 GB CUDA/Metal** and does not run on 6 GB Arc. Operators still need **local text-to-image** on the same zerollama API (`POST /api/generate`, `zerollama run`) without a second server. Two subprocess backends reuse the existing **`external-image` hook** — one global env for Vulkan sd.cpp, per-manifest wrapper override for OpenVINO so stacks coexist.

**Why two stacks on one GPU class:** stable-diffusion.cpp shares the ggml/Vulkan toolchain with llama.cpp (one operator mental model); OpenVINO GenAI ships Intel-tuned INT8 IR models that can beat sd.cpp on Arc for some checkpoints. Bench both via **`zerollama ls` PERF** after `zerollama bench sd15 --force`.

- **Vulkan (stable-diffusion.cpp)** — `sd15-vulkan`, `sd15-q8-vulkan`, `sd15-turbo-vulkan`, `sdxl-vulkan` (experimental); manifest `modality_backends.image=external-image`; `backend_paths.sd_cli` / `sd_model`; **`diffusion_fa: true`** required on Mesa ANV (noise without it).
- **OpenVINO GenAI** — `sd15-openvino`, `sdxl-openvino`; `modality_backends.image=openvino-image`; `backend_paths.ov_model_dir` + per-model `external_image_bin` → `ov_external_image.sh` (no global env change vs Vulkan).
- **`ImageGenerationConfig`** — per-model width/height/steps/cfg/sampler and SD flags in manifest `image_generation` (env `OLLAMA_SD_*` / `OLLAMA_OV_*` at subprocess boundary).
- **`zerollama ls`** — **PERF** column (was TOK/S): tok/s for chat, seconds for image/video; filter `zerollama ls image`; cloud **image** routes visible, completion-only Eliza stubs hidden.
- **`zerollama bench`** — image models: non-stream `/api/generate`, `TotalDuration` → `gen_sec`; capped at 2 timed epochs; **`effectiveMin` clamped** to cap so `--min-epochs 3` does not always fail on image tags; video_gen: `POST /v1/videos` poll.
- **Bench health / manifest search** — `envconfig.ModelsSearchDirs()` prefers `/usr/share/ollama/.ollama/models` when `OLLAMA_MODELS` unset so root/service installs match `zerollama bench` skips.
- **Scripts** — `install_stable_diffusion.sh`, `sd_external_image.sh`, `register_sd_models.sh`; `install_openvino_diffusion.sh`, `ov_image_generate.py`, `ov_external_image.sh`, `register_ov_models.sh`; `build_zerollama_a380.sh`, `install_a380_llama_server.sh`, `zerollama-a380.service`.
- **Doc:** [sd-vulkan-a380.md](docs/sd-vulkan-a380.md), [sd-openvino-a380.md](docs/sd-openvino-a380.md), [a380-runbook.md](docs/a380-runbook.md), [bench-cache.md](docs/bench-cache.md), [multimodal-backends.md](docs/multimodal-backends.md).

### Radix v2 track (L3-R2–L3-R5) — Jun 2026

**Why:** Cross-slot Radix v1 (donor→cold target) closed the main agent-fleet prefill gap but left four vLLM RadixAttention gaps: warm targets behind donors, overlapping block registrations, fleet metadata federation, and Gemma-style hybrid models. Each milestone adds admission-layer logic only — physical KV pages and cross-node blobs stay deferred.

Doc: [radix-prefix-share.md](docs/radix-prefix-share.md), [ROADMAP L3-R](docs/ROADMAP.md#radix-v2-l3-r--product-gaps).

### Radix warm-target catch-up (L3-R2) — Jun 2026

**Why:** Agent threads sometimes hold partial KV on a cache key while another slot already prefilled the full shared system prompt. v1 Radix skipped any target with `seq_pos > 0`, forcing redundant prefill on the tail.

- **`find_radix_share_plan`** — warm catch-up when donor matched > target `seq_pos`; `verify_target_slot_prefix` ensures target slot owns prefix block metadata.
- **`PrefixBlockPool.find_donor_slot_prefix`** — skip target-owned block entries while walking hash chain to find a longer donor.
- **`engine._prefix_cache_admission`** — Radix runs on warm slots; trace adds `warm_catchup` / `target_seq_pos_before`.

### Radix ref-count block DAG (L3-R3) — Jun 2026

**Why:** Two slots can register the same prefix block hash after independent prefills. v1 stored a single `slot_id` per entry — donor search picked the wrong chain when overlaps were partial (short donor vs full donor) and eviction could drop metadata another slot still needed.

- **`PrefixBlockEntry.holder_slots`** — ref-counted multi-slot holders per block hash; `release_slot_holders()` for slot teardown.
- **`_best_donor_from_chain`** — picks donor with longest contiguous prefix from token 0 across holder sets (warm skip segments still count).
- **Eviction** — prefer removing entries with zero holders; health exposes `multi_holder_blocks`.

### Remote LMCache tier (L3-R4) — Jun 2026

**Why:** Fleet nodes need a shared prefix block index after restart or on a cold node — local `file://` does not federate across hosts. KV blobs remain on each host's llama-server slot files; Redis carries **metadata only** until NIXL/Mooncake blob pull ships.

- **`redis://` LMCache backend** — `runtime/kv/lmcache_redis.py` (stdlib RESP GET/SET/PING; no redis-py dependency).
- **`ZEROLLAMA_LMCACHE_URI=redis://host:6379/0`** — optional **`ZEROLLAMA_LMCACHE_TTL_SEC`** for key expiry.
- **Block pool** — hydrates donor metadata from Redis on lookup (same contract as file tier).

### Hybrid-memory Radix (L3-R5) — Jun 2026

**Why:** v1 skipped all hybrid `seq_cp`, blocking Gemma-style SWA models whose copied prefix fits the coordinated window. True attn+recurrent memory (some LFM2 paths) can still abort `seq_cp` — operators keep `ZEROLLAMA_RADIX_HYBRID_SEQ_COPY=0` until live-probed.

- **`radix_seq_copy_policy.py`** — `radix_seq_copy_allowed(spec, plan)`; hybrid allowed when copy ≤ SWA window and `swa_allows_cache_prompt` passes.
- **`ZEROLLAMA_RADIX_HYBRID_SEQ_COPY`** — default on; set `0` for conservative skip on attn+recurrent probes.
- **Engine** — policy-driven skip reasons replace blanket `hybrid_memory_seq_cp_unsupported`.

### T6 unified queue — operator smoke + status (Jun 2026)

**Why:** T6 policy (idle-wait, defer queue, allowed window, cross-queue FIFO) was implemented but lacked a single operator runbook and regression smoke.

- **`docs/t6-unified-queue.md`** — env table, priority matrix, defer/window/FIFO, production checklist, code map.
- **`scripts/e2e/e2e_t6_queue_smoke.sh`** — offline Go + pytest; optional live `/api/status` + runtime `/health` fifo fields.
- **`GET /api/status`** — `inference.training.queue_policy` exposes configured T6 gates and live defer depth.

### Production serve wrapper (`~/bin/serve.sh`) — Jun 2026

**Why:** CT 1564 operators copied `scripts/serve/serve_gpu_example.sh` verbatim to `~/bin/serve.sh`. That script resolves repo root as `dirname(BASH_SOURCE)/..`, which becomes **`$HOME`** when the file lives in `~/bin` — not `~/zerollama`. Result: `sched_watchdog_env.sh`, `training_uv_venv.sh`, and `PYTHONPATH` never load; `zerollama serve` exits immediately or embed fails with `ModuleNotFoundError: uvicorn` while the operator sees an idle screen (logs go nowhere unless `SERVE_LOG` is set).

**Resolution:** keep **`serve_gpu_example.sh` in-repo only**; install **`scripts/serve/serve_production_wrapper.sh`** as `~/bin/serve.sh`. The wrapper sets `ZEROLLAMA_REPO` and `exec`s the in-repo example. **`serve_gpu_example.sh`** now resolves repo via `$ZEROLLAMA_REPO` / `~/zerollama` when not under `scripts/`, and auto-picks vendor `llama-server` when built (fork QJL + Radix `/kv/seq-copy`).

- **`scripts/serve/serve_production_wrapper.sh`** — thin `~/bin` installer; default `SERVE_LOG=/tmp/zerollama-serve.log`.
- **`scripts/serve/serve_gpu_example.sh`** — repo-root detection fix; vendor `LLAMA_SERVER_BIN` when `vendor/llama-cpp-*` is built.
- **Doc:** [5080-runbook.md](docs/5080-runbook.md#production-serve-binserve-sh), [gpu-5080-operator-guide.md](docs/gpu-5080-operator-guide.md#production-serve-binserve-sh), [gpu-training.md](docs/gpu-training.md#installing-python-deps-embedded-interpreter), [runtime-embed.md](docs/runtime-embed.md).

### Training venv ABI + serve scripts (Jun 2026)

**Why:** CT 1564 operators saw `training worker not started` when `.venv-training` was built for Python 3.11 but the installed `zerollama` embeds `libpython3.10`. PyTorch wheels are ABI-specific; pointing `PYTHONPATH` at the wrong `pythonX.Y/site-packages` fails before torch imports. Legacy repo-root `venv-training/` and pkg-config-only detection made serve scripts pick the wrong path or version.

**Resolution on 5080/CT 1564:** rebuild `zerollama` with **`libpython3.11`** (matches `runtime/.venv` and uv defaults), keep a single `.venv-training` on 3.11, delete legacy 3.10 trees (~15 GiB).

- **`scripts/training/training_uv_venv.sh`** — always uses `$REPO/.venv-training` (drops auto-pick of legacy `venv-training/`); `embedded_training_python_ver()` prefers `ldd zerollama` over `pkg-config python3-embed`; new `--embed-py` flag for serve/5080 scripts.
- **`scripts/training/training_embed_build_env.sh`** — PKG_CONFIG overlay so `go build` links `libpython3.11` (or chosen version) when distro `python3-embed` is still 3.10. **`5080_build_zerollama`** sources it when `python-3.11-embed` exists.
- **`scripts/serve/serve_gpu_example.sh`** — in-repo production env only; **`scripts/serve/serve_production_wrapper.sh`** → `~/bin/serve.sh` (do **not** copy the example verbatim — breaks `_ROOT`). Sets `TRAINING_UV_SITE_PACKAGES` from linked libpython before `zerollama serve`; auto-run `training_uv_venv.sh --verify` when site-packages missing.
- **`x/trainingworker/pyembed/training_shim.c`** — clearer operator error: `embedded Python X.Y requires .venv-training/lib/pythonX.Y/site-packages`.
- **`.gitignore`** — `venv-training/` (legacy duplicate venv; only `.venv-training/` is canonical).
- **Doc:** [gpu-training.md](docs/gpu-training.md) (ABI matching, 3.11 build, cleanup), [5080-runbook.md](docs/5080-runbook.md), [development.md](docs/development.md) (Linux CGO embed build).

### ANE in-process dflash draft — B1–B6 lab track (Jun 2026)

**Why:** maderix ANE bridge requires **IOSurface in the same PID** as ggml Metal; subprocess draft daemons proved compile-once scheduling but cannot hand off activations to llama-server. In-process hook on **dflash speculative decode** validates map+eval latency and sidecar weight extract **before** routing draft tokens from ANE (B7+). **Why not production `:11434`:** hook adds measurable e2e overhead and draft tokens remain Metal ggml until full subgraph MIL lands.

- **B1 in-process ANE session** — `ane_draft_session.mm` compile-once + IOSurface in llama-server; links `libane_bridge` when `ANE_REPO/bridge/libane_bridge.dylib` exists; stub otherwise. **Why:** eliminates cross-process `IOSurfaceLookup` failure.
- **B2 ggml IOSurface handoff** — `common_ane_draft_handoff_after_decode()` packs draft pre-norm hidden via `ggml_backend_dev_buffer_from_iosurface()` after decode. **Why:** same bytes as Metal activations without CPU copy; enables real sidecar-driven input.
- **B3 sidecar weight bundle** — `MaterializeANEDraftWeightBundle()` extracts `ffn_gate` conv proxy + norm gamma from eliza drafter GGUF (Q8_0 dequant for 27B); gamma on host pack because MIL broadcast mul failed. **Why:** ANE MIL needs BLOBFILE weights from real checkpoints, not synthetic smoke only.
- **B4 A/B bench** — `zerollama ane-draft-ab-smoke --e2e` micro ANE step vs Metal-only dflash on **`ZEROLLAMA_ANE_LAB_PORT` (11435)**. **Why:** quantify hook overhead without touching daily serve.
- **B5 per-step handoff** — handoff on every draft `llama_decode` (not once). **Why:** matches speculative loop; step telemetry for parity work.
- **B7 subgraph expansion** — manifest v3: `ffn_up` second conv via **dual conv1 kernels**; GGUF FFN weights transposed to ANE `[out,in]` conv layout; maderix blob header fields (`wsize@72`, `payload@128`); golden CPU ref cosine **1.0** vs ANE; cached IOSurface + gamma; `--e2e-telemetry` opt-in.
- **B7 draft token drive (lab)** — `ZEROLLAMA_ANE_DRAFT_DRIVE=shadow|force` routes ANE conv output → host tied-embed argmax → draft token ID (`force` replaces Metal sampler). Requires manifest v4 `token_embd` cache; use `ZEROLLAMA_ANE_DRAFT_DRIVE_VOCAB_CAP=8192` for fast lab argmax.
- **CLI** — `ane-draft-mil-bundle`, `ane-draft-mil-map`, `ane-draft-ab-smoke`, `ane-inprocess-smoke` (existing), export env in bundle JSON.
- **Build** — `scripts/vendor/sync_ane_hook_to_llama_cpp.sh`, `build_llama_server.sh` copies `libane_bridge.dylib`; **`install_name_tool`** on `libllama-common` for `@loader_path/libane_bridge.dylib`. **Why:** dyld fails if install name left as bare `libane_bridge.dylib`.
- **Doc:** [docs/ane-draft-inprocess.md](docs/ane-draft-inprocess.md); updated [ane-hybrid-path.md](docs/ane-hybrid-path.md), [ane-ggml-iosurface-hook.md](docs/ane-ggml-iosurface-hook.md).

**Not in scope:** ANE-driven draft token IDs; full `dflash_fc` MIL; vendor-only tree without sibling rebuild; enabling hook on production serve by default.

### L3 env consolidation (Jun 2026)

**Why:** operator shells leaked flags (e.g. `ZEROLLAMA_LLAMA_CACHE_DISK=0` from Metal) into unrelated CUDA smokes; too many redundant L3 toggles.

- **`runtime/runtime/env.py`** — centralized tri-state env parsing + `configure_runtime_env()` from engine init (backend, `n_parallel`); YAML **`l3:`** block (`radix_share`, `block_size`, `trace`, `lmcache_uri`).
- **`runtime/configs/l3_agent_subprocess.yaml`** — example multi-slot agent profile (Radix via YAML, not six env vars).
- **`resolve_llama_cpp_root()` / `resolve_llama_server_bin()`** — prefer vendor pin over stale bare sibling `LLAMA_CPP_ROOT`; smokes use resolver instead of manual exports.
- **`ZEROLLAMA_LLAMA_CACHE_DISK`** — smart default when unset: off on Darwin, on for Linux subprocess; explicit `0`/`1` still wins.
- **`ZEROLLAMA_PREFIX_BLOCK_POOL`** — auto-on when L3 + `n_parallel > 1`, Radix share, or LMCache URI; `=0` to disable.
- **`ZEROLLAMA_LMCACHE_URI`** — URI alone enables tier; `ZEROLLAMA_LMCACHE_TIER=1` deprecated alias.
- **Migrated to `env.py`:** `PREFIX_CACHE_BLOCK_SIZE`, `PREFIX_CACHE_TRACE`, `DECODE_GRAPH_*`, retention interval, cache root/TTL/salt, infer trace.
- **`ZEROLLAMA_L3_PROFILE=agent`** — loads `l3_agent_subprocess.yaml` when `ZEROLLAMA_RUNTIME_CONFIG` unset.
- **`ZEROLLAMA_DEBUG=l3|infer`** — enables prefix trace / infer trace without separate flags.
- **`/health.llama_cache.runtime_env`** — effective L3/env snapshot for operators.
- **KV env helpers** — `ZEROLLAMA_KV_NATIVE_*` / `ZEROLLAMA_KV_AUTO_BATCH*` centralized in `env.py`.
- **`./scripts/runtime/runtime_env_doctor.sh`** — offline effective env report (no server).
- **`./scripts/vendor/llama_patch_doctor.sh`** — patch file + in-tree + vendor commit count + resolved llama-server binary checks; fails on stale sibling builds missing `/kv/seq-copy`. **Why:** patches live in git but binaries often come from unpatch `../llama.cpp`.
- **`LLAMA_CPP_VENDOR_HEAD`** — tracked expected vendor git SHA after full patch apply; doctor + patch doctor warn on drift.
- **`/health.llama_patches`** — compact patch doctor snapshot on live sidecar.
- **`zerollama doctor`** — `llama.cpp patches` check (in-tree seq-copy, patch files, binary probe).
- **`build_llama_server.sh`** — fails vendor builds when binary lacks `/kv/seq-copy`.
- **VRAM env migration (complete)** — all `ZEROLLAMA_RUNTIME_VRAM_*`, RAM overhead/margin, weight-tensor auto mode, clamp/suggest/probe-calibrate/autotune/persist, GGUF layout flags, and `INFERENCE_POLICY` / `SHARED_PYTHON` read via `env.py`; consumers delegate (`gpu_vram`, `host_memory`, `vram_suggest`, `vram_calibration`, `vram_autotune_persist`, `vram_env_apply`, `gguf_estimate`, `gpu/inference_policy`). Expanded `/health.llama_cache.runtime_env.vram`.
- **Doc pin sync** — `c84b3020` / elizaOS unified as current pin in ggml migration + Phase 17 docs (supersedes b9781 tables).
- **`runtime/cli.py` fix** — `serve` used `resolve_default_config_path()` (Darwin → `apple_silicon.yaml`) and ignored `ZEROLLAMA_L3_PROFILE=agent`; Radix live smokes now load `l3_agent_subprocess.yaml`. **Live gate (Jun 2026):** donor slot 0 → target slot 2, `radix_seed` 128 tokens, target wall **0.52s** vs donor **8.83s**, `POST /kv/seq-copy` OK on vendor llama-server.
- **Doc:** [docs/runtime-env.md](docs/runtime-env.md) — operator env reference (profiles, L3, KV, VRAM).

### L3 prefix cache — hybrid coordinator, block pool, LMCache tier (Jun 2026)

**Why:** vLLM selective retention + block-hash prefix verification closes correctness gaps on SWA/hybrid models and detects prompt drift before stale KV reuse.

- **Hybrid KV coordinator** — `runtime/runtime/kv/hybrid_kv_coordinator.py`; per-layer full/SWA groups; coordinated `cache_prompt` gate via min SWA window on hybrid GGUF layouts.
- **Hash-chained prefix block pool** — `ZEROLLAMA_PREFIX_BLOCK_POOL=1`; `kv/prefix_block_hash.py` + `kv/prefix_block_pool.py`; denies reuse on `prefix_block_hash_mismatch`.
- **Optional LMCache tier** — `ZEROLLAMA_LMCACHE_TIER=1` + `ZEROLLAMA_LMCACHE_URI=file://…`; filesystem metadata tier hydrates block index after restart.
- **Cross-slot Radix sharing** — `ZEROLLAMA_RADIX_PREFIX_SHARE=1`; donor slot KV seed via hash chain + `llama_memory_seq_cp` / `POST /kv/seq-copy`. **Why:** L3 pins one slot per cache key; agents sharing a system prompt but different keys otherwise repeat prefill.
- **Radix operator polish** — `record_radix_share` trace (`radix_seed`); `SubprocessSlotState.seed_seq_pos()`; `./scripts/phase/l3_radix_prefix_smoke.sh` (`L3_RADIX_LIVE=1` forces **vendor** llama-server); decode-graph reason `radix_cross_slot_seed`; patch **`0017-ollama-kv-seq-copy-endpoint`**; hybrid models skip `seq_cp` in engine. **Why vendor binary:** bare sibling `../llama.cpp` lacks `/kv/seq-copy`; stale `LLAMA_CPP_ROOT` in shell was a common live-smoke footgun.
- **Health/trace** — `/health.llama_cache.prefix_block_pool`, `/health.kv_resume.prefix_block_pool`; trace rows include `prefix_block_matched_tokens` and `radix_seed` events.
- **Smoke** — `./scripts/phase/l3_prefix_block_pool_smoke.sh`; `./scripts/phase/l3_radix_prefix_smoke.sh` (offline + optional live).
- **Doc** — [docs/radix-prefix-share.md](docs/radix-prefix-share.md).

### Radix product gaps — documentation (Jun 2026)

**Why:** v1 Radix shipped without a published gap matrix; operators comparing to vLLM RadixAttention needed explicit scope (cold target only, no ref-count DAG, no remote tier, hybrid skip).

- **`docs/radix-prefix-share.md`** — **Product gaps** section: scope table, validation status (Mac live PASS, 5080 pending), operator checklist, roadmap pointers.
- **`docs/ROADMAP.md`** — **Radix v2 (L3-R)** milestones L3-R0…L3-R5 with WHY + exit criteria.
- **`docs/gpu-profiles-l3.md`** — Radix live gate row in sign-off table; link to product gaps.
- **README** — expanded WHY for Radix vs same-key L3; explicit non-goals pointer.
- **Code comments** — `radix_prefix_share.py`, `prefix_block_pool.py`, `env.radix_prefix_share_enabled`, `engine._apply_radix_prefix_share` — WHY cold-only, block pool first, hybrid skip.

### llama.cpp pin bump — `b9781` (upstream Ollama v0.30.11)

**Why:** Upstream Ollama advanced from v0.30.10 (`07ed7523` / `b9672`) to **v0.30.11** (`32a97b74` / **`b9781`**). Staying on `b9672` widened the cherry-pick gap for Phase 17 (`llm/llama_server.go`, discovery, native chat on `/api/generate`, CUDA/Vulkan fixes). Rebasing the **vendor patch quilt** onto `b9781` keeps mergeability without replacing Mac ggml Metal default (~+7% decode vs upstream on M4 Max).

**Why rebase patches onto the pin tag, not llama.cpp HEAD:** zerollama ships **16 ordered format-patches** on a fixed tag. Rebasing onto moving upstream HEAD would rewrite the entire series every week; rebasing onto upstream Ollama’s pin matches vanilla release cadence and keeps Phase 15 kv-ext diffs reviewable.

- **`LLAMA_CPP_VERSION=b9781`**, **`Makefile.sync`** → `vendor/llama-cpp-b9781`, `FETCH_HEAD=b9781`.
- **Vendor:** all **16** Ollama patches applied on `b9781` (vendor HEAD `b10675c` after manual conflict resolution on **0010** GPU discovery, **0012** mtmd C API, **0015** compat clip loader).
- **In-tree sync:** `./scripts/vendor/sync_vendor_llama.sh` — rsync patched vendor → `ml/backend/ggml/ggml` + `llama/llama.cpp`; regen `build-info.cpp` + Metal embed.
- **`Makefile.sync` fix:** `make sync` no longer runs `git checkout` on vendor (that reset wiped patch commits before rsync). **Why:** operators reported pin `b9781` in build-info while CGO still built bare upstream ggml.
- **`sync_vendor_llama.sh` guard:** aborts if vendor has zero commits on top of `FETCH_HEAD`. **Why:** fail fast instead of silent unpatch sync.
- **Doc:** [docs/ggml-b9509-migration.md](docs/ggml-b9509-migration.md) (pin table, patch conflicts, workflow).

**Patch apply notes (b9781):** `git am` fails when format-patches lack blob SHAs (`index` lines) after upstream file moves — use `git apply --reject` + manual hunks for **0010** (`ggml-cuda.cu` struct/NVML), **0012** (`mtmd_progress_callback` inserted in b9781), **0015** (`clip.cpp` tensor load loop). **Why no `device_mutex` on b9781:** upstream removed it; NVML/HIP memory reporting applies without the lock.

### Upstream Ollama v0.30.11 — Go port (excluding Claude/OpenCode auto-install)

**Why:** v0.30.11 ships renderer/parser fixes, CUDA driver 550+ compat, Vulkan Windows discovery, native chat on `/api/generate`, and MLX speculative-decode refactors zerollama needs for merge parity. **Explicitly skipped:** Claude/OpenCode **auto-install** launchers, Kimi, desktop app launchers, wholesale Python removal, Mac-default llama-server flip — different product choices documented in [upstream-ollama-diff.md](docs/upstream-ollama-diff.md).

- **Server:** `chatExecutionModeNative`, `prepareNativeChatRequest`, `truncateNativeChatMessages`, `imageTaggedMessages`; `Model.HasChatTemplate` / `HasGoTemplate`; `ApplyChatTemplate` on `llm.LlamaServer`.
- **Discover:** Vulkan Windows fix; **`discover/cuda_compat.go`** — driver 550+, Jetson, sm_86 gating for flash-attention. **Why:** upstream gates FA from device metadata; ggml path still needs accurate props after b9509 API trim.
- **llm:** `ps` mmap accounting; context-shift headroom; mmproj offload sizing; **`llm/vulkan_windows.go`**.
- **Models:** Ornith 9B parser/renderer; Qwen35 renderer tweak.
- **MLX:** speculative decode refactor (`speculate.go`, depth helpers); gemma4 updates; `NewClient(modelName, numCtx)`; preserved zerollama tokenize cache + `PromptTokens` passthrough. **Why:** upstream MTP budget checks need exact token IDs after tail-truncate.
- **Imagegen:** upstream `compile.go`; `keepDuringRead` / `ReleaseAll` in `mlx.go`.
- **Launch:** Hermes official install URLs only (no Claude/OpenCode auto-install).
- **Convert / tools:** JSON brace fix; qwen25vl converter.
- **Compat:** `llama/compat/001-llama-cpp-hooks.patch` refreshed from upstream.

**Not ported:** `codex_app`, Kimi launch, desktop launchers, full upstream rebase, deleting `runner/ollamarunner`.

### Flash-MoE (anemll) + ANE probe (maderix) — experimental Mac inference upgrades

**Why:** Scouting [@anemll](https://x.com/anemll) and [@maderix](https://x.com/maderix) surfaced two complementary Apple Silicon paths: **Flash-MoE** runs MoE models **larger than RAM** by streaming routed experts from SSD (immediate operator value via Phase 17 llama-server), while **ANE probe** validates private Neural Engine access for a **future hybrid** (embeddings / vision front-ends) without touching the ggml Metal hot path.

**Flash-MoE — why llama-server only, not ggml default:** anemll's slot-bank runtime lives in a forked `llama-server` with `--moe-*` flags; Mac ggml stays ~+7% faster for in-RAM models (M7). Passthrough via existing Go→llama-server seam is the smallest correct integration.

- **`envconfig/flash_moe.go`** — `ZEROLLAMA_FLASH_MOE*`, `FLASH_MOE_REPO`, `FlashMoELlamaServerBin()`; **why:** centralize operator knobs and surface in `zerollama envconfig`.
- **`llm/flash_moe.go`** — `appendFlashMoEArgs`, `FindFlashMoELlamaServer`, sidecar-gated activation; auto **`-fit on`** + **`-ub 1`**; **why:** anemll documents these as required for MoE prefill correctness and dense/slot VRAM balance.
- **`api/types.go`** — `moe_mode`, `moe_sidecar`, `moe_slot_bank`, `moe_topk`, `moe_prefetch_temporal` on runner options; **why:** per-model Modelfile config without global env.
- **`discover/flash_moe_inventory.go`** + **`zerollama flash-moe-resolve`** — scan `~/.ollama/models` for MoE tags, blob paths, manifest `moe_sidecar`, default `~/Models/flash/<tag>`; **why:** smoke/operators should not hand-copy GGUF paths zerollama already knows from `pull`.
- **`scripts/gpu/flash_moe_extract_sidecar.sh`** — operator wrapper for `flashmoe_sidecar.py extract` + verify; prints serve env hints.
- **`scripts/build/build_flash_moe_llama_server.sh`** — builds `anemll-flash-llama.cpp` → `build/flash-moe-llama-server-darwin/bin/llama-server`.
- **`scripts/phase/flash_moe_smoke.sh`** — tier 0 toolchain (go tests + `--moe-sidecar` binary); tier 1 startup; tier 2 zerollama E2E; **why:** MoE sidecar/GGUF are operator-local — CI validates wiring without 100GB fixtures.
- **`cmd/doctor_flash_moe.go`** — doctor check reports repo/binary readiness even when disabled; **why:** operators should see "binary ready" before setting env vars.
- **Doc:** [docs/flash-moe.md](docs/flash-moe.md).

**ANE probe — why subprocess, not CGO:** maderix `libane_bridge.dylib` uses private `_ANEClient` APIs that break across macOS updates; isolating compile/eval in `ane-probe` keeps `zerollama` stable and lets doctor warn without blocking serve.

- **`tools/ane-probe/`** — minimal ObjC MIL conv smoke test; `@loader_path` dylib + `install_name_tool`.
- **`discover/ane_probe.go`** — find/run probe, JSON parse; `cmd.Dir` set for dyld.
- **`cmd/doctor_ane.go`** + hidden **`zerollama ane-probe`**; **`scripts/ane/ane_probe_build.sh`**, **`scripts/ane/ane_probe_smoke.sh`**.
- **`envconfig/ANERepo()`**, **`internal/reporoots`** — shared checkout discovery for local build artifacts.
- **Doc:** [docs/ane-probe.md](docs/ane-probe.md).

**Not in scope:** Flash-MoE in ggml Metal; automatic sidecar extract on `pull`; ANE on inference hot path; CUDA Flash-MoE build script.

### Model bench cache (`zerollama bench` + `TOK/S` in `ls`)

**Why:** Parameter size and disk footprint in `zerollama ls` do not tell operators which local tag decodes fastest on *their* GPU/backend. A lightweight client-side bench avoids N× manual `run --verbose` sessions and keeps numbers visible in daily `list` output without server or manifest changes.

- **`zerollama bench [MODEL...]`** — warmup + averaged `/api/generate` epochs; reports decode tok/s from `EvalCount` / `EvalDuration`; **why:** same HTTP path agents use, reflects live serve policy (ggml vs llama-server).
- **`~/.ollama/bench.json`** — keyed by manifest **digest** (not name); atomic write after each model; **why:** re-pull invalidates stale entries; Ctrl-C mid-run keeps completed models.
- **`zerollama ls`** — **TOK/S** column (`--` when not benchmarked); soft-fail load; **why:** list must never break when cache is missing.
- **`cmd/benchcache/`** — load/save helper shared by bench + list.
- **Filters** — skips remote catalog stubs (except LM Studio), embedding/image/video_gen/speech-only tags; **why:** generate bench is meaningless on non-completion models.
- **Doc:** [docs/bench-cache.md](docs/bench-cache.md).

**Not in scope:** auto-bench on pull; prefill/TTFT in `ls`; fleet-wide bench aggregation (use `cmd/bench/bench.go` or L1 gates for CI).

### Phase 16 — thin edge daemon (v0–v2)

**Why:** Upstream Ollama’s default GGUF path is **Go → llama-server** with no Python chat middleman. Zerollama keeps training, Eliza, fleet, and Mac ggml speed as differentiators, but operators deploying upstream-shaped Linux/edge nodes need a **thin Go daemon** that routes all GGUF chat/generate through llama-server and turns off runtime chat by default — without forking the API surface.

- **`zerollama serve --edge`** / **`ZEROLLAMA_EDGE=1`** — forces `ZEROLLAMA_LLAMA_SERVER=1`, `ZEROLLAMA_RUNTIME=0`, and legacy-runner routing; **why:** one flag for “upstream-shaped edge” without hand-tuning five env vars.
- **`server/edge_ggml_policy.go`** — `schedSkipGgmlRunnerLoad` returns HTTP 400 when edge policy would spawn ggml; **why:** fail fast with a clear operator message instead of hanging on a runner that will never load.
- **`server/inference_backend_policy.go`** + **`GET /api/status`** — `inference.backend` snapshot (`edge`, `edge_build`, `ggml_linked`, `llama_server`, `gguf_path`, `runtime_chat`); **why:** fleet ops and smokes need one JSON probe, not log archaeology.
- **`GET /api/version`** + **`zerollama -v`** — `edge_build` compile marker; **why:** CI and operators must distinguish `-tags edge` binaries from runtime `--edge` alone.
- **`-tags edge` (v1)** — `GgmlRunnerLinked=false`, stubs `zerollama runner` subprocess, `ggmlRunnerRequired` fail-fast without llama-server; **`build_zerollama_edge.sh`**, **`phase16_edge_build_smoke.sh`** (CI regression).
- **`-tags edge` (v2)** — `llm/server.go` + `server_score.go` excluded (`//go:build !edge`); **`llm/server_edge.go`** llama-server-only `NewLlamaServer`; edge main dep tree drops `llama`/`model` CGO; **why:** smaller edge binary and no accidental ggml load paths at link time.
- **`-tags edge` (v3)** — `RuntimeEmbedEnabled` / `RuntimeDarwinSidecarLikely` false when `EdgeBuildTag`; **`phase16_edge_binary_smoke.sh`** E2E on edge artifact without `--edge` flag; **why:** deployed edge binaries must not spawn Python runtime even if operator sets `ZEROLLAMA_EDGE=0`.
- **`discover/gpu_discovery_upstream.go`** — skip ggml bootstrap when `!GgmlRunnerLinked()`; **why:** edge discovery must use llama-server probe, not spawn a ggml runner subprocess that is stubbed out.
- **Scripts:** `serve_edge.sh`, `serve_linux_auto.sh`, `phase16_edge_smoke.sh`; **`RUN_E2E_UPSTREAM_GGUF=1`** bundles P17 + P17_LINUX_AUTO + EDGE on 5080 session.
- **Doc:** [docs/phase16-thin-edge.md](docs/phase16-thin-edge.md).

### Phase 17 — upstream GGUF path alignment (operator + CI)

**Why:** Cherry-picking upstream’s Go→llama-server integration reduces merge pain and removes Go→Python→llama hops for default text GGUF on Linux — while Mac keeps ggml Metal (~+7% decode on M4 Max).

- **Linux auto-default** — plain `zerollama serve` sets `ZEROLLAMA_LLAMA_SERVER=auto` when llama-server is on disk; routes **all** GGUF (text + vision + thinking); **`serve_linux_auto.sh`**, **`phase17_linux_auto_smoke.sh`**.
- **Explicit opt-in on Mac** — `--llama-server-backend` / `ZEROLLAMA_LLAMA_SERVER=1`; Mac default unchanged (M7 bench).
- **`ErrEdgeGgmlRunnerDisabled`** remedy text — points to llama-server env, not `--edge` (which *enables* the restriction); **why:** operators were told to use the mode that caused the error.
- **`phase17_l2_pin_status.sh`**, **`phase15_upstream_kv_watch.sh`** — JSON pin/L2/upstream-KV reports for CI regression.
- **Doc:** [docs/phase17-llama-server.md](docs/phase17-llama-server.md).

### Build / CI fixes

**Why:** `go build ./...` must succeed on dev Macs; smoke scripts must not false-fail when subprocesses correctly exit non-zero.

- **`app/webview/webview.go`** — Darwin LDFLAGS `-lc++`; **why:** header-only webview C++ compiled without libc++ link caused undefined `std::` symbols on current Xcode toolchains.
- **`x/mlxrunner/mlx/generator/main.go`** — `//go:build ignore`; **`go generate -tags ignore`**; **why:** codegen tool pulled incomplete vendor tree-sitter into `./...` builds.
- **`scripts/phase/phase16_edge_build_smoke.sh`** — capture `zerollama -v` and runner stderr before `grep` (no pipes under `pipefail`); **why:** `grep -q` closes early (SIGPIPE 141) and stubbed runner exits non-zero — both caused false FAILs despite correct output.

### Mac Phase 11 → 13 → 15 sign-off (Jun 2026)

**Why:** CUDA `gpu_5080_session.sh` validates Phase 11 admission and Phase 13 VRAM on discrete NVML; Mac operators need the same evidence on **unified memory** (metal-unified probe, `apple_silicon.yaml` defaults, in-process Phase 15 KV) without copying the 5080 playbook.

- **`./scripts/phase/phase11_13_15_metal_signoff.sh`** — ordered gate: Phase 11 → 13 → CPU `phase15_kv_native_ci` → optional `phase15_metal_signoff` + upstream KV watch; **`METAL_SELF_START=1`** bootstraps sidecar `:8081` + Go `:8080`.
- **`phase11_metal_admission_smoke.sh`** — admission/inference-policy pytest + live `e2e_coordination_smoke` + `/health` admission fields (`vram_min_free`, training reserve, `metal-unified`).
- **`phase13_metal_vram_smoke.sh`** — VRAM suggest/autotune pytest + live `/internal/vram-estimate` + `gpu_phase13_snapshot.sh`.
- **`macos_export_llama_cpp_paths()`** — prefer **`vendor/llama-cpp-<pin>`** over bare sibling `../llama.cpp` for kv-ext symbols + linked `_kv_native`; wired into Mac serve lib and Phase 15 sign-off.
- **`phase15_runtime_kv_env.sh`** — `rm -rf build` before GPU linked ext build; export `LLAMA_CPP_LIB`; **why:** CPU CI’s unlinked `_kv_native` must not be copied over when decode loop links libllama.
- **`runtime/native/kv_decode_loop.c`** — weak `llama_context_cuda_graph_invalidate` no-op when patched libllama lacks the symbol; **why:** vendor Metal builds link decode loop without zerollama’s CUDA graph hook.
- **Smoke heredoc fixes** — Phase 11 (`HEALTH_JSON`) and Phase 13 (`ESTIMATE_JSON`) pass JSON via env, not `python3 <<'PY' "$json"` (treated as script path).
- **`macos_runtime_serve_lib.sh`** — `/api/tags` wait uses 15s curl timeout + `MACOS_GO_TAGS_MAX` (default 30); **why:** LM Studio model sync can exceed 2s and false-fail bootstrap.
- **Test fixes** — `test_generate_l3_second_turn_passes_current_pos` (four `_decode_current_pos` mocks); `trace_path()` invalidates when trace dir changes; `test_prefix_block_pool` resets runtime env hints; **`runtime/env.py`** — repaired corrupted `configure_runtime_env` merge.
- **Sign-off (M4 Max):** `./scripts/phase/phase11_13_15_metal_signoff.sh` PASS — Phase 11 (56 pytest + live), Phase 13 (64 pytest + live estimate), Phase 15 CI (248 pytest), Phase 15 Metal (5 steps incl. `batch_decode_in_c=True`).

**Not in scope:** replacing `metal_signoff.sh` / qwen35 gate; CUDA Phase 11/13 tuning on 5080 (still `gpu_5080_session.sh`).

### M9 `metal_signoff.sh` full gate — eliza-1 qwen35 + linked tensor bind (Jun 2026)

**Why:** the ordered Phase 11→13→15 script validates admission/VRAM/KV in isolation; operators still need one **daily Mac gate** that runs M3 + optional qwen35 Go ggml + Phase 15 on the same Metal device. After vendor `llama-kv-ext` linked builds, GPU smokes must accept **`status=bound` + `bind_level=tensor`** — not only pre-ext **`partial` + `seq_position`** — or sign-off false-fails despite a healthy decode path.

- **`./scripts/gpu/metal_signoff.sh`** — M4 Max **PASS** (~2.6 min): coordination → Phase 13 snapshot → Phase 14 inprocess → qwen35 → Phase 15 (5 steps incl. `batch_decode_in_c=True`, tensor bind probe).
- **Canonical qwen35 tag:** **`eliza-1-2b:latest`** via `RUN_E2E_QWEN35_MODEL=…` — **why:** `eliza-1-*` is the ship qwen35 family in this repo (2B is fast enough for CI handoff/resume); `qwen3.6:latest` remains valid but heavier and not required for the gate.
- **`smoke_runtime_assert_kv_snapshot()`** — accepts `partial`+`seq_position` **or** `bound`+`tensor`/`physical`/`seq_position` on `/health` and `partial`/`bound` on `/internal/kv-snapshot`; v33+ allows `physical_pages_bound` when writable page-map is linked.
- **Vendor libllama:** `macos_export_llama_cpp_paths()` → `vendor/llama-cpp-c84b3020` for linked `_kv_native`; weak CUDA graph invalidate in `kv_decode_loop.c` for Metal vendor builds without zerollama’s CUDA hook.
- **Doc sync:** ROADMAP M9/M10, README Mac tier-2, [apple-silicon-metal.md](docs/apple-silicon-metal.md), [testing-smoke.md](docs/testing-smoke.md).

**Example (text sign-off blob + eliza qwen35):**

```bash
M3_LLAMA_MODEL=~/.ollama/models/blobs/sha256-… \
RUN_E2E_QWEN35=1 RUN_E2E_QWEN35_MODEL=eliza-1-2b:latest \
./scripts/gpu/metal_signoff.sh
```

### 5080 operator runbook (Jun 2026)

**Why:** Mac sign-off is consolidated in `metal_signoff.sh`; CUDA operators had the same story split across gpu-5080-operator-guide, L1/L3 profiles, Phase 15, and Phase 17 docs — easy to miss **`RUN_E2E_UPSTREAM_GGUF=1`** (open Phase 16/17 gate on ship hardware).

- **`docs/5080-runbook.md`** — ordered tiers 0–5: base `gpu_5080_session.sh` → L1/L3 production → Phase 15 → upstream GGUF bundle; `RUN_E2E_*` table; CT 1564 status matrix; full re-sign-off script block.
- **Cross-links** — [gpu-5080-operator-guide.md](docs/gpu-5080-operator-guide.md), [docs/README.md](docs/README.md), [README.md](README.md), [ROADMAP.md](docs/ROADMAP.md) operator ladder.

### 5080 full re-sign-off PASS — CT 1564 (Jun 28 2026)

**Why:** Mac `metal_signoff.sh` closed M9; CUDA needed the same evidence on ship hardware after `fd7042bc` + NVML 590.48.01 fix + embed `PYTHONPATH`/`LD_LIBRARY_PATH` hygiene.

- **Tier 1–3 PASS** — `gpu_5080_session.sh`, L1 concurrent **+~16–20%** / single-stream **−5%** @ 8k on eliza-1 9B, L3 8k + 27k, `phase15_inprocess_signoff.sh`.
- **Tier 4 PASS (individual smokes)** — `phase17_llama_server_smoke.sh`, `phase17_linux_auto_smoke.sh`, `phase16_edge_smoke.sh` (`P17_NUM_PREDICT=32`); artifacts under `/tmp/phase17-*`, `/tmp/phase16-edge-smoke.json`.
- **Still open:** L2 fork merge @ 8k/27k (stock wins — fork profiles opt-in). Radix live and `RUN_E2E_UPSTREAM_GGUF=1` bundle PASS on 5080 Jun 2026.
- **Doc:** [5080-runbook.md](docs/5080-runbook.md) — now **self-contained** (serve startup, sm_120 libllama build, troubleshooting); operator guide = extended reference.

### 5080 runbook self-contained expansion (Jun 2026)

**Why:** Re-sign-off on CT 1564 exposed a gap — the runbook listed tiers but said “serve must already be up” without copy/paste serve env, and duplicated footguns lived only in the operator guide.

- **`docs/5080-runbook.md`** — added: Proxmox/CT layout; cpp-httplib / `-lstdc++` fixes; **Build patched libllama (sm_120)**; **Start serve** (PYTHONPATH, health wait, `serve_gpu_example.sh` pointer); **Troubleshooting** table; **After green re-sign-off**; full sequence no longer uses `# … start serve …` placeholder.
- **`docs/gpu-5080-operator-guide.md`** — defers primary re-sign-off path to runbook; remains MLX/L2/code-map reference.

### Upstream Ollama — launch model inventory (v0.30.10)

**Why:** Each integration used to call `/api/show` per model when writing agent configs — slow, easy to timeout, and inconsistent with the model picker (`/api/tags`). Upstream loads tags once per launch run and passes rich `LaunchModel` structs to every integration. Zerollama ports that pattern for mergeability and faster `zerollama launch`.

- **`cmd/launch/model_inventory.go`** — `LaunchModel`, per-run `modelInventory` (load/refresh/resolve); local miss triggers refresh after `pull`.
- **`cmd/launch/launch.go`** — `resolveRunModels`, `liveConfigMatches` drift guard; managed catalog uses inventory only (not picker recommendations).
- **`api/types.go`**, **`server/routes.go`**, **`server/model_details.go`** — list tags carry `Capabilities`, context/embedding length from GGUF kv.
- **Integrations** — `Edit`/`Run`/`ConfigureWithModels` take `[]LaunchModel`; OpenCode, OpenClaw, Pi, Droid, OMP stop per-edit Show calls.
- **Fixes** — Cline `Models()` ignores stale legacy state when `providers.json` active provider ≠ ollama; `Makefile.sync` rsync-before-build-info to prevent pin skew.
- **Doc:** [docs/launch-model-inventory.md](docs/launch-model-inventory.md).

**Not ported:** Kimi launch, desktop app launchers (`claude-desktop`, `codex-app`, `hermes-desktop`).

### Upstream Ollama — Qwen Code + Pool launch integrations

**Why:** Upstream ships first-class agent launchers for Qwen Code CLI and Poolside; zerollama keeps Eliza/Zoey but should expose the same upstream integrations for operator parity.

- **`cmd/launch/qwen.go`** — ManagedSingleModel config for `~/.qwen/settings.json`; auto-install; `ConnectableHost()` `/v1` base URL.
- **`cmd/launch/poolside.go`** — `POOLSIDE_STANDALONE_BASE_URL` + API key wiring; hidden on Windows.
- **`cmd/launch/registry.go`** — register `qwen` and `pool` in launcher order.

### Phase 17 — vision/thinking llama-server opt-in

**Why:** Upstream routes all GGUF through llama-server; zerollama blocked split projectors even with explicit `--llama-server-backend`. Opt-in now matches upstream for vision (mmproj) and thinking (`enable_thinking`); Linux auto-default stays plain text only.

- **`llm/llama_server_routing.go`** — check explicit `ZEROLLAMA_LLAMA_SERVER=1` before plain-text gate; projectors allowed on opt-in path.
- **`llm/llama_server_routing_test.go`** — explicit-on + projectors; reject projectors without explicit flag.
- **`scripts/phase/phase17_llama_server_vision_smoke.sh`** — opt-in E2E chat+image on llama-server (`RUN_E2E_P17_VISION=1`).

### Upstream Ollama — Cline providers.json (#16402)

**Why:** Upstream Cline reads `~/.cline/data/settings/providers.json` (OpenAI-compatible `/v1` base URL). Legacy `globalState.json` alone was insufficient for provider drift detection and launch reconfigure.

- **`cmd/launch/cline.go`** — dual-write `providers.json` + legacy state; `ensureClineInstalled()` npm prompt; `Models()` prefers active ollama provider config.
- **`cmd/launch/cline_test.go`** — providers.json coverage; **`TestLaunchIntegration_ClineRewritesWhenLiveProviderDrifted`** in launch tests.

### Upstream Ollama — Phase 17 smoke + integration tests

**Why:** E2E proof that Go → llama-server generates on ship hardware; integration tests aligned with upstream context-shift and HF CLI behavior.

- **`scripts/phase/phase17_llama_server_smoke.sh`** — uses pulled model tag (`P17_MODEL` / `RUN_E2E_PROXY_MODEL`); accepts `thinking` when `response` empty.
- **`integration/context_test.go`** — context-limit 4xx tolerance for llama-server oversized initial prompts (#16764).
- **`integration/create_test.go`** — prefer `hf` CLI over deprecated `huggingface-cli` (#16765).
- **`x/create/create_test.go`** — FP8 error message + Qwen35 MTP tensor expectations aligned with upstream.

### Upstream Ollama — OMP launch (#16410)

**Why:** Upstream added [oh-my-pi / OMP](https://omp.sh) as a first-class `ollama launch omp` integration with Ollama provider discovery and optional web-search plugin.

- **`cmd/launch/omp.go`** — ManagedSingleModel; writes `~/.omp/agent/models.yml` + `config.yml`; launches `omp --model ollama/<name>`.
- **Registry** — `zerollama launch omp`; web search plugin uses shared `shouldManagePiWebSearch()` gate.

### Upstream Ollama — Cohere2 MoE MLX (#16670)

**Why:** Upstream added Command A / North safetensors (`Cohere2MoeForCausalLM`) with a distinct chat template and MoE architecture. Without parser/renderer/MLX model registration, `ollama create` and inference would fall back to passthrough.

- **`model/parsers/cohere.go`**, **`model/renderers/cohere.go`** — North template tokens (`<|START_OF_TURN_TOKEN|>`, thinking/action blocks, tool results).
- **`x/models/cohere2_moe/`** — MLX forward (parallel residual, sliding-window + NoPE layers, sparse MoE).
- **`x/create/cohere2moe.go`** — layer-position quantization heuristic for import.
- **`x/create/client/create.go`** — auto-detect `cohere` template from `config.json` architecture.

### Chunked prefill abort — Phase 15 v31 (Jun 2026)

**Why:** Long context prefill (100k+ tokens) can take 30–60 s on a single GPU. If the client disconnects mid-prefill, the runtime has no way to cancel — it burns the GPU until the last chunk completes. This adds the mechanism to signal abort between page-aligned chunks.

- **`kv_decode_loop_abort_set/clear/check`** (`runtime/native/kv_decode_loop.h`, `.c`) — `atomic_int` abort flag; `run_prefill` checks between chunks and returns `KV_DECODE_LOOP_ERR_ABORT` (-3). GIL is released during prefill, so a Python thread can safely call `abort_set()` concurrently.
- **`decode_loop_abort_set/clear`** Python bindings (`runtime/native/kv_block_pool.c`) — exposed as `METH_NOARGS`; no GIL release needed (atomic write).
- **`PrefillAbortedError`** (`runtime/runtime/kv/native_decode_loop.py`) — distinct exception type so callers can distinguish abort from decode failure (`RuntimeError`) and page bind (`LlamaServerError`).
- **`prefill_abort_set()` / `prefill_abort_clear()`** — thin Python wrappers; swallow `ImportError` in non-linked builds.
- **`run_prefill` auto-clear** — clears abort flag before each call so a stale flag from a previous cancel doesn't immediately abort the next request.
- **Tests** — `test_run_prefill_raises_prefill_aborted_error_on_minus_three`, `test_run_prefill_page_bind_error_still_raises_on_minus_two`, `test_prefill_abort_helpers_no_op_when_not_linked`.
- **Engine wiring (Jun 2026 follow-up):** `PrefillCancelToken` (`runtime/kv/prefill_cancel.py`); passed through inprocess `_decode_stream` (native + ctypes chunk loops); `/api/generate` + `/api/chat` streaming use `disconnect_stream.ndjson_stream_on_disconnect` to call `prefill_abort_set()` when `request.is_disconnected()`; `done_reason=cancelled` on abort.
- **Non-stream disconnect (Jun 2026):** `run_sync_on_disconnect` polls disconnect while `engine.generate()` runs; wired on non-stream `/api/generate`, `/api/chat`, `/v1/chat/completions` (ctypes + llama-server HTTP close + llama-cpp-python wheel via internal streaming).
- **ViT embed cache sizing log (Jun 2026):** `vision embed cache may be undersized` when latest-user frames exceed `OLLAMA_IMAGE_EMBED_CACHE_SIZE`.

### ViT embed cache — configurable slots (Jun 2026)

**Why:** Video agent turns resend the same clip frames; the llamarunner vision encoder (`ImageContext`) had a fixed 4-slot LRU. For 32-frame clips this evicts and re-encodes frames on every turn.

- **`OLLAMA_IMAGE_EMBED_CACHE_SIZE`** — per-runner ViT embed LRU initial slots (default 4).
- **`OLLAMA_IMAGE_EMBED_CACHE_MAX`** — auto-grow cap per multimodal turn (default 64).
- **`growCacheForDistinctFrames`** — expands LRU at encode time for video agents.
- **Session ViT embed overlay (Jun 2026):** `prompt_cache_key` flows to llamarunner `MultimodalTokenize` and **ollama-engine** `VisionEmbedCache`; per-session embed map (32 sessions, 30m TTL) survives global LRU eviction on agent turn 2+. Log: `vision embed session cache hit`.

### Qwen3-VL `padded_input_ids` ggml runner inject (Jun 2026)

**Why:** SGLang preprocessed clients send pretokenized layouts; native path previously re-tokenized text and ignored layout at the runner.

- **`BuildPaddedCompletionPromptTokens`** — splices latest-user `padded_input_ids` into Qwen3-VL HF rendered template (`server/modality/build_padded_prompt.go`).
- **`qwen3vl_hf_runner_inject`** — routes pass `prompt_tokens` to llamarunner; full mtmd chunks replace each `<|vision_start|>…<|vision_end|>` block. Works on **HF** and production **`[img-N]`** paths (renderer skips duplicate tags when `padded_input_ids` set). **llama-server subprocess:** vision blocks → `prompt_string` + `multimodal_data`.
- **Multimodal agent history** — splice targets **each** user block when counts align; prior turns with rendered images OK; multi-turn `padded_input_ids` on both user turns supported.
- **Runtime disconnect cancel** — `/v1/chat/completions` SSE uses `sse_stream_on_disconnect` (same prefill abort as `/api/chat`); `finish_reason: cancelled` on client disconnect.

### Qwen3-VL `padded_input_ids` audit hardening (Jun 2026)

**Why:** Tool-calling agents broke multi-turn padded splice: Qwen3-VL renders tool results inside pseudo `<|im_start|>user\n<tool_response>…` blocks. Span mismatch caused silent inject failure after the renderer had already stripped vision placeholders — prompts lost both layout ids and `[img-N]` markers.

- **Tool-span exclusion** — `qwenVLUserContentSpans` skips `<tool_response>` pseudo-user blocks so span count matches `role=user` messages.
- **`MessageSkipsVisionPlaceholdersForChat`** — latest user skips duplicate vision tags when padded; prior padded users keep placeholders when the chat has tool messages (safe fallback if splice fails).
- **`deferred_multimodal_history`** — `ggmlPaddedCompletionPromptTokens` + `routes.go` downgrade and log when inject cannot build pretokenized ids.
- **llama-server pretokenized truncate with media** — vision-block-aware middle discard; no longer skips truncation when images are attached.
- **llama-server inject fallback** — pretokenized layout without `vision_start` but with media → detokenize + standard media-marker path.

### Gemma4 `padded_input_ids` ggml runner inject (Jun 2026)

**Why:** SGLang preprocessed clients send pretokenized layouts for Gemma4 VLMs; native path previously deferred as `deferred_non_qwen3vl`.

- **`BuildGemma4PaddedCompletionPromptTokens`** — splices `padded_input_ids` into `<|turn>user\n…<turn|>` blocks.
- **`gemma4_img_runner_inject`** — runner injects mtmd chunks at `<|image|>`, `<|video|>` (N frames per clip), `<|audio|>` soft tokens; llama-server maps slots to media markers.
- **`MessageSkipsVisionPlaceholdersForChat`** on Gemma4 render — skip duplicate `[img-N]` when layout is consumed.
- **Unclosed vision blocks** — no phantom `multimodal_data` entries for incomplete `vision_start` spans.
- **OpenAI SSE** — `finish_reason: cancelled` preserved (not coerced to `stop`).

**Docs:** [sglang-multimodal-borrowings.md](docs/sglang-multimodal-borrowings.md) §25; [phase17-llama-server.md](docs/phase17-llama-server.md#padded-multimodal-inject-llama-server).

### Ollama-engine + mllama `padded_input_ids` inject (Jun 2026)

**Why:** Mac Metal default uses **ollama-engine** (`OllamaEngineRequired` families). Padded inject on llamarunner/llama-server alone missed Qwen3-VL, Gemma4, and mllama on the hot path.

- **`runner/ollamarunner/padded_inputs.go`** — `inputsFromPaddedPromptTokens`: Qwen3-VL vision blocks, Gemma4 soft tokens, mllama slot `128256` → `EncodeMultimodal` + `PostTokenize`.
- **`BuildMllamaPaddedCompletionPromptTokens`** — splices `padded_input_ids` into Llama3 `<|start_header_id|>user…<|eot_id|>` spans; `mllama_img_runner_inject` consume mode.
- **`runner/ollamarunner/runner.go`** — `NewSequence` branches on `promptTokens` + `paddedLayoutConsume`; passes Gemma4 soft-token resolution at sequence start.

**Docs:** [sglang-multimodal-borrowings.md](docs/sglang-multimodal-borrowings.md) §32.

### Ollama-engine ViT embed cache + session overlay (Jun 2026)

**Why:** Mac Metal default uses ollama-engine; session ViT overlay was llamarunner-only — agent turn 2+ still re-ran `EncodeMultimodal` on the hot path after global LRU eviction.

- **`runner/ollamarunner/vision_embed_cache.go`** — materialized vision tensors (float32 + grid metadata) in per-runner LRU; auto-grow via `OLLAMA_IMAGE_EMBED_CACHE_MAX`.
- **Session overlay** — `prompt_cache_key` wired through `CompletionRequest` → `NewSequence`; 32 sessions, 30m TTL (same as llamarunner).
- **Log:** `vision embed session cache hit` with `engine=ollama`.
- **`PromptEvalCachedCount` on ollama-engine** — input-cache prefix hits wired to `cached_prompt_tokens` for agent E2E gate (`runner/ollamarunner/runner.go`).

**Docs:** [sglang-multimodal-borrowings.md](docs/sglang-multimodal-borrowings.md) §28–§29.

### Gemma3 `padded_input_ids` ollama-engine inject (Jun 2026)

**Why:** Gemma3 vision is `OllamaEngineRequired` and popular for native image chat; SGLang preprocessed layouts were still `deferred_non_qwen3vl`.

- **`BuildGemma3PaddedCompletionPromptTokens`** — splices `padded_input_ids` into `<start_of_turn>user\n…<end_of_turn>` spans.
- **`gemma3_img_runner_inject`** — ollama-engine injects at `<start_of_image>` token `255999` (`EncodeMultimodal` + `PostTokenize`).
- **gemma3n** — text-only; remains `deferred_non_qwen3vl`.

**Docs:** [sglang-multimodal-borrowings.md](docs/sglang-multimodal-borrowings.md) §32.

### Llama4 `padded_input_ids` ollama-engine inject (Jun 2026)

**Why:** Llama4 Scout/Maverick vision is `OllamaEngineRequired`; pretokenized layouts with tile `<|patch|>` tokens were still `deferred_non_qwen3vl`.

- **`BuildLlama4PaddedCompletionPromptTokens`** — splices into `<|header_start|>user<|header_end|>\n\n…<|eot|>` spans.
- **`llama4_img_runner_inject`** — replaces each `<|image_start|>…<|image_end|>` block (`200080`) with `EncodeMultimodal` + `PostTokenize` tile expansion.

**Docs:** [sglang-multimodal-borrowings.md](docs/sglang-multimodal-borrowings.md) §32.

### LFM2 `padded_input_ids` ollama-engine inject (Jun 2026)

**Why:** LFM2-VL is `OllamaEngineRequired`; pretokenized ChatML layouts with tiled `<|img_row_N_col_M|>` blocks were still `deferred_non_qwen3vl`.

- **`BuildLfm2PaddedCompletionPromptTokens`** — splices into `<|im_start|>user\n…<|im_end|>` spans (same delimiters as Qwen3-VL HF; tool turns use `role=tool` wrappers).
- **`lfm2_img_runner_inject`** — replaces each `<|image_start|>…<|image_end|>` block (or contiguous `<image>` runs when special tokens disabled) with `EncodeMultimodal` + `PostTokenize` tile expansion.
- **LFM2 renderer** — skips `[img-N]` / `<image>` placeholders when latest user carries `padded_input_ids`.

**Docs:** [sglang-multimodal-borrowings.md](docs/sglang-multimodal-borrowings.md) §33.

### GLM-OCR `padded_input_ids` ollama-engine inject (Jun 2026)

**Why:** GLM-OCR vision is `OllamaEngineRequired`; pretokenized image_start…image_end layouts were still `deferred_non_qwen3vl`.

- **`BuildGlmocrPaddedCompletionPromptTokens`** — splices into `<|user|>\n…` spans (ends at next role tag).
- **`glmocr_img_runner_inject`** — replaces each image_start…image_end block with `EncodeMultimodal` + `PostTokenize` M-RoPE expansion.
- **GLM-OCR renderer** — skips `[img-N]` when latest user carries `padded_input_ids`.

**Docs:** [sglang-multimodal-borrowings.md](docs/sglang-multimodal-borrowings.md) §34.

### Mistral3 `padded_input_ids` ollama-engine inject (Jun 2026)

**Why:** Pixtral/Mistral3 vision is `OllamaEngineRequired`; pretokenized `[IMG]…[IMG_END]` layouts were still `deferred_non_qwen3vl`.

- **`BuildMistral3PaddedCompletionPromptTokens`** — splices into `[INST] … [/INST]` user spans (mistral-instruct jinja).
- **`mistral3_img_runner_inject`** — one `EncodeMultimodal` per image block; skip through `[IMG_END]`; `PostTokenize` expands Pixtral patch rows.
- **`chatPrompt`** — skips `[img-N]` prefix when latest user carries `padded_input_ids` (template path, no Go renderer).

**Docs:** [sglang-multimodal-borrowings.md](docs/sglang-multimodal-borrowings.md) §35.

### DeepSeek-OCR `padded_input_ids` ollama-engine inject (Jun 2026)

**Why:** DeepSeek-OCR vision is `OllamaEngineRequired`; pretokenized `<image>` token (`128815`) runs were still `deferred_non_qwen3vl`.

- **`BuildDeepseekOcrPaddedCompletionPromptTokens`** — splices by content-order user spans (deepseek-ocr template has no role wrappers).
- **`deepseekocr_img_runner_inject`** — one `EncodeMultimodal` per contiguous image-token run; `PostTokenize` expands SAM+CLIP layout.
- **`chatPrompt`** — skips `[img-N]` when latest user carries `padded_input_ids` (same gate as Mistral3 template path).

**Docs:** [sglang-multimodal-borrowings.md](docs/sglang-multimodal-borrowings.md) §36.

**Note:** `qwen25vl` / `qwen2vl` already route through `qwen3vl_hf_runner_inject` via `isQwen3VLModel` (vision tokens `151652…151653`).

### SGLang Tier 2 — `sglext` + MM per-request caps (Jun 2026)

**Why:** Latest SGLang (`v0.5.14`) exposes prefix-cache tier breakdown on OpenAI responses and caps multimodal attachments per turn. Zerollama clients migrating from SGLang need compatible shapes without radix/HiCache fully wired.

- **`openai/openai.go`** — `sglext.cached_tokens_details` on non-stream and stream usage chunks; `device`/`host`/`storage` from `api.Metrics`.
- **`api/types.go`** — `cached_tokens_host`, `cached_tokens_storage`, `cached_tokens_storage_backend` on metrics (populated when L3/HiCache backends report them).
- **`server/inference_access_log.go`** — `cached_tokens_device`, `cached_tokens_host`, `cached_tokens_storage` on `inference response out`.
- **`OLLAMA_LIMIT_MM_DATA_PER_REQUEST`** — JSON caps on latest user turn (`server/modality/limit_mm_data.go`); preflight before ffmpeg expand on `/api/chat`.

**Docs:** [sglang-multimodal-borrowings.md](docs/sglang-multimodal-borrowings.md) §7b.

### llama-server padded inject — all native VLM families (Jun 2026)

**Why:** Phase 17 edge/Linux nodes route through llama-server subprocess; SGLang preprocessed clients need the same pretokenized layout → media marker mapping as ollama-engine, not only Qwen3-VL/Gemma4.

- **`llm/padded_prompt_llama_server.go`** — block/slot/image-run builders for mllama, Gemma3, Llama4, LFM2, GLM-OCR, Mistral3, DeepSeek-OCR.
- **`llm/llama_server.go`** — `completionPromptForRequest` wires all `*_runner_inject` consume modes; detokenize fallback when layout lacks vision tokens but `Media` is attached.

**Docs:** [phase17-llama-server.md](docs/phase17-llama-server.md#padded-multimodal-inject-llama-server), [sglang-multimodal-borrowings.md](docs/sglang-multimodal-borrowings.md) §30–36.

### SGLang `precomputed_embedding` ingest — partial (Jun 2026)

**Why:** SGLang clients can supply ViT output rows instead of PNG bytes when `padded_input_ids` already encodes vision token positions — avoids duplicate encode on edge nodes.

- **`api.Message`** — flexible `images` JSON (raw bytes or `{format, feature}`); `precomputed_embeddings` field; requires `padded_input_ids`.
- **`llm.ImageData.PrecomputedFeature`** — row-major vision embeds passed to runners.
- **`runner/llamarunner`** — `encodeImageChunks` skips mtmd when precomputed rows present; inject via padded paths for all native VLM families.
- **ollama-engine (all native VLMs)** — `MultimodalFromPrecomputed` where implemented; padded inject logs `engine=ollama`.
- **llama-server** — explicit 400 (subprocess accepts base64 rasters only).

**Docs:** [sglang-multimodal-borrowings.md](docs/sglang-multimodal-borrowings.md) §7c.

### SGLang `processor_output` ingest — partial (Jun 2026)

**Why:** SGLang clients can send HF `pixel_values` + `image_grid_thw` with pretokenized layouts so ollama-engine skips PNG decode and runs only the vision tower.

- **`api.ProcessorOutput`** — flexible tensor JSON; `images[]` object or `processor_outputs` field.
- **ollama-engine (Qwen3-VL / qwen25vl / Gemma3 / Gemma4 / glmocr / mistral3 / llama4 / lfm2)** — `MultimodalFromProcessorOutput`; log `processor_output runner inject`.
- **Preflight** — `PreflightPreprocessedInputs` (combined with precomputed rules).

**Docs:** [sglang-multimodal-borrowings.md](docs/sglang-multimodal-borrowings.md) §7d.

### ggml llamarunner padded inject — all native VLM families (Jun 2026)

**Why:** Edge/Linux ggml path previously injected pretokenized layouts only for Qwen3-VL and Gemma4; SGLang clients on LFM2/GLM-OCR/Mistral3/etc. fell through to duplicate `[img-N]` render.

- **`runner/llamarunner/padded_families.go`** — mtmd chunk inject for Gemma3, mllama, Llama4, LFM2, GLM-OCR, Mistral3, DeepSeek-OCR consume modes (mirrors ollama-engine token rules).
- **`runner/llamarunner/runner.go`** — unified `inputsFromPaddedLayoutConsume` dispatch + single inject log line.

### Gemma3 preprocessed ingest — partial (Jun 2026)

**Why:** Gemma3 is `OllamaEngineRequired` on Mac; SGLang precomputed/processor clients need parity with Gemma4.

- **`model/models/gemma3/precomputed.go`**, **`processor_output.go`** — post-projector rows and HF `pixel_values` at fixed `vision.image_size`.

**Docs:** [sglang-multimodal-borrowings.md](docs/sglang-multimodal-borrowings.md) §7c–§7d.

### Llama4 / LFM2 / DeepSeek-OCR preprocessed ingest (Jun 2026)

**Why:** Close the last ollama-engine VLM gaps for SGLang preprocessed clients on padded layouts.

- **llama4** — `MultimodalFromPrecomputed` with optional multi-tile `grid_thw [1,tile_h,tile_w]` (+ global chunk); single-tile `MultimodalFromProcessorOutput`.
- **lfm2** — single-tile precomputed rows only (multi-tile still PNG).
- **deepseekocr** — post-projector precomputed stack.

### LFM2 processor_output + SGLang `enable_prefix_mm_cache` hint (Jun 2026)

**Why:** LFM2 is a native Go VLM on Mac Metal; SGLang clients sending HF `pixel_values` should not fall back to PNG when pretokenized layout is already supplied. **`enable_prefix_mm_cache`** is SGLang's operator flag for keeping ViT outputs hot per conversation — zerollama already pins session overlay when `prompt_cache_key` is set; the flag documents intent and warns when clients forget the session key (overlay + layout caches both require it).

- **lfm2** — single-tile `MultimodalFromProcessorOutput` (`[1,H,W]` pixels).
- **`server/modality/prefix_mm_cache.go`** — honor `enable_prefix_mm_cache` in options; log when set without `prompt_cache_key`.
- **`server/routes.go`** — warn on all `/api/chat` turns before preprocessed preflight (before ffmpeg — cheap hint, no ViT work wasted).
- **`openai/openai.go`** — top-level `enable_prefix_mm_cache` → `options` map; test in `openai_test.go`.
- **`scripts/video/video_agent_infer_smoke.sh`** — agent turns send `enable_prefix_mm_cache`; optional `VIDEO_AGENT_INFER_PREFIX_MM_WARN=1` leg greps the hint log line.

**Docs:** [sglang-multimodal-borrowings.md](docs/sglang-multimodal-borrowings.md) §29, §7d; [testing-smoke.md](docs/testing-smoke.md).

### SGLang ViT static cache parity — precomputed + overlay gating (Jun 2026)

**Why:** Agent turn 2+ should skip redundant ViT work for PNG, `precomputed_embedding`, and `processor_output` when the same clip/features were seen on the thread — matching SGLang `MultiModalStaticCache` + `enable_prefix_mm_cache`.

- **`SessionViTOverlayEnabled`** — overlay ON when `prompt_cache_key` is set; opt out with `enable_prefix_mm_cache: false`.
- **ollama-engine** — global + session cache for precomputed/processor on `[img-N]` and padded inject; `GetOrEncodePrecomputed` / `GetOrEncodeProcessorOutput`.
- **llamarunner** — `GetPrecomputedChunks` global + session LRU; `[img-N]` and padded paths use cached precomputed rows; PNG `MultimodalTokenize` session store gated by overlay flag.
- **Logs:** `precomputed_embedding global cache hit`, `precomputed_embedding session cache hit`, `vision embed session cache hit`.

**Observability (Jun 2026 follow-up):** `vision embed global cache hit` on PNG global LRU (both runners); smoke/gate scripts grep precomputed/processor global+session cache hits; `vision_embed_preprocessed_test.go` covers overlay gating.

### mtmd `grid_thw` forward + strict ViT session smoke (Jun 2026)

**Why:** SGLang clients attach `[1,H,W]` per frame; pixel-only smart_resize caused embed count drift vs pretokenized layouts.

- **`mtmd_bitmap_set_grid_hint`** — M-RoPE vision models resize to `W*patch × H*patch` and skip `smart_resize` when hint set; Go bind in `llama/llama.go`.
- **`VideoSpan.GridTHWExplicit`** — only client pre-expanded `grid_thw` forwards to runner; server ffmpeg estimates stay preflight-only.
- **`VIDEO_AGENT_INFER_VIT_SESSION=1`** / **`VIDEO_AGENT_INFER_GRID_THW=1`** — strict smoke legs for ViT session overlay and mtmd grid forward.

**Docs:** [mtmd-grid-thw-handoff.md](docs/mtmd-grid-thw-handoff.md), [testing-smoke.md](docs/testing-smoke.md), [sglang-multimodal-borrowings.md](docs/sglang-multimodal-borrowings.md) §31.

### SGLang multimodal documentation pass (Jun 2026)

**Why:** Operators migrating from SGLang need WHY-oriented docs — not just feature lists — for preprocessed ingest, session ViT pinning, and infer smoke legs.

- **README.md** — precomputed/processor ingest, `enable_prefix_mm_cache`, smoke env vars.
- **docs/ROADMAP.md** — Option 2 shipped table + next steals updated for preprocessed matrix.
- **docs/video-understanding.md** — § precomputed_embedding, processor_output, prefix-mm.
- **docs/sglang-multimodal-borrowings.md** — operator checklist + code map rows.
- **Code comments** — `prefix_mm_cache.go`, precomputed/processor preflight, ollamarunner inject, `routes.go`, smoke scripts.

### `grid_thw` mtmd Go seam + video agent infer smoke hardening (Jun 2026)

- **`llama.MtmdContext.MultimodalTokenize(..., gridTHW []int)`** — Go signature accepts per-frame `[1,H,W]`; debug log when hint present until `mtmd_bitmap_set_grid_hint` lands in llama.cpp. **Why:** wire callers now so upstream bump is a one-line C bind, not a runner refactor.
- **`runner/llamarunner/image.go`** — `ImageContext.MultimodalTokenize` passes `gridTHW` from `llm.ImageData`; cache keys remain image-bytes only (hints do not affect LRU). **Why:** same frame bytes → same ViT embed regardless of client grid metadata.
- **`runner/llamarunner/runner.go`** — `encodeImageChunks` and `[img-N]` path pass `img.GridTHW`; `logVisionGridHint` compares hint vs embed count after encode.
- **`./scripts/video/video_agent_infer_smoke.sh`** — log read **after** all HTTP legs (preproc included); `VIDEO_AGENT_INFER_PREPROC=1` requires `VIDEO_AGENT_GO_LOG`; greps `preprocessed layout session cache hit`, `padded_input_ids runner inject`, `vision grid hints`; ollama-engine detected via `engine=ollama` on inject log; separate `preproc_report.verdict` (`pass`/`soft`/`fail`). **Why:** expand-only smoke does not prove L3/input-cache on real vision prefill; preproc leg must not grep stale logs.
- **`./scripts/video/video_agent_infer_gate_report.sh`** — prints preproc verdict; exits non-zero on preproc `fail`. **Why:** operator sign-off parity with `l3_gate_report.sh`.

**Docs:** [mtmd-grid-thw-handoff.md](docs/mtmd-grid-thw-handoff.md), [testing-smoke.md](docs/testing-smoke.md#video-agent--padded-multimodal-operator), [sglang-multimodal-borrowings.md](docs/sglang-multimodal-borrowings.md) §37.

### MLX agent prompts — context, tokenize, keepalive, observability (Jun 2026)

**Why:** Agent clients (Hermes, Mercury) send 100k+ token megaprompts to MLX safetensors models. Gemma4 multimodal `config.json` can store `vocab_size` in `text_config.max_position_embeddings` (262144) instead of real ctx (131072) → no tail truncate, double tokenize, 5m keep-alive evictions, empty SSE during multi-minute prefill. Fixes are **generic**, not client-specific.

- **Context length correction** (`x/server/show.go`, `server/mlx_model_config.go`) — ignore bogus `text_config.max_position_embeddings == vocab_size`; enrich MLX `ContextLen` at load + schedule. **Why:** VRAM tier defaults inherited 262144; tail truncate never fired.
- **`capMLXScheduleOptions`** — cap `num_ctx` / `num_predict` to MLX model max before scheduler load. **Why:** clients and 128 GB tier defaults request more than the model supports.
- **Tail truncate + budget** (`server/prompt.go`, `effectiveChatPromptBudget`) — token-ID front-drop for single megaprompts; message binary search for multi-turn. **Why:** message drop alone cannot shrink one huge user blob.
- **`PromptTokens` passthrough** — `chatPrompt` always captures MLX token IDs once; routes → `CompletionRequest.PromptTokens` → `Prepare` skips re-encode. **Why:** detokenize→re-tokenize diverges after front-drop; MLX MTP needs exact IDs.
- **Tokenize LRU** (`x/mlxrunner/tokenize_cache.go`) — 16 entries / 2M tokens / 15 m TTL per MLX client. **Why:** duplicate `/v1/tokenize` within one request + identical megaprompts every agent turn (~500ms each).
- **MLX keep-alive floor** (`mlxKeepAliveFloor`, 30 m when unset) — **Why:** MLX load 3–10s; default 5m evicts during agent think pauses.
- **SSE stream keepalive** (`server/stream_keepalive.go`, `OLLAMA_STREAM_KEEPALIVE_INTERVAL`) — status chunks until first token. **Why:** agent HTTP clients abort on empty streams during long MLX prefill.
- **Prefill tuning** (`x/mlxrunner/prefill_config.go`) — chunk size, snapshot disable for long prompts, MTP off above threshold. **Why:** 131k prefill with MTP snapshots can stall or OOM on unified memory.
- **Operator logs** — `runner needs reload` + `reload_reason`; `prefill complete` + `tok_per_sec`; `inference response out` + `prompt_tokens` / `truncated_tokens` / `messages_dropped`; debug `mlx tokenize cache hit`. **Why:** `03:12` audit showed load vs prefill vs template time invisible in one line.

**Docs:** [docs/mlx-agent-prompts.md](docs/mlx-agent-prompts.md); [ROADMAP.md](docs/ROADMAP.md#m15--mlx-agent-prompt-hardening-jun-2026).

### Phase 17 upstream cherry-picks (Jun 2026)

**Why:** Vanilla Ollama @ `07ed7523` (v0.30.10) routes text GGUF as **Go → llama-server** and ships renderer/discovery/pin deltas zerollama lacked. Full rebase would drop training, Python runtime, and Eliza. Phase 17 cherry-picks **integration seams** so merges stay cheap while **Mac ggml Metal remains default** (~+7% decode vs upstream on M4 Max).

- **`llm/llama_server.go` path** — DisableJinja, context shift, MTP/draft-mtp, Linux auto-default when binary found. **Why:** upstream shape without Python hop on the critical path.
- **`discover/llama_server.go`** — short-lived llama-server probe; CUDA arch + ROCm gfx filter; **ggml fallback** on Mac default. **Why:** upstream scheduler inputs; Mac must not spawn llama-server when Phase 17 is off (tests + latency).
- **`LeadingBOSForRenderer`** — `Renderer.LeadingBOS()` + `CompletionRequest.LeadingBOS`. **Why:** with `--no-jinja`, llama-server must not prepend BOS Go already rendered (Gemma4, LFM2, Cogito, …).
- **`chatPrompt` → `PromptTokens`** — tail-truncation returns pre-tokenized IDs to the runner. **Why:** avoid re-tokenize drift after front-drop; MLX MTP long-prefill path.
- **`PreservedTokens()` wiring** — parser interface + harmony + generate/chat routes. **Why:** llama-server preserved-token slots must match Go parser vocabulary.
- **`cmd/launch`** — OpenCode thinking models; `liveConfigMatches` drift guard. **Why:** agent integrations break when inline config diverges from disk.
- **LFM2 optional thinking** — parser + renderer parity with upstream.
- **`llama/llama.go` `-lc++` LDFLAGS** — darwin/linux test + link fix for jinja C++ in CGO graph. **Why:** `go test ./discover/` failed with missing `libc++` on Mac without production build scripts.
- **`LLAMA_CPP_VERSION=b9672`** — vendor synced: `vendor/llama-cpp-b9672`, 16 patches, in-tree ggml/llama.cpp rsync + Metal embed regen. **Why:** match upstream Ollama v0.30.10; Phase 17 mergeability + qwen35/Metal fixes on current llama.cpp.
- **Native `gpu-discover` subcommand** — hidden CLI + Linux/Windows CGO probes (`native_probe_*.go`). **Why:** llama-server discovery merges PCI IDs, compute capability, and gfx targets from a crash-isolated subprocess without loading a model.
- **Integrated GPU policy (`gfx1151`)** — upstream allowlist for Strix Halo 8060S; `OLLAMA_IGPU_ENABLE`. **Why:** default iGPU filter dropped the only GPU on Ryzen AI Max+ boxes.
- **Metal discovery retry** — `ShouldRetryWithMetalTensorDisabled` + `RunnerEnvOverrides`. **Why:** Metal tensor API probe can pass while real context init fails; conservative path must stick for later loads.
- **`llm/llama_server_test.go`** — upstream regression suite (health, SSE ping, context shift, load stall, mmproj). **Why:** Phase 17 mergeability without full C++ link in CI.
- **OpenAI `/v1/models` list** — prefer `ListModelResponse.model` over `name` (#16556). **Why:** tagged manifests expose canonical model id separately from legacy name.
- **`filteredEnv` secret redaction** + **`cudaFlashAttentionSupported`** — upstream log hygiene and old-GPU FA default-off.
- **`filterOldCUDADriver`** — drops pre-Volta GPUs on `cuda_v12` when NVIDIA driver < 570. **Why:** upstream parity; old Pascal cards fail at load with newer CUDA blobs.
- **Docs:** [phase17-llama-server.md](docs/phase17-llama-server.md), [upstream-ollama-diff.md](docs/upstream-ollama-diff.md), [ROADMAP.md](docs/ROADMAP.md#phase-17--upstream-gguf-path-alignment-directional).

**Explicitly not ported:** Mac-default llama-server routing, Python runtime removal, wholesale `sched.go` replace.

### SGLang-inspired multimodal borrowings (Jun 2026)

**Why:** SGLang’s Jun 2026 stack optimizes agent VLMs (repeat clips, OpenAI usage breakdown, prefix KV observability). Zerollama keeps native ffmpeg + vision runners but ports **narrow** patterns that fit Option 2 — without RadixAttention, required SGLang servers, or full `grid_thw` processor parity.

**Caches (three layers):**
- **Pooled `video_url` HTTP** (`openai/video_url.go`) — shared transport + per-request context deadlines. **Why:** repeat HTTPS fetches should reuse TLS/idle conns; timeouts must not be frozen on the client at init.
- **Remote video body LRU** (`openai/video_fetch_cache.go`, 32 entries / 30 m) — keyed by URL after SSRF checks. **Why:** agent turns re-hit the same CDN URL; skip network before ffmpeg.
- **Global video expansion LRU** (`server/modality/video_cache.go`) — `(policy, video digest) → PNG frames`. **Why:** ffmpeg is the dominant cost on repeat bytes.
- **Session expansion cache** (`server/modality/session_video_cache.go`) — `prompt_cache_key` / `eliza.conversationId` pins frames per thread (16 clips / session, 256 sessions). **Why:** global LRU evicts under fleet load; L3 session keys keep **their** clips warm.

**Pipeline and API:**
- **Preprocessed fast path** — `video_spans` + `images`, no `videos` → skip ffmpeg; reject inconsistent spans. **Why:** clients may send already-expanded frames (SGLang-style).
- **Capability on pre-expanded video** — `video_spans` trigger vision/video checks, not only raw `videos`. **Why:** pre-expanded clients must not bypass model capability gates.
- **Preflight scoping** — raw `videos` on any message; pre-expanded spans / audio on **latest user** only. **Why:** echoed multimodal history must not false-reject follow-up turns.
- **mllama preflight** — `PreflightMllamaSingleImage` before ffmpeg. **Why:** one-image models should fail fast with `max_frames=1` guidance, not after subprocess work.
- **OpenAI `prompt_tokens_details`** — `image_tokens`, `video_tokens`, `audio_tokens` (heuristic post-expand); `input_audio` → `AudioClips`. **Why:** billing/debug parity with OpenAI/SGLang; audio must not ride in `Images`.
- **`cached_tokens` usage** — `timings.cache_n` → `api.Metrics` → OpenAI `cached_tokens`; runtime `llama_timings.py` + access log `cached_prompt_tokens`. **Why:** L3 prefix hits were invisible to operators and API clients.
- **Access log modality tokens** — `image_tokens`, `video_tokens`, `audio_tokens` on `inference response out` (post-expand heuristic). **Why:** fleet logs should mirror OpenAI `prompt_tokens_details` without parsing response bodies.
- **OpenAI session keys** — `/v1/chat/completions` accepts `prompt_cache_key` + `options`. **Why:** OpenAI-shaped clients need the same L3 + session expansion pins as `/api/chat`.
- **Gemma4 span placeholders** (`model/renderers/gemma4.go`) — HF `<|video|>` path when `!RenderImgTags`; production uses `[img-N]` per frame.
- **`grid_thw` runner hints** — per-frame `[1,H,W]` on `llm.ImageData`; Info `vision grid hints` after encode on llamarunner (mtmd) and **ollama-engine** (Mac default); `MultimodalTokenize(..., gridTHW)` Go seam (debug until mtmd C API); handoff [mtmd-grid-thw-handoff.md](docs/mtmd-grid-thw-handoff.md).

**Tests and smoke:**
- **Policy golden tests** (`video_policy_golden_test.go`). **Why:** Phase D regression without ffmpeg fixtures in git.
- **ffmpeg golden test** (`video_ffmpeg_golden_test.go`, skips without ffmpeg). **Why:** hook tests do not exercise real ffmpeg argv.
- **Agent turn test** (`TestExpandVideosInChatRequest_agentSecondTurn`). **Why:** LRU unit tests ≠ turn-2 resend-clip agent pattern.
- **OpenAI agent session test** (`openai/video_agent_session_test.go`). **Why:** `/v1/chat/completions` `video_url` + `prompt_cache_key` must hit session expansion cache like `/api/chat`.
- **Qwen3-VL video span render tests** (`qwen3vl_video_test.go`). **Why:** document SGLang per-frame vision blocks vs production `[img-N]` list.
- **`./scripts/video/video_expand_cache_smoke.sh`** — unit gate (modality + openai). **Why:** CI proof without GPU/VLM.
- **`./scripts/video/video_agent_cache_smoke.sh`** — agent loop + OpenAI + Qwen3-VL render; live E2E adds `/v1/chat/completions` probe (`RUN_E2E_VIDEO_AGENT=1`). **Why:** proves session cache in the HTTP loop across API shapes.
- **`./scripts/video/video_l3_agent_gate.sh`** — unit video gate + optional L3 text smoke (`RUN_E2E_L3=1`). **Why:** video expansion and L3 prefix cache are separate layers; one operator entry point.
- **`./scripts/video/video_agent_infer_smoke.sh`** — live VLM inference + turn-2 `cached_prompt_tokens` (`RUN_E2E_VIDEO_AGENT_INFER=1`); greps `padded_input_ids runner inject`, `vision grid hints`, `preprocessed layout session cache hit` (preproc leg, after all requests); optional `VIDEO_AGENT_INFER_PREPROC=1` padded+`grid_thw` leg with `turn2_cached_ok`. **Why:** expand-only `_debug_render_only` smoke does not prove L3 on real vision prefill.
- **`./scripts/video/video_agent_infer_gate_report.sh`** — infer JSON verdict printer. **Why:** operator sign-off parity with `l3_gate_report.sh`.
- **Preprocessed metadata (partial)** — session layout LRU; Qwen3-VL HF skips duplicate vision placeholders when `padded_input_ids` set (`qwen3vl_hf_skip_placeholders`).
- **Token estimate scoping** — `EstimateMultimodalTokens` counts only latest user message (matches preflight). **Why:** agent history echo inflated `image_tokens`/`video_tokens` in usage and access logs.
- **`./scripts/video/gen_video_testdata.sh`** — lavfi MP4 for local ffmpeg golden debugging (no checked-in blobs). **Why:** Phase D fixtures on demand.

**Docs:** [docs/sglang-multimodal-borrowings.md](docs/sglang-multimodal-borrowings.md); [video-understanding.md](docs/video-understanding.md), [video-parity.md](docs/video-parity.md), [ROADMAP.md](docs/ROADMAP.md), [testing-smoke.md](docs/testing-smoke.md).

**Deferred:** full runner consume of `padded_input_ids` (family processor hook); chunked-prefill abort; ViT embedding cache; int8 linear-attn pool (Phase 15).

### LocalAI control-plane borrowings + manifest hygiene (Jun 2026)

**Why:** LocalAI invests in cheap GGUF metadata, scheduler lifecycle, and fleet routing at the daemon—not in replacing llama.cpp. Zerollama keeps ggml/runtime/training but adopts the same **control-plane** patterns so operators spend less time on manifest footguns, VRAM thrash, and blind fleet picks.

- **Fast GGUF metadata** (`fs/ggml.DecodeMetadata`, `llm.LoadModelMetadata`) — skip vocab bodies and tensor weight walks on metadata probes. **Why:** large models on slow disks hung pull/show/load ([LocalAI #9790](https://github.com/mudler/LocalAI/issues/9790)).
- **GGUF guess hooks** (`server/gguf_guess.go`, `gguf_guess_parser.go`) — auto-fill arch, capped `num_ctx` (8192), MTP, stops, `parser`; kill-switch `ZEROLLAMA_DISABLE_GGUF_GUESS` / `LOCALAI_DISABLE_GUESSING`. **Why:** train-context manifests pre-allocate KV and hang before first token.
- **Scheduler watchdog** (`server/sched_watchdog.go`) — LRU `lastUsedAt`, optional VRAM reclaim (`ZEROLLAMA_MEMORY_RECLAIM_THRESHOLD`), busy timeout (`ZEROLLAMA_RUNNER_BUSY_TIMEOUT`), load coalescing, pull `singleflight`. **Why:** multi-model agents evict thrash and stuck runners without manual `stop`.
- **Concurrency groups** (`server/concurrency_groups.go`, `types/model/config.go`) — Modelfile `PARAMETER concurrency_groups`; scheduler evicts conflicting residents before load. **Why:** imagegen + chat on 16 GB need mutual exclusion, not just `OLLAMA_MAX_LOADED_MODELS`.
- **Post-load metadata probe** (`server/runner_metadata.go`, `api.LoadedModelMetadata`) — `num_ctx`, parser, thinking/tools flags, `has_chat_template` on `/api/ps` and fleet `loaded_model_details`. **Why:** manifest vs effective load drift; fleet ops need ground truth without re-pull.
- **Fleet filter-then-score** (`fleet/score.go`, `prefixcache.go`, `probecache.go`, `POST /internal/score`) — warm/queue/affinity scoring; probe cache TTL. **Why:** deterministic routing beats ad-hoc warm+queue sorts; fewer `/health` storms on assign.
- **Status semantics** — fleet `loaded` = ready runners; `InferenceBacklog().loaded` = resident map size for training idle-wait. **Why:** warm routing must not count still-loading runners; training block must see VRAM occupied during load.
- **Docs:** [docs/localai-borrowings.md](docs/localai-borrowings.md) — shipped vs deferred, manifest hygiene, env table.
- **`zerollama repair`:** metadata-only manifest rewrite (`repair MODEL`, `--all`, `--write`) — cap `num_ctx`, fill parser/arch/template without re-downloading weights.
- **Pull enrich + `POST /api/repair`:** new pulls auto-apply GGUF metadata hints; HTTP repair evicts resident runners when `--write`.
- **Fleet capacity scoring:** penalize warm nodes with other loaded models / high effective `num_ctx` from `loaded_model_details`.
- **HF pull (LA8):** `zerollama pull huggingface://org/repo[/file.gguf]` — GGUF from Hugging Face Hub without registry.
- **Logprob score API (LA9):** `POST /api/score` — joint log-probability of candidate continuations for agent routing without generation; llamarunner, ollamarunner, and llama-server backends.

### llama.cpp backend unification (Jun 2026)

**Why:** Multiple checkouts (`../llama.cpp`, `../eliza-llama.cpp`, in-process vendor) caused pin drift and argv crashes (e.g. `--spec-draft-backend-sampling` on stale eliza builds).

- **`llm/llama_cpp_unified.go`** — `UnifiedLlamaCppRoot`, pin read, legacy checkout detection, doctor report.
- **`FindLlamaServer`** — prefers `$LLAMA_CPP_ROOT/build/bin/llama-server` after explicit `LLAMA_SERVER_BIN`.
- **`zerollama doctor`** — **`llama.cpp unified`** check (pin, HEAD, env override warnings).
- **`ApplyUnifiedLlamaCppEnv()`** — at `zerollama serve` + Darwin sidecar: redirects legacy `LLAMA_SERVER_BIN` to unified build when present.
- **`runtime/llama_cpp_unified.py`** + `/health.llama_cpp_unified` — Python parity + operator visibility.
- **`scripts/vendor/migrate_llama_cpp_unified.sh`** — migration helper for shells still exporting eliza-llama paths.
- **U3 vendor rebase:** `vendor/llama-cpp-c84b3020/` (elizaOS @ pin + Ollama patches), `scripts/vendor/rebase_vendor_unified.sh`, `scripts/vendor/sync_vendor_llama.sh`, `Makefile.sync` pin. In-tree `llama/llama.cpp` + `ml/backend/ggml/ggml` rsync from vendor for CGO.
- **Doc:** [docs/llama-cpp-unification.md](docs/llama-cpp-unification.md) — U1–U5 rollout (U4 Phase-17 default, U5 Flash-MoE fold still open).

### L3 prefix cache policy — spec × SWA guards (Jun 2026)

**Why:** vLLM-style selective retention — draft speculative decode (eagle3/mtp/dflash) must not persist KV slot blobs; pure sliding-window models must not `cache_prompt` when the prefix exceeds the SWA window.

- **`llm/llama_server_flags.go`** — probe `llama-server --help` before passing `--spec-draft-backend-sampling`. **Why:** eliza fork / older builds reject the flag and abort MTP/eagle3 startup (`invalid argument: --spec-draft-backend-sampling`).

- **`runtime/runtime/prefix_cache_policy.py`** — GGUF classification (standard / sliding_window / hybrid), `allow_cache_prompt` / `allow_disk_persist`, SWA `effective_window`, draft-spec disable.
- **`cache_bridge.py`**, **`engine.py`**, in-process worker — policy-aware `cache_prompt`, disk TTL on eviction, `/health.llama_cache.policy`.
- **`scripts/phase/l3_spec_cache_smoke.sh`** — subprocess policy gate (`L3_SPEC_METHOD=ngram` default; `eagle3` + `LLAMA_DRAFT_MODEL` for draft leg). Optional: `L3_RUN_SPEC_CACHE=1` on `l3_cuda_full_gate.sh`.
- **Parser tests** — Qwen3 streaming `<` in JSON tool args (`qwen3_test.go`, `runtime_parse_golden_test.go`).
- **Subprocess SWA guard** — `subprocess_slot_state.py` tracks llama-server slot length from completion timings; turn-2 `cache_prompt` respects `pos + n_prompt` vs SWA window; `GET /slots` backfill after restart/disk restore.
- **In-process SWA enforcement** — `engine._prefix_cache_request()` pairs `(cache_prompt, resume_pos)`; `libllama_ctypes._prepare_seq_for_decode(cache_prompt=False)` clears slot and skips disk restore when policy blocks prefix reuse.
- **`kv_cache_spec.py`** — pluggable `KVCacheSpec` (standard / sliding_window / hybrid / disabled); `prefix_cache_policy.py` delegates to specs.
- **Prefix cache trace replay** — `ZEROLLAMA_PREFIX_CACHE_TRACE=1` writes JSONL decisions; `prefix_cache_trace.replay_trace_file()` for offline regression.
- **Spec × page bind** — `runtime/kv/spec_bind.py` validates SWA window at decode; `/health.llama_cache.spec_bind`.
- **Draft spec × prefix cache (vLLM `drop_eagle_block`)** — RAM `cache_prompt` stays enabled under eagle3/mtp/dflash; disk persist disabled; last prefix block dropped on resume (`drop_last_prefix_block`, default block 512 via `ZEROLLAMA_PREFIX_CACHE_BLOCK_SIZE`). In-process path trims KV with `llama_memory_seq_rm` when live seq exceeds adjusted resume pos.
- **`cache_salt` tenant isolation** — `options.cache_salt` / `eliza.cacheSalt` / `ZEROLLAMA_CACHE_SALT` mixed into `derive_slot_id` hash and in-process owner key `cache:{salt}:{key}`.
- **Subprocess prefix invalidation** — epoch bumps + `POST /cuda-graph/invalidate` on llama-server when SWA/draft deny; draft drop-last-block falls back to `cache_prompt=false`.

### Decode graph invalidation — vLLM breakable-graph bind (Jun 2026)

**Why:** L3 prefix cache clears KV slots on SWA block, owner change, or `cache_prompt=false`, but ggml-cuda reuses captured decode graphs keyed by compute-graph topology — not by sequence id. Without invalidation, a stale CUDA graph can replay after KV changed (wrong logits / silent corruption). vLLM bumps an epoch and breaks graphs; zerollama wires the same pattern: Python epoch for future capture keys + native ggml clear today.

- **`decode_graph_policy.py`** — per-slot + global epoch; `graph_capture_key(slot)` → `slot:slot_epoch:global_epoch`. **Why global epoch:** model swap / session close must invalidate slots that were never individually bumped.
- **`llama_context_cuda_graph_invalidate`** (sibling `../llama.cpp`) — iterates context sched CUDA backends → `ggml_backend_cuda_invalidate_graphs` (stream sync + `cuda_graphs.clear()`). **Why sibling patch:** ggml graph cache is internal; no upstream per-slot API yet.
- **`POST /cuda-graph/invalidate`** (sibling llama-server) — task-queue handler calls the same API on `ctx_tgt` for subprocess backend. **Why HTTP not ctypes:** default backend runs inference in a child process; zerollama Python cannot reach that address space.
- **`runtime/kv/cuda_graph_invalidate.py`** — native extension first, ctypes fallback, subprocess HTTP to `POST /cuda-graph/invalidate`; `ZEROLLAMA_DECODE_GRAPH_INVALIDATE=0` kill-switch. **Why three paths:** in-process native/ctypes reach ggml directly; subprocess llama-server owns its own context.
- **`engine._prefix_cache_request`** — passes `base_url` to `bump_decode_graph_epoch` when subprocess policy denies prefix resume. **Why before completion:** stale graphs must be cleared before the next decode step reuses ggml capture.
- **`libllama_ctypes._prepare_seq_for_decode`** — passes `ctx_ptr` on `cache_prompt_disabled`, `spec_bind_swa_block`, `slot_clear`; session `close()` → `bump_all_decode_graph_epochs`. **Why ctx on bump:** epoch alone does not reach ggml’s map.
- **`DecodeGraphCache` stub** — `lookup()` always misses; health exposes epochs + `llama_cpp_probe` (CUDA graph compile flags, pin drift). **Why stub:** capture handles deferred until llama.cpp export exists.
- **`llama_cpp_probe.py`** — probes sibling `LLAMA_CPP_ROOT` for `GGML_CUDA_GRAPHS`, `libllama` path, git pin vs `LLAMA_CPP_VERSION`.
- **`scripts/build/build_llama_server.sh`** — `-DGGML_CUDA_GRAPHS=ON` on CUDA builds. **Why explicit:** graphs are off by default on some cmake presets; invalidation is useless without capture enabled.
- **Tests:** `test_decode_graph_policy.py`, `test_decode_graph_cache.py`, `test_cuda_graph_invalidate.py`, `test_prefix_cache_subprocess.py`, `test_llama_cpp_probe.py`.
- **Metal note:** invalidate API compiles but returns `0` backends cleared (`GGML_CUDA=OFF`); L3 epoch + policy still apply for trace and future capture scaffold.

**Docs:** [docs/decode-graph-invalidation.md](docs/decode-graph-invalidation.md).

### MLX imagegen — Z-Image Turbo on CUDA 16GB (Jun 2026)

**Why:** RTX 5080-class hosts need `x/z-image-turbo` working alongside the existing three VRAM consumers (ggml, Python runtime, training). Upstream MLX imagegen targets Apple Metal; this fork adds CUDA survival tactics, scheduler/VRAM integration, and operator docs — without a second public scheduling system.

- **Staged VRAM pipeline (`x/imagegen/models/zimage/zimage.go`)** — defer text encoder until first generate on CUDA; free encoder before transformer load; keep transformer resident between requests. **Why:** ~12 GB weights + activations cannot coexist on 16 GB; reloading transformer on every request OOM'd the second load.
- **Batched weight materialization (`x/imagegen/mlx/mlx.go`, `manifest/weights.go`)** — `EvalErrBatched(16, …)` + periodic `TrimVRAM`. **Why:** one `mlx_eval` over the full transformer spikes VRAM right after encoder release.
- **CPU VAE subprocess (`decode_latents`, `zimage.go`)** — export latents after denoise; fresh process decodes on CPU. **Why:** in-process CUDA VAE after denoise hit allocator heap issues on 5080.
- **Dimension resolution (`x/imagegen/size`, `server/routes.go`)** — Go validates aspect presets only; runner resolves `width`/`height` with `mlx.GPUIsAvailable()`. **Why:** serve process has no MLX context → wrong `maxSide` caused silent double-clamp (1024→384).
- **Scheduler stability (`server/sched.go`)** — `clearActiveLoading`, `activeLoadingKey`, defer in-use runner unload on `UnloadAllRunners`, nil-safe `needsReload` Ping. **Why:** VRAM handoff during load probe caused nil deref panic; training handoff killed active image streams.
- **VRAM broker (`server/vram/broker.go`)** — `PrepareForImageGen` evicts other runners + runtime sidecar; skip when image model already loaded. **Why:** imagegen needs exclusive GPU; re-prep mid-request tore down in-flight generation.
- **Streaming errors (`server/routes.go`, `x/imagegen/server.go`, `cli.go`)** — propagate `error: …` on NDJSON after stream start; fail if `done` without `image`. **Why:** clients saw generic “completed without image data” when subprocess failed late.
- **ggml GPU discovery (`ml/backend/ggml/ggml.go`)** — `ggml_backend_dev_name` / `description` instead of props string pointers. **Why:** `/info` panic and CPU fallback on RTX 5080 drivers.
- **MLX CUDA allocator patch (`scripts/mlx/patch_mlx_cuda_vram.sh`)** — `cudaMalloc` vs async pool, 90% limit, disable buffer recycle. **Why:** async pool reservations exhaust physical VRAM on 16 GB.
- **MLX-C `array.cpp` patches** (`scripts/mlx/patch_mlx_c_array.sh` + `patch_mlx_c_debug_cleanup.sh`) — add `mlx_array_detach` (break compute graph before free; prevents use-after-free on weight release) and `mlx_go_export_latents_bin_d2h` (direct `cudaMemcpy` D2H of latents post-denoise; bypasses `mlx::core::copy` which faults on 5080 post-checkpoint). Debug cleanup script removes `fprintf` instrumentation added during OOM diagnosis. **Why separate scripts:** allocator patch touches `mlx-src`; array patches touch `mlx-c-src`; both are idempotent and must be re-applied after a clean cmake configure.
- **Docs:** [docs/imagegen-zimage-turbo.md](docs/imagegen-zimage-turbo.md), [x/imagegen/README.md](x/imagegen/README.md); expanded [gpu-5080-operator-guide.md](docs/gpu-5080-operator-guide.md) MLX section.

### L1/L3 full gates + Proxmox build/serve ops (Jun 2026)

**Why:** L1/L3 exit criteria need one-shot orchestrators (not three manual scripts). Proxmox CT 1564 ships inference to remote Ruby clients but minimal checkouts cannot `go build` without vendored `cpp-httplib`; default `OLLAMA_HOST` binds localhost only.

- **`scripts/phase/l1_cuda_full_gate.sh`** + **`l1_gate_report.sh`** — calibrate + concurrent bench → merged `gate.json` + PASS/REGRESS verdict. **Why:** single-stream may show **−5%** np overhead @ 8k; concurrent N=2 is the ship bar (**+~16–20%** on eliza-1 9B).
- **`scripts/phase/l1_full_gate.sh`** / **`l1_metal_gate.sh`** — platform wrappers (CUDA vs Metal).
- **`scripts/phase/l3_cuda_full_gate.sh`** + **`l3_gate_report.sh`** — 8k smoke + 27k production gate → merged verdict. **Why:** wiring @ 8k ≠ agent-scale win @ 27k.
- **`scripts/phase/l3_full_gate.sh`** — dispatches CUDA vs Mac smoke paths.
- **`gpu_5080_session.sh`** — optional `RUN_E2E_L1=1`, `RUN_E2E_L3=1` (need `CUDA_LLAMA_MODEL` or `LLAMA_MODEL`).
- **CGO build on minimal checkout** — root `.gitignore` `vendor/` excludes `llama/llama.cpp/vendor/cpp-httplib/`; copy from sibling `llama.cpp` or run `./scripts/vendor/sync_vendor_llama.sh` after full vendor clone. Doc: [gpu-5080-operator-guide.md](docs/gpu-5080-operator-guide.md#building-zerollama-cgo-on-proxmox-ct).
- **Production serve** — `OLLAMA_HOST=0.0.0.0:8080` for remote clients; embed `127.0.0.1:8081`. **`scripts/serve/serve_production_wrapper.sh`** → `~/bin/serve.sh` (in-repo `serve_gpu_example.sh` only — do not copy to `~/bin`); logs `/tmp/zerollama-serve.log`.
- **`linux_runtime_serve_lib.sh`** — `LINUX_RT_CURL_TIMEOUT=15` (cold `/health` ~9s on 5080); kill `llama-server` on `runtime_port+1` when stopping sidecar. **Why:** 2s curl timeout caused false “runtime failed to start”; orphan llama-server held VRAM across A/B legs.

### Phase 15 v32 — scheduler-driven auto-batch (Jun 2026)

**Why:** v27–v30 batch decode only ran on explicit ``generate_batch`` / ``/internal/generate-batch``. Concurrent ``/api/generate`` threads each called ``completion()`` separately — N ``llama_decode`` calls per token step. v32 opt-in coalesces admitted requests within a short window into ``completions_parallel``.

- **`runtime/runtime/kv/auto_batch.py`** — ``AutoBatchCoordinator``; flush on ``parallel_slots`` fill or ``ZEROLLAMA_KV_AUTO_BATCH_MS`` timeout; batch key includes sampler options hash.
- **`InferenceEngine.generate()``** — routes through coordinator when ``ZEROLLAMA_KV_AUTO_BATCH=1`` + in-process multiseq + linked batch decode.
- **`/health.kv_auto_batch`** — operator stats (``pending``, ``flush_count``, ``batched_requests``).
- **Tests:** ``tests/test_kv_auto_batch.py`` in ``phase15_kv_native_ci.sh``.
- **`runtime/runtime/server/app.py`** — ``Optional[dict[str, Any]]`` on ``InternalBatchGenerateBody.options`` (Python 3.9 + FastAPI compat).

**Env:** ``ZEROLLAMA_KV_AUTO_BATCH=1`` (default off); ``ZEROLLAMA_KV_AUTO_BATCH_MS=5`` (default). Streaming ``generate`` unchanged.

### L1 concurrent + L3 production gates — 5080 CT 1564 (Jun 2026)

**Why:** Single-stream L1 calibration: 1B **+0.5%**; 9B **−5%** @ 8k with `np=2` (expected on one stream). `n_parallel=2` must win under concurrent agent load. L3 @ 8k strict PASS did not prove cache at production ctx (27k).

- **`l1_cuda_concurrent_bench.sh` PASS** — eliza-1 9B, `L1C_N=2` @ 8k: profile ON **~65** vs OFF **~55** agg tok/s (**+~16–20%**); ON leg 0 errors; OFF leg 1×502 (expected at `n_parallel=1`). Re-measure after vendor `libllama.so` pairing fix (Jun 2026).
- **`l3_production_gate.sh` PASS** — eliza-1 9B @ `L3_NUM_CTX=26624`, `L3_PREFIX_REPEAT=150`: cached turn2 **0.72s** vs no-cache **1.48s**; `turn2/turn1=1.02` (strict ratio ≤0.75 not met — decode-bound after warm prefill).
- **`linux_runtime_serve_lib.sh`** — `curl -m` 15s on `/health` wait (WHY: cold health probe ~9s on 5080); kill llama-server on `runtime_port+1` on stop.
- **Docs:** [gpu-profiles-l1.md](docs/gpu-profiles-l1.md), [gpu-profiles-l3.md](docs/gpu-profiles-l3.md), [gpu-5080-operator-guide.md](docs/gpu-5080-operator-guide.md), [ROADMAP.md](docs/ROADMAP.md).

### Phase 15 v32b — writable bind upstream tracker (Jun 2026)

**Why:** Criterion #5 (writable PA→tensor page bind) is upstream-blocked; operators need a static probe and CI watch for when llama.cpp ships page-handle APIs — without requiring a live decode context.

- **`llama_memory_kv_ext_writable_bind_probe`** — staging C API in `llama-kv-ext.h`; returns available when `LLAMA_KV_EXT_WRITABLE_PAGE_MAP` is defined at libllama build time.
- **`page_bind_writable_probe()`** — native ext + Python facade; `/health.kv_page_bind` exposes `writable_bind_available`, `writable_bind_api`, `writable_bind_blocker`.
- **`scripts/phase/phase15_llama_kv_ext_pin_check.sh`** — greps upstream `llama.h` for writable page-map symbol names and prints NOTICE when detected.

### L1 CUDA 5080 calibration — rtx-5080.json tuned on ship hardware (Jun 2026)

**Why:** Eliza-ported profile (`-np 4 -b 2048`) regressed single-stream on 1B Q8 (−12.5%); production 9B only −1% — slot overhead dominates on tiny models.

- **`scripts/phase/l1_cuda_calibrate.sh`** — OFF vs ON (+ `L1_SWEEP_NP`) through `l2_cuda_bench.sh`; sets `ZEROLLAMA_GPU_PROFILE_CTX=0` + `ZEROLLAMA_LLAMA_FORK=0`; cleans `${L1_OUT_DIR}` each run.
- **`scripts/phase/l2_cuda_bench.sh`** — `ZEROLLAMA_GPU_PROFILE` overridable (default `1`) for L1 OFF baseline.
- **`runtime/configs/gpu/rtx-5080.json`** — `n_parallel=2`, `batch_size=1024`, `ubatch_size=256` (half 4090 batch for 16 GiB). Measured @ 8k: 1B **+0.5%** single-stream; 9B **−5%** single-stream / **+~16–20%** concurrent (`L1C_N=2`).
- **Docs:** [gpu-profiles-l1.md](docs/gpu-profiles-l1.md), [gpu-5080-operator-guide.md](docs/gpu-5080-operator-guide.md), [ROADMAP.md](docs/ROADMAP.md).

### RTX 5080 CUDA gates — Phase 15 PASS, L2 FAIL merge, L3 STRICT PASS (Jun 2026)

**Why:** Metal sign-off (M5/M9) proved Phase 15 batch decode on Apple Silicon; CUDA 5080 (CT 1564, Proxmox) needed the same evidence before claiming cross-platform Phase 15 + borrowings L2/L3 status.

- **Phase 15 `phase15_inprocess_signoff.sh` PASS** — KV decode hook (`kv_decode_steps` native), multiseq `kv_inprocess_n_seq_max=2`, continuous batch decode (`batch_decode_in_c=true`) on RTX 5080 with patched b9611 `libllama.so` (`120-real`).
- **`scripts/phase/phase15_inprocess_multiseq_smoke.sh`** — `ZEROLLAMA_GPU_PROFILE=0` on multiseq serve. **Why:** L1 `rtx-5080` sets `n_parallel=4`, overriding temp YAML `llama_parallel_slots: 2` (same pattern as `phase15_metal_signoff.sh`).
- **L2 `l2_cuda_full_gate.sh`** — stock **79.3** vs fork **56.9** tok/s @ 8192 ctx (OuteTTS 1B Q8; reruns ±1 tok/s); **FAIL merge** verdict (exit 1 = verdict fail, not broken run); compat smoke PASS.
- **L2 long-ctx 5080 (Jun 2026):** eliza-1 9B @ 8k/26624 — stock **~18.5** vs fork **~14.3** tok/s (~−22%); **no fork salvage at 27k**. 131k fork blocked: 9B needs ~31 GiB VRAM; 1B QJL `qjl1_256` incompatible with model head dim.
- **`gpu_profiles.py`** — emit `--checkpoint-every-n-tokens` (fork @ 96dd1a8); old `--ctx-checkpoint-interval` crashed llama-server on 131k leg.
- **`scripts/build/build_eliza_llama_server.sh`** + **`build_llama_server.sh`** — `LLAMA_BUILD_WEBUI=OFF` on Linux. **Why:** headless CT builds fail cmake `xxd.cmake` when HF WebUI download/npm build fails.
- **L3 `l3_cache_smoke.sh`** — **STRICT PASS** on eliza-1 9B (CT 1564): `L3_PREFIX_REPEAT=150`, cached turn2 **0.66s** vs no-cache **1.13s**; 1B Q8 remains SOFT PASS.
- **`scripts/phase/l3_cache_smoke.sh`** — Linux/CUDA via `linux_runtime_serve_lib.sh`; `CUDA_LLAMA_MODEL` alias; `L3_PREFIX_REPEAT`; `ZEROLLAMA_GPU_PROFILE_CTX=1` on Linux (WHY: avoid n_ctx=1024 on long agent prefix).
- **`scripts/gpu/gpu_5080_session.sh`** — `RUN_E2E_PREFLIGHT` now respects env (default `1`). **Why:** Proxmox CT 1564 lacks vendored `cpp-httplib` for CGO; hardcoding preflight on blocked the official GPU gate — use `RUN_E2E_PREFLIGHT=0` on minimal trees; CI still runs `phase12_golden_ci.sh`.
- **Docs:** [gpu-5080-operator-guide.md](docs/gpu-5080-operator-guide.md) (Proxmox CT layout, gate sequence, preflight skip, WHYs), [gpu-profiles-l1.md](docs/gpu-profiles-l1.md), [gpu-profiles-l2.md](docs/gpu-profiles-l2.md), [gpu-profiles-l3.md](docs/gpu-profiles-l3.md), [ROADMAP.md](docs/ROADMAP.md).

### Phase 15 v31 — llama-kv-ext pin tracking + hybrid/iSWA resolve (Jun 2026)

**Why:** `llama-kv-ext.h` lived untracked in-tree — vendor sync could wipe it on pin bumps. Hybrid/iSWA models returned `unsupported_memory_type` even though attn KV is reachable via `get_base()` / `get_mem_attn()`.

- **`llama/patches/0015-ollama-llama-kv-ext-phase15.patch`** — formal patch on b9611 pin (cell map, tensor info, CMake entry).
- **`llama/llama.cpp/src/llama-memory-kv-ext.cpp`** — resolve `llama_kv_cache_iswa`, `llama_memory_hybrid`, `llama_memory_hybrid_iswa` to attn base cache; `llama_memory_kv_ext_classify`.
- **`scripts/phase/phase15_llama_kv_ext_pin_check.sh`** — CI gate: patch + in-tree files + upstream `llama.h` deps at pin.
- **`runtime/native/kv_tensor_probe.c`** — exports `memory_kind` / `memory_kind_name` on probe.
- **Docs:** [phase15-llama-kv-ext-upstream.md](docs/phase15-llama-kv-ext-upstream.md).
- **GPU sign-off:** `./scripts/phase/phase15_inprocess_signoff.sh` **PASS** (RTX 5080, CT 1564, Jun 2026) — native decode + batch decode + multiseq; see [gpu-5080-operator-guide.md](docs/gpu-5080-operator-guide.md#phase-15-cuda-libllama--sign-off).
- **`scripts/phase/phase15_inprocess_multiseq_smoke.sh`** — sets `ZEROLLAMA_GPU_PROFILE=0` on serve (parity with Metal sign-off; rtx-5080 L1 otherwise overrides `llama_parallel_slots: 2` → `n_parallel: 4`).

**Still blocked:** writable cross-allocator PA→tensor page bind (needs upstream page-handle API); pure recurrent-only models.

### Phase 15 v30 — per-row C sampling in batch decode (Jun 2026)

**Why:** v29 batch steps sampled in Python via ctypes because v26 C path used one shared sampler (accept-state bleed) and `run_sample` always read the last logit row. v30 passes one sampler pointer per batch row so sampling stays in C with correct logit indices and isolated accept state.

- **`runtime/native/kv_decode_loop.c`** — `kv_decode_loop_run_batch_step` takes `smpl_ptrs[]` per row.
- **`runtime/native/kv_block_pool.c`** — `decode_loop_batch_step` accepts int (legacy) or list of smpl pointers.
- **`runtime/runtime/kv/native_decode_loop.py`** — `run_batch_step(..., smpl_ptrs=)`.
- **`runtime/runtime/worker/libllama_ctypes.py`** — `_decode_parallel_stream` uses C batch sampling when all row samplers are available.

### Phase 15 v29 — streaming batch decode + GPU sign-off (Jun 2026)

**Why:** v27 batched autoregressive steps for `generate_batch` but only returned full text at the end. v29 streams `seq_idx`-tagged chunks through the same C `run_batch_step` path so callers can consume interleaved tokens from N sequences. GPU sign-off needed a loopback hook because batch APIs are engine-internal (no public `/api/generate` batch yet).

- **`runtime/runtime/worker/libllama_ctypes.py`** — `_decode_parallel_stream()`; `complete_parallel_stream()`; shared `_parallel_jobs_and_smpls` / `_finalize_parallel_jobs`; non-stream collects from stream.
- **`runtime/runtime/worker/llama_inprocess.py`** — `completions_parallel_stream()` with batch path + sequential fallback.
- **`runtime/runtime/engine.py`** — `stream_generate_batch()` admits N requests and yields tagged NDJSON-shaped chunks.
- **`runtime/runtime/server/app.py`** — `POST /internal/generate-batch` (loopback-only) for GPU smokes and operator debug.
- **`scripts/phase/phase15_batch_decode_smoke.sh`** — GPU batch + stream smoke; wired into `phase15_metal_signoff.sh` (step 3/5) and `phase15_inprocess_multiseq_smoke.sh`.
- **`scripts/phase/phase15_runtime_kv_env.sh`** — **Why fix:** prefer sibling `../llama.cpp` (has `ggml.h`) over in-repo vendor stub; single venv Python build (avoids 3.9 universal overwrite that caused first-run Metal segfault).
- **Audit fixes:** post-prefill-only native sample (`batch_idx == -1`); sampler cleanup on `_parallel_jobs_and_smpls` failure; `RequestState.FINISHED` on stream batch stop; L3 disk-restore test binds `_prepare_seq_for_decode`.
- **GPU sign-off:** `./scripts/phase/phase15_metal_signoff.sh` **PASS** (M4 Max, Jun 2026) — batch decode step reports `batch_decode_in_c=True`, non-stream + stream batch OK.
- **Tests:** `test_kv_decode_engine_batch.py`; `test_l3_inprocess_disk.py`.

### Phase 15 v28 — `/health` continuous batch plan export (Jun 2026)

**Why:** v27 wired batch decode but operators had no merged view of what `run_batch_step` would consume for N running sequences — only per-request `kv_forward_plans`. v28 adds `kv_continuous_batch` on `/health` for GPU sign-off.

- **`runtime/runtime/kv/forward_plan.py`** — `kv_continuous_batch_forward_plan()` merges running decode-phase rows into `kv_continuous_batch_step_plan`.
- **`runtime/runtime/engine.py`** — `kv_continuous_batch` health field + `kv_snapshot` export.
- **Tests:** `test_kv_forward_plan.py` — batch candidate filtering + `would_batch`.

### Phase 15 v27 — engine wiring for C continuous batch decode (Jun 2026)

**Why:** v26 shipped the C batch primitive but `generate_batch` still called `completion()` per row — N sequential `llama_decode` hot paths. v27 prefills each admitted sequence then merges autoregressive steps through `run_batch_step` when `n_seq_max>1` and the linked ext is available.

- **`runtime/runtime/worker/libllama_ctypes.py`** — `_prepare_seq_for_decode()` (extracted resume/clear); `complete_parallel()` + `_decode_parallel_non_stream()`; `n_batch` sized for `n_seq_max`; **one sampler chain per sequence** (audit fix: shared chain bled `llama_sampler_accept` state across rows).
- **`runtime/runtime/worker/llama_inprocess.py`** — `completions_parallel()` uses batch path when `native_batch_decode_available()`; sequential fallback when disabled or on error.
- **`runtime/runtime/kv/native_decode_loop.py`** — `native_batch_decode_available()`; env `ZEROLLAMA_KV_NATIVE_BATCH=0` disables.
- **Tests:** `test_kv_decode_engine_batch.py`; resume stub binds `_prepare_seq_for_decode`.

### Phase 15 v26 — continuous batch decode in C (Jun 2026)

**Why:** With `llama_parallel_slots>1`, each sequence previously called `llama_decode` separately from Python — N scheduler ticks, N GIL transitions. v26 batches N single-token rows into one C `llama_decode` (continuous batching scaffold).

- **`runtime/native/kv_decode_loop.c`** — `kv_decode_loop_run_batch_step()`; per-row page-bind validation; optional per-row C sampling.
- **`runtime/native/kv_block_pool.c`** — `decode_loop_batch_step`, `decode_batch_layout_multi` bindings; `batch_decode_in_c` on `decode_loop_status`.
- **`runtime/runtime/kv/native_decode_loop.py`** — `run_batch_step()`.
- **`runtime/runtime/kv/decode_plan.py`** — `kv_continuous_batch_step_plan()` for `/health` export.
- **Tests:** `test_kv_decode_batch_loop.py`.

### Phase 15 v25 — auto-link decode loop + 131k long-ctx validation (Jun 2026)

**Why:** C prefill/decode is the hot path but required `ZEROLLAMA_KV_DECODE_LOOP=1` on every build when libllama was present. The page-bind registry capped at 4096 pages (65536 tokens @ block_size=16), blocking 131072 ctx validation that L2 fork-only bench legs depend on.

- **`runtime/setup.py`** — auto-links libllama when found under `LLAMA_CPP_ROOT` / `LLAMA_CPP_LIB`; `ZEROLLAMA_KV_DECODE_LOOP=0` forces unlinked ext (CI); `=1` requires libllama and **exits non-zero** when missing (audit fix: was silently unlinked).
- **`runtime/native/kv_page_bind_internal.h`** — `KV_MAX_PAGES_PER_BIND` 4096 → 8192 (131072 ctx @ block_size=16).
- **`runtime/native/kv_decode_loop.c`** — post-prefill tensor probe moved to `kv_decode_loop_post_prefill_probe()` called after `Py_END_ALLOW_THREADS` (GIL-held registry write; fixes data race from v24 audit).
- **`scripts/phase/phase15_kv_native_ci.sh`** — default unlinked build (`ZEROLLAMA_KV_DECODE_LOOP=0`); includes `test_kv_decode_long_ctx.py`.
- **Tests:** `test_kv_decode_long_ctx.py` — 8192-chunk prefill plan, page-bind boundary at 131072, C bind validation; `test_kv_native_build.py` — forced-link fail-fast.

### Phase 15 v24 — C decode loop page-bind validation + post-prefill tensor probe (Jun 2026)

**Why:** Native C prefill (`kv_decode_loop_run_prefill`) validated page tables only in Python before the GIL-released call; a ctypes bypass or future direct-C caller could write past PA-reserved pages. Tensor bind flags (`cell_pages_bound`, `tensor_pages_bound_slot`) only updated at `complete()` — too late for `/health` polling during long streaming prefills.

- **`runtime/native/kv_page_bind_internal.h`** — `kv_page_bind_validate_range()` (endpoint check, matches Python `validate_token_positions`).
- **`runtime/native/kv_decode_loop.c`** — validate each prefill chunk + decode step before `llama_decode`; post-prefill tensor probe (v25: moved to `kv_decode_loop_post_prefill_probe` in binding after GIL re-acquire).
- **`runtime/native/kv_block_pool.c`** — map bind validation failure (`-2`) to `ValueError` in Python bindings.
- **`runtime/runtime/kv/native_decode_loop.py`** — wrap C bind errors as `LlamaServerError`.
- **`scripts/phase/phase15_metal_signoff.sh`** — step 4: `phase15_tensor_bind_probe.sh`; document why post-generate health cannot assert `tensor_pages_bound`.
- **Tests:** `test_decode_loop_prefill_c_page_bind_validation`.

### L3 — in-process disk cache audit fixes (Jun 2026)

**Why:** Audit found three correctness bugs in the initial disk parity implementation.

- **`runtime/worker/libllama_ctypes.py`** — `_save_slot_cache_disk` now derives token count from live `pos_max` via `sequence_kv_usage` instead of the caller's current-turn `prompt_tokens`; the token metadata written to the blob now matches the full KV history (prompt + all generated tokens). Removed `prompt_tokens` parameter from the API.
- **`runtime/worker/libllama_ctypes.py`** — disk restore guard no longer requires `decode_pos == 0`; any `not is_resume` + pinned slot attempts restore, fixing the case where a running sidecar has a stale owner and non-zero decode_pos.
- **`runtime/cache_bridge.py`** — `prepare_slot_cache_dir` takes `evict: bool = False`; eviction now runs at most once per session (on worker `start()`), not on every save turn.
- **`runtime/worker/llama_inprocess.py`** — calls `prepare_slot_cache_dir(evict=True)` at session start.
- **`runtime/tests/test_l3_inprocess_disk.py`** — updated test stubs to match new `_save_slot_cache_disk` signature (uses `sequence_kv_usage` mock).

### L3 — in-process disk cache parity (Jun 2026)

**Why:** subprocess L3 used llama-server `--slot-save-path`; in-process only had RAM resume (v17). Agent threads lost prefix KV on sidecar restart.

- **`runtime/cache_bridge.py`** — `inprocess_disk_cache_enabled`, `slot_cache_filename`, `slot_cache_file_path`, `prepare_slot_cache_dir`; `/health.llama_cache.inprocess_disk_cache`.
- **`runtime/worker/libllama_ctypes.py`** — `llama_state_seq_{save,load}_file` ctypes; save after pinned decode; restore before clear when slot cold.
- **`runtime/worker/llama_inprocess.py`** — `slot_cache_model_hash` from L1 argv cache types.
- **Env:** `ZEROLLAMA_LLAMA_CACHE_DISK=0` disables disk only.
- **Scripts:** `l3_inprocess_smoke.sh`, `l3_agent_bench.sh`.
- **Tests:** `test_l3_inprocess_disk.py`, cache_bridge disk helpers.

### L2 — CUDA gate audit fixes (Jun 2026)

**Why:** Post-creation audit found four bugs before the CUDA gate scripts were run in CI.

- **`scripts/phase/l2_cuda_full_gate.sh`** — called `l2_runtime_compat_smoke.sh` (Darwin/Metal only: `macos_runtime_serve_lib.sh`, `lsof`, `.dylib`); replaced with `l2_cuda_runtime_compat_smoke.sh`.
- **`scripts/runtime/linux_runtime_serve_lib.sh`** — had `set -euo pipefail` at the top; removed. Sourced library scripts must not set shell options — the caller's `set -euo pipefail` must govern (a sourced `-e` would override caller error handling and cause unexpected exits).
- **`scripts/phase/l2_cuda_bench.sh`** — redundantly sourced `runtime_uv_venv.sh` and `runtime_smoke_lib.sh` before sourcing `linux_runtime_serve_lib.sh`, which already sources both; removed the duplicate `source` calls.
- **`scripts/phase/l2_cuda_bench.sh`, `scripts/phase/l2_metal_bench.sh`** — Python bench core read `llama_server_args` for reporting from a static YAML file, which may not match the arguments chosen by `ZEROLLAMA_AUTO_CONFIG=1` at runtime. Fixed: now prefer `gpu_profile.llama_server_args` from the live `/health` response; fall back to YAML only when that field is absent.
- **`scripts/check_gpu_scripts.sh`** — added new CUDA gate scripts to the syntax-check array and added `grep` assertions for their key content.
- **`docs/gpu-profiles-l2.md`** — step 5 in the CUDA build/run section incorrectly called `l2_runtime_compat_smoke.sh` (Mac-only); updated to `l2_cuda_runtime_compat_smoke.sh`.

### L2 — CUDA bench gate scripts (Jun 2026)

**Why:** `l2_metal_bench.sh` is Darwin/Metal only (dylib, apple_silicon.yaml, lsof). CUDA sign-off needs a parallel bench path for RTX 5080-class hosts before the vendor-merge decision.

- **`scripts/runtime/linux_runtime_serve_lib.sh`** — shared sidecar start/stop for Linux; mirrors `macos_runtime_serve_lib.sh` (fuser, .so, single_gpu.yaml).
- **`scripts/phase/l2_cuda_bench.sh`** — Linux A/B: stock vs fork decode tok/s + VRAM JSON at configurable `num_ctx`; same `L2_HIGH_CTX_WARMUPS` logic as Metal.
- **`scripts/phase/l2_cuda_full_gate.sh`** — CUDA gate orchestrator: eval + compat + 8k/27k/131k bench legs + `l2_gate_report.sh` verdict.
- **`docs/gpu-profiles-l2.md`** — CUDA build/run section; updated runtime integration table; gate status entry.

### L2 — 131k decode bench warmups (Jun 2026)

**Why:** fork-only 131k leg measured first-touch KV alloc, not steady-state decode tok/s.

- **`scripts/phase/l2_metal_bench.sh`** — `L2_HIGH_CTX_WARMUPS` (default 2 when `num_ctx >= 65536`); reports warmup count in JSON.
- **`scripts/phase/l2_full_gate.sh`** — 131k leg uses `L2_BENCH_RUNS=2`, `L2_NUM_PREDICT=64`, warmups.

### Phase 15 v23 — unified prefill chunker + sign-off C pool defaults (Jun 2026)

**Why:** `kv_decode_prefill_plan` and `_prefill_prompt` duplicated chunk boundaries; exported `logits_last` did not match execution (v15 requires final prefill chunk True). Sign-off left C block pool off despite built ext.

- **`runtime/runtime/kv/decode_plan.py`** — `iter_prefill_execute_chunks()`; plan export uses it; final chunk `logits_last=True`.
- **`runtime/runtime/worker/libllama_ctypes.py`** — ctypes prefill calls shared chunker.
- **`scripts/phase/phase15_runtime_kv_env.sh`** — shared env (`ZEROLLAMA_RUNTIME_KV_NATIVE=1`, native decode/sample); `phase15_runtime_kv_ext_build`.
- **`scripts/phase/phase15_metal_signoff.sh`**, **`phase15_inprocess_signoff.sh`** — source env; build ext when `PHASE15_BUILD_KV_EXT=1` (default).
- **Tests:** updated `test_kv_decode_plan.py` for logits_last + execute parity.

### Phase 15 v22 — fix stale decode_pos after sequence clear; re-enable native sampling (Jun 2026)

**Root cause:** In `LlamaLoadedSession._complete_locked` (multiseq / shared-ctx path), `decode_pos` was read from the live KV sequence *before* the `is_resume` check. On the non-resume path `_clear_sequence` wiped the slot, but `decode_pos` still held the previous sequence's final position (7–13 in practice) and was forwarded unchanged as `current_pos` into `_decode_stream` / `_decode_non_stream`. The native C prefill skipped entirely (`start_pos >= n_prompt`) and `llama_sampler_sample` was called with no valid logits → intermittent segfault on Metal. Repro: `ZEROLLAMA_GPU_PROFILE=1` (n_seq_max=8), 5 sequential generates without resume — crashed on loop 4 reliably.

**Fix (one line):** reset `decode_pos = 0` immediately after `_clear_sequence`. The slot is empty; position 0 is the only correct value.

- **`runtime/runtime/worker/libllama_ctypes.py`** — `decode_pos = 0` after `_clear_sequence` on non-resume path; `infer_trace complete.clear stale_decode_pos=N` emitted for observability.
- **`runtime/runtime/infer_trace.py`** — new opt-in trace module (`ZEROLLAMA_INFER_TRACE=1`); wired into `engine.py` and `libllama_ctypes.py` for reload/reuse/prefill/sample phase logging.
- **`scripts/phase/phase15_metal_signoff.sh`** — `ZEROLLAMA_KV_NATIVE_SAMPLE` default changed back to `1` (workaround removed now root cause is fixed).
- **`scripts/e2e/e2e_runtime_smoke.sh`** — removed Darwin `sleep` workaround before `/api/chat`.
- **`scripts/phase/phase15_metal_crash_repro.sh`** — new repro bisect harness (runtime_loop / broker_gguf / phase14_full scenarios).
- **`runtime/tests/test_infer_trace.py`** — unit tests for `infer_trace` enable/disable.

**Verified:** `phase15_metal_signoff.sh` PASS with `ZEROLLAMA_KV_NATIVE_SAMPLE=1`; bisect 10/10 invocations × 5 generates = 50 calls with no crash.

### Phase 15 v21b — tensor bind probe correctness fixes (Jun 2026)

**Why:** v21 audit found five correctness issues in `kv_tensor_bind_attempt`: wrong early-exit blocker for `lctx==NULL`, stale-flag clear happened before `llama_get_memory` (obscuring failure path), `seq_max+1` could overflow `int32_t`, `blocker` on `/health` used a static fallback string even when a probe had run, and `accounting_ok` was compared as raw int.

- **`runtime/native/kv_tensor_probe.c`** — restructured guard order: `lctx==NULL` sets `KV_TENSOR_BLOCKER_NO_PAGE_API`; stale-flag clear is now inside the same block that precedes `llama_get_memory`; overflow guard replaces `seq_max+1` with `base + llama_token_cells` (avoids `INT32_MAX + 1` wrap).
- **`runtime/runtime/kv/page_bind.py`** — `blocker` field now uses the probe's own blocker string whenever a probe ran (cell_bound or not); `accounting_ok` normalised via `bool()` before `None`-guard; `accounting_aligned` in output is `None` when no probe ran.
- **`runtime/tests/test_kv_tensor_probe.py`** — new: `test_page_bind_health_blocker_from_probe_when_cell_bound`, `test_page_bind_health_blocker_fallback_when_no_probe`.
- **`runtime/tests/test_kv_page_bind.py`** — `test_page_bind_health_without_native_ext` now asserts `slots == []` and `bind_level is None`.
- **`docs/phase15-native-kv.md`** — `kv_page_bind` health field row expanded with `status`/`bind_level`/`blocker`/`slots` value catalogue.

### Phase 15 v21 — per-slot bind registry + post-decode warnings (Jun 2026)

**Why:** v20 bind state lived only inside ephemeral probe results; operators could not see per-slot `cell_pages_bound` on `/health` without a running request + linked probe.

- **`page_bind_slots()`** — C export of active registry rows; `/health.kv_page_bind.slots`.
- **`libllama_ctypes.py`** — post-decode warns on incomplete bind (`cell_map_gap`, etc.) when accounting is ok.
- **Scripts** — health smoke + metal signoff assert `slots`/`bind_level`; decode loop build prefers vendored fork.

### Phase 15 v20b — tensor bind audit fixes (Jun 2026)

**Why:** v20 audit found compile bug (C++ `cell_index_for` in `.c` file), wrong tensor for multi-stream, shifted-position cell map, stale bind flags, and health state machine inconsistency.

- **`kv_tensor_probe.c`** — use `llama_memory_kv_cell_for_pos` for stream; probe only live token pages + partial last page; clear stale bind flags before attempt.
- **`llama-kv-cache.cpp`** — `kv_tensor_k/v(kv_layer, stream)` returns per-stream 2D view when `n_stream>1`.
- **`llama-kv-ext.h`** — `llama_memory_kv_tensor_info(..., stream, ...)`.
- **`runtime/kv/page_bind.py`** — misaligned does not override `status=bound`.
- **Tests** — bound-not-overridden-by-misaligned health case.

### Phase 15 v20 — cell + tensor bind via llama-kv-ext (Jun 2026)

**Why:** v19 accounting bind could not resolve PA pages to llama KV storage. v20 adds a staging API in the pinned llama.cpp fork and wires zerollama's tensor probe to cell-index + K/V tensor verification after decode.

- **`llama/llama.cpp/include/llama-kv-ext.h`** — `llama_memory_kv_cell_for_pos`, `llama_memory_kv_cell_map_range`, `llama_memory_kv_tensor_info`.
- **`llama/llama.cpp/src/llama-memory-kv-ext.cpp`** — implementation for standard `llama_kv_cache`.
- **`llama/llama.cpp/src/llama-kv-cache.{h,cpp}`** — `cell_index_for`, `kv_tensor_k/v`.
- **`runtime/native/kv_tensor_probe.c`** — v20 bind attempt; `cell_pages_bound`, `tensor_pages_bound`.
- **`runtime/kv/page_bind.py`** — `status=bound`, `bind_level=tensor` when probe succeeds.
- **`runtime/setup.py`** — prefer `llama/llama.cpp` vendored root for linked builds.

**Requires:** rebuild libllama from fork before `ZEROLLAMA_KV_DECODE_LOOP=1` link.

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v20-ops--cell--tensor-bind-via-llama-kv-ext-jun-2026).

### Phase 15 v20a — native page table on forward plans (Jun 2026)

**Why:** v19 `page_bind_table` was script-only; operators comparing `kv_forward_plans.pages[]` to the C registry had no single JSON view. v20a mirrors the native registry on admitted plans with a parity flag.

- **`runtime/kv/tensor_probe.py`** — `page_table_native_parity()`.
- **`runtime/kv/forward_plan.py`** — `native_page_table`, `page_table_native_parity` when C registry populated.
- **`scripts/phase/phase15_kv_native_ci.sh`** — adds `test_kv_tensor_probe.py`, `test_kv_decode_engine_resume.py`.
- **Tests** — forward plan native mirror; misaligned health status.

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v20a-ops--native-page-table-in-forward-plans-jun-2026).

### Phase 15 v19 — tensor bind scaffold (Jun 2026)

**Why:** v8–v18 seq-position bind could validate token ranges but could not map PA `block_ids` onto llama KV tensor pages — blocked on missing public llama.cpp page-handle API. v19 unblocks the path with accounting-level verify + operator probes so full tensor bind is a thin layer when upstream ships handles.

- **`native/kv_tensor_probe.c`** — `kv_tensor_probe_run`: `llama_get_memory`, seq positions, PA page fit vs live cells.
- **`native/kv_page_bind_internal.h`** — shared page bind registry for pool + probe.
- **`native/kv_block_pool.c`** — `page_bind_table(kv_slot)` export; `page_bind_tensor_probe(ctx_ptr, seq_id, kv_slot)` when linked.
- **`runtime/kv/tensor_probe.py`** — Python facade.
- **`runtime/kv/page_bind.py`** — health includes `tensor_probe`, `tensor_bind_ready`, `blocker`, `accounting_aligned`.
- **`runtime/worker/libllama_ctypes.py`** — post-decode tensor probe warn/strict.
- **`scripts/phase/phase15_tensor_bind_probe.sh`** — build + table export smoke.
- **Tests** — `tests/test_kv_tensor_probe.py`.

**Still blocked for full tensor bind:** public llama.cpp API to attach external page ids to KV tensor storage.

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v19-ops--tensor-bind-scaffold-jun-2026).

### Phase 15 v18 — kv_resume health + L3 two-turn gate (Jun 2026)

**Why:** v16–v17 resume state lived only inside `LlamaLoadedSession._seq_last_owner` with no operator visibility. v18 exposes `/health.kv_resume` and adds a Metal sign-off step for two-turn L3 `prompt_cache_key` traffic.

- **`runtime/worker/libllama_ctypes.py`** — `resume_owner_snapshot()` for health export.
- **`runtime/engine.py`** — `_kv_resume_health()` on `/health` and `kv_snapshot`.
- **`scripts/phase/phase15_metal_signoff.sh`** — step 3: two-turn L3 generate + `kv_resume` assert.
- **`scripts/phase/phase15_health_smoke.sh`** — asserts `kv_resume` key.
- **Tests** — `test_kv_resume_health_*`, `test_generate_l3_second_turn_passes_current_pos`.

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v18-ops--kv_resume-health--l3-gate-jun-2026).

### Phase 15 v17 — L3 session resume owner (Jun 2026)

**Why:** v16b keyed slot ownership on `request_id`, but L3 pinned sessions (`slot_pinned=True`) allocate a **new** `request_id` every HTTP turn while reusing `prompt_cache_key` and `kv_slot`. Multi-turn agent chat therefore always failed the owner check, cleared good prefix KV, and re-prefilled from scratch — defeating L3 cache for in-process backends.

- **`runtime/cache_bridge.py`** — `slot_resume_owner_key(kv_bind_req)`: pinned → `cache:{prompt_cache_key}`; otherwise → `request_id`.
- **`runtime/worker/libllama_ctypes.py`** — `_seq_last_owner` (renamed from v16b `_seq_last_request_id`); resume guard uses `slot_resume_owner_key`; owner cleared on sequence clear and on `close()` (model teardown).
- **Tests** — `test_slot_resume_owner_key_*`, `test_complete_skips_clear_l3_second_turn`, `test_complete_clears_sequence_different_pinned_session`, `test_close_clears_seq_last_owner`.

**Still open:** tensor page bind (blocked on llama.cpp API).

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v17-ops--l3-session-resume-owner-jun-2026).

### Phase 15 v16b — slot-ownership guard for resume (Jun 2026)

**Why:** v16 added `decode_pos > 0` as the guard for skipping `_clear_sequence`, but that condition alone is insufficient: a *different* request can land on the same slot after the first one completes.  Without an ownership check, the second request would resume into stale KV from the first, producing corrupted generations.  v16b adds `_seq_last_owner` on `LlamaLoadedSession` (shipped as `_seq_last_request_id`, renamed in v17) so `complete()` only skips the clear when the incoming owner matches the last writer of that slot.

- **`runtime/worker/libllama_ctypes.py`**
  - `LlamaLoadedSession._seq_last_owner: dict[int, str]` — tracks last owning key per seq slot (v16b: `request_id` only).
  - `complete()` — skip `_clear_sequence` only when `decode_pos > 0` **and** owner matches; writes owner back after decode (stream and non-stream paths).
  - `_resolve_decode_current_pos` — `WHY` docstring explaining the no-op on the single-seq path.
- **`runtime/engine.py`** — `_decode_current_pos_for_request` gets a `WHY` docstring documenting the read-outside-lock pattern, why it is safe, and how `_seq_last_owner` re-validates under the lock.
- **Tests** — `tests/test_kv_decode_engine_resume.py`:
  - `test_complete_skips_clear_same_request_id` — same request resumes; no clear.
  - `test_complete_clears_sequence_different_request_id` — different request on same slot; clears.
  - `test_complete_clears_sequence_no_req_id_with_decode_pos` — `kv_bind_req=None`; conservative clear.
  - `test_complete_clears_sequence_when_current_pos_zero` — refactored onto shared helper.
  - `test_engine_decode_current_pos_for_request` — asserts exact `(lib, ctx, seq_id)` args to `current_pos_for_seq`.

**Still open:** tensor page bind (blocked on llama.cpp API).

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v16b-ops--slot-ownership-guard-jun-2026).

### Phase 15 v16 — engine resume wiring (Jun 2026)

**Why:** v15 added `_decode_stream(current_pos=)` but generate always started from position 0 and cleared the seq. v16 reads live llama seq positions at completion time and skips `_clear_sequence` when resuming.

- **`runtime/kv/physical.py`** — `current_pos_for_seq(lib, ctx, seq_id)`.
- **`runtime/engine.py`** — `_decode_current_pos_for_request()`; passed to `completion` / `completion_stream` / batch.
- **`runtime/worker/libllama_ctypes.py`** — `complete(..., current_pos=)`; skip clear when `decode_pos > 0`.
- **`runtime/worker/llama_inprocess.py`** — forwards `current_pos` / `current_positions`.
- **Tests** — `tests/test_kv_decode_engine_resume.py`.

**Still open:** tensor page bind (blocked on llama.cpp API). **Hardened by v16b** (slot-ownership guard).

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v16-ops--engine-resume-wiring-jun-2026).

### Phase 15 v15 — sampling in C + resume prefill wiring (Jun 2026)

**Why:** v14 hardened the C decode loop but sampling still round-tripped through ctypes each token, and `_decode_stream` always prefilled from position 0 despite `decode_work` exporting live `current_pos`.

- **`native/kv_decode_loop.c`** — `kv_decode_loop_sample`; optional `smpl` on `kv_decode_loop_run_step` (decode + sample in one GIL-released block).
- **`native/kv_block_pool.c`** — `decode_loop_sample(smpl_ptr, ctx_ptr)`; `decode_loop_step(..., smpl_ptr=0)` returns `(steps, token)` when sampling; `/health.kv_decode_loop.sampling_in_c`.
- **`runtime/kv/native_decode_loop.py`** — `run_sample`, `run_step(..., smpl_ptr=)`; `greedy_decode_tokens` uses C sampling when linked.
- **`runtime/worker/libllama_ctypes.py`** — `_decode_stream(..., current_pos=0)` wires remaining prefill + decode resume; C sampling on native fast path.
- **Tests** — `tests/test_kv_decode_stream_resume.py`; E2E patches `run_sample` for ctypes baseline.

**Still open:** tensor page bind; engine passes `current_pos` from `kv_physical` into generate (API ready on `_decode_stream`).

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v15-ops--sampling-in-c--resume-prefill-jun-2026).

### Phase 15 v14 — harden C decode loop (Jun 2026)

**Why:** v13 moved `llama_decode` into C but still held the GIL and lacked resume prefill + operator E2E confidence. v14 releases the GIL, validates page bind before C calls, supports `pos_start` for remaining prefill, and adds optional linked-build parity smoke.

- **`native/kv_block_pool.c`** — `Py_BEGIN_ALLOW_THREADS` around `kv_decode_loop_run_prefill` / `run_step`; `/health.kv_decode_loop.gil_released`; `decode_loop_prefill(..., pos_start=0)`.
- **`native/kv_decode_loop.c`** — `pos_start` on prefill (llama positions = `pos_start + tok_off`).
- **`runtime/kv/native_decode_loop.py`** — `pos_start`, `kv_slot` + `validate_token_positions`; `greedy_decode_tokens()` for E2E parity.
- **`runtime/worker/libllama_ctypes.py`** — passes `kv_slot` into native prefill/step calls.
- **Tests** — `tests/test_kv_decode_loop_e2e.py` (gated: `RUN_E2E_DECODE_LOOP=1`, `LLAMA_MODEL`, linked ext).
- **CI** — `scripts/phase/phase15_kv_decode_loop_build.sh` checks `gil_released`; runs E2E when env set.

**Still open (v15+):** tensor page bind.

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v14-ops--harden-c-decode-loop-jun-2026).

### Phase 15 v13 — native C decode loop via llama_decode (Jun 2026)

**Why:** v12 proved `libllama` links; v13 moves the hot `llama_decode` call into C, reducing GIL contention. Sampling (`llama_sampler_sample`) stays in Python — it's negligible relative to the forward pass and allows reuse of the existing ctypes sampler chain.

- **`native/kv_decode_loop.c`** — `kv_decode_loop_run_prefill` (page-aligned chunking + repeated `llama_decode`); `kv_decode_loop_run_step` (single-token decode step). Manual heap `llama_batch` — **why:** `llama_batch_get_one` is stack-unsafe for chunked prefill; `llama_batch_init` leaves arrays uninitialized unless we fill every field.
- **`native/kv_decode_loop.h`** — declarations for both entry points (gated by `ZEROLLAMA_KV_DECODE_LOOP`).
- **`native/kv_block_pool.c`** — Python bindings `decode_loop_prefill(ctx_ptr, tokens, seq_id, block_size)` and `decode_loop_step(ctx_ptr, token, seq_id, current_pos)`, `#ifdef` gated.
- **`runtime/kv/native_decode_loop.py`** — `run_prefill()` / `run_step()` with `ctx_ptr: int` (ctypes `c_void_p` value); returns `None` when not linked.
- **`runtime/worker/libllama_ctypes.py`** (`_decode_stream`) — v13 fast path: C prefill → sample (ctypes) → C step loop; ctypes fallback when ext not linked or encoder model.
- **Tests** — `test_kv_decode_work_plan.py` covers `run_prefill` / `run_step` no-op safety when not linked.
- **CI** — `scripts/phase/phase15_kv_decode_loop_build.sh` verifies `decode_loop_prefill` + `decode_loop_step` symbols.

**Audit (v13):** documented reserved `n_seq_max` in `kv_batch_alloc` (inner seq arrays are length 1 today); removed stale `sampled_out` from header comment; conftest only resets `vram_yaml_defaults._APPLIED` when a test actually applied YAML defaults.

**Runtime test hermeticity (Jun 2026):** **why** full pytest was failing on Python 3.9 / macOS from env leaks and syntax — `runtime/server/app.py` uses `Optional[]` (FastAPI needs live annotations); `tests/conftest.py` clears native `page_bind` slots and restores VRAM YAML env keys between tests; `engine.py` drops `zip(strict=True)`.

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v13-ops--native-c-decode-loop-jun-2026). **Next:** [v14 — harden C decode loop](docs/phase15-native-kv.md#v14-ops--harden-c-decode-loop-jun-2026) (shipped Jun 2026).

### Phase 15 v12 — libllama link build + probe (Jun 2026)

**Why:** v11 shipped `decode_loop_status` but always reported `ctypes`. v12 wires optional libllama linking at build time so operators can verify the extension links before a full C decode loop lands.

- **`runtime/setup.py`** — `ZEROLLAMA_KV_DECODE_LOOP=1` + `LLAMA_CPP_LIB` / `LLAMA_CPP_ROOT` → `-DZEROLLAMA_KV_DECODE_LOOP`, link `-lllama`, rpath.
- **`native/kv_decode_loop.c`** — calls `llama_max_devices()` as link probe; exposed on `/health.kv_decode_loop.llama_max_devices`.
- **`scripts/phase/phase15_kv_decode_loop_build.sh`** — optional smoke (skips when libllama not built).
- **Audit (v11):** removed duplicate top-level `decode_steps` on forward plans (`decode_work.decode` is canonical); `_empty_prefill_plan` sets `prefill_complete: true`.

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v12-ops--libllama-link-build--probe-jun-2026).

### Phase 15 v11 — unified decode work plan + libllama link scaffold (Jun 2026)

**Why:** v9–v10 export separate `decode_prefill` and `decode_steps`; operators and a future C loop need one phase indicator. Linking libllama into the native ext is a separate build — v11 adds the probe and contract before the loop ships.

- **`kv_decode_work_plan()`** — unified `{phase, prefill, decode}` (`admit` / `prefill` / `decode` / `done`).
- **`kv_forward_plans[].decode_work`** — always present when admitted; includes live phase when `current_pos` known.
- **`current_pos_by_request_from_physical()`** — testable helper; engine uses it for forward-plan wiring.
- **`/health.kv_decode_loop`** — `{available, reason, link}`; C `decode_loop_status()` (linked loop blocked until `ZEROLLAMA_KV_DECODE_LOOP=1` + `LLAMA_CPP_LIB` at build time).
- **Tests / CI** — `tests/test_kv_decode_work_plan.py` in `phase15_kv_native_ci.sh`.

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v11-ops--unified-decode-work-plan--libllama-link-scaffold-jun-2026).

### Phase 15 v10 — in-progress decode plans (Jun 2026)

**Why:** v9 exported admit-time prefill plans (`pos_start=0`). Running requests need the **current** llama write position so operators see remaining prefill + planned decode steps on `/health` without guessing.

- **`kv_decode_prefill_plan(..., pos_start=)`** — plan remaining prompt from `current_pos`; `prefill_complete` when prefill done.
- **`kv_decode_step_plan()`** — single-token decode steps with `logits_last=True` (matches `_decode_stream`); `pending_prefill` while `current_pos < n_prompt`.
- **`kv_forward_plans[].decode_steps`** + **`plan_current_pos`** — when live seq positions available from `kv_physical`.
- **`next_pos_from_llama()`** — `llama_pos_max + 1` → next write position.
- **Engine** — `_kv_forward_plans_health()` wires `kv_physical.running[].llama_pos_max` into forward plans.

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v10-ops--in-progress-decode-plans-jun-2026).

### Phase 15 v9 — decode prefill plan on forward plans (Jun 2026)

**Why:** exit criterion #6 requires C batch layout wired to `kv_forward_plans`. v8 page-chunks at decode time; v9 exports the same plan on `/health` and `/internal/kv-snapshot` so operators and a future native decode loop share one contract without running inference.

**What shipped:**

- **`runtime/kv/decode_plan.py`** — `kv_decode_prefill_plan()`; page-aligned chunks + optional native `batch_layout` summary. **Why shared chunker:** calls the same `iter_prefill_decode_chunks` as `libllama_ctypes._prefill_prompt` — plan boundaries cannot drift from real decode.
- **`kv_forward_plans[].decode_prefill`** — present when request has admitted `block_ids` and `prompt_tokens`. **Why both guards:** waiting requests have no page table yet; exporting a plan without reserved blocks would mislead operators.
- **`logits_last: false` on every prefill chunk** — matches ctypes prefill (`_prefill_prompt` never sets logits on prefill batches). **Why not True on the last chunk:** the first sampled token’s logit comes from the decode loop’s separate single-token batch, not the final prefill batch.
- **`pos_start=0`** — plan covers the full prompt at admit time; continuation positions (`n_pos` after generation or multi-turn) stay a decode-time concern (v10+).
- **Tests / CI** — `tests/test_kv_decode_plan.py` (10 tests) in `phase15_kv_native_ci.sh`.

**Audit fixes (same release):**

- **`logits_last` on final prefill chunk** — earlier draft marked the last chunk `True`; corrected after tracing `_prefill_prompt`. **Why:** llama prefill batches do not emit sampling logits; marking the last chunk True would mislead a future C decode loop.
- **Test structure** — split `test_forward_plan_omits_decode_prefill_without_block_table` out of the page-boundary test; removed unused `pytest` import. **Why:** unrelated assertions in one test name hide failures.

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v9-ops--decode-prefill-plan-export-jun-2026).

### Phase 15 v8 — seq-position page bind + C decode batch layout (Jun 2026)

**Why:** ROADMAP exit criteria 5–6 need llama.cpp **tensor** APIs that do not exist on the public surface. Operators still need (a) PA page tables enforced before decode so generation cannot silently exceed reserved blocks, and (b) batch metadata built off the GIL hot path so a future native decode loop can wire to `kv_forward_plans` without another refactor. v8 ships the bookkeeping + layout layer; tensor mapping and `llama_decode` in C remain blocked upstream.

**What shipped:**

- **`runtime/native/kv_block_pool.c`** — `page_bind_set/clear/resolve/stats`, `decode_batch_layout`, `decode_prefill_chunks`. **Why C:** admission and decode share the same page table; keeping resolve + batch layout in the extension avoids Python list churn on every token batch.
- **`runtime/kv/page_bind.py`** — register PA page tables on scheduler admit, clear on `complete()`; `/health` `kv_page_bind.status=partial`, `bind_level=seq_position` when native ext built. **Why admit-time register:** decode validates against the table that was reserved at admission — not a stale export from `/health`.
- **`runtime/kv/native_decode_batch.py`** — C-built `llama_batch` field lists; page-aligned prefill chunks for prompts longer than one PA page. **Why page-aligned chunks:** matches PA page boundaries so bind validation and future tensor bind share the same token ranges.
- **`runtime/runtime/worker/libllama_ctypes.py`** — ctypes decode uses native batch builder; overrun raises `LlamaServerError` (not raw C exceptions). **Why:** stream/generate paths already catch `LlamaServerError`; operators see a clear KV bind failure instead of a traceback.
- **`runtime/runtime/scheduler/loop.py`** — `register_request_bind` on admit, `unregister_request_bind` on complete; block size from `pools[0].block_size`. **Why pool block_size:** `SchedulerLoop` has no separate config field — the pool is the source of truth for page size.

**M14 — `zerollama doctor --fix` clones llama.cpp:**

- **`cmd/doctor.go`** — runs `ensure_llama_cpp_sibling.sh` (with `ZEROLLAMA_REPO`) before `build_llama_server.sh`. **Why:** fresh clones failed opaquely at Metal build time; `mac_setup` already cloned first — doctor should match that order so `--fix` is a one-command bootstrap.

**Audit fixes (same release):**

- **`self.block_size` crash** — admission called `register_request_bind(..., block_size=self.block_size)` but `SchedulerLoop` has no such field → `AttributeError` on first admit. **Why:** use `pools[0].block_size` like every other KV callsite.
- **`tensor_pages_bound` type** — C `page_bind_stats` now returns Python `False` (not `0`); Python normalizes to `bool`. **Why:** `/health` JSON should not mix int/bool for the same semantic flag.
- **Duplicate bind validation** — validate once in `build_batch_from_tokens` / `_make_batch`, not again in chunk iterator. **Why:** hot path; same check twice bought nothing.
- **`n_predict=0` prefill** — decode loop now decodes the prompt when `limit=0`. **Why:** old condition `n_pos + batch.n_tokens < n_prompt + limit` skipped the only prefill decode when `limit=0`.

**Still blocked:** PA `block_ids` → llama **tensor** KV pages; full decode loop in C without ctypes.

Doc: [phase15-native-kv.md](docs/phase15-native-kv.md#v8-ops--seq-position-page-bind--c-decode-batch-jun-2026) · ROADMAP Phase 15 criteria 5–6 · [mac-dev-setup.md](docs/mac-dev-setup.md) (M14 doctor).

### Ggml manifest `num_ctx` suggest + opt-in clamp (M12, Jun 2026)

**Why:** High-VRAM tier sets server default `num_ctx=262144`; merged manifest defaults pre-allocate full KV at ggml load and can hang qwen35/recurrent models before the first token. Phase 13 runtime already exposes `suggested_max_num_ctx` + opt-in clamp — the Go ggml scheduler had docs only, so operators on the legacy/Metal path had no API signal when manifest defaults exceeded VRAM.

**What shipped:**

- **`server/ggml_num_ctx.go`** — binary-search `suggested_max_num_ctx` from `fs/ggml.GraphSize` + file size vs **free** VRAM (same overhead floor as `sched.go`); optional clamp in `scheduleRunner` before `GetRunner`.
- **`GET /api/show`** — `ggml_num_ctx.suggested_max_num_ctx`; `merged_num_ctx` when the merged default exceeds the suggestion (separate fields so clients are not confused about clamped vs requested context).
- **`/api/chat` / `/api/generate`** — `ggml_num_ctx` only when clamp applied (mirrors runtime `vram_num_ctx`; default off preserves operator trust).
- **Env (default off):** `ZEROLLAMA_GGML_CLAMP_NUM_CTX`, `ZEROLLAMA_GGML_SUGGEST_CTX_MAX`, `ZEROLLAMA_GGML_VRAM_MARGIN`.

**Audit fixes (same release):**

- **No total-VRAM fallback** — early code used startup `totalVRAM` when free was unknown; that over-suggested context (total ≠ free). **Why:** fail-open suggest is safer than pretending all installed VRAM is available.
- **Free-VRAM cache (~2s) for `/api/show`** — avoids live `GPUDevices` probe on every show (CLI startup calls show often). Load path refreshes via `LoadedRunnersForDiscovery()`.
- **Removed silent `recover()` in suggest hi-bound** — `llm.LoadModel` always returns parsed GGUF; panics should not be swallowed.

Doc: [scheduling-vram-policy.md](docs/scheduling-vram-policy.md#ggml-vram-suggest-and-opt-in-clamp-m12-jun-2026).

### Go ollama-engine Metal stability — qwen35moe on Apple Silicon (Jun 2026)

**Why:** `qwen35*` is in `OllamaEngineRequired()` — the Go ollama-engine path is the long-term default on every OS. On M-series Macs, load aborted in `ggml_backend_sched_reserve` with `GGML_ASSERT(tensor->buffer == NULL)` during worst-case graph reserve (after Metal init succeeded). C aborts do not return Go errors, so a darwin-only legacy gate had blocked the Go engine for qwen35. Operators also saw stale Metal free-memory during scheduling and per-load `/health` latency on the training submit path.

**Root cause (sched_reserve):** `newTensor` eagerly allocated backing buffers for **every** graph intermediate while `LoadOperationFit` also called `sched_reserve`, which assigns the same tensors via `ggml_backend_tensor_alloc`. Double assignment tripped the assert on qwen35moe’s large MoE + SSM graphs on Metal.

**What shipped:**

- **`Context.Persistent()`** — `ml/backend.go`, `ml/backend/ggml/ggml.go`: KV/recurrent buffer contexts mark tensors for eager alloc; transient graph intermediates defer to `sched_reserve` / `sched_alloc_graph`. **Why:** KV cells must exist before forward; graph scratch must not pre-claim buffers the scheduler owns.
- **kvcache** — `causal.go`, `encoder.go`, `recurrent.go`, `recurrent_checkpoints.go`: buffer contexts call `.Persistent()`.
- **Darwin routing** — `llm/server.go` `pickOllamaEngine`: removed darwin-only legacy fallback for `qwen35*`/`qwen3next`. **Why:** root allocator fix makes Go engine viable; legacy llamarunner + compat remains for llama-server path and investigation.
- **Worst-case reserve** — removed qwen35 arch blocklist from `runner/ollamarunner/runner.go` `reserveWorstCaseGraph` (was masking the assert instead of fixing allocation).
- **Metal unified free memory** — `discover/runner.go` no longer skips free-memory refresh on darwin/arm64; `discover/metal_unified.go` + `capMetalUnifiedFree()` fill Metal device free bytes from host `vm_stat`. **Why:** bootstrap subprocess free memory is process-local; scheduler layer fit needs unified pool headroom, not stale zeros.
- **Runtime health cache** — `server/inference_workload.go`: 500ms TTL on `runtimeInferenceHealth()`. **Why:** training submit idle-wait hit `:8081/health` on every ggml load attempt.
- **Smoke** — `scripts/runtime/qwen35_mac_smoke.sh`: accept `thinking` when `response` is empty (qwen3.6 thinking models); `scripts/runtime/runtime_smoke_lib.sh` `smoke_unload_ggml_runners`: single-quoted Python for unload payload (shell brace expansion broke `{...}` dicts).

**Sign-off (M4 Max):** full gate `./scripts/gpu/metal_signoff.sh` with `RUN_E2E_QWEN35=1 RUN_E2E_QWEN35_MODEL=qwen3.6:latest` — Phase 13–15 + qwen35 Go ollama-engine generate/unload PASS (Jun 2026).

Doc: [qwen35-apple-silicon.md](docs/qwen35-apple-silicon.md), [apple-silicon-metal.md](docs/apple-silicon-metal.md#go-ollama-engine-sched_reserve-jun-2026). ROADMAP: [M10](docs/ROADMAP.md#apple-silicon--metal-track).

### Metal sign-off gate repairs (Jun 2026)

**Why:** After the sched_reserve fix, `./scripts/gpu/metal_signoff.sh` still failed on unrelated smoke wiring — not Metal regressions. Each failure blocked validating the full M3 + Phase 15 + qwen35 chain on one command.

**What shipped:**

- **Sign-off order** — `scripts/phase/m3_metal_signoff.sh` runs **qwen35 before Phase 15**. **Why:** Phase 15’s exit trap stops the `:8081` sidecar; qwen35 handoff/resume needs runtime `/health` after ggml unload.
- **Phase 15 multiseq** — `scripts/phase/phase15_metal_signoff.sh` sets `ZEROLLAMA_GPU_PROFILE=0` for the `llama_parallel_slots=2` temp YAML. **Why:** L1 `apple-silicon-128g` sets `n_parallel=8`, overriding yaml `2` and breaking `kv_inprocess_n_seq_max` assertions.
- **L3 + inprocess** — `llama_inprocess.py` / `llama_cpp_python.py` accept `cache_prompt` (ignored on inprocess). **Why:** L3 cache bridge passes the kwarg from `engine.py`; Phase 14 generate returned HTTP 500 without it.
- **Unload payload** — `smoke_unload_ggml_runners` uses single-quoted Python `-c` (see Metal stability smoke bullet above).

**Full gate (M4 Max, Jun 2026):**

```bash
LLAMA_CPP_ROOT=../llama.cpp ./scripts/build/build_llama_server.sh
RUN_E2E_QWEN35=1 RUN_E2E_QWEN35_MODEL=qwen3.6:latest ./scripts/gpu/metal_signoff.sh
```

### L2 elizaOS/llama.cpp fork evaluation spike (Jun 2026)

**Why:** L1 tuned flags on stock `q8_0` cache types cap VRAM and tok/s vs eliza-v3’s QJL/Polar/TurboQuant kernels. Replacing `vendor/llama-cpp-b9611/` blindly would break Ollama patches and qwen35 compat — L2 evaluates the fork in an isolated sibling tree first.

**What shipped:**

- **`runtime/llama_fork.py`** — fork detection via `ZEROLLAMA_LLAMA_FORK` or `llama-server --help` probe for `--ctx-checkpoints`.
- **`gpu_profiles.py`** — merges `_eliza_fork_llama_server_flags` (QJL/Polar cache types) when fork enabled; emits `--ctx-checkpoints` on fork builds only.
- **`scripts/build/build_eliza_llama_server.sh`** — clone/build `elizaOS/llama.cpp` @ `96dd1a8` into `../eliza-llama.cpp`.
- **`scripts/phase/l2_fork_eval.sh`** — probe + profile argv smoke.
- **GPU JSON** — `_eliza_fork_llama_server_flags` on 3090/4090/5080/5090/H200 profiles.
- **`/health.llama_fork`** — observability for operators.
- **`scripts/phase/l2_metal_bench.sh`**, **`l2_runtime_compat_smoke.sh`**, **`l2_gate_report.sh`**, **`l2_full_gate.sh`** — Metal A/B + verdict orchestrator.
- **`m3_metal_signoff.sh`** — `RUN_E2E_L2=1` runs full gate.
- Doc: [gpu-profiles-l2.md](docs/gpu-profiles-l2.md).

**Gate (open):** measured tok/s + VRAM win on 5080 + M-series before vendor merge.

**Metal gate run (M4 Max 128 GiB, Jun 2026):**

| Model | ctx | Stock | Fork |
|-------|-----|-------|------|
| eliza-1-2b | 8192 | 37.6 tok/s, q8_0 | 20.5 tok/s, tbq4_0/tbq3_0 |
| eliza-1-27b | 26624 | 13.2 tok/s | 12.7 tok/s |
| eliza-1-27b | 131072 | admission fail (est.) | 5.0 tok/s (fork-only leg) |

**Runtime compat smoke:** PASS (`l2_runtime_compat_smoke.sh`). **Verdict:** stock wins decode on measured A/B legs; fork enables 131k ctx where stock path rejects. **Vendor merge blocked** pending CUDA 5080 bench + qwen35 ggml smoke. Scripts: `l2_full_gate.sh`, `l2_gate_report.sh`.

### L3 prompt cache key → llama-server slot bridge (Jun 2026)

**Why:** L1 raises peak tok/s via batch/parallel flags; L2 may shrink KV footprint. Neither fixes **repeat prefill** — agent threads and fixed system prompts re-run the full prompt every turn when Phase 15 assigns a fresh dynamic `id_slot` and releases it on `complete()`. Eliza-v3 solves this with stable cache keys hashed into llama-server slots plus optional on-disk slot save. L3 ports that bridge without pulling in Eliza’s device UI or bundle catalog.

**What shipped:**

- **`runtime/cache_bridge.py`** — key resolution (`conversationId` → segments → prefix → `promptCacheKey`), `derive_slot_id(key, parallel)`, `--slot-save-path` argv, mtime TTL eviction, `/health` snapshot. **Why separate module from `gpu_profiles`:** cache keys are per-request/session; GPU JSON is per-hardware.
- **Admission** — `_admit_one()` sets `prompt_cache_key`, pinned `kv_slot`, `slot_pinned` before scheduler tick. **Why at admit:** slot must be reserved before `/completion` so llama-server sees a stable `id_slot`.
- **Phase 15 loop** — `SlotAllocator.try_acquire()` for pinned slots; concurrent same-slot requests re-queue; allocator releases tracking on complete but llama-server keeps KV (next turn re-derives same slot from key hash).
- **Subprocess completions** — `cache_prompt: true` when a cache key is present. **Why:** tells llama-server to persist prefix KV into the slot (RAM + optional disk under `--slot-save-path`).
- **Inprocess / wheel workers** — `cache_prompt` kwarg accepted and ignored on `LlamaInprocessWorker` and `llama-cpp-python` backend. **Why:** `engine.py` always passes the flag after L3 admit; Phase 14 Metal sign-off hit HTTP 500 without a matching worker signature.
- **Disk cache** — `~/.cache/zerollama/llama-cache/<modelHash>/`; hash includes GGUF path, draft model, `--cache-type-k/v` so fork vs stock profiles do not collide. TTL via `ZEROLLAMA_LLAMA_CACHE_TTL_MS` (default 1h) for llama-server `slot_*.bin` names.
- **`/health.llama_cache`** — enabled, root, `model_hash`, `slot_save_path`, file stats; `model_loaded` false when GGUF path configured but not on disk yet.
- **Batch** — `generate_batch` + `completions_parallel` pass per-request `cache_prompt`; `options.prompt_cache_keys[]` for per-row keys.
- **`scripts/phase/l3_cache_smoke.sh`**, **`l3_gate_report.sh`** — two-turn cache latency gate; `RUN_E2E_L3=1` on `m3_metal_signoff.sh`.
- **Fixes:** gpu profile emits `-fa on` (stock llama-server rejected bare `-fa`); `macos_runtime_serve_lib` preserves `ZEROLLAMA_RUNTIME_LLAMA_BACKEND=subprocess` for L2/L3 smokes.

**Audit fixes (Jun 2026, second pass):**

- **Canonical GGUF path in `model_hash`** — `_canonical_model_path()` resolves symlinks before hashing. **Why:** same weights via LM Studio symlink vs absolute path must share one `--slot-save-path` directory; fragmented hashes miss disk-restored prefix KV on restart.
- **Orphan model-hash dir sweep** — `evict_orphaned_cache_dirs()` on llama-server cold start. **Why:** L2 profile switches change cache-type fields in the hash; stale sibling dirs accumulate expired `slot_*.bin` files and waste disk without touching the active model hash.
- **Batch `prompt_cache_keys` semantics** — when the list is present, out-of-range indices get no key (no flat-key fallback). **Why:** unrelated batch rows must not accidentally share one pinned slot when only some rows specify keys.
- **`complete()` ordering** — document and enforce `unregister_request_bind` before `SlotAllocator.release`. **Why:** releasing first lets the next tick `try_acquire` the slot while native page bind is still being torn down — stale block ids on decode.

**Operator env:** `ZEROLLAMA_LLAMA_CACHE=0` (disable), `ZEROLLAMA_LLAMA_CACHE_ROOT`, `ZEROLLAMA_LLAMA_CACHE_TTL_MS`. Requires `-np > 1` (L1 profile) for multi-session pinning; subprocess backend for full disk save. Doc: [gpu-profiles-l3.md](docs/gpu-profiles-l3.md).

**Remaining:** in-process disk parity; agent-scale bench on CUDA 5080.

### L1 GPU profiles — per-hardware llama-server autotune (Jun 2026)

**Why:** Phase 13 prevents OOM and suggests `num_ctx`; it does not pick batch size, parallel slots, flash-attn, or MTP draft depth. Operators on 5080 vs 4090 vs M4 Max were either using conservative one-size YAML defaults or hand-copying flags from eliza. Wrong `-b`/`-np` wastes VRAM headroom or silently caps throughput.

**What shipped:**

- **`runtime/gpu_profiles.py`** — loads `runtime/configs/gpu/*.json`, merges flags into `RuntimeConfig.llama_server_args()`. **Why at config load:** runtime inprocess/subprocess paths already consume `llama_server_args()`; Go ggml is a separate track (Phase 17).
- **NVIDIA** — match by `nvidia-smi` name or VRAM bucket (`index.json` `fallback_buckets`). Applied only when loading `single_gpu.yaml` / `dual_4090.yaml`.
- **Apple Silicon** — RAM tiers from `hw.memsize` (`apple_silicon_16g` … `128g`). Applied only when loading `apple_silicon.yaml`. **Why tiers not chip ID:** same M-series SKU ships multiple unified-memory sizes.
- **Stock safety** — fork-only cache types → `q8_0`; fork-only argv (`ctx_checkpoints`) never emitted; `mlock` default **off** in JSON (opt-in `LLAMA_SERVER_EXTRA_ARGS=--mlock`).
- **`runtime/nvidia_probe.py`** — shared cached `nvidia-smi` probes for autoconfig + profiles. **Why split module:** avoids circular import between `autoconfig` and `gpu_profiles`.
- **Observability** — `/health.gpu_profile` (`id`, `bucket_label`, `unified_memory_gb`, `n_parallel`, emit flags). `macos_metal_smoke.sh` prints profile fields.
- **M4 Max sign-off** — `apple-silicon-128g`: `-np 8`, `-c 131072` aligned with Phase 13 `suggested_max_num_ctx`.

**Operator env:** `ZEROLLAMA_GPU_PROFILE=0` (disable), `ZEROLLAMA_GPU_PROFILE_CTX=0` (skip profile `-c`), `LLAMA_SERVER_EXTRA_ARGS` (override). Doc: [gpu-profiles-l1.md](docs/gpu-profiles-l1.md).

**Remaining:** CUDA 5080 benchmark gate to validate/tune `rtx-5080.json` (Apple side marked done in ROADMAP L1).

### Mac GPU bootstrap discovery — Metal reported as CPU (Jun 2026)

**Why:** Operators on M-series Macs saw `inference compute library=cpu`, `total_vram="0 B"`, and `offloaded 0/N layers to GPU` even though llama.cpp init logged `Apple M4 Max`. The scheduler uses bootstrap GPU discovery to pick layer layout and VRAM-tiered default `num_ctx`; empty discovery forced CPU-only scheduling and capped context at 4096 on 128 GiB machines.

**Root cause:** Bootstrap discovery asks the ollama-engine runner `GET /info`. The handler loaded a dummy model with **zero GPU layers**, which calls `ensureDevices(true)` on first init and sets `GGML_DISABLE_METAL=1` permanently (`sync.Once`). Metal never registered in the discovery subprocess, so `/info` returned no GPUs. Inference runners are separate processes and still init Metal, but the **main server** believed there was no GPU.

**What shipped:**

- **`DiscoverBackendDevices()`** — `ml/backend/ggml/ggml.go`: probe path that calls `ensureDevices(false)` so Metal registers during bootstrap only. **Why:** discovery must not reuse the `num_gpu=0` CPU-only gate meant to avoid runtime sidecar contention on actual loads.
- **Ollama-engine `/info`** — `runner/ollamarunner/runner.go`: use `DiscoverBackendDevices()` when no model is loaded instead of dummy zero-layer `model.New`. **Why:** faster (no temp GGUF) and correct on darwin unified memory.
- **Docs** — [apple-silicon-metal.md](docs/apple-silicon-metal.md#gpu-bootstrap-discovery-jun-2026): symptoms, fix, startup speed env vars.

**Verify after rebuild:** `./scripts/build/build_zerollama_mac.sh && ./zerollama serve` → log shows `inference compute library=Metal`, `total_vram` ~100+ GiB, model load logs `offloaded N/N layers to GPU`.

**Related (harmless load/chat warnings):** qwen35 GGUFs may log `control-looking token … was not control-type` (bad `token_type` in blob; llama.cpp overrides) and `embeddings required but some input tokens were not marked as outputs` (llamarunner embedding mode vs chat batch). See [qwen35-apple-silicon.md — Token warnings](docs/qwen35-apple-silicon.md#token-warnings-jun-2026).

### Prompt truncation surfaced in API responses (Jun 2026)

**Why:** When input exceeded `num_ctx`, runners logged `truncating input prompt` but clients got a normal 200 with no indication that most of the prompt was dropped.

**What shipped:** Final `/api/chat` and `/api/generate` responses (and streams' last chunk) now include:

- `prompt_truncated: true` and `original_prompt_tokens` when token-level truncation occurred in the runner
- `messages_truncated: true` and `messages_dropped` when `chatPrompt` removed older messages

Set `"truncate": false` on the request to get HTTP 400 instead of silent truncation.

**Follow-up (Jul 2026):** propagate chatPrompt pre-drop size + runtime `detect_context_overflow` for proxy/sidecar paths — see [Unreleased — Explicit context-overflow fields](#explicit-context-overflow-fields-jul-2026).

### Model unload after create/stop + manifest `num_ctx` vs load-time KV (Jun 2026)

**Why:** Operators updated `num_ctx` via `/api/create` (manifest showed 262144) but `/api/ps` still reported `context_length: 4096` — the in-memory runner was never evicted. `zerollama stop` returned success while the model stayed loaded when `refCount > 0` or the scheduler key did not match the loaded runner. Separately, persisting **`num_ctx: 262144` as the model default** caused generation to hang: llamarunner pre-allocates KV for the manifest context at **load** time, not per request.

**Root causes:**

- **`/api/create`** updated manifest blobs only; the ggml scheduler reused the warm runner (`needsReload` never ran until the next inference request with different merged options).
- **`expireRunner`** only queued unload when `refCount <= 0`; active or leaked references left the runner resident while HTTP returned `done_reason: "unload"`.
- **Manifest `parameters.num_ctx`** is merged in `modelOptions()` and passed to `llama.Load` — KV/recurrent buffers are sized for that value immediately. Large defaults (262K on qwen35moe) can block or hang before first token; **request `options.num_ctx`** on an already-loaded smaller context is a different path (Hermes / runtime may grow per request; ggml may still `needsReload` when runner options differ).

**What shipped:**

- **`expireRunner`** — always schedules unload; `processExpiredRunner` retries while `refCount > 0`. **`findLoadedRunner`** — match by `ModelPath`, then `ShortName` / `Name`. **Why:** stop and empty-prompt unload must not silently no-op.
- **`/api/create`** — after successful manifest write, evicts any loaded runner for that model. **Why:** parameter changes (`num_ctx`, `num_gpu`, …) must apply on next load without manual stop.
- **Docs** — [qwen35-apple-silicon.md](docs/qwen35-apple-silicon.md#manifest-num_ctx-vs-request-options), [scheduling-vram-policy.md](docs/scheduling-vram-policy.md#go-ggml-scheduler-keep_alive-unload-and-num_ctx-at-load).

**Operator guidance:** Keep manifest default **`num_ctx` modest (e.g. 4096)** for fast reliable loads on ggml; pass large context via **`options.num_ctx` per request** when Hermes or the runtime detects need. Verify unload: `curl …/api/generate -d '{"model":"…","prompt":"","keep_alive":0}'` then empty `/api/ps`. `/api/ps` `context_length` reflects the **loaded** runner, not the manifest or a single request's options.

### Mac dev bootstrap tiers (Jun 2026)

**Why:** Another developer cloning the repo could not run `./scripts/runtime/mac_setup.sh` successfully without your exact layout: **`../llama.cpp` had to exist manually**, **metal sign-off ran by default** (needs a pulled text GGUF in `~/.ollama/models`), and **CI smokes default `OLLAMA_HOST=:8080`** while daily **`zerollama serve` uses `:11434`** — copy-pasting smoke curl against the wrong port looked like a broken server.

**What shipped:**

| Tier | Goal | Command |
|------|------|---------|
| **0** | Build + serve | `./scripts/runtime/dev_bootstrap.sh` |
| **1** | Chat | `./zerollama pull llama3.2:3b` |
| **2** | Metal sign-off | `MAC_SETUP_SIGNOFF=1 MAC_SETUP_GO=0 MAC_SETUP_BUILD=0 ./scripts/runtime/mac_setup.sh` |
| **3** | qwen35 smoke | `RUN_E2E_QWEN35_MODEL=tag ./scripts/runtime/qwen35_mac_smoke.sh` |

- **`dev_bootstrap.sh`** — thin entry; sets `MAC_SETUP_SIGNOFF=0`, `MAC_SETUP_LLAMA_CLONE=1`. **Why:** one command name for “fresh clone, any checkout path.”
- **`ensure_llama_cpp_sibling.sh`** — shallow-clone `LLAMA_CPP_VERSION` pin to `${REPO}/../llama.cpp`. **Why:** runtime inprocess and sign-off need `libllama.dylib`; failing late in `build_llama_server.sh` was opaque.
- **`zerollama doctor --fix`** — runs `ensure_llama_cpp_sibling.sh` before Metal `build_llama_server.sh` (same order as `mac_setup`). **Why:** tier-0 bootstrap should not require knowing about sibling clone scripts — `doctor --fix` is the self-service path on fresh checkouts.
- **`mac_setup.sh`** — sign-off **off** by default; `mac_setup_has_signoff_model()` skips sign-off with pull instructions when no GGUF; tier hints at end. **`MAC_SETUP_LLAMA_OPTIONAL=1`** for ggml-only dev without llama build.
- **`build_llama_server.sh`** — default `LLAMA_CPP_ROOT=${REPO}/../llama.cpp` (was `${REPO}/../../llama.cpp`). **Why:** sibling path must not depend on repo nesting depth.
- **`qwen35_mac_smoke.sh`** — documents `OLLAMA_HOST=:11434` override for daily serve.
- **Docs** — [mac-dev-setup.md](docs/mac-dev-setup.md) (tiers, port table), [development.md](docs/development.md), [README.md](README.md).

**Ports (why two Go ports):** upstream Ollama convention is `:11434`; CI/sign-off scripts historically used `:8080` to avoid clashing with a system Ollama. Smokes set `OLLAMA_HOST` internally — daily dev does not.

ROADMAP: [M14](docs/ROADMAP.md#apple-silicon--metal-track).

### Unified Mac build (Jun 2026)

**Why:** Operators had to remember two scripts — `build_zerollama_mac.sh` (ggml) and `build_production_mac.sh` (MLX dylibs) — to run safetensors from repo-root `./zerollama`.

**What shipped:**

- **`build_mlx_dylibs_mac.sh`** — shared CMake install for MLX Metal v3/v4 (dev `build/metal-v*/` or production `dist/darwin-arm64/` via `INSTALL_PREFIX`).
- **`build_zerollama_mac.sh`** — `BUILD_MLX=auto` (default): builds MLX dylibs when `../mlx` exists and `build/metal-v3/.../libmlxc.dylib` is missing; `BUILD_MLX=0` for fast ggml-only; `BUILD_MLX=1` / `MLX_FORCE=1` to force rebuild.
- **`build_production_mac.sh`** — regenerates `ggml-metal-embed.metal` before `build_darwin.sh`; arm64 MLX cmake delegated to `build_mlx_dylibs_mac.sh`.
- **`zerollama doctor`** — MLX fix hint points at `BUILD_MLX=1 ./scripts/build/build_zerollama_mac.sh` for dev.

Doc: [mac-dev-setup.md](docs/mac-dev-setup.md#dev-vs-production-mlx-layout).

### Go ollama engine on darwin (qwen35) — superseded (Jun 2026)

**Superseded by:** [Go ollama-engine Metal stability — qwen35moe on Apple Silicon](#go-ollama-engine-metal-stability--qwen35moe-on-apple-silicon-jun-2026) above. The darwin legacy gate and qwen35 worst-case reserve blocklist are removed; sched_reserve root fix + M4 Max sign-off landed.

### Mac smoke gaps (Jun 2026)

**Why:** M4 Max sign-off exposed three Mac-only failure modes that looked like one “broken Metal” bug but were independent: SSE proxies hung without terminal frames, runtime + legacy ggml both touched Metal on one device, and `num_gpu=0` still registered the Metal backend at first ggml init.

**What shipped:**

- **Proxy v1 SSE hang** — Python runtime always yields `data: [DONE]` on error mid-stream; Go proxy uses `copyRuntimeResponseBody` (flush after each chunk). **Why:** Gin buffered `io.Copy` until EOF; partial SSE streams hung curl/CI for 20+ minutes.
- **Darwin Metal contention** — `server/darwin_ggml_policy.go`: block ggml when runtime `llama_server=true`; contention checks run **before** `PrepareForLegacyRunner` so the sidecar is not evicted for a load that will be skipped. **Why:** dual Metal residency wedged the GPU silently.
- **`num_gpu=0` Metal init** — `GGML_DISABLE_METAL` gates Metal backend registration in C++; Go sets it before first `OnceLoad` on CPU-only loads. **Why:** embed/CPU-only smokes still initialized Metal and contended with the runtime sidecar.
- **Scheduler HTTP status** — `ErrRuntimeInferenceModel` → 400, `ErrDarwinMetalContention` → 503 (was generic 500). **Why:** operators need actionable routing errors, not “internal server error.”
- **e2e smokes** — `RUN_E2E_STREAM_MAX` cap; legacy ggml skipped on darwin when runtime holds Metal unless `RUN_E2E_LEGACY_FORCE=1`.

Doc: [docs/apple-silicon-metal.md](docs/apple-silicon-metal.md#scheduler-errors-http-status).

### Apple Silicon polish (Jun 2026)

**Why:** M10 qwen35 VL manifests could pick `clip` as primary family; LM Studio MLX imports need full disk copy but listed models anyway; qwen35 parser lost thinking text when streams ended without `</think>`; catalog hid MLX models silently on `statfs` errors; GGUF+config.json dirs failed pull; Windows CI broke on `syscall.Statfs`.

**What shipped:**

- **`PrimaryFamily()`** — routes renderers/parsers/thinking for VL manifests where projector (`clip`) was stored first; returns `""` for projector-only manifests (`server/model_family.go`). **Why:** create-time layer order stored `ModelFamily=clip` on qwen35 VL blobs.
- **Qwen35 parser** — `flushDoneEvents` on stream end (thinking, whitespace-after-close, trailing content). **Why:** truncated streams dropped reasoning text that never received a close tag.
- **LM Studio integration (v0.0.1)** — discover `~/.lmstudio/models`, merge into `list`/`/api/tags`, pull-from-cache for GGUF and MLX safetensors. **Why:** avoid re-downloading weights LM Studio already fetched.
  - **Native MLX import** — `ImportSafetensorsFromDirectory` when `config.json` + `.safetensors` present. **Why:** MLX→GGUF conversion fails on dtypes like `U32`.
  - **Disk checks** — `ImportCopyBytes` / `HasDiskForImport`; catalog hides MLX when `OLLAMA_MODELS` volume lacks ~model size + 512 MiB; pull fails early with readable error. **Why:** repack doubles disk use; mid-import ENOSPC wastes operator time.
  - **`OLLAMA_LMSTUDIO_LIST_ALL=1`** — list all discoverable models anyway (pull still enforces). **Why:** hidden models confused operators on tight disks.
  - **`dirIsMLXSafetensors`** — only MLX-layout dirs count toward copy bytes. **Why:** legacy safetensors without `config.json` symlink like GGUF and must stay visible in catalog.
  - **`weightFilesOnly`** — strip `config.json` before GGUF convert. **Why:** multi-file dirs (GGUF + HF metadata) sent JSON to the GGUF parser.
  - **Portable free space** — `diskspace_unix.go` / `diskspace_windows.go`. **Why:** `syscall.Statfs` is not available on Windows CI.
- **Opt-in qwen35 smoke** — `./scripts/runtime/qwen35_mac_smoke.sh` (runtime handoff → Go ollama-engine generate); `RUN_E2E_QWEN35=1` on `metal_signoff.sh` / `m3_metal_signoff.sh` (**runs before Phase 15** — sidecar teardown order). **Why:** qwen35 uses ggml on `:8080`, not the runtime inprocess path covered by Phase 14 alone.
- **Mac build** — `build_zerollama_mac.sh` passes `-ldflags` version (`VERSION` env, default `0.0.1`).

Docs: [lmstudio-import.md](docs/lmstudio-import.md), [apple-silicon-metal.md](docs/apple-silicon-metal.md), [qwen35-apple-silicon.md](docs/qwen35-apple-silicon.md), [mlx-routing-policy.md](docs/mlx-routing-policy.md).

## [0.0.1] — 2026-06-12

First tagged zerollama build with embedded version string. Includes LM Studio cache import, MLX disk policy, and Mac polish items above. Operators: `./scripts/build/build_zerollama_mac.sh && ./zerollama serve`.

### Fleet LAN discovery (F4)

**Why:** Static `ZEROLLAMA_FLEET_PEERS` works for K8s and fixed IPs but homelab operators want zero-config LAN discovery without maintaining peer lists.

**What shipped:**

- **Nodes:** `ZEROLLAMA_MDNS=1` on `zerollama serve` advertises `_zerollama._tcp` (TXT: `role=node`, `version`).
- **Fleet manager:** `--mdns` / `ZEROLLAMA_FLEET_MDNS=1` browses LAN and merges with static peers; peers optional when browse enabled.
- **Fleet advertise:** `--mdns-advertise` / `ZEROLLAMA_FLEET_MDNS_ADVERTISE=1` registers `_zerollama-fleet._tcp` for agents.
- **Package:** `fleet/mdns/` (register, browse, peer URL helpers).

Doc: [docs/fleet-management.md](docs/fleet-management.md#lan-discovery-f4-mdns).

### Qwen 3.5 / 3.6 on Apple Silicon (Jun 2026)

**Why:** Library tags like `qwen3.6:latest` (`qwen35moe`) and LM Studio–style `qwen35` VL GGUFs became loadable after the b9611 pin, but Mac operators hit **three independent failures**: Go-engine Metal SIGSEGV during init, llama.cpp `rope.dimension_sections` mismatch on published blobs, and missing unary Metal kernels on first decode. Each looked like one “Metal bug”; fixes belong to different layers.

**What shipped:**

- **Engine routing (`llm/server.go`):** On **darwin**, `qwen35`, `qwen35moe`, and `qwen3next` use the **legacy llamarunner** (CGO llama.cpp + Metal) instead of the Go ollama engine. **Why:** `OllamaEngineRequired()` sends these archs through `ggml.New()`; a C Metal segfault does not surface as a Go error, so there is no fallback. Legacy path + mtmd handles qwen35 after llama.cpp arch support landed.
- **In-process compat (`llama/compat/`):** Wired the same Ollama GGUF translation layer used by **llama-server** into the **CGO llamarunner** (`compat.go`, hooks in `llama-model-loader.cpp` / `mtmd/clip.cpp`, import from `llama/llama.go`). **Why:** Published qwen35moe stores M-RoPE `rope.dimension_sections` as **3** elements; llama.cpp expects **4**. Compat existed but only ran on CMake-fetched builds—not the Mac default binary.
- **Metal shader embed:** `build_zerollama_mac.sh` runs `go generate ./ml/backend/ggml/ggml/src/ggml-metal/` before `go build`. **Why:** macOS JIT-compiles from embedded `ggml-metal-embed.metal`; when ggml adds kernels (e.g. `kernel_unary_f32_f32` for sigmoid in gated SSM) without regenerating the embed, **load succeeds** but **first token** crashes with `Function kernel_unary_f32_f32 was not found`.
- **Defensive ggml backend init:** Skip nil device backends in `ml/backend/ggml/ggml.go` scheduler setup (belt-and-suspenders for Metal device init edge cases).

**Operator notes:**

- Rebuild + **restart serve** after pulling.
- Cap **`num_ctx`** (2048–8192) for first tests; `n_ctx_seq < n_ctx_train` log is informational.
- `tensor API disabled for pre-M5` on M4 Max is **expected**, not the crash cause.

Doc: [docs/qwen35-apple-silicon.md](docs/qwen35-apple-silicon.md).

### Fleet management node (F3)

**Why:** Agents and integrations often see **many zerollama hosts**, not one. Per-node schedulers answer local FIFO and VRAM correctly but not “which box has model M warm?” Scatter-gather and long reservation quotes **waste GPU work** on constrained fleets. F3 adds a **thin management process** that polls F2 `/api/status`, builds a warm-model map, and returns `{url, node_id}` — it never loads or evicts remotely.

**What shipped:**

- **`zerollama fleet serve`** — static `ZEROLLAMA_FLEET_PEERS`; poll interval default 3s; listen default `0.0.0.0:11450`.
- **HTTP API:** `GET /health`, `GET /api/fleet/status`, `POST /api/fleet/assign` (warm-first, lowest-queue routing; `warm_only`, `exclude`).
- **Package:** `fleet/` (manager, assign logic, tests); env `ZEROLLAMA_FLEET_*` in `envconfig`.

Doc: [docs/fleet-management.md](docs/fleet-management.md), [docs/fleet-scheduling.md](docs/fleet-scheduling.md#shipped-f3-management-node-v0).

### macOS runtime serve logging

**Why:** `./scripts/serve/serve_mac_runtime.sh` backgrounded sidecar and Go to log files with no terminal progress — operators thought the script hung. Startup now prints wait dots, log paths, and a ready banner with `tail -f` hints.

**What changed:** `scripts/runtime/macos_runtime_serve_lib.sh`, `scripts/serve/serve_mac_runtime.sh` — `MACOS_RT_LOG`, `MACOS_GO_LOG` documented.

### llama.cpp b9611 + MLX pin bump

**Why:** Stay on current upstream llama.cpp (latest tag `b9611`, ahead of vanilla Ollama’s `b9509`) with a **reviewable 14-patch series** instead of in-tree-only deltas. MLX pins aligned with upstream Ollama for MTP/speculation parity. **Why not wait for vanilla Ollama:** zerollama’s in-process ggml runner needs Ollama deltas (no-alloc fit, device props, grammar) on a **clean vendor base**; lagging b9509 blocked Metal fixes and Phase 17 mergeability.

**What shipped:**

- **Pin:** `LLAMA_CPP_VERSION=b9611`, vendor `vendor/llama-cpp-b9611/`, **14 patches** (0013 mtmd C API, 0014 ollama_vocab grammar; 0011 rebased for b9611 CUDA struct layout).
- **Sync:** `./scripts/vendor/sync_vendor_llama.sh` (replaces `sync_vendor_b9509.sh` shim); strips mtmd CLI `main()` after rsync.
- **Go fixes:** `llama.go` mtmd `bitmap_wrapper` + `placeholder=false` (b9611 API); `build-info.cpp` @ `1aefee58`.
- **Build:** `./scripts/build/build_zerollama_mac.sh` uses `GOFLAGS=-mod=mod` so CGO builds succeed when `vendor/` is incomplete.
- **Sibling runtime tree:** `../llama.cpp` @ b9611 + `./scripts/build/build_llama_server.sh` for vanilla `llama-server` / `libllama.dylib` (Python runtime subprocess — separate from patched in-tree ggml).
- **MLX:** `MLX_VERSION` → `2165dc08`, `MLX_C_VERSION` → `fba4470b` (upstream Ollama pins). Rebuild dylibs after pin bump — see **MLX dylib rebuild** below.

Doc: [docs/ggml-b9509-migration.md](docs/ggml-b9509-migration.md) (filename kept; pin is b9611).

### MLX dylib rebuild (Jun 2026)

**Why:** `MLX_VERSION` / `MLX_C_VERSION` are independent of the ggml llama.cpp pin — safetensors inference uses **`libmlx.dylib` + `libmlxc.dylib`**, not CGO ggml. Bumping pins without rebuilding leaves `mlxrunner` on stale Metal code (wrong kernels, ABI drift vs regenerated Go/C shims). Use **`BUILD_MLX=1 ./scripts/build/build_zerollama_mac.sh`** (dev) or **`./scripts/build/build_production_mac.sh`** (release layout) after pin bumps.

**What shipped:**

- Rebuilt **Metal v3** (macOS 14+) and **Metal v4** (macOS 26+ / NAX) at pins `2165dc08` / `fba4470b`.
- **Install layout:** `dist/darwin-arm64/lib/ollama/mlx_metal_v3/` and `mlx_metal_v4/` (`libmlx.dylib`, `libmlxc.dylib`, `mlx.metallib`).
- **Dev discovery:** `build/metal-v4/lib/ollama/libmlxc.dylib` (repo-root `./zerollama doctor` loads newest variant tree).
- **`GOFLAGS=-mod=mod`** in `build_production_mac.sh` — **why:** CMake runs `go generate ./x/...` during MLX configure; incomplete `vendor/` otherwise aborts configure.

**Operator commands:**

```bash
./scripts/mlx/ensure_mlx_sources.sh          # verify ../mlx ../mlx-c have pinned SHAs
export GOFLAGS=-mod=mod
./scripts/build/build_production_mac.sh      # MLX + release zerollama → dist/darwin-arm64/
# MLX dylibs only (no Go binary): cmake --build build/metal-v3 --target mlx mlxc && cmake --install …
```

Doc: [docs/apple-silicon-metal.md](docs/apple-silicon-metal.md#mlx-engine-optional), [docs/mac-dev-setup.md](docs/mac-dev-setup.md#dev-vs-production-mlx-layout).

### Apple Silicon sign-off (Jun 2026, M4 Max)

**Why:** Operators need a repeatable gate beyond `doctor` — runtime Metal inprocess is the **daily Mac path** (`apple_silicon.yaml`), while qwen35 validates the **Go ollama-engine ggml** path on the same Metal device. Sign-off proves Phase 13–15, optional qwen35, and tools without CUDA-centric `gpu_5080_session.sh`.

**One command (recommended):**

```bash
LLAMA_CPP_ROOT=../llama.cpp ./scripts/build/build_llama_server.sh
RUN_E2E_QWEN35=1 RUN_E2E_QWEN35_MODEL=qwen3.6:latest ./scripts/gpu/metal_signoff.sh
```

**Why qwen35 before Phase 15 in the script:** Phase 15 stops the runtime sidecar on exit; qwen35 needs `:8081` for training-handoff and resume after ggml unload.

**Passed (GPU, Jun 2026):**

- **Phase 13:** `/tmp/metal-session.json` — `metal-unified` probe, L1 GPU profile, autotune catalog.
- **Phase 14:** inprocess from YAML (`llama_backend_source=config`); generate/chat/stream; tokenize + render-chat; Go proxy.
- **Qwen35 (opt-in):** `qwen3.6:latest` via Go `--ollama-engine`; generate + API unload; thinking field OK.
- **Phase 15:** KV decode hook (`kv_decode_steps`); multi-seq (`llama_parallel_slots=2`, `kv_inprocess_n_seq_max=2` with `ZEROLLAMA_GPU_PROFILE=0`).
- **Tools:** runtime + proxy `/api/chat` and `/v1/chat/completions` with tools (HTTP 200, not legacy 501).
- **CPU (no model load):** Phase 12 golden, Phase 15 KV native CI, `go test ./server/... -short`, coordination smoke.

**Known gaps (documented, not blockers for b9611):**

| Gap | Why it happened | Fix (Jun 2026) |
|-----|-----------------|----------------|
| **Proxy v1 stream hang** | SSE missing `[DONE]` on errors; curl waited for EOF | Runtime always emits `[DONE]`; Go proxy flushes SSE; e2e `--max-time` via `RUN_E2E_STREAM_MAX` |
| **Legacy ggml + runtime Metal** | Two stacks on one device | Scheduler blocks ggml when runtime `llama_server=true`; legacy smoke skips on darwin unless `RUN_E2E_LEGACY_FORCE=1` |
| **`num_gpu=0` init Metal** | Metal registered at first ggml init | First CPU-only load sets `GGML_DISABLE_METAL` before backend register |
| **`go test ./ml/backend/ggml/...`** | Dummy GGUF fixture segfault | `doctor` + `metal_signoff.sh` as gate |
| **MLX dylib** | Pin bump without rebuild | `./scripts/mlx/ensure_mlx_sources.sh` + `GOFLAGS=-mod=mod ./scripts/build/build_production_mac.sh` |
| **Qwen35 in full gate** | Phase 15 killed sidecar before qwen35 resume | qwen35 runs **before** Phase 15 in `m3_metal_signoff.sh`; opt-in `RUN_E2E_QWEN35=1` |
| **Phase 15 multiseq vs L1** | `apple-silicon-128g` forced `n_parallel=8` | Multiseq step uses `ZEROLLAMA_GPU_PROFILE=0` + temp yaml `llama_parallel_slots: 2` |
| **Phase 14 + L3 cache_prompt** | Inprocess worker lacked kwarg | Accept/ignore `cache_prompt` on inprocess and wheel backends |
| **Scheduler 400/503** | Contention returned generic HTTP 500 | `handleScheduleError` maps runtime-routed → 400, Metal contention → 503 |

Scripts: `./scripts/gpu/metal_signoff.sh`, `./scripts/runtime/qwen35_mac_smoke.sh`. Guide: [docs/apple-silicon-metal.md](docs/apple-silicon-metal.md).

### ggml @ llama.cpp b9509 (real vendored tree)

**Why this release:** Zerollama’s **in-process ggml Metal runner** was pinned to an old llama.cpp base with **27/36 patches failing** on upstream’s current pin (`b9509`). Overlay-regenerating patches produced **multi‑MB fork snapshots**, not a clean b9509 ggml. Without rebasing, every upstream bump increased merge pain and blocked Phase 17 alignment with vanilla Ollama’s `LLAMA_CPP_VERSION`.

**What shipped:**

- **Clean b9509 base:** `vendor/llama-cpp-b9509/` + **12 rebased patches** in `llama/patches/` (backup: `llama/patches.pre-b9509-20260612/`).
- **Synced in-tree trees:** `ml/backend/ggml/ggml/` and `llama/llama.cpp/` via `./scripts/vendor/sync_vendor_b9509.sh`; `Makefile.sync` pins `FETCH_HEAD=b9509`.
- **Ollama API ports for b9509:** `ggml_backend_sched_new_ext` (fit/no-alloc sizing), extended `ggml_backend_dev_props` + NVML/ROCm mem helpers, `ollama_vocab` grammar, mtmd C API, LoRA plural API, device props in Go.
- **CGO build fixes for b9509 common/:** `jinja_wrap.cpp`, `httplib_wrap.cpp`, `llama/build-info.cpp`; exclude CLI `main()` from mtmd; `models.go` include path for `src/`.
- **Build verified:** `go build`, `zerollama doctor` on Apple Silicon.

**Not in this release:** full CUDA no-alloc pool overrides (`reserving_graph` stubs); automatic replacement of ggml with Go→llama-server (Phase 17 remains opt-in).

Doc: [docs/ggml-b9509-migration.md](docs/ggml-b9509-migration.md).

### Wan text-to-video (v1)

**Why this release:** OpenAI clients expect async video jobs (`POST /v1/videos` → poll → download). Wan runs in a **separate PyTorch stack** from GGUF chat; bolting it into the runtime or ggml runner would duplicate VRAM policy and job lifecycle. Reusing the **embedded training worker** (`run_script`) gives one GPU handoff story (Phase 8 broker, T6 defer queue) without a second public daemon.

**What shipped:**

- OpenAI-compatible **`POST /v1/videos`**, **`GET /v1/videos/:id`**, **`GET /v1/videos/:id/content`** for local Wan presets (`wan2.1-t2v:1.3b`, `wan2.2-ti2v-5b`, 16g manifests).
- **`video_gen`** capability + `video_generation` / `backend_paths` in model config; config-only registration (no GGUF blob).
- Go **`server/video_generate.go`**: payload build, frame caps on 16g, artifact sandbox under `$OLLAMA_MODELS/generated/`.
- Python **`scripts/video/wan_video_generate.py`** wrapper → upstream `generate.py` with `--save_file`, venv interpreter, progress lines.
- **`training.py`**: `python_bin` / `WAN_VENV` for wrapper; `{job_id}` in `output_path` and `WAN_OUTPUT_PATH`; merged stderr/stdout logging.
- **Defer queue**: `defer-*` ids pollable with `trainingworker.ErrJobNotFound` → HTTP 404; `videoModel` / `videoSize` / stable `submitted_at` on wire.
- **CLI:** `zerollama run <wan-model> "prompt"` via `x/videogen`.
- Install: `scripts/video/install_wan_video.sh`, `scripts/video/register_wan_models.sh`.

**Not in v1:** list/cancel on `/v1/videos`, TI2V image input, `:cloud` video, artifact TTL.

Doc: [docs/wan-t2v.md](docs/wan-t2v.md).

### Phase 15 — native KV block pool (v0)

**Why:** Continuous batching allocates KV block ids on every scheduler tick; pure Python competes for the GIL when training and runtime share one embedded interpreter.

**What shipped:**

- C extension `runtime.kv._kv_native` (`runtime/native/kv_block_pool.c`) — same API as Python `BlockPool`.
- Opt-in `ZEROLLAMA_RUNTIME_KV_NATIVE=1`; default remains Python; `/health` `kv` object reports `backend`, `native_requested`, `native_available`, and a `note` when the env is set but the extension is missing.
- CI: `setup.py build_ext --inplace` + `test_kv_native_parity.py` in regression workflow; `./scripts/phase/phase15_kv_native_ci.sh`.

**Phase 15 v1 (scheduler KV):** `/health` `kv_scheduler`; block reserve for `max(prompt+max_tokens, num_ctx)`; subprocess `kv_slot` → llama-server `id_slot`.

**Phase 15 v2 (in-process multi-seq KV):** when `llama_parallel_slots`>1 and backend `inprocess`, one shared `llama_context` with per-sequence KV clear + `seq_id` batch decode; scheduler assigns `kv_slot` for in-process too.

**Phase 15 audit fixes:** `resolve_parallel_slots()` — `-np` in `llama_server_args()` wins over YAML; admission uses real tokenize when GGUF known; `/health` `kv_inprocess_n_seq_max` only when multi-seq; docs aligned.

**Phase 15 v3 (logical KV bind):** `runtime/runtime/kv/bind.py`; `/health` `kv_bind` + per-request `block_ids`; `assert_kv_capacity` at forward; in-process `kv_token_budget`; batch uses real tokenize + scheduler `kv_slot` in parallel decode.

**Phase 15 audit fixes:** clarify `block_ids[i]` = pool id for sequence page *i*; batch `kv_token_budgets`; in-process multi-seq `n_ctx` capped by pool `num_blocks * block_size`; doc two-KV-cap note for subprocess.

**Phase 15 v4:** `kv/physical.py` — llama `seq_pos` vs PA reserve; `/health` `kv_physical` + `kv_native_scheduler_tick`; native `scheduler_tick()`; `ZEROLLAMA_RUNTIME_KV_PHYSICAL_STRICT`.

**Phase 15 audit (health):** `physical_bind_level` when in-process weights loaded; `kv_physical` PA-only snapshot + note for single-seq; strict errors include `request_id` / `kv_slot`.

**Phase 15 v5:** `/health` `kv_scheduler_tick` `{value, source}` with Python fallback; `kv_physical_recent` ring buffer; expanded `phase15_kv_native_ci.sh`; [handoff-phase15-native-kv.md](docs/handoff-phase15-native-kv.md).

**Phase 15 v6:** native `decode_step(n)` on in-process `llama_decode`; `/health` `kv_decode_steps`; env `ZEROLLAMA_RUNTIME_KV_DECODE_HOOK`.

**Phase 15 audit (health/ops):** `kv_decode_steps` inactive reason for subprocess; per-completion `kv_decode_steps` on generate/stream; `kv_physical_recent` mismatches only; atomic C counters; scheduler tick health note.

**Not in v0:** GPU KV tensors, in-process llama KV bind, native decode loop. Doc: [docs/phase15-native-kv.md](docs/phase15-native-kv.md).

### Phase 14 — in-process llama forward (summary)

**Why this release:** The Python runtime already scheduled work and estimated VRAM, but every completion still crossed loopback HTTP to a `llama-server` child. That added latency, complicated GPU handoff with training/ggml, and left Go tools render-chat without a real tokenizer on the runtime path (heuristic truncation only).

**What shipped:**

- Three forward backends behind `ZEROLLAMA_RUNTIME_LLAMA_BACKEND`: **`subprocess`** (default), **`inprocess`** (ctypes + pinned `libllama.so`), **`llama-cpp-python`** (pip wheel, CPU-default GPU layers).
- **`POST /internal/tokenize`** (vocab-only, cached) + Go render path → `truncate_mode: tokenize` when no ggml runner.
- Ollama-shaped **sampling** on all backends (`runtime/runtime/worker/sampler_options.py`).
- Operator/smoke scripts: `phase14_serve_env.sh`, `phase14_backend_smoke.sh`, `phase14_both_backends.sh`; `RUN_E2E_PHASE14=1` in `e2e_runtime_smoke.sh`.
- **5080:** inprocess GPU + wheel CPU smokes pass; see [docs/gpu-5080-operator-guide.md](docs/gpu-5080-operator-guide.md).

**Not in scope (Phase 15+):** native KV scheduler, grammar/mirostat in-process, subprocess removal as default.

Full design: [docs/phase14-inprocess-llama.md](docs/phase14-inprocess-llama.md).

### Fixed

- **Phase 14 render tokenize (embed):** Go `tokenizeForRuntimeModel` uses `runtimeProxyConfigured()` (embedded loopback via `runtimeworker.BaseURL()`), not only `ZEROLLAMA_RUNTIME_URL`. **Why:** embed leaves URL unset; render-chat stayed `truncate_mode=heuristic`.
- **Phase 14 smoke proxy:** `RUN_E2E_PHASE14=1` sends `X-Zerollama-Runtime: 1` on Go proxy steps (smoke-only; `ZEROLLAMA_RUNTIME=1` is not `OLLAMA_RUNTIME_ALL`).
- **server sched tests:** `TestMain` unsets runtime env from operator shells so synthetic GGUF sched tests still exercise ggml load.
- **Phase 14 sign-off script:** `scripts/phase/phase14_both_backends.sh` restarts serve for `inprocess` and `llama-cpp-python` smokes (embed-safe: does not export `ZEROLLAMA_RUNTIME_URL` to serve).
- **Phase 14 llama-cpp-python GPU:** wheel backend defaults to **CPU** (`n_gpu_layers=0`); GPU via `-ngl` or `ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS` (negative env values fall back to CPU with warning). **Why:** cu124 wheel can `free(): invalid pointer` on GPU decode on some hosts while ctypes inprocess works.
- **Phase 14 `phase14_both_backends.sh`:** fails if every backend skipped; clears stale `RUN_E2E_INPROCESS` between runs. **5080 guide:** Phase 14 sign-off checklist in [docs/gpu-5080-operator-guide.md](docs/gpu-5080-operator-guide.md).
- **Phase 14 in-process stream:** `llama_free` / sampler no longer run before stream chunks are consumed (segfault on streaming generate/chat). **Why:** `return generator` inside `try/finally` freed the context immediately.
- **Phase 14 lib path:** auto-detect `libllama.so` via zerollama repo root (`parents[3]`), not `/` + wrong sibling.
- **Phase 14 load parity:** apply `-mg` / `-sm` / `-ts` from `llama_server_args()`; reject speculative/draft on in-process backend.
- **Phase 14 backend config:** `resolve_llama_backend()` uses `RuntimeConfig.llama_backend` when env unset.
- **Phase 14 YAML `llama_backend`:** autoconfig files (e.g. `single_gpu.yaml`) load `llama_backend`; env still wins; invalid values fail at load (`canonical_llama_backend`). `/health` `llama_backend_source`: `env` | `config` (explicit YAML key) | `default` (packaged subprocess). **`llama_cpp`** on wheel backend reports `gpu_mode` / `n_gpu_layers` for operator GPU offload visibility.
- **Embed port conflict:** Go embed preflight refuses busy loopback `:8081` and matches `/health` `embed_boot` to this process (no silent attach to stale runtime). **Why:** `address already in use` while Go logged embed success on cudallama-style restarts.
- **Phase 14 in-process decode:** ctypes path uses heap `llama_batch_init` batches with explicit `pos[]` (fixes `llama_batch_get_one` UAF and uninitialized positions). **Why:** inprocess generate returned 502 `llama_decode failed` on multi-token prompts.
- **Phase 14 YAML smoke:** `phase14_yaml_config_smoke.sh` infers `RUN_E2E_*` backend flags from `/health` when `llama_backend_source=config` (rejects subprocess).
- **Phase 14 status:** ROADMAP exit criteria 3–4 signed off on 5080 dev host (`phase14_inprocess_smoke`, `phase14_wheel_cpu_smoke`).

- **Phase 13 VRAM audit:** IQ2_XS (`GGML_TYPE_IQ2_XS`) block size corrected to 74 bytes (was 138 — over-estimated weights). `VRAM_ESTIMATE_FACTOR` applies once on the outer estimate (speculative draft no longer scaled twice). Calibration uses fresh probe reads and precomputed estimates; multi-GPU `scope_warning` on `/health`. **Why:** audit found false rejections on IQ2_XS models and inflated draft VRAM.
- **Phase 11 dequeue fairness:** scheduler no longer stalls **`priority: normal`** when inference-first metrics are on (only **`low`** at queue head waits). **Why:** enqueue already allowed normal chat under defer/ggml/backlog; dequeue had been blocking all non-high work.
- **Phase 13 load `num_ctx` parity:** `generate` / `stream_generate` pass **admitted** `active.num_ctx` to `llama-server` (not pre-clamp request values). **Why:** VRAM clamp and precheck applied capped context to the queue but load still used the client’s larger `num_ctx` → estimate/load mismatch and possible OOM.
- **Phase 12 tools + clamp:** tools chat render and load share `InferenceEngine.resolve_num_ctx_for_request()` so Go `/internal/render-chat` truncation and `-c` agree when `VRAM_CLAMP_NUM_CTX` is on. **Why:** render used uncapped ctx while load used capped ctx.
- **`runtime_vram_estimate.sh`:** `--num-ctx` now works (`NUM_CTX` exported before JSON payload). **Why:** shell variable was not visible to Python builder.

### Added

- **Phase 14 5080 session:** `RUN_E2E_PHASE14_SIGNOFF=1` and `RUN_E2E_PHASE15=1` in `gpu_5080_session.sh` / `gpu_smoke_all.sh` (sign-off needs `LLAMA_CPP_LIB`).
- **Phase 14 5080 sign-off:** `phase14_5080_signoff.sh` — one-shot gate (`both_backends` + YAML config full + Phase 15 multi-seq). ROADMAP Phase 14 marked **Done**; subprocess remains packaged default.
- **Phase 14 smoke:** `phase14_backend_smoke.sh` and `RUN_E2E_PHASE14=1` in `e2e_runtime_smoke.sh`; sign-off wrappers `phase14_inprocess_smoke.sh`, `phase14_wheel_cpu_smoke.sh`; provenance via `phase14_yaml_config_smoke.sh`, `phase14_subprocess_default_smoke.sh`; optional `phase14_wheel_gpu_smoke.sh` and `phase14_enable_yaml_inprocess.sh`; optional `RUN_E2E_PHASE14=1` in `gpu_smoke_all.sh` / `gpu_5080_session.sh`.
- **Phase 15 inprocess KV smoke:** `phase15_inprocess_kv_smoke.sh` — self-contained inprocess serve + asserts `kv_decode_steps` on generate and `/health` (v6 decode hook); KV snapshot assertion now runs in a subshell so `exec` in the backend smoke does not swallow it.
- **Phase 15 in-process sign-off:** `phase15_inprocess_signoff.sh` — self-contained KV decode hook + multi-seq smokes; `phase15_inprocess_kv_smoke.sh` now starts its own serve.
- **Phase 15 KV snapshot helper:** `smoke_runtime_assert_kv_snapshot()` in `runtime_smoke_lib.sh`; used by Phase 15 GPU smokes.
- **Phase 14 5080 sign-off:** step 3 now runs full `phase15_inprocess_signoff.sh` (KV hook + multi-seq + snapshot).
- **CI:** regression workflow runs `go test ./x/runtimeworker/...` (embed `:8081` preflight tests).
- **Phase 14 YAML full smoke:** `phase14_yaml_config_full_smoke.sh` — optional #6 without editing packaged `single_gpu.yaml`.
- **Phase 14 render tokenize:** `llama-cpp-python` backend uses wheel `vocab_only` for `/internal/tokenize` (no `libllama.so`); subprocess can fall back to wheel vocab when ctypes lib is missing.
- **Phase 14 llama-cpp-python backend:** `ZEROLLAMA_RUNTIME_LLAMA_BACKEND=llama-cpp-python` uses the pip wheel (no `libllama.so` build); tokenize via loaded model when GGUF matches.
- **Phase 14 sampling:** `options.temperature`, `top_k`, `top_p`, penalties, and `seed` forwarded to subprocess `llama-server` and in-process libllama sampler chains; no sampling keys → greedy in-process default. `temperature: 0` sends `temperature: 0` on subprocess; render-chat tokenize falls back to heuristic only when runtime is unreachable (not on HTTP/model errors).
- **Phase 14 tokenize for render:** `POST /internal/tokenize` (libllama vocab-only, cached); Go `/internal/render-chat` uses `truncate_mode: tokenize` when no ggml runner and runtime URL is set. **Why:** tools chat truncation matched ggml only when a runner was loaded; runtime path had heuristic-only.
- **Phase 14 (v1) in-process llama:** `ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess` loads pinned `libllama.so` in the runtime process (ctypes); subprocess `llama-server` remains default. `/health` includes `llama_backend`. Doc: [docs/phase14-inprocess-llama.md](docs/phase14-inprocess-llama.md). **Why:** remove loopback HTTP and second process on the hot path; foundation for in-process tokenize (render) and Phase 15 native KV.
- **Phase 13 operator CLI:** `scripts/runtime/runtime_vram_estimate.sh` — `POST /internal/vram-estimate` for a GGUF (budget, `suggested_max_num_ctx`, host RAM). **Why:** tune context and quant choice before load on a tight GPU.
- **Phase 13 docs:** [docs/phase13-runtime-vram.md](docs/phase13-runtime-vram.md) — WHY-oriented estimate/clamp/autoconfig guide and 5080 workflow.
- **`InferenceEngine.resolve_num_ctx_for_request()`** — resolve + optional clamp without queuing; used by `_admit_one`, `/api/generate`, `/api/chat` (including tools render). **Why:** one code path for context policy; avoids render/load drift.
- **Phase 11 VRAM headroom env:** `ZEROLLAMA_RUNTIME_VRAM_MIN_FREE`, `ZEROLLAMA_RUNTIME_TRAINING_VRAM_RESERVE` (size strings; defaults 1 GiB / 2 GiB). `/health` exposes `vram_min_free_configured` and `vram_training_reserve_configured`. **Why:** 5080 operators can tune without editing Python constants.
- **Phase 13 `num_ctx` clamp (opt-in):** `ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX` default **off** (`auto`/`1` lowers request ctx to suggestion); API/stream include `vram_num_ctx` when clamped; tools stream forwards `vram_num_ctx`. **Why:** avoid silent context reduction; still available for single-GPU smoke.
- **GPU smokes:** `e2e_runtime_smoke.sh` calls `/internal/vram-estimate` and post-generate `vram_calibration`; `e2e_coordination_smoke.sh` prints Phase 13 health fields; `serve_gpu_example.sh` documents single-GPU VRAM env defaults. **Why:** CI/ops can validate estimate path without reading Python.
- **Phase 13 on `/v1/chat/completions`:** `prepare_v1_chat()` + `v1_request_options()` — resolve/clamp `num_ctx`, tools render parity, optional `options.gguf` on JSON body; non-stream `vram_num_ctx` in response. **Why:** OpenAI clients hitting :8081 directly should not bypass estimate/clamp policy.
- **Go v1 runtime proxy (Phase 9 + 13):** `runtimeV1ProxyOptions` injects manifest `options.gguf` and merges client `options` before forwarding to Python `/v1/chat/completions` (parity with `/api/chat` proxy). v1 legacy gate matches `/api/chat` for logprobs, vision, think, and OpenAI `reasoning` / `reasoning_effort`. **Why:** OpenAI-shaped clients via `:8080` had no GGUF path for VRAM precheck/clamp on the runtime.
- **GPU smoke:** `e2e_runtime_smoke.sh` exercises generate/chat/v1 (non-stream + stream) on runtime and Go proxy; `gpu_smoke_all.sh` runs coordination + full GPU/proxy smokes; `gpu_health_report.sh` prints calibration/autotune tuning hints from `/health`.
- **Go proxy tests:** chat tools + options forwarding; v1 SSE stream passthrough.
- **`gpu_health_report.sh`:** shared formatter `runtime.gpu_health_report` + tests; `/health` keys match calibration/autotune schema.
- **v1 `v1_request_options`:** top-level `max_tokens` promoted to `options.num_predict` for Phase 13 VRAM resolve (parity with Go proxy).
- **CI:** `scripts/check_gpu_scripts.sh` in regression workflow; `python -m runtime.gpu_health_report` CLI.
- **5080 operator guide:** [docs/gpu-5080-operator-guide.md](docs/gpu-5080-operator-guide.md) — WHY-oriented single-GPU workflow (session gate, API unload, snapshot, harmony/host-RAM limits).
- **Apple Silicon / Metal (M1–M2):** [docs/apple-silicon-metal.md](docs/apple-silicon-metal.md); `runtime/configs/apple_silicon.yaml` autoconfig on darwin; `metal-unified` VRAM probe (`vm_stat`); `read_host_memory()` for host budget on Mac; `macos_metal_smoke.sh`. **Why:** unified memory is not NVIDIA VRAM — CUDA `single_gpu.yaml` and `nvidia-smi` probes rejected valid Mac loads.
- **Apple Silicon fix:** `check_gguf_host_budget` and `/health` host mem now call `read_host_memory()` (was Linux-only); `vm.swapusage` parsed from real macOS `free = N.M` format. **Why:** audit found host pre-load checks silently skipped on Darwin and swap budget always zero.
- **MLX routing (M4):** [docs/mlx-routing-policy.md](docs/mlx-routing-policy.md); `modelUsesRuntimeInference` rejects `IsMLX()` even with mistaken Modelfile backend; Go tests. **Why:** safetensors must stay on mlxrunner, not Python GGUF runtime.
- **Mac session gate (M3):** `gpu_metal_session.sh` — smoke + Phase 13 snapshot + optional Phase 14. **Why:** Mac operators need the same repeatable gate as `gpu_5080_session.sh`.
- **Phase 13 Python:** `single_gpu.yaml` `vram:` block applied at runtime start (`vram_yaml_defaults.py`). **Why:** 16GB autoconfig installs get admission/autotune defaults without a long systemd env block; operator env still wins.
- **Phase 13 snapshot:** `python -m runtime.gpu_snapshot` reads session JSON and prints env hints (autotune `persist`, budget warnings, harmony skip). **Why:** portable tuning record after `gpu_5080_session.sh` without re-scraping `/health`.
- **GPU smoke unload:** `smoke_unload_ggml_runners` evicts stale ggml via `/api/ps` + `keep_alive:0` before Phase 8 broker; `mapfile` per model; `SMOKE_UNLOAD_MAX_WAIT` default 30s; 503+runner retry. **Why:** Go 503 before broker left runners loaded; `pkill` bypassed public unload and false-positive `pgrep` on shell lines.
- **GPU smoke:** `gpu_harmony_capture.sh` uses API unload instead of `pkill`; `gpu_5080_session.sh` runs snapshot + recommendations.
- **GPU smoke:** optional `RUN_E2E_TOOLS=1` for `/api/chat` with tools on `:8081` and `:8080` proxy (asserts 200 + not legacy 501); Go `TestRuntimeChatProxyStream`.
- **gpu_health_report:** export hint only when `suggested_estimate_factor` is in 0.1–3.
- **Phase 12 render:** `/internal/render-chat` uses ggml `Tokenize` when a runner for the model is already loaded (`truncate_mode`: `tokenize` | `heuristic` | `none`); `truncated` true only when prefix messages dropped; `has_tool_support` uses prepared messages; Python tools `meta.truncate_mode` / `meta.truncated` mirror Go; golden parity tests `runtime_render_golden_test.go`; single `prepareRenderMessages` in handler via `renderChatPromptPrepared`.
- **Phase 12 parse golden:** `runtime_parse_golden_test.go` — functiongemma one-shot + streaming chunks vs `model/parsers`; render→parse roundtrip.
- **Phase 12 truncation:** ggml `chatPrompt` uses `chatPromptTokenBudget` (same reserve as render-chat when `num_predict > 0`); single `prepareToolsForRender` in render handler.
- **Phase 12 parse golden:** harmony tool-call fixture in `runtime_parse_golden_test.go`; Python asserts `requires_go_tool_parser` for harmony parser meta.
- **GPU ops:** `gpu_harmony_capture.sh` + `--harmony` on `phase12_capture_tool_transcript.sh` (fix gguf override for pulled models); `gpu_5080_session.sh`; Phase 11 threshold env overrides; `phase12_golden_ci.sh` (`all|go|py`); clamp/snapshot smokes. Audit: legacy only via `RUN_E2E_LEGACY_ONLY` / legacy-only e2e; `smoke_prepare_vram` fails on 503+ggml runner; Phase 11 doc aligned with env overrides.
- **Go:** `TestRuntimeV1ChatCompletionsProxyForwardsTools`; CI `check_gpu_scripts` greps tools smoke markers; **fix** regression workflow runs `check_gpu_scripts` from repo root (not `runtime/` cwd).
- **GPU smoke:** `RUN_E2E_TOOLS=1` also exercises `/v1/chat/completions` with tools on `:8081` and `:8080` proxy.
- **Phase 12 render truncation:** `/internal/render-chat` heuristic reserves `num_predict` (or ~256 default) for completion headroom; Python tools path passes `n_predict` into Go render.
- **GPU smoke:** `RUN_E2E_VRAM_CLAMP=1` asserts `/health` `vram_num_ctx_policy.clamp_enabled` and probes high-`num_ctx` generate for `vram_num_ctx` or budget reject; default `OLLAMA_HOST` is `:8080`.
- **v1 parity:** `v1_max_tokens` accepts float JSON numbers and `options.num_predict`; `think:false` stays on runtime (Go + Python).
- **T6 training queue policy (Go):** optional `ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE` rejects training submit while ggml and/or runtime inference are busy; **`defer-*` job queue** when `ZEROLLAMA_TRAINING_QUEUE_ON_BUSY` (or `priority: low` / `queue_on_busy`) with auto-promote, tombstone TTL, retries, cancel, and list merge. Submit **`priority`**: `high` bypasses idle-wait, `low` prefers defer. **Why:** single-GPU hosts need inference-first scheduling without a second Python listener; batch training should wait for chat to finish instead of failing with opaque 409s.
- **`ZEROLLAMA_TRAINING_WAIT_GGML_LOADED`:** resident ggml runners (`OLLAMA_KEEP_ALIVE`) count as busy when idle-wait is on (opt-out for multi-GPU). **Why:** a loaded legacy model holds VRAM even with an empty queue.
- **Training blocks inference (monitor):** while training occupies GPU, pause ggml loads, evict runners, block runtime proxy (`ZEROLLAMA_BLOCK_INFERENCE_DURING_TRAINING`, default on). **Why:** training and chat must not fight for the same 16 GB without policy.
- **Phase 11 runtime admission (opinionated):** Python VRAM + inference-first policy with **two operator envs** only — `ZEROLLAMA_RUNTIME_INFERENCE_POLICY=off`, `ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM=0`. Thresholds, **2 GiB** training reserve, **1 GiB** min-free (1.5× for `low`), and backlog/defer/ggml constants live in code. **Why:** single-GPU operators should not tune a dozen `ADMISSION_*` flags; we measure and adjust constants instead. See [docs/phase11-runtime-admission.md](docs/phase11-runtime-admission.md).
- **Phase 11 enqueue GGUF precheck:** host + `check_gguf_vram_budget` **before** the waiting queue when the model path is known. **Why:** fail fast instead of filling the queue with work that cannot load.
- **Phase 11 inference priority:** `options.priority` (`high` / `normal` / `low` with aliases); high jumps queue and bypasses min-free gate (not model fit); `generate_batch` defaults to low. **Why:** align with Go training T6 without one global FIFO.
- **Scheduler VRAM re-check:** `SchedulerLoop.tick` re-runs admission before KV allocate (per-request `priority`). **`VRAM_MMAP_FACTOR`** scales weight estimate. **Why:** VRAM can drop while requests sit in queue; mmap’d GGUF may not need full tensor bytes on GPU.
- **Admit cleanup:** `cancel_waiting` on failed tick/batch; tick distinguishes `AdmissionRejected` (re-queue) vs misconfig (fail). **Why:** avoid stray queued requests after a failed admit.
- **Go → Python training GPU busy:** `POST /internal/training-gpu-busy` from training policy monitor; admission reserve follows Go `trainingOccupiesGPU`. **Why:** reserve headroom on direct `:8081` runtime traffic while training holds the card.
- **Go coordination on runtime `/health`:** `POST /internal/go-coordination` mirrors training defer queue counts and policy flags. **Why:** single-GPU operators see Go+Python queue state from the runtime sidecar.
- **Defer / ggml backlog (inference-first):** when Go mirror is fresh, **`priority: low`** is rejected at enqueue and stalled at dequeue; **normal** is not. Wired via `go-coordination` + `inference_policy.py` (no per-gate env). **Why:** inference-first SLO on a shared GPU without blocking default chat.
- **Ggml pause when runtime busy (Go):** `ZEROLLAMA_GGML_PAUSE_WHEN_RUNTIME_BUSY=auto` (on when runtime URL or embed is configured) pauses new ggml loads while `runtime_waiting + runtime_running` ≥ `ZEROLLAMA_GGML_PAUSE_RUNTIME_MIN_BACKLOG` (default 4); one `/health` probe per tick; mirror exposes `ggml_loads_paused`. **Why:** symmetric single-GPU policy — Python already blocks low-priority work when ggml is busy.
- **Phase 11 `/health` gates:** `admission.gates_active` uses explicit names (`low_would_wait`, `runtime_backlog_pressure`, …); legacy keys under `gates_active_compat`. **Why:** old `batch_backpressure: true` with `backlog: 0` confused operators — true means “low would wait,” not “everything blocked.”
- **Phase 13 startup env apply:** `ZEROLLAMA_RUNTIME_VRAM_APPLY_EXPORTED_ENV=1` loads `vram_estimate_factor.env` once at runtime start (skips when `VRAM_ESTIMATE_FACTOR` already set; autotune persist still wins per GGUF). **Why:** operators can `source` export without hand-copying into systemd/unit env.
- **Phase 13 KV head dims:** when GGUF omits `key_length`, derive from `attn_k` / `attn_v` tensor shapes + `embedding_length` / `head_count_kv`. **Why:** tighter KV VRAM estimates on sparse manifests.
- **Phase 13 quant KV bytes:** `VRAM_KV_BLOCK_LAYOUT=1` (default) uses ggml block layout for IQ/TQ/MXFP `key_type`/`value_type`; classic Q4/Q8 KV stays ≥2 bytes/element (conservative). **Why:** less KV over-estimate on IQ models without loosening Q4 admission.
- **T6 cross-queue FIFO:** global monotonic tickets (`POST /internal/cross-queue-seq`); Go mirror `fifo_go_oldest_*` / runtime `fifo_runtime_oldest` (waiting+running); Python blocks batch when **ggml** is ahead; ggml pending/loading yields to older runtime tickets; defer promotion waits for inference (runtime or ggml). **Why:** single-GPU ordering across ggml, runtime, and defer without operator knobs.
- **Coordination shutdown:** `finalizeInferenceCoordination` on daemon exit resumes ggml loads and pushes cleared `ggml_loads_paused` / `training_gpu_blocked` (avoids stale mirror until TTL). **Why:** rolling restarts should not leave Python thinking ggml is paused for 30s.
- **Phase 13 IQ/TQ block layouts:** VRAM weight estimates use ggml block sizes for IQ2/IQ3/IQ4/IQ1 and TQ1/TQ2 types (synced from ggml-common.h). `ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR` scales final estimates for operator calibration. **Why:** rare quants no longer over-count via fp16 fallback bytes/element.
- **Phase 13 probe calibration:** `ZEROLLAMA_RUNTIME_VRAM_PROBE_CALIBRATE=auto` records NVML/smi free VRAM before/after llama-server load; `/health` `vram_calibration` exposes `suggested_estimate_factor` (observed/raw). **Why:** operators can tune estimates on real hardware without automatic policy changes.
- **Phase 13 estimate autotune:** per-model factors in `STATE_DIR/vram_autotune.json` (v2); **`VRAM_AUTOTUNE_PERSIST=auto`**; **`VRAM_ESTIMATE_FACTOR_EXPORT=auto`** writes `vram_estimate_factor.env` + catalog after calibration. Pre-check uses `effective_vram_estimate_factor(gguf=...)`. **Why:** different models on one GPU; operators can `source` suggested env without hand-editing.
- **Phase 13 VRAM hardening:** per-layer `sliding_window` UINT32 arrays (hybrid SWA); unknown ggml KV types → fp16; Q4/Q8 sized conservatively; separate K/V dims; `ZEROLLAMA_RUNTIME_VRAM_SCRATCH_FACTOR`; cached `gguf_arch_hints` for `/health`; heuristic path uses resolved `num_ctx` consistently.
- **Phase 13 llama-server flag parity:** VRAM pre-check and `/health` `vram_estimate` parse config/`LLAMA_SERVER_EXTRA_ARGS` (`-c`, `-ngl`, `-np`) plus YAML `llama_parallel_slots`. **Why:** operators cap context with `-c 8192` but estimates used full GGUF `context_length` and ignored parallel slots.
- **Phase 13 speculative draft VRAM:** when `speculative.method` is a draft plugin, estimates add the draft GGUF (`--model-draft`, `--spec-draft-ngl`). **Why:** dual-model speculative decode needs both weight+KV budgets on a tight GPU.
- **Phase 13 ngram scratch:** ngram speculative methods add `ZEROLLAMA_RUNTIME_VRAM_NGRAM_SCRATCH_BYTES` (default 128MiB) to VRAM estimates. **Why:** ngram cache has no draft GGUF but still consumes GPU memory.
- **Phase 13 per-tensor GPU weights:** `VRAM_WEIGHT_TENSOR` sums GGUF tensors by layer for `-ngl`; `VRAM_WEIGHT_BLOCK_LAYOUT` uses ggml quant block sizes (Q4_0, Q4_K, …). Exact KV per GPU layer; `-ngl 0` skips GPU KV. **Why:** linear `ne×2` and full-layer KV over-estimated VRAM on partial offload and quants.
- **Phase 13 runtime VRAM (Python):** `ZEROLLAMA_RUNTIME_VRAM_PROBE` (`auto` / `nvml` / `nvidia-smi`), optional `nvidia-ml-py` via `pip install -e 'runtime/.[gpu]'`, unified-memory fallback (`host-unified`), KV scale from request `num_ctx` → env → GGUF `context_length`, layer scale from GGUF `block_count`. **Exact KV path:** `estimate_kv_cache_bytes` from GGUF layers, GQA heads, K/V dims, optional `attention.key_type`/`value_type` (quantized KV), `sliding_window` cap; `ZEROLLAMA_RUNTIME_VRAM_KV_EXACT` (default on); `/health` `vram_estimate`; request `options` passed into pre-check. **Why:** weights + explicit K+V avoids double-counting ctx/layer multipliers when metadata is complete.
- **T6 training night window (Go):** `ZEROLLAMA_TRAINING_ALLOWED_WINDOW` (e.g. `22:00-06:00`), `ZEROLLAMA_TRAINING_WINDOW_TZ`; `priority: high` bypasses; defer queue can hold jobs until the window opens when `ZEROLLAMA_TRAINING_QUEUE_ON_BUSY=1`. Invalid window env → **503** + warn-once log (fail closed). **Why:** first SLO hook for batch training on a shared GPU without a unified FIFO.
- **`gguf_model_hints`:** read block count / context length from GGUF metadata for VRAM estimates. **Why:** one file parse drives both layer and context scaling.
- **CI regression (Phase 10):** `.github/workflows/zerollama-regression.yaml` — Go `server`/`envconfig`/`trainingworker` + runtime pytest. **Why:** cross-language policy wiring regresses silently without a gate.
- **CPU training route smoke (T4 partial):** Go tests register `/api/train/*` and exercise `GET /api/train/status` without embedded Python. **Why:** catch route wiring breaks without CUDA on every PR.

- **Manifest → runtime (Phase 9):** Go proxy adds `options.gguf` from the Ollama manifest; Python runtime loads or swaps `llama-server` per request. **Why:** `ollama run <pulled-model>` on `:8080` without a global `LLAMA_MODEL` or `smoke` name.
- **Go VRAM broker (`server/vram`, Phase 8):** before ggml runner load → `training-handoff` on embedded runtime; before runtime proxy → unload all runners + `inference/resume`; before training job submit → both (OOM path unchanged). **Why:** single-GPU hosts no longer need manual curl between legacy and runtime stacks.
- **Runtime inference smoke:** [`docs/testing-smoke.md`](docs/testing-smoke.md) and [`scripts/e2e/e2e_runtime_smoke.sh`](scripts/e2e/e2e_runtime_smoke.sh) (`RUN_E2E_GPU`, `RUN_E2E_PROXY`). **Why:** two local stacks (Python runtime vs ggml runner) need an explicit checklist so operators do not confuse 404/503/OOM with API bugs.
- **`POST /internal/inference/resume`** on the Python runtime (internal only; Go broker calls it before runtime proxy). **Why:** `training-handoff` leaves inference `unloaded`; without resume, `:8081` generate returns 503 until process restart.
- **`runtime/configs/single_gpu.yaml`** and default `device_count: 1`. **Why:** `dual_4090.yaml` tensor split on one GPU makes `llama-server` fail fitting (`SPLIT_MODE_TENSOR`).
- **`LLAMA_SERVER_EXTRA_ARGS`** env (appended to `llama-server` argv). **Why:** operators need `-c` / other flags without editing YAML for every host.
- **Health `server_revision`** field (`fastapi-body-v3`). **Why:** confirm embedded Python reloaded after deploy without guessing from 422 bodies.

- **GPU training integration (Go + embedded Python):** when `OLLAMA_TRAINING` is true (default), the Go daemon embeds **CPython** via CGO (`x/trainingworker/pyembed`), loads repo-root **`training.py`**, exposes **`/api/train/*`** over HTTP, and optionally **TCP `:9500`** (newline JSON, legacy-compatible). **No** separate `python3` subprocess, **no** gRPC/`grpcio`, **no** UDS control plane. **Why:** one public process (Ollama), Python owns PyTorch/CUDA while Go owns ports, scheduler integration, and VRAM policy (inference-first OOM bridge: pause loads → evict runners → ack Python).
- **Zerollama → Eliza Cloud (default remote inference):** default upstream `https://www.elizacloud.ai`, `ELIZACLOUD_API_KEY` sent as `X-API-Key` on `/api/v1/...`; **Ed25519 request signing only** when `OLLAMA_CLOUD_BASE_URL` targets `ollama.com` (legacy cloud). Client paths `/v1/*` are rewritten to Eliza `/api/v1/*`; `/api/embed` and `/api/embeddings` map to `/api/v1/embeddings`. **Why:** OpenAI/Anthropic-compatible APIs and API-key auth match how agents integrate; legacy signing stays opt-in for ollama.com users.
- **Cloud model catalog merge:** `GET /api/v1/models` merged into local tag lists when cloud is enabled, with **singleflight** on fetch, **Cache-Control**–aware TTL (clamped), and dedupe by model name. **Why:** one combined list for operators; avoids stampedes and duplicate rows.

- **Native video sampling policy:** env `OLLAMA_VIDEO_SAMPLE_MODE` / `OLLAMA_VIDEO_STRIDE`, optional manifest `video_sampling` and `tokens_per_image`, centralized ffmpeg filter builder, structured **Info** logs after sampling, **`video_spans`** on `api.Message`, context **preflight** against `num_ctx` (messages with video only), and **[video-parity.md](docs/video-parity.md)** (Option 2 matrix).
- **Video understanding (VLM)** for OpenAI-compatible chat: `content` parts with `type: "video_url"` are merged into a single user message, decoded (data URI or remote HTTPS by default), sampled to frames via **ffmpeg**, and fed through the existing vision path as additional images (`docs/video-understanding.md`).
- **`api.Message.videos`** for raw video bytes on `POST /api/chat`; expansion runs before prompt rendering.
- **Manifest / capabilities:** `modality_backends.video_understanding` values `native` (default) or `sglang`; **`video`** capability alongside vision where applicable.
- **Optional SGLang proxy:** when `video_understanding=sglang` and `OLLAMA_SGLANG_URL` is set, `POST /v1/chat/completions` bodies that include `video_url` can be forwarded in full to SGLang’s `/v1/chat/completions`.
- **Environment variables** for limits and behavior: `OLLAMA_FFMPEG`, `OLLAMA_SGLANG_URL`, `OLLAMA_VIDEO_*` (see `docs/multimodal-backends.md`).
- **`FromChatRequestWithContext`** so remote `video_url` fetches respect request cancellation; `FromChatRequest` remains for callers without a context.

### Security

- Remote `video_url` fetches use **HTTPS by default**; `http://` requires `OLLAMA_VIDEO_ALLOW_INSECURE_HTTP=1`.
- **SSRF mitigation:** DNS resolution before GET with rejection of loopback/private/link-local targets (see `docs/video-understanding.md` for limitations).

### Changed

- **Phase 11 admission env removed:** per-gate `ZEROLLAMA_RUNTIME_ADMISSION_*`, `TRAINING_VRAM_RESERVE`, `ADMISSION_VRAM_BYPASS_PRIORITY`, etc. are **no longer read**. Behavior is fixed in `runtime/runtime/gpu/admission.py` and `inference_policy.py`. **Why:** product decision — opinionated defaults, tune in code after GPU measurement.
- **Phase 11 VRAM gate coupling:** `admission_vram_gate_enabled()` follows `CHECK_GPU_VRAM` only (not `INFERENCE_POLICY`). **Why:** disabling scheduling policy must not disable VRAM safety rails.
- **Phase 11 model budget:** `check_gguf_vram_budget` applies `max(model×margin, min_free×priority)` and training reserve on all load/enqueue/dequeue paths. **Why:** one coherent budget check instead of a separate 1 GiB probe when GGUF is known.
- **GPU training control plane:** subprocess `python3 -m trainingdaemon` + gRPC/UDS replaced by **embedded CPython** (`x/trainingworker/pyembed`). **`OLLAMA_TRAINING_PYTHONPATH`** now means the repo root containing **`training.py`**. **`grpcio`** is no longer required for training IPC.
- **Eliza outbound auth:** `X-API-Key` is applied to all proxied paths toward non-`ollama.com` upstreams (not only `/api/v1/...`); missing key logs **once** per process on first such request. **Path rewrite:** only `/v1` and `/v1/...` are mapped to `/api/v1/...` (avoids mangling paths like `/v1chat`). **Signing:** Ed25519 uses `isOllamaComUpstream()` instead of a redundant `signingHost` return value from `OLLAMA_CLOUD_BASE_URL` resolution.
- OpenAI multimodal `content` arrays are converted to **one** internal `api.Message` per assistant/user turn (text + images + videos) instead of multiple messages per part, preserving array order for vision inputs.
- **Native video:** invalid manifest `video_sampling.mode` logs a warning and falls back to **fps**; **`ExternalVideoDecodeHook`** runs only after empty/size checks (same as ffmpeg path).

### Fixed

- **Python runtime FastAPI `/api/generate`:** request body binding via `Body()`; removed `from __future__ import annotations` in `runtime/server/app.py`. **Why:** postponed annotations made FastAPI treat `req` as a **query** parameter (`422` “Field required” on `query.req`).
- **Runtime proxy `num_predict`:** no longer defaults to **128** when Ollama options omit it (`NumPredict: -1`). **Why:** answers looked “cut off” even though the legacy runner would run until stop/EOS.
- **`scripts/build/build_llama_server.sh`:** validate `nvcc` exists; search `cuda-12.8`; do not trust a broken `CUDA_HOME`. **Why:** `CUDACXX=/usr/local/cuda-13/bin/nvcc` failed when only CUDA 12.8 was installed.
- **Go tests:** `server/sched_test.go` `TestMain` sets `OLLAMA_NO_CLOUD=true` when unset. **Why:** Eliza catalog merge on dev machines broke list/tags tests expecting small fixture counts.
- **Runtime `llama_server` errors:** HTTP/JSON failures surface as `502` with detail, not empty 500 bodies.
- **Shared-interpreter `/health` hang (mitigation):** when training + embedded runtime share CPython, Go sets `ZEROLLAMA_RUNTIME_SHARED_PYTHON=1`; VRAM probe defaults to `nvidia-smi` (GIL-friendly), skips NVML when smi missing; `/health` TTL cache + single-flight with invalidation on handoff/resume/training-gpu-busy; `vram_probe_effective` on `/health`. **Why:** concurrent `/health` + `pynvml` + training threads could stall uvicorn (see `docs/bugs/shared-interpreter-health-hang.md`).
- **Runtime proxy `options.gguf`:** client-supplied `gguf` wins over manifest path (smoke `RUN_E2E_GGUF` on `:8080` proxy). **Why:** explicit override for VRAM smoke and ad-hoc weights.
- **Runtime dequeue VRAM precheck:** `check_gguf_vram_budget` at admit uses `resolve_vram_num_ctx` (options / env / `-c` / GGUF), not only `req.num_ctx`. **Why:** underestimated VRAM when `num_ctx` lived in `options` only.
- **GPU training OOM wait:** Python now keeps a **single** `threading.Event` from `_prepare_vram_relief_wait` through `_wait_vram_relief_after_oom` (stored on `BridgeState`). **Why:** re-registering a new event after Go had already ack’d caused a **lost wakeup** and up to a 120s stall.
- **GPU training shutdown:** `shutdown_ollama_training` signals `_pending_oom_event` before joining the job thread. **Why:** a thread blocked in the OOM wait could otherwise keep `join(30s)` from finishing cleanly.
- **GPU training repo path:** auto-detect walks cwd and `$HOME/zerollama`; explicit `OLLAMA_TRAINING_PYTHONPATH` / `ZEROLLAMA_REPO` must contain `training.py` or Start fails (no silent fallback). **Why:** typos must not load a different checkout.
- **`list_jobs` bridge:** `_job_to_dict` accepts `Job.to_dict()` output directly so the handler does not re-lock the queue per job. **Why:** less lock churn under load.

### Documentation

- **[scheduling-vram-policy.md](docs/scheduling-vram-policy.md)** — **why** inference and training are separate queues, VRAM broker, T6 defer queue, Phase 11–13 heuristics, tight-host checklist, code map.
- **[testing-smoke.md](docs/testing-smoke.md)** — dual-stack smoke, GPU handoff, 5080 build notes, troubleshooting (WHY-oriented).
- **GPU training (WHY-oriented):** expanded [`docs/gpu-training.md`](docs/gpu-training.md) (OOM event ordering, lost-wakeup prevention, progress polling vs push, init failure / restart, shutdown); [`docs/development.md`](docs/development.md) (embedded CPython build); [`x/trainingworker/README.md`](x/trainingworker/README.md); new [`x/trainingworker/pyembed/README.md`](x/trainingworker/pyembed/README.md); [`README.md`](README.md) in-repo link text; **code comments** in `x/trainingworker/client.go`, `server/training_api.go`, `x/trainingworker/pyembed/shim*.go`, `training_shim.c`, `bootstrap.py`, `training.py`, `envconfig/config.go`.
- **[ROADMAP.md](docs/ROADMAP.md)** — **GPU training (fine-tuning)** section; **why** the roadmap file exists; **Option 2** video phases; **[Zerollama remote cloud (Eliza)](docs/ROADMAP.md#zerollama-remote-cloud-eliza)** follow-ups and non-goals.
- **[eliza-cloud.md](docs/eliza-cloud.md)** — **why** Eliza is the default upstream, **why** `X-API-Key` vs Ed25519 signing, path rewrites, catalog merge/cache, raw upstream JSON on some routes, account stubs off ollama.com.
- **[video-understanding.md](docs/video-understanding.md)** — **why** merged OpenAI messages, ffmpeg→PNG, **why** preflight scopes to messages with video, **why** `video_spans`, logging at Info.
- **[multimodal-backends.md](docs/multimodal-backends.md)** — **why** env + manifest both apply to sampling.
- **[video-parity.md](docs/video-parity.md)** — **why** a parity matrix and reference workloads.
- Code comments in **`server/cloud_proxy.go`** / **`server/eliza_catalog.go`** (remote proxy defaults, path rewrite, singleflight) and **`server/modality`** (video policy, preflight, ffmpeg, expansion) plus **`types/model` / `api`** types where relevant — **why** decisions, not only **what**.
- Code comments in **`server/routes.go`**, **`server/sched.go`** — training route wiring and scheduler hooks for VRAM eviction.
