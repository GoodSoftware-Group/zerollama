# GLiNER.cpp — findings (WHYs)

Lab notes for the **GL\*** track. Keep WHYs here so the next change does not bolt NER onto `/v1/decisions` or ship Torch Ray as the default serve path.

## Finding 1 — Separate modality from decisions

**What:** GLiNER returns labeled spans; Laya/CLM/OpenJev return typed `choice`/`score`/`noul`.

**Why:** Mixing NER into Decider packing would lie about calibration and break Jev clients.

**Learning:** Public routes are `/v1/extract` + `/v1/gliner` only. Non-goal: chat JSON NER prompts as the product surface.

## Finding 2 — Sibling C++ / ONNX, not prod llama-server

**What:** Inference runs Knowledgator GLiNER.cpp + ONNX Runtime in `gliner-server`.

**Why:** ORT + Rust tokenizers must not risk prod `b10615` chat/Laya/CLM emb. Same sibling WHY as DiffusionGemma.

**Learning:** Pin `GLINER_CPP_COMMIT`; never set `LLAMA_SERVER_BIN` to the GLiNER binary.

## Finding 3 — Dual wire (abstract + mechanical)

**What:** `/v1/extract` is portable (`text`, `labels`, `threshold`). `/v1/gliner` adds GLiNER.cpp knobs (`max_width`, `max_length`, `model_type`, `device_id`, `flat_ner`, `multi_label`).

**Why:** Agents want one NER contract; operators need engine knobs without a second product.

**Learning:** Go strips unknown engine keys on the abstract path; C++ rejects unknown keys on `/v1/gliner` with 400.

## Finding 4 — Config knobs vs per-request inference knobs

**What:** `Model` constructs with `Config{maxWidth,maxLength,modelType}` and optional `device_id`; `inference()` takes `flatNer`, `threshold`, `multiLabel`.

**Why:** Reloading ONNX per request is expensive; mismatched config in a `/v1/gliner` body returns 409 unless the server reloads under lock (lab).

**Learning:** Document startup flags as source of truth; request may echo config and must match or request reload.

## Finding 5 — Lab ports only

**What:** Default sibling `:18094` (CPU) / `:18095` (CUDA smoke), Go lab `:11437`.

**Why:** Production `:11434` / `:8081` / CT `:8080` are reserved.

**Learning:** Smokes refuse PVE hypervisor bind; run inside CT 1564.

## Finding 6 — Static link gliner into gliner-server

**What:** `GLINER_BUILD_SHARED=OFF` when building `gliner/server` — shared `libgliner.so` failed with `tokenizers_cpp.a` lacking `-fPIC`.

**Why:** Upstream defaults to shared; tokenizers static archive is not PIC-safe for `.so` link on this host.

**Learning:** Keep static in `gliner/server/CMakeLists.txt` until upstream tokenizers are rebuilt with PIC.

## Finding 7 — VRAM coexistence on 5080 16 GB (GL4)

**What:** CUDA `gliner_small` smoke moved used VRAM **6414 → 7499 MiB** (~1.1 GiB delta) with `device_id=0`. Host already had other residents.

**Why:** DiffusionGemma Q4 `-ngl 25` alone peaks ~15 GiB; emb/CLM can leave &lt;200 MiB free. Co-resident CUDA GLiNER + OpenJev denoise is not safe on 16 GB.

**Learning:** Free OpenJev / heavy emb before CUDA GLiNER, **or** keep GLiNER on CPU ORT while Diffusion owns the card. Lab CUDA smoke uses port `:18095` so it does not fight a CPU server on `:18094`.

## Finding 8 — GPU ORT without upstream GPU_CHECK

**What:** `GLINER_ORT=cuda` links `onnxruntime-linux-x64-gpu` and runs `--device-id 0`. Upstream `-DGPU_CHECK=ON` fails: `find_library(cudnn)` does not see pip/ollama `libcudnn.so.9`.

**Why:** GLiNER.cpp’s cmake PATHS for cudnn are wrong for this layout; runtime still needs cuDNN on `LD_LIBRARY_PATH` (pip `nvidia/cudnn/lib` or `ollama/mlx_cuda_v12`).

## Finding 9 — RelEx / bi-encoder blocked in GLiNER.cpp (GL5)

**What:** Upstream GLiNER.cpp checklist still open for bi-encoder; RelEx (joint NER+relations) ships in Python/ONNX (`gliner-relex-*`) but not in the C++ `Model::inference` API we wrap.

**Why:** GL5a can only exercise what the sibling binary supports today — span + **token-level** multitask ONNX (`--model-type token`).

**Learning:** Keep RelEx/bi-encoder **Parked** until Knowledgator lands C++ paths (or we accept a temporary Python RelEx sidecar — non-goal for this track). Use `ensure_gliner_model.sh multitask` + `gliner_token_smoke.sh` for token-level load.

## Finding 10 — Multitask token ONNX empty spans (channel-first logits)

**What:** `onnx-community/gliner-multitask-large-v0.5` with `--model-type token` loaded and echoed `engine.model_type=token`, but `/v1/gliner` returned `entities:[]` until fixed.

**Why:** Transformers.js / onnx-community export emits `logits` as **`[3, batch, num_words, num_classes]`** (channel-first). Upstream `TokenDecoder` indexes **`[batch, num_words, num_classes, 3]`**. Same element count, wrong layout → garbage scores near 0. Span `gliner_small` is unaffected (`[B, …]`).

**Learning:** Patch `gliner/patches/0001-token-logits-channel-first-transpose.patch` (applied by `ensure_gliner_cpp.sh`) rearranges to decoder layout. Smoke asserts Kyiv/Ukraine hits. Prefer span `gliner_small` for default product extract; token multitask is available when needed.
