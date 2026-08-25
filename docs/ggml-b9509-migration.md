# ggml @ llama.cpp — vendor migration guide

> **Current pin:** **`b10615`** (`LLAMA_CPP_VERSION`, `LLAMA_CPP_COMMIT`, `Makefile.sync` `FETCH_HEAD`). **ggml-org/llama.cpp** @ `f280b26983ad0fdb705a0d9ebf0503e76f2899b0` (tag **b10615**; supersedes **`b10488`**).

Zerollama’s **in-process ggml Metal runner** (`runner/ollamarunner`, `ml/backend/ggml`) is built from a **pinned llama.cpp tree** plus a **small set of Ollama-specific deltas**. The June–August 2026 migration rebased from an old fork snapshot onto **`b9509`** → … → **`b10488`** → **`b10615`** with **119** formal patch commits on this pin (Metal kernel-split absorbed many old `.metal` ports).

This document explains **what changed**, **why**, and **how to maintain** the vendored ggml/llama.cpp trees without drifting back to a stale fork snapshot.

---

## Why we migrated

| Problem (old state) | Why it hurt | What we did |
|---------------------|-------------|-------------|
| ggml pinned to an **old llama.cpp base** (~36 patches on `ec98e2002`) | 27/36 patches **failed** on b9509; merge cost grew every upstream bump | Rebase onto **clean upstream tags** + **16 small patches** |
| One “regenerate” path **overlaid** the entire old `ml/backend/ggml/ggml` tree | Produced **multi‑MB patches** that were a **fork snapshot**, not real upstream ggml | **Vendor clone** → apply patches → **rsync** into in-tree trees |
| Upstream b9509+ **removed/changed** C APIs Ollama Go still calls | Build broke (`sched_new_ext`, device props, LoRA plural API, jinja in common/) | Port **minimal Ollama deltas** (documented below) |
| Mac default is **ggml Metal**, not Python llama | We still need **correct Metal build + fit sizing** on unified memory | Keep **no-alloc scheduler** + **device discovery** extensions |
| **`make sync` ran `git checkout`** on vendor | Reset vendor to bare tag **before rsync** — shipped unpatch ggml while `build-info` reported new pin | **`make sync` → `sync_vendor_llama.sh` only**; script errors if zero patch commits |

**Non-goals of this migration:** replacing ggml with MLX; deleting the Python runtime; full upstream Ollama rebase; CUDA-only no-alloc pool overrides on Mac (Metal uses buft-level dummy buffers).

---

## Architecture (where ggml lives)

```text
zerollama serve
    └── Go scheduler (server/sched.go)
            └── runner/ollamarunner  (default text GGUF on Mac)
                    └── ml/backend/ggml  (CGO)
                            └── ml/backend/ggml/ggml/   ← vendored ggml + Ollama deltas
            └── llama/ (llamarunner path)
                    └── llama/llama.cpp/              ← vendored llama.cpp + Ollama deltas
```

**Why two trees:** Ollama historically split **ggml** (direct backend for `ollamarunner`) and **llama.cpp** (CGO for `llamarunner`). Both must stay on the **same pin** or symbol/layout drift breaks CGO links.

**Phase 17 (parallel track):** upstream routes plain text GGUF as **Go → llama-server** subprocess — see [phase17-llama-server.md](./phase17-llama-server.md). **ggml remains Mac default** after M7 benchmark (~164 vs ~158 tok/s @ 4k ctx). This migration keeps ggml **mergeable** while Phase 17 lands upstream-shaped routing.

---

## Pin and vendor layout

| File | Purpose |
|------|---------|
| `LLAMA_CPP_VERSION` | Human pin (`86d86ed4`) — scripts grep this without Make |
| `LLAMA_CPP_COMMIT` | Full git ref for vendor checkout |
| `LLAMA_CPP_VENDOR_HEAD` | Expected vendor HEAD after full patch apply (CI/doctor) |
| `Makefile.sync` | `FETCH_HEAD=b10615`, `WORKDIR=vendor/llama-cpp-b10615`, `UPSTREAM=ggml-org` |
| `vendor/llama-cpp-b10615/` | Fresh clone + Ollama patch commits (gitignored) |
| `llama/patches/` | format-patches; **79** on `86d86ed4` (merged upstream Metal ports dropped: snake #25459, Q2_0 #25419) |
| `llama/patches.pre-8f114a9b-20260717/` | Backup of pre-`86d86ed4` series |

**Patch drift:** `./scripts/vendor/llama_patch_doctor.sh` · `./scripts/runtime/runtime_env_doctor.sh` (`llama_patches`) · `/health.llama_patches`

**Why vendor is gitignored:** it is a **materialization workspace** for `git am` / `format-patch`, not a second source of truth. Truth is: **patches + synced in-tree trees**.

**Why pin bumps rebase patches, not chase llama.cpp HEAD daily:** keep a reviewed tip so Phase 17 diffs and zerollama-only routes (`POST /kv/seq-copy`) stay reviewable. This pin is intentionally **ggml-org master** at the time of the Jul 2026 bump.

---

## Patch series (current pin)

Applied on top of **`8f114a9b`** via `llama/patches/*.patch`. Clean re-apply: **`fail=0`** for all **25**.

### Applied on `8f114a9b` (25 commits)

| # | Subject |
|---|---------|
| 0001 | chore: stage compat + kv-ext |
| 0002–0013 | grammar → MoE split / ctx-checkpoints (core Ollama + zerollama) |
| 0014 | wire ANE into `common/CMakeLists.txt` |
| 0015–0018 | string-arr KV, CUDA q6_k get_rows, GPU discovery, mtmd C API |
| 0019–0022 | kv-ext Phase 15, compat hooks, cell_index/CMake, seq-copy |
| 0023–0025 | ANE dflash lab, multi-GPU SWA/mmproj, `n_ubatch` SWA cap |

### Deferred on ggml-org

| Topic | Status |
|-------|--------|
| Eliza fused QJL/Polar/TBQ CUDA | **Applied** — patches 0026-0030 extract types/ops/kernels from elizaOS into ggml-org vendor |

| # | Subject | Why Ollama/zerollama still needs it |
|---|---------|-------------------------------------|
| 0001 | Grammar rule ordering | Stable grammar sampling for constrained JSON |
| 0002 | String-arr KV loading | GGUF KV edge cases in loader |
| 0003 | Graph memory reporting on failure | Actionable OOM errors in runner |
| 0004 | mtmd-audio Windows build | Cross-platform multimodal |
| 0005 | Uncaught exception registration | Safer C++ boundary for CGO |
| 0006 | CUDA skip large batches | CUDA stability (no-op on Mac) |
| ~~0007~~ | ~~bakllava regression~~ | **Retired** — `llama/compat/` `handle_missing_llava_projector_type` |
| 0008 | CUDA get_rows q6_k | Quantized op support |
| 0009 | `ggml_backend_dev_reset` | Go calls reset on device unload — **required** |
| 0010 | GPU discovery enhancements | Extended `ggml_backend_dev_props`, NVML/HIP VRAM, CUDA/Metal library props |
| 0011 | No-alloc scheduler mode | `ggml_backend_sched_new_ext` for `LoadOperationFit` VRAM sizing without allocation |
| 0012 | mtmd C API | `mtmd_input_text_init/free` for CGO multimodal |
| 0013 | ollama_vocab grammar | Go-side token pieces into C++ grammar without full `llama_model` |
| 0014 | **llama-kv-ext Phase 15** | Staging KV cell map + tensor introspection for PA page bind |
| 0015 | **compat loader hooks** | Call sites for `llama/compat/` — canonical patch; symlinked as `llama/compat/001-llama-cpp-hooks.patch` |
| 0016 | **ggml scheduler + Metal gate** | `GGML_SCHED_MAX_SPLIT_INPUTS 128`, `alloc_buffers` guard for LoadOperationFit, `GGML_DISABLE_METAL` runtime gate |
| 0017 | **kv seq-copy endpoint** | Radix cross-slot KV seed (`POST /kv/seq-copy`) |
| 0018 | **ANE dflash hook (lab)** | In-process IOSurface draft hook — optional Mac lab build — **needs rebase** |
| 0019 | **llama-kv-ext alias validate (v47)** | External PA buffer alias feasibility probe + validate (no tensor mutation) |
| 0020–0029 | See table above | Eliza L2 + SWA follow-ups |

---

## Pin bump to b9781 (Jun 2026) — conflicts and WHY manual steps

**Why `git am` failed on b9781:** format-patches from the b9672 vendor omit blob SHAs upstream moved (`ggml-cuda.cu` struct relocated ~line 4900 vs ~670 in patch context; `mtmd.h` gained `mtmd_progress_callback`; `clip.cpp` load loop refactored). `git am --3way` cannot build a fake ancestor without index lines.

**Resolution pattern:**

1. `make -f Makefile.sync clean apply-patches` — applies 0001–0009 cleanly; stops on first failure.
2. For failed patch: `sed '/^index /d' llama/patches/NNNN-*.patch | git -C vendor/llama-cpp-b9781 apply --reject -p1`
3. Fix `.rej` hunks manually; `git commit` with the patch subject; `touch llama/patches/.NNNN-*.patched`; continue `apply-patches`.

| Patch | b9781 manual fix | Why upstream moved |
|-------|------------------|-------------------|
| **0010** | Add struct fields + NVML/HIP in `get_memory`; skip `device_mutex` | b9781 removed per-device mutex; props still need NVML |
| **0012** | Declare `mtmd_input_text_*` before `mtmd_progress_callback` typedef | b9781 added progress callback between typedefs |
| **0015** | `maybe_load_tensor` in clip weight loop (~line 2831) | Same logic, different line numbers / `no_alloc` branch |

**Why not copy whole files from b9672 vendor:** b9781 contains upstream fixes (Vulkan batching, mtmd progress, UMA memory) that would be lost.

---

## Ollama deltas retained in-tree

These exist because **Go/CGO contracts** or **build layout** differ from upstream CMake builds.

### ggml (`ml/backend/ggml/ggml/`)

| Delta | Why |
|-------|-----|
| `ggml_backend_sched_new_ext` + buft `no_alloc` | **`LoadOperationFit`** must size VRAM **without allocating** weights/graph buffers |
| Extended `ggml_backend_dev_props` (`library`, `compute_*`, `driver_*`, `integrated`) | Go **flash-attention gating**, GPU routing, and `/api/tags` device info |
| `mem_nvml.cpp`, `mem_hip.cpp` | **Accurate VRAM** on CUDA/ROCm (`cudaMemGetInfo` lies on some Windows setups) |
| `ggml_backend_dev_reset` (patch 0009) | Clear backend state between model loads |

### llama.cpp (`llama/llama.cpp/`)

| Delta | Why |
|-------|-----|
| `ollama_vocab` + `o_vocab` in grammar | Go passes **token pieces from Go-side vocab** into C++ grammar without full `llama_model` |
| `mtmd_input_text_init/free` | CGO multimodal path constructs mtmd prompts |
| `jinja_wrap.cpp` | b9509 added jinja under `common/jinja/`; **CGO does not compile subdirs** — unity include |
| `httplib_wrap.cpp` | b9509 `download.cpp` / `hf-cache.cpp` link httplib; CGO needs `.cpp` in `common/` |
| `build-info.cpp` (repo `llama/build-info.cpp`) | CMake generates this upstream; **CGO build has no configure step** |
| Exclude `mtmd-cli.cpp`, `deprecation-warning.cpp` | Would define **`main()`** and break the Go link |

### Go glue

| File | Why |
|------|-----|
| `ml/backend/ggml/ggml.go` — `applyDeviceProps`, `deviceLibraryFromProps` | b9509 **removed** `props.library` / `props.id`; reconstruct from registry + extensions |
| `llama/llama.go` — `llama_set_adapters_lora` | b9509 **plural** LoRA API |
| `llama/llama.cpp/src/models/models.go` — `-I..` | `llama-model.h` lives in `src/`, not `include/` |

---

## Workflow: sync vendor → in-tree

```bash
# 1. Ensure vendor exists (once per pin bump)
./scripts/vendor/rebase_vendor_unified.sh --apply --sync
# or manually:
# git clone https://github.com/elizaOS/llama.cpp.git vendor/llama-cpp-c84b3020
# cd vendor/llama-cpp-c84b3020 && git checkout c84b3020
# make -f Makefile.sync clean apply-patches

# 2. Rsync into in-tree vendored trees (preserves zerollama-only files)
./scripts/vendor/sync_vendor_llama.sh
# or: make -f Makefile.sync sync   # alias — does NOT git checkout vendor

# 4. Build + doctor
eval "$(./scripts/runtime/mac_cgo_env.sh --export)"
./scripts/build/build_zerollama_mac.sh
./zerollama doctor
```

**Why `sync_vendor_llama.sh` excludes certain paths:** rsync `--delete` would otherwise **wipe** Ollama-only files (`mem_nvml.cpp`, CGO wrappers, `build-info.cpp`) that **do not exist** in upstream vendor — script re-copies them from vendor patch commits into ggml, and preserves CGO wrappers via exclude list.

**Why the script checks `rev-list FETCH_HEAD..HEAD`:** syncing bare `b9781` produces a tree missing `dev_reset`, no-alloc scheduler, and kv-ext while `build-info.cpp` still prints `b9781`.

**Why `GOFLAGS=-mod=mod` in `build_zerollama_mac.sh`:** `go build` must not fail on inconsistent `vendor/` when only CGO trees are synced.

**Legacy shim:** `./scripts/vendor/sync_vendor_b9509.sh` forwards to `sync_vendor_llama.sh` — **why kept:** old docs/scripts referenced the b9509 name; pin tracks `Makefile.sync` `FETCH_HEAD`.

### cpp-httplib on CUDA / Proxmox CT (CGO build)

**Why not in git:** root `.gitignore` has `vendor/` — matches `llama/llama.cpp/vendor/cpp-httplib/` even though `miniaudio` / `nlohmann` / `stb` are tracked.

| Symptom | Fix |
|---------|-----|
| `cpp-httplib/httplib.h: No such file or directory` | `rsync -a ~/llama.cpp/vendor/cpp-httplib/ llama/llama.cpp/vendor/cpp-httplib/` then `CGO_ENABLED=1 go build -o zerollama .` |
| GPU gate fails on Go golden, not GPU | `RUN_E2E_PREFLIGHT=0 ./scripts/gpu/gpu_5080_session.sh` |

---

## Regenerating patches (after editing vendor)

```bash
# Edit commits in vendor/llama-cpp-b9781 (on top of b9781)
make -f Makefile.sync format-patches   # writes llama/patches/*.patch
./scripts/vendor/sync_vendor_llama.sh
# build + smoke
```

**Why format-patch from vendor:** produces **reviewable, ordered commits** instead of a monolithic diff of the entire ggml tree.

---

## Verification checklist

| Check | Why |
|-------|-----|
| `git -C vendor/llama-cpp-b9781 rev-list --count b9781..HEAD` = **16** | Patches materialized before sync |
| `go build` succeeds | CGO + new common/jinja/httplib compile |
| `zerollama doctor` | Metal ggml loads; sidecar optional |
| `ZEROLLAMA_LEGACY_RUNNER=1 ./zerollama serve` + small model | **ggml runner** path (not runtime) |
| `LoadOperationFit` / model show memory | **no-alloc scheduler** + fit sizing |
| Compare tok/s vs upstream @ 4k ctx | Regression guard (M7 baseline) |

Known gap: `go test ./ml/backend/ggml/...` may segfault on dummy GGUF fixture — **binary build + doctor** are the gate for this migration.

---

## Related docs

- [upstream-ollama-diff.md](./upstream-ollama-diff.md) — Go → llama-server vs ggml vs Python runtime; v0.30.11 cherry-picks
- [phase17-llama-server.md](./phase17-llama-server.md) — upstream GGUF path scaffold
- [apple-silicon-metal.md](./apple-silicon-metal.md) — why ggml Metal stays default on Mac
- [runtime/LLAMA_CPP_PIN.md](../runtime/LLAMA_CPP_PIN.md) — sibling llama.cpp for Python/llama-server
- [llama/README.md](../llama/README.md) — llama.cpp bump checklist
