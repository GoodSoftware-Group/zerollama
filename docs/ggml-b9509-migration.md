# ggml @ llama.cpp — vendor migration guide

> **Current pin:** `b9611` (`LLAMA_CPP_VERSION`, `Makefile.sync` `FETCH_HEAD`). Vanilla upstream Ollama still pins `b9509`; zerollama tracks latest llama.cpp tag for ggml/Metal.

Zerollama’s **in-process ggml Metal runner** (`runner/ollamarunner`, `ml/backend/ggml`) is built from a **pinned llama.cpp tree** plus a **small set of Ollama-specific deltas**. The June 2026 migration rebased from an old fork snapshot onto **`b9509`**, then bumped to **`b9611`** with formal patches 0011–0014.

This document explains **what changed**, **why**, and **how to maintain** the vendored ggml/llama.cpp trees without drifting back to a stale fork snapshot.

---

## Why we migrated

| Problem (old state) | Why it hurt | What we did |
|---------------------|-------------|-------------|
| ggml pinned to an **old llama.cpp base** (~36 patches on `ec98e2002`) | 27/36 patches **failed** on b9509; merge cost grew every upstream bump | Rebase onto **clean b9509** + **10 small patches** that apply cleanly |
| One “regenerate” path **overlaid** the entire old `ml/backend/ggml/ggml` tree | Produced **multi‑MB patches** that were a **fork snapshot**, not “real ggml at b9509” | **Vendor clone** → apply patches → **rsync** into `ml/backend/ggml/ggml` and `llama/llama.cpp` |
| Upstream b9509 **removed/changed** C APIs Ollama Go still calls | Build broke (`sched_new_ext`, device props, LoRA plural API, jinja in common/) | Port **minimal Ollama deltas** in-tree (documented below) |
| Mac default is **ggml Metal**, not Python llama | We still need **correct Metal build + fit sizing** on unified memory | Keep **no-alloc scheduler** + **device discovery** extensions from old Ollama patches |

**Non-goals of this migration:** replacing ggml with MLX; deleting the Python runtime; full upstream Ollama rebase; CUDA-only no-alloc pool overrides on Mac (Metal uses buft-level dummy buffers).

---

## Architecture (where ggml lives)

```text
zerollama serve
    └── Go scheduler (server/sched.go)
            └── runner/ollamarunner  (default text GGUF on Mac)
                    └── ml/backend/ggml  (CGO)
                            └── ml/backend/ggml/ggml/   ← vendored b9509 ggml + Ollama deltas
            └── llama/ (llamarunner path)
                    └── llama/llama.cpp/              ← vendored b9509 llama.cpp + Ollama deltas
```

**Why two trees:** Ollama historically split **ggml** (direct backend for `ollamarunner`) and **llama.cpp** (CGO for `llamarunner`). Both must stay on the **same pin** or symbol/layout drift breaks CGO links.

**Phase 17 (parallel track):** upstream routes plain text GGUF as **Go → llama-server** subprocess — see [phase17-llama-server.md](./phase17-llama-server.md). **ggml remains Mac default** after M7 benchmark (~164 vs ~158 tok/s @ 4k ctx). This migration keeps ggml **mergeable** while Phase 17 lands upstream-shaped routing.

---

## Pin and vendor layout

| File | Purpose |
|------|---------|
| `LLAMA_CPP_VERSION` | Human pin (`b9611`) |
| `Makefile.sync` | `FETCH_HEAD=b9611`, `WORKDIR=vendor/llama-cpp-b9611` |
| `vendor/llama-cpp-b9611/` | Fresh clone + Ollama patch commits (gitignored) |
| `llama/patches/` | **15** format-patches on b9611 |
| `llama/patches.pre-b9509-20260612/` | Backup of pre-migration patch series |

**Why vendor is gitignored:** it is a **materialization workspace** for `git am` / `format-patch`, not a second source of truth. Truth is: **patches + synced in-tree trees**.

---

## Patch series (b9611)

Applied on top of upstream `b9611` (vendor HEAD after patches: `1aefee58`):

| # | Subject | Why Ollama still needs it on b9509 |
|---|---------|-----------------------------------|
| 0001 | Grammar rule ordering | Stable grammar sampling for constrained JSON |
| 0002 | String-arr KV loading | GGUF KV edge cases in loader |
| 0003 | Graph memory reporting on failure | Actionable OOM errors in runner |
| 0004 | mtmd-audio Windows build | Cross-platform multimodal |
| 0005 | Uncaught exception registration | Safer C++ boundary for CGO |
| 0006 | CUDA skip large batches | CUDA stability (no-op on Mac) |
| 0007 | bakllava regression | Vision model fix |
| 0008 | Win exit instead of abort | Windows runner lifecycle |
| 0009 | CUDA get_rows q6_k | Quantized op support |
| 0010 | `ggml_backend_dev_reset` | Go calls reset on device unload — **required** |
| 0011 | GPU discovery enhancements | Extended `ggml_backend_dev_props`, NVML/HIP VRAM, CUDA/Metal library props |
| 0012 | No-alloc scheduler mode | `ggml_backend_sched_new_ext` for `LoadOperationFit` VRAM sizing without allocation |
| 0013 | mtmd C API | `mtmd_input_text_init/free` for CGO multimodal |
| 0014 | ollama_vocab grammar | Go-side token pieces into C++ grammar without full `llama_model` |
| 0015 | **llama-kv-ext Phase 15** | Staging KV cell map + tensor introspection for PA page bind; hybrid/iSWA resolve to attn base cache |

**CGO-only (not patches):** `jinja_wrap.cpp`, `httplib_wrap.cpp`, `llama/build-info.cpp`; exclude mtmd CLI mains after sync.

**Old patches intentionally dropped or deferred:**

| Old patch theme | Why dropped / deferred |
|-----------------|----------------------|
| `ggml_backend_sched_set_batch_size` | **Removed upstream**; Go no longer calls it |
| Full CUDA no-alloc pool (`reserving_graph` memcpy stubs) | **CUDA-only**; Mac fit works via buft `no_alloc` dummy buffers |
| Some gemma4 / metal kernel patches | Re-evaluate per model after b9509 native support |

---

## Ollama deltas retained in-tree

These exist because **Go/CGO contracts** or **build layout** differ from upstream CMake builds.

### ggml (`ml/backend/ggml/ggml/`)

| Delta | Why |
|-------|-----|
| `ggml_backend_sched_new_ext` + buft `no_alloc` | **`LoadOperationFit`** must size VRAM **without allocating** weights/graph buffers |
| Extended `ggml_backend_dev_props` (`library`, `compute_*`, `driver_*`, `integrated`) | Go **flash-attention gating**, GPU routing, and `/api/tags` device info |
| `mem_nvml.cpp`, `mem_hip.cpp` | **Accurate VRAM** on CUDA/ROCm (cudaMemGetInfo lies on some Windows setups) |
| `ggml_backend_dev_reset` (patch 0010) | Clear backend state between model loads |

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
# 1. Ensure vendor exists (once)
git clone https://github.com/ggml-org/llama.cpp.git vendor/llama-cpp-b9611
cd vendor/llama-cpp-b9611 && git checkout b9611

# 2. Apply Ollama patches into vendor
make -f Makefile.sync clean apply-patches

# 3. Rsync into in-tree vendored trees (preserves zerollama-only files)
./scripts/sync_vendor_llama.sh

# 4. Build
eval "$(./scripts/mac_cgo_env.sh --export)"
./scripts/build_zerollama_mac.sh
./zerollama doctor
```

**Why `sync_vendor_llama.sh` excludes certain paths:** rsync `--delete` would otherwise **wipe** Ollama-only files (`mem_nvml.cpp`, CGO wrappers, `build-info.cpp`) that **do not exist** in upstream vendor.

**Why `GOFLAGS=-mod=mod` in `build_zerollama_mac.sh`:** `go build` must not fail on inconsistent `vendor/` when only CGO trees are synced; module mode uses `go.mod` sum files instead.

**Legacy shim:** `./scripts/sync_vendor_b9509.sh` forwards to `sync_vendor_llama.sh` — **why kept:** old docs/scripts referenced the b9509 name; pin is now b9611 in `Makefile.sync`.

---

## Regenerating patches (after editing vendor)

```bash
# Edit commits in vendor/llama-cpp-b9509 (on top of b9509)
make -f Makefile.sync format-patches   # writes llama/patches/*.patch
./scripts/sync_vendor_llama.sh
# build + smoke
```

**Why format-patch from vendor:** produces **reviewable, ordered commits** instead of a monolithic diff of the entire ggml tree.

---

## Verification checklist

| Check | Why |
|-------|-----|
| `go build` succeeds | CGO + new common/jinja/httplib compile |
| `zerollama doctor` | Metal ggml loads; sidecar optional |
| `ZEROLLAMA_LEGACY_RUNNER=1 ./zerollama serve` + small model | **ggml runner** path (not runtime) |
| `LoadOperationFit` / model show memory | **no-alloc scheduler** + fit sizing |
| Compare tok/s vs upstream @ 4k ctx | Regression guard (M7 baseline) |

Known gap: `go test ./ml/backend/ggml/...` may segfault on dummy GGUF fixture — **binary build + doctor** are the gate for this migration.

---

## Related docs

- [upstream-ollama-diff.md](./upstream-ollama-diff.md) — Go → llama-server vs ggml vs Python runtime
- [phase17-llama-server.md](./phase17-llama-server.md) — upstream GGUF path scaffold
- [apple-silicon-metal.md](./apple-silicon-metal.md) — why ggml Metal stays default on Mac
- [runtime/LLAMA_CPP_PIN.md](../runtime/LLAMA_CPP_PIN.md) — sibling llama.cpp for Python/llama-server
- [llama/README.md](../llama/README.md) — llama.cpp bump checklist
