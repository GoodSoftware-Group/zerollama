# Llama

## Two llama.cpp surfaces in zerollama

| Surface | Tree | Patch path | Why separate |
|---------|------|------------|--------------|
| **In-process ggml** (`llamarunner`) | `ml/backend/ggml/ggml/` + `llama/llama.cpp/` | `llama/patches/` + **`llama/compat/` (CGO)** | Mac default runner; compat translates published GGUF at load time |
| **llama-server** (Phase 17 / Python runtime) | Sibling `../llama.cpp` | `llama/compat/` at CMake fetch | Subprocess GGUF path; upstream-shaped |

Both pins must stay aligned on **`LLAMA_CPP_VERSION`** (`b9672` target; vendor tree may lag — see [runtime/LLAMA_CPP_PIN.md](../runtime/LLAMA_CPP_PIN.md)). For ggml vendor sync, patch series, and Ollama deltas see [docs/ggml-b9509-migration.md](../docs/ggml-b9509-migration.md).

**Why pin file can lead vendor tree:** bumping `LLAMA_CPP_VERSION` documents upstream intent immediately; `./scripts/sync_vendor_llama.sh` and Metal sign-off run on a separate cadence so daily Mac dev is not blocked on every upstream tag.

### CGO link (`-lc++`)

`llama.go` sets `#cgo darwin|linux LDFLAGS: -lc++` because **common/** (jinja) is C++. Plain `go test ./discover/` links that object graph without `build_production_mac.sh` `CGO_LDFLAGS`. **Why:** dev/CI unit tests must not require a full Metal release build.

Phase 17 operator doc: [docs/phase17-llama-server.md](../docs/phase17-llama-server.md).

### In-process compat (Mac llamarunner)

**Why this exists:** CMake `llama/server` applied `llama/compat/` only to **fetched** llama-server builds. The Mac **default binary** links llama.cpp via CGO (`llama/llama.go` → `runner/llamarunner`) and never ran those hooks—so published qwen35/qwen35moe blobs failed with metadata errors (e.g. `rope.dimension_sections` length 3 vs 4) even though llama-server would have worked.

**What we did:** `llama/compat/compat.go` links the compat `.cpp` files into the Go binary; hook call sites live in patch **0016** (`llama/patches/0016-ollama-compat-loader-hooks.patch`, symlinked as `llama/compat/llama-cpp-hooks.patch` for CMake fetch). Blank import: `_ "github.com/ollama/ollama/llama/compat"` from `llama/llama.go`.

**Operator doc:** [docs/qwen35-apple-silicon.md](../docs/qwen35-apple-silicon.md).

## Updating llama.cpp

`LLAMA_CPP_VERSION` pins Ollama's llama.cpp source. An update can change more
than compilation: it can affect model loading, GPU discovery, scheduler inputs,
runtime logs, streaming, and compatibility patches. Validate the upstream diff,
the patched source Ollama actually builds, and the affected local paths.

### Workflow

Record the old ref from the base branch and choose an explicit new llama.cpp
tag or commit. After updating `LLAMA_CPP_VERSION`, materialize the source
through Ollama's normal build path:

```sh
cmake -S llama/server --preset cpu
```

This configure step fetches the pinned source and applies `llama/compat/`
patches. Confirm the resulting checkout, usually
`build/llama-server-cpu/_deps/llama_cpp-src`, resolves to the intended new ref.
Do not trust an old or dirty `_deps/` checkout as validation.
This is only a source and patch-application check; it is not runtime
validation.

Review the upstream diff using Git refs from the llama.cpp checkout:

```sh
git diff <old-ref> <new-ref> -- <path>
git show <new-ref>:<path>
```

Avoid treating patched working-tree files as pristine upstream source.

For build prerequisites, platform notes, and backend selection, see the
[developer guide](../docs/development.md).

### What to review

- Build option and dependency drift: changed `GGML_*` or `LLAMA_*` options,
  new `find_package` calls, generated assets, shader tools, or backend
  dependencies. Compare against `llama/server/CMakeLists.txt`,
  `llama/server/CMakePresets.json`, `cmake/local.cmake`, Dockerfiles, CI, and
  build scripts as needed.
- Backend discovery contracts: GGML symbols used by `discover/native_probe*.go`,
  `ggml_backend_dev_props`, backend device type enums, backend registry loading,
  device ordering, visible-device filtering, and CUDA/ROCm/Vulkan/Metal runtime
  library behavior.
- llama-server contracts: launch args and defaults, status and error payloads,
  memory/offload log lines, `system_info:`, flash-attention logging,
  `--main-gpu`, split-mode behavior, and scheduler-sensitive flags consumed by
  `llm/llama_server.go` or `server/sched.go`.
- Streaming: any new SSE frame shape, heartbeat, keepalive ping, completion
  marker, or response cadence on paths Ollama parses directly.
- Model and conversion surfaces: new architectures, tensor names, GGUF
  metadata, tokenizer behavior, speculative/MTP paths, sampler defaults, and
  server capabilities that may require updates under `convert/`, `model/`,
  `x/create/`, `llm/`, or `llama/compat/`. A model load alone is not enough;
  affected paths should run a real request and assert the expected result.

### Compatibility patches

**In-tree ggml (b9672):** patches live in `llama/patches/` (**16** format-patches). Materialize with `make -f Makefile.sync apply-patches`, then `./scripts/sync_vendor_llama.sh`. Do **not** edit synced trees directly — regenerate patches from vendor.

**llama-server (compat):** patches under `llama/compat/` are applied during CMake configure. If a patch
insertion point moved, regenerate the patch against a fresh checkout of the new
ref rather than editing an already-patched `_deps/` tree.

If compatibility sources, model patches, `llama/server/CMakeLists.txt`, or
`cmake/local.cmake` changed, build the CPU target:

```sh
cmake --build build/llama-server-cpu --target llama-server --parallel 12
```

Configure-only validation can miss missing sources, template instantiation
problems, and link errors. Also check whether upstream now supports a locally
patched model natively; if it does, the local patch may need removal or rebase.

### Local checks

Run the Go tests:

```sh
go test ./...
```
Then proceed to build the full Ollama release and verify.

### End-to-end Testing

For runtime validation, build the full applicable native payload for the
platform using the [developer guide](../docs/development.md): Metal on macOS
arm64, and the available CUDA, ROCm, and Vulkan backends on Linux and Windows.

Then run the [integration tests](../integration/README.md) on the platforms
being validated. Use them to exercise real Ollama requests and inspect logs for
device discovery, offload, memory accounting, flash attention, and
request/response behavior. macOS, Windows, and Linux behavior must be validated
on those platforms.
