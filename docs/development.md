# Development

Install prerequisites:

- [Go](https://go.dev/doc/install)
- C/C++ Compiler e.g. Clang on macOS, [TDM-GCC](https://github.com/jmeubank/tdm-gcc/releases/latest) (Windows amd64) or [llvm-mingw](https://github.com/mstorsjo/llvm-mingw) (Windows arm64), GCC/Clang on Linux.
- **Linux:** `python3-dev` (or `python3-devel`) and `pkg-config` — required so CGO can link **embedded CPython** for GPU training (`pkg-config python3-embed`, used by `x/trainingworker/pyembed`). Example: `sudo apt install python3-dev pkg-config`. **Why:** the Go binary embeds the interpreter for `training.py`; without headers and `libpython3`, the link step fails early instead of shipping a binary that cannot load PyTorch at runtime.

**Linux training embed (5080 / Ubuntu 22.04):** default `python3-embed` is **3.10** even when `runtime/.venv` uses **3.11**. To align training with runtime on one Python generation:

```bash
sudo apt install python3.11-dev pkg-config
source ./scripts/training/training_embed_build_env.sh 3.11   # WHY: overlay pkg-config before go build
CGO_ENABLED=1 go build -o zerollama .
TRAINING_UV_PYTHON_VER=3.11 ./scripts/training/training_uv_venv.sh --verify
ldd ./zerollama | grep libpython   # expect 3.11
```

See [gpu-training.md](./gpu-training.md#installing-python-deps-embedded-interpreter). **`5080_build_zerollama`** in [`scripts/gpu/5080_env.sh`](../scripts/gpu/5080_env.sh) sources the embed overlay automatically when `python-3.11-embed` is installed.

**5080 production serve:** `cp scripts/serve/serve_production_wrapper.sh ~/bin/serve.sh` — **WHY not copy `serve_gpu_example.sh` to `~/bin`:** breaks repo-root detection (`_ROOT=$HOME`). See [5080-runbook.md](./5080-runbook.md#production-serve-binserve-sh).

Then build and run Ollama from the root directory of the repository:

```shell
go run . serve
```

The CLI binary is **`zerollama`**. A plain `go build` writes an executable named after this directory (`ollama`); use `go build -o zerollama .` if you want the on-disk name to match installs and integration tests.

**GPU inference smoke** (runtime + legacy runner, VRAM handoff): see [testing-smoke.md](./testing-smoke.md). **Why separate from `go test`:** unit tests do not run `llama-server` on your GPU or prove the two local inference stacks can share one card safely.

**Architecture (directional):** Zerollama aggregates **many inference callers** and **optional training jobs** into **queued work** sharing one or few GPUs per node; the roadmap spells out today’s split schedulers vs a future **unified policy** (priorities, idle training) on each host, and a **fleet management node** for multi-node warm routing and agent status. See [ROADMAP.md](./ROADMAP.md#product-model-queues-stakeholders-and-gpu-time) and [fleet-scheduling.md](./fleet-scheduling.md).

**Compare with upstream Ollama:** Zerollama adds Python runtime, training, and Eliza; upstream routes default GGUF as **Go → llama-server** (no Python sidecar). Clone vanilla Ollama beside this repo for A/B without merging: `./scripts/gpu/clone_upstream_ollama.sh` → [upstream-ollama-diff.md](./upstream-ollama-diff.md).

> [!NOTE]
> Ollama includes native code compiled with CGO.  From time to time these data structures can change and CGO can get out of sync resulting in unexpected crashes.  You can force a full build of the native code by running `go clean -cache` first. 


## macOS (Apple Silicon)

macOS Apple Silicon supports **Metal** built into the main binary for **GGUF** models — no CUDA steps required. For **runtime admission** on unified memory (Phase 11/13), autoconfig picks `apple_silicon.yaml` and probes via `metal-unified`.

**First-time setup:** see **[mac-dev-setup.md](./mac-dev-setup.md)** (tier 0 bootstrap for any Mac / any checkout path).

```bash
./scripts/runtime/dev_bootstrap.sh   # clone ../llama.cpp if needed; sign-off off by default
zerollama serve              # default :11434; auto sidecar on :8081, autoconfig, training venv
zerollama doctor             # check CGO env, uv, libllama, sidecar /health
zerollama doctor --fix       # uv venv + Mac CGO build; clones ../llama.cpp then builds libllama when missing (M14)
```

**Why `doctor --fix` clones before build:** `build_llama_server.sh` assumes a sibling checkout at `../llama.cpp`. Fresh clones used to fail with an opaque CMake error; `mac_setup` already ran `ensure_llama_cpp_sibling.sh` first — doctor now matches that order so tier-0 bootstrap is one command.

**Serve on macOS:** `zerollama serve` listens on **`OLLAMA_HOST`** (default `:11434`). On Apple Silicon it automatically ensures `runtime/.venv`, starts the Python runtime sidecar on loopback `:8081`, enables autoconfig (`ZEROLLAMA_AUTO_CONFIG=1`), and prepares the training venv when `OLLAMA_TRAINING` is on. No wrapper scripts required for daily use.

**CI / sign-off only:** `./scripts/gpu/metal_signoff.sh` and `./scripts/gpu/macos_metal_smoke.sh` use **`OLLAMA_HOST=:8080`** + sidecar `:8081` — not the default `:11434` daily layout.

**Escape hatches:** set `ZEROLLAMA_RUNTIME_URL` to an existing sidecar (skip spawn), `ZEROLLAMA_RUNTIME_DARWIN_SIDECAR=0` for ggml-only, or `OLLAMA_TRAINING=false` to skip training deps.

**Build (CGO):** use **`./scripts/build/build_zerollama_mac.sh`** (or `./scripts/runtime/mac_setup.sh`). Details: [mac-dev-setup.md](./mac-dev-setup.md).

Optional **MLX engine** for safetensors models: see [MLX Engine](#mlx-engine-optional) below.

## macOS (Intel)

Install prerequisites:

- [CMake](https://cmake.org/download/) or `brew install cmake`

Then, configure and build the project:

```shell
cmake -B build
cmake --build build
```

Lastly, run Ollama:

```shell
go run . serve
```

## Windows

Install prerequisites:

- [CMake](https://cmake.org/download/)
- [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) including the Native Desktop Workload
- (Optional) AMD GPU support
    - [ROCm](https://rocm.docs.amd.com/en/latest/)
    - [Ninja](https://github.com/ninja-build/ninja/releases)
- (Optional) NVIDIA GPU support
    - [CUDA SDK](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network)
- (Optional) VULKAN GPU support
    - [VULKAN SDK](https://vulkan.lunarg.com/sdk/home) - useful for AMD/Intel GPUs
- (Optional) MLX engine support
    - [CUDA 13+ SDK](https://developer.nvidia.com/cuda-downloads)
    - [cuDNN 9+](https://developer.nvidia.com/cudnn)

Then, configure and build the project:

```shell
cmake -B build
cmake --build build --config Release
```

> Building for Vulkan requires VULKAN_SDK environment variable:
> 
> PowerShell
> ```powershell
> $env:VULKAN_SDK="C:\VulkanSDK\<version>"
> ```
> CMD
> ```cmd
> set VULKAN_SDK=C:\VulkanSDK\<version>
> ```

> [!IMPORTANT]
> Building for ROCm requires additional flags:
> ```
> cmake -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
> cmake --build build --config Release
> ```



Lastly, run Ollama:

```shell
go run . serve
```

## Windows (ARM)

Windows ARM does not support additional acceleration libraries at this time.  Do not use cmake, simply `go run` or `go build`.

## Linux

Install prerequisites:

- [CMake](https://cmake.org/download/) or `sudo apt install cmake` or `sudo dnf install cmake`
- `python3-dev` and `pkg-config` (for embedded training; CGO + `python3-embed`)
- (Optional) AMD GPU support
    - [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html)
- (Optional) NVIDIA GPU support
    - [CUDA SDK](https://developer.nvidia.com/cuda-downloads)
- (Optional) VULKAN GPU support
    - [VULKAN SDK](https://vulkan.lunarg.com/sdk/home) - useful for AMD/Intel GPUs
    - Or install via package manager: `sudo apt install vulkan-sdk` (Ubuntu/Debian) or `sudo dnf install vulkan-sdk` (Fedora/CentOS)
- (Optional) MLX engine support
    - [CUDA 13+ SDK](https://developer.nvidia.com/cuda-downloads)
    - [cuDNN 9+](https://developer.nvidia.com/cudnn)
    - OpenBLAS/LAPACK: `sudo apt install libopenblas-dev liblapack-dev liblapacke-dev` (Ubuntu/Debian)
> [!IMPORTANT]
> Ensure prerequisites are in `PATH` before running CMake.


Then, configure and build the project:

```shell
cmake -B build
cmake --build build
```

Lastly, run Ollama:

```shell
go run . serve
```

## MLX Engine (Optional)

The MLX engine enables running safetensor based models. It requires building the [MLX](https://github.com/ml-explore/mlx) and [MLX-C](https://github.com/ml-explore/mlx-c) shared libraries separately via CMake.  On MacOS, MLX leverages the Metal library to run on the GPU, and on Windows and Linux, runs on NVIDIA GPUs via CUDA v13.

### macOS (Apple Silicon)

Requires the Metal toolchain. Install [Xcode](https://developer.apple.com/xcode/) first, then:

```shell
xcodebuild -downloadComponent MetalToolchain
```

Verify it's installed correctly (should print "no input files"):

```shell
xcrun metal
```

Then build:

```shell
cmake -B build --preset MLX
cmake --build build --preset MLX --parallel
cmake --install build --component MLX
```

> [!NOTE]
> Without the Metal toolchain, cmake will silently complete with Metal disabled. Check the cmake output for `Setting MLX_BUILD_METAL=OFF` which indicates the toolchain is missing.

### Windows / Linux (CUDA)

Requires CUDA 13+ and [cuDNN](https://developer.nvidia.com/cudnn) 9+.

```shell
cmake -B build --preset "MLX CUDA 13"
cmake --build build --target mlx --target mlxc --config Release --parallel
cmake --install build --component MLX --strip
```

### Local MLX source overrides

To build against a local checkout of MLX and/or MLX-C (useful for development), set environment variables before running CMake:

```shell
export OLLAMA_MLX_SOURCE=/path/to/mlx
export OLLAMA_MLX_C_SOURCE=/path/to/mlx-c
```

Clone sibling repos with **full history** (not `--depth 1`) so pinned commits stay available for diffs:

```shell
git clone https://github.com/ml-explore/mlx.git ../mlx
git clone https://github.com/ml-explore/mlx-c.git ../mlx-c
./scripts/mlx/ensure_mlx_sources.sh   # fetch MLX_VERSION / MLX_C_VERSION if missing
```

For example, using the helper scripts with local mlx and mlx-c repos:
```shell
OLLAMA_MLX_SOURCE=../mlx OLLAMA_MLX_C_SOURCE=../mlx-c ./scripts/build/build_linux.sh

OLLAMA_MLX_SOURCE=../mlx OLLAMA_MLX_C_SOURCE=../mlx-c ./scripts/build/build_darwin.sh

# arm64 production layout (dist/darwin-arm64/) — see mac-dev-setup.md
./scripts/build/build_production_mac.sh
```

```powershell
$env:OLLAMA_MLX_SOURCE="../mlx"
$env:OLLAMA_MLX_C_SOURCE="../mlx-c"
./scripts/build_darwin.ps1
```

## Compare with upstream Ollama

Zerollama is a fork with **Python runtime**, **training**, and **Eliza cloud**. Upstream [ollama/ollama](https://github.com/ollama/ollama) removed in-process ggml for text GGUF and uses **`llm/llama_server.go`** instead.

```bash
./scripts/gpu/clone_upstream_ollama.sh          # ../ollama-upstream
cd ../ollama-upstream && go build -o ollama .
OLLAMA_HOST=127.0.0.1:11435 ./ollama serve   # A/B vs zerollama :11434
```

Full delta table, pin gaps, and cherry-pick list: [upstream-ollama-diff.md](./upstream-ollama-diff.md). Experimental zerollama routing toward llama.cpp: [llama-cpp-backend.md](./llama-cpp-backend.md).

## Docker

```shell
docker build .
```

### ROCm

```shell
docker build --build-arg FLAVOR=rocm .
```

## Running tests

To run tests, use `go test`:

```shell
go test ./...
```

> NOTE: In rare circumstances, you may need to change a package using the new
> "synctest" package in go1.24.
>
> If you do not have the "synctest" package enabled, you will not see build or
> test failures resulting from your change(s), if any, locally, but CI will
> break.
>
> If you see failures in CI, you can either keep pushing changes to see if the
> CI build passes, or you can enable the "synctest" package locally to see the
> failures before pushing.
>
> To enable the "synctest" package for testing, run the following command:
>
> ```shell
> GOEXPERIMENT=synctest go test ./...
> ```
>
> If you wish to enable synctest for all go commands, you can set the
> `GOEXPERIMENT` environment variable in your shell profile or by using:
>
> ```shell
> go env -w GOEXPERIMENT=synctest
> ```
>
> Which will enable the "synctest" package for all go commands without needing
> to set it for all shell sessions.
>
> The synctest package is not required for production builds.

## Library detection

Ollama looks for acceleration libraries in the following paths relative to the `ollama` executable:

* `./lib/ollama` (Windows)
* `../lib/ollama` (Linux)
* `.` (macOS)
* `build/lib/ollama` (for development)

If the libraries are not found, Ollama will not run with any acceleration libraries.
