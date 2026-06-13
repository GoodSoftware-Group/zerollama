# macOS developer setup

Short onboarding for **any Mac** (Apple Silicon or Intel) building and running zerollama locally. No Homebrew required for a standard build.

**Full guide:** [apple-silicon-metal.md](./apple-silicon-metal.md) · **Build details:** [development.md](./development.md#macos-apple-silicon)

---

## Prerequisites (once per machine)

| Tool | Install |
|------|---------|
| **Xcode CLI tools** | `xcode-select --install` |
| **Go 1.22+** | [go.dev/dl](https://go.dev/dl/) |
| **uv** | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **llama.cpp sibling** (runtime inprocess) | clone/build at `../llama.cpp` — `mac_setup` can build it |

Optional: **Homebrew** `python@3.12 pkg-config` if you want to link against Homebrew Python instead of Xcode’s bundled 3.9.

---

## One command setup

From the repo root:

```bash
./scripts/mac_setup.sh
```

This will:

1. Configure **CGO** (Xcode `clang` + `python3-embed` from Xcode — fixes common `stddef.h` / embed errors)
2. **`go build -o zerollama .`**
3. Create **`runtime/.venv`** (uv, Python 3.11+)
4. Build **Metal libllama** (`../llama.cpp`)
5. Run **`zerollama doctor`**
6. Run **metal sign-off** (skip with `MAC_SETUP_SIGNOFF=0`)

For MPS LoRA training deps:

```bash
MAC_SETUP_TRAINING=1 ./scripts/mac_setup.sh
```

---

## Daily use

```bash
./zerollama serve          # :11434; auto sidecar :8081 on Darwin
./zerollama doctor
./zerollama run llama3.2:3b
```

No wrapper scripts or extra env vars needed for normal dev.

---

## If `go build` fails on your Mac

Many dev machines put a non-Apple **`clang`** first on `PATH` (elan/Lean, Homebrew llvm). CGO needs **Xcode’s** compiler and **`python3-embed`**.

**Preferred:**

```bash
./scripts/build_zerollama_mac.sh
```

**Or load env into your shell:**

```bash
eval "$(./scripts/mac_cgo_env.sh --export)"
go build -o zerollama .
```

**Verify:**

```bash
./scripts/mac_cgo_env.sh --check
```

| Error | Fix |
|-------|-----|
| `stddef.h file not found` | Use `build_zerollama_mac.sh` or `mac_cgo_env --export` |
| `python3-embed not found` | `xcode-select --install` (Xcode ships the `.pc` file) |
| `Library not loaded: @rpath/Python3.framework` | Rebuild/re-test after pulling latest (CGO embeds framework rpath); or `./scripts/build_zerollama_mac.sh` |
| `go test` abort trap on Mac | `eval "$(./scripts/mac_cgo_env.sh --export)"` then `go test …`, or `./scripts/phase12_golden_ci.sh go` |
| `go.mod not found` | `cd` to the zerollama repo root |
| `Undefined symbols` / `std::` linker errors building llama.cpp | Shell has `CXX=.../clang`; use `./scripts/build_llama_server.sh` (forces `clang++`) or `eval "$(./scripts/mac_cgo_env.sh --export)"` |
| Training torch missing at runtime | `MAC_SETUP_TRAINING=1 ./scripts/mac_setup.sh` or `./scripts/training_uv_venv.sh --verify` |
| `CHECK failed: mlx_distributed_group_new_` at startup | Stale flat `build/lib/ollama/libmlxc.dylib` — `rm` it, or `./scripts/build_production_mac.sh` and run from `dist/darwin-arm64/` |
| MLX / safetensors models fail | `./scripts/build_production_mac.sh` (needs local `mlx` + `mlx-c` at pins in `MLX_VERSION` / `MLX_C_VERSION`); `export GOFLAGS=-mod=mod` if cmake fails on `go generate`; Metal Toolchain: `xcodebuild -downloadComponent MetalToolchain` |
| CMake MLX configure: `inconsistent vendoring` | `export GOFLAGS=-mod=mod` before `build_production_mac.sh` — **why:** MLX configure runs `go generate ./x/...` |

---

## Dev vs production MLX layout

| Workflow | Build | Run from |
|----------|-------|----------|
| **Daily dev** (Go, sidecar, ggml Metal) | `./scripts/build_zerollama_mac.sh` | repo root: `./zerollama serve` |
| **llama.cpp backend (experimental)** | `./scripts/build_llama_server.sh` | `./scripts/serve_llama_cpp_backend.sh` or `./zerollama serve --llama-cpp-backend` — [llama-cpp-backend.md](./llama-cpp-backend.md) |
| **Upstream Ollama A/B** | `./scripts/build_upstream_ollama_mac.sh` | `OLLAMA_HOST=127.0.0.1:11435 ../ollama-upstream/ollama serve` — [upstream-ollama-diff.md](./upstream-ollama-diff.md) |
| **MLX / release smoke** | `./scripts/build_production_mac.sh` | `dist/darwin-arm64/`: `./zerollama serve` |

Dev builds do **not** rebuild MLX native libs. After bumping `MLX_VERSION` or `MLX_C_VERSION`, run `./scripts/ensure_mlx_sources.sh`, checkout pins in `../mlx` / `../mlx-c`, then `GOFLAGS=-mod=mod ./scripts/build_production_mac.sh`. A leftover flat `build/lib/ollama/libmlxc.dylib` can shadow fresher `build/metal-v*/lib/ollama/` trees and cause startup CHECK errors. `zerollama doctor` warns when this happens.

**MLX dylibs only** (skip full zerollama binary): after `ensure_mlx_sources` and one successful `build_production_mac.sh` configure, `cmake --build build/metal-v3 --target mlx mlxc && cmake --install build/metal-v3 --component MLX` (repeat for `metal-v4` on Xcode 26+ SDK).

Production output:

```text
dist/darwin-arm64/
├── zerollama
└── lib/ollama/
    ├── libggml-*.dylib
    ├── mlx_metal_v3/
    └── mlx_metal_v4/   # macOS 26+ / Xcode 26+ SDK only
```

Override MLX sources: `OLLAMA_MLX_SOURCE=../mlx OLLAMA_MLX_C_SOURCE=../mlx-c ./scripts/build_production_mac.sh`

---

## CI / regression only

These scripts use explicit `:8080` + `:8081` ports for smoke tests — not required for daily dev:

- `./scripts/serve_mac_runtime.sh`
- `./scripts/macos_metal_smoke.sh`
- `./scripts/metal_signoff.sh`

---

## Optional shell hook

Add to `~/.zshrc` if you often run plain `go build` outside the helper scripts:

```bash
# zerollama macOS CGO (only when in repo — optional)
if [[ -f "$HOME/Sites/inference/zerollama/scripts/mac_cgo_env.sh" ]]; then
  eval "$("$HOME/Sites/inference/zerollama/scripts/mac_cgo_env.sh" --export 2>/dev/null)" || true
fi
```

Adjust the path to your checkout. Project scripts do **not** require editing `~/.zshrc`.
