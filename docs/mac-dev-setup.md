# macOS developer setup

Short onboarding for **any Mac** (Apple Silicon or Intel), **any checkout path**.

**Full guide:** [apple-silicon-metal.md](./apple-silicon-metal.md) · **Build details:** [development.md](./development.md#macos-apple-silicon) · **Pin:** [runtime/LLAMA_CPP_PIN.md](../runtime/LLAMA_CPP_PIN.md) (`f95de977` / b10159)

---

## Prerequisites (once per machine)

| Tool | Install | Why |
|------|---------|-----|
| **Go 1.24.1+** | [go.dev/dl](https://go.dev/dl/) | Matches `go.mod` (not 1.22) |
| **Full Xcode.app** | App Store / developer.apple.com | CGO `python3-embed` pc lives under `/Applications/Xcode.app/...`; `mac_cgo_env.sh` expects it |
| **Xcode CLI tools** | `xcode-select --install` | Compilers / SDKs; not sufficient alone for embed |
| **cmake** | `brew install cmake` (or kitware) | Default bootstrap builds sibling Metal `libllama` / `llama-server` |
| **uv** | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | `runtime/.venv` (Python **3.11+** via uv) |
| **pkg-config** | Ships with Xcode / `brew install pkg-config` | Locates `python3-embed` |

**CLI-tools-only Macs:** install Homebrew **`python@3.12 pkg-config cmake`**, then `eval "$(./scripts/runtime/mac_cgo_env.sh --export)"` (or use `build_zerollama_mac.sh`). Without that, `python3-embed not found` is expected.

**Not required up front:** pulled models (sign-off is opt-in), `../mlx` (safetensors only), `../bmtl` (UMA only). Sibling `../llama.cpp` is **cloned by bootstrap** when missing.

---

## Onboarding tiers

| Tier | Goal | Command | Needs |
|------|------|---------|-------|
| **0** | Build + daily serve | `./scripts/runtime/dev_bootstrap.sh` | Go 1.24.1+, Xcode.app (or Homebrew Python), cmake, uv |
| **1** | Pull a model + chat | `./zerollama serve` then `./zerollama pull llama3.2:3b` | Tier 0 |
| **2** | Metal sign-off (CI regression) | `MAC_SETUP_SIGNOFF=1 MAC_SETUP_GO=0 MAC_SETUP_BUILD=0 ./scripts/runtime/mac_setup.sh` | Tier 1 (local text GGUF) |
| **3** | qwen35 ggml smoke | `RUN_E2E_QWEN35_MODEL=eliza-1-2b:latest ./scripts/runtime/qwen35_mac_smoke.sh` | Pulled eliza-1 / qwen tag + serve — **why eliza-1-2b:** ship qwen35 family, fast sign-off handoff |

**Why tiers:** sign-off and qwen smokes used to run inside default `mac_setup` and failed on fresh clones with no models. Tier 0 is the only required path for another developer.

---

## Fresh clone (recommended)

From any directory (repo can live anywhere):

```bash
git clone <zerollama-repo-url> zerollama
cd zerollama
./scripts/runtime/dev_bootstrap.sh
./zerollama serve
# other terminal:
./zerollama pull llama3.2:3b
./zerollama run llama3.2:3b
```

`dev_bootstrap.sh` is `mac_setup.sh` with safe defaults:

- clones **`../llama.cpp`** at pin when missing (`MAC_SETUP_LLAMA_CLONE=1`)
- **skips metal sign-off** by default (`MAC_SETUP_SIGNOFF=0`)
- builds **`./zerollama`**, **`runtime/.venv`**, Metal **libllama** when clone/build succeed

**Sibling clone repo:** `ensure_llama_cpp_sibling.sh` defaults to **elizaOS/llama.cpp** (fork kernels). For a public pin matching [LLAMA_CPP_PIN.md](../runtime/LLAMA_CPP_PIN.md):

```bash
LLAMA_CPP_REPO=https://github.com/ggml-org/llama.cpp.git ./scripts/runtime/dev_bootstrap.sh
```

Equivalent:

```bash
./scripts/runtime/mac_setup.sh   # same defaults (sign-off off, auto-clone llama.cpp)
```

For MPS LoRA training deps:

```bash
MAC_SETUP_TRAINING=1 ./scripts/runtime/dev_bootstrap.sh
```

---

## Script map (post-reorg)

| Need | Path |
|------|------|
| Fresh-clone bootstrap | `./scripts/runtime/dev_bootstrap.sh` |
| Mac setup knobs | `./scripts/runtime/mac_setup.sh` |
| CGO env / python-embed | `./scripts/runtime/mac_cgo_env.sh` |
| Runtime uv venv | `./scripts/runtime/runtime_uv_venv.sh` |
| Build `./zerollama` | `./scripts/build/build_zerollama_mac.sh` |
| Sibling llama.cpp | `./scripts/vendor/ensure_llama_cpp_sibling.sh` |
| Metal libllama / llama-server | `./scripts/build/build_llama_server.sh` |
| Metal sign-off | `./scripts/gpu/metal_signoff.sh` |

Flat `scripts/dev_bootstrap.sh`, `scripts/build_zerollama_mac.sh`, `scripts/runtime_uv_venv.sh`, etc. **do not exist** after the scripts reorg — use the table above (and `zerollama doctor --fix`, which follows these paths).

---

## Ports: daily dev vs CI smokes

| Layout | Go API | Runtime sidecar | When |
|--------|--------|-----------------|------|
| **`./zerollama serve`** (default) | **`:11434`** | `:8081` | Daily dev |
| **Sign-off / e2e smokes** | **`:8080`** | `:8081` | `metal_signoff.sh`, `macos_metal_smoke.sh` |

**Why two Go ports:** upstream Ollama uses `:11434`; CI scripts historically bound `:8080` to avoid clashing with a system Ollama. Smokes set `OLLAMA_HOST=http://127.0.0.1:8080` internally — do not copy smoke curl examples against a default `:11434` serve without changing the host. Agents must not kill production listeners on **11434** / **8081**.

---

## Daily use

```bash
./zerollama serve          # :11434; auto sidecar :8081 on Darwin
./zerollama doctor
./zerollama doctor --fix   # uv venv + build + clone ../llama.cpp + Metal libllama
./zerollama run llama3.2:3b
```

**Why `doctor --fix` on fresh clones:** runtime in-process and sign-off need `libllama.dylib` at `../llama.cpp/build/bin/`. Without the sibling checkout, `build_llama_server.sh` fails late. `--fix` runs `scripts/vendor/ensure_llama_cpp_sibling.sh` first (same as `mac_setup` tier 0).

No wrapper scripts or extra env vars needed for normal dev.

---

## Sign-off after you have a model

```bash
./zerollama pull llama3.2:3b
MAC_SETUP_SIGNOFF=1 MAC_SETUP_GO=0 MAC_SETUP_BUILD=0 ./scripts/runtime/mac_setup.sh
# or: ./scripts/gpu/metal_signoff.sh   # expects OLLAMA_HOST=:8080 layout; see serve_mac_runtime.sh
```

Or point at a GGUF blob:

```bash
M3_LLAMA_MODEL="$HOME/.ollama/models/blobs/sha256-...." MAC_SETUP_SIGNOFF=1 ...
```

---

## If `go build` fails on your Mac

Many dev machines put a non-Apple **`clang`** first on `PATH` (elan/Lean, Homebrew llvm). CGO needs **Xcode’s** compiler and **`python3-embed`**.

**Preferred:**

```bash
./scripts/build/build_zerollama_mac.sh
```

**Or load env into your shell:**

```bash
eval "$(./scripts/runtime/mac_cgo_env.sh --export)"
GOFLAGS=-mod=mod go build -o zerollama .
```

**Verify:**

```bash
./scripts/runtime/mac_cgo_env.sh --check
```

| Error | Fix |
|-------|-----|
| `stddef.h file not found` | Use `build_zerollama_mac.sh` or `mac_cgo_env --export` |
| `python3-embed not found` | Install **full Xcode.app**, or Homebrew `python@3.12 pkg-config`, then `mac_cgo_env --export` |
| `cmake: command not found` | `brew install cmake` (needed for sibling Metal libllama) |
| `llama.cpp not found` | `./scripts/vendor/ensure_llama_cpp_sibling.sh`, `zerollama doctor --fix`, or `MAC_SETUP_LLAMA_CLONE=1 ./scripts/runtime/mac_setup.sh` |
| `inconsistent vendoring` on `go build` | `GOFLAGS=-mod=mod` (set by `build_zerollama_mac.sh`) |
| `Library not loaded: @rpath/Python3.framework` | Rebuild via `./scripts/build/build_zerollama_mac.sh` |
| `go test` abort trap on Mac | `eval "$(./scripts/runtime/mac_cgo_env.sh --export)"` then `go test …` |
| Training torch missing at runtime | `MAC_SETUP_TRAINING=1 ./scripts/runtime/dev_bootstrap.sh` |
| MLX / safetensors models fail | `BUILD_MLX=1 ./scripts/build/build_zerollama_mac.sh` when `../mlx` exists |
| Sign-off: `Set M3_LLAMA_MODEL` | `zerollama pull …` first, or `MAC_SETUP_SIGNOFF=0` for tier 0 only |

**Ggml-only (skip llama.cpp build):**

```bash
MAC_SETUP_BUILD=0 MAC_SETUP_LLAMA_OPTIONAL=1 ./scripts/runtime/dev_bootstrap.sh
```

Runtime inprocess on `:8081` needs `libllama.dylib`; ggml chat via `:11434` still works.

---

## Dev vs production MLX layout

| Workflow | Build | Run from |
|----------|-------|----------|
| **Daily dev** (ggml + optional MLX) | `./scripts/build/build_zerollama_mac.sh` | repo root: `./zerollama serve` |
| **Fresh clone bootstrap** | `./scripts/runtime/dev_bootstrap.sh` | same |
| **ggml only (fast)** | `BUILD_MLX=0 ./scripts/build/build_zerollama_mac.sh` | repo root |
| **Release / dist tarball** | `./scripts/build/build_production_mac.sh` | `dist/darwin-arm64/` |

See [apple-silicon-metal.md](./apple-silicon-metal.md) for MLX pin bumps and `zerollama doctor` hints.

---

## CI / regression only

These use **`OLLAMA_HOST=:8080`** + **`ZEROLLAMA_RUNTIME_URL=:8081`** — not the default `:11434` serve:

- `./scripts/serve/serve_mac_runtime.sh`
- `./scripts/gpu/macos_metal_smoke.sh`
- `./scripts/gpu/metal_signoff.sh`

Full gate with qwen35 (M4 Max PASS Jun 2026): `RUN_E2E_QWEN35=1 RUN_E2E_QWEN35_MODEL=eliza-1-2b:latest ./scripts/gpu/metal_signoff.sh` — **why eliza-1-2b:** ship qwen35 family; qwen35 runs before Phase 15 inside the script (Phase 15 stops `:8081`).

---

## Optional shell hook

Add to `~/.zshrc` if you often run plain `go build` outside the helper scripts:

```bash
# zerollama macOS CGO (optional; adjust path to your checkout)
ZEROLLAMA_DIR="$HOME/code/zerollama"
if [[ -f "${ZEROLLAMA_DIR}/scripts/runtime/mac_cgo_env.sh" ]]; then
  eval "$("${ZEROLLAMA_DIR}/scripts/runtime/mac_cgo_env.sh" --export 2>/dev/null)" || true
fi
```

Project scripts do **not** require editing `~/.zshrc`.
