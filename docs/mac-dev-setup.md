# macOS developer setup

Short onboarding for **any Mac** (Apple Silicon or Intel), **any checkout path**. No Homebrew required for a standard build.

**Full guide:** [apple-silicon-metal.md](./apple-silicon-metal.md) · **Build details:** [development.md](./development.md#macos-apple-silicon)

---

## Prerequisites (once per machine)

| Tool | Install |
|------|---------|
| **Xcode CLI tools** | `xcode-select --install` |
| **Go 1.22+** | [go.dev/dl](https://go.dev/dl/) |
| **uv** | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

Optional: **Homebrew** `python@3.12 pkg-config` if you want to link against Homebrew Python instead of Xcode’s bundled 3.9.

**Not required up front:** `../llama.cpp` (bootstrap clones it), pulled models (sign-off is opt-in), `../mlx` (safetensors only).

---

## Onboarding tiers

| Tier | Goal | Command | Needs |
|------|------|---------|-------|
| **0** | Build + daily serve | `./scripts/dev_bootstrap.sh` | Xcode CLI, Go, uv |
| **1** | Pull a model + chat | `./zerollama serve` then `./zerollama pull llama3.2:3b` | Tier 0 |
| **2** | Metal sign-off (CI regression) | `MAC_SETUP_SIGNOFF=1 MAC_SETUP_GO=0 MAC_SETUP_BUILD=0 ./scripts/mac_setup.sh` | Tier 1 (local text GGUF) |
| **3** | qwen35 ggml smoke | `RUN_E2E_QWEN35_MODEL=eliza-1-2b:latest ./scripts/qwen35_mac_smoke.sh` | Pulled eliza-1 / qwen tag + serve — **why eliza-1-2b:** ship qwen35 family, fast sign-off handoff |

**Why tiers:** sign-off and qwen smokes used to run inside default `mac_setup` and failed on fresh clones with no models. Tier 0 is the only required path for another developer.

---

## Fresh clone (recommended)

From any directory (repo can live anywhere):

```bash
git clone <zerollama-repo-url> zerollama
cd zerollama
./scripts/dev_bootstrap.sh
./zerollama serve
# other terminal:
./zerollama pull llama3.2:3b
./zerollama run llama3.2:3b
```

`dev_bootstrap.sh` is `mac_setup.sh` with safe defaults:

- clones **`../llama.cpp`** at pin `LLAMA_CPP_VERSION` when missing (`MAC_SETUP_LLAMA_CLONE=1`)
- **skips metal sign-off** by default (`MAC_SETUP_SIGNOFF=0`)
- builds **`./zerollama`**, **`runtime/.venv`**, Metal **libllama** when clone/build succeed

Equivalent:

```bash
./scripts/mac_setup.sh   # same defaults since Jun 2026 (sign-off off, auto-clone llama.cpp)
```

For MPS LoRA training deps:

```bash
MAC_SETUP_TRAINING=1 ./scripts/dev_bootstrap.sh
```

---

## Ports: daily dev vs CI smokes

| Layout | Go API | Runtime sidecar | When |
|--------|--------|-----------------|------|
| **`./zerollama serve`** (default) | **`:11434`** | `:8081` | Daily dev |
| **Sign-off / e2e smokes** | **`:8080`** | `:8081` | `metal_signoff.sh`, `macos_metal_smoke.sh` |

**Why two Go ports:** upstream Ollama uses `:11434`; CI scripts historically bound `:8080` to avoid clashing with a system Ollama. Smokes set `OLLAMA_HOST=http://127.0.0.1:8080` internally — do not copy smoke curl examples against a default `:11434` serve without changing the host.

---

## Daily use

```bash
./zerollama serve          # :11434; auto sidecar :8081 on Darwin
./zerollama doctor
./zerollama doctor --fix   # optional: uv venv + zerollama build + clone ../llama.cpp + Metal libllama
./zerollama run llama3.2:3b
```

**Why `doctor --fix` on fresh clones:** runtime in-process and sign-off need `libllama.dylib` at `../llama.cpp/build/bin/`. Without the sibling checkout, `build_llama_server.sh` fails late. `--fix` runs `ensure_llama_cpp_sibling.sh` first (same as `mac_setup` tier 0).

No wrapper scripts or extra env vars needed for normal dev.

---

## Sign-off after you have a model

```bash
./zerollama pull llama3.2:3b
MAC_SETUP_SIGNOFF=1 MAC_SETUP_GO=0 MAC_SETUP_BUILD=0 ./scripts/mac_setup.sh
# or: ./scripts/metal_signoff.sh   # expects OLLAMA_HOST=:8080 layout; see serve_mac_runtime.sh
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
./scripts/build_zerollama_mac.sh
```

**Or load env into your shell:**

```bash
eval "$(./scripts/mac_cgo_env.sh --export)"
GOFLAGS=-mod=mod go build -o zerollama .
```

**Verify:**

```bash
./scripts/mac_cgo_env.sh --check
```

| Error | Fix |
|-------|-----|
| `stddef.h file not found` | Use `build_zerollama_mac.sh` or `mac_cgo_env --export` |
| `python3-embed not found` | `xcode-select --install` (Xcode ships the `.pc` file) |
| `llama.cpp not found` | `./scripts/ensure_llama_cpp_sibling.sh`, `zerollama doctor --fix`, or `MAC_SETUP_LLAMA_CLONE=1 ./scripts/mac_setup.sh` |
| `inconsistent vendoring` on `go build` | `GOFLAGS=-mod=mod` (set by `build_zerollama_mac.sh`) |
| `Library not loaded: @rpath/Python3.framework` | Rebuild via `./scripts/build_zerollama_mac.sh` |
| `go test` abort trap on Mac | `eval "$(./scripts/mac_cgo_env.sh --export)"` then `go test …` |
| Training torch missing at runtime | `MAC_SETUP_TRAINING=1 ./scripts/dev_bootstrap.sh` |
| MLX / safetensors models fail | `BUILD_MLX=1 ./scripts/build_zerollama_mac.sh` when `../mlx` exists |
| Sign-off: `Set M3_LLAMA_MODEL` | `zerollama pull …` first, or `MAC_SETUP_SIGNOFF=0` for tier 0 only |

**Ggml-only (skip llama.cpp build):**

```bash
MAC_SETUP_BUILD=0 MAC_SETUP_LLAMA_OPTIONAL=1 ./scripts/dev_bootstrap.sh
```

Runtime inprocess on `:8081` needs `libllama.dylib`; ggml chat via `:11434` still works.

---

## Dev vs production MLX layout

| Workflow | Build | Run from |
|----------|-------|----------|
| **Daily dev** (ggml + optional MLX) | `./scripts/build_zerollama_mac.sh` | repo root: `./zerollama serve` |
| **Fresh clone bootstrap** | `./scripts/dev_bootstrap.sh` | same |
| **ggml only (fast)** | `BUILD_MLX=0 ./scripts/build_zerollama_mac.sh` | repo root |
| **Release / dist tarball** | `./scripts/build_production_mac.sh` | `dist/darwin-arm64/` |

See [apple-silicon-metal.md](./apple-silicon-metal.md) for MLX pin bumps and `zerollama doctor` hints.

---

## CI / regression only

These use **`OLLAMA_HOST=:8080`** + **`ZEROLLAMA_RUNTIME_URL=:8081`** — not the default `:11434` serve:

- `./scripts/serve_mac_runtime.sh`
- `./scripts/macos_metal_smoke.sh`
- `./scripts/metal_signoff.sh`

Full gate with qwen35 (M4 Max PASS Jun 2026): `RUN_E2E_QWEN35=1 RUN_E2E_QWEN35_MODEL=eliza-1-2b:latest ./scripts/metal_signoff.sh` — **why eliza-1-2b:** ship qwen35 family; qwen35 runs before Phase 15 inside the script (Phase 15 stops `:8081`).

---

## Optional shell hook

Add to `~/.zshrc` if you often run plain `go build` outside the helper scripts:

```bash
# zerollama macOS CGO (optional; adjust path to your checkout)
ZEROLLAMA_DIR="$HOME/code/zerollama"
if [[ -f "${ZEROLLAMA_DIR}/scripts/mac_cgo_env.sh" ]]; then
  eval "$("${ZEROLLAMA_DIR}/scripts/mac_cgo_env.sh" --export 2>/dev/null)" || true
fi
```

Project scripts do **not** require editing `~/.zshrc`.
