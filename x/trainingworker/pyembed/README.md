# `pyembed` — embedded CPython for GPU training

This package is the **CGO boundary** between Go and CPython: it compiles `training_shim.c` against `libpython3` (`pkg-config python3-embed`), embeds `bootstrap.py`, and loads repo-root **`training.py`**.

## Why this exists (instead of subprocess + gRPC)

- **One process:** no second `python3` daemon, no Unix socket control plane, no `grpcio` dependency for training IPC.
- **VRAM policy in one place:** CUDA OOM in Python can call into Go synchronously (C → `//export` → scheduler). **Why:** inference and training share one GPU; Go already owns runner load/unload.
- **Explicit link dependency:** the binary always links embedded Python on supported builds. **Why:** fail at link time if headers/libs are missing, instead of optional tags or silent fallbacks.

## What lives here

| File | Role |
|------|------|
| `shim.go` | Go wrappers around the C API (`InitEmbeddedPython`, JSON helpers, `Shutdown`). |
| `shim_exports.go` | `//export go_training_oom_hook` — Go callback invoked from C when Python reports OOM. Mutex protects handler registration. **Why mutex:** OOM can race with `Close()` / handler replacement. |
| `bootstrap_embed.go` | `//go:embed bootstrap.py` — must be separate from `import "C"` (CGO rule). |
| `bootstrap.py` | Runs in `__main__` after compile+eval: installs `BridgeState`, wires `ollama_training_native`, starts `job_processor`. |
| `training_shim.c` | `Py_Initialize`, GIL discipline, JSON IPC, native module `fire_oom`. Resolves `$REPO/.venv-training/lib/pythonX.Y/site-packages` where **X.Y = embedded libpython** (see `embedded_training_python_ver` in [`scripts/training/training_uv_venv.sh`](../../../scripts/training/training_uv_venv.sh)). **Before init:** `embedded_prepare_pytorch_ld_path` ([`x/pyembed_common/`](../../pyembed_common/)) prepends `torch/lib` and strips ggml `hostlibs` from `LD_LIBRARY_PATH`. **Why no `Py_Finalize` after torch:** unsafe with PyTorch and the Go runtime; process exit cleans up. |
| `training_shim.h` | C API surface for the shim. |

## Reading order

1. [`docs/gpu-training.md`](../../docs/gpu-training.md) — full architecture, env vars, OOM bridge, troubleshooting.
2. [`../README.md`](../README.md) — why `trainingworker` is a separate package from `server/`.
