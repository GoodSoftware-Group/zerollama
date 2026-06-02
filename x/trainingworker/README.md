# `trainingworker` (Go)

Embeds **CPython** via CGO (`x/trainingworker/pyembed`), loads repo-root **`training.py`**, and exposes:

- **Public TCP** — legacy newline JSON (default `:9500`, configurable via `OLLAMA_TRAINING_TCP`)
- **HTTP** — `server/training_api.go` calls into the embedded interpreter for `/api/train/*`

## Why this package exists (separate from `server/`)

- **CGO boundary isolation:** Python headers, `libpython3`, and the C shim live behind a small API (`pyembed`). The rest of the daemon stays mostly plain Go; **why:** faster rebuilds, clearer ownership, and less risk of accidental CGO in unrelated packages.
- **Scheduler contract:** `VRAMEvictor` is a three-method interface (`PauseNewLoads`, `UnloadAllRunners`, `ResumeLoads`). **Why:** the OOM callback must not import `server` types (import cycles); the server implements the interface and passes itself into `trainingworker.Start`.
- **TCP + JSON compatibility:** newline-delimited JSON on `:9500` matches historical clients; HTTP is the preferred surface for new work. **Why:** one place encodes wire quirks (idle deadlines vs long `train` jobs) without spreading them across `server/`.

## Why embedded Python (not a subprocess)

- One OS process: no `python3` on PATH for a sidecar, no gRPC, no UDS between Go and Python for the control plane.
- OOM path is a **C → Go export** from the thread that hit CUDA OOM; **why:** synchronous coordination with the inference scheduler beats polling or a second socket for “please unload.”

Full rationale: [`docs/gpu-training.md`](../../docs/gpu-training.md). Shim file index: [`pyembed/README.md`](pyembed/README.md).
