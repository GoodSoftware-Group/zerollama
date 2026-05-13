# `trainingworker` (Go)

Spawns the Python **`trainingdaemon`**, dials **gRPC over a Unix socket**, and exposes:

- **Public TCP** — legacy newline JSON (default `:9500`, configurable via `OLLAMA_TRAINING_TCP`)
- **Indirect HTTP** — `server/training_api.go` calls `Client.GRPC()` for `/api/train/*`

**Why this package exists:** isolate subprocess + gRPC + TCP bridging from the rest of `server/`, keep the scheduler (`VRAMEvictor`) as a small interface, and document the OOM bridge in one place.

Full rationale: [`docs/gpu-training.md`](../../docs/gpu-training.md).
