# `trainingdaemon` — Python GPU sidecar for Ollama training

This package is **not** the public training server. The **Go** daemon listens on HTTP and (by default) TCP `:9500`; this process listens on a **private Unix socket** and serves **gRPC** defined in `proto/training.proto`.

## Why a separate process?

- **PyTorch / CUDA** live naturally in Python.
- **Crash isolation** — a bad training job should not take down the main inference server (within reason; OOM coordination still matters).
- **One gRPC boundary** — Go stays free of Python GIL and torch ABI concerns.

## Running (normally you do not)

Ollama starts this automatically with:

```bash
python3 -m trainingdaemon --uds-path /path/to/socket.sock
```

`PYTHONPATH` must include the parent of the `trainingdaemon` package (the `x/trainingdaemon` directory). **Why:** so `import trainingdaemon` resolves; the repo root is also injected so `import training` finds repo-root `training.py`.

## Local development

From the repository root:

```bash
cd x/trainingdaemon
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
# or: pip install grpcio torch ... per pyproject.toml
```

If you run the daemon by hand for debugging, you still need a Go (or other) gRPC **client** dialing the same UDS path.

## Dependencies

See `pyproject.toml`. Heavy deps (`torch`, `transformers`, `peft`, …) are intentional—**why:** real fine-tuning, not a stub.

## Generated code

`training_pb2.py` and `training_pb2_grpc.py` are generated from `proto/training.proto`. Regenerate when the proto changes (same `protoc` flow as Go stubs in `x/trainingworker/trainingpb/`).

## Documentation

Full architecture and rationale: [`docs/gpu-training.md`](../../docs/gpu-training.md).
