# Bug: `GET /health` hangs with shared CPython (training + embedded runtime)

**Status:** mitigated (May 2026) — see **Mitigation** below; `./scripts/repro_shared_interpreter_health_hang.sh` exits 0 (5× `/health` @ 200, verified on dev host)  
**Severity:** medium — training HTTP/TCP work; runtime HTTP stalls; Go `go-coordination` pushes time out  
**Affects:** `OLLAMA_TRAINING=true` + embedded runtime (`ZEROLLAMA_RUNTIME_EMBED` on, `ZEROLLAMA_RUNTIME_URL` unset)

## Summary

When the Go daemon embeds **both** GPU training (`x/trainingworker/pyembed`) and the inference runtime (`x/runtimeworker/pyembed`) in one process, uvicorn listens on the runtime port but **`GET /health` often stops responding** (client times out, 0 bytes). Repro is **intermittent**: the first one or two `/health` calls may return 200 (sometimes after 10–20s); later calls hang until timeout. The same binary with `OLLAMA_TRAINING=false` serves `/health` reliably in ~1–2s.

Training surfaces remain usable:

- `GET /api/train/status` on the main HTTP port
- Legacy TCP `{"cmd":"ping"}\n` on `OLLAMA_TRAINING_TCP`

## Environment (observed May 2026)

- Host: Linux, RTX 5080, `zerollama` built with `CGO_ENABLED=1`, `python3-embed`
- Python: `fastapi`, `uvicorn` installed; `torch` present for training
- **Not** required: GPU generate/load — hang reproduces on `/health` alone

## Minimal repro

From repo root (uses **non-production** ports `19180` / `19181` / `19650`):

```bash
./scripts/repro_shared_interpreter_health_hang.sh
```

Expected output (after training checks pass):

```text
health try 1: http=200 bytes=... time=...s
health try 2: http=200 bytes=... time=...s   # may be slow
health try 3: http=000 bytes=0 time=20s        # hang
FAIL: /health hung on try 3 (shared-interpreter bug)
```

Set `REPRO_HEALTH_ROUNDS=1` for a single-shot check (may not trigger the hang).

## Manual steps

```bash
cd ~/zerollama
export CGO_ENABLED=1
go build -o zerollama .

# Isolate from production (:8080) and from a stale sidecar URL
unset ZEROLLAMA_RUNTIME_URL
export OLLAMA_HOST=127.0.0.1:19180
export ZEROLLAMA_RUNTIME_EMBED_PORT=19181
export ZEROLLAMA_RUNTIME_EMBED=1
export OLLAMA_TRAINING=true
export OLLAMA_TRAINING_TCP=:19650
export OLLAMA_TRAINING_PYTHONPATH=$PWD
export OLLAMA_NO_CLOUD=true
export LLAMA_MODEL=/path/to/small.gguf   # any readable GGUF
export LLAMA_SERVER_BIN=/path/to/llama-server
export ZEROLLAMA_RUNTIME_CONFIG=$PWD/runtime/configs/single_gpu.yaml

./zerollama serve &
sleep 8

# Fast (baseline)
curl -sS -m 3 http://127.0.0.1:19181/ -o /dev/null -w 'root http=%{http_code}\n'

# Hang
curl -sS -m 15 http://127.0.0.1:19181/health -o /tmp/h.json -w 'health http=%{http_code} bytes=%{size_download}\n' || true
wc -c /tmp/h.json

# Still works
curl -sS http://127.0.0.1:19180/api/train/status | head -c 200
printf '{"cmd":"ping"}\n' | nc -w 3 127.0.0.1 19650
```

## Control (training off)

Same ports, but `OLLAMA_TRAINING=false` — `/health` should return JSON with `"status":"ok"` within a few seconds.

## Related code paths

| Step | Location |
|------|----------|
| Training init, `PyEval_SaveThread` | `x/trainingworker/pyembed/training_shim.c` (`training_init`) |
| Runtime embed, uvicorn thread | `x/runtimeworker/pyembed/runtime_shim.c`, `runtime/runtime/embed/serve_thread.py` |
| Startup order (training → runtime) | `server/routes.go` (~1754–1808) |
| `/health` handler | `runtime/runtime/server/app.py` → `InferenceEngine.health()` |
| Go mirror push (times out when hung) | `internal/runtimeclient/client.go` → `POST /internal/go-coordination` |

## Hypothesis (unconfirmed)

Single interpreter: training job thread + Go `runTrainingGPUPolicyMonitor` / `go-coordination` HTTP client + uvicorn worker threads contend on the GIL or `InferenceEngine` / coordinator locks. `/health` is synchronous and heavy (`nvidia_free_vram_by_device`, `vram_estimate_and_budget`, policy snapshot); after the first responses, a waiter may never run. Confirm with `py-spy dump --pid <zerollama>` while hung.

## Mitigation (shipped)

1. **`ZEROLLAMA_RUNTIME_SHARED_PYTHON=1`** — set automatically when Go embeds training + runtime (`server/routes.go`). Runtime defaults VRAM probe to **`nvidia-smi`** (subprocess waits release the GIL; `pynvml` does not).
2. **No `nvidia-smi` on shared embed** — one warning; VRAM free-memory probes are skipped (fail-open) instead of calling NVML.
3. **`InferenceEngine.health()`** — ~400ms TTL cache + single-flight rebuild; cache cleared on `training-handoff`, `inference/resume`, and `training-gpu-busy`. Response includes `vram_probe_mode` (configured) and `vram_probe_effective` (`nvml`, `nvidia-smi`, or `skipped`).

Override probe with `ZEROLLAMA_RUNTIME_VRAM_PROBE=nvml` only if you accept possible stalls on shared embed.

## Workarounds

- **Production inference only:** `OLLAMA_TRAINING=false` (current `serve.sh` default).
- **Training + inference:** run training in-process but runtime as **external** sidecar: set `ZEROLLAMA_RUNTIME_URL=http://127.0.0.1:8081` and `ZEROLLAMA_RUNTIME_EMBED=0` (separate `zerollama-runtime` process).

## Not the same bug

- **`ZEROLLAMA_RUNTIME_URL` set to another daemon’s `:8081`** while a second `zerollama serve` runs on a test port — embed is off, `/health` on the test embed port never listens; use `unset ZEROLLAMA_RUNTIME_URL` in repros.
