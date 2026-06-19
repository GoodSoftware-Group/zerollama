# Phase 17 — Go → llama-server (upstream GGUF path)

Zerollama’s default Mac path remains **in-process ggml Metal**. Phase 17 ports upstream Ollama’s **Go → llama-server** integration so plain text GGUF can run without the Python runtime hop. See [upstream-ollama-diff.md](./upstream-ollama-diff.md) and [ROADMAP.md](./ROADMAP.md#phase-17--upstream-gguf-path-alignment-directional).

## Status (scaffold)

| Item | State |
|------|--------|
| `LLAMA_CPP_VERSION` pin (`b9509`) | Done |
| `llama/compat/`, `llama/server/` CMake tree | Imported |
| `llm/llama_server.go`, `llm/llama_binary.go` | Ported; compiles |
| MTP / draft-mtp (GGUF, llama-server path) | Done — manifest `DRAFT` → `draft-mtp`; embedded MTP auto when `draft_num_predict` set |
| DFlash / draft-eagle3 (GGUF sidecar) | Done — `PARAMETER spec_type draft-eagle3` + `DRAFT` or `draft_model_path` |
| Eliza ngram-simple | Opt-in — `ZEROLLAMA_ELIZA_NGRAM=1` or `PARAMETER spec_type ngram-simple` |
| Context shift + DisableJinja (llama-server) | Done — upstream parity on Phase 17 path only |
| MLX MTP (safetensors) | Done — Gemma4 assistant drafts, `DRAFT` modelfile |
| GGUF `DRAFT` create | Done — `DraftFiles`, `--draft-quantize` |
| `ZEROLLAMA_LLAMA_SERVER=1`, `--llama-server-backend` | Wired via `llm.NewLlamaServer` |
| `scripts/build_ollama_llama_server_darwin.sh` | Done |
| Default routing policy (replace ggml on Mac) | **Not yet** — opt-in flag only |
| Linux plain-text auto-default (llama-server when binary found) | **Shipped** — `ZEROLLAMA_LLAMA_SERVER=0` to disable |
| `OLLAMA_NEW_ENGINE` | **Deprecated** — warns; use llama-server path |
| E2E smoke | `./scripts/phase17_llama_server_smoke.sh` |
| M7 Metal benchmark | Done — keep ggml default (see below) |

## Enable the path

```bash
# Build llama-server into zerollama tree (Metal)
./scripts/build_ollama_llama_server_darwin.sh

# Build zerollama
./scripts/build_zerollama_mac.sh

# Serve (sets ZEROLLAMA_LEGACY_RUNNER=1 so Python runtime does not steal text GGUF)
./scripts/serve_llama_server_backend.sh
# equivalent:
./zerollama serve --llama-server-backend
```

Binary discovery (`llm.FindLlamaServer`) checks, in order:

- `LLAMA_SERVER_BIN` if set
- `build/llama-server-darwin/bin/llama-server` (relative to executable or cwd)
- Packaged `lib/ollama/` layouts

Reuse upstream’s build if you already have it:

```bash
BUILD_UPSTREAM_GO=0 ./scripts/build_upstream_ollama_mac.sh
LLAMA_SERVER_BIN=../ollama-upstream/build/llama-server-darwin/bin/llama-server \
  ./zerollama serve --llama-server-backend
```

## Routing interaction

| Flag / env | Text GGUF path |
|------------|----------------|
| (default Mac) | ggml Metal runner; sidecar for tokenize/VRAM only (`ZEROLLAMA_RUNTIME=1` or `--llama-cpp-backend` to proxy text GGUF) |
| `--llama-cpp-backend` | Go → Python runtime → inprocess / llama-server |
| `--llama-server-backend` | Go → llama-server subprocess (upstream shape) |

`ApplyLlamaServerBackendDefaults()` sets `ZEROLLAMA_LEGACY_RUNNER=1` when unset so Darwin sidecar runtime routing does not intercept eligible text models.

Models with manifest `zerollama-runtime` backend or vision/thinking capabilities still use ggml or Python paths as today.

## M7 benchmark (Metal, Jun 2026)

Fair run: idle GPU, `llama3.2:3b`, `num_ctx=4096`, 6 epochs.

| Arm | Host | Generate tok/s |
|-----|------|----------------|
| Upstream Go → llama-server | `:11435` | ~158.3 |
| Zerollama ggml Metal | `:11436` (`ZEROLLAMA_LEGACY_RUNNER=1`) | ~164.1 |

**Decision:** keep **ggml Metal** as Mac default; Phase 17 is for **mergeability** and upstream parity, not immediate throughput wins.

Reproduce:

```bash
go run ./cmd/bench -host 127.0.0.1:11435 -model llama3.2:3b -num-ctx 4096 -epochs 6 -format csv -output upstream-4k.csv
ZEROLLAMA_LEGACY_RUNNER=1 go run ./cmd/bench -host 127.0.0.1:11436 -model llama3.2:3b -num-ctx 4096 -epochs 6 -format csv -output ggml-4k.csv
```

Phase 17 zerollama arm (once smoke-tested):

```bash
./scripts/serve_llama_server_backend.sh
go run ./cmd/bench -model llama3.2:3b -num-ctx 4096 -epochs 6 -format csv -output zerollama-llama-server-4k.csv
```

## Remaining work

1. ~~Smoke-test `--llama-server-backend` end-to-end on ship hardware~~ — `phase17_llama_server_smoke.sh`.
2. ~~Reduce duplicate `llama/patches/` vs `llama/compat/` overlay.~~ — **Done:** retired patch 0007 (BakLLaVA MLP default → compat); formalized compat hook call sites as patch **0016** (symlinked for CMake, includes shard-loop skip); patch **0017** preserves ggml scheduler/Metal customizations through vendor sync; fixed 0015 header hunk.
3. Port upstream `discover/llama_server.go` GPU probe if needed.
4. ~~Deprecate `OLLAMA_NEW_ENGINE` for plain text GGUF~~ — **partial**: deprecated + Linux auto-default; Mac still uses ggml llamarunner until policy flip.
5. Policy decision: when (if ever) to flip Mac default from ggml to Go → llama-server.

## Related docs

- [upstream-ollama-diff.md](./upstream-ollama-diff.md) — architecture comparison
- [llama-cpp-backend.md](./llama-cpp-backend.md) — Python runtime test harness (not long-term default)
- [apple-silicon-metal.md](./apple-silicon-metal.md) — Mac operator guide
- [llama/README.md](../llama/README.md) — pin bump workflow
