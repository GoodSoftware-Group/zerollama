---
name: benchmark-model-speed
description: "Benchmark local model decode speed (chat) or generation wall time (image/video) on a zerollama server and cache results for zerollama ls."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, bench, benchmark, performance, tok-s]
    category: mlops
    related_skills: [zerollama-integration, fleet-vram-admission, download-model, doctor-model]
---

# Benchmark Model Speed Skill

Measure real decode throughput (chat) or wall time per generation
(image/video) for local models on a [zerollama](https://github.com/GoodSoftware-Group/zerollama)
server via `zerollama bench`, and cache the result so `zerollama ls` shows a
`PERF` column without re-running inference every time. This answers "which
of my local models is actually fastest on this GPU" — model size/params
alone don't tell you that (a 3B Q4 can beat a 7B Q8 depending on backend and
VRAM pressure).

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/api/generate -d '{}'   # 400/422 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/v1/videos -d '{}'   # 400/422 = route exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

## When to Use

- Choosing between several locally available models for a latency-sensitive
  agent task
- After a serve-flag or GPU change, to re-measure real throughput instead
  of assuming the old numbers still hold
- Comparing backends (ggml vs llama-server) on the same hardware

## How to Run

```bash
# Benchmark everything local (completion, image, video_gen) — skips already-cached
zerollama bench

# Benchmark specific models
zerollama bench llama3.2:3b qwen3-coder-next:6bit

# Force re-bench even if cached
zerollama bench llama3.2:3b --force

# Tune epochs/tokens/context for the run
zerollama bench llama3.2:3b --epochs 5 --tokens 256 --num-ctx 8192

# See the cached results without re-running anything
zerollama ls
```

Key flags: `--epochs` (default 3, timed runs to average), `--tokens`
(default 128, max output tokens/epoch), `--warmup` (default 1),
`--num-ctx` (default 8192 — lower this to fit large models on
tensor-parallel dual-GPU setups), `--timeout`/`--load-timeout` (image
models / cold load), `--video-timeout` (async video poll), `--min-epochs`
(allow partial results if some epochs fail), `--skip-health-check`.

## What gets measured

| Model kind | Call path | Metric |
|---|---|---|
| Completion (chat/text) | `POST /api/generate` decode | avg **tok/s** |
| Image | `POST /api/generate` (non-streaming) | avg **seconds** per generation |
| `video_gen` | `POST /v1/videos` → poll until completed | wall-clock **seconds** |

Results persist to `~/.ollama/bench.json`, **keyed by manifest digest** (not
name) — re-pulling a tag invalidates stale numbers automatically, and
`zerollama cp`/renamed tags don't carry over stale results.

## Pitfalls

- **Client-only measurement** — this runs through the same HTTP path any
  agent uses; results reflect *your* serve flags and currently loaded
  backend (ggml vs llama-server vs external-image subprocess), not a
  vendor benchmark. Different serve config → different numbers.
- **Cached results are skipped by default** — always pass `--force` if you
  changed serve flags/GPU/quant and want fresh numbers instead of stale
  cached ones.
- **Chat models are unloaded between benches** (`keep_alive: 0`) — this is
  intentional so each model gets a fair cold-load-and-decode measurement;
  don't interpret a slow first epoch as a bug, it includes load time
  (separately bounded by `--load-timeout`).
- **Large/tensor-parallel models may need `OLLAMA_LOAD_TIMEOUT=30m`** on
  the **serve** process — the client's `--load-timeout` flag alone won't
  help if the server's own load timeout (default 5m) fires first.
- **Image/video bench is minutes-class** — a full multi-model sweep
  including image/video tags can take a long time; results save after each
  model so `Ctrl-C` keeps partial progress.
- **`PERF` is one column with mixed units** — tok/s (`.1f`) for chat vs.
  seconds (`.0fs`) for image/video, distinguished by the cache entry's
  `kind`; don't assume every `PERF` number is directly comparable across
  model types.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `fleet-vram-admission` — checking GPU headroom before a long bench sweep
- `doctor-model` — rule out config traps before publishing a benchmark number
- `download-model` — pulling models before benchmarking them
