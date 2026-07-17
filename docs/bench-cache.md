# Model bench cache (`zerollama bench` + `PERF` in `ls`)

**Audience:** operators choosing models on one machine; contributors touching CLI or list output.

**Related:** [phase17-llama-server.md](./phase17-llama-server.md) (engine A/B), [gpu-profiles-l1.md](./gpu-profiles-l1.md) (throughput tuning), [sd-vulkan-a380.md](./sd-vulkan-a380.md) (local image tags), [launch-model-inventory.md](./launch-model-inventory.md) (launch metadata — separate from bench cache).

---

## Why this exists

Operators routinely ask: *“Which of my local models is fastest on **this** GPU?”* Size and parameter count in `zerollama ls` do not answer that — a 3B Q4 model can beat a 7B Q8 on the same box depending on backend, VRAM pressure, and context.

For **image** and **video** models, tok/s is meaningless — you need **wall seconds per generation**. **`zerollama bench`** is the lightweight operator path for all three:

1. Warm up + timed epochs per model (chat: `/api/generate` decode; image: non-stream generate + PNG; video: async `/v1/videos` poll).
2. Average **decode tok/s** (chat) or **seconds** (image/video) from server metrics.
3. Persist to **`~/.ollama/bench.json`** keyed by **manifest digest** (not name — re-pull invalidates stale numbers automatically).
4. Show **`PERF`** in **`zerollama ls`** without re-running inference.

**Why one column (`PERF`) not separate TOK/S and SEC:** operators scan one table; the cache `kind` field picks the right unit (`.1f` tok/s vs `.0fs`).

**Why client-only:** Bench runs through the same HTTP path agents use; no server schema change, no manifest layer, no fleet coupling. Results reflect *your* serve flags, GPU, and loaded backend (ggml vs llama-server vs external-image subprocess).

**Why digest keys:** Tag names can be copied (`zerollama cp`) or reused; digest changes when weights change. Orphan cache entries are harmless — `ls` only looks up the current digest.

---

## Data flow

```text
zerollama bench [MODEL...]
        │
        ▼
  GET /api/tags          filter: local completion / image / video_gen; optional prefix
        │
        ├── completion ──► POST /api/generate (Raw, num_predict=128)
        │                  EvalCount / EvalDuration → avg tok/s
        │
        ├── image ───────► POST /api/generate (stream=false, short prompt)
        │                  Metrics.TotalDuration → avg seconds (max 2 timed epochs)
        │
        └── video_gen ───► POST /v1/videos → poll until completed
                           wall clock → gen_sec
        │
        ▼
  ~/.ollama/bench.json   save after each model (Ctrl-C keeps earlier results)
        │
        ▼
zerollama ls             PERF column (-- when no entry for digest)
```

**Why save after each model:** Multi-model bench can take tens of minutes (image tags are minutes-class on low-end GPUs); partial progress should survive interruption.

**Why unload between chat models (`KeepAlive: 0`):** Each model gets a fair VRAM slot; otherwise a warm predecessor skews load time and scheduler state for the next tag.

**Dual-GPU / large-model sweeps:** Default server `OLLAMA_LOAD_TIMEOUT` is 5m — 30B+ GGUF loads can exceed that on tensor-parallel hosts. Set `OLLAMA_LOAD_TIMEOUT=30m` on the serve process and use client flags:

```bash
source /etc/profile.d/zerollama-cli.sh
OLLAMA_HOST=http://127.0.0.1:2083 OLLAMA_MAX_LOADED_MODELS=1 \
  ./zerollama bench --load-timeout 1800 --timeout 900 --force qwen3.6:35b
```

Re-bench after a bad run with `--force` (digest-keyed cache keeps stale numbers like bogus partial-stream tok/s until forced).

**Why cap image timed epochs at 2:** SD on 6 GB Arc is ~30–60 s per frame; three+ epochs × many tags becomes hours. Chat keeps the full `--epochs` default.

**Why clamp `min-epochs` for image:** CLI `--min-epochs 3` with default `--epochs 3` would always fail when the image path caps timed runs at 2 — clamp `effectiveMin` to the cap so operator flags stay intuitive.

---

## Cache file

**Path:** `~/.ollama/bench.json` (same directory as `config.json`).

**Shape:**

```json
{
  "sha256:abc123…": {
    "model": "llama3.2:latest",
    "kind": "completion",
    "tok_per_sec": 42.3,
    "benched_at": "2026-06-20T08:00:00Z"
  },
  "sha256:def456…": {
    "model": "sd15-turbo-vulkan:latest",
    "kind": "image",
    "gen_sec": 35,
    "benched_at": "2026-07-03T12:00:00Z"
  }
}
```

| Field | Why |
|-------|-----|
| Map key = full digest | Stable across renames; auto-invalidates on re-pull |
| `model` | Human-readable label when inspecting JSON |
| `kind` | `completion`, `image`, or `video_gen` — drives PERF formatting |
| `tok_per_sec` | Chat decode rate only |
| `gen_sec` | Image/video wall seconds per generation |
| `benched_at` | Operator audit; future TTL policies could use this |

Writes use atomic temp + rename (`fileutil.WriteWithBackup`) — same durability pattern as integration config.

---

## Command reference

```bash
zerollama bench                    # completion + image + video_gen locals
zerollama bench llama3.2           # prefix filter (same idea as ls)
zerollama bench sd15 --force       # all sd* image tags on this host
zerollama bench --force            # ignore cache
zerollama bench --epochs 5 --tokens 256 --warmup 2 --timeout 180
zerollama ls                       # PERF column populated from cache
zerollama ls image                 # filter models with image capability
```

| Flag | Default | Why |
|------|---------|-----|
| `--epochs` | 3 | Average reduces noise from scheduler jitter (chat); image uses min(epochs, 2) |
| `--tokens` | 128 | Enough decode work without dominating wall time |
| `--warmup` | 1 | Pays load + first-graph cost outside timed epochs |
| `--force` | off | Skip re-bench when digest already cached |
| `--timeout` | 120s | Large chat models may need long first load |
| `--video-timeout` | 2h | Wan T2V jobs can run a long time |

**Skipped models:** remote completion-only catalog stubs (except LM Studio imports), embedding/speech-only tags, empty digest.

**Image models:** fixed product-photo prompt; uses `TotalDuration` from non-stream `/api/generate` (includes subprocess sd.cpp / OpenVINO wall time).

**Video models:** one `POST /v1/videos` job polled to completion; requires Wan stack installed.

**Warmup failures:** logged as warnings; timed epochs still run — **why:** slow first load should not abort the whole tag when decode would succeed.

---

## What the number means (and does not)

| Measures | Does not measure |
|----------|------------------|
| Sustained **generation** tok/s on **this host** (chat) | Cloud / remote models |
| **Seconds per image** or **video job** (media) | Prefill tok/s or TTFT |
| Same backend path as `zerollama run` | Agent tool-call or vision paths |
| Average of N epochs with varied prompts (chat) | Pixel-perfect SD quality scores |

**Why `Raw: true` for chat:** Template rendering and thinking tags add variable overhead; bench targets decode engine speed, comparable to `cmd/bench/bench.go`.

**Why `TotalDuration` for image:** External-image backends are subprocesses — server-reported total wall time is what operators feel; splitting load vs diffusion is backend-specific.

**Why varied word-list prompts per chat epoch:** Defeats KV prefix reuse across epochs so timed runs measure fresh decode, not cached-prefix shortcuts.

---

## Relationship to other benchmarks

| Tool | Role |
|------|------|
| **`zerollama bench`** | Operator cache → `ls` PERF column; minimal flags |
| **`cmd/bench/bench.go`** | Developer harness; benchstat/CSV; TTFT/prefill/load columns |
| **`scripts/phase/m4_upstream_vs_zerollama_bench.sh`** | Phase 17 ggml vs upstream llama-server decision |
| **`scripts/gpu/a380_vulkan_smoke.sh`** | A380 chat API with load_ms / total_duration gates |
| **L1 gates (`l1_cuda_full_gate.sh`)** | CI throughput regression with profile env |

Do not expect PERF in `ls` to match script benchstat lines byte-for-byte — prompts, epochs, and backends differ by design.

---

## Code map

| File | Responsibility |
|------|----------------|
| `cmd/bench_cmd.go` | Cobra command, chat generate loop, kind dispatch |
| `cmd/bench_media.go` | Image/video bench paths, epoch cap + min-epochs clamp |
| `cmd/benchcache/benchcache.go` | JSON load/save, `PerfString()`, `Cached()` |
| `cmd/cmd.go` `ListHandler` | PERF column; `listShowsRemoteCatalogEntry` / `listMatchesFilter` |
| `envconfig/models_dirs.go` | Multi-root manifest search for bench health skips |

**Why soft-fail in `ls`:** Listing models must never break when bench has never been run or cache is corrupt; missing entry shows `--`.

**Why `ModelsSearchDirs` prefers system store:** Service installs under `/usr/share/ollama` were invisible to bench health when CLI ran as root without `OLLAMA_MODELS` — dual roots match production layout.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| PERF always `--` | Never ran `bench`, or digest changed after re-pull |
| Bench skips model | Cached entry; use `--force` |
| `error` in summary table | Timed epoch failed (timeout, OOM, missing external-image bin) |
| Image bench `only 0/2 epochs` | `OLLAMA_EXTERNAL_IMAGE_BIN` unset, sd-cli missing, or VRAM OOM |
| Lower than `run --verbose` (chat) | Bench uses Raw + fixed prompt; thinking models may differ |
| Stale after backend change | Re-bench — cache does not track ggml vs llama-server |

Manual cache reset: delete `~/.ollama/bench.json` or bench with `--force`.

**A380 image compare:**

```bash
zerollama bench sd15 --force --epochs 1 --warmup 0
zerollama ls sd15    # PERF: sd15-vulkan vs sd15-turbo-vulkan vs sd15-openvino
```
