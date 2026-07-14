# LM Studio cache import

Zerollama can **discover**, **list**, and **pull** models already downloaded by [LM Studio](https://lmstudio.ai/) under `~/.lmstudio/models/` (and optional extra roots). The goal is to avoid re-downloading multi‑gigabyte weights from the Ollama registry when the files already exist locally.

**Why this exists:** LM Studio and zerollama use different on-disk layouts and naming, but many operators install both on the same Mac. Re-fetching a 30–70 GB checkpoint wastes time and disk. Import-from-cache turns `zerollama pull <name>` into a local registration step when a matching directory is found.

**Why not always “run in place”:** Zerollama’s blob store expects content-addressed layers under `OLLAMA_MODELS`. GGUF files can be **symlinked** into that store with near-zero extra space. MLX safetensors trees must be **repacked** into zerollama tensor blobs for the MLX runner — that costs roughly **one full model copy** plus headroom. See [Why MLX copies](#why-mlx-copies-instead-of-symlinks) below.

---

## Quick start

```bash
# Build (embeds version 0.0.1 by default; override with VERSION=…)
./scripts/build/build_zerollama_mac.sh
./zerollama serve

# List local + discoverable LM Studio caches
./zerollama list

# Pull registers from LM Studio cache when names match (no registry download)
./zerollama pull lmstudio-community/gemma-4-31b-it:q8_0

# See every LM Studio model even when disk is too tight for MLX import
OLLAMA_LMSTUDIO_LIST_ALL=1 ./zerollama serve
```

LM Studio entries in `zerollama list` show `remote_host=lmstudio` and `family=lmstudio` in API details until you pull and register them locally.

---

## Three import paths

| Cache layout | Detection | Import behavior | Extra disk |
|--------------|-----------|-----------------|------------|
| **GGUF** (`.gguf` weights) | Weight files in model dir | Symlink into blob store → GGUF convert/create | ~0 |
| **Legacy safetensors** (`.safetensors`, no `config.json`) | Safetensors without HF layout | Symlink + safetensors→GGUF conversion | ~0 |
| **MLX / HF safetensors** (`config.json` + `.safetensors`) | `IsSafetensorsModelDir` | Native `CreateSafetensorsModel` (no GGUF conversion) | ~model size + 512 MiB |

**Why two safetensors paths:** LM Studio MLX models use Hugging Face-style trees (`config.json`, sharded `.safetensors`). Converting those tensors to GGUF fails on MLX-specific dtypes (e.g. `U32`). Native import registers MLX tensor blobs instead. Older safetensors-only trees without `config.json` still go through the legacy conversion path.

**Why GGUF dirs with `config.json` still use the GGUF path:** Some LM Studio folders ship both metadata and a `.gguf` quant (e.g. `dria-agent-a-0.5b.Q4_K_M.gguf`). Native import requires **safetensors weights**, not GGUF. Only dirs with both `config.json` and `.safetensors` files select the MLX path.

---

## Why MLX copies instead of symlinks

1. **Blob layout:** MLX inference reads packed tensor layers in zerollama’s manifest format, not arbitrary LM Studio shard names on disk.
2. **Integrity:** Content-addressed blobs dedupe and match what `mlxrunner` expects after create.
3. **Tradeoff:** Import temporarily needs **~size(model) + 512 MiB** free on the filesystem that holds `OLLAMA_MODELS`. GGUF imports do not.

**Future direction:** In-place or read-through loading from LM Studio paths would remove the copy requirement but needs manifest/runner changes. Tracked on [ROADMAP.md](./ROADMAP.md#lm-studio-integration).

---

## Environment variables

| Variable | Default | Why |
|----------|---------|-----|
| `OLLAMA_LMSTUDIO_IMPORT` | `true` | Master switch for cache discovery, list merge, and pull shortcut. |
| `OLLAMA_LMSTUDIO_MODELS` | (unset) | Extra scan roots (comma- or OS-separated). **Why:** custom cache locations or test fixtures. |
| `OLLAMA_LMSTUDIO_LIST_ALL` | (unset) | When `1`, list **all** discoverable LM Studio models even if MLX import would exceed free disk. **Why:** operators were confused when large MLX models disappeared from `list`; pull still enforces disk checks. |
| `OLLAMA_MODELS` | `~/.ollama/models` | Volume used for **free-space** checks (`statfs` / Windows equivalent). |

On macOS, default roots include `~/.lmstudio/models` and `~/Library/Application Support/LM Studio/models` when present.

---

## Disk space policy

**Catalog (`zerollama list`, `/api/tags`):**

- **Default:** Hide MLX safetensors entries when `free < model_size + 512 MiB` on the `OLLAMA_MODELS` volume.
- **GGUF and legacy safetensors:** Always listed (symlink path, zero copy bytes).
- **`OLLAMA_LMSTUDIO_LIST_ALL=1`:** List everything; use when you want visibility while planning disk cleanup.

**Pull:**

- Always checks disk **before** starting MLX repack.
- Fails with a human-readable error, e.g. `insufficient disk space for LM Studio MLX import (~32 GB needed, 10 GB free on OLLAMA_MODELS volume)`.
- **Why fail early:** Mid-import “no space left on device” leaves partial blobs and wastes time.

**Headroom constant:** `ImportHeadroomBytes = 512 MiB` — manifest layers, metadata, and temporary packing during import.

---

## Naming and matching

LM Studio directory names are mapped to Ollama-style names (`publisher/model:tag`), e.g.:

```text
~/.lmstudio/models/lmstudio-community/gemma-4-31B-it-GGUF/
  → lmstudio-community/gemma-4-31b-it:q8_0
```

Multi-quant GGUF folders produce **one list entry per quant**; `MatchSelection` picks the weight file whose basename matches the requested tag (`q4_k`, `q8_0`, etc.).

Fuzzy matching applies when the exact name is missing but a single unambiguous directory matches the requested model family and tag.

---

## Operator troubleshooting

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| Model in LM Studio but not in `zerollama list` | MLX + low disk (default filter) | Free space on `OLLAMA_MODELS` volume, or `OLLAMA_LMSTUDIO_LIST_ALL=1` |
| Pull says insufficient disk | MLX import needs ~full model size free | Delete unused models or move `OLLAMA_MODELS` to a larger volume |
| `supplied file was not in GGUF format` on GGUF pull | Fixed Jun 2026: `config.json` was passed to GGUF parser | Rebuild; `weightFilesOnly` strips non-weight files before convert |
| MLX pull errors like `unknown data type: U32` | Safetensors routed through GGUF conversion | Rebuild; ensure `config.json` + `.safetensors` present (native path) |
| `zerollama list` shows model, pull downloads from registry | Name mismatch or `OLLAMA_LMSTUDIO_IMPORT=false` | Check exact name from list; enable import |

---

## Version and build

```bash
./scripts/build/build_zerollama_mac.sh          # VERSION defaults to 0.0.1
VERSION=0.0.2 ./scripts/build/build_zerollama_mac.sh
./zerollama --version
curl -s http://127.0.0.1:11434/api/version
```

**Why embed version via `-ldflags`:** Operators and `/api/version` must reflect the built binary, not a stale default in `version/version.go`.

---

## Code map

| Area | File | Role |
|------|------|------|
| Discovery | `internal/lmstudio/lmstudio.go` | Scan roots, `Entry`, `MatchSelection`, `ImportCopyBytes`, `HasDiskForImport` |
| Disk (Unix) | `internal/lmstudio/diskspace_unix.go` | `syscall.Statfs` — **why separate file:** Windows has no `Statfs` |
| Disk (Windows) | `internal/lmstudio/diskspace_windows.go` | `GetDiskFreeSpaceExW` |
| List merge | `server/lmstudio_catalog.go` | Append LM Studio entries to `/api/tags`; disk filter + `LIST_ALL` |
| Pull import | `server/lmstudio_import.go` | Route MLX vs GGUF; disk check; `weightFilesOnly` |
| MLX create | `x/create/client/create.go` | `ImportSafetensorsFromDirectory` |
| Config | `envconfig/config.go` | `OLLAMA_LMSTUDIO_*` variables |

Related: [MLX routing policy](./mlx-routing-policy.md) (when inference uses MLX vs ggml Metal).

---

## Tests

```bash
GOFLAGS=-mod=mod go test ./internal/lmstudio/... ./server -run 'LMStudio|ImportCopy|DirImport|HasDisk|RelativePaths|NativeSafetensors'
```

Integration smoke (manual): rebuild → `zerollama serve` → pull a small GGUF from cache → `curl /api/generate`.
