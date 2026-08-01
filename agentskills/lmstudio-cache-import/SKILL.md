---
name: lmstudio-cache-import
description: "Import models already downloaded by LM Studio into zerollama without re-downloading, via zerollama list/pull cache discovery."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, lmstudio, import, cache, disk-space, mlx, gguf]
    category: mlops
    related_skills: [zerollama-integration, download-model, model-suggester]
---

# LM Studio Cache Import Skill

Register models already downloaded by [LM Studio](https://lmstudio.ai/)
(under `~/.lmstudio/models/`) into [zerollama](https://github.com/GoodSoftware-Group/zerollama)
without re-downloading multi-gigabyte weights from the registry. Useful
when both tools are installed on the same machine.

## When to Use

- The user has models in LM Studio and wants to use them with zerollama
  instead of re-pulling the same weights
- `zerollama pull <name>` unexpectedly hits the network instead of using a
  local LM Studio cache
- Debugging disk-space errors during an MLX (safetensors) import

## How to Run

```bash
# See local + discoverable LM Studio models (family=lmstudio until pulled)
zerollama list

# Register from LM Studio cache — no registry download if the name matches
zerollama pull lmstudio-community/gemma-4-31b-it:q8_0

# Show every discoverable LM Studio model even if MLX import would exceed free disk
OLLAMA_LMSTUDIO_LIST_ALL=1 zerollama serve
```

## Three import paths (by cache layout)

| Cache layout | Detection | Import behavior | Extra disk |
|---|---|---|---|
| GGUF (`.gguf` weights) | Weight file in model dir | Symlink into blob store | ~0 |
| Legacy safetensors (no `config.json`) | Safetensors without HF layout | Symlink + safetensors→GGUF conversion | ~0 |
| MLX / HF safetensors (`config.json` + `.safetensors`) | HF-style tree | Native `CreateSafetensorsModel` (no GGUF conversion) | ~model size + 512 MiB |

**Why MLX needs a copy, not a symlink:** MLX inference reads packed tensor
layers in zerollama's own manifest format, not arbitrary LM Studio shard
filenames — the import repacks tensors into content-addressed blobs.

**Why a GGUF folder with `config.json` still uses the GGUF path:** native
MLX import requires safetensors weights; a dir with metadata **and** a
`.gguf` quant still goes through the GGUF path, only `config.json` +
`.safetensors` together select the MLX path.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_LMSTUDIO_IMPORT` | `true` | Master switch for discovery/list-merge/pull-shortcut |
| `OLLAMA_LMSTUDIO_MODELS` | unset | Extra scan roots (comma/OS-separated) |
| `OLLAMA_LMSTUDIO_LIST_ALL` | unset | `1` lists every discoverable model even if disk would block MLX import (pull still enforces the disk check) |
| `OLLAMA_MODELS` | `~/.ollama/models` | Volume checked for free space |

Default scan roots on macOS: `~/.lmstudio/models` and `~/Library/Application
Support/LM Studio/models`.

## Naming

LM Studio directory names map to `publisher/model:tag`, e.g.
`~/.lmstudio/models/lmstudio-community/gemma-4-31B-it-GGUF/` →
`lmstudio-community/gemma-4-31b-it:q8_0`. Multi-quant folders produce one
list entry per quant; fuzzy matching applies when the exact name is missing
but exactly one directory matches the requested family/tag.

## Pitfalls

- **MLX model missing from `zerollama list`** — default policy hides MLX
  entries when free disk on the `OLLAMA_MODELS` volume is less than
  `model_size + 512 MiB`. Free space, or set `OLLAMA_LMSTUDIO_LIST_ALL=1`
  to see it anyway (pull will still fail cleanly if disk is actually
  short).
- **Pull fails with "insufficient disk space"** — MLX import needs close
  to a full model-size copy; this is checked **before** starting the
  repack so you don't end up with a half-written import.
- **`pull` hits the registry instead of the cache** — usually a name
  mismatch (copy the exact name from `zerollama list`) or
  `OLLAMA_LMSTUDIO_IMPORT=false` set somewhere in the environment.
- **`supplied file was not in GGUF format`** — historical bug where
  `config.json` leaked into the GGUF parser; fixed by `weightFilesOnly`
  stripping non-weight files before conversion — rebuild if you hit this on
  an old binary.
- **`unknown data type: U32`** — a safetensors model got routed through
  GGUF conversion instead of the native MLX path; ensure both
  `config.json` and `.safetensors` files are present so it's detected as
  an MLX/HF tree.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `download-model` — the general pull/registry flow this import shortcuts
- `model-suggester` — checking the LM Studio cache before deciding to pull
