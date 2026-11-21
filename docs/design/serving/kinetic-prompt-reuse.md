# Prompt reuse (Kinetic ideas in llama.cpp)

**Patches:** [`0162`](../../../llama/patches/0162-server-mark-hybrid-prompts-on-a-256-token-grid.patch)–[`0186`](../../../llama/patches/0186-llama-restore-a-long-prompt-when-the-copy-is-faster-than-running-it.patch) on pin **b12557**
**Source of the ideas:** [Veloresearch/kinetic](https://github.com/Veloresearch/kinetic) — behavior only. The Rust kernels stay there.
**Findings:** [kinetic-prompt-reuse-findings.md](kinetic-prompt-reuse-findings.md)
**Pin:** [runtime/LLAMA_CPP_PIN.md](../../../runtime/LLAMA_CPP_PIN.md)

## Why this exists

A single new prompt runs the same Metal, CUDA, or Arc matmul it did before. The slow part of an agent turn is reading the same tools and rules again. llama.cpp already kept that prefix in RAM for the life of the process. These patches make the save restorable after a restart, skip a restore that would be slower than running the prompt, and leave a draft that stops before the last word so the finished chat continues from the saved state.

**Why not the Kinetic GEMM kernels:** Burst int8 is a CUDA/SYCL matmul. It was not ported. It would not change an M4 decode, and llama.cpp already has CUDA graphs. Decode tok/s on a cold unique generate is the wrong measurement for this series. `zerollama bench` times that generate (`num_predict` 128, raw). A flat bench is the expected result.

**Why no new HTTP knobs:** `LLAMA_WARM_DISK`, `LLAMA_SHORTLIST`, `LLAMA_RAD1`, `LLAMA_SETTLE`, `LLAMA_PAGE_IN`, and the MoE cache are process-wide and default on. One client turning them off would change the next client's slot. The per-request control that already exists is a cache miss: `options.cache_reset` or OpenAI `cache_prompt: false`.

## Where it runs

| Path | What this series does |
|------|------------------------|
| **llama-server** built from `vendor/llama-cpp-b12557` after **0186** | Disk and RAM prompt state, cost gate, prefill route, mark grid, session-suffix draft |
| **In-process ggml** (Mac default) | Library pieces compiled into CGO (`src/`): host output shortlist, host-expert cache. The files under `tools/server/` are not in that binary |
| **Linux serve** | `ZEROLLAMA_LLAMA_SERVER=auto` routes GGUF through llama-server, so the server pieces apply once that binary is rebuilt |
| **MLX** | Unchanged. `POST /api/prefill` returns **501** |

**Why the Mac default is different:** Darwin does not set `ZEROLLAMA_LLAMA_SERVER`. Daily `:11434` chat still uses the ggml runner. The agent-visible disk cache and `/api/prefill` show up when the model is on llama-server (`ZEROLLAMA_LLAMA_SERVER=1`, Linux `auto`, or `--edge`).

The November 2026 `llama-server` binary next to this tree (built 2026-11-19) and the `zerollama` binary from 2026-11-10 do not contain the series. Restarting those files does not change speed. Rebuild both, on a lab port, before comparing.

## What an agent calls

`POST /api/prefill` and `POST /v1/prefill` take the same `model`, `messages`, `tools`, and `think` as the finished `/api/chat`. The last message is cut before its last word. The response is `cached_tokens` and `prompt_tokens`.

**Why Go renders the draft:** `/api/chat` renders with `chatPrompt`, then calls llama-server `/completion`. Proxying the chat to llama-server `/v1/prefill` would run that server's template. The cached string would not be a prefix of the next chat, and the restore would miss. Go appends U+F8FF after the cut, renders, and truncates at the sentinel so the end-of-turn marker stays off the cached text. A template that drops the sentinel is **400**.

**Why `n_predict` is sent as `0` in a raw body:** the normal completion struct tags `n_predict` `omitempty`. Zero would be omitted and the server would sample. `prefill_only` also forces `n_predict` 0. Sampling fields are copied onto the draft so a saved top-k row still covers the follow-up: a saved k covers a later k no larger than it.

An empty cut (the last message is one unfinished word) returns zeros and does not call the server. Media is **400**. Cloud and MLX are **501**.

`/api/version` advertises `"prefill": true` on a binary that includes the route. **501** from the handler is the signal that the loaded runner is not llama-server.

## Two caches, two directories

| Cache | Where | When |
|-------|--------|------|
| L3 slot bridge | `~/.cache/zerollama/llama-cache/` | Named with `prompt_cache_key`. On a Mac this save stays off until `ZEROLLAMA_L3_PROFILE=agent`. See [gpu-profiles-l3.md](gpu-profiles-l3.md) |
| This series | `~/Library/Caches/llama.cpp/warm` on macOS, `~/.cache/llama.cpp/warm` on Linux (`LLAMA_CACHE` overrides the parent) | Inside llama-server, on unless `LLAMA_WARM_DISK=0` |

A file is restored only for the same model file (size, mtime, leading megabyte), context, batch, RoPE, and device. Metal state does not restore onto CPU. Version stays **2**. **Why:** version 2 already rejects the old one-token-at-a-time states. A bump would throw away every good file. Extra trailers are optional.

## Patch series

| Patch | What it does | Why |
|-------|----------------|-----|
| **0162** | Mark a hybrid or recurrent prompt on a grid | A later request can resume at a mark. `checkpoint_every_nt` 0 is auto, `-1` is off. Multi-GPU sliding window turns checkpoints off after the grid is chosen. Breaks at 4 and `4 + n_ubatch` tokens before the end stay, so the System One primer still lands |
| **0163** | CPU shortlist of a host Q6_K or Q8_0 output head | Score the rows a top-k sampler can keep. **Why host only:** a Metal or CUDA buffer is not that matrix, and the GPU matmul stays. `LLAMA_SHORTLIST=0` disables it. Logprobs, grammar, penalties, and logit bias keep the full head |
| **0164** | `POST /prefill` and `/v1/prefill` on llama-server | The last word is the only part left to run. `prefill_only` returns before sampling |
| **0165** | Size the MoE cache from free memory | A reserve stays free for the rest of the process. The slot map's max batch stays 32 |
| **0166** | Session-suffix draft when no drafter was chosen | Repeated text in the session can be accepted from the lookup. An explicit `--spec-type` (ngram, dflash, eagle, draft-mtp, or `none`) does not add it. An empty Go spec type omits `--spec-type`, so the lookup still turns on. `LLAMA_RAD1=0` disables it. Qwen 3.5 draft-mtp stays disabled in Go |
| **0167** | Write the prompt state to disk | The RAM cache dies with the process |
| **0168** | Restore only when the copy is faster than running the prompt | A short prompt should not pay for a state load |
| **0169** | Resume a GPU prompt from the last full batch | A partial batch is the wrong place to continue |
| **0170** | Read host expert weights into RAM when 3 GiB would remain | Mapped experts otherwise fault one page at a time on the first tokens. `LLAMA_PAGE_IN=0` skips the read |
| **0171** | Copy every expert into the cache when it fits | A miss then hits RAM instead of the file |
| **0172** | Top-k shortlist of the output rows | Same idea as **0163**, kept when the sampler's k is known |
| **0173** | Keep a frequent expert when a rare one misses | A one-off expert must not evict a hot one |
| **0174** | Sample a repeated prompt from its saved logits | The next request should not recompute the last row |
| **0175** | Space marks by the cost of one recurrent cell | A fixed 8192 is the wrong distance when one cell is expensive |
| **0176** | Mark a sliding window on the same grid | A windowed model can resume too. A mark inside the window is the one a wrapped prompt can still reach |
| **0177** | Drop a cache entry a longer prompt cannot resume | A short save must not win over a usable longer one |
| **0178** | Keep a short in-window entry a wrapped prompt cannot cut back to | The window, not the token count, decides whether the save is still a prefix |
| **0179** | Restore only on the device that wrote the state | CPU and GPU round differently. The key includes the device name. Metal and CPU do not share a file |
| **0180** | Keep the last logits with the saved state | The row has to travel with the state or the next sample misses |
| **0181** | Drop the least recently used disk file past the budget | One model must not fill the directory forever. A successful restore becomes newest |
| **0182** | Keep a resumed prompt in RAM for the next slot | The following request on another slot should not read the file again |
| **0183** | New cache when the model file changes | Same path, replaced bytes, must not restore the old state |
| **0184** | Key includes release, physical batch, and RoPE | Those change the state a later token continues from. The git commit is not in the key. **Why:** a rebuild that does not change the forward would throw away every file |
| **0185** | Rank a save by the prefill rate that wrote it | A later choice can prefer the state that was cheap to produce |
| **0186** | Restore a long prompt when the timed copy beats running it | The untimed path still keeps the 0.25 keep gate. The timed path uses the measured gain and skips that gate |

## Process env

Unset means on. Set the value to `0` to disable that one policy.

| Env | Effect |
|-----|--------|
| `LLAMA_WARM_DISK` | Disk copy of prompt state |
| `LLAMA_WARM_DIR` | Directory instead of `llama.cpp/warm` |
| `LLAMA_WARM_COST` | Skip a restore that is slower than running the prompt |
| `LLAMA_SHORTLIST` | Host output shortlist |
| `LLAMA_RAD1` | Session-suffix draft |
| `LLAMA_SETTLE` | Wait for a short prompt to settle before caching it |
| `LLAMA_PAGE_IN` | Fault host experts into RAM at load |
| `LLAMA_PREFILL_END` | Replacement for the U+F8FF cut marker |

`ZEROLLAMA_WARM_HYSTERESIS` is a different policy. It chooses which ggml runner to unload. It does not toggle this cache.

Go copies `os.Environ()` into the llama-server process, so an operator export reaches the server. Go does not set these itself.

## What stayed out

| Kinetic piece | Why it stayed out |
|---------------|-------------------|
| Burst int8 GEMM | CUDA/SYCL kernels. Decode tok/s was not the goal, and the Mac path would not run them |
| Torque | llama.cpp already has CUDA graphs |
| Convoy | A 5 ms sleep, not a faster forward |
| Prefetch | Needs a second copy stream the graph does not have |
| Flywheel EWMA | The timer wraps the wrong interval, so the estimate would move the cache the wrong way |
| Branch mark and the 32-token grid | They change the CUDA GDN kernel at 64 tokens |
| Shrinking the settle piece below one ubatch | A restore shorter than `llama_n_ubatch` (typically 512) is not a full batch the GPU resume can continue |
| Draft trees, ensemble, MTP | Lookup is the session index. Qwen 3.5 MTP stays off |
| DRY, layer split, fit, shared cache | Already in llama.cpp |
| Receipts | A report, not a faster prompt |
| Private cache directory as a speed change | The path does not change the work |

## Rebuild

```bash
./scripts/build/build_llama_server.sh
./scripts/build/build_zerollama_mac.sh
# lab port only — do not bind :11434 or :8081
OLLAMA_HOST=127.0.0.1:11435 ZEROLLAMA_LLAMA_SERVER=1 ./zerollama serve
```

`LLAMA_CPP_VENDOR_HEAD` is still `6e18bf0`, the commit before **0162**. The vendor checkout that contains the series is `a520b5b` (186 commits past the tag). **Why the stamp file was left:** `apply-patches` treats a matching stamp plus a short commit count as "new patch files still need to be applied." Promoting the stamp early would make a checkout parked on `6e18bf0` look finished. The count check skips a second apply when the checkout is already `a520b5b`.
