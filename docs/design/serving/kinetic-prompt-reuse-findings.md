# Prompt reuse — findings

**Doc:** [kinetic-prompt-reuse.md](kinetic-prompt-reuse.md)
**Quilt:** `llama/patches/` **0162–0186** on **b12557**. Vendor checkout `a520b5b` (186 commits past `b94959a6`). `LLAMA_CPP_VENDOR_HEAD` remains `6e18bf0`.

## What was measured

Nothing in this series has a tok/s delta. The comparison that was available is the operator bench cache and the binaries that were actually running.

| Item | Result |
|------|--------|
| `~/.ollama/bench.json` | 58 completion rows, almost all **2026-07-26**. Fast end `llama3.2:1b` at 272 tok/s. Slow end is the 27B–70B tags, about 3–8 tok/s. Several fast rows are MLX |
| What that bench times | Decode tok/s of `POST /api/generate`, raw, `num_predict` 128. It unloads between models |
| Serve restarted 2026-11-21 15:24 | `zerollama` dated **2026-11-10**, `llama-server` dated **2026-11-19**. `strings` found no `PrefillBeforeLastWord`, `prefill_only`, `LLAMA_SHORTLIST`, or `LLAMA_WARM_DISK`. `GET /api/version` had no `prefill` capability |
| Verdict | That restart runs the July decode speeds. The quilt is in the source tree only |

## WHYs

1. **A cold prompt does not get faster.** The matmul was not replaced. Burst int8 stayed in Kinetic. Torque was already CUDA graphs. On an M4 the output shortlist also stays off when the head lives on Metal: `ggml_backend_buffer_is_host` is false for that buffer, and `llama_shortlist_mul` returns null. The GPU scores the whole vocab.

2. **The win is the next request that shares the prefix.** An agent resends the thread. The saved state is copied back when that copy is faster than running the prompt, including after a restart. The last logits ride with the state so the next sample does not recompute them. A draft cut before the last word leaves that word for the finished chat. A reply that repeats text already in the session can accept those tokens from the suffix index.

3. **The in-RAM prefix cache was already there.** **0167–0186** are what make a save survive the process, skip a restore that loses, and cover recurrent and sliding-window models. A fixed 8192-token mark is the wrong spacing when one recurrent cell is the expensive part (**0175**). A sliding window can resume on the same grid only while the mark is still inside the window (**0176**, **0178**).

4. **Host experts are a different machine than a model that fits.** **0170–0173** copy experts that are still host weights, and only when a GPU backend exists. A MoE that fits in unified memory keeps those tensors on Metal and never enters the cache. Discrete CUDA hits this path more often because the experts spill. `LLAMA_PAGE_IN` reads the pages at load when 3 GiB would remain, including the macOS available-memory check, so the first tokens do not fault the file.

5. **`tools/server` is not the Mac default binary.** Disk warm, the prefill route, the mark grid, and the cost gate are compiled into llama-server. The Mac ggml runner links `src/`. Darwin does not set `ZEROLLAMA_LLAMA_SERVER`. Linux `auto` does. Documenting the series as "on for every M4 chat" would be wrong until that runner is llama-server built from this tree.

6. **Go has to render the draft.** `/api/chat` always calls `chatPrompt` and then `/completion`. llama-server `/v1/prefill` renders with its own template. Those two strings are not prefixes of each other, so a cache written by one misses on the other. U+F8FF is the cut marker because a private-use character survives the template and the cut can happen after render, leaving the end-of-turn marker out of the cached text.

7. **`n_predict: 0` cannot ride the normal struct.** `llamaServerCompletionRequest.NPredict` is `omitempty`. Zero is dropped, and the server samples. Prefill posts a raw map. The sampling fields are repeated so the stored top-k row covers the follow-up.

8. **`cache_prompt` was hardcoded on.** A client `cache_reset` or OpenAI `cache_prompt: false` was ignored. Both option maps are read because the OpenAI adapter writes `options.cache_reset` and harnesses write `options.zerollama.cache_reset`.

9. **An explicit drafter suppresses the session lookup.** `common_speculative_init` adds the lookup only when the type list is a single `none` and `LLAMA_RAD1` is unset or does not start with `0`. `--spec-type none` is two `none`s and does not add it. Go omits `--spec-type` when the model has no spec type, so the lookup still turns on. Setting a type for Eliza ngram or a draft sidecar turns it off. Qwen 3.5 MTP stays disabled. **Why that split:** the lookup is the fallback for "no drafter," not a second drafter beside one the operator chose.

10. **The disk key is the forward, not the git commit.** Release, `n_ubatch`, and RoPE/YaRN change the state a later token continues from. A rebuild that does not change those would still match. Mixing `llama_build_info()` into the key would invalidate every file on every rebuild. The model file's size, mtime, and leading megabyte are in the key so a replaced GGUF at the same path starts over.

11. **File version stays 2.** Version 2 exists to reject states written one token at a time, which a GPU resume must not load. Optional trailers (logits, rate) do not need a new version. A bump deletes the directory's useful files.

12. **The near-end checkpoint offsets stay.** Prefill still breaks a batch at 4 and `4 + n_ubatch` tokens before the end. The System One primer scores the checkpoint 4 tokens before the end. Moving the grid must not remove those breaks.

13. **Multi-GPU sliding window turns checkpoints off after the grid is chosen.** The spacing is still computed. The force-off is the multi-GPU constraint, not "this model has no grid."

14. **Do not add per-request `LLAMA_*` fields.** The policies are properties of the process. The one request-scoped switch is a forced miss. `ZEROLLAMA_WARM_HYSTERESIS` is the ggml runner victim policy and is a different knob.

15. **`LLAMA_CPP_VENDOR_HEAD` stays `6e18bf0`.** That file is what `apply-patches` compares to `HEAD`. A stamp that matches an old checkout while new patch files exist is the signal to apply the missing files, not to skip them. The checkout that already has **0162–0186** is `a520b5b`. The count check (`commits past the tag` ≥ patch files) skips a second `git am` there. Promoting the stamp is an operator step after the in-tree mirror and the vendor checkout are that commit, not part of writing the patches.

16. **The Python runtime launcher was left alone.** `runtime/runtime/worker/llama_server.py` already drops `cache_prompt` on `cache_reset`. It was not pointed at the new env vars. Two launchers with two policies is the current shape. Forcing every path onto llama-server was out of scope.

## Left on purpose

Burst kernels, a flywheel EWMA on the wrong interval, a convoy sleep, a prefetch that needs a second stream, a branch mark that edits the CUDA GDN kernel, a settle piece smaller than one ubatch, and draft trees. DRY, layer split, fit, and shared cache were already llama.cpp. Receipts are a report. There is no patch **0187**.

## Open

- Rebuild `llama-server` and `zerollama` from this tree and time a repeated prompt on a lab port (`:11435`). Compare time to first token against the pre-series binary. Do not `--force` `~/.ollama/bench.json`. Do not use `:11434` or `:8081`.
- Promote `LLAMA_CPP_VENDOR_HEAD` to `a520b5b` when the operator wants the doctor to expect that checkout.
