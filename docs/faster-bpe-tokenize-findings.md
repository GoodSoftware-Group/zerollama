# Faster BPE tokenize — findings & learnings

**Why this doc:** Capture what we tried, what failed measurement-wise, and what actually shipped so the next agent does not re-learn the same traps. Operator how-to stays in [faster-bpe-tokenize.md](./faster-bpe-tokenize.md).

**Patch:** [`llama/patches/0106-llama-vocab-id-pair-bpe-merge-gigatoken-lessons.patch`](../llama/patches/0106-llama-vocab-id-pair-bpe-merge-gigatoken-lessons.patch)  
**Date:** Jul 2026 · **ROADMAP:** **M15d** · **Lab host:** Mac aarch64 · **Pin surface:** `vendor/llama-cpp-c84b3020` / sibling `../llama.cpp` @ `c84b3020`

---

## Problem (why we looked)

Agent and tool workflows re-tokenize **megaprompts** (100 KiB–1 MiB) on every turn via `llama_tokenize` (CGO ggml path and Python runtime → llama.cpp). Production `/v1/tokenize` logs showed **~270–640 ms** per call on Qwen/Gemma-class vocabs. That is pure CPU before any forward pass — felt latency, not theoretical.

**Why not port Rust gigatoken wholesale?**

| Gigatoken / BMTL | Zerollama need |
|------------------|----------------|
| Bulk corpus GB/s on x86_64/Linux (+ limited Windows) | Per-request encode on Darwin + Linux, identical IDs to stock |
| Own BPE stack / optional SIMD pretok | Must stay inside vendored `llama-vocab.cpp` |
| PairRankTable packs one value | llama.cpp needs **rank + merged_id** (two 32-bit fields) |

**Why borrow techniques anyway:** T19 (id-pair rank table) and T20 (tiered short merge) attack the same hot path stock llama.cpp has: string-keyed merge lookups. Lessons without the Rust crate.

---

## What we shipped (and why each piece)

| Piece | Why |
|-------|-----|
| **`(id,id) → (rank, merged_id)` open-addressed table** | Removes `std::string` concat + `unordered_map<pair<string,string>>` on every candidate bigram. Rank is merge-list index; merged_id is the vocab token — **must pack both** (BMTL’s single slot cannot). |
| **`llm_symbol.id`** | Soft-delete stale check without rebuilding left+right strings. |
| **piece4 LUT (UTF-8 len ≤ 4 → token id)** | Initial BPE symbols are always **one Unicode codepoint**. Char→id via `std::string` + `token_to_id` cancelled the merge win on Qwen2 (byte-encode, many short words). Packed keys avoid that alloc/hash. |
| **Tiered merge ≤64 symbols (linear min-scan)** | Priority-queue overhead dominates tiny words (gigatoken T20). Tie-break uses strict `<` on rank while scanning left→right so leftmost wins — **matches** `llm_bigram_bpe::comparator`. |
| **Per-call pretok→ids cache (T25/T26, patch 0107)** | After merge is cheap, repeated 4–15B identifiers still re-merge. Memoize ≤4 result ids within one `tokenize()`; miss = same path (identity-safe). Skip 1–3B keys and &lt;3-symbol words (probe tax &gt; savings). |
| **Pretok materialize (patch 0108)** | After 0106+0107, English Qwen still spends time in `unicode_regex_split`. Kill double `cpts_from_utf8`, per-cpt temp strings, and GPT-2 byte-encode noop roundtrip — without a SIMD pretok rewrite. |
| **Lazy collapse + ASCII consume + byte-span (0109)** | Custom paths never used `text_collapsed` but still built it; letter loops hit a ~2MiB flags table; materialize rebuilt UTF-8. |
| **ASCII pretok LUT + Qwen2 fast path (0110)** | Mostly-ASCII scanners still indexed the big flags table; all-ASCII segments can use a tight loop. |
| **LTR specials partition (T47, 0111)** | Chat megaprompts: ~294 specials × full-text find (+ fragment explosion) dominated tokenize. |
| **Specials trie + memchr (0112)** | At each `<`, avoid scanning every special that shares the first byte; `memchr` when one interesting byte. |
| **Byte-indexed specials trie (0113)** | O(1) child steps + load-time interesting masks; harness `FORCE_LEGACY_SPECIALS` identity. |
| **Qwen2 ASCII byte pretok (0114)** | All-ASCII Qwen2: pretok on bytes, skip cpt decode; 8-wide letter consume. |
| **GPT-2/Llama3 ASCII byte pretok (0115)** | Same for GPT-2 + Llama3 custom regexes. |
| **Qwen3.5 ASCII byte pretok (0116)** | All-ASCII Qwen3.5 ≡ Qwen2 scanner (`\p{M}` empty on ASCII). |
| **Byte-encode LUT + printable skip (0117)** | Flat enc table; skip remap for `!`–`~` words. |
| **Fuse materialize + encode (0118)** | One-pass offsets → remapped words on ASCII byte path. |
| **Pretok blob (0119)** | `unicode_pretok_blob` + BPE `(ptr,len)`; skip ~370k strings/MiB; printable-all views into text. |
| **Session pretok cache (0120)** | Init cache once per BPE session — specials fragments reuse the warmed table. |
| **Mixed pretok blob (0121)** | General-path blob + 8-wide ASCII cpt fill on decode. |
| **ASCII islands mixed Qwen2 (0122)** | ASCII gaps → `ascii_seg`; letter/punct non-ASCII islands → `unicode_seg`; keep ` ·` / ` 🚀` spans. |
| **ASCII islands GPT-2/Llama3/Qwen3.5 (0123)** | Family-specific mixed islands (GPT-2 ` ?` space; Llama3 N{1,3}; Qwen3.5 marks). |
| **Byte-level mixed pretok (0124)** | ASCII gaps via `ascii_bytes_seg`; decode only non-ASCII islands. |
| **Space+printable byte-encode (0125)** | Leading-space pretok remaps space once, memcpy rest. |
| **SWAR/NEON pretok consume (0126)** | Borrow-safe SWAR + NEON letter/digit consume on ASCII byte paths. |
| **`LLAMA_BPE_FORCE_LEGACY=1`** | Same binary, two paths → bit-identical A/B without a second build. **Re-read getenv each `tokenize()`** so the harness can flip mid-process (do not cache forever). |
| **`LLAMA_BPE_NO_PRETOK_CACHE=1`** | Measure 0107 contribution vs 0106-only without a second build. |
| **`LLAMA_BPE_NO_PRETOK_BLOB=1`** | Measure 0119 vs `vector<string>` materialize. |
| **`LLAMA_BPE_NO_ASCII_PRETOK=1`** | Disable ASCII pretok + 0122 islands (full Unicode scanner). |
| **`LLAMA_BPE_NO_SIMD_PRETOK=1`** | Disable 0126 SWAR/NEON; keep 8-wide LUT consume. |
| **Legacy string path kept** | Empty id-pair table or force-legacy still works; identity gate needs a ground-truth path. |

**Unchanged on purpose:** pretok regex *structure*, specials, BOS/EOS, `ignore_merges`, Gemma4 newline whole-word hit. **Why:** those are correctness-sensitive; 0126 only accelerates letter/digit **consume** on proven-ASCII byte paths.

**Deferred:** dense PairRankTable grid, full pretok mask-scanner rewrite, Rust submodule, cross-request word→ids cache. **Why:** identity risk and diminishing returns after piece4 + id-pair + ASCII blob + 0126 consume on ship vocabs.

---

## Measurement lessons (why early numbers lied)

1. **Throwaway `/tmp` benches are not evidence.** Early “6–22×” numbers came from uncommitted scripts and **did not reproduce**. Always keep the harness in-repo (`scripts/bench/tokenize_bpe_identity_bench.cpp`).
2. **In-binary `FORCE_LEGACY` A/B is valid only if both paths run after warmup and you alternate order.** Cold first run + host load looked like a “regression” on Qwen2 until fair alternating A/B.
3. **`NO_PRETOK_CACHE` faster than cache ON was a fragment-init bug, not “cache is bad.”** Before 0120, mixed/chat megaprompts paid `slots.assign(4096)` once per specials fragment; disabling cache skipped that tax and looked like a win. After 0120, cache ON is faster on mixed too.
4. **Pristine second binary confirms the toggle.** Building HEAD without the patch in isolation matched `FORCE_LEGACY=1` within noise — so the env gate is trustworthy.
5. **Vocab-dependent wins.** After piece4 (1 MiB seed, fair A/B):

   | Vocab | Speedup | Why that shape |
   |-------|--------:|----------------|
   | Gemma4 | **~2.8×** | Heavy merge; `byte_encode=false`; merge was a large fraction of time |
   | Llama3 BPE | **~1.27×** | Real but smaller |
   | Qwen2 | **~1.07×** | Pretok (`unicode_regex_split`) still dominates; merge was never the whole story |

6. **Identity ≠ golden HF vectors.** Fast vs legacy in one binary proves we did not change *our* algorithm. Prefer also `test-tokenizer-0` when the full CMake test target builds (this lab tree had unrelated ANE WIP linker breaks).
7. **Mixed Unicode / chat specials are not “English pretok.”** The old identity-bench seed embedded CJK/emoji (Qwen **byte-encode** merge cost) and `<|im_start|>` every ~150 B (**special-token scan**). Both look like ~500–900 ms/MiB. Pure ASCII without specials is ~**8–9 ms/MiB** after 0114–0120. Use `mega_1mib_ascii` for pretok claims.
8. **Island splits must keep ` ?[^\s\p{L}\p{N}]+` spans.** Early 0122 drafts split ` ·` / ` 🚀` into space + special and broke `NO_ASCII_PRETOK` identity even when `FORCE_LEGACY` still matched (merge-only gate). Harness now A/Bs islands vs `NO_ASCII_PRETOK` on mixed seeds.
9. **NEON `vminvq` on an inverted letter mask is “any letter”, not “all letters.”** Early 0126 used `vminvq(vmvn(is_let))==0` and swallowed spaces into letter runs → ~10× BPE blow-up. Use `vmaxvq(vmvn(...))==0` (or `vminvq(is_let)==0xFF`). SWAR letter checks must be **borrow-safe** (`hasless`/`hasgreater`). Always A/B `LLAMA_BPE_NO_SIMD_PRETOK` — `FORCE_LEGACY` does not cover pretok.

---

## Wiring traps (why three trees)

| Tree | Role | Why it matters |
|------|------|----------------|
| `vendor/llama-cpp-*` | `git am` target for `llama/patches/` | Source of truth for patch series |
| `llama/llama.cpp/` | **What Go `#cgo` compiles** (`llama/llama.go`) | Shipping only sibling/vendor leaves Mac ggml tokenize **unpatched** |
| `../llama.cpp` | Lab `llama-server` / `libllama` benches | Easy to “prove” a win that never reaches CGO |

**Why audit found a silent miss:** a truncated/empty patch “applied” with exit 0 while in-tree vocab still had zero `has_bpe_id_pairs`. Always `grep has_bpe_id_pairs llama/llama.cpp/src/llama-vocab.cpp` after apply, then `./scripts/vendor/sync_vendor_llama.sh`.

---

## Correctness rules we locked in

- **Bit-identical** fast vs `LLAMA_BPE_FORCE_LEGACY=1` on size ladder + edge snippets.
- Load skips merge rows whose left/right/merged text is missing from `token_to_id` (**lenient**) — **why:** some GGUFs have orphan merge strings; failing the whole load is worse than falling back for those pairs.
- Duplicate id-pair keys keep **first** insert — **why:** matches `unordered_map::emplace` first-wins.
- piece4 keeps first insert for duplicate packed pieces.

---

## Open follow-ups (with why)

| Idea | Why consider | Why not yet / status |
|------|--------------|----------------------|
| **Per-call pretok→ids cache (T25/T26)** | Repeated identifiers re-merge after 0106 | **Shipped 0107** — helps code-like; wash on English pretok-bound |
| **Pretok materialize (0108)** | Double decode + temp strings + byte-encode noop | **Shipped 0108** — GPT-2 ~1.9× / Llama ~1.4× / Gemma ~1.2× vs pristine unicode; Qwen English ≈ wash |
| **Lazy collapse + ASCII + byte-span (0109)** | Wasted collapse; flags-table letter loops; cpt→utf8 rebuild | **Shipped 0109** — identity green; **≈noise vs 0108** on Qwen English (scanner still dominates) |
| **ASCII pretok LUT + Qwen2 fast path (0110)** | Mostly-ASCII still used big flags table | **Shipped 0110** — pure ASCII ~18 ms/MiB; ~1.0–1.07× vs 0109 on that shape |
| **LTR specials partition (T47, 0111)** | Repeated `<|im_start|>` → hundreds of ms | **Shipped 0111** — chat seed ~**8.5×** (~850→~99 ms); `FORCE_LEGACY_SPECIALS` identity green |
| **Specials trie + memchr (0112)** | 0111 still memcmp'd ~200 `<…` specials per candidate | **Shipped 0112** — dense markers ~**65 ms** vs ~**217 ms** legacy specials (~3.3×); identity green |
| **Byte-indexed specials trie (0113)** | map walk + per-call interesting rebuild | **Shipped 0113** — ≈noise vs 0112 on dense chat; harness specials identity; load-time gates |
| **Qwen2 ASCII byte pretok (0114)** | uint32 decode + 4×RAM before ascii_seg | **Shipped 0114** — ~**11.4 ms**/MiB vs ~15 ms `NO_ASCII_PRETOK` (~1.3×); identity green |
| **GPT-2/Llama3 ASCII byte pretok (0115)** | Same skip-decode for other common BPE families | **Shipped 0115** — GPT-2 ~1.34× / Llama3 ~1.19× vs `NO_ASCII_PRETOK`; identity green |
| **Qwen3.5 ASCII byte pretok (0116)** | `[\p{L}\p{M}]+` but no ASCII is `\p{M}` | **Shipped 0116** — reuse Qwen2 scanner; ~1.24× vs `NO_ASCII_PRETOK`; identity green |
| **Byte-encode LUT + printable skip (0117)** | GPT-2 remap used map+`+=` per byte | **Shipped 0117** — ~**1.33×** vs `NO_BYTE_ENC_FAST` on Qwen/GPT-2 ASCII; identity green |
| **Fuse materialize + encode (0118)** | Double-copy printable pretok words | **Shipped 0118** — ~**1.42×** vs `NO_BYTE_ENC_FAST`; ~8.8 ms/MiB Qwen2 ASCII; identity green |
| **Pretok blob / no vector&lt;string&gt; (0119)** | ~370k `std::string`/MiB after offsets | **Shipped 0119** — ~**1.09×** vs `NO_PRETOK_BLOB` (~8.7 vs ~9.5 ms); printable-all views into text; identity green |
| **Session pretok cache (0120)** | Cache re-init per specials fragment | **Shipped 0120** — mixed ~**21 ms** (was ~91); chat ~**15 ms** (was ~94); ascii unchanged ~9 ms; identity green |
| **Mixed pretok blob + ASCII cpt fill (0121)** | Mixed still built N strings; per-byte UTF-8 decode | **Shipped 0121** — mixed ~**19.5 ms** (~1.1× vs `NO_PRETOK_BLOB`); identity green |
| **ASCII islands mixed Qwen2 (0122)** | One non-ASCII byte forced full Unicode scanner on MiB | **Shipped 0122** — mixed ~**19.5 ms** (~1.1× vs `NO_ASCII_PRETOK`); keep ` ·`/` 🚀` spans; identity green (incl. ascii-islands gate) |
| **ASCII islands GPT-2/Llama3/Qwen3.5 (0123)** | Same gap for other BPE families | **Shipped 0123** — identity green on GPT-2/Llama3/Qwen3.5; dense mixed ~noise–1.05× |
| **Byte-level mixed pretok (0124)** | Full uint32 decode before cpt islands | **Shipped 0124** — ~**1.07×** vs `NO_BYTE_MIXED` on dense Qwen2 mixed; identity green |
| **Space+printable byte-encode (0125)** | Space failed printable skip → per-letter LUT | **Shipped 0125** — Qwen2 ASCII ~**7.5 ms** (was ~8.5); GPT-2 ~**6.7 ms**; identity green |
| **SWAR/NEON pretok consume (0126)** | 8-wide LUT letter/digit consume | **Shipped 0126** — borrow-safe SWAR + NEON; ~**1.3×** ASCII / ~**1.1–1.25×** chat vs `NO_SIMD`; identity green (`simd-pretok`) |
| Cross-request pretok cache | Agent threads re-send same system prompt across HTTP calls | Lifecycle + RAM; measure production first |
| Full pretok mask-scanner rewrite | Remaining gain after consume acceleration | High footgun; 0126 covers the hot consume only |
| Production before/after `/v1/tokenize` on `:11434` | Only real user-facing proof | Operator window; do not restart prod ports from agents |
| Upstream `test-tokenizer-0` in CI | Golden HF vectors | Full llama.cpp test link blocked by unrelated WIP here |

---

## References

- Operator doc: [faster-bpe-tokenize.md](./faster-bpe-tokenize.md)
- BMTL techniques: `../bmtl/docs/sources/gigatoken_techniques.md` (T19/T20)
- Upstream ask: [ggml-org/llama.cpp#26139](https://github.com/ggml-org/llama.cpp/issues/26139)
- Not used: [chynggi/gigatoken-llama.cpp](https://github.com/chynggi/gigatoken-llama.cpp) (Rust + non-Darwin)
