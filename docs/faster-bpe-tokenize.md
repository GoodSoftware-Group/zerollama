# Faster BPE tokenize (gigatoken lessons)

**Status:** vendor patches  
[`0106-…id-pair…`](../llama/patches/0106-llama-vocab-id-pair-bpe-merge-gigatoken-lessons.patch) +  
[`0107-…pretok-id-cache…`](../llama/patches/0107-llama-vocab-pretok-id-cache-gigatoken-t25.patch) +  
[`0108-…pretok-materialize…`](../llama/patches/0108-unicode-faster-pretok-materialize-shared-cpts.patch) +  
[`0109-…lazy-collapse…`](../llama/patches/0109-unicode-lazy-collapse-ascii-consume-byte-span.patch) +  
[`0110-…ascii-pretok…`](../llama/patches/0110-unicode-ascii-pretok-lut-qwen2-fastpath.patch) +  
[`0111-…ltr-special…`](../llama/patches/0111-llama-vocab-ltr-special-token-partition-t47.patch) +  
[`0112-…specials-trie…`](../llama/patches/0112-llama-vocab-specials-trie-ltr-partition.patch) +  
[`0113-…byte-indexed-specials…`](../llama/patches/0113-llama-vocab-byte-indexed-specials-trie.patch) +  
[`0114-…qwen2-ascii-byte…`](../llama/patches/0114-unicode-qwen2-ascii-byte-pretok-skip-cpt-decode.patch) +  
[`0115-…gpt2-llama3-ascii…`](../llama/patches/0115-unicode-gpt2-llama3-ascii-byte-pretok.patch) +  
[`0116-…qwen35-ascii…`](../llama/patches/0116-unicode-qwen35-ascii-byte-pretok.patch) +  
[`0117-…byte-encode-lut…`](../llama/patches/0117-unicode-byte-encode-flat-lut-printable-skip.patch) +  
[`0118-…fuse-materialize…`](../llama/patches/0118-unicode-fuse-ascii-materialize-byte-encode.patch) +  
[`0119-…pretok-blob…`](../llama/patches/0119-unicode-pretok-blob-no-vector-string.patch) +  
[`0120-…pretok-cache-once…`](../llama/patches/0120-llama-vocab-pretok-cache-once-per-session.patch) +  
[`0121-…pretok-blob-general…`](../llama/patches/0121-unicode-pretok-blob-general-mixed-path.patch) +  
[`0122-…ascii-islands…`](../llama/patches/0122-unicode-qwen2-ascii-islands-mixed-pretok.patch) +  
[`0123-…gpt2-llama3-qwen35-islands…`](../llama/patches/0123-unicode-gpt2-llama3-qwen35-ascii-islands.patch) +  
[`0124-…byte-mixed…`](../llama/patches/0124-unicode-byte-mixed-pretok-islands.patch) +  
[`0125-…space-plus-printable…`](../llama/patches/0125-unicode-space-plus-printable-byte-encode.patch) +  
[`0126-…simd-swar-ascii-pretok…`](../llama/patches/0126-unicode-simd-swar-ascii-pretok-consume.patch)  
**Applies to:** `src/llama-vocab.{h,cpp}` (**0106/0107/0111–0113/0119–0121**) and `src/unicode.{h,cpp}` (**0108–0110/0114–0126**) in `vendor/llama-cpp-*`, sibling `../llama.cpp`, and in-tree `llama/llama.cpp/` (Go/CGO).  
**Findings / audit trail:** [faster-bpe-tokenize-findings.md](./faster-bpe-tokenize-findings.md)  
**ROADMAP:** Apple Silicon **M15d**

## Why this exists

Agent stacks re-tokenize **megaprompts** every turn. Stock `llama_tokenize` BPE merge did `std::string` ×2 + `unordered_map<pair<string,string>>` on every candidate pair — production `/v1/tokenize` logs showed **~270–640 ms** on Qwen/Gemma-class vocabs before any forward.

**Why not vendor Rust gigatoken?** It is a bulk-corpus CPU tokenizer (x86_64/Linux-oriented), not an inference-time drop-in that guarantees bit-identical IDs on Darwin. **Why borrow its ideas anyway?** Pair-rank tables (T19), tiered short merge (T20), and short pretok→ids cache (T25/T26) map onto llama.cpp’s BPE session without a second tokenizer.

## What changed (and why)

### Patch 0106 — merge path

1. **Load-time id-pair table** (after `token_to_id` is populated — merges load earlier as strings):
   - key = `(uint64_t)id1 << 32 | id2`
   - value = `(uint64_t)rank << 32 | merged_id`
   - flat open-addressed hash, load factor ≤ ½  
   - **Why not BMTL’s single packed value:** llama.cpp’s rank (merge-list index) ≠ merged token id; both are required at merge time.
2. **Packed piece4 table** (UTF-8 piece len ≤ 4 → token id):
   - **Why:** initial symbols are always one Unicode codepoint. Per-char `std::string` + `token_to_id` cancelled the merge win on Qwen2.
3. **ID-based soft-delete merge** with stale checks via stored `(left_id, right_id)`.
4. **Tiered short merge (T20):** ≤64 symbols → stack linear min-scan; longer words keep the priority queue.
5. **`LLAMA_BPE_FORCE_LEGACY=1`:** string-keyed path for identity A/B (re-read getenv each `tokenize()`).

### Patch 0107 — pretok→ids cache (T25/T26)

6. **Per-`llama_tokenize()` short pretok cache:**
   - key = word bytes with length **4–15**; value = **≤4** result token ids
   - open-addressed, cap 4096, load ≤ ½; first-wins on duplicate
   - skip insert when initial symbol count &lt; 3 (**why:** 1–2 symbol words are already a piece4 lookup; probe tax &gt; savings)
   - skip 1–3 byte keys (**why:** spaces / tiny tokens dominate English pretok count and regress when always hashed)
   - **`LLAMA_BPE_NO_PRETOK_CACHE=1`:** disable for A/B vs 0106-only
   - **Why per-call not cross-request:** zero lifecycle / memory growth; still captures within-prompt repetition (system prompts, code identifiers)

### Patch 0108 — pretok materialize (`unicode.cpp`)

7. **Share codepoints with custom splitters:** `unicode_regex_split` decoded UTF-8 once, then each custom qwen2/gpt2/llama3/… path decoded again — drop the second pass.
8. **`unicode_cpt_append_utf8`:** materializing words used `s += unicode_cpt_to_utf8(cpt)` (temp `std::string` per codepoint).
9. **Skip noop in `unicode_byte_encoding_process`:** words were decoded to cpts then re-encoded to the same UTF-8 before GPT-2 byte remap.

### Patch 0109 — pretok scanner hygiene (`unicode.cpp`)

10. **Lazy `text_collapsed`:** old code built the collapsed string whenever any regex mentioned `\p{L}`/`\p{N}`/… even though GPT-2/Llama/Qwen **custom** splitters never read it. Build only when `std::regex` fallback needs it.
11. **ASCII letter/digit consume (T01):** English megaprompts are ASCII-dominant; arithmetic predicates match the flags table for `c < 128`.
12. **Byte-span materialize:** keep cpt→byte offsets while decoding; on valid UTF-8, pretok words are `substr` of the original text (invalid/FFFD keeps cpt→utf8 rebuild for identity).

### Patch 0110 — ASCII pretok LUT + Qwen2 fast path

13. **128-entry ASCII flags LUT** for `c < 128` in GPT-2/Llama3/Qwen2 `_get_flags` (skip ~2MiB unicode flags table on mostly-ASCII text).
14. **Qwen2 all-ASCII segment loop** when every codepoint in a pretok segment is `< 128`.
15. **`LLAMA_BPE_NO_ASCII_PRETOK=1`:** disable the all-ASCII loop for same-binary A/B (LUT still used).

### Patch 0111 — LTR special-token partition (T47 lessons)

16. **First-byte-gated LTR longest match** for `tokenizer_st_partition` — avoid O(|specials| × fragments × find) on chat megaprompts.
17. **`LLAMA_BPE_FORCE_LEGACY_SPECIALS=1`:** old nested-find algorithm for identity A/B.

### Patch 0112 — specials trie + memchr skip

18. **Load-time `naive_trie` over `cache_special_tokens`:** at each candidate, walk match length instead of memcmp'ing every special that shares the first byte (Qwen ~200 start with `<`).
19. **`memchr` skip** when only one interesting first byte — jump to the next `<` (etc.) without byte-by-byte scanning.

### Patch 0113 — byte-indexed specials trie + load-time gates

20. **`llm_specials_byte_trie`:** `child[256]` indices (no `std::map` per step).
21. **Load-time `specials_interesting` / `_ud` masks** — avoid rebuilding the first-byte gate every `tokenize()`.
22. **Harness:** identity A/B vs `LLAMA_BPE_FORCE_LEGACY_SPECIALS` on mixed/chat seeds.

### Patch 0114 — all-ASCII Qwen2 byte pretok

23. **Skip uint32 decode** when text bytes are all `< 0x80` and the sole regex is Qwen2 custom — pretok on bytes, `substr` words.
24. **8-wide letter consume** on the byte/cpt ASCII scanners.
25. Still gated by `LLAMA_BPE_NO_ASCII_PRETOK=1` (falls back to cpt + unicode path).

### Patch 0115 — GPT-2 + Llama3 all-ASCII byte pretok

26. Same skip-decode path for GPT-2 and Llama3 custom regex strings (case-sensitive GPT-2 contractions; Llama3 `\p{N}{1,3}`).

### Patch 0116 — Qwen3.5 all-ASCII byte pretok

27. **Qwen3.5** (`[\p{L}\p{M}]+`) routes all-ASCII to the Qwen2 byte/ascii_seg scanners — no U+00–7F codepoint is `\p{M}`.

### Patch 0117 — GPT-2 byte-encode flatten + printable skip

28. **Flat `enc[256]` LUT** instead of `unordered_map` + `string +=` per byte.
29. **Skip remap** when a pretok word is only `0x21..0x7E` (identity under bytes↔unicode).
30. **`LLAMA_BPE_NO_BYTE_ENC_FAST=1`** for same-binary A/B.

### Patch 0118 — fuse ASCII materialize + byte-encode

31. **One-pass** offsets → remapped words (no intermediate substring vector).

### Patch 0119 — pretok blob (no `vector<string>`)

32. **`unicode_pretok_blob`:** ASCII custom pretok emits `storage` + `lens` (or views into `text` when all words are printable ASCII).
33. **BPE session** walks `(ptr,len)` — avoids ~370k `std::string`/MiB.
34. **`LLAMA_BPE_NO_PRETOK_BLOB=1`:** fall back to `unicode_regex_split` → `vector<string>` for A/B.

### Patch 0120 — pretok cache once per session

35. **Init pretok→ids cache once per BPE session**, not at the start of every `tokenize()` call.
36. **Why:** `tokenizer_st_partition` invokes `session.tokenize()` once per fragment between specials; re-`assign(4096)` + cold cache each fragment made mixed/chat megaprompts **~2.6× slower** with cache ON than `NO_PRETOK_CACHE`.
37. Session still lives for one `llama_tokenize` only — no cross-request lifecycle.

### Patch 0121 — pretok blob on mixed + ASCII bulk cpt decode

38. **`unicode_regex_split_try_blob` always fills** (ASCII custom or general cpt path) unless `NO_PRETOK_BLOB`.
39. **8-wide ASCII cpt fill** before `unicode_cpt_from_utf8` on the general path (ASCII byte == codepoint).

### Patch 0122 — ASCII islands in mixed Qwen2 pretok

40. **When a segment is not all-ASCII**, split into ASCII gaps (`ascii_seg`) and non-ASCII islands (`unicode_seg`) instead of running the Unicode scanner on the whole MiB.
41. **Letter islands** expand through contiguous `\p{L}` (keeps `café` / `hello世界`).
42. **Punctuation/emoji islands** keep optional leading space and the `?[^\s\p{L}\p{N}]+[\r\n]*` span (` ·`, ` 🚀\n`) — required for identity.
43. Still gated by `LLAMA_BPE_NO_ASCII_PRETOK=1`. Dense mixed ~**1.1×** vs that toggle; ASCII-majority with rare CJK is the intended win.

### Patch 0123 — ASCII islands for GPT-2 / Llama3 / Qwen3.5

44. Same mixed-island idea as 0122 for the other common custom pretok families.
45. **GPT-2:** optional leading space on letter/number/punct (` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s…]`).
46. **Llama3:** Qwen2-like letter/punct islands; numbers stay `\p{N}{1,3}` inside `unicode_seg`.
47. **Qwen3.5:** letter islands expand through `\p{L}`/`\p{M}`; ASCII gaps reuse Qwen2 `ascii_seg`.

### Patch 0124 — byte-level mixed pretok islands

48. **Before full uint32 decode**, split mixed custom pretok on raw bytes: `ascii_bytes_seg` on ASCII gaps; decode only non-ASCII islands into cpts for `unicode_seg`.
49. Same island boundary rules as 0122/0123 (letter/punct/number + optional space).
50. **`LLAMA_BPE_NO_BYTE_MIXED=1`:** fall through to the cpt path (still has 0122/0123 islands) for A/B.
51. Dense mixed Qwen2 ~**1.05–1.1×** vs `NO_BYTE_MIXED`; identity green (byte-mixed gate).

### Patch 0125 — space + printable byte-encode fast path

52. **Leading-space pretok words** (` ?\p{L}+`) failed the 0117 printable skip because space remaps to U+0120.
53. **Remap space once, `memcpy` the printable rest** (identity under bytes↔unicode for `0x21..0x7E`).
54. Still gated by `LLAMA_BPE_NO_BYTE_ENC_FAST=1`. Qwen2 ASCII ~**7.5 ms/MiB** (was ~8.5); GPT-2 ASCII ~**6.7 ms**.

### Patch 0126 — SWAR/NEON ASCII letter+digit consume

55. **Borrow-safe SWAR** (`hasless`/`hasgreater`) + **aarch64 NEON** accelerate `\p{L}+` / `\p{N}+` consume on proven-ASCII byte pretok paths (BMTL T01/T05 lessons — consume only, not a full mask-scanner rewrite).
56. **SWAR-first**; NEON only after a full 8-letter/digit hit (short English words skip NEON tax).
57. **`LLAMA_BPE_NO_SIMD_PRETOK=1`:** keep the 0114 8-wide LUT loop for A/B. Harness gates `simd-pretok`.
58. Median A/B (Qwen2, load-sensitive abs ms): ASCII ~**1.3×** vs `NO_SIMD`; chat/mixed ~**1.1–1.25×**.

Specials / BOS / EOS / `ignore_merges` / Gemma4 newline whole-word hit unchanged. Full pretok mask-scanner rewrite still deferred.

## Pin / apply

**Why three trees:** `vendor/` is `git am` target; `llama/llama.cpp/` is what Go `#cgo` compiles; `../llama.cpp` is lab `libllama`.

```bash
./scripts/vendor/apply_llama_vendor_patches.sh   # includes 0106 → 0126
./scripts/vendor/sync_vendor_llama.sh            # rsync → llama/llama.cpp (CGO) + ggml
# sibling (optional):
patch -p1 -d ../llama.cpp < llama/patches/0106-llama-vocab-id-pair-bpe-merge-gigatoken-lessons.patch
patch -p1 -d ../llama.cpp < llama/patches/0107-llama-vocab-pretok-id-cache-gigatoken-t25.patch
patch -p1 -d ../llama.cpp < llama/patches/0108-unicode-faster-pretok-materialize-shared-cpts.patch
patch -p1 -d ../llama.cpp < llama/patches/0109-unicode-lazy-collapse-ascii-consume-byte-span.patch
patch -p1 -d ../llama.cpp < llama/patches/0110-unicode-ascii-pretok-lut-qwen2-fastpath.patch
patch -p1 -d ../llama.cpp < llama/patches/0111-llama-vocab-ltr-special-token-partition-t47.patch
patch -p1 -d ../llama.cpp < llama/patches/0112-llama-vocab-specials-trie-ltr-partition.patch
patch -p1 -d ../llama.cpp < llama/patches/0113-llama-vocab-byte-indexed-specials-trie.patch
patch -p1 -d ../llama.cpp < llama/patches/0114-unicode-qwen2-ascii-byte-pretok-skip-cpt-decode.patch
patch -p1 -d ../llama.cpp < llama/patches/0115-unicode-gpt2-llama3-ascii-byte-pretok.patch
patch -p1 -d ../llama.cpp < llama/patches/0116-unicode-qwen35-ascii-byte-pretok.patch
patch -p1 -d ../llama.cpp < llama/patches/0117-unicode-byte-encode-flat-lut-printable-skip.patch
patch -p1 -d ../llama.cpp < llama/patches/0118-unicode-fuse-ascii-materialize-byte-encode.patch
patch -p1 -d ../llama.cpp < llama/patches/0119-unicode-pretok-blob-no-vector-string.patch
patch -p1 -d ../llama.cpp < llama/patches/0120-llama-vocab-pretok-cache-once-per-session.patch
patch -p1 -d ../llama.cpp < llama/patches/0121-unicode-pretok-blob-general-mixed-path.patch
patch -p1 -d ../llama.cpp < llama/patches/0122-unicode-qwen2-ascii-islands-mixed-pretok.patch
patch -p1 -d ../llama.cpp < llama/patches/0123-unicode-gpt2-llama3-qwen35-ascii-islands.patch
patch -p1 -d ../llama.cpp < llama/patches/0124-unicode-byte-mixed-pretok-islands.patch
patch -p1 -d ../llama.cpp < llama/patches/0125-unicode-space-plus-printable-byte-encode.patch
patch -p1 -d ../llama.cpp < llama/patches/0126-unicode-simd-swar-ascii-pretok-consume.patch
```

```bash
grep -c has_bpe_id_pairs llama/llama.cpp/src/llama-vocab.cpp   # > 0
grep -c bpe_piece4 llama/llama.cpp/src/llama-vocab.cpp          # > 0
grep -c pretok_cache llama/llama.cpp/src/llama-vocab.cpp        # > 0
grep -c unicode_cpt_append_utf8 llama/llama.cpp/src/unicode.cpp # > 0
grep -cE 'ensure_text_collapsed|had_invalid_utf8' llama/llama.cpp/src/unicode.cpp # > 0
grep -c unicode_regex_split_qwen2_ascii_seg llama/llama.cpp/src/unicode.cpp # > 0
grep -c LLAMA_BPE_FORCE_LEGACY_SPECIALS llama/llama.cpp/src/llama-vocab.cpp # > 0
grep -c llm_specials_byte_trie llama/llama.cpp/src/llama-vocab.cpp # > 0
grep -c unicode_regex_split_qwen2_ascii_bytes llama/llama.cpp/src/unicode.cpp # > 0
grep -c unicode_regex_split_gpt2_ascii_bytes llama/llama.cpp/src/unicode.cpp # > 0
grep -c unicode_is_qwen35_regex_expr llama/llama.cpp/src/unicode.cpp # > 0
grep -c LLAMA_BPE_NO_BYTE_ENC_FAST llama/llama.cpp/src/unicode.cpp # > 0
grep -c unicode_byte_enc_table llama/llama.cpp/src/unicode.cpp # > 0
grep -c unicode_regex_split_try_blob llama/llama.cpp/src/unicode.cpp # > 0
grep -c pretok_cache_ready llama/llama.cpp/src/llama-vocab.cpp # > 0
grep -c unicode_fill_blob_from_cpt_offsets llama/llama.cpp/src/unicode.cpp # > 0
grep -c unicode_regex_split_qwen2_mixed_seg llama/llama.cpp/src/unicode.cpp # > 0
grep -c unicode_regex_split_gpt2_mixed_seg llama/llama.cpp/src/unicode.cpp # > 0
grep -c LLAMA_BPE_NO_BYTE_MIXED llama/llama.cpp/src/unicode.cpp # > 0
grep -c unicode_word_is_space_plus_printable llama/llama.cpp/src/unicode.cpp # > 0
grep -c LLAMA_BPE_NO_SIMD_PRETOK llama/llama.cpp/src/unicode.cpp # > 0
```

`runtime/LLAMA_CPP_PIN.md` lists through **0126**.

## Benchmarks (Mac aarch64, vocab-only GGUFs)

**Method:** warmup; alternate order; `FORCE_LEGACY` / `NO_PRETOK_CACHE` toggles; for 0108 use pristine-`unicode.cpp` rebuild A/B (no env toggle). Absolute ms are load-sensitive.

### English repeating research seed (1 MiB) — after 0106 (+0107 ≈ wash)

| Vocab | legacy | id-pair (±cache) | speedup |
|-------|------:|------:|------:|
| Qwen2 | ~720–970 ms | ~675–900 ms | **~1.07×** (pretok-bound) |
| Llama3 BPE | ~74 ms | ~59 ms | **~1.27×** |
| Gemma4 | ~108 ms | ~38 ms | **~2.8×** |

### Code-like repeated identifiers (1 MiB) — where 0107 shows up

| Vocab | legacy | nocache (0106) | cache (0106+0107) |
|-------|------:|------:|------:|
| Qwen2 | ~142 ms | ~58 ms | **~46 ms (~1.25× vs 0106, ~3.1× vs legacy)** |
| Gemma4 | ~136 ms | ~50 ms | ~49 ms (neutral vs 0106) |
| Llama3 BPE | ~91 ms | ~45 ms | ~48 ms (≈noise vs 0106) |

### 0108 vs pristine unicode (mega_1mib fast path, same 0106+0107 merge)

| Vocab | pristine unicode | +0108 | speedup |
|-------|------:|------:|------:|
| GPT-2 | ~37 ms | ~20 ms | **~1.9×** (byte-encode path) |
| Llama3 BPE | ~71 ms | ~49 ms | **~1.4×** |
| Gemma4 | ~38 ms | ~32 ms | **~1.2×** |
| Qwen2 | ~721 ms | ~728 ms | ≈ wash (custom regex still dominates) |

### 0109 vs 0108 (mega_1mib interleaved)

≈ **noise / ~1.0–1.05×** on Qwen2/Llama/GPT-2. Ships because it removes a wasted collapse pass and cpt→utf8 rebuild on valid UTF-8.

### 0110 + measurement trap (Qwen2, Mac aarch64)

| Seed | What it stresses | ~ms / 1 MiB (after 0126) |
|------|------------------|------------:|
| `mega_1mib` (mixed Unicode) | byte-encode + specials | **~19–20** (~1.1× vs `NO_SIMD` median on top of 0122–0125) |
| `mega_1mib_chat` (ASCII + `<|im_start|>`) | special-token partition | **~14** (0120; was ~94 before session cache) |
| `mega_1mib_ascii` (no specials) | pretok + BPE on plain English | **~7–8** (0125 encode + 0126 SWAR; A/B ~1.3× vs `NO_SIMD`) |

Use `LLAMA_BPE_FORCE_LEGACY_SPECIALS=1` to A/B specials partition. Do **not** cite pre-0111 mixed/`<|im_start|>` seeds as “English pretok-bound.”

**Why English wash / code win (0107):** after 0106, English megaprompts are mostly `unicode_regex_split`; code-like text has longer 4–15B identifiers that still need multi-symbol merge — memoizing those skips merge on every repeat within the call.

**Why 0108 helps GPT-2 more than Qwen:** GPT-2 pays for byte remapping + word rebuild; Qwen English time is inside the custom category scanner, not materialize.

Do **not** cite unreproducible early “6–22×” `/tmp` figures ([findings](./faster-bpe-tokenize-findings.md)).

## Identity gate

```bash
./scripts/bench/run_tokenize_bpe_identity_bench.sh
./scripts/bench/run_tokenize_bpe_identity_bench.sh --bench
```

Fast (cache on) vs `LLAMA_BPE_FORCE_LEGACY=1` must be bit-identical (merge path). **0108/0109** must also match prior-`unicode.cpp` token dumps (pretok path — `FORCE_LEGACY` does not cover this). **0122** harness also A/Bs `LLAMA_BPE_NO_ASCII_PRETOK` on mixed seeds (islands vs full Unicode pretok). **0124** A/Bs `LLAMA_BPE_NO_BYTE_MIXED` (byte islands vs cpt islands). **0126** A/Bs `LLAMA_BPE_NO_SIMD_PRETOK` (SWAR/NEON vs LUT consume). Load log: `bpe_id_pairs = N`, `bpe_piece4 = M`.

## Non-goals (why)

- No Rust gigatoken / Cargo — Darwin + CGO simplicity.
- No cross-request pretok cache yet — lifecycle/memory; measure first.
- No MT corpus encode on the serve path.
- No full pretok mask-scanner rewrite — **0126** is letter/digit **consume** only on proven-ASCII byte paths.
- No vendoring BMTL’s `bmtl_pair_rank_table`.

## References

- Findings: [faster-bpe-tokenize-findings.md](./faster-bpe-tokenize-findings.md)
- BMTL: `../bmtl/docs/sources/gigatoken_techniques.md` (T19/T20/T25/T26)
- Upstream: [ggml-org/llama.cpp#26139](https://github.com/ggml-org/llama.cpp/issues/26139)
