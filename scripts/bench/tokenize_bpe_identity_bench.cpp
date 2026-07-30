// Identity + throughput bench for llama.cpp BPE tokenize (patches 0106–0121 / M15d).
// Prefer: ./scripts/bench/run_tokenize_bpe_identity_bench.sh [--bench]
//
// WHY: wrong token IDs silently break chat templates, tool parsers, and L3 keys.
// LLAMA_BPE_FORCE_LEGACY=1 selects the string-keyed path inside the same binary
// so fast vs legacy can be compared without a second build (identity gate).
// WHY flip getenv mid-process: has_bpe_id_pairs() re-reads each tokenize() on purpose.
// Docs: docs/faster-bpe-tokenize.md , docs/faster-bpe-tokenize-findings.md

#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<llama_token> tokenize_all(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> tok(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tok.data(), (int32_t) tok.size(), true, true);
    if (n < 0) {
        tok.resize((size_t) (-n));
        n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tok.data(), (int32_t) tok.size(), true, true);
    }
    if (n < 0) {
        return {};
    }
    tok.resize((size_t) n);
    return tok;
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now() - t0).count();
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s vocab.gguf [--bench] [--no-identity]\n", argv[0]);
        return 1;
    }
    const char * path = argv[1];
    bool do_bench = false;
    bool do_identity = true;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--bench") == 0) {
            do_bench = true;
        }
        if (strcmp(argv[i], "--no-identity") == 0) {
            do_identity = false;
        }
    }

    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mparams);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", path);
        return 2;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // WHY two seeds: mixed Unicode (café/🚀/CJK) triggers Qwen byte-encode merge cost and
    // was misread as "pretok-bound English". Pure ASCII matches real English agent megaprompts.
    std::string seed_mixed = "Hello world! The quick brown fox jumps over the lazy dog. "
                             "Qwen测试 日本語 café 🚀\n"
                             "<|im_start|>user\nWhat is 2+2?<|im_end|>\n";
    // Pure ASCII without chat specials — specials (`<|im_start|>` every ~150B) dominate
    // via the special-token scanner (patches 0111/0112), which is a separate lever from pretok/BPE.
    std::string seed_ascii = "Hello world! The quick brown fox jumps over the lazy dog. "
                             "Research notes and agent tool transcripts. "
                             "User: What is 2+2?\n";
    // ASCII + repeated chat markers — stresses tokenizer_st_partition (0111/0112).
    std::string seed_chat = "Hello world! The quick brown fox jumps over the lazy dog. "
                            "Research notes.\n"
                            "<|im_start|>user\nWhat is 2+2?<|im_end|>\n";

    struct Case {
        const char * name;
        size_t nbytes;
        int iters;
        int seed_kind; // 0=mixed, 1=ascii, 2=chat
    };
    Case cases[] = {
        {"tiny", 64, 200, 0},
        {"chat_2kib", 2u << 10, 50, 0},
        {"tools_32kib", 32u << 10, 20, 0},
        {"mega_256kib", 256u << 10, 5, 0},
        {"mega_1mib", 1u << 20, 3, 0},
        {"mega_1mib_ascii", 1u << 20, 5, 1},
        {"mega_1mib_chat", 1u << 20, 5, 2},
    };

    int failures = 0;
    for (const auto & c : cases) {
        const std::string & seed = (c.seed_kind == 1) ? seed_ascii : (c.seed_kind == 2) ? seed_chat : seed_mixed;
        std::string text;
        text.reserve(c.nbytes + seed.size());
        while (text.size() < c.nbytes) {
            text += seed;
        }
        text.resize(c.nbytes);

        if (do_identity) {
            unsetenv("LLAMA_BPE_FORCE_LEGACY");
            unsetenv("LLAMA_BPE_FORCE_LEGACY_SPECIALS");
            auto fast = tokenize_all(vocab, text);
            setenv("LLAMA_BPE_FORCE_LEGACY", "1", 1);
            auto legacy = tokenize_all(vocab, text);
            unsetenv("LLAMA_BPE_FORCE_LEGACY");
            if (fast != legacy) {
                fprintf(stderr, "IDENTITY FAIL %s: fast=%zu legacy=%zu\n", c.name, fast.size(), legacy.size());
                size_t n = std::min(fast.size(), legacy.size());
                for (size_t i = 0; i < n; ++i) {
                    if (fast[i] != legacy[i]) {
                        fprintf(stderr, "  first diff at %zu: fast=%d legacy=%d\n", i, fast[i], legacy[i]);
                        break;
                    }
                }
                failures++;
            } else {
                printf("IDENTITY OK  %-12s tokens=%zu\n", c.name, fast.size());
            }
            // Blob vs vector materialize (0119/0121) — ASCII + mixed.
            if (c.seed_kind == 0 || c.seed_kind == 1) {
                unsetenv("LLAMA_BPE_NO_PRETOK_BLOB");
                auto with_blob = tokenize_all(vocab, text);
                setenv("LLAMA_BPE_NO_PRETOK_BLOB", "1", 1);
                auto no_blob = tokenize_all(vocab, text);
                unsetenv("LLAMA_BPE_NO_PRETOK_BLOB");
                if (with_blob != no_blob) {
                    fprintf(stderr, "IDENTITY FAIL pretok-blob %s: blob=%zu no_blob=%zu\n", c.name, with_blob.size(),
                            no_blob.size());
                    failures++;
                } else {
                    printf("IDENTITY OK  %-12s pretok-blob\n", c.name);
                }
            }
            // Specials path A/B (0111–0113) — chat / mixed seeds stress partition.
            if (c.seed_kind == 0 || c.seed_kind == 2) {
                unsetenv("LLAMA_BPE_FORCE_LEGACY_SPECIALS");
                auto fast_sp = tokenize_all(vocab, text);
                setenv("LLAMA_BPE_FORCE_LEGACY_SPECIALS", "1", 1);
                auto legacy_sp = tokenize_all(vocab, text);
                unsetenv("LLAMA_BPE_FORCE_LEGACY_SPECIALS");
                if (fast_sp != legacy_sp) {
                    fprintf(stderr, "IDENTITY FAIL specials %s: fast=%zu legacy=%zu\n", c.name, fast_sp.size(),
                            legacy_sp.size());
                    failures++;
                } else {
                    printf("IDENTITY OK  %-12s specials\n", c.name);
                }
            }
            // ASCII islands (0122) vs full Unicode pretok — mixed seeds only.
            if (c.seed_kind == 0) {
                unsetenv("LLAMA_BPE_NO_ASCII_PRETOK");
                auto with_islands = tokenize_all(vocab, text);
                setenv("LLAMA_BPE_NO_ASCII_PRETOK", "1", 1);
                auto no_ascii = tokenize_all(vocab, text);
                unsetenv("LLAMA_BPE_NO_ASCII_PRETOK");
                if (with_islands != no_ascii) {
                    fprintf(stderr, "IDENTITY FAIL ascii-islands %s: islands=%zu no_ascii=%zu\n", c.name,
                            with_islands.size(), no_ascii.size());
                    failures++;
                } else {
                    printf("IDENTITY OK  %-12s ascii-islands\n", c.name);
                }
            }
            // Byte-mixed islands (0124) vs cpt-mixed path — mixed seeds only.
            if (c.seed_kind == 0) {
                unsetenv("LLAMA_BPE_NO_BYTE_MIXED");
                auto with_byte = tokenize_all(vocab, text);
                setenv("LLAMA_BPE_NO_BYTE_MIXED", "1", 1);
                auto no_byte = tokenize_all(vocab, text);
                unsetenv("LLAMA_BPE_NO_BYTE_MIXED");
                if (with_byte != no_byte) {
                    fprintf(stderr, "IDENTITY FAIL byte-mixed %s: byte=%zu cpt=%zu\n", c.name, with_byte.size(),
                            no_byte.size());
                    failures++;
                } else {
                    printf("IDENTITY OK  %-12s byte-mixed\n", c.name);
                }
            }
            // SWAR/NEON letter+digit consume (0126) vs 8-wide LUT — ASCII + mixed + chat.
            {
                unsetenv("LLAMA_BPE_NO_SIMD_PRETOK");
                auto with_simd = tokenize_all(vocab, text);
                setenv("LLAMA_BPE_NO_SIMD_PRETOK", "1", 1);
                auto no_simd = tokenize_all(vocab, text);
                unsetenv("LLAMA_BPE_NO_SIMD_PRETOK");
                if (with_simd != no_simd) {
                    fprintf(stderr, "IDENTITY FAIL simd-pretok %s: simd=%zu lut=%zu\n", c.name, with_simd.size(),
                            no_simd.size());
                    failures++;
                } else if (c.nbytes >= (1u << 20)) {
                    printf("IDENTITY OK  %-12s simd-pretok\n", c.name);
                }
            }
        }

        if (do_bench) {
            unsetenv("LLAMA_BPE_FORCE_LEGACY");
            auto warm = tokenize_all(vocab, text);
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < c.iters; ++i) {
                warm = tokenize_all(vocab, text);
            }
            double per = ms_since(t0) / c.iters;
            double mib_s = (c.nbytes / (1024.0 * 1024.0)) / (per / 1000.0);
            printf("BENCH fast   %-12s tokens=%7zu  ms=%8.3f  thruput=%.2f MiB/s\n", c.name, warm.size(), per, mib_s);

            setenv("LLAMA_BPE_FORCE_LEGACY", "1", 1);
            warm = tokenize_all(vocab, text);
            t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < c.iters; ++i) {
                warm = tokenize_all(vocab, text);
            }
            per = ms_since(t0) / c.iters;
            mib_s = (c.nbytes / (1024.0 * 1024.0)) / (per / 1000.0);
            unsetenv("LLAMA_BPE_FORCE_LEGACY");
            printf("BENCH legacy %-12s tokens=%7zu  ms=%8.3f  thruput=%.2f MiB/s\n", c.name, warm.size(), per, mib_s);
        }
    }

    {
        std::vector<std::string> snippets = {
            "",
            "\n",
            "\n\n\n",
            " ",
            "Hello",
            " Hello World!",
            "🚀",
            "нещо на Български",
            "a\nb\nc",
            std::string(100, '\n'),
            std::string(1000, 'x'),
            "你好世界",
            "<|im_start|>",
            "café",
            "hello世界",
            "hello ·世界",
            " café 🚀\n",
            "Qwen测试",
        };
        for (size_t si = 0; si < snippets.size(); ++si) {
            const auto & text = snippets[si];
            unsetenv("LLAMA_BPE_FORCE_LEGACY");
            auto fast = tokenize_all(vocab, text);
            setenv("LLAMA_BPE_FORCE_LEGACY", "1", 1);
            auto legacy = tokenize_all(vocab, text);
            unsetenv("LLAMA_BPE_FORCE_LEGACY");
            if (fast != legacy) {
                fprintf(stderr, "IDENTITY FAIL snippet[%zu] len=%zu fast=%zu legacy=%zu\n", si, text.size(), fast.size(),
                        legacy.size());
                failures++;
            }
        }
        if (failures == 0) {
            printf("IDENTITY OK  snippets (%zu cases)\n", snippets.size());
        }
    }

    llama_model_free(model);
    llama_backend_free();
    return failures ? 3 : 0;
}
