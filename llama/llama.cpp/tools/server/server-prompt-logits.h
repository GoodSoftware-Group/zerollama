#pragma once

#include <cstdint>
#include <cstdlib>

// Logits after the last token of a prompt. An identical later prompt samples
// from this row instead of running that token again. A masked row (saved_k > 0)
// only covers a sampler that needs no more than those rows. saved_k < 0 is the
// whole vocabulary. LLAMA_PROMPT_LOGITS=0 keeps the one-token rewind.

inline bool llama_prompt_logits_on_env(const char * env) {
    return !(env && env[0] == '0');
}

inline bool llama_prompt_logits_on() {
    return llama_prompt_logits_on_env(std::getenv("LLAMA_PROMPT_LOGITS"));
}

// n_past is how much of this prompt is already in the cache.
inline bool llama_prompt_logits_match(int32_t n_past, int32_t n_prompt, int32_t n_saved) {
    return n_past > 0 && n_saved > 0 && n_past == n_prompt && n_saved == n_prompt;
}

// need_k == 0 means the sampler reads the whole row (penalties, grammar, bias).
inline bool llama_prompt_logits_cover(int32_t saved_k, int32_t need_k) {
    if (saved_k < 0) {
        return true;
    }
    return need_k > 0 && saved_k >= need_k;
}

inline uint64_t llama_prompt_logits_fnv_init() {
    return 14695981039346656037ull;
}

inline uint64_t llama_prompt_logits_mix(uint64_t h, int32_t token) {
    h ^= (uint32_t) token;
    h *= 1099511628211ull;
    return h;
}

inline uint64_t llama_prompt_logits_fnv_finish(uint64_t h) {
    return h == 0 ? 1 : h;
}

inline uint64_t llama_prompt_logits_fnv(const int32_t * tokens, int32_t n) {
    uint64_t h = llama_prompt_logits_fnv_init();
    for (int32_t i = 0; i < n; ++i) {
        h = llama_prompt_logits_mix(h, tokens[i]);
    }
    return llama_prompt_logits_fnv_finish(h);
}

#if defined(LLAMA_PROMPT_LOGITS_TEST)
#include <cstdio>

int main() {
    if (!llama_prompt_logits_match(4, 4, 4)) {
        return 1;
    }
    if (llama_prompt_logits_match(3, 4, 4) || llama_prompt_logits_match(4, 4, 0) || llama_prompt_logits_match(0, 0, 0)) {
        return 1;
    }
    if (!llama_prompt_logits_cover(-1, 0) || !llama_prompt_logits_cover(-1, 40)) {
        return 1;
    }
    if (!llama_prompt_logits_cover(40, 40) || !llama_prompt_logits_cover(40, 1)) {
        return 1;
    }
    if (llama_prompt_logits_cover(40, 41) || llama_prompt_logits_cover(40, 0) || llama_prompt_logits_cover(0, 40)) {
        return 1;
    }
    if (!llama_prompt_logits_on_env(nullptr) || !llama_prompt_logits_on_env("") || !llama_prompt_logits_on_env("1")) {
        return 1;
    }
    if (llama_prompt_logits_on_env("0") || llama_prompt_logits_on_env("0ff")) {
        return 1;
    }
    const int32_t a[] = { 1, 2, 3 };
    const int32_t b[] = { 1, 2, 4 };
    const int32_t c[] = { 3, 2, 1 };
    const uint64_t ha = llama_prompt_logits_fnv(a, 3);
    if (ha == 0 || ha != llama_prompt_logits_fnv(a, 3) || ha == llama_prompt_logits_fnv(b, 3) || ha == llama_prompt_logits_fnv(c, 3)) {
        return 1;
    }
    std::printf("prompt logits ok\n");
    return 0;
}
#endif
