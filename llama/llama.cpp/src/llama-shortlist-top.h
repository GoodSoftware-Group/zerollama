#pragma once

// How many output-head rows decide a sample. Zero means the whole row.
//
// A top-k that runs before every sampler that looks at the tail is enough:
// top-p, min-p, typical and XTC then only see those rows. Penalties, DRY,
// a grammar, a bias, or a filter that runs first need every logit.

#include <cstdint>

enum llama_shortlist_step : int32_t {
    LLAMA_SHORTLIST_STEP_PENALTIES   = 1,
    LLAMA_SHORTLIST_STEP_DRY         = 2,
    LLAMA_SHORTLIST_STEP_TOP_N_SIGMA = 3,
    LLAMA_SHORTLIST_STEP_TOP_K       = 4,
    LLAMA_SHORTLIST_STEP_TYPICAL     = 5,
    LLAMA_SHORTLIST_STEP_TOP_P       = 6,
    LLAMA_SHORTLIST_STEP_MIN_P       = 7,
    LLAMA_SHORTLIST_STEP_XTC         = 8,
    LLAMA_SHORTLIST_STEP_TEMPERATURE = 9,
    LLAMA_SHORTLIST_STEP_OTHER       = 10,
};

struct llama_shortlist_sample {
    float   temp              = 0.0f;
    int32_t top_k             = 0;
    float   top_p             = 1.0f;
    float   min_p             = 0.0f;
    float   typ_p             = 1.0f;
    float   xtc_probability   = 0.0f;
    float   top_n_sigma       = -1.0f;
    float   dynatemp_range    = 0.0f;
    int32_t penalty_last_n    = 0;
    float   penalty_repeat    = 1.0f;
    float   penalty_freq      = 0.0f;
    float   penalty_present   = 0.0f;
    float   dry_multiplier    = 0.0f;
    float   dry_base          = 1.75f;
    int32_t dry_penalty_last_n = 0;
    int32_t mirostat          = 0;
    int32_t n_probs           = 0;
    int32_t reasoning_budget  = -1;
    float   adaptive_target   = -1.0f;
    bool    grammar           = false;
    bool    logit_bias        = false;
    bool    preserved         = false;
    bool    backend_sampling  = false;
    bool    ignore_eos        = false;
};

inline bool llama_shortlist_penalties_off(const llama_shortlist_sample & s) {
    return s.penalty_last_n == 0 ||
        (s.penalty_repeat == 1.0f && s.penalty_freq == 0.0f && s.penalty_present == 0.0f);
}

inline bool llama_shortlist_dry_off(const llama_shortlist_sample & s) {
    return s.dry_multiplier == 0.0f || s.dry_base < 1.0f || s.dry_penalty_last_n == 0;
}

// Steps are the sampler chain, in order. An empty chain is greedy when
// temperature is not positive, and the whole row otherwise.
inline int32_t llama_shortlist_sample_k(const llama_shortlist_sample & s, const int32_t * steps, int32_t n_steps) {
    if (s.n_probs != 0 || s.mirostat != 0 || s.reasoning_budget >= 0 || s.adaptive_target >= 0.0f) {
        return 0;
    }
    if (s.grammar || s.logit_bias || s.preserved || s.backend_sampling || s.ignore_eos) {
        return 0;
    }
    bool saw_top_k = false;
    bool top_k_on  = false;
    for (int32_t i = 0; i < n_steps; ++i) {
        switch (steps[i]) {
            case LLAMA_SHORTLIST_STEP_PENALTIES:
                if (!llama_shortlist_penalties_off(s)) {
                    return 0;
                }
                break;
            case LLAMA_SHORTLIST_STEP_DRY:
                if (!llama_shortlist_dry_off(s)) {
                    return 0;
                }
                break;
            case LLAMA_SHORTLIST_STEP_TOP_N_SIGMA:
                if (s.top_n_sigma > 0.0f) {
                    return 0;
                }
                break;
            case LLAMA_SHORTLIST_STEP_TOP_K:
                saw_top_k = true;
                top_k_on  = s.top_k > 0;
                break;
            case LLAMA_SHORTLIST_STEP_TOP_P:
                if (!saw_top_k && s.top_p < 1.0f) {
                    return 0;
                }
                break;
            case LLAMA_SHORTLIST_STEP_MIN_P:
                if (!saw_top_k && s.min_p > 0.0f) {
                    return 0;
                }
                break;
            case LLAMA_SHORTLIST_STEP_TYPICAL:
                if (!saw_top_k && s.typ_p < 1.0f) {
                    return 0;
                }
                break;
            case LLAMA_SHORTLIST_STEP_XTC:
                if (!saw_top_k && s.xtc_probability > 0.0f) {
                    return 0;
                }
                break;
            case LLAMA_SHORTLIST_STEP_TEMPERATURE:
                if (!saw_top_k && s.dynatemp_range != 0.0f) {
                    return 0;
                }
                break;
            default:
                return 0;
        }
    }
    if (s.temp <= 0.0f) {
        return 1;
    }
    if (top_k_on) {
        return s.top_k;
    }
    return 0;
}

#ifdef LLAMA_SHORTLIST_TOP_TEST
#include <cstdio>
int main() {
    const int32_t chain[] = {
        LLAMA_SHORTLIST_STEP_PENALTIES,
        LLAMA_SHORTLIST_STEP_DRY,
        LLAMA_SHORTLIST_STEP_TOP_N_SIGMA,
        LLAMA_SHORTLIST_STEP_TOP_K,
        LLAMA_SHORTLIST_STEP_TYPICAL,
        LLAMA_SHORTLIST_STEP_TOP_P,
        LLAMA_SHORTLIST_STEP_MIN_P,
        LLAMA_SHORTLIST_STEP_XTC,
        LLAMA_SHORTLIST_STEP_TEMPERATURE,
    };
    llama_shortlist_sample s;
    s.temp = 0.8f;
    s.top_k = 40;
    s.top_p = 0.95f;
    s.min_p = 0.05f;
    s.penalty_last_n = 64;
    s.dry_base = 1.75f;
    s.dry_penalty_last_n = 64;
    if (llama_shortlist_sample_k(s, chain, 9) != 40) {
        std::fputs("default sampling is the top 40 rows\n", stderr);
        return 1;
    }
    s.temp = 0.0f;
    if (llama_shortlist_sample_k(s, chain, 9) != 1) {
        std::fputs("temperature zero is the one best row\n", stderr);
        return 1;
    }
    s.temp = 0.8f;
    s.dry_multiplier = 0.8f;
    if (llama_shortlist_sample_k(s, chain, 9) != 0) {
        std::fputs("a penalty needs the whole row\n", stderr);
        return 1;
    }
    s.dry_multiplier = 0.0f;
    s.top_k = 0;
    if (llama_shortlist_sample_k(s, chain, 9) != 0) {
        std::fputs("top-p over the whole vocabulary needs every row\n", stderr);
        return 1;
    }
    s.top_k = 40;
    const int32_t early_p[] = { LLAMA_SHORTLIST_STEP_TOP_P, LLAMA_SHORTLIST_STEP_TOP_K };
    if (llama_shortlist_sample_k(s, early_p, 2) != 0) {
        std::fputs("top-p before top-k needs the whole row\n", stderr);
        return 1;
    }
    s.grammar = true;
    const int32_t just_k[] = { LLAMA_SHORTLIST_STEP_TOP_K };
    if (llama_shortlist_sample_k(s, just_k, 1) != 0) {
        std::fputs("a grammar needs the whole row\n", stderr);
        return 1;
    }
    std::puts("shortlist top ok");
    return 0;
}
#endif
