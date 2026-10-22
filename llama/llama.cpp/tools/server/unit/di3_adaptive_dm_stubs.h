// Minimal stubs so DI3 can unit-test server-adaptive-dm.h without linking llama.
#pragma once

#include <cstdint>
#include <string>

enum common_speculative_dm_controller {
    COMMON_SPECULATIVE_DM_CONTROLLER_OFF,
    COMMON_SPECULATIVE_DM_CONTROLLER_PROFIT,
};

struct common_params_speculative_draft {
    int32_t n_min = 0;
    float p_split = 0.0f;
    float p_min = 0.0f;
    int32_t backend_sampling = 0;
    int32_t cache_type_k = 0;
    int32_t cache_type_v = 0;
};

struct common_params_speculative {
    common_params_speculative_draft draft;
    common_speculative_dm_controller dm_controller = COMMON_SPECULATIVE_DM_CONTROLLER_PROFIT;
    float   dm_profit_min               = 0.05f;
    float   dm_profit_raise_margin      = 0.05f;
    float   dm_profit_lower_margin      = 0.05f;
    float   dm_profit_ewma_alpha        = 0.15f;
    int32_t dm_profit_min_samples       = 3;
    int32_t dm_profit_warmup            = 0;
    int32_t dm_profit_baseline_interval = 1024;
};

struct common_params_sampling {
    int32_t top_k = 0;
    float temp = 0.0f;
    float top_p = 1.0f;
    float min_p = 0.0f;
    bool grammar_lazy = false;
    std::string grammar;
};

inline const std::string & common_grammar_value(const std::string & g) {
    return g;
}
