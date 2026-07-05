#pragma once

#include "ggml.h"

#include <cstdint>
#include <vector>

#define LLAMA_DFLASH_MAX_TARGET_LAYERS 8

struct llama_dflash_export_params {
    bool     enabled             = false;
    uint32_t n_target_features   = 0;
    uint32_t n_target_layers     = 0;
    uint32_t target_layer_ids[LLAMA_DFLASH_MAX_TARGET_LAYERS] = {};
};

static inline bool llama_dflash_export_enabled(const llama_dflash_export_params & exp) {
    return exp.enabled && exp.n_target_layers > 0;
}

static inline bool llama_dflash_export_layer(const llama_dflash_export_params & exp, int il) {
    if (!llama_dflash_export_enabled(exp)) {
        return false;
    }
    for (uint32_t i = 0; i < exp.n_target_layers; ++i) {
        if ((int) exp.target_layer_ids[i] == il) {
            return true;
        }
    }
    return false;
}

static inline ggml_tensor * llama_dflash_concat_layer_tensors(
        ggml_context * ctx0,
        const std::vector<ggml_tensor *> & ordered_layers) {
    if (ordered_layers.empty()) {
        return nullptr;
    }
    ggml_tensor * cat = ordered_layers[0];
    for (size_t i = 1; i < ordered_layers.size(); ++i) {
        cat = ggml_concat(ctx0, cat, ordered_layers[i], 0);
    }
    return cat;
}
