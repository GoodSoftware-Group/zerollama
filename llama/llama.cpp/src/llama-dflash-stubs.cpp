// Temporary stubs for ANE draft-hook dflash activation getters / export APIs.
// Full buffer-backed implementations belong on the vendor/WIP context path.
// These keep CGO linkable for lab MTP / non-dflash builds (return nullptr / 0).
#include "llama-ext.h"
#include "llama.h"
#include "llama-model.h"

#include <cstdint>

void llama_set_dflash_target_export(struct llama_context *, const struct llama_model *) {}

struct ggml_tensor * llama_get_dflash_target_features(struct llama_context *) { return nullptr; }
float * llama_get_dflash_target_features_ith(struct llama_context *, int32_t) { return nullptr; }

float * llama_get_dflash_attn_out_ith(struct llama_context *, int32_t) { return nullptr; }
float * llama_get_dflash_tok_embd_ith(struct llama_context *, int32_t) { return nullptr; }
float * llama_get_dflash_attn_norm_ith(struct llama_context *, int32_t) { return nullptr; }
float * llama_get_dflash_q_mm_ith(struct llama_context *, int32_t) { return nullptr; }
float * llama_get_dflash_q_pre_rope_ith(struct llama_context *, int32_t) { return nullptr; }
float * llama_get_dflash_q_rope_ith(struct llama_context *, int32_t) { return nullptr; }
float * llama_get_dflash_k_noise_ith(struct llama_context *, int32_t) { return nullptr; }
float * llama_get_dflash_v_noise_ith(struct llama_context *, int32_t) { return nullptr; }
float * llama_get_dflash_k_cat_ith(struct llama_context *, int32_t, int32_t) { return nullptr; }
float * llama_get_dflash_v_cat_ith(struct llama_context *, int32_t, int32_t) { return nullptr; }
int32_t llama_get_dflash_kv_cat_n(struct llama_context *) { return 0; }
float * llama_get_dflash_fused_target_ith(struct llama_context *, int32_t, int32_t) { return nullptr; }
int32_t llama_get_dflash_fused_n(struct llama_context *) { return 0; }
float * llama_get_dflash_k_ctx_ith(struct llama_context *, int32_t, int32_t) { return nullptr; }
float * llama_get_dflash_k_ctx_pre_ith(struct llama_context *, int32_t, int32_t) { return nullptr; }
int32_t llama_get_dflash_k_ctx_n(struct llama_context *) { return 0; }
float * llama_get_dflash_layer_hidden_ith(struct llama_context *, int32_t, int32_t) { return nullptr; }

int32_t llama_model_layer_has_swa(const struct llama_model * model, int32_t il) {
    if (!model || il < 0) {
        return 0;
    }
    return model->hparams.is_swa((uint32_t) il) ? 1 : 0;
}

// llama_model_dflash_* metadata getters live in llama-model.cpp (dflash-draft arch).

int32_t llama_context_cross_n_enc(const struct llama_context *) { return 0; }
