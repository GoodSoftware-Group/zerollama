#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
//

//
// cross-attention encoder cache (dflash target_hidden window; T5 encoder path)
//

// True when cross.v_embd has at least one row (n_embd × n_enc).
LLAMA_API bool llama_context_cross_has_v_embd(const struct llama_context * ctx);

// Copy row `pos` from cross.v_embd into dst (at most dst_cap floats). Returns copied length.
LLAMA_API int32_t llama_context_cross_row(
        const struct llama_context * ctx,
        int32_t pos,
        float * dst,
        int32_t dst_cap);

// Upsert one cross row (grows n_enc as needed). Used by ANE dflash_fc handoff (B8 lab).
LLAMA_API void llama_context_cross_upsert_row(
        struct llama_context * ctx,
        int32_t pos,
        const float * src,
        int32_t n_feat);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// Number of staged cross rows (0 when cross.v_embd empty).
LLAMA_API int32_t llama_context_cross_n_enc(const struct llama_context * ctx);

//
// pre-norm embeddings (hidden state before the final output norm)
// implemented as thin aliases over the embeddings_nextn machinery (unmasked mode)
//

// mirrors:
// LLAMA_API void llama_set_embeddings(struct llama_context * ctx, bool embeddings);
LLAMA_API void llama_set_embeddings_pre_norm(struct llama_context * ctx, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_pre_norm(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_pre_norm_ith(struct llama_context * ctx, int32_t i);

//
// dflash-draft model metadata (B8 target_hidden / cross.v_embd wiring)
//

// Returns dflash.n_target_features for dflash-draft models, else 0.
LLAMA_API int32_t llama_model_dflash_n_target_features(const struct llama_model * model);

// Returns dflash.mask_token_id for dflash-draft models, else -1.
LLAMA_API int32_t llama_model_dflash_mask_token_id(const struct llama_model * model);

// Returns dflash.block_size for dflash-draft models, else 0.
LLAMA_API int32_t llama_model_dflash_block_size(const struct llama_model * model);

// Returns count of dflash.target_layer_ids[], else 0.
LLAMA_API int32_t llama_model_dflash_n_target_layers(const struct llama_model * model);

// Returns dflash.target_layer_ids[i] or -1 when out of range / not dflash-draft.
LLAMA_API int32_t llama_model_dflash_target_layer_id(const struct llama_model * model, int32_t i);

// Enable per-layer target hidden export on ctx_tgt using draft-model dflash metadata.
LLAMA_API void llama_set_dflash_target_export(struct llama_context * ctx_tgt, const struct llama_model * draft_model);

// Row from the last target decode ([n_target_features]); NULL when export inactive.
LLAMA_API float * llama_get_dflash_target_features_ith(struct llama_context * ctx, int32_t i);

// Full tensor view for ANE pack sizing; may be NULL when export inactive.
LLAMA_API struct ggml_tensor * llama_get_dflash_target_features(struct llama_context * ctx);

// Row from the last dflash-draft decode (kqv_out before attn_wo); NULL when inactive.
LLAMA_API float * llama_get_dflash_attn_out_ith(struct llama_context * ctx, int32_t i);

// Row from the last dflash-draft decode (tok_embd before attn_norm); NULL when inactive.
LLAMA_API float * llama_get_dflash_tok_embd_ith(struct llama_context * ctx, int32_t i);

// Row from the last dflash-draft decode (attn_norm on tok_embd); NULL when inactive.
LLAMA_API float * llama_get_dflash_attn_norm_ith(struct llama_context * ctx, int32_t i);

// Row from the last dflash-draft decode (Q after wq matmul); NULL when inactive.
LLAMA_API float * llama_get_dflash_q_mm_ith(struct llama_context * ctx, int32_t i);

// Row from the last dflash-draft decode (Q after attn_q_norm, before RoPE); NULL when inactive.
LLAMA_API float * llama_get_dflash_q_pre_rope_ith(struct llama_context * ctx, int32_t i);

// Row from the last dflash-draft decode (Q after RoPE); NULL when inactive.
LLAMA_API float * llama_get_dflash_q_rope_ith(struct llama_context * ctx, int32_t i);

// dflash-draft K/V noise after RoPE and concat ctx+noise K/V cat (requires embeddings_pre_norm).
LLAMA_API float * llama_get_dflash_k_noise_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_dflash_v_noise_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_dflash_k_cat_ith(struct llama_context * ctx, int32_t i, int32_t slot);
LLAMA_API float * llama_get_dflash_v_cat_ith(struct llama_context * ctx, int32_t i, int32_t slot);
LLAMA_API int32_t llama_get_dflash_kv_cat_n(struct llama_context * ctx);
LLAMA_API float * llama_get_dflash_fused_target_ith(struct llama_context * ctx, int32_t i, int32_t slot);
LLAMA_API int32_t llama_get_dflash_fused_n(struct llama_context * ctx);
LLAMA_API float * llama_get_dflash_k_ctx_ith(struct llama_context * ctx, int32_t i, int32_t slot);
LLAMA_API float * llama_get_dflash_k_ctx_pre_ith(struct llama_context * ctx, int32_t i, int32_t slot);
LLAMA_API int32_t llama_get_dflash_k_ctx_n(struct llama_context * ctx);

// Row from the last dflash-draft decode (hidden after blk.layer); NULL when inactive.
LLAMA_API float * llama_get_dflash_layer_hidden_ith(struct llama_context * ctx, int32_t layer, int32_t i);

// P51: re-pull dflash export tensors from the last graph into host output buffers.
LLAMA_API void llama_dflash_pull_graph_exports(struct llama_context * ctx);

// Returns 1 if layer `il` uses sliding-window attention, 0 otherwise (or on invalid input).
LLAMA_API int32_t llama_model_layer_has_swa(const struct llama_model * model, int32_t il);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);
