#pragma once

#include "llama.h"

#include <string>
#include <vector>

// CPU post for LLM_ARCH_LAYA decision head (encoder embeddings → choice logits + act).
// WHY CPU: head is ~26MB; encoder already GPU via ggml. Avoids new Metal/CUDA kernels
// while keeping per-MASK states (RANK pooling cannot). See docs/laya-llama-cpp-findings.md.

struct laya_linear {
    std::vector<float> weight; // [ne0=in, ne1=out] column-major ggml layout
    std::vector<float> bias;   // empty if absent
    int64_t n_in  = 0;
    int64_t n_out = 0;
};

struct laya_ln {
    std::vector<float> weight;
    std::vector<float> bias; // empty if absent
};

struct laya_head_layer {
    laya_ln     attn_norm;
    laya_linear attn_qkv;
    laya_linear attn_out;
    laya_ln     ffn_norm;
    laya_linear ffn_up;
    laya_linear ffn_down;
};

struct laya_head_weights {
    int n_embd        = 0;
    int n_head        = 1;
    int n_head_layers = 0;
    int n_act         = 2;
    float eps         = 1e-5f;

    std::vector<float> type_embd; // [n_embd, 3]
    std::vector<laya_head_layer> layers;

    laya_ln     scorer_norm;
    laya_linear scorer_fc1;
    laya_linear scorer_fc2;

    laya_linear act_fc1;
    laya_linear act_fc2;
};

struct laya_decision_input {
    std::vector<llama_token> tokens;
    std::vector<int32_t>     marker_pos;
    int32_t                  qtype = 0;
    std::string              question_id; // echoed back in result; empty if not provided
};

struct laya_decision_result {
    std::vector<float> logits; // per-marker scorer logits
    std::vector<float> act;    // softmax over act head
    int                n_tokens = 0;
};

// True if model carries Laya decision-head tensors.
bool laya_model_has_head(const llama_model * model);

// Load (and dequantize) head weights from the model. Returns false on missing/invalid tensors.
bool laya_head_load(const llama_model * model, laya_head_weights & out, std::string & err);

// Encoder pass: embeddings mode, all-token outputs → embd_out[n_tokens * n_embd].
bool laya_encode_embeddings(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        std::vector<float> & embd_out,
        std::string & err);

// Apply decision head on per-token embeddings H[t].
bool laya_head_apply(
        const laya_head_weights & w,
        const float * H,
        int n_tokens,
        const laya_decision_input & input,
        laya_decision_result & out,
        std::string & err);

// Encode + head for one tokenized input.
bool laya_decide(
        llama_context * ctx,
        const laya_head_weights & w,
        const laya_decision_input & input,
        laya_decision_result & out,
        std::string & err);
