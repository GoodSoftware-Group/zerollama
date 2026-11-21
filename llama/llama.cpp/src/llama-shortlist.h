#pragma once

#include "llama.h"

struct ggml_context;
struct ggml_tensor;
struct llama_model;
struct llama_shortlist;

void llama_shortlist_free(llama_shortlist * s);

// k > 0 keeps a set that contains the k highest logits. k <= 0 writes the full head.
void llama_shortlist_set_k(llama_shortlist * s, int32_t k);

// Output-head matmul for a host Q6_K / Q8_0 weight, or nullptr when this
// product must stay ggml_mul_mat (GPU weight, LoRA, scale, bias, or a batch).
struct ggml_tensor * llama_shortlist_mul(
        struct ggml_context * ctx,
        const struct llama_model & model,
        struct ggml_tensor * w,
        struct ggml_tensor * cur);
