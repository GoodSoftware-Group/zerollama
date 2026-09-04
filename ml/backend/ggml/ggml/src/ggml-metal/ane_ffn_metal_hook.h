#pragma once

#include "ggml-metal-ops.h"

#ifdef __cplusplus
extern "C" {
#endif

// Lab ANE FFN intercept for Metal mul_mat. Returns >0 to skip that many encode
// nodes (same convention as ggml_metal_op_mul_mat). Returns 0 → fall through.
// Lives in its own TU: inlining this into ggml-metal-ops.cpp regressed MoE
// quality on b10488+ even when ZEROLLAMA_ANE_FFN was off.
int ane_ffn_metal_op_mul_mat_try(
    ggml_metal_op_t ctx,
    int idx,
    struct ggml_tensor * op,
    int ic, int oc, int seq);

#ifdef __cplusplus
}
#endif
