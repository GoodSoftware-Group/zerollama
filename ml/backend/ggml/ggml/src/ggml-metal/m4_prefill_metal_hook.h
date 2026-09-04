#pragma once

#include "ggml-metal-device.h"
#include "ggml-metal-ops.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opt-in Metal fused Gate+Up Q4_0 SwiGLU (ZEROLLAMA_M4_PREFILL_SWIGLU=1).
// Returns >0 encode-skip count through the GLU node (down stays stock Metal).
// Returns 0 → fall through to ANE hook / stock mul_mat.
int m4_prefill_metal_op_mul_mat_try(
    ggml_metal_op_t ctx,
    ggml_metal_library_t lib,
    ggml_metal_encoder_t enc,
    int idx,
    struct ggml_tensor * op,
    int ic,
    int oc,
    int seq);

#ifdef __cplusplus
}
#endif
