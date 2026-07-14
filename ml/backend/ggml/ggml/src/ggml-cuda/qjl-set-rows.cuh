#pragma once

// Device-side QJL quantize for GGML_OP_SET_ROWS (KV cache writes).
#if defined(GGML_CUDA_QJL)

#include "common.cuh"
#include "ggml-common.h"

// Host uploads the default projection matrix before SET_ROWS kernels run.
extern __device__ float * ggml_cuda_qjl_prj_dev;

__device__ __forceinline__ uint16_t qjl_fp32_to_bf16_dev_setrows(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return (uint16_t) (u >> 16);
}

// Quantize one 128-float key row into block_qjl1_256 (matches qjl_quantize_row_ref).
__device__ inline void quantize_f32_qjl1_256_block(const float * x, block_qjl1_256 * y) {
    const float * prj = ggml_cuda_qjl_prj_dev;
    if (prj == nullptr) {
        return;
    }

    float norm_sq = 0.0f;
    for (int i = 0; i < 128; ++i) {
        const float v = x[i];
        norm_sq += v * v;
    }
    y->d = qjl_fp32_to_bf16_dev_setrows(sqrtf(norm_sq));

    for (int byte_i = 0; byte_i < (QK_QJL / 8); ++byte_i) {
        uint8_t packed = 0;
        for (int bit = 0; bit < 8; ++bit) {
            const int j = byte_i * 8 + bit;
            float acc = 0.0f;
            for (int i = 0; i < 128; ++i) {
                acc += x[i] * prj[i * QK_QJL + j];
            }
            if (acc > 0.0f) {
                packed |= (uint8_t) (1u << bit);
            }
        }
        y->signs[byte_i] = packed;
    }
}

#endif // GGML_CUDA_QJL
