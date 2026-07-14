#pragma once

// Device-side PolarQuant Q4 encode for GGML_OP_SET_ROWS (V-cache writes).
#if defined(GGML_CUDA_POLARQUANT)

#include "common.cuh"
#include "ggml-common.h"

__device__ __forceinline__ float polar_qjl_sign_setrows(int i) {
    uint32_t state = 42u;
    if (state == 0u) {
        state = 1u;
    }
    float bit = -1.0f;
    for (int k = 0; k <= i; ++k) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        bit = (state & 1u) ? 1.0f : -1.0f;
    }
    return bit;
}

__device__ __forceinline__ void polar_hadamard_inplace_setrows(float * x) {
    for (int h = 1; h < QK_POLAR; h <<= 1) {
        for (int i = 0; i < QK_POLAR; i += (h << 1)) {
            for (int j = i; j < i + h; ++j) {
                const float a = x[j];
                const float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
}

__device__ __forceinline__ uint8_t polar_q4_bucketize_setrows(float v) {
    const float boundaries[15] = {
        -2.423958757e+00f, -1.868302108e+00f, -1.461390631e+00f, -1.121190365e+00f,
        -8.175165476e-01f, -5.351446099e-01f, -2.648274426e-01f,  4.996003611e-16f,
         2.648274426e-01f,  5.351446099e-01f,  8.175165476e-01f,  1.121190365e+00f,
         1.461390631e+00f,  1.868302108e+00f,  2.423958757e+00f,
    };
    uint8_t code = 0;
    for (int i = 0; i < 15; ++i) {
        if (v > boundaries[i]) {
            code = (uint8_t) (i + 1);
        }
    }
    return code;
}

__device__ inline void quantize_f32_q4_polar_block(const float * x, block_q4_polar * y) {
    const float centroids[16] = {
        -2.754354807e+00f, -2.093562707e+00f, -1.643041510e+00f, -1.279739752e+00f,
        -9.626409783e-01f, -6.723921169e-01f, -3.978971029e-01f, -1.317577823e-01f,
         1.317577823e-01f,  3.978971029e-01f,  6.723921169e-01f,  9.626409783e-01f,
         1.279739752e+00f,  1.643041510e+00f,  2.093562707e+00f,  2.754354807e+00f,
    };

    float norm_sq = 0.0f;
    for (int i = 0; i < QK_POLAR; ++i) {
        const float v = x[i];
        norm_sq += v * v;
    }
    const float l2     = sqrtf(norm_sq);
    const float inv_l2 = (l2 > 1e-10f) ? (1.0f / l2) : 0.0f;
    y->d = __float2half(l2);

    float buf[QK_POLAR];
    for (int i = 0; i < QK_POLAR; ++i) {
        buf[i] = x[i] * inv_l2;
    }
    polar_hadamard_inplace_setrows(buf);

    uint8_t codes[QK_POLAR];
    for (int i = 0; i < QK_POLAR; ++i) {
        codes[i] = polar_q4_bucketize_setrows(buf[i]);
    }
    for (int i = 0; i < QK_POLAR / 2; ++i) {
        const uint8_t lo = codes[2 * i];
        const uint8_t hi = codes[2 * i + 1];
        y->qs[i] = (uint8_t) ((hi << 4) | (lo & 0x0Fu));
    }

    for (int i = 0; i < QJL_RESIDUAL_BYTES; ++i) {
        y->qjl[i] = 0;
    }
    float proj = 0.0f;
    for (int i = 0; i < QK_POLAR; ++i) {
        const float c = centroids[codes[i]];
        proj += (buf[i] - c) * polar_qjl_sign_setrows(i);
    }
    y->qjl[0] = (proj >= 0.0f) ? 1u : 0u;
}

#endif // GGML_CUDA_POLARQUANT
