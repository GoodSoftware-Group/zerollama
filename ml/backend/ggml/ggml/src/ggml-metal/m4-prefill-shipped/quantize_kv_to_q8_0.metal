// Copyright 2026 Mohammed Hossam. Licensed under the Apache License, Version 2.0.
// Source: https://github.com/mohamedhossammohamed/m4-prefill-engine
// Staged for zerollama ggml-metal borrow (NOT wired into GGML_METAL_LIBS yet).
// Research PoC assumptions: head_dim=64, Q4_0 K%32==0, prefill-oriented tiles.
// Wire work must generalize ne/nb and reuse ggml-common block_q4_0 / block_q8_0.

#include <metal_stdlib>
using namespace metal;

struct block_q4_0 {
    half d;
    uint8_t qs[16];
};

struct block_q8_0 {
    half d;
    int8_t qs[32];
};

inline uint read_u32_unaligned(thread const uint8_t* p) {
    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

// Dynamic FP16 to Q8_0 KV Cache Quantization
kernel void quantize_kv_to_q8_0(
    device const half*       src [[buffer(0)]],
    device block_q8_0*       dst [[buffer(1)]],
    constant uint&           num_blocks [[buffer(2)]],
    uint blk_id [[thread_position_in_grid]])
{
    if (blk_id >= num_blocks) return;

    device const half* s = src + blk_id * 32;
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        float val = fabs((float)s[i]);
        if (val > amax) amax = val;
    }

    float d = amax / 127.0f;
    dst[blk_id].d = (half)d;
    float id_scale = (d > 0.0f) ? (1.0f / d) : 0.0f;

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int q = (int)round((float)s[i] * id_scale);
        q = clamp(q, -128, 127);
        dst[blk_id].qs[i] = (int8_t)q;
    }
}
