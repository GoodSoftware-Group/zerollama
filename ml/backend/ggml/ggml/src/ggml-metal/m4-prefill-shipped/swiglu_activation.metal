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

// High-Throughput Standalone SwiGLU Activation: S = SiLU(Gate) * Up
kernel void swiglu_activation(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    constant uint&     num_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint base_idx = id * 4;
    if (base_idx < num_elements) {
        device const half4* g_ptr = reinterpret_cast<device const half4*>(gate + base_idx);
        device const half4* u_ptr = reinterpret_cast<device const half4*>(up + base_idx);
        device half4* o_ptr = reinterpret_cast<device half4*>(out + base_idx);

        half4 g_h = *g_ptr;
        half4 u_h = *u_ptr;

        float4 g_f = float4(g_h);
        float4 u_f = float4(u_h);

        float4 silu_g = g_f / (float4(1.0f) + exp(-g_f));
        float4 res = silu_g * u_f;

        *o_ptr = half4(res);
    }
}
