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

// ============================================================================
// 5. FUSED GATE + UP Q4_0 GEMM WITH IN-KERNEL SWIGLU EPILOGUE
// ============================================================================
// Computes SiLU(A @ W_gate) * (A @ W_up) directly into Out [M, N_mlp]
// Completely eliminates writing Gate and Up intermediate matrices to DRAM!
kernel void fused_gate_up_swiglu_q4_0(
    device const half*         A      [[buffer(0)]], // [M, K]
    device const block_q4_0*   B_gate [[buffer(1)]], // [N_mlp, K/32]
    device const block_q4_0*   B_up   [[buffer(2)]], // [N_mlp, K/32]
    device half*               Out    [[buffer(3)]], // [M, N_mlp]
    constant uint&             M      [[buffer(4)]],
    constant uint&             N_mlp  [[buffer(5)]],
    constant uint&             K      [[buffer(6)]],
    threadgroup half*          shmem  [[threadgroup(0)]], // [2][32][32] = 4KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]])
{
    uint tg_row_start = tg_id.y * 32;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N_mlp) return;

    uint col_idx = tg_col_start + simd_lane_id;
    bool valid_col = (col_idx < N_mlp);
    uint num_k_blocks = K / 32;

    threadgroup half (*sh_A)[32][32] = (threadgroup half (*)[32][32])shmem;

    float acc_g[32];
    float acc_u[32];
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        acc_g[r] = 0.0f;
        acc_u[r] = 0.0f;
    }

    auto load_A = [&](uint buf_idx, uint kb) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint idx = simd_lane_id * 4 + i;
            uint r = idx / 4;
            uint c = (idx % 4) * 8;
            uint global_r = tg_row_start + r;
            uint global_c = kb * 32 + c;
            float4 val = float4(0.0f);
            if (global_r < M && global_c < K) {
                val = *reinterpret_cast<device const float4*>(&A[global_r * K + global_c]);
            }
            *reinterpret_cast<threadgroup float4*>(&sh_A[buf_idx][r][c]) = val;
        }
    };

    load_A(0, 0);
    block_q4_0 qg_curr, qu_curr;
    if (valid_col) {
        qg_curr = B_gate[col_idx * num_k_blocks + 0];
        qu_curr = B_up[col_idx * num_k_blocks + 0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cur_buf = 0;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint nxt_buf = cur_buf ^ 1;
        uint next_kb = kb + 1;
        block_q4_0 qg_next, qu_next;

        if (next_kb < num_k_blocks) {
            load_A(nxt_buf, next_kb);
            if (valid_col) {
                qg_next = B_gate[col_idx * num_k_blocks + next_kb];
                qu_next = B_up[col_idx * num_k_blocks + next_kb];
            }
        }

        if (valid_col) {
            // Unpack Gate weights
            half dg = qg_curr.d;
            half4 hdg = half4(dg);
            half4 h_off_g = half4(-8.0h * dg);
            uint gw0 = read_u32_unaligned(qg_curr.qs + 0);
            uint gw1 = read_u32_unaligned(qg_curr.qs + 4);
            uint gw2 = read_u32_unaligned(qg_curr.qs + 8);
            uint gw3 = read_u32_unaligned(qg_curr.qs + 12);

            half4 g_low[4], g_high[4];
            g_low[0]  = fma(half4(as_type<uchar4>(gw0 & 0x0F0F0F0Fu)), hdg, h_off_g);
            g_high[0] = fma(half4(as_type<uchar4>((gw0 >> 4) & 0x0F0F0F0Fu)), hdg, h_off_g);
            g_low[1]  = fma(half4(as_type<uchar4>(gw1 & 0x0F0F0F0Fu)), hdg, h_off_g);
            g_high[1] = fma(half4(as_type<uchar4>((gw1 >> 4) & 0x0F0F0F0Fu)), hdg, h_off_g);
            g_low[2]  = fma(half4(as_type<uchar4>(gw2 & 0x0F0F0F0Fu)), hdg, h_off_g);
            g_high[2] = fma(half4(as_type<uchar4>((gw2 >> 4) & 0x0F0F0F0Fu)), hdg, h_off_g);
            g_low[3]  = fma(half4(as_type<uchar4>(gw3 & 0x0F0F0F0Fu)), hdg, h_off_g);
            g_high[3] = fma(half4(as_type<uchar4>((gw3 >> 4) & 0x0F0F0F0Fu)), hdg, h_off_g);

            // Unpack Up weights
            half du = qu_curr.d;
            half4 hdu = half4(du);
            half4 h_off_u = half4(-8.0h * du);
            uint uw0 = read_u32_unaligned(qu_curr.qs + 0);
            uint uw1 = read_u32_unaligned(qu_curr.qs + 4);
            uint uw2 = read_u32_unaligned(qu_curr.qs + 8);
            uint uw3 = read_u32_unaligned(qu_curr.qs + 12);

            half4 u_low[4], u_high[4];
            u_low[0]  = fma(half4(as_type<uchar4>(uw0 & 0x0F0F0F0Fu)), hdu, h_off_u);
            u_high[0] = fma(half4(as_type<uchar4>((uw0 >> 4) & 0x0F0F0F0Fu)), hdu, h_off_u);
            u_low[1]  = fma(half4(as_type<uchar4>(uw1 & 0x0F0F0F0Fu)), hdu, h_off_u);
            u_high[1] = fma(half4(as_type<uchar4>((uw1 >> 4) & 0x0F0F0F0Fu)), hdu, h_off_u);
            u_low[2]  = fma(half4(as_type<uchar4>(uw2 & 0x0F0F0F0Fu)), hdu, h_off_u);
            u_high[2] = fma(half4(as_type<uchar4>((uw2 >> 4) & 0x0F0F0F0Fu)), hdu, h_off_u);
            u_low[3]  = fma(half4(as_type<uchar4>(uw3 & 0x0F0F0F0Fu)), hdu, h_off_u);
            u_high[3] = fma(half4(as_type<uchar4>((uw3 >> 4) & 0x0F0F0F0Fu)), hdu, h_off_u);

            #pragma unroll
            for (int r = 0; r < 32; r++) {
                half4 a0 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][0]);
                half4 a1 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][4]);
                half4 a2 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][8]);
                half4 a3 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][12]);
                half4 a4 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][16]);
                half4 a5 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][20]);
                half4 a6 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][24]);
                half4 a7 = *reinterpret_cast<threadgroup const half4*>(&sh_A[cur_buf][r][28]);

                // Gate dot product
                half4 pg0 = a0 * g_low[0] + a1 * g_low[1];
                half4 pg1 = a2 * g_low[2] + a3 * g_low[3];
                half4 pg2 = a4 * g_high[0] + a5 * g_high[1];
                half4 pg3 = a6 * g_high[2] + a7 * g_high[3];
                half4 sg = (pg0 + pg1) + (pg2 + pg3);
                acc_g[r] += (float)(sg[0] + sg[1] + sg[2] + sg[3]);

                // Up dot product
                half4 pu0 = a0 * u_low[0] + a1 * u_low[1];
                half4 pu1 = a2 * u_low[2] + a3 * u_low[3];
                half4 pu2 = a4 * u_high[0] + a5 * u_high[1];
                half4 pu3 = a6 * u_high[2] + a7 * u_high[3];
                half4 su = (pu0 + pu1) + (pu2 + pu3);
                acc_u[r] += (float)(su[0] + su[1] + su[2] + su[3]);
            }
        }

        if (next_kb < num_k_blocks) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            qg_curr = qg_next;
            qu_curr = qu_next;
            cur_buf = nxt_buf;
        }
    }

    if (valid_col) {
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            uint global_r = tg_row_start + r;
            if (global_r < M) {
                // SwiGLU Activation: S = SiLU(Gate) * Up
                float g = acc_g[r];
                float u = acc_u[r];
                float silu_g = g / (1.0f + exp(-g));
                float swiglu = silu_g * u;
                Out[global_r * N_mlp + col_idx] = (half)swiglu;
            }
        }
    }
}
