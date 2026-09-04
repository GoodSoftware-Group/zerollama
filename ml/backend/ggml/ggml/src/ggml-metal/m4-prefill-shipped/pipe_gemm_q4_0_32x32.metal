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
// 3. PIPELINED QUEUE-SATURATED DOUBLE-BUFFERED Q4_0 GEMM (32x32)
// ============================================================================
kernel void pipe_gemm_q4_0_32x32(
    device const half*         A [[buffer(0)]],
    device const block_q4_0*   B [[buffer(1)]],
    device half*               C [[buffer(2)]],
    constant uint&             M [[buffer(3)]],
    constant uint&             N [[buffer(4)]],
    constant uint&             K [[buffer(5)]],
    threadgroup half*          shmem [[threadgroup(0)]], // [2][32][32] = 4KB
    uint2 tg_id   [[threadgroup_position_in_grid]],
    uint  simd_lane_id [[thread_index_in_simdgroup]])
{
    uint tg_row_start = tg_id.y * 32;
    uint tg_col_start = tg_id.x * 32;

    if (tg_row_start >= M || tg_col_start >= N) return;

    uint col_idx = tg_col_start + simd_lane_id;
    bool valid_col = (col_idx < N);
    uint num_k_blocks = K / 32;

    threadgroup half (*sh_A)[32][32] = (threadgroup half (*)[32][32])shmem;

    float acc[32];
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        acc[r] = 0.0f;
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
    block_q4_0 q_curr;
    if (valid_col) {
        q_curr = B[col_idx * num_k_blocks + 0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cur_buf = 0;

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint nxt_buf = cur_buf ^ 1;
        uint next_kb = kb + 1;
        block_q4_0 q_next;

        if (next_kb < num_k_blocks) {
            load_A(nxt_buf, next_kb);
            if (valid_col) {
                q_next = B[col_idx * num_k_blocks + next_kb];
            }
        }

        if (valid_col) {
            half d = q_curr.d;
            half4 hd = half4(d);
            half4 h_off = half4(-8.0h * d);

            uint w0 = read_u32_unaligned(q_curr.qs + 0);
            uint w1 = read_u32_unaligned(q_curr.qs + 4);
            uint w2 = read_u32_unaligned(q_curr.qs + 8);
            uint w3 = read_u32_unaligned(q_curr.qs + 12);

            half4 v_low[4], v_high[4];
            v_low[0]  = fma(half4(as_type<uchar4>(w0 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[0] = fma(half4(as_type<uchar4>((w0 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[1]  = fma(half4(as_type<uchar4>(w1 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[1] = fma(half4(as_type<uchar4>((w1 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[2]  = fma(half4(as_type<uchar4>(w2 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[2] = fma(half4(as_type<uchar4>((w2 >> 4) & 0x0F0F0F0Fu)), hd, h_off);
            v_low[3]  = fma(half4(as_type<uchar4>(w3 & 0x0F0F0F0Fu)), hd, h_off);
            v_high[3] = fma(half4(as_type<uchar4>((w3 >> 4) & 0x0F0F0F0Fu)), hd, h_off);

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

                half4 p0 = a0 * v_low[0] + a1 * v_low[1];
                half4 p1 = a2 * v_low[2] + a3 * v_low[3];
                half4 p2 = a4 * v_high[0] + a5 * v_high[1];
                half4 p3 = a6 * v_high[2] + a7 * v_high[3];

                half4 s0 = p0 + p1;
                half4 s1 = p2 + p3;
                half4 s = s0 + s1;

                acc[r] += (float)(s[0] + s[1] + s[2] + s[3]);
            }
        }

        if (next_kb < num_k_blocks) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            q_curr = q_next;
            cur_buf = nxt_buf;
        }
    }

    if (valid_col) {
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            uint global_r = tg_row_start + r;
            if (global_r < M) {
                C[global_r * N + col_idx] = (half)acc[r];
            }
        }
    }
}
