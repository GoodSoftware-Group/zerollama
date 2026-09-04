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

template<ushort TG_SIZE, ushort BC>
inline void load_kv_tile_fp16(
    device const half* K,
    device const half* V,
    threadgroup half* smem_K,
    threadgroup half* smem_V,
    uint h,
    uint c_start,
    uint M,
    uint tid)
{
    constexpr uint total_half4 = (BC * 64) / 4;
    #pragma unroll
    for (uint vec_idx = tid; vec_idx < total_half4; vec_idx += TG_SIZE) {
        uint row = vec_idx / 16;
        uint col_vec = vec_idx % 16;
        uint global_tok = c_start + row;
        
        threadgroup half4* k_dst = (threadgroup half4*)(smem_K + row * 64);
        threadgroup half4* v_dst = (threadgroup half4*)(smem_V + row * 64);
        
        if (global_tok < M) {
            device const half4* k_src = (device const half4*)(K + (h * M + global_tok) * 64);
            device const half4* v_src = (device const half4*)(V + (h * M + global_tok) * 64);
            k_dst[col_vec] = k_src[col_vec];
            v_dst[col_vec] = v_src[col_vec];
        } else {
            k_dst[col_vec] = half4(0.0h);
            v_dst[col_vec] = half4(0.0h);
        }
    }
}

template<ushort TG_SIZE, ushort BC>
inline void load_kv_tile_q8_0(
    device const block_q8_0* K_q8,
    device const block_q8_0* V_q8,
    threadgroup half* smem_K,
    threadgroup half* smem_V,
    uint h,
    uint c_start,
    uint M,
    uint tid)
{
    constexpr uint total_blocks = BC * 2;
    #pragma unroll
    for (uint blk_idx = tid; blk_idx < total_blocks; blk_idx += TG_SIZE) {
        uint row = blk_idx / 2;
        uint sub_blk = blk_idx % 2;
        uint global_tok = c_start + row;
        
        threadgroup half* k_dst = smem_K + row * 64 + sub_blk * 32;
        threadgroup half* v_dst = smem_V + row * 64 + sub_blk * 32;
        
        if (global_tok < M) {
            uint blk_offset = (h * M + global_tok) * 2 + sub_blk;
            block_q8_0 k_blk = K_q8[blk_offset];
            block_q8_0 v_blk = V_q8[blk_offset];
            
            half kd = k_blk.d;
            half vd = v_blk.d;
            
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                k_dst[i] = (half)k_blk.qs[i] * kd;
                v_dst[i] = (half)v_blk.qs[i] * vd;
            }
        } else {
            threadgroup half4* k_dst4 = (threadgroup half4*)(smem_K + row * 64 + sub_blk * 32);
            threadgroup half4* v_dst4 = (threadgroup half4*)(smem_V + row * 64 + sub_blk * 32);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                k_dst4[i] = half4(0.0h);
                v_dst4[i] = half4(0.0h);
            }
        }
    }
}


// FlashAttention Q8_0 (64x32 Tile - Peak Throughput with Q8_0 KV Cache)
kernel void flash_attn_q8_0_causal(
    device const half*       Q     [[buffer(0)]], // [H, M, 64]
    device const block_q8_0* K_q8  [[buffer(1)]], // [H, M, 2] blocks
    device const block_q8_0* V_q8  [[buffer(2)]], // [H, M, 2] blocks
    device half*             O     [[buffer(3)]], // [M, H * 64] output directly in [M, H*D] layout!
    constant uint&           M     [[buffer(4)]],
    constant uint&           H     [[buffer(5)]],
    constant float&          scale [[buffer(6)]],
    threadgroup half*        shmem [[threadgroup(0)]], // [2][32*64 + 32*64] = 16KB
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]])
{
    constexpr ushort BR = 64;
    constexpr ushort BC = 32;
    constexpr ushort TG_SIZE = 64;

    uint b_r = tg_pos.x;
    uint h   = tg_pos.y;
    
    uint r_in_tile = tid;
    uint row_idx = b_r * BR + r_in_tile;
    bool is_valid_row = (r_in_tile < BR) && (row_idx < M);
    
    half4 q_reg[16];
    if (is_valid_row) {
        device const half4* q_ptr = (device const half4*)(Q + (h * M + row_idx) * 64);
        #pragma unroll
        for (int d = 0; d < 16; d++) {
            q_reg[d] = q_ptr[d];
        }
    } else {
        #pragma unroll
        for (int d = 0; d < 16; d++) {
            q_reg[d] = half4(0.0h);
        }
    }
    
    float running_max = -1e30f;
    float running_sum = 0.0f;
    half4 o_acc[16];
    #pragma unroll
    for (int d = 0; d < 16; d++) {
        o_acc[d] = half4(0.0h);
    }
    
    uint num_key_tiles = (M + BC - 1) / BC;
    uint r_max = min((b_r + 1) * BR, M) - 1;
    uint max_causal_tile = r_max / BC;
    uint loop_tiles = min(max_causal_tile + 1, num_key_tiles);
    
    threadgroup half (*smem_K)[BC * 64] = (threadgroup half (*)[BC * 64])shmem;
    threadgroup half (*smem_V)[BC * 64] = (threadgroup half (*)[BC * 64])(shmem + 2 * BC * 64);

    uint cur_buf = 0;
    if (loop_tiles > 0) {
        load_kv_tile_q8_0<TG_SIZE, BC>(K_q8, V_q8, smem_K[0], smem_V[0], h, 0, M, tid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint b_c = 0; b_c < loop_tiles; b_c++) {
        uint nxt_buf = cur_buf ^ 1;
        uint next_b_c = b_c + 1;
        if (next_b_c < loop_tiles) {
            load_kv_tile_q8_0<TG_SIZE, BC>(K_q8, V_q8, smem_K[nxt_buf], smem_V[nxt_buf], h, next_b_c * BC, M, tid);
        }
        
        if (is_valid_row) {
            float s_tile[BC];
            float tile_local_max = -1e30f;
            
            #pragma unroll
            for (uint c = 0; c < BC; c++) {
                uint col_idx = b_c * BC + c;
                if (col_idx <= row_idx && col_idx < M) {
                    threadgroup const half4* k_ptr = (threadgroup const half4*)(smem_K[cur_buf] + c * 64);
                    half4 dot4 = half4(0.0h);
                    #pragma unroll
                    for (int d = 0; d < 16; d++) {
                        dot4 += q_reg[d] * k_ptr[d];
                    }
                    float dot = (float)(dot4[0] + dot4[1] + dot4[2] + dot4[3]) * scale;
                    s_tile[c] = dot;
                    if (dot > tile_local_max) tile_local_max = dot;
                } else {
                    s_tile[c] = -1e30f;
                }
            }
            
            float new_max = max(running_max, tile_local_max);
            float alpha = exp(running_max - new_max);
            running_max = new_max;
            running_sum = running_sum * alpha;
            
            #pragma unroll
            for (int d = 0; d < 16; d++) {
                o_acc[d] = o_acc[d] * (half)alpha;
            }
            
            float p_tile[BC];
            #pragma unroll
            for (uint c = 0; c < BC; c++) {
                if (s_tile[c] > -1e20f) {
                    float p = exp(s_tile[c] - running_max);
                    p_tile[c] = p;
                    running_sum += p;
                } else {
                    p_tile[c] = 0.0f;
                }
            }
            
            #pragma unroll
            for (uint c = 0; c < BC; c++) {
                if (p_tile[c] > 0.0f) {
                    half p_val = (half)p_tile[c];
                    threadgroup const half4* v_ptr = (threadgroup const half4*)(smem_V[cur_buf] + c * 64);
                    #pragma unroll
                    for (int d = 0; d < 16; d++) {
                        o_acc[d] = fma(v_ptr[d], half4(p_val), o_acc[d]);
                    }
                }
            }
        }
        
        if (next_b_c < loop_tiles) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            cur_buf = nxt_buf;
        }
    }
    
    if (is_valid_row) {
        half inv_sum = (running_sum > 0.0f) ? (half)(1.0f / running_sum) : 0.0h;
        device half4* o_out = (device half4*)(O + row_idx * (H * 64) + h * 64);
        #pragma unroll
        for (int d = 0; d < 16; d++) {
            o_out[d] = o_acc[d] * inv_sum;
        }
    }
}
