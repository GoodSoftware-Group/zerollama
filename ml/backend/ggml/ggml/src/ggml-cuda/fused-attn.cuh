#pragma once

#include "common.cuh"

#if defined(GGML_CUDA_FUSED_ATTN_QJL)

#ifdef __cplusplus
extern "C" {
#endif

void fused_attn_qjl_tbq_cuda(
        const float * q_sketch_d,
        const void  * packed_k_d,
        const void  * packed_v_d,
        int n_heads, int n_kv_heads, int n_q_pos, int n_kv,
        float sm_scale, int kv_tile,
        float * out_d,
        cudaStream_t stream);

void fused_attn_qjl_polar_cuda(
        const float * q_sketch_d,
        const void  * packed_k_d,
        const void  * packed_v_d,
        int n_heads, int n_kv_heads, int n_q_pos, int n_kv,
        float sm_scale, int v_use_qjl, int kv_tile, int causal, int q_pos_base,
        float * out_d,
        cudaStream_t stream);

void qjl_score_dp4a_cuda(
        const int8_t * q_sketch_i8_d,
        const float  * q_scale_d,
        const void   * packed_k_d,
        int n_heads, int n_kv_heads, int n_kv_tokens,
        float * scores_d,
        cudaStream_t stream);

#ifdef __cplusplus
}
#endif

void ggml_cuda_op_fused_attn_qjl_tbq(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_attn_score_qjl(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#endif // GGML_CUDA_FUSED_ATTN_QJL
