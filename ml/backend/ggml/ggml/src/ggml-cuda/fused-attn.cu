#include "fused-attn.cuh"

#if defined(GGML_CUDA_FUSED_ATTN_QJL)

#include "qjl.cuh"
#include "ggml-cuda.h"

#include <cmath>

static const float * ggml_cuda_get_tensor_f32(const ggml_tensor * t, int dev) {
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    GGML_UNUSED(dev);
    return (const float *) t->data;
}

static float * ggml_cuda_get_tensor_f32_mut(ggml_tensor * t) {
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    return (float *) t->data;
}

void ggml_cuda_op_fused_attn_qjl_tbq(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q  = dst->src[0];
    const ggml_tensor * pk = dst->src[1];
    const ggml_tensor * pv = dst->src[2];

    GGML_ASSERT(q && pk && pv);
    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(pk->type == GGML_TYPE_QJL1_256);
    GGML_ASSERT(pv->type == GGML_TYPE_TBQ3_0 || pv->type == GGML_TYPE_Q4_POLAR);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_rows(q));
    GGML_ASSERT(ggml_is_contiguous_rows(pk));
    GGML_ASSERT(ggml_is_contiguous_rows(pv));
    GGML_ASSERT(ggml_is_contiguous_rows(dst));

    const int32_t * params = (const int32_t *) dst->op_params;
    const int n_kv_heads = params[0];
    union { int32_t i; float f; } scale_bits;
    scale_bits.i = params[1];
    const int v_use_qjl = params[2];
    const int kv_tile   = params[3];
    const int causal    = params[4];
    const int q_pos_base = params[5];

    const int n_heads = (int) q->ne[1];
    const int n_q_pos = (int) q->ne[2];
    const int n_kv    = (int) pk->ne[1];
    const int64_t ne3 = q->ne[3];

    GGML_ASSERT(n_kv_heads > 0);
    GGML_ASSERT((n_heads % n_kv_heads) == 0);
    GGML_ASSERT(pk->ne[0] == 128 && pv->ne[0] == 128 && dst->ne[0] == 128);
    GGML_ASSERT(pk->ne[2] == n_kv_heads && pv->ne[2] == n_kv_heads);
    GGML_ASSERT(pv->ne[1] == n_kv);
    GGML_ASSERT(dst->ne[1] == n_heads && dst->ne[2] == n_q_pos);

    const int head_dim = (int) q->ne[0];
    const bool project_q = head_dim == 128;
    const size_t sketch_elems = project_q ? (size_t) n_q_pos * n_heads * QK_QJL : 0;

    cudaStream_t stream = ctx.stream();

    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        const float * q_src = ggml_cuda_get_tensor_f32(q, ctx.device)
            + (size_t) i3 * q->nb[3] / sizeof(float);
        const void * pk_src = (const char *) pk->data + (size_t) i3 * pk->nb[3];
        const void * pv_src = (const char *) pv->data + (size_t) i3 * pv->nb[3];
        float * out_src = ggml_cuda_get_tensor_f32_mut(dst)
            + (size_t) i3 * dst->nb[3] / sizeof(float);

        const float * q_sketch = q_src;
        ggml_cuda_pool_alloc<float> q_sketch_scratch;
        if (project_q) {
            q_sketch_scratch.alloc(ctx.pool(), sketch_elems);
#if defined(GGML_CUDA_QJL)
            qjl_project_q_cuda(q_src, q_sketch_scratch.get(), head_dim, QK_QJL, n_heads, n_q_pos, stream);
#else
            GGML_ABORT("fused_attn_qjl: Q projection requires GGML_CUDA_QJL");
#endif
            q_sketch = q_sketch_scratch.get();
        } else {
            GGML_ASSERT(head_dim == QK_QJL);
        }

        if (pv->type == GGML_TYPE_Q4_POLAR) {
            fused_attn_qjl_polar_cuda(
                q_sketch, pk_src, pv_src,
                n_heads, n_kv_heads, n_q_pos, n_kv,
                scale_bits.f, v_use_qjl, kv_tile, causal, q_pos_base,
                out_src, stream);
        } else {
            fused_attn_qjl_tbq_cuda(
                q_sketch, pk_src, pv_src,
                n_heads, n_kv_heads, n_q_pos, n_kv,
                scale_bits.f, kv_tile,
                out_src, stream);
        }
    }
}

void ggml_cuda_op_attn_score_qjl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q  = dst->src[0];
    const ggml_tensor * pk = dst->src[1];

    GGML_ASSERT(q && pk);
    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(pk->type == GGML_TYPE_QJL1_256);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_rows(q));
    GGML_ASSERT(ggml_is_contiguous_rows(pk));
    GGML_ASSERT(ggml_is_contiguous_rows(dst));

    const int32_t * params = (const int32_t *) dst->op_params;
    const int n_kv_heads = params[0];
    const int n_heads = (int) q->ne[1];
    const int n_kv_tokens = (int) pk->ne[1];
    const int64_t ne3 = q->ne[3];

    GGML_ASSERT(n_kv_heads > 0);
    GGML_ASSERT((n_heads % n_kv_heads) == 0);
    GGML_ASSERT(q->ne[0] == QK_QJL || q->ne[0] == 128);
    GGML_ASSERT(pk->ne[0] == 128);

    cudaStream_t stream = ctx.stream();
    const int head_dim = (int) q->ne[0];
    const bool project_q = head_dim == 128;

    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        const float * q_src = ggml_cuda_get_tensor_f32(q, ctx.device)
            + (size_t) i3 * q->nb[3] / sizeof(float);
        const void * pk_src = (const char *) pk->data + (size_t) i3 * pk->nb[3];
        float * scores_src = ggml_cuda_get_tensor_f32_mut(dst)
            + (size_t) i3 * dst->nb[3] / sizeof(float);

        const float * q_sketch = q_src;
        if (project_q) {
            const int n_q_pos = (int) q->ne[2];
            const size_t sketch_elems = (size_t) n_q_pos * n_heads * QK_QJL;
            ggml_cuda_pool_alloc<float> q_sketch_scratch(ctx.pool(), sketch_elems);
#if defined(GGML_CUDA_QJL)
            qjl_project_q_cuda(q_src, q_sketch_scratch.get(), head_dim, QK_QJL, n_heads, n_q_pos, stream);
#else
            GGML_ABORT("attn_score_qjl: Q projection requires GGML_CUDA_QJL");
#endif
            q_sketch = q_sketch_scratch.get();
        }

#if defined(GGML_CUDA_QJL)
        attn_score_qjl_cuda(q_sketch, pk_src, n_heads, n_kv_heads, n_kv_tokens, scores_src, stream);
#else
        GGML_UNUSED(stream);
        GGML_ABORT("attn_score_qjl: GGML_CUDA_QJL required");
#endif
    }
}

#endif // GGML_CUDA_FUSED_ATTN_QJL
