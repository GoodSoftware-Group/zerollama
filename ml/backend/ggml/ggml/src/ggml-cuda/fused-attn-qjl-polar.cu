// Fused attention: QJL-K score + Q4_POLAR-V mix (online softmax).
// Mirrors fused_attn_qjl_polar.comp / fused_attn_qjl_polar.metal.

#include "ggml.h"
#include "ggml-impl.h"
#include "common.cuh"

#if defined(GGML_CUDA_FUSED_ATTN_QJL)

#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <cstring>

#define FUSED_PROJ_DIM   QK_QJL
#define FUSED_HEAD_DIM   128
#define FUSED_WARP       32
#define FUSED_MIN_BLOCKS_PER_SM 16
#define FUSED_POLAR_BLK_BYTES 82

#define FUSED_QJL_SQRT_PI_OVER_2 1.2533141373155003f

__constant__ float k_fused_polar_q4_centroids[16] = {
    -2.754354807f, -2.093562707f, -1.643041510f, -1.279739752f,
    -0.962640978f, -0.672392117f, -0.397897103f, -0.131757782f,
     0.131757782f,  0.397897103f,  0.672392117f,  0.962640978f,
     1.279739752f,  1.643041510f,  2.093562707f,  2.754354807f,
};

__constant__ float k_fused_polar_qjl_signs[128] = {
    -1.f, -1.f,  1.f, -1.f, -1.f, -1.f, -1.f, -1.f,
     1.f,  1.f, -1.f,  1.f, -1.f, -1.f,  1.f,  1.f,
    -1.f, -1.f,  1.f, -1.f,  1.f,  1.f, -1.f, -1.f,
    -1.f,  1.f, -1.f,  1.f,  1.f,  1.f, -1.f, -1.f,
    -1.f, -1.f,  1.f, -1.f,  1.f, -1.f,  1.f, -1.f,
    -1.f, -1.f, -1.f,  1.f, -1.f,  1.f,  1.f,  1.f,
     1.f,  1.f, -1.f,  1.f, -1.f, -1.f,  1.f,  1.f,
     1.f,  1.f, -1.f, -1.f, -1.f,  1.f, -1.f,  1.f,
     1.f, -1.f,  1.f, -1.f,  1.f,  1.f,  1.f,  1.f,
    -1.f, -1.f, -1.f, -1.f,  1.f, -1.f, -1.f, -1.f,
     1.f, -1.f, -1.f,  1.f,  1.f,  1.f,  1.f, -1.f,
    -1.f,  1.f, -1.f,  1.f,  1.f, -1.f,  1.f,  1.f,
     1.f, -1.f, -1.f, -1.f,  1.f,  1.f, -1.f,  1.f,
    -1.f,  1.f,  1.f, -1.f, -1.f,  1.f, -1.f, -1.f,
     1.f, -1.f,  1.f, -1.f, -1.f,  1.f, -1.f, -1.f,
     1.f,  1.f,  1.f, -1.f,  1.f, -1.f, -1.f,  1.f,
};

static __device__ __forceinline__ float fused_bf16_to_fp32(uint16_t bits) {
    uint32_t u = ((uint32_t) bits) << 16;
    float f; memcpy(&f, &u, sizeof(f)); return f;
}
static __device__ __forceinline__ float fused_fp16_to_fp32(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

static __device__ __forceinline__ float fused_polar_qjl_score_partial(
        const block_qjl1_256 & blk, const float qreg[8], int lane) {
    const uint8_t sb = blk.signs[lane];
    float partial = 0.f;
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        partial += ((sb >> b) & 1) ? qreg[b] : -qreg[b];
    }
    return partial;
}

static __device__ __forceinline__ void fused_polar_hadamard128(float * x) {
    for (int h = 1; h < FUSED_HEAD_DIM; h <<= 1) {
        for (int i = 0; i < FUSED_HEAD_DIM; i += (h << 1)) {
            for (int j = 0; j < h; ++j) {
                const float a = x[i + j];
                const float b = x[i + j + h];
                x[i + j]     = a + b;
                x[i + j + h] = a - b;
            }
        }
    }
}

__global__ void __launch_bounds__(FUSED_WARP, FUSED_MIN_BLOCKS_PER_SM)
fused_attn_qjl_polar_kernel(
        const float * __restrict__ q_sketch,
        const block_qjl1_256 * __restrict__ k_blocks,
        const block_q4_polar * __restrict__ v_blocks,
        int proj_dim, int n_heads, int n_kv_heads, int n_q_pos, int n_kv,
        float sm_scale, int v_use_qjl, int kv_tile, int causal, int q_pos_base,
        float * __restrict__ out) {
    const int lane  = threadIdx.x;
    const int hq    = blockIdx.x;
    const int q_pos = blockIdx.y;
    if (hq >= n_heads || q_pos >= n_q_pos || lane >= FUSED_WARP) return;

    const int gqa = n_heads / n_kv_heads;
    const int hk  = hq / gqa;
    const float * qh = q_sketch + ((size_t) q_pos * n_heads + hq) * proj_dim;
    const block_qjl1_256 * pk = k_blocks + (size_t) hk * n_kv;
    const block_q4_polar * pv = v_blocks + (size_t) hk * n_kv;
    float * oh = out + ((size_t) q_pos * n_heads + hq) * FUSED_HEAD_DIM;

    const float qjl_scl = FUSED_QJL_SQRT_PI_OVER_2 / (float) proj_dim;
    const int q_abs = q_pos_base + q_pos;

    float qreg[8];
    {
        const float * qbase = qh + lane * 8;
        if (((uintptr_t) qbase & 0xF) == 0) {
            const float4 a = reinterpret_cast<const float4 *>(qbase)[0];
            const float4 b = reinterpret_cast<const float4 *>(qbase)[1];
            qreg[0] = a.x; qreg[1] = a.y; qreg[2] = a.z; qreg[3] = a.w;
            qreg[4] = b.x; qreg[5] = b.y; qreg[6] = b.z; qreg[7] = b.w;
        } else {
            #pragma unroll
            for (int b = 0; b < 8; ++b) qreg[b] = qbase[b];
        }
    }

    __shared__ float sh_vbuf[FUSED_HEAD_DIM];

    float acc_lane[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) acc_lane[c] = 0.f;
    float m = -INFINITY;
    float l = 0.f;

    if (n_kv == 0) {
        for (int i = lane; i < FUSED_HEAD_DIM; i += FUSED_WARP) {
            oh[i] = 0.f;
        }
        return;
    }

    const int tile = (kv_tile > 0) ? kv_tile : n_kv;
    for (int t0 = 0; t0 < n_kv; t0 += tile) {
        const int t1 = (t0 + tile < n_kv) ? (t0 + tile) : n_kv;
        for (int t = t0; t < t1; ++t) {
            if (causal && t > q_abs) break;

            float partial = fused_polar_qjl_score_partial(pk[t], qreg, lane);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(0xFFFFFFFFu, partial, off);
            }
            const float score = qjl_scl * fused_bf16_to_fp32(__ldg(&pk[t].d)) * partial * sm_scale;

            const float m_new = fmaxf(m, score);
            const float corr  = __expf(m - m_new);
            const float w     = __expf(score - m_new);
            l = l * corr + w;
            m = m_new;

            const block_q4_polar & vblk = pv[t];
            if (lane < FUSED_HEAD_DIM / 2) {
                const uint8_t byte = vblk.qs[lane];
                sh_vbuf[2 * lane]     = k_fused_polar_q4_centroids[byte & 0x0Fu];
                sh_vbuf[2 * lane + 1] = k_fused_polar_q4_centroids[(byte >> 4) & 0x0Fu];
            }
            __syncwarp();

            if (v_use_qjl) {
                const uint8_t bit = vblk.qjl[0] & 1u;
                const float sign_v = bit ? 1.f : -1.f;
                const float scaled = sign_v * 0.5f * 0.08838834764831845f;
                if (lane < FUSED_HEAD_DIM) {
                    sh_vbuf[lane] += scaled * k_fused_polar_qjl_signs[lane];
                }
                __syncthreads();
            }

            if (lane == 0) {
                fused_polar_hadamard128(sh_vbuf);
            }
            __syncthreads();

            const float l2 = fused_fp16_to_fp32(__ldg(reinterpret_cast<const uint16_t *>(&vblk.d)));
            const float scale = w * l2 * (1.f / (float) FUSED_HEAD_DIM);
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int idx = c * 32 + lane;
                acc_lane[c] = acc_lane[c] * corr + scale * sh_vbuf[idx];
            }
            __syncwarp();
        }
    }

    const float inv_l = (l > 0.f && isfinite(m)) ? (1.f / l) : 0.f;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        oh[c * 32 + lane] = acc_lane[c] * inv_l;
    }
}

extern "C" void fused_attn_qjl_polar_cuda(
        const float * q_sketch_d,
        const void  * packed_k_d,
        const void  * packed_v_d,
        int n_heads, int n_kv_heads, int n_q_pos, int n_kv,
        float sm_scale, int v_use_qjl, int kv_tile, int causal, int q_pos_base,
        float * out_d,
        cudaStream_t stream) {
    GGML_ASSERT(n_heads > 0 && n_kv_heads > 0 && n_q_pos > 0 && n_kv >= 0);
    GGML_ASSERT((n_heads % n_kv_heads) == 0);
    const dim3 grid(n_heads, n_q_pos, 1);
    fused_attn_qjl_polar_kernel<<<grid, FUSED_WARP, 0, stream>>>(
        q_sketch_d,
        (const block_qjl1_256 *) packed_k_d,
        (const block_q4_polar   *) packed_v_d,
        FUSED_PROJ_DIM, n_heads, n_kv_heads, n_q_pos, n_kv,
        sm_scale, v_use_qjl, kv_tile, causal, q_pos_base, out_d);
}

#endif // GGML_CUDA_FUSED_ATTN_QJL
