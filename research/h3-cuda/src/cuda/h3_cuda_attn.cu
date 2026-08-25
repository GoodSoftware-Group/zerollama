#include "h3_cuda_internal.h"

#include <cuda_bf16.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif


__device__ inline float h3_bf16_to_f32(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
__device__ inline __nv_bfloat16 h3_f32_to_bf16(float v) {
  return __float2bfloat16(v);
}

/* Online-softmax SDPA (FlashAttention-style) — O(head_dim) scratch, not O(seq).
 * Layout: Q/K/V/out are row-major [seq, heads, dim]. One block = one (row, head). */
__global__ void k_sdpa_bf16_online(const __nv_bfloat16 *q, const __nv_bfloat16 *k,
                                   const __nv_bfloat16 *v, __nv_bfloat16 *out,
                                   uint32_t sequence, uint32_t heads,
                                   uint32_t head_dim, float scale) {
  uint32_t head = blockIdx.y;
  uint32_t row = blockIdx.x;
  if (head >= heads || row >= sequence) return;
  extern __shared__ float smem[];
  /* smem: q_row[head_dim] | acc[head_dim] | red[blockDim] */
  float *q_row = smem;
  float *acc = smem + head_dim;
  float *red = smem + 2 * head_dim;
  uint32_t tid = threadIdx.x;

  const __nv_bfloat16 *qr = q + ((size_t)row * heads + head) * head_dim;
  for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
    q_row[d] = h3_bf16_to_f32(qr[d]);
    acc[d] = 0.f;
  }
  __syncthreads();

  float m_i = -INFINITY;
  float l_i = 0.f;

  for (uint32_t col = 0; col < sequence; col++) {
    const __nv_bfloat16 *kc = k + ((size_t)col * heads + head) * head_dim;
    float partial = 0.f;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x)
      partial = fmaf(q_row[d], h3_bf16_to_f32(kc[d]), partial);
    red[tid] = partial;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s; s >>= 1) {
      if (tid < s) red[tid] += red[tid + s];
      __syncthreads();
    }
    float score = red[0] * scale;
    float m_new = fmaxf(m_i, score);
    float alpha = expf(m_i - m_new);
    float beta = expf(score - m_new);
    float l_new = alpha * l_i + beta;
    const __nv_bfloat16 *vc = v + ((size_t)col * heads + head) * head_dim;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x)
      acc[d] = alpha * acc[d] + beta * h3_bf16_to_f32(vc[d]);
    m_i = m_new;
    l_i = l_new;
    __syncthreads();
  }

  float inv = 1.f / l_i;
  __nv_bfloat16 *o = out + ((size_t)row * heads + head) * head_dim;
  for (uint32_t d = tid; d < head_dim; d += blockDim.x)
    o[d] = h3_f32_to_bf16(acc[d] * inv);
}

int h3_gpu_sdpa_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale) {
  if (!gpu || !output || !query || !key || !value) return 0;
  size_t n = (size_t)sequence * heads * head_dim;
  if (query->elements < n || key->elements < n || value->elements < n ||
      output->elements < n)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  if (head_dim > 256) {
    h3_cuda_vset_error(gpu, "sdpa_bf16 head_dim > 256 unsupported");
    return 0;
  }
  dim3 grid(sequence, heads);
  int threads = 128;
  if (head_dim >= 128) threads = 128;
  size_t shmem = (size_t)(2 * head_dim + threads) * sizeof(float);
  k_sdpa_bf16_online<<<grid, threads, shmem, gpu->stream>>>(
      (const __nv_bfloat16 *)query->device, (const __nv_bfloat16 *)key->device,
      (const __nv_bfloat16 *)value->device, (__nv_bfloat16 *)output->device,
      sequence, heads, head_dim, scale);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "sdpa_bf16")) return 0;
  gpu->stats.mps_sdpa_dispatches++;
  return 1;
}

__global__ void k_grouped_qkv_rope_bf16(
    const __nv_bfloat16 *qkv, const __nv_bfloat16 *q_norm,
    const __nv_bfloat16 *k_norm, const __nv_bfloat16 *rope_cos,
    const __nv_bfloat16 *rope_sin, __nv_bfloat16 *query, __nv_bfloat16 *key,
    __nv_bfloat16 *value, uint32_t sequence, uint32_t heads, uint32_t head_dim,
    uint32_t rope_half, float epsilon) {
  uint32_t row = blockIdx.x;
  uint32_t head = blockIdx.y;
  if (row >= sequence || head >= heads) return;
  uint32_t tid = threadIdx.x;
  const __nv_bfloat16 *src =
      qkv + (size_t)row * heads * 3 * head_dim + (size_t)head * 3 * head_dim;
  extern __shared__ float red[];
  float qsum = 0.f, ksum = 0.f;
  for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
    float qv = h3_bf16_to_f32(src[d]);
    float kv = h3_bf16_to_f32(src[head_dim + d]);
    qsum = fmaf(qv, qv, qsum);
    ksum = fmaf(kv, kv, ksum);
  }
  red[tid] = qsum;
  red[tid + blockDim.x] = ksum;
  __syncthreads();
  for (uint32_t s = blockDim.x / 2; s; s >>= 1) {
    if (tid < s) {
      red[tid] += red[tid + s];
      red[tid + blockDim.x] += red[tid + blockDim.x + s];
    }
    __syncthreads();
  }
  float qinv = rsqrtf(red[0] / (float)head_dim + epsilon);
  float kinv = rsqrtf(red[blockDim.x] / (float)head_dim + epsilon);
  __nv_bfloat16 *qo = query + ((size_t)row * heads + head) * head_dim;
  __nv_bfloat16 *ko = key + ((size_t)row * heads + head) * head_dim;
  __nv_bfloat16 *vo = value + ((size_t)row * heads + head) * head_dim;
  for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
    float qv = h3_bf16_to_f32(src[d]) * qinv * h3_bf16_to_f32(q_norm[d]);
    float kv =
        h3_bf16_to_f32(src[head_dim + d]) * kinv * h3_bf16_to_f32(k_norm[d]);
    float vv = h3_bf16_to_f32(src[2 * head_dim + d]);
    qo[d] = h3_f32_to_bf16(qv);
    ko[d] = h3_f32_to_bf16(kv);
    vo[d] = h3_f32_to_bf16(vv);
  }
  __syncthreads();
  for (uint32_t d = tid; d < rope_half; d += blockDim.x) {
    float c = h3_bf16_to_f32(rope_cos[(size_t)row * rope_half + d]);
    float s = h3_bf16_to_f32(rope_sin[(size_t)row * rope_half + d]);
    float q0 = h3_bf16_to_f32(qo[d]);
    float q1 = h3_bf16_to_f32(qo[d + rope_half]);
    float k0 = h3_bf16_to_f32(ko[d]);
    float k1 = h3_bf16_to_f32(ko[d + rope_half]);
    qo[d] = h3_f32_to_bf16(q0 * c - q1 * s);
    qo[d + rope_half] = h3_f32_to_bf16(q0 * s + q1 * c);
    ko[d] = h3_f32_to_bf16(k0 * c - k1 * s);
    ko[d + rope_half] = h3_f32_to_bf16(k0 * s + k1 * c);
  }
}

int h3_gpu_grouped_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                                 h3_gpu_tensor *key, h3_gpu_tensor *value,
                                 const h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t sequence, uint32_t heads,
                                 uint32_t head_dim, uint32_t rope_half,
                                 float epsilon) {
  if (!gpu || !query || !key || !value || !qkv || !q_norm || !k_norm ||
      !rope_cos || !rope_sin)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  dim3 grid(sequence, heads);
  int threads = 256;
  size_t shmem = (size_t)threads * 2 * sizeof(float);
  k_grouped_qkv_rope_bf16<<<grid, threads, shmem, gpu->stream>>>(
      (const __nv_bfloat16 *)qkv->device, (const __nv_bfloat16 *)q_norm->device,
      (const __nv_bfloat16 *)k_norm->device,
      (const __nv_bfloat16 *)rope_cos->device,
      (const __nv_bfloat16 *)rope_sin->device, (__nv_bfloat16 *)query->device,
      (__nv_bfloat16 *)key->device, (__nv_bfloat16 *)value->device, sequence,
      heads, head_dim, rope_half, epsilon);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "grouped_qkv_rope_bf16"))
    return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_grouped_qkv_linear_rope_bf16(
    h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
    h3_gpu_tensor *value, h3_gpu_tensor *qkv, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *q_norm,
    const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos,
    const h3_gpu_tensor *rope_sin, uint32_t rows, uint32_t input_dim,
    uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon) {
  if (!h3_gpu_linear_bf16(gpu, qkv, input, weight, NULL, rows, input_dim,
                          heads * 3 * head_dim))
    return 0;
  return h3_gpu_grouped_qkv_rope_bf16(gpu, query, key, value, qkv, q_norm,
                                      k_norm, rope_cos, rope_sin, rows, heads,
                                      head_dim, rope_half, epsilon);
}

#ifdef __cplusplus
} /* extern "C" */
#endif
