/* P1 text / GQA path: embedding, head RMS, text RoPE, GQA causal SDPA. */
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

__global__ static void k_embedding_bf16(const __nv_bfloat16 *weight,
                                        const uint32_t *ids,
                                        __nv_bfloat16 *out, uint32_t tokens,
                                        uint32_t vocab, uint32_t width) {
  uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t tok = blockIdx.y;
  if (tok >= tokens || col >= width) return;
  uint32_t id = ids[tok];
  out[(size_t)tok * width + col] =
      id < vocab ? weight[(size_t)id * width + col] : __float2bfloat16(0.f);
}

int h3_gpu_embedding_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *token_ids, uint32_t tokens,
                          uint32_t vocab_size, uint32_t width) {
  if (!gpu || !output || !weight || !token_ids) return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  dim3 block(256);
  dim3 grid((width + 255) / 256, tokens);
  k_embedding_bf16<<<grid, block, 0, gpu->stream>>>(
      (const __nv_bfloat16 *)weight->device, (const uint32_t *)token_ids->device,
      (__nv_bfloat16 *)output->device, tokens, vocab_size, width);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "embedding_bf16")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

__global__ static void k_silu_mul_bf16(const __nv_bfloat16 *gate,
                                      const __nv_bfloat16 *up,
                                      __nv_bfloat16 *out, uint32_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = __bfloat162float(gate[i]);
  float u = __bfloat162float(up[i]);
  out[i] = __float2bfloat16((g / (1.f + expf(-g))) * u);
}

int h3_gpu_silu_mul_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *gate, const h3_gpu_tensor *up,
                         uint32_t elements) {
  if (!gpu || !output || !gate || !up) return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  int threads = 256;
  int blocks = (int)((elements + threads - 1) / threads);
  k_silu_mul_bf16<<<blocks, threads, 0, gpu->stream>>>(
      (const __nv_bfloat16 *)gate->device, (const __nv_bfloat16 *)up->device,
      (__nv_bfloat16 *)output->device, elements);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "silu_mul_bf16")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

/* One thread owns one (row, head) — race-free in-place RMS. */
__global__ static void k_head_rms_norm_bf16(__nv_bfloat16 *tensor,
                                            const __nv_bfloat16 *weight,
                                            uint32_t sequence, uint32_t heads,
                                            uint32_t head_dim, float epsilon) {
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t head = blockIdx.y;
  if (row >= sequence || head >= heads) return;
  size_t base = ((size_t)row * heads + head) * head_dim;
  float sum = 0.f;
  for (uint32_t d = 0; d < head_dim; d++) {
    float v = h3_bf16_to_f32(tensor[base + d]);
    sum = fmaf(v, v, sum);
  }
  float inv = rsqrtf(sum / (float)head_dim + epsilon);
  for (uint32_t d = 0; d < head_dim; d++) {
    float v = h3_bf16_to_f32(tensor[base + d]);
    tensor[base + d] =
        h3_f32_to_bf16(v * inv * h3_bf16_to_f32(weight[d]));
  }
}

int h3_gpu_head_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *tensor,
                              const h3_gpu_tensor *weight, uint32_t sequence,
                              uint32_t heads, uint32_t head_dim,
                              float epsilon) {
  if (!gpu || !tensor || !weight) return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  dim3 block(32);
  dim3 grid((sequence + 31) / 32, heads);
  k_head_rms_norm_bf16<<<grid, block, 0, gpu->stream>>>(
      (__nv_bfloat16 *)tensor->device, (const __nv_bfloat16 *)weight->device,
      sequence, heads, head_dim, epsilon);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "head_rms_norm_bf16")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

/* In-place RoPE; F32 cos/sin tables. */
__global__ static void k_rope_text_bf16(__nv_bfloat16 *query, __nv_bfloat16 *key,
                                       const float *rope_cos,
                                       const float *rope_sin, uint32_t sequence,
                                       uint32_t query_heads, uint32_t kv_heads,
                                       uint32_t head_dim) {
  uint32_t row = blockIdx.x;
  uint32_t head = blockIdx.y;
  if (row >= sequence) return;
  uint32_t half = head_dim / 2;
  if (head < query_heads) {
    size_t base = ((size_t)row * query_heads + head) * head_dim;
    for (uint32_t d = threadIdx.x; d < half; d += blockDim.x) {
      float first = h3_bf16_to_f32(query[base + d]);
      float second = h3_bf16_to_f32(query[base + half + d]);
      float c = rope_cos[(size_t)row * half + d];
      float s = rope_sin[(size_t)row * half + d];
      query[base + d] = h3_f32_to_bf16(first * c - second * s);
      query[base + half + d] = h3_f32_to_bf16(second * c + first * s);
    }
  }
  if (head < kv_heads) {
    size_t base = ((size_t)row * kv_heads + head) * head_dim;
    for (uint32_t d = threadIdx.x; d < half; d += blockDim.x) {
      float first = h3_bf16_to_f32(key[base + d]);
      float second = h3_bf16_to_f32(key[base + half + d]);
      float c = rope_cos[(size_t)row * half + d];
      float s = rope_sin[(size_t)row * half + d];
      key[base + d] = h3_f32_to_bf16(first * c - second * s);
      key[base + half + d] = h3_f32_to_bf16(second * c + first * s);
    }
  }
}

int h3_gpu_rope_text_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                          h3_gpu_tensor *key, const h3_gpu_tensor *rope_cos_f32,
                          const h3_gpu_tensor *rope_sin_f32, uint32_t sequence,
                          uint32_t query_heads, uint32_t kv_heads,
                          uint32_t head_dim) {
  if (!gpu || !query || !key || !rope_cos_f32 || !rope_sin_f32) return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  uint32_t max_heads = query_heads > kv_heads ? query_heads : kv_heads;
  dim3 grid(sequence, max_heads);
  k_rope_text_bf16<<<grid, 32, 0, gpu->stream>>>(
      (__nv_bfloat16 *)query->device, (__nv_bfloat16 *)key->device,
      (const float *)rope_cos_f32->device, (const float *)rope_sin_f32->device,
      sequence, query_heads, kv_heads, head_dim);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "rope_text_bf16")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

/* Combined Q/K RMS + RoPE (BF16 rope tables) — Metal text_qk_rope shape. */
__global__ static void k_text_qk_rope_bf16(
    const __nv_bfloat16 *query_in, const __nv_bfloat16 *key_in,
    const __nv_bfloat16 *q_weight, const __nv_bfloat16 *k_weight,
    const __nv_bfloat16 *rope_cos, const __nv_bfloat16 *rope_sin,
    __nv_bfloat16 *query_out, __nv_bfloat16 *key_out, uint32_t sequence,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim, float epsilon) {
  uint32_t dim = threadIdx.x;
  uint32_t head = blockIdx.y;
  uint32_t row = blockIdx.z;
  if (dim >= head_dim || head >= query_heads || row >= sequence) return;
  uint32_t half = head_dim / 2;
  uint32_t pair = dim < half ? dim + half : dim - half;
  float c = h3_bf16_to_f32(rope_cos[(size_t)row * half + (dim % half)]);
  float s = h3_bf16_to_f32(rope_sin[(size_t)row * half + (dim % half)]);

  size_t q_base = ((size_t)row * query_heads + head) * head_dim;
  float q_sum = 0.f;
  for (uint32_t d = 0; d < head_dim; d++) {
    float v = h3_bf16_to_f32(query_in[q_base + d]);
    q_sum = fmaf(v, v, q_sum);
  }
  float q_inv = rsqrtf(q_sum / (float)head_dim + epsilon);
  float q0 = h3_bf16_to_f32(query_in[q_base + dim]) * q_inv *
             h3_bf16_to_f32(q_weight[dim]);
  float q1 = h3_bf16_to_f32(query_in[q_base + pair]) * q_inv *
             h3_bf16_to_f32(q_weight[pair]);
  float q_rot = dim < half ? q0 * c - q1 * s : q0 * c + q1 * s;
  query_out[q_base + dim] = h3_f32_to_bf16(q_rot);

  if (head < kv_heads) {
    size_t k_base = ((size_t)row * kv_heads + head) * head_dim;
    float k_sum = 0.f;
    for (uint32_t d = 0; d < head_dim; d++) {
      float v = h3_bf16_to_f32(key_in[k_base + d]);
      k_sum = fmaf(v, v, k_sum);
    }
    float k_inv = rsqrtf(k_sum / (float)head_dim + epsilon);
    float k0 = h3_bf16_to_f32(key_in[k_base + dim]) * k_inv *
               h3_bf16_to_f32(k_weight[dim]);
    float k1 = h3_bf16_to_f32(key_in[k_base + pair]) * k_inv *
               h3_bf16_to_f32(k_weight[pair]);
    float k_rot = dim < half ? k0 * c - k1 * s : k0 * c + k1 * s;
    key_out[k_base + dim] = h3_f32_to_bf16(k_rot);
  }
}

int h3_gpu_text_qk_rope_bf16(
    h3_gpu *gpu, h3_gpu_tensor *query_output, h3_gpu_tensor *key_output,
    const h3_gpu_tensor *query_input, const h3_gpu_tensor *key_input,
    const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm,
    const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
    uint32_t sequence, uint32_t query_heads, uint32_t kv_heads,
    uint32_t head_dim, float epsilon) {
  if (!gpu || !query_output || !key_output || !query_input || !key_input ||
      !q_norm || !k_norm || !rope_cos || !rope_sin)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  if (head_dim > 1024) {
    h3_cuda_vset_error(gpu, "text_qk_rope head_dim too large");
    return 0;
  }
  dim3 block(head_dim);
  dim3 grid(1, query_heads, sequence);
  k_text_qk_rope_bf16<<<grid, block, 0, gpu->stream>>>(
      (const __nv_bfloat16 *)query_input->device,
      (const __nv_bfloat16 *)key_input->device,
      (const __nv_bfloat16 *)q_norm->device, (const __nv_bfloat16 *)k_norm->device,
      (const __nv_bfloat16 *)rope_cos->device,
      (const __nv_bfloat16 *)rope_sin->device,
      (__nv_bfloat16 *)query_output->device, (__nv_bfloat16 *)key_output->device,
      sequence, query_heads, kv_heads, head_dim, epsilon);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "text_qk_rope_bf16")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

/* Causal GQA with online softmax; scale applied to Q (Metal/MLX order). */
__global__ static void k_gqa_causal_online(
    const __nv_bfloat16 *query, const __nv_bfloat16 *key,
    const __nv_bfloat16 *value, __nv_bfloat16 *output, uint32_t sequence,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim, float scale) {
  uint32_t q_row = blockIdx.x;
  uint32_t q_head = blockIdx.y;
  if (q_row >= sequence || q_head >= query_heads) return;
  uint32_t group = query_heads / kv_heads;
  uint32_t kv_head = q_head / group;
  extern __shared__ float smem[];
  /* q_scaled[head_dim] | acc[head_dim] | red[blockDim] */
  float *q_scaled = smem;
  float *acc = smem + head_dim;
  float *red = smem + 2 * head_dim;
  uint32_t tid = threadIdx.x;

  size_t q_base = ((size_t)q_row * query_heads + q_head) * head_dim;
  for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
    float qv = h3_bf16_to_f32(query[q_base + d]) * scale;
    q_scaled[d] = h3_bf16_to_f32(h3_f32_to_bf16(qv));
    acc[d] = 0.f;
  }
  __syncthreads();

  float m_i = -INFINITY;
  float l_i = 0.f;
  uint32_t key_count = q_row + 1;
  for (uint32_t k_row = 0; k_row < key_count; k_row++) {
    size_t k_base = ((size_t)k_row * kv_heads + kv_head) * head_dim;
    float partial = 0.f;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x)
      partial = fmaf(q_scaled[d], h3_bf16_to_f32(key[k_base + d]), partial);
    red[tid] = partial;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s; s >>= 1) {
      if (tid < s) red[tid] += red[tid + s];
      __syncthreads();
    }
    float score = red[0];
    float m_new = fmaxf(m_i, score);
    float alpha = expf(m_i - m_new);
    float beta = expf(score - m_new);
    float l_new = alpha * l_i + beta;
    size_t v_base = ((size_t)k_row * kv_heads + kv_head) * head_dim;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x)
      acc[d] = alpha * acc[d] + beta * h3_bf16_to_f32(value[v_base + d]);
    m_i = m_new;
    l_i = l_new;
    __syncthreads();
  }
  float inv = 1.f / l_i;
  for (uint32_t d = tid; d < head_dim; d += blockDim.x)
    output[q_base + d] = h3_f32_to_bf16(acc[d] * inv);
}

int h3_gpu_gqa_causal_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key, const h3_gpu_tensor *value,
                           uint32_t sequence, uint32_t query_heads,
                           uint32_t kv_heads, uint32_t head_dim, float scale) {
  if (!gpu || !output || !query || !key || !value) return 0;
  if (!query_heads || !kv_heads || (query_heads % kv_heads) != 0) {
    h3_cuda_vset_error(gpu, "invalid GQA head counts");
    return 0;
  }
  if (!h3_cuda_require_encoding(gpu)) return 0;
  if (head_dim > 256) {
    h3_cuda_vset_error(gpu, "gqa head_dim > 256 unsupported");
    return 0;
  }
  dim3 grid(sequence, query_heads);
  int threads = 128;
  size_t shmem = (size_t)(2 * head_dim + threads) * sizeof(float);
  k_gqa_causal_online<<<grid, threads, shmem, gpu->stream>>>(
      (const __nv_bfloat16 *)query->device, (const __nv_bfloat16 *)key->device,
      (const __nv_bfloat16 *)value->device, (__nv_bfloat16 *)output->device,
      sequence, query_heads, kv_heads, head_dim, scale);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "gqa_causal_bf16")) return 0;
  gpu->stats.mps_sdpa_dispatches++;
  return 1;
}

#ifdef __cplusplus
}
#endif
