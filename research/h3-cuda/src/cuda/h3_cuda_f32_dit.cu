/* F32 DiT helpers + ungrouped BF16 QKV/RoPE (Metal h3_*_f32 / h3_qkv_rope_bf16). */
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

/* ---- AdaLN / gate F32 ---- */

__global__ static void k_adaln_f32(const float *input, const float *weight,
                                   const float *modulation,
                                   const uint32_t *row_map, float *output,
                                   uint32_t rows, uint32_t width,
                                   uint32_t slots, uint32_t shift_slot,
                                   uint32_t scale_slot, float epsilon) {
  uint32_t row = blockIdx.x;
  if (row >= rows) return;
  extern __shared__ float red[];
  uint32_t tid = threadIdx.x;
  const float *x = input + (size_t)row * width;
  float local = 0.f;
  for (uint32_t k = tid; k < width; k += blockDim.x)
    local = fmaf(x[k], x[k], local);
  red[tid] = local;
  __syncthreads();
  for (uint32_t s = blockDim.x / 2; s; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / (float)width + epsilon);
  size_t base = (size_t)row_map[row] * slots * width;
  for (uint32_t c = tid; c < width; c += blockDim.x) {
    float n = x[c] * inv * weight[c];
    float shift = modulation[base + (size_t)shift_slot * width + c];
    float scale = modulation[base + (size_t)scale_slot * width + c];
    output[(size_t)row * width + c] = n * (1.f + scale) + shift;
  }
}

__global__ static void k_gate_f32(const float *residual, const float *branch,
                                  const float *modulation,
                                  const uint32_t *row_map, float *output,
                                  uint32_t rows, uint32_t width,
                                  uint32_t slots, uint32_t gate_slot) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = (size_t)rows * width;
  if (idx >= n) return;
  uint32_t row = (uint32_t)(idx / width);
  uint32_t col = (uint32_t)(idx % width);
  size_t base = (size_t)row_map[row] * slots * width;
  float gate = modulation[base + (size_t)gate_slot * width + col];
  output[idx] = residual[idx] + branch[idx] * gate;
}

int h3_gpu_adaln_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon) {
  if (!gpu || !output || !input || !norm_weight || !modulation || !row_map)
    return 0;
  if (shift_slot >= slots || scale_slot >= slots) return 0;
  size_t count = (size_t)rows * width;
  if (input->elements < count || output->elements < count ||
      norm_weight->elements < width || row_map->elements < rows)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  int threads = 256;
  size_t shmem = (size_t)threads * sizeof(float);
  k_adaln_f32<<<rows, threads, shmem, gpu->stream>>>(
      (const float *)input->device, (const float *)norm_weight->device,
      (const float *)modulation->device, (const uint32_t *)row_map->device,
      (float *)output->device, rows, width, slots, shift_slot, scale_slot,
      epsilon);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "adaln_f32")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_gate_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *residual, const h3_gpu_tensor *branch,
                    const h3_gpu_tensor *modulation,
                    const h3_gpu_tensor *row_map, uint32_t rows,
                    uint32_t width, uint32_t slots, uint32_t gate_slot) {
  if (!gpu || !output || !residual || !branch || !modulation || !row_map)
    return 0;
  if (gate_slot >= slots) return 0;
  size_t count = (size_t)rows * width;
  if (residual->elements < count || branch->elements < count ||
      output->elements < count || row_map->elements < rows)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  int threads = 256;
  int blocks = (int)((count + threads - 1) / threads);
  k_gate_f32<<<blocks, threads, 0, gpu->stream>>>(
      (const float *)residual->device, (const float *)branch->device,
      (const float *)modulation->device, (const uint32_t *)row_map->device,
      (float *)output->device, rows, width, slots, gate_slot);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "gate_f32")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

/* ---- Ungrouped QKV + RoPE (F32 / BF16) ----
 * Layout: [Q | K | V] with each block [seq, heads, dim] contiguous. */

__global__ static void k_qkv_rope_f32(const float *qkv, const float *q_norm,
                                      const float *k_norm, const float *rope_cos,
                                      const float *rope_sin, float *query,
                                      float *key, float *value,
                                      uint32_t sequence, uint32_t heads,
                                      uint32_t head_dim, uint32_t rope_half,
                                      float epsilon) {
  uint32_t row = blockIdx.x;
  uint32_t head = blockIdx.y;
  if (row >= sequence || head >= heads) return;
  uint32_t tid = threadIdx.x;
  uint32_t inner = heads * head_dim;
  size_t row_base = (size_t)row * inner * 3;
  size_t q_base = row_base + (size_t)head * head_dim;
  size_t k_base = q_base + inner;
  size_t v_base = q_base + (size_t)inner * 2;
  extern __shared__ float red[];
  float qsum = 0.f, ksum = 0.f;
  for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
    float qv = qkv[q_base + d];
    float kv = qkv[k_base + d];
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
  float *qo = query + ((size_t)row * heads + head) * head_dim;
  float *ko = key + ((size_t)row * heads + head) * head_dim;
  float *vo = value + ((size_t)row * heads + head) * head_dim;
  for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
    qo[d] = qkv[q_base + d] * qinv * q_norm[d];
    ko[d] = qkv[k_base + d] * kinv * k_norm[d];
    vo[d] = qkv[v_base + d];
  }
  __syncthreads();
  for (uint32_t d = tid; d < rope_half; d += blockDim.x) {
    float c = rope_cos[(size_t)row * rope_half + d];
    float s = rope_sin[(size_t)row * rope_half + d];
    float q0 = qo[d], q1 = qo[d + rope_half];
    float k0 = ko[d], k1 = ko[d + rope_half];
    qo[d] = q0 * c - q1 * s;
    qo[d + rope_half] = q0 * s + q1 * c;
    ko[d] = k0 * c - k1 * s;
    ko[d + rope_half] = k0 * s + k1 * c;
  }
}

__global__ static void k_qkv_rope_bf16(
    const __nv_bfloat16 *qkv, const __nv_bfloat16 *q_norm,
    const __nv_bfloat16 *k_norm, const __nv_bfloat16 *rope_cos,
    const __nv_bfloat16 *rope_sin, __nv_bfloat16 *query, __nv_bfloat16 *key,
    __nv_bfloat16 *value, uint32_t sequence, uint32_t heads, uint32_t head_dim,
    uint32_t rope_half, float epsilon) {
  uint32_t row = blockIdx.x;
  uint32_t head = blockIdx.y;
  if (row >= sequence || head >= heads) return;
  uint32_t tid = threadIdx.x;
  uint32_t inner = heads * head_dim;
  size_t row_base = (size_t)row * inner * 3;
  size_t q_base = row_base + (size_t)head * head_dim;
  size_t k_base = q_base + inner;
  size_t v_base = q_base + (size_t)inner * 2;
  extern __shared__ float red[];
  float qsum = 0.f, ksum = 0.f;
  for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
    float qv = h3_bf16_to_f32(qkv[q_base + d]);
    float kv = h3_bf16_to_f32(qkv[k_base + d]);
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
    qo[d] = h3_f32_to_bf16(h3_bf16_to_f32(qkv[q_base + d]) * qinv *
                           h3_bf16_to_f32(q_norm[d]));
    ko[d] = h3_f32_to_bf16(h3_bf16_to_f32(qkv[k_base + d]) * kinv *
                           h3_bf16_to_f32(k_norm[d]));
    vo[d] = qkv[v_base + d];
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

/* Video VAE attention: interleaved [Q|K|V] per head, RMS only (no q/k norm). */
__global__ static void k_video_qkv_rope_f32(const float *qkv,
                                            const float *rope_cos,
                                            const float *rope_sin, float *query,
                                            float *key, float *value,
                                            uint32_t sequence, uint32_t heads,
                                            uint32_t head_dim,
                                            uint32_t rope_half, float epsilon) {
  uint32_t row = blockIdx.x;
  uint32_t head = blockIdx.y;
  if (row >= sequence || head >= heads) return;
  uint32_t tid = threadIdx.x;
  size_t base = ((size_t)row * heads + head) * head_dim * 3;
  extern __shared__ float red[];
  float qsum = 0.f, ksum = 0.f;
  for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
    float qv = qkv[base + d];
    float kv = qkv[base + head_dim + d];
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
  float *qo = query + ((size_t)row * heads + head) * head_dim;
  float *ko = key + ((size_t)row * heads + head) * head_dim;
  float *vo = value + ((size_t)row * heads + head) * head_dim;
  for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
    qo[d] = qkv[base + d] * qinv;
    ko[d] = qkv[base + head_dim + d] * kinv;
    vo[d] = qkv[base + 2 * head_dim + d];
  }
  __syncthreads();
  for (uint32_t d = tid; d < rope_half; d += blockDim.x) {
    float c = rope_cos[(size_t)row * rope_half + d];
    float s = rope_sin[(size_t)row * rope_half + d];
    float q0 = qo[d], q1 = qo[d + rope_half];
    float k0 = ko[d], k1 = ko[d + rope_half];
    qo[d] = q0 * c - q1 * s;
    qo[d + rope_half] = q0 * s + q1 * c;
    ko[d] = k0 * c - k1 * s;
    ko[d + rope_half] = k0 * s + k1 * c;
  }
}

int h3_gpu_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
                        h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
                        const h3_gpu_tensor *q_norm,
                        const h3_gpu_tensor *k_norm,
                        const h3_gpu_tensor *rope_cos,
                        const h3_gpu_tensor *rope_sin, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim, uint32_t rope_half,
                        float epsilon) {
  if (!gpu || !query || !key || !value || !qkv || !q_norm || !k_norm ||
      !rope_cos || !rope_sin)
    return 0;
  if (rope_half * 2 > head_dim) return 0;
  size_t inner = (size_t)heads * head_dim;
  size_t count = (size_t)sequence * inner;
  if (qkv->elements < count * 3 || query->elements < count ||
      key->elements < count || value->elements < count ||
      q_norm->elements < head_dim || k_norm->elements < head_dim ||
      rope_cos->elements < (size_t)sequence * rope_half ||
      rope_sin->elements < (size_t)sequence * rope_half)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  dim3 grid(sequence, heads);
  int threads = 256;
  size_t shmem = (size_t)threads * 2 * sizeof(float);
  k_qkv_rope_f32<<<grid, threads, shmem, gpu->stream>>>(
      (const float *)qkv->device, (const float *)q_norm->device,
      (const float *)k_norm->device, (const float *)rope_cos->device,
      (const float *)rope_sin->device, (float *)query->device,
      (float *)key->device, (float *)value->device, sequence, heads, head_dim,
      rope_half, epsilon);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "qkv_rope_f32")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
                         h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
                         const h3_gpu_tensor *q_norm,
                         const h3_gpu_tensor *k_norm,
                         const h3_gpu_tensor *rope_cos,
                         const h3_gpu_tensor *rope_sin, uint32_t sequence,
                         uint32_t heads, uint32_t head_dim, uint32_t rope_half,
                         float epsilon) {
  if (!gpu || !query || !key || !value || !qkv || !q_norm || !k_norm ||
      !rope_cos || !rope_sin)
    return 0;
  if (rope_half * 2 > head_dim) return 0;
  size_t inner = (size_t)heads * head_dim;
  size_t count = (size_t)sequence * inner;
  if (qkv->elements < count * 3 || query->elements < count ||
      key->elements < count || value->elements < count)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  dim3 grid(sequence, heads);
  int threads = 256;
  size_t shmem = (size_t)threads * 2 * sizeof(float);
  k_qkv_rope_bf16<<<grid, threads, shmem, gpu->stream>>>(
      (const __nv_bfloat16 *)qkv->device, (const __nv_bfloat16 *)q_norm->device,
      (const __nv_bfloat16 *)k_norm->device,
      (const __nv_bfloat16 *)rope_cos->device,
      (const __nv_bfloat16 *)rope_sin->device, (__nv_bfloat16 *)query->device,
      (__nv_bfloat16 *)key->device, (__nv_bfloat16 *)value->device, sequence,
      heads, head_dim, rope_half, epsilon);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "qkv_rope_bf16")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_video_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                              h3_gpu_tensor *key, h3_gpu_tensor *value,
                              const h3_gpu_tensor *qkv,
                              const h3_gpu_tensor *rope_cos,
                              const h3_gpu_tensor *rope_sin, uint32_t sequence,
                              uint32_t heads, uint32_t head_dim,
                              uint32_t rope_half, float epsilon) {
  if (!gpu || !query || !key || !value || !qkv || !rope_cos || !rope_sin)
    return 0;
  if (rope_half * 2 > head_dim) return 0;
  size_t inner = (size_t)heads * head_dim;
  size_t count = (size_t)sequence * inner;
  if (qkv->elements < count * 3 || query->elements < count ||
      key->elements < count || value->elements < count ||
      rope_cos->elements < (size_t)sequence * rope_half ||
      rope_sin->elements < (size_t)sequence * rope_half)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  dim3 grid(sequence, heads);
  int threads = 256;
  size_t shmem = (size_t)threads * 2 * sizeof(float);
  k_video_qkv_rope_f32<<<grid, threads, shmem, gpu->stream>>>(
      (const float *)qkv->device, (const float *)rope_cos->device,
      (const float *)rope_sin->device, (float *)query->device,
      (float *)key->device, (float *)value->device, sequence, heads, head_dim,
      rope_half, epsilon);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "video_qkv_rope_f32")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

/* ---- Non-causal F32 SDPA (row-major [seq,heads,dim]) ---- */

__global__ static void k_sdpa_f32_online(const float *q, const float *k,
                                         const float *v, float *out,
                                         uint32_t sequence, uint32_t heads,
                                         uint32_t head_dim, float scale) {
  uint32_t head = blockIdx.y;
  uint32_t row = blockIdx.x;
  if (head >= heads || row >= sequence) return;
  extern __shared__ float smem[];
  float *q_row = smem;
  float *acc = smem + head_dim;
  float *red = smem + 2 * head_dim;
  uint32_t tid = threadIdx.x;
  const float *qr = q + ((size_t)row * heads + head) * head_dim;
  for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
    q_row[d] = qr[d];
    acc[d] = 0.f;
  }
  __syncthreads();
  float m_i = -INFINITY;
  float l_i = 0.f;
  for (uint32_t col = 0; col < sequence; col++) {
    const float *kc = k + ((size_t)col * heads + head) * head_dim;
    float partial = 0.f;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x)
      partial = fmaf(q_row[d], kc[d], partial);
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
    const float *vc = v + ((size_t)col * heads + head) * head_dim;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x)
      acc[d] = alpha * acc[d] + beta * vc[d];
    m_i = m_new;
    l_i = l_new;
    __syncthreads();
  }
  float inv = 1.f / l_i;
  float *o = out + ((size_t)row * heads + head) * head_dim;
  for (uint32_t d = tid; d < head_dim; d += blockDim.x) o[d] = acc[d] * inv;
}

int h3_gpu_sdpa_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                    const h3_gpu_tensor *value, uint32_t sequence,
                    uint32_t heads, uint32_t head_dim, float scale) {
  if (!gpu || !output || !query || !key || !value) return 0;
  size_t n = (size_t)sequence * heads * head_dim;
  if (query->elements < n || key->elements < n || value->elements < n ||
      output->elements < n)
    return 0;
  if (head_dim > 256) {
    h3_cuda_vset_error(gpu, "sdpa_f32 head_dim > 256 unsupported");
    return 0;
  }
  if (!h3_cuda_require_encoding(gpu)) return 0;
  dim3 grid(sequence, heads);
  int threads = 128;
  size_t shmem = (size_t)(2 * head_dim + threads) * sizeof(float);
  k_sdpa_f32_online<<<grid, threads, shmem, gpu->stream>>>(
      (const float *)query->device, (const float *)key->device,
      (const float *)value->device, (float *)output->device, sequence, heads,
      head_dim, scale);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "sdpa_f32")) return 0;
  gpu->stats.mps_sdpa_dispatches++;
  return 1;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
