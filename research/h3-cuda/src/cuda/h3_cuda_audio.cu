/* AudioVAE activations + attention helpers; vision QKV RoPE. */
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

__device__ inline int h3_clamp_i(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

__global__ static void k_snake1d_f32(const float *input, const float *alpha,
                                     float *output, uint32_t batch,
                                     uint32_t length, uint32_t channels) {
  size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = (size_t)batch * length * channels;
  if (gid >= n) return;
  float a = alpha[gid % channels];
  float x = input[gid];
  float wave = sinf(a * x);
  output[gid] = x + wave * wave / (a + 1e-9f);
}

__global__ static void k_alias_free_snake_f32(
    const float *input, const float *alpha_log, const float *beta_log,
    const float *upsample_filter, const float *downsample_filter, float *output,
    uint32_t batch, uint32_t length, uint32_t channels) {
  uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t time = blockIdx.y;
  uint32_t b = blockIdx.z;
  if (channel >= channels || time >= length || b >= batch) return;
  float alpha = expf(alpha_log[channel]);
  float beta = expf(beta_log[channel]);
  float result = 0.f;
  for (int down_k = 0; down_k < 12; down_k++) {
    int up_time = (int)(time * 2) + down_k - 5;
    up_time = h3_clamp_i(up_time, 0, (int)(length * 2) - 1);
    int raw_time = up_time + 15;
    float upsampled = 0.f;
    for (int up_k = 0; up_k < 12; up_k++) {
      int numerator = raw_time - up_k;
      if (numerator < 0 || (numerator & 1)) continue;
      int padded_time = numerator / 2;
      int source_time = h3_clamp_i(padded_time - 5, 0, (int)length - 1);
      size_t source =
          ((size_t)b * length + (size_t)source_time) * channels + channel;
      upsampled = fmaf(input[source], 2.f * upsample_filter[up_k], upsampled);
    }
    float sine = sinf(alpha * upsampled);
    float activated = upsampled + sine * sine / (beta + 1e-9f);
    result = fmaf(activated, downsample_filter[down_k], result);
  }
  size_t dest = ((size_t)b * length + time) * channels + channel;
  output[dest] = result;
}

__global__ static void k_audio_qkv_split_f32(
    const float *qkv, const float *q_bias, const float *k_bias,
    const float *v_bias, float *query, float *key, float *value, uint32_t batch,
    uint32_t length, uint32_t heads, uint32_t head_dim) {
  size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t width = heads * head_dim;
  size_t count = (size_t)batch * length * width;
  if (gid >= count) return;
  uint32_t column = (uint32_t)(gid % width);
  size_t row = gid / width;
  size_t base = row * width * 3;
  query[gid] = qkv[base + column] + q_bias[column];
  key[gid] = qkv[base + width + column] + k_bias[column];
  value[gid] = qkv[base + width * 2 + column] + v_bias[column];
}

/* Layout [batch, seq, heads, dim]; causal online softmax. */
__global__ static void k_sdpa_causal_f32_online(
    const float *q, const float *k, const float *v, float *out, uint32_t batch,
    uint32_t sequence, uint32_t heads, uint32_t head_dim, float scale) {
  uint32_t head = blockIdx.y;
  uint32_t row = blockIdx.x;
  uint32_t b = blockIdx.z;
  if (head >= heads || row >= sequence || b >= batch) return;
  extern __shared__ float smem[];
  float *q_row = smem;
  float *acc = smem + head_dim;
  float *red = smem + 2 * head_dim;
  uint32_t tid = threadIdx.x;
  size_t q_base =
      (((size_t)b * sequence + row) * heads + head) * head_dim;
  for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
    q_row[d] = q[q_base + d];
    acc[d] = 0.f;
  }
  __syncthreads();
  float m_i = -INFINITY;
  float l_i = 0.f;
  for (uint32_t col = 0; col <= row; col++) {
    size_t k_base =
        (((size_t)b * sequence + col) * heads + head) * head_dim;
    float partial = 0.f;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x)
      partial = fmaf(q_row[d], k[k_base + d], partial);
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
    size_t v_base =
        (((size_t)b * sequence + col) * heads + head) * head_dim;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x)
      acc[d] = alpha * acc[d] + beta * v[v_base + d];
    m_i = m_new;
    l_i = l_new;
    __syncthreads();
  }
  float inv = 1.f / l_i;
  for (uint32_t d = tid; d < head_dim; d += blockDim.x)
    out[q_base + d] = acc[d] * inv;
}

__global__ static void k_audio_attention_pool_f32(
    const float *attended, float *output, uint32_t batch, uint32_t length,
    uint32_t heads, uint32_t head_dim, uint32_t output_dim) {
  size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t count = (size_t)batch * length * output_dim;
  if (gid >= count) return;
  uint32_t column = (uint32_t)(gid % output_dim);
  size_t row = gid / output_dim;
  uint32_t pool = head_dim / output_dim;
  float sum = 0.f;
  for (uint32_t head = 0; head < heads; head++) {
    size_t base =
        (row * heads + head) * head_dim + (size_t)column * pool;
    for (uint32_t item = 0; item < pool; item++) sum += attended[base + item];
  }
  output[gid] = sum / (float)(heads * pool);
}

__global__ static void k_vision_qkv_rope_bf16(
    const __nv_bfloat16 *qkv, const __nv_bfloat16 *rope_cos,
    const __nv_bfloat16 *rope_sin, __nv_bfloat16 *query, __nv_bfloat16 *key,
    __nv_bfloat16 *value, uint32_t sequence, uint32_t heads, uint32_t head_dim,
    uint32_t rope_half) {
  uint32_t dimension = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t head = blockIdx.y;
  uint32_t row = blockIdx.z;
  if (dimension >= head_dim || head >= heads || row >= sequence) return;
  uint32_t inner = heads * head_dim;
  size_t row_base = (size_t)row * inner * 3;
  size_t q_base = row_base + (size_t)head * head_dim;
  size_t k_base = row_base + inner + (size_t)head * head_dim;
  size_t v_base = row_base + (size_t)inner * 2 + (size_t)head * head_dim;
  uint32_t half_dim = rope_half;
  size_t rope_index = (size_t)row * half_dim + dimension % half_dim;
  float c = h3_bf16_to_f32(rope_cos[rope_index]);
  float s = h3_bf16_to_f32(rope_sin[rope_index]);
  uint32_t pair =
      dimension < half_dim ? dimension + half_dim : dimension - half_dim;
  float q0 = h3_bf16_to_f32(qkv[q_base + dimension]);
  float k0 = h3_bf16_to_f32(qkv[k_base + dimension]);
  float q1 = h3_bf16_to_f32(qkv[q_base + pair]);
  float k1 = h3_bf16_to_f32(qkv[k_base + pair]);
  float qr = dimension < half_dim ? q0 * c - q1 * s : q0 * c + q1 * s;
  float kr = dimension < half_dim ? k0 * c - k1 * s : k0 * c + k1 * s;
  size_t out_i = ((size_t)row * heads + head) * head_dim + dimension;
  query[out_i] = h3_f32_to_bf16(qr);
  key[out_i] = h3_f32_to_bf16(kr);
  value[out_i] = qkv[v_base + dimension];
}

int h3_gpu_snake1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input, const h3_gpu_tensor *alpha,
                       uint32_t batch, uint32_t length, uint32_t channels) {
  if (!gpu || !output || !input || !alpha) return 0;
  size_t n = (size_t)batch * length * channels;
  if (output->dtype != H3_GPU_F32 || input->dtype != H3_GPU_F32 ||
      alpha->dtype != H3_GPU_F32 || output->elements < n ||
      input->elements < n || alpha->elements < channels)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  int thr = 256;
  int blk = (int)((n + thr - 1) / thr);
  k_snake1d_f32<<<blk, thr, 0, gpu->stream>>>(
      (const float *)input->device, (const float *)alpha->device,
      (float *)output->device, batch, length, channels);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "snake1d")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_alias_free_snake_f32(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *alpha_log, const h3_gpu_tensor *beta_log,
    const h3_gpu_tensor *upsample_filter, const h3_gpu_tensor *downsample_filter,
    uint32_t batch, uint32_t length, uint32_t channels) {
  if (!gpu || !output || !input || !alpha_log || !beta_log || !upsample_filter ||
      !downsample_filter)
    return 0;
  size_t n = (size_t)batch * length * channels;
  if (output->dtype != H3_GPU_F32 || input->dtype != H3_GPU_F32 ||
      output->elements < n || input->elements < n ||
      alpha_log->elements < channels || beta_log->elements < channels ||
      upsample_filter->elements < 12 || downsample_filter->elements < 12)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  dim3 block(64);
  dim3 grid((channels + 63) / 64, length, batch);
  k_alias_free_snake_f32<<<grid, block, 0, gpu->stream>>>(
      (const float *)input->device, (const float *)alpha_log->device,
      (const float *)beta_log->device, (const float *)upsample_filter->device,
      (const float *)downsample_filter->device, (float *)output->device, batch,
      length, channels);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "alias_free_snake")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_audio_qkv_split_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                               h3_gpu_tensor *key, h3_gpu_tensor *value,
                               const h3_gpu_tensor *qkv,
                               const h3_gpu_tensor *q_bias,
                               const h3_gpu_tensor *k_bias,
                               const h3_gpu_tensor *v_bias, uint32_t batch,
                               uint32_t length, uint32_t heads,
                               uint32_t head_dim) {
  if (!gpu || !query || !key || !value || !qkv || !q_bias || !k_bias || !v_bias)
    return 0;
  uint32_t width = heads * head_dim;
  size_t n = (size_t)batch * length * width;
  if (query->elements < n || key->elements < n || value->elements < n ||
      qkv->elements < n * 3 || q_bias->elements < width ||
      k_bias->elements < width || v_bias->elements < width)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  int thr = 256;
  int blk = (int)((n + thr - 1) / thr);
  k_audio_qkv_split_f32<<<blk, thr, 0, gpu->stream>>>(
      (const float *)qkv->device, (const float *)q_bias->device,
      (const float *)k_bias->device, (const float *)v_bias->device,
      (float *)query->device, (float *)key->device, (float *)value->device, batch,
      length, heads, head_dim);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "audio_qkv_split")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_sdpa_causal_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value, uint32_t batch,
                           uint32_t sequence, uint32_t heads, uint32_t head_dim,
                           float scale) {
  if (!gpu || !output || !query || !key || !value) return 0;
  size_t n = (size_t)batch * sequence * heads * head_dim;
  if (query->elements < n || key->elements < n || value->elements < n ||
      output->elements < n)
    return 0;
  if (head_dim > 256) {
    h3_cuda_vset_error(gpu, "sdpa_causal head_dim > 256 unsupported");
    return 0;
  }
  if (!h3_cuda_require_encoding(gpu)) return 0;
  int threads = 128;
  size_t shmem = (size_t)(2 * head_dim + threads) * sizeof(float);
  dim3 grid(sequence, heads, batch);
  k_sdpa_causal_f32_online<<<grid, threads, shmem, gpu->stream>>>(
      (const float *)query->device, (const float *)key->device,
      (const float *)value->device, (float *)output->device, batch, sequence,
      heads, head_dim, scale);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "sdpa_causal_f32")) return 0;
  gpu->stats.mps_sdpa_dispatches++;
  return 1;
}

int h3_gpu_audio_attention_pool_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                    const h3_gpu_tensor *attended,
                                    uint32_t batch, uint32_t length,
                                    uint32_t heads, uint32_t head_dim,
                                    uint32_t output_dim) {
  if (!gpu || !output || !attended || !output_dim ||
      (head_dim % output_dim) != 0)
    return 0;
  size_t out_n = (size_t)batch * length * output_dim;
  size_t in_n = (size_t)batch * length * heads * head_dim;
  if (output->elements < out_n || attended->elements < in_n) return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  int thr = 256;
  int blk = (int)((out_n + thr - 1) / thr);
  k_audio_attention_pool_f32<<<blk, thr, 0, gpu->stream>>>(
      (const float *)attended->device, (float *)output->device, batch, length,
      heads, head_dim, output_dim);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "audio_attention_pool")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_vision_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                                h3_gpu_tensor *key, h3_gpu_tensor *value,
                                const h3_gpu_tensor *qkv,
                                const h3_gpu_tensor *rope_cos,
                                const h3_gpu_tensor *rope_sin,
                                uint32_t sequence, uint32_t heads,
                                uint32_t head_dim, uint32_t rope_half) {
  if (!gpu || !query || !key || !value || !qkv || !rope_cos || !rope_sin)
    return 0;
  if (!rope_half || head_dim < rope_half) return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  dim3 block(64);
  dim3 grid((head_dim + 63) / 64, heads, sequence);
  k_vision_qkv_rope_bf16<<<grid, block, 0, gpu->stream>>>(
      (const __nv_bfloat16 *)qkv->device, (const __nv_bfloat16 *)rope_cos->device,
      (const __nv_bfloat16 *)rope_sin->device, (__nv_bfloat16 *)query->device,
      (__nv_bfloat16 *)key->device, (__nv_bfloat16 *)value->device, sequence,
      heads, head_dim, rope_half);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "vision_qkv_rope")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

#ifdef __cplusplus
}
#endif
