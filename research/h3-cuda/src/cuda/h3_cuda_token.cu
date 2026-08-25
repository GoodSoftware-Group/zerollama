/* Token-reduction fusions: pool+AdaLN and expand(+AdaLN). */
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

/* One reduced row per block: pool → residual + optional baseline, then AdaLN. */
__global__ static void k_token_pool_adaln_bf16(
    const __nv_bfloat16 *input, const uint32_t *pairs, __nv_bfloat16 *residual,
    __nv_bfloat16 *baseline, const uint32_t *baseline_indices,
    __nv_bfloat16 *original, const __nv_bfloat16 *weight,
    const __nv_bfloat16 *modulation, const uint32_t *row_map,
    __nv_bfloat16 *output, size_t input_offset, size_t original_offset,
    size_t baseline_offset, uint32_t rows, uint32_t width, uint32_t slots,
    uint32_t shift_slot, uint32_t scale_slot, float epsilon) {
  uint32_t row = blockIdx.x;
  if (row >= rows) return;
  extern __shared__ float smem[];
  float *red = smem;
  /* pooled values as float scratch after reduction buffer */
  float *pooled_f = smem + blockDim.x;
  uint32_t tid = threadIdx.x;
  uint32_t left = pairs[row * 2];
  uint32_t right = pairs[row * 2 + 1];
  uint32_t bi = baseline_indices[row];

  float local = 0.f;
  for (uint32_t c = tid; c < width; c += blockDim.x) {
    __nv_bfloat16 first =
        input[input_offset + (size_t)left * width + c];
    original[original_offset + (size_t)left * width + c] = first;
    float pf = h3_bf16_to_f32(first);
    if (left != right) {
      __nv_bfloat16 second =
          input[input_offset + (size_t)right * width + c];
      original[original_offset + (size_t)right * width + c] = second;
      pf = 0.5f * (pf + h3_bf16_to_f32(second));
    }
    __nv_bfloat16 pooled = h3_f32_to_bf16(pf);
    residual[(size_t)row * width + c] = pooled;
    pooled_f[c] = h3_bf16_to_f32(pooled);
    if (bi != 0xffffffffu)
      baseline[baseline_offset + (size_t)bi * width + c] = pooled;
    local = fmaf(pooled_f[c], pooled_f[c], local);
  }
  red[tid] = local;
  __syncthreads();
  for (uint32_t s = blockDim.x / 2; s; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / (float)width + epsilon);
  size_t base = (size_t)row_map[row] * slots * width;
  for (uint32_t c = tid; c < width; c += blockDim.x) {
    float n = pooled_f[c] * inv * h3_bf16_to_f32(weight[c]);
    float shift =
        h3_bf16_to_f32(modulation[base + (size_t)shift_slot * width + c]);
    float scale =
        h3_bf16_to_f32(modulation[base + (size_t)scale_slot * width + c]);
    output[(size_t)row * width + c] = h3_f32_to_bf16(n * (1.f + scale) + shift);
  }
}

__global__ static void k_token_expand_delta_bf16(
    const __nv_bfloat16 *original, const __nv_bfloat16 *reduced,
    const __nv_bfloat16 *baseline, const uint32_t *baseline_indices,
    const uint32_t *parents, __nv_bfloat16 *output, size_t original_offset,
    size_t baseline_offset, uint32_t rows, uint32_t width,
    uint32_t exact_prefix_rows, float update_scale) {
  uint32_t row = blockIdx.y;
  uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || col >= width) return;
  uint32_t parent = parents[row];
  size_t dest = (size_t)row * width + col;
  size_t red_i = (size_t)parent * width + col;
  if (row < exact_prefix_rows) {
    output[dest] = reduced[red_i];
    return;
  }
  uint32_t baseline_row = baseline_indices[parent];
  if (baseline_row == 0xffffffffu) {
    output[dest] = reduced[red_i];
    return;
  }
  float update = h3_bf16_to_f32(reduced[red_i]) -
                 h3_bf16_to_f32(
                     baseline[baseline_offset + (size_t)baseline_row * width +
                              col]);
  output[dest] = h3_f32_to_bf16(
      h3_bf16_to_f32(original[original_offset + dest]) +
      update_scale * update);
}

__global__ static void k_token_expand_adaln_bf16(
    const __nv_bfloat16 *original, const __nv_bfloat16 *reduced,
    const __nv_bfloat16 *baseline, const uint32_t *baseline_indices,
    const uint32_t *parents, __nv_bfloat16 *residual,
    const __nv_bfloat16 *weight, const __nv_bfloat16 *modulation,
    const uint32_t *row_map, __nv_bfloat16 *output, size_t original_offset,
    size_t baseline_offset, uint32_t rows, uint32_t width,
    uint32_t exact_prefix_rows, float update_scale, uint32_t slots,
    uint32_t shift_slot, uint32_t scale_slot, float epsilon) {
  uint32_t row = blockIdx.x;
  if (row >= rows) return;
  extern __shared__ float smem[];
  float *red = smem;
  float *restored_f = smem + blockDim.x;
  uint32_t tid = threadIdx.x;
  uint32_t parent = parents[row];
  uint32_t baseline_row = baseline_indices[parent];
  int direct = (row < exact_prefix_rows) || (baseline_row == 0xffffffffu);

  float local = 0.f;
  for (uint32_t c = tid; c < width; c += blockDim.x) {
    size_t dest = (size_t)row * width + c;
    size_t red_i = (size_t)parent * width + c;
    float rv = h3_bf16_to_f32(reduced[red_i]);
    if (!direct) {
      float update =
          rv - h3_bf16_to_f32(
                   baseline[baseline_offset + (size_t)baseline_row * width + c]);
      rv = h3_bf16_to_f32(original[original_offset + dest]) +
           update_scale * update;
    }
    __nv_bfloat16 restored = h3_f32_to_bf16(rv);
    residual[dest] = restored;
    restored_f[c] = h3_bf16_to_f32(restored);
    local = fmaf(restored_f[c], restored_f[c], local);
  }
  red[tid] = local;
  __syncthreads();
  for (uint32_t s = blockDim.x / 2; s; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / (float)width + epsilon);
  size_t base = (size_t)row_map[row] * slots * width;
  for (uint32_t c = tid; c < width; c += blockDim.x) {
    float n = restored_f[c] * inv * h3_bf16_to_f32(weight[c]);
    float shift =
        h3_bf16_to_f32(modulation[base + (size_t)shift_slot * width + c]);
    float scale =
        h3_bf16_to_f32(modulation[base + (size_t)scale_slot * width + c]);
    output[(size_t)row * width + c] = h3_f32_to_bf16(n * (1.f + scale) + shift);
  }
}

int h3_gpu_token_pool_adaln_bf16(
    h3_gpu *gpu, h3_gpu_tensor *residual, h3_gpu_tensor *output,
    const h3_gpu_tensor *input, size_t input_offset, h3_gpu_tensor *original,
    size_t original_offset, h3_gpu_tensor *baseline, size_t baseline_offset,
    const h3_gpu_tensor *baseline_indices, const h3_gpu_tensor *pairs,
    const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation,
    const h3_gpu_tensor *row_map, uint32_t input_rows, uint32_t rows,
    uint32_t baseline_rows, uint32_t width, uint32_t slots, uint32_t shift_slot,
    uint32_t scale_slot, float epsilon) {
  (void)input_rows;
  (void)baseline_rows;
  if (!gpu || !residual || !output || !input || !original || !baseline ||
      !baseline_indices || !pairs || !norm_weight || !modulation || !row_map)
    return 0;
  if (width > 8192 || shift_slot >= slots || scale_slot >= slots) return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  int threads = 256;
  size_t shmem = (size_t)(threads + width) * sizeof(float);
  k_token_pool_adaln_bf16<<<rows, threads, shmem, gpu->stream>>>(
      (const __nv_bfloat16 *)input->device, (const uint32_t *)pairs->device,
      (__nv_bfloat16 *)residual->device, (__nv_bfloat16 *)baseline->device,
      (const uint32_t *)baseline_indices->device,
      (__nv_bfloat16 *)original->device, (const __nv_bfloat16 *)norm_weight->device,
      (const __nv_bfloat16 *)modulation->device, (const uint32_t *)row_map->device,
      (__nv_bfloat16 *)output->device, input_offset, original_offset,
      baseline_offset, rows, width, slots, shift_slot, scale_slot, epsilon);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "token_pool_adaln")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_token_expand_delta_bf16(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *original,
    size_t original_offset, const h3_gpu_tensor *reduced,
    const h3_gpu_tensor *baseline, size_t baseline_offset,
    const h3_gpu_tensor *baseline_indices, const h3_gpu_tensor *parents,
    uint32_t rows, uint32_t reduced_rows, uint32_t baseline_rows, uint32_t width,
    uint32_t exact_prefix_rows, float update_scale) {
  (void)reduced_rows;
  (void)baseline_rows;
  if (!gpu || !output || !original || !reduced || !baseline ||
      !baseline_indices || !parents)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  dim3 block(256);
  dim3 grid((width + 255) / 256, rows);
  k_token_expand_delta_bf16<<<grid, block, 0, gpu->stream>>>(
      (const __nv_bfloat16 *)original->device,
      (const __nv_bfloat16 *)reduced->device,
      (const __nv_bfloat16 *)baseline->device,
      (const uint32_t *)baseline_indices->device, (const uint32_t *)parents->device,
      (__nv_bfloat16 *)output->device, original_offset, baseline_offset, rows,
      width, exact_prefix_rows, update_scale);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "token_expand_delta")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_token_expand_adaln_bf16(
    h3_gpu *gpu, h3_gpu_tensor *residual, h3_gpu_tensor *output,
    const h3_gpu_tensor *original, size_t original_offset,
    const h3_gpu_tensor *reduced, const h3_gpu_tensor *baseline,
    size_t baseline_offset, const h3_gpu_tensor *baseline_indices,
    const h3_gpu_tensor *parents, const h3_gpu_tensor *norm_weight,
    const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map,
    uint32_t rows, uint32_t reduced_rows, uint32_t baseline_rows, uint32_t width,
    uint32_t exact_prefix_rows, float update_scale, uint32_t slots,
    uint32_t shift_slot, uint32_t scale_slot, float epsilon) {
  (void)reduced_rows;
  (void)baseline_rows;
  if (!gpu || !residual || !output || !original || !reduced || !baseline ||
      !baseline_indices || !parents || !norm_weight || !modulation || !row_map)
    return 0;
  if (width > 8192 || shift_slot >= slots || scale_slot >= slots) return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  int threads = 256;
  size_t shmem = (size_t)(threads + width) * sizeof(float);
  k_token_expand_adaln_bf16<<<rows, threads, shmem, gpu->stream>>>(
      (const __nv_bfloat16 *)original->device,
      (const __nv_bfloat16 *)reduced->device,
      (const __nv_bfloat16 *)baseline->device,
      (const uint32_t *)baseline_indices->device, (const uint32_t *)parents->device,
      (__nv_bfloat16 *)residual->device, (const __nv_bfloat16 *)norm_weight->device,
      (const __nv_bfloat16 *)modulation->device, (const uint32_t *)row_map->device,
      (__nv_bfloat16 *)output->device, original_offset, baseline_offset, rows,
      width, exact_prefix_rows, update_scale, slots, shift_slot, scale_slot,
      epsilon);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "token_expand_adaln")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

#ifdef __cplusplus
}
#endif
