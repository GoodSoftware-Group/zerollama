#pragma once
#include <cuda_bf16.h>
#include <math.h>

__device__ inline float h3_bf16_to_f32(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

__device__ inline __nv_bfloat16 h3_f32_to_bf16(float v) {
  return __float2bfloat16(v);
}

/* One row per block: RMSNorm * weight. */
__global__ void k_rms_norm_bf16(const __nv_bfloat16 *input,
                                const __nv_bfloat16 *weight,
                                __nv_bfloat16 *output, uint32_t rows,
                                uint32_t width, float epsilon) {
  uint32_t row = blockIdx.x;
  if (row >= rows) return;
  extern __shared__ float red[];
  uint32_t tid = threadIdx.x;
  const __nv_bfloat16 *x = input + (size_t)row * width;
  float local = 0.f;
  for (uint32_t k = tid; k < width; k += blockDim.x) {
    float v = h3_bf16_to_f32(x[k]);
    local = fmaf(v, v, local);
  }
  red[tid] = local;
  __syncthreads();
  for (uint32_t s = blockDim.x / 2; s; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / (float)width + epsilon);
  for (uint32_t c = tid; c < width; c += blockDim.x) {
    float n = h3_bf16_to_f32(x[c]) * inv * h3_bf16_to_f32(weight[c]);
    output[(size_t)row * width + c] = h3_f32_to_bf16(n);
  }
}

/* AdaLN: rms * w * (1+scale) + shift; modulation layout [map][slots][width]. */
__global__ void k_adaln_bf16(const __nv_bfloat16 *input,
                             const __nv_bfloat16 *weight,
                             const __nv_bfloat16 *modulation,
                             const uint32_t *row_map, __nv_bfloat16 *output,
                             uint32_t rows, uint32_t width, uint32_t slots,
                             uint32_t shift_slot, uint32_t scale_slot,
                             float epsilon) {
  uint32_t row = blockIdx.x;
  if (row >= rows) return;
  extern __shared__ float red[];
  uint32_t tid = threadIdx.x;
  const __nv_bfloat16 *x = input + (size_t)row * width;
  float local = 0.f;
  for (uint32_t k = tid; k < width; k += blockDim.x) {
    float v = h3_bf16_to_f32(x[k]);
    local = fmaf(v, v, local);
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
    float n = h3_bf16_to_f32(x[c]) * inv * h3_bf16_to_f32(weight[c]);
    float shift =
        h3_bf16_to_f32(modulation[base + (size_t)shift_slot * width + c]);
    float scale =
        h3_bf16_to_f32(modulation[base + (size_t)scale_slot * width + c]);
    output[(size_t)row * width + c] = h3_f32_to_bf16(n * (1.f + scale) + shift);
  }
}

__global__ void k_gate_bf16(const __nv_bfloat16 *residual,
                            const __nv_bfloat16 *branch,
                            const __nv_bfloat16 *modulation,
                            const uint32_t *row_map, __nv_bfloat16 *output,
                            uint32_t rows, uint32_t width, uint32_t slots,
                            uint32_t gate_slot) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = (size_t)rows * width;
  if (idx >= n) return;
  uint32_t row = (uint32_t)(idx / width);
  uint32_t col = (uint32_t)(idx % width);
  size_t base = (size_t)row_map[row] * slots * width;
  float gate =
      h3_bf16_to_f32(modulation[base + (size_t)gate_slot * width + col]);
  float v = h3_bf16_to_f32(residual[idx]) + h3_bf16_to_f32(branch[idx]) * gate;
  output[idx] = h3_f32_to_bf16(v);
}

/* fused = [gate | up] per row, width each → silu(gate)*up */
__global__ void k_swiglu_bf16(const __nv_bfloat16 *fused, __nv_bfloat16 *output,
                              uint32_t rows, uint32_t width) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = (size_t)rows * width;
  if (idx >= n) return;
  uint32_t row = (uint32_t)(idx / width);
  uint32_t col = (uint32_t)(idx % width);
  size_t base = (size_t)row * width * 2;
  float gate = h3_bf16_to_f32(fused[base + col]);
  float up = h3_bf16_to_f32(fused[base + width + col]);
  float silu = gate / (1.f + expf(-gate));
  output[idx] = h3_f32_to_bf16(silu * up);
}

__global__ void k_add_bias_bf16(__nv_bfloat16 *inout, const __nv_bfloat16 *bias,
                                uint32_t rows, uint32_t width) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = (size_t)rows * width;
  if (idx >= n) return;
  uint32_t col = (uint32_t)(idx % width);
  float v = h3_bf16_to_f32(inout[idx]) + h3_bf16_to_f32(bias[col]);
  inout[idx] = h3_f32_to_bf16(v);
}
