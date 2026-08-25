/* P2 shared elementwise / norm ops (VAE + vision helpers). */
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

__global__ static void k_silu_f32(const float *in, float *out, uint32_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = in[i];
  out[i] = x / (1.f + expf(-x));
}

__global__ static void k_silu_bf16(const __nv_bfloat16 *in, __nv_bfloat16 *out,
                                   uint32_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = h3_bf16_to_f32(in[i]);
  out[i] = h3_f32_to_bf16(x / (1.f + expf(-x)));
}

__global__ static void k_gelu_bf16(const __nv_bfloat16 *in, __nv_bfloat16 *out,
                                   uint32_t n, int approximate) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = h3_bf16_to_f32(in[i]);
  float y;
  if (approximate) {
    float cube = x * x * x;
    y = 0.5f * x *
        (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * cube)));
  } else {
    y = 0.5f * x * (1.f + erff(x * 0.7071067811865476f));
  }
  out[i] = h3_f32_to_bf16(y);
}

__global__ static void k_clip_f32(const float *in, float *out, uint32_t n,
                                  float lo, float hi) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = fminf(hi, fmaxf(lo, in[i]));
}

__global__ static void k_add_scaled_f32(const float *left, const float *right,
                                        float *out, float ls, float rs,
                                        uint32_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = left[i] * ls + right[i] * rs;
}

__global__ static void k_scale_add_f32(const float *residual, const float *branch,
                                       const float *scale, float *out,
                                       uint32_t rows, uint32_t width) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = (size_t)rows * width;
  if (idx >= n) return;
  uint32_t col = (uint32_t)(idx % width);
  out[idx] = residual[idx] + branch[idx] * scale[col];
}

__global__ static void k_geglu_f32(const float *gate, const float *linear,
                                   float *out, uint32_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = gate[i];
  float cube = x * x * x;
  float gelu = 0.5f * x *
               (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * cube)));
  out[i] = gelu * linear[i];
}

__global__ static void k_swiglu_f32(const float *fused, float *out, uint32_t rows,
                                   uint32_t width) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = (size_t)rows * width;
  if (idx >= n) return;
  uint32_t row = (uint32_t)(idx / width);
  uint32_t col = (uint32_t)(idx % width);
  size_t base = (size_t)row * width * 2;
  float gate = fused[base + col];
  float up = fused[base + width + col];
  out[idx] = (gate / (1.f + expf(-gate))) * up;
}

__global__ static void k_rms_norm_f32(const float *input, const float *weight,
                                     float *output, uint32_t rows, uint32_t width,
                                     float epsilon) {
  uint32_t row = blockIdx.x;
  if (row >= rows) return;
  extern __shared__ float rms_red[];
  uint32_t tid = threadIdx.x;
  const float *x = input + (size_t)row * width;
  float local = 0.f;
  for (uint32_t k = tid; k < width; k += blockDim.x)
    local = fmaf(x[k], x[k], local);
  rms_red[tid] = local;
  __syncthreads();
  for (uint32_t s = blockDim.x / 2; s; s >>= 1) {
    if (tid < s) rms_red[tid] += rms_red[tid + s];
    __syncthreads();
  }
  float inv = rsqrtf(rms_red[0] / (float)width + epsilon);
  for (uint32_t c = tid; c < width; c += blockDim.x)
    output[(size_t)row * width + c] = x[c] * inv * weight[c];
}

__global__ static void k_layer_norm_f32(const float *input, const float *weight,
                                        const float *bias, float *output,
                                        uint32_t rows, uint32_t width,
                                        float epsilon) {
  uint32_t row = blockIdx.x;
  if (row >= rows) return;
  extern __shared__ float ln_red[];
  uint32_t tid = threadIdx.x;
  const float *x = input + (size_t)row * width;
  float local = 0.f;
  for (uint32_t k = tid; k < width; k += blockDim.x) local += x[k];
  ln_red[tid] = local;
  __syncthreads();
  for (uint32_t s = blockDim.x / 2; s; s >>= 1) {
    if (tid < s) ln_red[tid] += ln_red[tid + s];
    __syncthreads();
  }
  float mean = ln_red[0] / (float)width;
  __syncthreads();
  local = 0.f;
  for (uint32_t k = tid; k < width; k += blockDim.x) {
    float c = x[k] - mean;
    local = fmaf(c, c, local);
  }
  ln_red[tid] = local;
  __syncthreads();
  for (uint32_t s = blockDim.x / 2; s; s >>= 1) {
    if (tid < s) ln_red[tid] += ln_red[tid + s];
    __syncthreads();
  }
  float inv = rsqrtf(ln_red[0] / (float)width + epsilon);
  for (uint32_t c = tid; c < width; c += blockDim.x)
    output[(size_t)row * width + c] =
        (x[c] - mean) * inv * weight[c] + bias[c];
}

__global__ static void k_layer_norm_bf16(const __nv_bfloat16 *input,
                                         const __nv_bfloat16 *weight,
                                         const __nv_bfloat16 *bias,
                                         __nv_bfloat16 *output, uint32_t rows,
                                         uint32_t width, float epsilon) {
  uint32_t row = blockIdx.x;
  if (row >= rows) return;
  extern __shared__ float lnb_red[];
  uint32_t tid = threadIdx.x;
  const __nv_bfloat16 *x = input + (size_t)row * width;
  float local = 0.f;
  for (uint32_t k = tid; k < width; k += blockDim.x)
    local += h3_bf16_to_f32(x[k]);
  lnb_red[tid] = local;
  __syncthreads();
  for (uint32_t s = blockDim.x / 2; s; s >>= 1) {
    if (tid < s) lnb_red[tid] += lnb_red[tid + s];
    __syncthreads();
  }
  float mean = lnb_red[0] / (float)width;
  __syncthreads();
  local = 0.f;
  for (uint32_t k = tid; k < width; k += blockDim.x) {
    float c = h3_bf16_to_f32(x[k]) - mean;
    local = fmaf(c, c, local);
  }
  lnb_red[tid] = local;
  __syncthreads();
  for (uint32_t s = blockDim.x / 2; s; s >>= 1) {
    if (tid < s) lnb_red[tid] += lnb_red[tid + s];
    __syncthreads();
  }
  float inv = rsqrtf(lnb_red[0] / (float)width + epsilon);
  for (uint32_t c = tid; c < width; c += blockDim.x) {
    float y = (h3_bf16_to_f32(x[c]) - mean) * inv * h3_bf16_to_f32(weight[c]) +
              h3_bf16_to_f32(bias[c]);
    output[(size_t)row * width + c] = h3_f32_to_bf16(y);
  }
}

/* Token pool: pairs are packed U32 [left,right] per reduced row. */
__global__ static void k_token_pool_bf16(
    const __nv_bfloat16 *input, const uint32_t *pairs, __nv_bfloat16 *output,
    __nv_bfloat16 *baseline, const uint32_t *baseline_indices,
    __nv_bfloat16 *original, size_t input_offset, size_t original_offset,
    size_t baseline_offset, uint32_t rows, uint32_t width) {
  uint32_t row = blockIdx.y;
  uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || col >= width) return;
  uint32_t left = pairs[row * 2];
  uint32_t right = pairs[row * 2 + 1];
  __nv_bfloat16 first =
      input[input_offset + (size_t)left * width + col];
  original[original_offset + (size_t)left * width + col] = first;
  __nv_bfloat16 pooled = first;
  if (left != right) {
    __nv_bfloat16 second =
        input[input_offset + (size_t)right * width + col];
    original[original_offset + (size_t)right * width + col] = second;
    float avg =
        (h3_bf16_to_f32(first) + h3_bf16_to_f32(second)) * 0.5f;
    pooled = h3_f32_to_bf16(avg);
  }
  output[(size_t)row * width + col] = pooled;
  uint32_t bi = baseline_indices[row];
  if (bi != 0xffffffffu)
    baseline[baseline_offset + (size_t)bi * width + col] = pooled;
}

#define LAUNCH1(fn, n, ...)                                                    \
  do {                                                                         \
    int thr = 256;                                                             \
    int blk = (int)(((n) + thr - 1) / thr);                                    \
    fn<<<blk, thr, 0, gpu->stream>>>(__VA_ARGS__);                             \
  } while (0)

int h3_gpu_silu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements) {
  if (!gpu || !output || !input || output->dtype != H3_GPU_F32 ||
      input->dtype != H3_GPU_F32 || elements > output->elements ||
      elements > input->elements)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  LAUNCH1(k_silu_f32, elements, (const float *)input->device,
          (float *)output->device, elements);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "silu_f32")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_silu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements) {
  if (!gpu || !output || !input || output->dtype != H3_GPU_BF16 ||
      input->dtype != H3_GPU_BF16 || elements > output->elements ||
      elements > input->elements)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  LAUNCH1(k_silu_bf16, elements, (const __nv_bfloat16 *)input->device,
          (__nv_bfloat16 *)output->device, elements);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "silu_bf16")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_gelu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements,
                     int approximate) {
  if (!gpu || !output || !input || output->dtype != H3_GPU_BF16 ||
      input->dtype != H3_GPU_BF16 || elements > output->elements ||
      elements > input->elements)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  LAUNCH1(k_gelu_bf16, elements, (const __nv_bfloat16 *)input->device,
          (__nv_bfloat16 *)output->device, elements, approximate);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "gelu_bf16")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_clip_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements, float minimum,
                    float maximum) {
  if (!gpu || !output || !input || output->dtype != H3_GPU_F32 ||
      input->dtype != H3_GPU_F32 || elements > output->elements ||
      elements > input->elements)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  LAUNCH1(k_clip_f32, elements, (const float *)input->device,
          (float *)output->device, elements, minimum, maximum);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "clip_f32")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_add_scaled_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                          float left_scale, float right_scale,
                          uint32_t elements) {
  if (!gpu || !output || !left || !right || output->dtype != H3_GPU_F32 ||
      left->dtype != H3_GPU_F32 || right->dtype != H3_GPU_F32 ||
      elements > output->elements || elements > left->elements ||
      elements > right->elements)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  LAUNCH1(k_add_scaled_f32, elements, (const float *)left->device,
          (const float *)right->device, (float *)output->device, left_scale,
          right_scale, elements);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "add_scaled_f32")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_scale_add_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *residual,
                         const h3_gpu_tensor *branch, const h3_gpu_tensor *scale,
                         uint32_t rows, uint32_t width) {
  if (!gpu || !output || !residual || !branch || !scale) return 0;
  size_t n = (size_t)rows * width;
  if (output->dtype != H3_GPU_F32 || residual->dtype != H3_GPU_F32 ||
      branch->dtype != H3_GPU_F32 || scale->dtype != H3_GPU_F32 ||
      output->elements < n || residual->elements < n || branch->elements < n ||
      scale->elements < width)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  LAUNCH1(k_scale_add_f32, n, (const float *)residual->device,
          (const float *)branch->device, (const float *)scale->device,
          (float *)output->device, rows, width);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "scale_add_f32")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_geglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *gate, const h3_gpu_tensor *linear,
                     uint32_t elements) {
  if (!gpu || !output || !gate || !linear) return 0;
  if (output->dtype != H3_GPU_F32 || gate->dtype != H3_GPU_F32 ||
      linear->dtype != H3_GPU_F32 || elements > output->elements ||
      elements > gate->elements || elements > linear->elements)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  LAUNCH1(k_geglu_f32, elements, (const float *)gate->device,
          (const float *)linear->device, (float *)output->device, elements);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "geglu_f32")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_swiglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *fused, uint32_t rows,
                      uint32_t width) {
  if (!gpu || !output || !fused) return 0;
  size_t n = (size_t)rows * width;
  if (output->dtype != H3_GPU_F32 || fused->dtype != H3_GPU_F32 ||
      output->elements < n || fused->elements < n * 2)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  LAUNCH1(k_swiglu_f32, n, (const float *)fused->device, (float *)output->device,
          rows, width);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "swiglu_f32")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_rms_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                        uint32_t rows, uint32_t width, float epsilon) {
  if (!gpu || !output || !input || !weight) return 0;
  if (output->dtype != H3_GPU_F32 || input->dtype != H3_GPU_F32 ||
      weight->dtype != H3_GPU_F32)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  int threads = 256;
  size_t shmem = (size_t)threads * sizeof(float);
  k_rms_norm_f32<<<rows, threads, shmem, gpu->stream>>>(
      (const float *)input->device, (const float *)weight->device,
      (float *)output->device, rows, width, epsilon);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "rms_norm_f32")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_layer_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
                          uint32_t rows, uint32_t width, float epsilon) {
  if (!gpu || !output || !input || !weight || !bias) return 0;
  if (output->dtype != H3_GPU_F32 || input->dtype != H3_GPU_F32 ||
      weight->dtype != H3_GPU_F32 || bias->dtype != H3_GPU_F32)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  int threads = 256;
  size_t shmem = (size_t)threads * sizeof(float);
  k_layer_norm_f32<<<rows, threads, shmem, gpu->stream>>>(
      (const float *)input->device, (const float *)weight->device,
      (const float *)bias->device, (float *)output->device, rows, width, epsilon);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "layer_norm_f32")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_layer_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
                           uint32_t rows, uint32_t width, float epsilon) {
  if (!gpu || !output || !input || !weight || !bias) return 0;
  if (output->dtype != H3_GPU_BF16 || input->dtype != H3_GPU_BF16 ||
      weight->dtype != H3_GPU_BF16 || bias->dtype != H3_GPU_BF16)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  int threads = 256;
  size_t shmem = (size_t)threads * sizeof(float);
  k_layer_norm_bf16<<<rows, threads, shmem, gpu->stream>>>(
      (const __nv_bfloat16 *)input->device, (const __nv_bfloat16 *)weight->device,
      (const __nv_bfloat16 *)bias->device, (__nv_bfloat16 *)output->device, rows,
      width, epsilon);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "layer_norm_bf16")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_token_pool_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input, size_t input_offset,
                           h3_gpu_tensor *original, size_t original_offset,
                           h3_gpu_tensor *baseline, size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs, uint32_t input_rows,
                           uint32_t rows, uint32_t baseline_rows,
                           uint32_t width) {
  (void)input_rows;
  (void)baseline_rows;
  if (!gpu || !output || !input || !original || !baseline || !baseline_indices ||
      !pairs)
    return 0;
  if (pairs->dtype != H3_GPU_U32 || baseline_indices->dtype != H3_GPU_U32)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  dim3 block(256);
  dim3 grid((width + 255) / 256, rows);
  k_token_pool_bf16<<<grid, block, 0, gpu->stream>>>(
      (const __nv_bfloat16 *)input->device, (const uint32_t *)pairs->device,
      (__nv_bfloat16 *)output->device, (__nv_bfloat16 *)baseline->device,
      (const uint32_t *)baseline_indices->device,
      (__nv_bfloat16 *)original->device, input_offset, original_offset,
      baseline_offset, rows, width);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "token_pool_bf16")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

#ifdef __cplusplus
}
#endif
