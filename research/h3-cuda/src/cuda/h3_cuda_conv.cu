/* AudioVAE Conv1d family — time-major [B,L,C], PyTorch OIK weights. */
#include "h3_cuda_internal.h"

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ static void k_conv1d_stride_f32(
    const float *input, const float *weight, const float *bias, float *output,
    uint32_t batch, uint32_t length, uint32_t in_c, uint32_t out_c,
    uint32_t kernel, uint32_t stride, uint32_t padding, uint32_t dilation,
    uint32_t out_len) {
  uint32_t oc = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t t = blockIdx.y;
  uint32_t b = blockIdx.z;
  if (oc >= out_c || t >= out_len || b >= batch) return;
  float sum = bias ? bias[oc] : 0.f;
  for (uint32_t ic = 0; ic < in_c; ic++) {
    for (uint32_t k = 0; k < kernel; k++) {
      int in_t = (int)(t * stride) - (int)padding + (int)(k * dilation);
      if (in_t < 0 || in_t >= (int)length) continue;
      size_t in_i =
          ((size_t)b * length + (size_t)in_t) * in_c + ic;
      size_t w_i = ((size_t)oc * in_c + ic) * kernel + k;
      sum = fmaf(input[in_i], weight[w_i], sum);
    }
  }
  size_t out_i = ((size_t)b * out_len + t) * out_c + oc;
  output[out_i] = sum;
}

__global__ static void k_conv_transpose1d_f32(
    const float *input, const float *weight, const float *bias, float *output,
    uint32_t batch, uint32_t length, uint32_t in_c, uint32_t out_c,
    uint32_t kernel, uint32_t stride, uint32_t padding, uint32_t out_len) {
  uint32_t oc = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t t = blockIdx.y;
  uint32_t b = blockIdx.z;
  if (oc >= out_c || t >= out_len || b >= batch) return;
  float sum = bias ? bias[oc] : 0.f;
  /* PyTorch ConvTranspose1d weight is IOK: [in_c, out_c, k] */
  for (uint32_t ic = 0; ic < in_c; ic++) {
    for (uint32_t k = 0; k < kernel; k++) {
      /* out_t = in_t * stride + k - padding  →  in_t = (out_t + padding - k) / stride */
      int numer = (int)t + (int)padding - (int)k;
      if (numer < 0 || (numer % (int)stride) != 0) continue;
      int in_t = numer / (int)stride;
      if (in_t < 0 || in_t >= (int)length) continue;
      size_t in_i = ((size_t)b * length + (size_t)in_t) * in_c + ic;
      size_t w_i = ((size_t)ic * out_c + oc) * kernel + k;
      sum = fmaf(input[in_i], weight[w_i], sum);
    }
  }
  size_t out_i = ((size_t)b * out_len + t) * out_c + oc;
  output[out_i] = sum;
}

__global__ static void k_weight_norm_f32(const float *vector,
                                         const float *magnitude, float *output,
                                         uint32_t outer, uint32_t inner) {
  uint32_t o = blockIdx.x;
  if (o >= outer) return;
  extern __shared__ float red[];
  uint32_t tid = threadIdx.x;
  const float *row = vector + (size_t)o * inner;
  float local = 0.f;
  for (uint32_t i = tid; i < inner; i += blockDim.x)
    local = fmaf(row[i], row[i], local);
  red[tid] = local;
  __syncthreads();
  for (uint32_t s = blockDim.x / 2; s; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  float scale = magnitude[o] * rsqrtf(red[0]);
  float *out = output + (size_t)o * inner;
  for (uint32_t i = tid; i < inner; i += blockDim.x) out[i] = row[i] * scale;
}

int h3_gpu_conv1d_stride_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t batch,
                             uint32_t length, uint32_t input_channels,
                             uint32_t output_channels, uint32_t kernel,
                             uint32_t stride, uint32_t padding,
                             uint32_t dilation) {
  if (!gpu || !output || !input || !weight || !batch || !length ||
      !input_channels || !output_channels || !kernel || !stride)
    return 0;
  uint64_t numer =
      (uint64_t)length + 2ull * padding - dilation * (kernel - 1ull) - 1ull;
  if (numer >= UINT32_MAX) return 0;
  uint32_t out_len = (uint32_t)(numer / stride) + 1u;
  size_t in_n = (size_t)batch * length * input_channels;
  size_t w_n = (size_t)output_channels * input_channels * kernel;
  size_t out_n = (size_t)batch * out_len * output_channels;
  if (input->dtype != H3_GPU_F32 || weight->dtype != H3_GPU_F32 ||
      output->dtype != H3_GPU_F32 || input->elements < in_n ||
      weight->elements < w_n || output->elements < out_n)
    return 0;
  if (bias && (bias->dtype != H3_GPU_F32 || bias->elements < output_channels))
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  dim3 block(64);
  dim3 grid((output_channels + 63) / 64, out_len, batch);
  k_conv1d_stride_f32<<<grid, block, 0, gpu->stream>>>(
      (const float *)input->device, (const float *)weight->device,
      bias ? (const float *)bias->device : nullptr, (float *)output->device,
      batch, length, input_channels, output_channels, kernel, stride, padding,
      dilation, out_len);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "conv1d_stride")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_conv1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t padding, uint32_t dilation) {
  return h3_gpu_conv1d_stride_f32(gpu, output, input, weight, bias, batch,
                                  length, input_channels, output_channels,
                                  kernel, 1, padding, dilation);
}

int h3_gpu_conv_transpose1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                const h3_gpu_tensor *input,
                                const h3_gpu_tensor *weight,
                                const h3_gpu_tensor *bias, uint32_t batch,
                                uint32_t length, uint32_t input_channels,
                                uint32_t output_channels, uint32_t kernel,
                                uint32_t stride, uint32_t padding) {
  if (!gpu || !output || !input || !weight || !batch || !length ||
      !input_channels || !output_channels || !kernel || !stride)
    return 0;
  if ((uint64_t)(length - 1) * stride + kernel < 2ull * padding) return 0;
  uint32_t out_len =
      (uint32_t)((uint64_t)(length - 1) * stride + kernel - 2ull * padding);
  size_t in_n = (size_t)batch * length * input_channels;
  size_t w_n = (size_t)input_channels * output_channels * kernel;
  size_t out_n = (size_t)batch * out_len * output_channels;
  if (input->dtype != H3_GPU_F32 || weight->dtype != H3_GPU_F32 ||
      output->dtype != H3_GPU_F32 || input->elements < in_n ||
      weight->elements < w_n || output->elements < out_n)
    return 0;
  if (bias && (bias->dtype != H3_GPU_F32 || bias->elements < output_channels))
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  dim3 block(64);
  dim3 grid((output_channels + 63) / 64, out_len, batch);
  k_conv_transpose1d_f32<<<grid, block, 0, gpu->stream>>>(
      (const float *)input->device, (const float *)weight->device,
      bias ? (const float *)bias->device : nullptr, (float *)output->device,
      batch, length, input_channels, output_channels, kernel, stride, padding,
      out_len);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "conv_transpose1d")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_weight_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *vector,
                           const h3_gpu_tensor *magnitude, uint32_t outer,
                           uint32_t inner) {
  if (!gpu || !output || !vector || !magnitude || !outer || !inner) return 0;
  size_t n = (size_t)outer * inner;
  if (output->dtype != H3_GPU_F32 || vector->dtype != H3_GPU_F32 ||
      magnitude->dtype != H3_GPU_F32 || output->elements < n ||
      vector->elements < n || magnitude->elements < outer)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  int threads = 256;
  size_t shmem = (size_t)threads * sizeof(float);
  k_weight_norm_f32<<<outer, threads, shmem, gpu->stream>>>(
      (const float *)vector->device, (const float *)magnitude->device,
      (float *)output->device, outer, inner);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "weight_norm")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

#ifdef __cplusplus
}
#endif
