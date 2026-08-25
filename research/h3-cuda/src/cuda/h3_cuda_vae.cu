/* Video VAE encoder: channels-last pad, Conv3d, group-norm+SiLU. */
#include "h3_cuda_internal.h"

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

__device__ inline int h3_reflect_coordinate(int coordinate, int length) {
  if (coordinate < 0) return -coordinate;
  if (coordinate >= length) return 2 * length - coordinate - 2;
  return coordinate;
}

__global__ static void k_vae_encoder_pad_f32(
    const float *input, float *output, uint32_t batch, uint32_t depth,
    uint32_t height, uint32_t width, uint32_t channels, uint32_t depth_front,
    uint32_t height_before, uint32_t height_after, uint32_t width_before,
    uint32_t width_after) {
  uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t out_x = blockIdx.y;
  uint32_t plane = blockIdx.z;
  uint32_t out_height = height + height_before + height_after;
  uint32_t out_width = width + width_before + width_after;
  uint32_t out_depth = depth + depth_front;
  if (channel >= channels || out_x >= out_width ||
      plane >= batch * out_depth * out_height)
    return;
  uint32_t out_y = plane % out_height;
  uint32_t temporal_plane = plane / out_height;
  uint32_t out_t = temporal_plane % out_depth;
  uint32_t b = temporal_plane / out_depth;
  size_t destination =
      ((((size_t)b * out_depth + out_t) * out_height + out_y) * out_width +
       out_x) *
          channels +
      channel;
  if (out_t < depth_front) {
    output[destination] = 0.f;
    return;
  }
  int source_y =
      h3_reflect_coordinate((int)out_y - (int)height_before, (int)height);
  int source_x =
      h3_reflect_coordinate((int)out_x - (int)width_before, (int)width);
  uint32_t source_t = out_t - depth_front;
  size_t source =
      ((((size_t)b * depth + source_t) * height + (uint32_t)source_y) * width +
       (uint32_t)source_x) *
          channels +
      channel;
  output[destination] = input[source];
}

__global__ static void k_vae_group_norm_silu_f32(
    const float *input, const float *weight, const float *bias, float *output,
    uint32_t batch, uint32_t depth, uint32_t height, uint32_t width,
    uint32_t channels, uint32_t groups, float epsilon) {
  uint32_t row = blockIdx.x;
  uint32_t rows = batch * depth * groups;
  if (row >= rows) return;
  extern __shared__ float red[];
  uint32_t tid = threadIdx.x;
  uint32_t channels_per_group = channels / groups;
  uint32_t group_index = row % groups;
  uint32_t temporal_plane = row / groups;
  uint32_t elements = height * width * channels_per_group;

  float local = 0.f;
  for (uint32_t index = tid; index < elements; index += blockDim.x) {
    uint32_t spatial = index / channels_per_group;
    uint32_t channel =
        group_index * channels_per_group + index % channels_per_group;
    size_t source =
        ((size_t)temporal_plane * height * width + spatial) * channels +
        channel;
    local += input[source];
  }
  red[tid] = local;
  __syncthreads();
  for (uint32_t s = blockDim.x / 2; s; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  float mean = red[0] / (float)elements;
  __syncthreads();
  local = 0.f;
  for (uint32_t index = tid; index < elements; index += blockDim.x) {
    uint32_t spatial = index / channels_per_group;
    uint32_t channel =
        group_index * channels_per_group + index % channels_per_group;
    size_t source =
        ((size_t)temporal_plane * height * width + spatial) * channels +
        channel;
    float centered = input[source] - mean;
    local = fmaf(centered, centered, local);
  }
  red[tid] = local;
  __syncthreads();
  for (uint32_t s = blockDim.x / 2; s; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / (float)elements + epsilon);
  for (uint32_t index = tid; index < elements; index += blockDim.x) {
    uint32_t spatial = index / channels_per_group;
    uint32_t channel =
        group_index * channels_per_group + index % channels_per_group;
    size_t destination =
        ((size_t)temporal_plane * height * width + spatial) * channels +
        channel;
    float value =
        (input[destination] - mean) * inv * weight[channel] + bias[channel];
    output[destination] = value / (1.f + expf(-value));
  }
}

/* Channels-last NDHWC + OIHWKD weights [oc,ic,kd,kh,kw]. */
__global__ static void k_conv3d_f32(
    const float *input, const float *weight, const float *bias, float *output,
    uint32_t batch, uint32_t depth, uint32_t height, uint32_t width,
    uint32_t in_c, uint32_t out_c, uint32_t kd, uint32_t kh, uint32_t kw,
    uint32_t sd, uint32_t sh, uint32_t sw, uint32_t od, uint32_t oh,
    uint32_t ow) {
  uint32_t oc = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t ox = blockIdx.y % ow;
  uint32_t oy = blockIdx.y / ow;
  uint32_t plane = blockIdx.z;
  if (oc >= out_c || oy >= oh || plane >= batch * od) return;
  uint32_t ot = plane % od;
  uint32_t b = plane / od;
  float sum = bias ? bias[oc] : 0.f;
  for (uint32_t ic = 0; ic < in_c; ic++) {
    for (uint32_t zd = 0; zd < kd; zd++) {
      uint32_t id = ot * sd + zd;
      for (uint32_t yh = 0; yh < kh; yh++) {
        uint32_t iy = oy * sh + yh;
        for (uint32_t xw = 0; xw < kw; xw++) {
          uint32_t ix = ox * sw + xw;
          size_t in_i =
              ((((size_t)b * depth + id) * height + iy) * width + ix) * in_c +
              ic;
          size_t w_i =
              ((((size_t)oc * in_c + ic) * kd + zd) * kh + yh) * kw + xw;
          sum = fmaf(input[in_i], weight[w_i], sum);
        }
      }
    }
  }
  size_t out_i =
      ((((size_t)b * od + ot) * oh + oy) * ow + ox) * out_c + oc;
  output[out_i] = sum;
}

int h3_gpu_vae_encoder_pad_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                               const h3_gpu_tensor *input, uint32_t batch,
                               uint32_t depth, uint32_t height, uint32_t width,
                               uint32_t channels, uint32_t depth_front,
                               uint32_t height_before, uint32_t height_after,
                               uint32_t width_before, uint32_t width_after) {
  if (!gpu || !output || !input || !batch || !depth || height < 2 || width < 2 ||
      !channels)
    return 0;
  if (height_before >= height || height_after >= height ||
      width_before >= width || width_after >= width)
    return 0;
  uint32_t out_depth = depth + depth_front;
  uint32_t out_height = height + height_before + height_after;
  uint32_t out_width = width + width_before + width_after;
  size_t in_n = (size_t)batch * depth * height * width * channels;
  size_t out_n =
      (size_t)batch * out_depth * out_height * out_width * channels;
  if (input->dtype != H3_GPU_F32 || output->dtype != H3_GPU_F32 ||
      input->elements < in_n || output->elements < out_n)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  dim3 block(64);
  dim3 grid((channels + 63) / 64, out_width,
            batch * out_depth * out_height);
  k_vae_encoder_pad_f32<<<grid, block, 0, gpu->stream>>>(
      (const float *)input->device, (float *)output->device, batch, depth,
      height, width, channels, depth_front, height_before, height_after,
      width_before, width_after);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "vae_encoder_pad")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_vae_encoder_group_norm_silu_f32(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch,
    uint32_t depth, uint32_t height, uint32_t width, uint32_t channels,
    uint32_t groups, float epsilon) {
  if (!gpu || !output || !input || !weight || !bias || !batch || !depth ||
      !height || !width || !channels || !groups || (channels % groups) != 0 ||
      !(epsilon > 0.f))
    return 0;
  size_t n = (size_t)batch * depth * height * width * channels;
  if (input->dtype != H3_GPU_F32 || output->dtype != H3_GPU_F32 ||
      weight->dtype != H3_GPU_F32 || bias->dtype != H3_GPU_F32 ||
      input->elements < n || output->elements < n ||
      weight->elements < channels || bias->elements < channels)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  uint32_t rows = batch * depth * groups;
  int threads = 256;
  size_t shmem = (size_t)threads * sizeof(float);
  k_vae_group_norm_silu_f32<<<rows, threads, shmem, gpu->stream>>>(
      (const float *)input->device, (const float *)weight->device,
      (const float *)bias->device, (float *)output->device, batch, depth, height,
      width, channels, groups, epsilon);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "vae_group_norm_silu")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_conv3d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch, uint32_t depth,
                      uint32_t height, uint32_t width, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel_depth,
                      uint32_t kernel_height, uint32_t kernel_width,
                      uint32_t stride_depth, uint32_t stride_height,
                      uint32_t stride_width) {
  if (!gpu || !output || !input || !weight || !batch || !depth || !height ||
      !width || !input_channels || !output_channels || !kernel_depth ||
      !kernel_height || !kernel_width || !stride_depth || !stride_height ||
      !stride_width)
    return 0;
  if (depth < kernel_depth || height < kernel_height || width < kernel_width)
    return 0;
  uint32_t od = (depth - kernel_depth) / stride_depth + 1;
  uint32_t oh = (height - kernel_height) / stride_height + 1;
  uint32_t ow = (width - kernel_width) / stride_width + 1;
  size_t in_n =
      (size_t)batch * depth * height * width * input_channels;
  size_t w_n = (size_t)output_channels * input_channels * kernel_depth *
               kernel_height * kernel_width;
  size_t out_n =
      (size_t)batch * od * oh * ow * output_channels;
  if (input->dtype != H3_GPU_F32 || weight->dtype != H3_GPU_F32 ||
      output->dtype != H3_GPU_F32 || input->elements < in_n ||
      weight->elements < w_n || output->elements < out_n)
    return 0;
  if (bias && (bias->dtype != H3_GPU_F32 || bias->elements < output_channels))
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  dim3 block(64);
  dim3 grid((output_channels + 63) / 64, oh * ow, batch * od);
  k_conv3d_f32<<<grid, block, 0, gpu->stream>>>(
      (const float *)input->device, (const float *)weight->device,
      bias ? (const float *)bias->device : nullptr, (float *)output->device,
      batch, depth, height, width, input_channels, output_channels, kernel_depth,
      kernel_height, kernel_width, stride_depth, stride_height, stride_width, od,
      oh, ow);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "conv3d_f32")) return 0;
  gpu->stats.mps_conv_dispatches++;
  return 1;
}

#ifdef __cplusplus
}
#endif
