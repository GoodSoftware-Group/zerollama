#include "h3_cuda_internal.h"

#include <cuda_bf16.h>

#include "kernels/epilogue.cuh"

#ifdef __cplusplus
extern "C" {
#endif


/* Y[rows, out] = X[rows, in] @ W[out, in]^T (+ bias) — matches Metal/MPS layout. */
int h3_gpu_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                       const h3_gpu_tensor *bias, uint32_t rows,
                       uint32_t input_dim, uint32_t output_dim) {
  if (!gpu || !output || !input || !weight) return 0;
  size_t in_n = (size_t)rows * input_dim;
  size_t w_n = (size_t)output_dim * input_dim;
  size_t out_n = (size_t)rows * output_dim;
  if (input->dtype != H3_GPU_BF16 || weight->dtype != H3_GPU_BF16 ||
      output->dtype != H3_GPU_BF16 || input->elements < in_n ||
      weight->elements < w_n || output->elements < out_n)
    return 0;
  if (bias && (bias->dtype != H3_GPU_BF16 || bias->elements < output_dim))
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;

  const float alpha = 1.f, beta = 0.f;
  /* Row-major C = A @ B^T via cublas column-major trick:
   * cublas: C_cm = op(A_cm) @ op(B_cm) with dims n,m,k → C is m×n row-major. */
  cublasStatus_t st = cublasGemmEx(
      gpu->cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)output_dim, (int)rows,
      (int)input_dim, &alpha, weight->device, CUDA_R_16BF, (int)input_dim,
      input->device, CUDA_R_16BF, (int)input_dim, &beta, output->device,
      CUDA_R_16BF, (int)output_dim, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (!h3_cuda_cublas_check(gpu, st, "linear_bf16 gemm")) return 0;

  if (bias) {
    int threads = 256;
    size_t n = out_n;
    int blocks = (int)((n + threads - 1) / threads);
    k_add_bias_bf16<<<blocks, threads, 0, gpu->stream>>>(
        (__nv_bfloat16 *)output->device, (const __nv_bfloat16 *)bias->device,
        rows, output_dim);
    if (!h3_cuda_check(gpu, cudaGetLastError(), "linear bias")) return 0;
  }
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_swiglu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *fused, uint32_t rows,
                       uint32_t width) {
  if (!gpu || !output || !fused) return 0;
  if (fused->elements < (size_t)rows * width * 2 ||
      output->elements < (size_t)rows * width)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  size_t n = (size_t)rows * width;
  int threads = 256;
  int blocks = (int)((n + threads - 1) / threads);
  k_swiglu_bf16<<<blocks, threads, 0, gpu->stream>>>(
      (const __nv_bfloat16 *)fused->device, (__nv_bfloat16 *)output->device, rows,
      width);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "swiglu_bf16")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_mlp_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, const h3_gpu_tensor *fc1_weight,
                    const h3_gpu_tensor *fc2_weight, uint32_t rows,
                    uint32_t input_dim, uint32_t hidden_dim,
                    uint32_t output_dim) {
  if (!gpu) return 0;
  h3_gpu_tensor *fused =
      h3_gpu_tensor_new_bf16(gpu, (size_t)rows * hidden_dim * 2);
  h3_gpu_tensor *act = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * hidden_dim);
  if (!fused || !act) {
    h3_gpu_tensor_free(fused);
    h3_gpu_tensor_free(act);
    return 0;
  }
  int ok = h3_gpu_linear_bf16(gpu, fused, input, fc1_weight, NULL, rows,
                              input_dim, hidden_dim * 2) &&
           h3_gpu_swiglu_bf16(gpu, act, fused, rows, hidden_dim) &&
           h3_gpu_linear_bf16(gpu, output, act, fc2_weight, NULL, rows,
                              hidden_dim, output_dim);
  h3_gpu_tensor_free(fused);
  h3_gpu_tensor_free(act);
  return ok;
}

int h3_gpu_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon) {
  if (!gpu || !output || !input || !weight) return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  int threads = 256;
  size_t shmem = (size_t)threads * sizeof(float);
  k_rms_norm_bf16<<<rows, threads, shmem, gpu->stream>>>(
      (const __nv_bfloat16 *)input->device, (const __nv_bfloat16 *)weight->device,
      (__nv_bfloat16 *)output->device, rows, width, epsilon);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "rms_norm_bf16")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon) {
  if (!gpu || !output || !input || !norm_weight || !modulation || !row_map)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  int threads = 256;
  size_t shmem = (size_t)threads * sizeof(float);
  k_adaln_bf16<<<rows, threads, shmem, gpu->stream>>>(
      (const __nv_bfloat16 *)input->device,
      (const __nv_bfloat16 *)norm_weight->device,
      (const __nv_bfloat16 *)modulation->device, (const uint32_t *)row_map->device,
      (__nv_bfloat16 *)output->device, rows, width, slots, shift_slot, scale_slot,
      epsilon);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "adaln_bf16")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_gate_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual, const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot) {
  if (!gpu || !output || !residual || !branch || !modulation || !row_map)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  size_t n = (size_t)rows * width;
  int threads = 256;
  int blocks = (int)((n + threads - 1) / threads);
  k_gate_bf16<<<blocks, threads, 0, gpu->stream>>>(
      (const __nv_bfloat16 *)residual->device,
      (const __nv_bfloat16 *)branch->device,
      (const __nv_bfloat16 *)modulation->device, (const uint32_t *)row_map->device,
      (__nv_bfloat16 *)output->device, rows, width, slots, gate_slot);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "gate_bf16")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}


int h3_gpu_adaln_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input, size_t input_offset,
                             const h3_gpu_tensor *norm_weight,
                             const h3_gpu_tensor *modulation,
                             const h3_gpu_tensor *row_map, uint32_t rows,
                             uint32_t width, uint32_t slots, uint32_t shift_slot,
                             uint32_t scale_slot, float epsilon) {
  if (!gpu || !output || !input || !norm_weight || !modulation || !row_map)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  const __nv_bfloat16 *in =
      (const __nv_bfloat16 *)input->device + input_offset;
  int threads = 256;
  size_t shmem = (size_t)threads * sizeof(float);
  k_adaln_bf16<<<rows, threads, shmem, gpu->stream>>>(
      in, (const __nv_bfloat16 *)norm_weight->device,
      (const __nv_bfloat16 *)modulation->device, (const uint32_t *)row_map->device,
      (__nv_bfloat16 *)output->device, rows, width, slots, shift_slot, scale_slot,
      epsilon);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "adaln_bf16_offset")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

/* ---- F32 linear + F32→BF16 patch projection (Metal parity) ---- */

__global__ static void k_add_bias_f32(float *inout, const float *bias,
                                      uint32_t rows, uint32_t width) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = (size_t)rows * width;
  if (idx >= n) return;
  inout[idx] += bias[idx % width];
}

__global__ static void k_cast_f32_to_bf16_range(const float *in,
                                               __nv_bfloat16 *out, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16(in[i]);
}

__global__ static void k_scatter_cast_f32_to_bf16(const float *tmp,
                                                  __nv_bfloat16 *out,
                                                  const uint32_t *row_map,
                                                  uint32_t rows,
                                                  uint32_t out_dim) {
  uint32_t r = blockIdx.y;
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows || c >= out_dim) return;
  out[(size_t)row_map[r] * out_dim + c] =
      __float2bfloat16(tmp[(size_t)r * out_dim + c]);
}

__global__ static void k_rms_inverse_bf16(const __nv_bfloat16 *input,
                                          float *inverse, uint32_t rows,
                                          uint32_t width, float epsilon) {
  uint32_t row = blockIdx.x;
  if (row >= rows) return;
  extern __shared__ float inv_red[];
  uint32_t tid = threadIdx.x;
  const __nv_bfloat16 *x = input + (size_t)row * width;
  float local = 0.f;
  for (uint32_t k = tid; k < width; k += blockDim.x) {
    float v = __bfloat162float(x[k]);
    local = fmaf(v, v, local);
  }
  inv_red[tid] = local;
  __syncthreads();
  for (uint32_t s = blockDim.x / 2; s; s >>= 1) {
    if (tid < s) inv_red[tid] += inv_red[tid + s];
    __syncthreads();
  }
  if (tid == 0) inverse[row] = rsqrtf(inv_red[0] / (float)width + epsilon);
}

static int h3_cuda_sgemm_nn_t(h3_gpu *gpu, float *out, const float *in,
                              const float *weight, uint32_t rows,
                              uint32_t input_dim, uint32_t output_dim) {
  const float alpha = 1.f, beta = 0.f;
  /* C = A @ B^T  (row-major) via cublas column-major trick. */
  cublasStatus_t st =
      cublasSgemm(gpu->cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)output_dim,
                  (int)rows, (int)input_dim, &alpha, weight, (int)input_dim, in,
                  (int)input_dim, &beta, out, (int)output_dim);
  return h3_cuda_cublas_check(gpu, st, "sgemm");
}

int h3_gpu_linear_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t input_dim, uint32_t output_dim) {
  if (!gpu || !output || !input || !weight) return 0;
  size_t in_n = (size_t)rows * input_dim;
  size_t w_n = (size_t)output_dim * input_dim;
  size_t out_n = (size_t)rows * output_dim;
  if (input->dtype != H3_GPU_F32 || weight->dtype != H3_GPU_F32 ||
      output->dtype != H3_GPU_F32 || input->elements < in_n ||
      weight->elements < w_n || output->elements < out_n)
    return 0;
  if (bias && (bias->dtype != H3_GPU_F32 || bias->elements < output_dim))
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  if (!h3_cuda_sgemm_nn_t(gpu, (float *)output->device, (const float *)input->device,
                          (const float *)weight->device, rows, input_dim,
                          output_dim))
    return 0;
  if (bias) {
    int threads = 256;
    int blocks = (int)((out_n + threads - 1) / threads);
    k_add_bias_f32<<<blocks, threads, 0, gpu->stream>>>(
        (float *)output->device, (const float *)bias->device, rows, output_dim);
    if (!h3_cuda_check(gpu, cudaGetLastError(), "linear_f32 bias")) return 0;
  }
  gpu->stats.direct_dispatches++;
  return 1;
}

/* Metal: F32 input/weight (+ F32 bias) → BF16 output. */
int h3_gpu_patch_linear_bf16_offset(
    h3_gpu *gpu, h3_gpu_tensor *output, size_t output_offset,
    const h3_gpu_tensor *input, size_t input_offset, const h3_gpu_tensor *weight,
    const h3_gpu_tensor *bias, uint32_t rows, uint32_t input_dim,
    uint32_t output_dim) {
  if (!gpu || !output || !input || !weight) return 0;
  size_t in_n = (size_t)rows * input_dim;
  size_t w_n = (size_t)output_dim * input_dim;
  size_t out_n = (size_t)rows * output_dim;
  if (input->dtype != H3_GPU_F32 || weight->dtype != H3_GPU_F32 ||
      output->dtype != H3_GPU_BF16)
    return 0;
  if (input_offset + in_n > input->elements || weight->elements < w_n ||
      output_offset + out_n > output->elements)
    return 0;
  if (bias && (bias->dtype != H3_GPU_F32 || bias->elements < output_dim))
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;

  h3_gpu_tensor *tmp = h3_gpu_tensor_new_f32(gpu, out_n);
  if (!tmp) return 0;
  const float *in = (const float *)input->device + input_offset;
  int ok = h3_cuda_sgemm_nn_t(gpu, (float *)tmp->device, in,
                              (const float *)weight->device, rows, input_dim,
                              output_dim);
  if (ok && bias) {
    int threads = 256;
    int blocks = (int)((out_n + threads - 1) / threads);
    k_add_bias_f32<<<blocks, threads, 0, gpu->stream>>>(
        (float *)tmp->device, (const float *)bias->device, rows, output_dim);
    ok = h3_cuda_check(gpu, cudaGetLastError(), "patch f32 bias");
  }
  if (ok) {
    int threads = 256;
    int blocks = (int)((out_n + threads - 1) / threads);
    __nv_bfloat16 *out =
        (__nv_bfloat16 *)output->device + output_offset;
    k_cast_f32_to_bf16_range<<<blocks, threads, 0, gpu->stream>>>(
        (const float *)tmp->device, out, out_n);
    ok = h3_cuda_check(gpu, cudaGetLastError(), "patch cast bf16");
  }
  h3_gpu_tensor_free(tmp);
  if (!ok) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_patch_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
  return h3_gpu_patch_linear_bf16_offset(gpu, output, 0, input, 0, weight, bias,
                                         rows, input_dim, output_dim);
}

int h3_gpu_patch_linear_bf16_map(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
    const h3_gpu_tensor *row_map, uint32_t output_rows, uint32_t rows,
    uint32_t input_dim, uint32_t output_dim) {
  (void)output_rows;
  if (!gpu || !output || !input || !weight || !row_map) return 0;
  size_t in_n = (size_t)rows * input_dim;
  size_t w_n = (size_t)output_dim * input_dim;
  size_t out_n = (size_t)rows * output_dim;
  if (input->dtype != H3_GPU_F32 || weight->dtype != H3_GPU_F32 ||
      output->dtype != H3_GPU_BF16 || row_map->dtype != H3_GPU_U32)
    return 0;
  if (input->elements < in_n || weight->elements < w_n ||
      row_map->elements < rows)
    return 0;
  if (bias && (bias->dtype != H3_GPU_F32 || bias->elements < output_dim))
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;

  h3_gpu_tensor *tmp = h3_gpu_tensor_new_f32(gpu, out_n);
  if (!tmp) return 0;
  int ok = h3_cuda_sgemm_nn_t(gpu, (float *)tmp->device,
                              (const float *)input->device,
                              (const float *)weight->device, rows, input_dim,
                              output_dim);
  if (ok && bias) {
    int threads = 256;
    int blocks = (int)((out_n + threads - 1) / threads);
    k_add_bias_f32<<<blocks, threads, 0, gpu->stream>>>(
        (float *)tmp->device, (const float *)bias->device, rows, output_dim);
    ok = h3_cuda_check(gpu, cudaGetLastError(), "patch_map bias");
  }
  if (ok) {
    dim3 block(256);
    dim3 grid((output_dim + 255) / 256, rows);
    k_scatter_cast_f32_to_bf16<<<grid, block, 0, gpu->stream>>>(
        (const float *)tmp->device, (__nv_bfloat16 *)output->device,
        (const uint32_t *)row_map->device, rows, output_dim);
    ok = h3_cuda_check(gpu, cudaGetLastError(), "patch_map scatter");
  }
  h3_gpu_tensor_free(tmp);
  if (!ok) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

/* Compose: RMS inverse + AdaLN (BF16) + linear_bf16 — Metal fused final head. */
int h3_gpu_adaln_linear_bf16(
    h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *inverse,
    const h3_gpu_tensor *input, size_t input_offset,
    const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation,
    const h3_gpu_tensor *row_map, const h3_gpu_tensor *weight,
    const h3_gpu_tensor *bias, uint32_t rows, uint32_t width,
    uint32_t output_dim, uint32_t slots, uint32_t shift_slot,
    uint32_t scale_slot, float epsilon) {
  if (!gpu || !output || !inverse || !input || !norm_weight || !modulation ||
      !row_map || !weight)
    return 0;
  if (inverse->dtype != H3_GPU_F32 || inverse->elements < rows) return 0;
  if (shift_slot >= slots || scale_slot >= slots) return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;

  const __nv_bfloat16 *in =
      (const __nv_bfloat16 *)input->device + input_offset;
  int threads = 256;
  size_t shmem = (size_t)threads * sizeof(float);
  k_rms_inverse_bf16<<<rows, threads, shmem, gpu->stream>>>(
      in, (float *)inverse->device, rows, width, epsilon);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "rms_inverse")) return 0;

  h3_gpu_tensor *normed = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * width);
  if (!normed) return 0;
  k_adaln_bf16<<<rows, threads, shmem, gpu->stream>>>(
      in, (const __nv_bfloat16 *)norm_weight->device,
      (const __nv_bfloat16 *)modulation->device, (const uint32_t *)row_map->device,
      (__nv_bfloat16 *)normed->device, rows, width, slots, shift_slot, scale_slot,
      epsilon);
  int ok = h3_cuda_check(gpu, cudaGetLastError(), "adaln_linear adaln");
  if (ok)
    ok = h3_gpu_linear_bf16(gpu, output, normed, weight, bias, rows, width,
                            output_dim);
  h3_gpu_tensor_free(normed);
  return ok;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
