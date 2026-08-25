/* Portable int8 MLP / linear (Metal semantics, not Apple NAX tiles).
 * Weights: one F32 scale per output row. Activations: one F32 scale per row. */
#include "h3_cuda_internal.h"

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

__device__ inline float h3_bf16_to_f32(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
__device__ inline __nv_bfloat16 h3_f32_to_bf16(float v) {
  return __float2bfloat16(v);
}

/* ---- Quantize BF16 → int8 (per-row absmax / 127), pad rows zero ---- */

__global__ static void k_quantize_bf16_int8_rows(
    const __nv_bfloat16 *input, int8_t *output, float *scales, uint32_t rows,
    uint32_t dispatch_rows, uint32_t columns, float clip) {
  uint32_t row = blockIdx.x;
  if (row >= dispatch_rows) return;
  uint32_t tid = threadIdx.x;
  extern __shared__ float red[];
  size_t base = (size_t)row * columns;
  if (row >= rows) {
    for (uint32_t c = tid; c < columns; c += blockDim.x) output[base + c] = 0;
    if (tid == 0) scales[row] = 1.f;
    return;
  }
  float local_max = 0.f;
  for (uint32_t c = tid; c < columns; c += blockDim.x)
    local_max = fmaxf(local_max, fabsf(h3_bf16_to_f32(input[base + c])));
  red[tid] = local_max;
  __syncthreads();
  for (uint32_t s = blockDim.x / 2; s; s >>= 1) {
    if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  float clipped = red[0] * clip;
  float scale = clipped > 0.f ? clipped / 127.f : 1.f / 127.f;
  float inv = clipped > 0.f ? 127.f / clipped : 127.f;
  if (tid == 0) scales[row] = scale;
  for (uint32_t c = tid; c < columns; c += blockDim.x) {
    int q = (int)rintf(h3_bf16_to_f32(input[base + c]) * inv);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    output[base + c] = (int8_t)q;
  }
}

/* Head-major [heads, rows, dim] → row-major int8 [rows, heads*dim]. */
__global__ static void k_quantize_bf16_int8_head_major(
    const __nv_bfloat16 *input, int8_t *output, float *scales, uint32_t rows,
    uint32_t padded_rows, uint32_t heads, uint32_t head_dim, float clip) {
  uint32_t row = blockIdx.x;
  if (row >= padded_rows) return;
  uint32_t tid = threadIdx.x;
  uint32_t columns = heads * head_dim;
  extern __shared__ float red[];
  size_t out_base = (size_t)row * columns;
  if (row >= rows) {
    for (uint32_t c = tid; c < columns; c += blockDim.x)
      output[out_base + c] = 0;
    if (tid == 0) scales[row] = 1.f;
    return;
  }
  float local_max = 0.f;
  for (uint32_t c = tid; c < columns; c += blockDim.x) {
    uint32_t head = c / head_dim;
    uint32_t d = c % head_dim;
    size_t src = ((size_t)head * rows + row) * head_dim + d;
    local_max = fmaxf(local_max, fabsf(h3_bf16_to_f32(input[src])));
  }
  red[tid] = local_max;
  __syncthreads();
  for (uint32_t s = blockDim.x / 2; s; s >>= 1) {
    if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  float clipped = red[0] * clip;
  float scale = clipped > 0.f ? clipped / 127.f : 1.f / 127.f;
  float inv = clipped > 0.f ? 127.f / clipped : 127.f;
  if (tid == 0) scales[row] = scale;
  for (uint32_t c = tid; c < columns; c += blockDim.x) {
    uint32_t head = c / head_dim;
    uint32_t d = c % head_dim;
    size_t src = ((size_t)head * rows + row) * head_dim + d;
    int q = (int)rintf(h3_bf16_to_f32(input[src]) * inv);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    output[out_base + c] = (int8_t)q;
  }
}

static int quantize_rows(h3_gpu *gpu, h3_gpu_tensor *output,
                         h3_gpu_tensor *scales, const h3_gpu_tensor *input,
                         uint32_t rows, uint32_t dispatch_rows,
                         uint32_t columns, float clip) {
  if (!gpu || !output || !scales || !input || !rows || dispatch_rows < rows ||
      !columns)
    return 0;
  if (input->dtype != H3_GPU_BF16 || output->dtype != H3_GPU_I8 ||
      scales->dtype != H3_GPU_F32)
    return 0;
  if (input->elements < (size_t)rows * columns ||
      output->elements < (size_t)dispatch_rows * columns ||
      scales->elements < dispatch_rows)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  int threads = 256;
  size_t shmem = (size_t)threads * sizeof(float);
  k_quantize_bf16_int8_rows<<<dispatch_rows, threads, shmem, gpu->stream>>>(
      (const __nv_bfloat16 *)input->device, (int8_t *)output->device,
      (float *)scales->device, rows, dispatch_rows, columns, clip);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "quantize_int8_rows")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_quantize_weight_int8(h3_gpu *gpu, h3_gpu_tensor *output,
                                h3_gpu_tensor *scales,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t columns) {
  return quantize_rows(gpu, output, scales, input, rows, rows, columns, 1.f);
}

/* ---- Int8 GEMM: Y[r,o] = sum_k A[r,k]*W[o,k] * sa[r]*sw[o] → BF16 ----
 * One block = one input row; threads cover output columns. Shared A tile
 * amortizes activation traffic across many output channels. */

__global__ static void k_linear_int8_bf16(const int8_t *a, const int8_t *w,
                                          const float *sa, const float *sw,
                                          __nv_bfloat16 *out, uint32_t rows,
                                          uint32_t input_dim,
                                          uint32_t output_dim) {
  uint32_t r = blockIdx.x;
  if (r >= rows) return;
  extern __shared__ int8_t a_tile[];
  const int8_t *ar = a + (size_t)r * input_dim;
  float scale_a = sa[r];
  uint32_t tid = threadIdx.x;
  uint32_t nthreads = blockDim.x;

  for (uint32_t o0 = 0; o0 < output_dim; o0 += nthreads) {
    uint32_t o = o0 + tid;
    int32_t acc = 0;
    for (uint32_t k0 = 0; k0 < input_dim; k0 += nthreads) {
      uint32_t k = k0 + tid;
      a_tile[tid] = (k < input_dim) ? ar[k] : 0;
      __syncthreads();
      if (o < output_dim) {
        const int8_t *wo = w + (size_t)o * input_dim + k0;
        uint32_t span = input_dim - k0;
        if (span > nthreads) span = nthreads;
        /* Vectorize when tile is 16B-aligned and full. */
        uint32_t t = 0;
        if (((uintptr_t)wo & 3u) == 0 && (span & 3u) == 0) {
          for (; t + 4 <= span; t += 4) {
            uint32_t wv, av;
            memcpy(&wv, wo + t, 4);
            memcpy(&av, a_tile + t, 4);
            acc += (int32_t)(int8_t)(av & 0xff) * (int32_t)(int8_t)(wv & 0xff);
            acc += (int32_t)(int8_t)((av >> 8) & 0xff) *
                   (int32_t)(int8_t)((wv >> 8) & 0xff);
            acc += (int32_t)(int8_t)((av >> 16) & 0xff) *
                   (int32_t)(int8_t)((wv >> 16) & 0xff);
            acc += (int32_t)(int8_t)((av >> 24) & 0xff) *
                   (int32_t)(int8_t)((wv >> 24) & 0xff);
          }
        }
        for (; t < span; t++) acc += (int32_t)a_tile[t] * (int32_t)wo[t];
      }
      __syncthreads();
    }
    if (o < output_dim) {
      float v = (float)acc * scale_a * sw[o];
      out[(size_t)r * output_dim + o] = h3_f32_to_bf16(v);
    }
  }
}

__global__ static void k_scale_int32_to_bf16(const int32_t *acc, const float *sa,
                                             const float *sw, __nv_bfloat16 *out,
                                             uint32_t rows, uint32_t output_dim) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = (size_t)rows * output_dim;
  if (idx >= n) return;
  uint32_t o = (uint32_t)(idx % output_dim);
  uint32_t r = (uint32_t)(idx / output_dim);
  float v = (float)acc[idx] * sa[r] * sw[o];
  out[idx] = h3_f32_to_bf16(v);
}

static int linear_int8_cublas(h3_gpu *gpu, const int8_t *a, const int8_t *w,
                              const float *sa, const float *sw,
                              __nv_bfloat16 *out, uint32_t rows,
                              uint32_t input_dim, uint32_t output_dim) {
  size_t need = (size_t)rows * output_dim;
  if (need > gpu->int8_acc_elems) {
    if (gpu->int8_acc) {
      cudaFree(gpu->int8_acc);
      gpu->int8_acc = nullptr;
      gpu->int8_acc_elems = 0;
    }
    if (!h3_cuda_check(gpu,
                       cudaMalloc((void **)&gpu->int8_acc,
                                  need * sizeof(int32_t)),
                       "int8 gemm scratch"))
      return 0;
    gpu->int8_acc_elems = need;
  }
  int32_t *acc = gpu->int8_acc;
  int32_t alpha = 1, beta = 0;
  /* Same layout trick as BF16 linear: C_rm[rows,out] via CM gemm. */
  cublasStatus_t st = cublasGemmEx(
      gpu->cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)output_dim, (int)rows,
      (int)input_dim, &alpha, w, CUDA_R_8I, (int)input_dim, a, CUDA_R_8I,
      (int)input_dim, &beta, acc, CUDA_R_32I, (int)output_dim,
      CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (st != CUBLAS_STATUS_SUCCESS) {
    st = cublasGemmEx(gpu->cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)output_dim,
                      (int)rows, (int)input_dim, &alpha, w, CUDA_R_8I,
                      (int)input_dim, a, CUDA_R_8I, (int)input_dim, &beta, acc,
                      CUDA_R_32I, (int)output_dim, CUBLAS_COMPUTE_32I,
                      CUBLAS_GEMM_DEFAULT);
  }
  if (!h3_cuda_cublas_check(gpu, st, "linear_int8 cublas")) return 0;
  int thr = 256;
  int blk = (int)((need + thr - 1) / thr);
  k_scale_int32_to_bf16<<<blk, thr, 0, gpu->stream>>>(acc, sa, sw, out, rows,
                                                      output_dim);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "int8 scale epilogue")) return 0;
  return 1;
}

static int linear_int8_core(h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales, uint32_t rows,
                            uint32_t input_dim, uint32_t output_dim,
                            int head_major, uint32_t heads, uint32_t head_dim,
                            int skip_quantize) {
  if (!gpu || !output || !quantized_input || !input_scales || !weight ||
      !weight_scales)
    return 0;
  if (!rows || !input_dim || !output_dim) return 0;
  uint32_t padded_rows = rows; /* portable path: no forced 128 pad */
  if (weight->dtype != H3_GPU_I8 || weight_scales->dtype != H3_GPU_F32 ||
      output->dtype != H3_GPU_BF16)
    return 0;
  if (weight->elements < (size_t)output_dim * input_dim ||
      weight_scales->elements < output_dim ||
      output->elements < (size_t)rows * output_dim ||
      quantized_input->elements < (size_t)padded_rows * input_dim ||
      input_scales->elements < padded_rows)
    return 0;
  if (!skip_quantize) {
    if (!input) return 0;
    if (head_major) {
      if (heads * head_dim != input_dim) return 0;
      if (!h3_cuda_require_encoding(gpu)) return 0;
      int threads = 256;
      size_t shmem = (size_t)threads * sizeof(float);
      k_quantize_bf16_int8_head_major<<<padded_rows, threads, shmem,
                                        gpu->stream>>>(
          (const __nv_bfloat16 *)input->device, (int8_t *)quantized_input->device,
          (float *)input_scales->device, rows, padded_rows, heads, head_dim,
          1.f);
      if (!h3_cuda_check(gpu, cudaGetLastError(), "quantize head-major"))
        return 0;
      gpu->stats.direct_dispatches++;
    } else if (!quantize_rows(gpu, quantized_input, input_scales, input, rows,
                              padded_rows, input_dim, 1.f)) {
      return 0;
    }
  }
  if (!h3_cuda_require_encoding(gpu)) return 0;

  const int8_t *a = (const int8_t *)quantized_input->device;
  const int8_t *w = (const int8_t *)weight->device;
  const float *sa = (const float *)input_scales->device;
  const float *sw = (const float *)weight_scales->device;
  __nv_bfloat16 *out = (__nv_bfloat16 *)output->device;

  /* cuBLAS int8 for mid/large GEMMs; tiny shapes keep the shared-A kernel. */
  int use_cublas = input_dim >= 64 && output_dim >= 16 && (input_dim % 4u) == 0 &&
                   getenv("H3_DISABLE_CUBLAS_INT8") == nullptr;
  int ok;
  if (use_cublas) {
    ok = linear_int8_cublas(gpu, a, w, sa, sw, out, rows, input_dim, output_dim);
    if (ok) gpu->stats.mps_linear_dispatches++;
  } else {
    int threads = 128;
    if (input_dim >= 512) threads = 256;
    size_t shmem = (size_t)threads * sizeof(int8_t);
    k_linear_int8_bf16<<<rows, threads, shmem, gpu->stream>>>(
        a, w, sa, sw, out, rows, input_dim, output_dim);
    ok = h3_cuda_check(gpu, cudaGetLastError(), "linear_int8_bf16");
    if (ok) gpu->stats.direct_dispatches++;
  }
  return ok;
}

int h3_gpu_linear_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales, uint32_t rows,
                            uint32_t input_dim, uint32_t output_dim,
                            int use_slower_uncached_int8_scales) {
  (void)use_slower_uncached_int8_scales;
  return linear_int8_core(gpu, output, quantized_input, input_scales, input,
                          weight, weight_scales, rows, input_dim, output_dim, 0,
                          0, 0, 0);
}

int h3_gpu_linear_int8_head_major_bf16(
    h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *quantized_input,
    h3_gpu_tensor *input_scales, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales,
    uint32_t rows, uint32_t heads, uint32_t head_dim, uint32_t output_dim) {
  return linear_int8_core(gpu, output, quantized_input, input_scales, input,
                          weight, weight_scales, rows, heads * head_dim,
                          output_dim, 1, heads, head_dim, 0);
}

int h3_gpu_mlp_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         h3_gpu_tensor *activated,
                         h3_gpu_tensor *quantized_activation,
                         h3_gpu_tensor *activation_scales,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *fc1_weight,
                         const h3_gpu_tensor *fc1_scales,
                         const h3_gpu_tensor *fc2_weight,
                         const h3_gpu_tensor *fc2_scales,
                         const h3_gpu_tensor *fc1_bf16,
                         const h3_gpu_tensor *fc2_bf16, uint32_t rows,
                         uint32_t input_dim, uint32_t hidden_dim,
                         uint32_t output_dim, int use_slower_grouped_quantizer,
                         int use_slower_dynamic_fc1_k, int use_int8_row_fc2,
                         int input_is_quantized) {
  (void)use_slower_grouped_quantizer;
  (void)use_slower_dynamic_fc1_k;
  (void)use_int8_row_fc2;
  (void)fc1_bf16;
  (void)fc2_bf16;
  if (!gpu || !output || !activated || !quantized_activation ||
      !activation_scales || !fc1_weight || !fc1_scales || !fc2_weight ||
      !fc2_scales)
    return 0;
  /* FC1 → fused gate|up BF16, then SwiGLU into activated. */
  h3_gpu_tensor *fused =
      h3_gpu_tensor_new_bf16(gpu, (size_t)rows * hidden_dim * 2);
  if (!fused) return 0;
  int ok = 1;
  if (input_is_quantized) {
    ok = linear_int8_core(gpu, fused, quantized_activation, activation_scales,
                          NULL, fc1_weight, fc1_scales, rows, input_dim,
                          hidden_dim * 2, 0, 0, 0, 1);
  } else {
    if (!input) {
      h3_gpu_tensor_free(fused);
      return 0;
    }
    ok = linear_int8_core(gpu, fused, quantized_activation, activation_scales,
                          input, fc1_weight, fc1_scales, rows, input_dim,
                          hidden_dim * 2, 0, 0, 0, 0);
  }
  ok = ok && h3_gpu_swiglu_bf16(gpu, activated, fused, rows, hidden_dim);
  ok = ok && linear_int8_core(gpu, output, quantized_activation,
                              activation_scales, activated, fc2_weight,
                              fc2_scales, rows, hidden_dim, output_dim, 0, 0, 0,
                              0);
  h3_gpu_tensor_free(fused);
  return ok;
}

/* Gate + AdaLN then quantize (compose; Metal fuses into one kernel). */
int h3_gpu_gate_adaln_quantize_int8(
    h3_gpu *gpu, h3_gpu_tensor *gated_residual, h3_gpu_tensor *quantized_output,
    h3_gpu_tensor *quantized_scales, const h3_gpu_tensor *residual,
    const h3_gpu_tensor *branch, const h3_gpu_tensor *norm_weight,
    const h3_gpu_tensor *gate_modulation, const h3_gpu_tensor *norm_modulation,
    const h3_gpu_tensor *row_map, uint32_t rows, uint32_t padded_rows,
    uint32_t width, uint32_t slots, uint32_t gate_slot, uint32_t shift_slot,
    uint32_t scale_slot, float epsilon) {
  if (!gpu || !gated_residual || !quantized_output || !quantized_scales ||
      !residual || !branch || !norm_weight || !gate_modulation ||
      !norm_modulation || !row_map)
    return 0;
  if (padded_rows < rows) return 0;
  h3_gpu_tensor *adaln_out = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * width);
  if (!adaln_out) return 0;
  int ok =
      h3_gpu_gate_bf16(gpu, gated_residual, residual, branch, gate_modulation,
                       row_map, rows, width, slots, gate_slot) &&
      h3_gpu_adaln_bf16(gpu, adaln_out, gated_residual, norm_weight,
                        norm_modulation, row_map, rows, width, slots, shift_slot,
                        scale_slot, epsilon) &&
      quantize_rows(gpu, quantized_output, quantized_scales, adaln_out, rows,
                    padded_rows, width, 1.f);
  h3_gpu_tensor_free(adaln_out);
  return ok;
}

int h3_gpu_grouped_qkv_linear_rope_int8(
    h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value,
    h3_gpu_tensor *quantized_input, h3_gpu_tensor *input_scales,
    const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
    const h3_gpu_tensor *weight_scales, const h3_gpu_tensor *q_norm,
    const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos,
    const h3_gpu_tensor *rope_sin, uint32_t rows, uint32_t input_dim,
    uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon,
    int input_is_quantized, int use_slower_unfused_qkv_rope,
    int use_slower_scalar_qkv_rms, int use_slower_uncached_int8_scales) {
  (void)use_slower_unfused_qkv_rope;
  (void)use_slower_scalar_qkv_rms;
  (void)use_slower_uncached_int8_scales;
  if (!gpu || !query || !key || !value || !quantized_input || !input_scales ||
      !weight || !weight_scales || !q_norm || !k_norm || !rope_cos || !rope_sin)
    return 0;
  uint32_t inner = heads * head_dim;
  h3_gpu_tensor *qkv = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * inner * 3);
  if (!qkv) return 0;
  int ok;
  if (input_is_quantized) {
    ok = linear_int8_core(gpu, qkv, quantized_input, input_scales, NULL, weight,
                          weight_scales, rows, input_dim, inner * 3, 0, 0, 0, 1);
  } else {
    if (!input) {
      h3_gpu_tensor_free(qkv);
      return 0;
    }
    ok = linear_int8_core(gpu, qkv, quantized_input, input_scales, input, weight,
                          weight_scales, rows, input_dim, inner * 3, 0, 0, 0, 0);
  }
  ok = ok && h3_gpu_grouped_qkv_rope_bf16(gpu, query, key, value, qkv, q_norm,
                                          k_norm, rope_cos, rope_sin, rows,
                                          heads, head_dim, rope_half, epsilon);
  h3_gpu_tensor_free(qkv);
  return ok;
}

/* Apple NAX BF16 MLP — not available on CUDA; keep stub semantics via caller
 * checking h3_gpu_has_nax_mlp()==0. Provide a BF16 fallback if invoked. */
int h3_gpu_mlp_nax_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                        h3_gpu_tensor *activated, const h3_gpu_tensor *input,
                        const h3_gpu_tensor *fc1_weight,
                        const h3_gpu_tensor *fc2_weight, uint32_t rows,
                        uint32_t input_dim, uint32_t hidden_dim,
                        uint32_t output_dim) {
  (void)activated;
  /* Fallback: ordinary BF16 MLP (allocates its own temps). */
  return h3_gpu_mlp_bf16(gpu, output, input, fc1_weight, fc2_weight, rows,
                         input_dim, hidden_dim, output_dim);
}

#ifdef __cplusplus
} /* extern "C" */
#endif
