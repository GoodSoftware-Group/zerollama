/* DiT-critical ops that were stubs: gate_adaln, add/sub, offsets, euler, head-major SDPA. */
#include "h3_cuda_internal.h"

#include <cuda_bf16.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ static void k_add_bf16(const __nv_bfloat16 *a, const __nv_bfloat16 *b,
                                  __nv_bfloat16 *out, uint32_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

__global__ static void k_sub_bf16(const __nv_bfloat16 *a, const __nv_bfloat16 *b,
                                  __nv_bfloat16 *out, uint32_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = __float2bfloat16(__bfloat162float(a[i]) - __bfloat162float(b[i]));
}

__global__ static void k_euler_bf16(__nv_bfloat16 *sample, size_t sample_offset,
                                    const __nv_bfloat16 *last,
                                    const __nv_bfloat16 *previous, uint32_t n,
                                    float delta, float ratio) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float s = __bfloat162float(sample[sample_offset + i]);
  float l = __bfloat162float(last[i]);
  float p = previous ? __bfloat162float(previous[i]) : l;
  float vel = l + ratio * (l - p);
  sample[sample_offset + i] = __float2bfloat16(s + delta * vel);
}

/* Head-major [heads,seq,dim] ← row-major [seq,heads,dim] after SDPA. */
__global__ static void k_transpose_seq_head(__nv_bfloat16 *dst,
                                            const __nv_bfloat16 *src,
                                            uint32_t sequence, uint32_t heads,
                                            uint32_t head_dim) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = (size_t)sequence * heads * head_dim;
  if (idx >= n) return;
  uint32_t d = (uint32_t)(idx % head_dim);
  size_t t = idx / head_dim;
  uint32_t head = (uint32_t)(t % heads);
  uint32_t row = (uint32_t)(t / heads);
  size_t src_i = ((size_t)row * heads + head) * head_dim + d;
  size_t dst_i = ((size_t)head * sequence + row) * head_dim + d;
  dst[dst_i] = src[src_i];
}

int h3_gpu_add_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
  if (!gpu || !output || !left || !right) return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  int threads = 256;
  int blocks = (int)((elements + threads - 1) / threads);
  k_add_bf16<<<blocks, threads, 0, gpu->stream>>>(
      (const __nv_bfloat16 *)left->device, (const __nv_bfloat16 *)right->device,
      (__nv_bfloat16 *)output->device, elements);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "add_bf16")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_sub_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
  if (!gpu || !output || !left || !right) return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  int threads = 256;
  int blocks = (int)((elements + threads - 1) / threads);
  k_sub_bf16<<<blocks, threads, 0, gpu->stream>>>(
      (const __nv_bfloat16 *)left->device, (const __nv_bfloat16 *)right->device,
      (__nv_bfloat16 *)output->device, elements);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "sub_bf16")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

int h3_gpu_euler_bf16(h3_gpu *gpu, h3_gpu_tensor *sample, size_t sample_offset,
                      const h3_gpu_tensor *last, const h3_gpu_tensor *previous,
                      uint32_t elements, float delta, float ratio) {
  if (!gpu || !sample || !last) return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  int threads = 256;
  int blocks = (int)((elements + threads - 1) / threads);
  k_euler_bf16<<<blocks, threads, 0, gpu->stream>>>(
      (__nv_bfloat16 *)sample->device, sample_offset,
      (const __nv_bfloat16 *)last->device,
      previous ? (const __nv_bfloat16 *)previous->device : nullptr, elements,
      delta, ratio);
  if (!h3_cuda_check(gpu, cudaGetLastError(), "euler_bf16")) return 0;
  gpu->stats.direct_dispatches++;
  return 1;
}

/* Compose gate then AdaLN — byte-compatible with unfused path; used as fused. */
int h3_gpu_gate_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *gated_residual,
                           h3_gpu_tensor *output, const h3_gpu_tensor *residual,
                           const h3_gpu_tensor *branch,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *gate_modulation,
                           const h3_gpu_tensor *norm_modulation,
                           const h3_gpu_tensor *row_map, uint32_t rows,
                           uint32_t width, uint32_t slots, uint32_t gate_slot,
                           uint32_t shift_slot, uint32_t scale_slot,
                           float epsilon) {
  if (!h3_gpu_gate_bf16(gpu, gated_residual, residual, branch, gate_modulation,
                        row_map, rows, width, slots, gate_slot))
    return 0;
  return h3_gpu_adaln_bf16(gpu, output, gated_residual, norm_weight,
                           norm_modulation, row_map, rows, width, slots,
                           shift_slot, scale_slot, epsilon);
}

/* h3_gpu_adaln_bf16_offset in gemm.cu */

/* h3_gpu_patch_linear_bf16_offset in gemm.cu */

int h3_gpu_sdpa_bf16_head_major_output(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *query,
    const h3_gpu_tensor *key, const h3_gpu_tensor *value, uint32_t sequence,
    uint32_t heads, uint32_t head_dim, float scale) {
  /* Compute row-major into a temp, then transpose to head-major. */
  size_t n = (size_t)sequence * heads * head_dim;
  h3_gpu_tensor *tmp = h3_gpu_tensor_new_bf16(gpu, n);
  if (!tmp) return 0;
  int ok = h3_gpu_sdpa_bf16(gpu, tmp, query, key, value, sequence, heads,
                            head_dim, scale);
  if (ok) {
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    k_transpose_seq_head<<<blocks, threads, 0, gpu->stream>>>(
        (__nv_bfloat16 *)output->device, (const __nv_bfloat16 *)tmp->device,
        sequence, heads, head_dim);
    ok = h3_cuda_check(gpu, cudaGetLastError(), "sdpa head-major transpose");
    if (ok) gpu->stats.direct_dispatches++;
  }
  h3_gpu_tensor_free(tmp);
  return ok;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
