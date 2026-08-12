/* Internal CUDA backend state for h3_gpu.h (lab P0). */
#ifndef H3_CUDA_INTERNAL_H
#define H3_CUDA_INTERNAL_H

#include "h3_gpu.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct h3_gpu {
  int device;
  cudaStream_t stream;
  cublasHandle_t cublas;
  char error[512];
  h3_gpu_stats stats;
  int encoding;
  char profile_label[128];
  /* Double-buffered pinned H2D staging for SSD weight streaming. */
  void *pin[2];
  size_t pin_chunk_bytes;
  cudaEvent_t pin_event[2];
  /* Reused INT32 accumulator for cuBLAS int8 → scale → BF16. */
  int32_t *int8_acc;
  size_t int8_acc_elems;
};

struct h3_gpu_tensor {
  h3_gpu *owner;
  void *device;
  size_t elements;
  size_t bytes;
  h3_gpu_dtype dtype;
};

void h3_cuda_vset_error(h3_gpu *gpu, const char *fmt, ...);
int h3_cuda_check(h3_gpu *gpu, cudaError_t err, const char *what);
int h3_cuda_cublas_check(h3_gpu *gpu, cublasStatus_t st, const char *what);
size_t h3_cuda_dtype_size(h3_gpu_dtype dtype);
h3_gpu_tensor *h3_cuda_tensor_alloc(h3_gpu *gpu, size_t elements,
                                    h3_gpu_dtype dtype);
int h3_cuda_require_encoding(h3_gpu *gpu);

#ifdef __cplusplus
}
#endif

#endif
