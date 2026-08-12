#include "h3_cuda_internal.h"
#include "h3_cuda.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif


void h3_cuda_vset_error(h3_gpu *gpu, const char *fmt, ...) {
  if (!gpu) return;
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(gpu->error, sizeof(gpu->error), fmt, ap);
  va_end(ap);
}

int h3_cuda_check(h3_gpu *gpu, cudaError_t err, const char *what) {
  if (err == cudaSuccess) return 1;
  h3_cuda_vset_error(gpu, "%s: %s", what, cudaGetErrorString(err));
  return 0;
}

int h3_cuda_cublas_check(h3_gpu *gpu, cublasStatus_t st, const char *what) {
  if (st == CUBLAS_STATUS_SUCCESS) return 1;
  h3_cuda_vset_error(gpu, "%s: cublas status %d", what, (int)st);
  return 0;
}

size_t h3_cuda_dtype_size(h3_gpu_dtype dtype) {
  switch (dtype) {
  case H3_GPU_F32:
    return 4;
  case H3_GPU_BF16:
  case H3_GPU_I8:
    return dtype == H3_GPU_I8 ? 1 : 2;
  case H3_GPU_U32:
    return 4;
  }
  return 0;
}

int h3_cuda_require_encoding(h3_gpu *gpu) {
  if (!gpu) return 0;
  if (!gpu->encoding) {
    h3_cuda_vset_error(gpu, "command encoding not begun (call h3_gpu_begin)");
    return 0;
  }
  return 1;
}

extern "C" int h3_cuda_probe(h3_device_info *info, char *error,
                             size_t error_size) {
  if (!info) {
    if (error && error_size) snprintf(error, error_size, "null info");
    return -1;
  }
  memset(info, 0, sizeof(*info));
  int n = 0;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (err != cudaSuccess || n < 1) {
    if (error && error_size)
      snprintf(error, error_size, "cudaGetDeviceCount: %s",
               cudaGetErrorString(err));
    return -1;
  }
  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, 0);
  if (err != cudaSuccess) {
    if (error && error_size)
      snprintf(error, error_size, "cudaGetDeviceProperties: %s",
               cudaGetErrorString(err));
    return -1;
  }
  snprintf(info->name, sizeof(info->name), "%.127s", prop.name);
  snprintf(info->architecture, sizeof(info->architecture), "sm_%d%d", prop.major,
           prop.minor);
  info->physical_memory = prop.totalGlobalMem;
  info->unified_memory = prop.unifiedAddressing ? 1 : 0;
  return 0;
}

h3_gpu *h3_gpu_create(const char *shader_source_path, char *error,
                      size_t error_size) {
  (void)shader_source_path;
  h3_gpu *gpu = (h3_gpu *)calloc(1, sizeof(*gpu));
  if (!gpu) {
    if (error && error_size) snprintf(error, error_size, "oom");
    return NULL;
  }
  if (cudaSetDevice(0) != cudaSuccess) {
    if (error && error_size)
      snprintf(error, error_size, "cudaSetDevice(0) failed");
    free(gpu);
    return NULL;
  }
  gpu->device = 0;
  if (cudaStreamCreate(&gpu->stream) != cudaSuccess) {
    if (error && error_size) snprintf(error, error_size, "stream create failed");
    free(gpu);
    return NULL;
  }
  if (cublasCreate(&gpu->cublas) != CUBLAS_STATUS_SUCCESS) {
    if (error && error_size) snprintf(error, error_size, "cublasCreate failed");
    cudaStreamDestroy(gpu->stream);
    free(gpu);
    return NULL;
  }
  cublasSetStream(gpu->cublas, gpu->stream);
  /* Prefer Tensor Core BF16 math where available. */
  cublasSetMathMode(gpu->cublas, CUBLAS_TF32_TENSOR_OP_MATH);

  gpu->pin_chunk_bytes = 4u << 20; /* 4 MiB chunks */
  for (int i = 0; i < 2; i++) {
    if (cudaHostAlloc(&gpu->pin[i], gpu->pin_chunk_bytes, cudaHostAllocDefault) !=
            cudaSuccess ||
        cudaEventCreateWithFlags(&gpu->pin_event[i], cudaEventDisableTiming) !=
            cudaSuccess) {
      if (error && error_size)
        snprintf(error, error_size, "pinned staging alloc failed");
      for (int j = 0; j <= i; j++) {
        if (gpu->pin_event[j]) cudaEventDestroy(gpu->pin_event[j]);
        if (gpu->pin[j]) cudaFreeHost(gpu->pin[j]);
      }
      cublasDestroy(gpu->cublas);
      cudaStreamDestroy(gpu->stream);
      free(gpu);
      return NULL;
    }
  }
  return gpu;
}

void h3_gpu_free(h3_gpu *gpu) {
  if (!gpu) return;
  for (int i = 0; i < 2; i++) {
    if (gpu->pin_event[i]) cudaEventDestroy(gpu->pin_event[i]);
    if (gpu->pin[i]) cudaFreeHost(gpu->pin[i]);
  }
  if (gpu->int8_acc) cudaFree(gpu->int8_acc);
  if (gpu->cublas) cublasDestroy(gpu->cublas);
  if (gpu->stream) cudaStreamDestroy(gpu->stream);
  free(gpu);
}

int h3_gpu_is_m5(const h3_gpu *gpu) {
  (void)gpu;
  return 0;
}
int h3_gpu_has_nax_mlp(const h3_gpu *gpu) {
  (void)gpu;
  return 0; /* Apple Neural Accelerators only */
}
int h3_gpu_has_int8_mlp(const h3_gpu *gpu) {
  (void)gpu;
  return 1; /* portable CUDA int8 linear/MLP path */
}

int h3_gpu_begin(h3_gpu *gpu) {
  if (!gpu) return 0;
  gpu->encoding = 1;
  gpu->error[0] = '\0';
  return 1;
}

int h3_gpu_continue(h3_gpu *gpu) {
  if (!gpu || !gpu->encoding) return 0;
  /* Flush current work; keep encoding open (Metal multi-buffer split). */
  return h3_cuda_check(gpu, cudaStreamSynchronize(gpu->stream), "continue sync");
}

int h3_gpu_submit(h3_gpu *gpu) {
  if (!gpu || !gpu->encoding) return 0;
  if (!h3_cuda_check(gpu, cudaStreamSynchronize(gpu->stream), "submit sync"))
    return 0;
  gpu->encoding = 0;
  gpu->stats.submissions++;
  return 1;
}

const char *h3_gpu_error(const h3_gpu *gpu) {
  return gpu ? gpu->error : "null gpu";
}

int h3_gpu_get_stats(const h3_gpu *gpu, h3_gpu_stats *stats) {
  if (!gpu || !stats) return 0;
  *stats = gpu->stats;
  return 1;
}

void h3_gpu_profile_set_label(h3_gpu *gpu, const char *label) {
  if (!gpu) return;
  snprintf(gpu->profile_label, sizeof(gpu->profile_label), "%s",
           label ? label : "");
}

void h3_gpu_profile_mark(h3_gpu *gpu, const char *phase) {
  (void)gpu;
  (void)phase;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
