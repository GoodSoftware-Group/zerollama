#include "h3_cuda_internal.h"

#include <cuda_bf16.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif


h3_gpu_tensor *h3_cuda_tensor_alloc(h3_gpu *gpu, size_t elements,
                                    h3_gpu_dtype dtype) {
  if (!gpu || !elements) {
    if (gpu) h3_cuda_vset_error(gpu, "invalid tensor alloc");
    return NULL;
  }
  size_t item = h3_cuda_dtype_size(dtype);
  if (!item || elements > SIZE_MAX / item) {
    h3_cuda_vset_error(gpu, "tensor size overflow");
    return NULL;
  }
  h3_gpu_tensor *t = (h3_gpu_tensor *)calloc(1, sizeof(*t));
  if (!t) {
    h3_cuda_vset_error(gpu, "oom tensor");
    return NULL;
  }
  t->owner = gpu;
  t->elements = elements;
  t->bytes = elements * item;
  t->dtype = dtype;
  if (!h3_cuda_check(gpu, cudaMalloc(&t->device, t->bytes), "cudaMalloc")) {
    free(t);
    return NULL;
  }
  gpu->stats.allocated_bytes += t->bytes;
  gpu->stats.live_bytes += t->bytes;
  if (gpu->stats.live_bytes > gpu->stats.peak_live_bytes)
    gpu->stats.peak_live_bytes = gpu->stats.live_bytes;
  gpu->stats.tensor_allocations++;
  return t;
}

h3_gpu_tensor *h3_gpu_tensor_new_f32(h3_gpu *gpu, size_t elements) {
  return h3_cuda_tensor_alloc(gpu, elements, H3_GPU_F32);
}
h3_gpu_tensor *h3_gpu_tensor_new_bf16(h3_gpu *gpu, size_t elements) {
  return h3_cuda_tensor_alloc(gpu, elements, H3_GPU_BF16);
}
h3_gpu_tensor *h3_gpu_tensor_new_i8(h3_gpu *gpu, size_t elements) {
  return h3_cuda_tensor_alloc(gpu, elements, H3_GPU_I8);
}

static h3_gpu_tensor *tensor_from_host(h3_gpu *gpu, const void *src,
                                       size_t elements, h3_gpu_dtype dtype) {
  h3_gpu_tensor *t = h3_cuda_tensor_alloc(gpu, elements, dtype);
  if (!t) return NULL;
  if (!h3_cuda_check(gpu,
                     cudaMemcpy(t->device, src, t->bytes, cudaMemcpyHostToDevice),
                     "H2D")) {
    h3_gpu_tensor_free(t);
    return NULL;
  }
  return t;
}

h3_gpu_tensor *h3_gpu_tensor_from_f32(h3_gpu *gpu, const float *values,
                                      size_t elements) {
  return tensor_from_host(gpu, values, elements, H3_GPU_F32);
}
h3_gpu_tensor *h3_gpu_tensor_from_bf16(h3_gpu *gpu, const uint16_t *values,
                                       size_t elements) {
  return tensor_from_host(gpu, values, elements, H3_GPU_BF16);
}
h3_gpu_tensor *h3_gpu_tensor_from_u32(h3_gpu *gpu, const uint32_t *values,
                                      size_t elements) {
  return tensor_from_host(gpu, values, elements, H3_GPU_U32);
}

static h3_gpu_tensor *load_file(h3_gpu *gpu, const char *path,
                                uint64_t file_offset, size_t elements,
                                h3_gpu_dtype dtype) {
  h3_gpu_tensor *t = h3_cuda_tensor_alloc(gpu, elements, dtype);
  if (!t) return NULL;
  char err[256];
  if (dtype == H3_GPU_BF16) {
    if (!h3_gpu_tensor_read_file_bf16(t, path, file_offset, elements, err,
                                      sizeof(err))) {
      h3_cuda_vset_error(gpu, "%s", err);
      h3_gpu_tensor_free(t);
      return NULL;
    }
  } else {
    /* F32 path: host staging */
    size_t bytes = t->bytes;
    void *host = malloc(bytes);
    if (!host) {
      h3_cuda_vset_error(gpu, "oom host staging");
      h3_gpu_tensor_free(t);
      return NULL;
    }
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
      h3_cuda_vset_error(gpu, "open %s: %s", path, strerror(errno));
      free(host);
      h3_gpu_tensor_free(t);
      return NULL;
    }
    size_t done = 0;
    while (done < bytes) {
      ssize_t n = pread(fd, (char *)host + done, bytes - done,
                        (off_t)(file_offset + done));
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) {
        h3_cuda_vset_error(gpu, "short read %s", path);
        close(fd);
        free(host);
        h3_gpu_tensor_free(t);
        return NULL;
      }
      done += (size_t)n;
    }
    close(fd);
    cudaError_t ce =
        cudaMemcpy(t->device, host, bytes, cudaMemcpyHostToDevice);
    free(host);
    if (!h3_cuda_check(gpu, ce, "H2D load")) {
      h3_gpu_tensor_free(t);
      return NULL;
    }
  }
  return t;
}

h3_gpu_tensor *h3_gpu_tensor_load_bf16(h3_gpu *gpu, const char *path,
                                       uint64_t file_offset, size_t elements) {
  return load_file(gpu, path, file_offset, elements, H3_GPU_BF16);
}
h3_gpu_tensor *h3_gpu_tensor_load_f32(h3_gpu *gpu, const char *path,
                                      uint64_t file_offset, size_t elements) {
  return load_file(gpu, path, file_offset, elements, H3_GPU_F32);
}

static int read_file_bf16_mode(h3_gpu_tensor *tensor, const char *path,
                               uint64_t file_offset, size_t elements,
                               int /*uncached*/, char *error,
                               size_t error_size) {
  if (error && error_size) error[0] = '\0';
  if (!tensor || !path || !*path || tensor->dtype != H3_GPU_BF16 ||
      elements != tensor->elements) {
    if (error && error_size)
      snprintf(error, error_size, "invalid BF16 file read");
    return 0;
  }
  h3_gpu *gpu = tensor->owner;
  if (!gpu || !gpu->pin[0] || !gpu->pin[1] || !gpu->pin_chunk_bytes) {
    if (error && error_size)
      snprintf(error, error_size, "missing pinned staging");
    return 0;
  }
  size_t bytes = elements * 2;
  int fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    if (error && error_size)
      snprintf(error, error_size, "open %s: %s", path, strerror(errno));
    return 0;
  }
#ifdef POSIX_FADV_DONTNEED
  (void)posix_fadvise(fd, (off_t)file_offset, (off_t)bytes, POSIX_FADV_SEQUENTIAL);
#endif

  /* Double-buffer: pread into pin[cur] while GPU drains pin[cur^1]. */
  size_t done = 0;
  int cur = 0;
  int outstanding = 0;
  while (done < bytes) {
    size_t chunk = bytes - done;
    if (chunk > gpu->pin_chunk_bytes) chunk = gpu->pin_chunk_bytes;
    if (outstanding) {
      cudaError_t we = cudaEventSynchronize(gpu->pin_event[cur]);
      if (we != cudaSuccess) {
        if (error && error_size)
          snprintf(error, error_size, "pin wait: %s", cudaGetErrorString(we));
        close(fd);
        return 0;
      }
    }
    size_t got = 0;
    while (got < chunk) {
      ssize_t n = pread(fd, (char *)gpu->pin[cur] + got, chunk - got,
                        (off_t)(file_offset + done + got));
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) {
        if (error && error_size)
          snprintf(error, error_size, "short BF16 read from %s", path);
        close(fd);
        return 0;
      }
      got += (size_t)n;
    }
    cudaError_t ce = cudaMemcpyAsync(
        (char *)tensor->device + done, gpu->pin[cur], chunk,
        cudaMemcpyHostToDevice, gpu->stream);
    if (ce != cudaSuccess) {
      if (error && error_size)
        snprintf(error, error_size, "H2D: %s", cudaGetErrorString(ce));
      close(fd);
      return 0;
    }
    cudaEventRecord(gpu->pin_event[cur], gpu->stream);
    done += chunk;
    cur ^= 1;
    outstanding = 1;
  }
  close(fd);
  cudaError_t se = cudaStreamSynchronize(gpu->stream);
  if (se != cudaSuccess) {
    if (error && error_size)
      snprintf(error, error_size, "H2D sync: %s", cudaGetErrorString(se));
    return 0;
  }
  return 1;
}

int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 char *error, size_t error_size) {
  return read_file_bf16_mode(tensor, path, file_offset, elements, 0, error,
                             error_size);
}

int h3_gpu_tensor_stream_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                   uint64_t file_offset, size_t elements,
                                   char *error, size_t error_size) {
  return read_file_bf16_mode(tensor, path, file_offset, elements, 1, error,
                             error_size);
}

void h3_gpu_tensor_free(h3_gpu_tensor *tensor) {
  if (!tensor) return;
  if (tensor->owner && tensor->bytes) {
    if (tensor->owner->stats.live_bytes >= tensor->bytes)
      tensor->owner->stats.live_bytes -= tensor->bytes;
    else
      tensor->owner->stats.live_bytes = 0;
  }
  if (tensor->device) cudaFree(tensor->device);
  free(tensor);
}

size_t h3_gpu_tensor_elements(const h3_gpu_tensor *tensor) {
  return tensor ? tensor->elements : 0;
}
h3_gpu_dtype h3_gpu_tensor_dtype(const h3_gpu_tensor *tensor) {
  return tensor ? tensor->dtype : H3_GPU_F32;
}

int h3_gpu_tensor_read_f32(const h3_gpu_tensor *tensor, float *values,
                           size_t elements) {
  return h3_gpu_tensor_read_f32_range(tensor, 0, values, elements);
}

int h3_gpu_tensor_read_f32_range(const h3_gpu_tensor *tensor,
                                 size_t source_offset, float *values,
                                 size_t elements) {
  if (!tensor || !values || tensor->dtype != H3_GPU_F32) return 0;
  if (source_offset + elements > tensor->elements) return 0;
  return cudaMemcpy(values, (const char *)tensor->device + source_offset * 4,
                    elements * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
}

int h3_gpu_tensor_read_bf16(const h3_gpu_tensor *tensor, uint16_t *values,
                            size_t elements) {
  if (!tensor || !values || tensor->dtype != H3_GPU_BF16) return 0;
  if (elements > tensor->elements) return 0;
  return cudaMemcpy(values, tensor->device, elements * 2,
                    cudaMemcpyDeviceToHost) == cudaSuccess;
}

int h3_gpu_tensor_write_f32(h3_gpu_tensor *tensor, const float *values,
                            size_t elements) {
  return h3_gpu_tensor_write_f32_range(tensor, 0, values, elements);
}

int h3_gpu_tensor_write_f32_range(h3_gpu_tensor *tensor,
                                  size_t destination_offset,
                                  const float *values, size_t elements) {
  if (!tensor || !values || tensor->dtype != H3_GPU_F32) return 0;
  if (destination_offset + elements > tensor->elements) return 0;
  return cudaMemcpy((char *)tensor->device + destination_offset * 4, values,
                    elements * 4, cudaMemcpyHostToDevice) == cudaSuccess;
}

int h3_gpu_tensor_write_bf16(h3_gpu_tensor *tensor, const uint16_t *values,
                             size_t elements) {
  return h3_gpu_tensor_write_bf16_range(tensor, 0, values, elements);
}

int h3_gpu_tensor_write_bf16_range(h3_gpu_tensor *tensor,
                                   size_t destination_offset,
                                   const uint16_t *values, size_t elements) {
  if (!tensor || !values || tensor->dtype != H3_GPU_BF16) return 0;
  if (destination_offset + elements > tensor->elements) return 0;
  return cudaMemcpy((char *)tensor->device + destination_offset * 2, values,
                    elements * 2, cudaMemcpyHostToDevice) == cudaSuccess;
}

int h3_gpu_copy_bf16(h3_gpu *gpu, h3_gpu_tensor *destination,
                     size_t destination_offset, const h3_gpu_tensor *source,
                     size_t source_offset, size_t elements) {
  if (!gpu || !destination || !source) return 0;
  if (destination->dtype != H3_GPU_BF16 || source->dtype != H3_GPU_BF16)
    return 0;
  if (destination_offset + elements > destination->elements ||
      source_offset + elements > source->elements)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  cudaError_t e = cudaMemcpyAsync(
      (char *)destination->device + destination_offset * 2,
      (const char *)source->device + source_offset * 2, elements * 2,
      cudaMemcpyDeviceToDevice, gpu->stream);
  if (!h3_cuda_check(gpu, e, "copy_bf16")) return 0;
  gpu->stats.blit_copies++;
  return 1;
}

int h3_gpu_copy_f32(h3_gpu *gpu, h3_gpu_tensor *destination,
                    size_t destination_offset, const h3_gpu_tensor *source,
                    size_t source_offset, size_t elements) {
  if (!gpu || !destination || !source) return 0;
  if (destination->dtype != H3_GPU_F32 || source->dtype != H3_GPU_F32) return 0;
  if (destination_offset + elements > destination->elements ||
      source_offset + elements > source->elements)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  cudaError_t e = cudaMemcpyAsync(
      (char *)destination->device + destination_offset * 4,
      (const char *)source->device + source_offset * 4, elements * 4,
      cudaMemcpyDeviceToDevice, gpu->stream);
  if (!h3_cuda_check(gpu, e, "copy_f32")) return 0;
  gpu->stats.blit_copies++;
  return 1;
}

__global__ static void k_cast_f32_to_bf16(const float *in, __nv_bfloat16 *out,
                                          size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16(in[i]);
}
__global__ static void k_cast_bf16_to_f32(const __nv_bfloat16 *in, float *out,
                                          size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __bfloat162float(in[i]);
}

int h3_gpu_cast_f32_to_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
  if (!gpu || !output || !input || output->dtype != H3_GPU_BF16 ||
      input->dtype != H3_GPU_F32 || elements > output->elements ||
      elements > input->elements)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  int threads = 256;
  int blocks = (int)((elements + threads - 1) / threads);
  k_cast_f32_to_bf16<<<blocks, threads, 0, gpu->stream>>>(
      (const float *)input->device, (__nv_bfloat16 *)output->device, elements);
  return h3_cuda_check(gpu, cudaGetLastError(), "cast_f32_to_bf16");
}

int h3_gpu_cast_bf16_to_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
  if (!gpu || !output || !input || output->dtype != H3_GPU_F32 ||
      input->dtype != H3_GPU_BF16 || elements > output->elements ||
      elements > input->elements)
    return 0;
  if (!h3_cuda_require_encoding(gpu)) return 0;
  int threads = 256;
  int blocks = (int)((elements + threads - 1) / threads);
  k_cast_bf16_to_f32<<<blocks, threads, 0, gpu->stream>>>(
      (const __nv_bfloat16 *)input->device, (float *)output->device, elements);
  return h3_cuda_check(gpu, cudaGetLastError(), "cast_bf16_to_f32");
}

#ifdef __cplusplus
} /* extern "C" */
#endif
