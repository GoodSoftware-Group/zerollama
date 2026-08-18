/*
 * gguf_min.h — minimal GGUF reader (tensor names, shapes, mmap data).
 *
 * Avoids linking full ggml; sufficient for Wan weight discovery.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  GGUF_DTYPE_F32 = 0,
  GGUF_DTYPE_F16 = 1,
  GGUF_DTYPE_UNKNOWN = -1
} gguf_dtype_t;

typedef struct gguf_tensor {
  char name[256];
  int ndim;
  int64_t shape[8];
  gguf_dtype_t dtype;
  size_t offset;
  size_t nbytes;
} gguf_tensor_t;

typedef struct gguf_file gguf_file;

gguf_file *gguf_open(const char *path);
void gguf_close(gguf_file *gf);

int gguf_tensor_count(const gguf_file *gf);
const gguf_tensor_t *gguf_tensor_at(const gguf_file *gf, int index);
const gguf_tensor_t *gguf_find_tensor(const gguf_file *gf, const char *name);

const void *gguf_tensor_data(const gguf_file *gf, const gguf_tensor_t *t);

/* Element count from shape[]. */
size_t gguf_tensor_nelems(const gguf_tensor_t *t);

/* Copy tensor to f32 (converts f16). dst_n is element capacity. */
int gguf_tensor_to_f32(const gguf_file *gf, const gguf_tensor_t *t, float *dst,
                       size_t dst_n);

#ifdef __cplusplus
}
#endif
