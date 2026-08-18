/*
 * zip_weight.h — load named tensors from torch .pth (zip, store) via JSON index.
 * Index maps logical name → absolute zip_offset + dtype/shape (no tensor copy).
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  ZW_DTYPE_F32 = 0,
  ZW_DTYPE_F16 = 1,
  ZW_DTYPE_BF16 = 2,
  ZW_DTYPE_UNKNOWN = -1
} zw_dtype_t;

typedef struct zw_tensor {
  char name[256];
  int ndim;
  int64_t shape[8];
  zw_dtype_t dtype;
  size_t zip_offset;
  size_t nbytes;
} zw_tensor_t;

typedef struct zw_file zw_file;

/* index_json lists tensors; pth_path is the torch zip on disk. */
zw_file *zw_open(const char *pth_path, const char *index_json);
void zw_close(zw_file *zf);

const zw_tensor_t *zw_find_tensor(const zw_file *zf, const char *name);
size_t zw_tensor_nelems(const zw_tensor_t *t);
int zw_tensor_to_f32(const zw_file *zf, const zw_tensor_t *t, float *dst,
                     size_t dst_n);

#ifdef __cplusplus
}
#endif
