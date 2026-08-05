/*
 * safetensors_min.h — mmap reader for HuggingFace safetensors (DiT weights as-is).
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  ST_DTYPE_F32 = 0,
  ST_DTYPE_F16 = 1,
  ST_DTYPE_BF16 = 2,
  ST_DTYPE_UNKNOWN = -1
} st_dtype_t;

typedef struct st_tensor {
  char name[256];
  int ndim;
  int64_t shape[8];
  st_dtype_t dtype;
  size_t data_begin; /* offset into data section (not file) */
  size_t data_end;
  size_t nbytes;
} st_tensor_t;

typedef struct st_file st_file;

st_file *st_open(const char *path);
void st_close(st_file *sf);

int st_tensor_count(const st_file *sf);
const st_tensor_t *st_find_tensor(const st_file *sf, const char *name);
size_t st_tensor_nelems(const st_tensor_t *t);
int st_tensor_to_f32(const st_file *sf, const st_tensor_t *t, float *dst,
                     size_t dst_n);

#ifdef __cplusplus
}
#endif
