#include "gguf_min.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define GGUF_MAGIC 0x46554747u /* "GGUF" */
#define GGUF_VERSION 3
#define GGUF_ALIGN 32

typedef enum {
  GGUF_VAL_U8 = 0,
  GGUF_VAL_I8 = 1,
  GGUF_VAL_U16 = 2,
  GGUF_VAL_I16 = 3,
  GGUF_VAL_U32 = 4,
  GGUF_VAL_I32 = 5,
  GGUF_VAL_F32 = 6,
  GGUF_VAL_BOOL = 7,
  GGUF_VAL_STRING = 8,
  GGUF_VAL_ARRAY = 9,
  GGUF_VAL_U64 = 10,
  GGUF_VAL_I64 = 11,
  GGUF_VAL_F64 = 12,
} gguf_val_type_t;

typedef enum {
  GGML_TYPE_F32 = 0,
  GGML_TYPE_F16 = 1,
} ggml_type_t;

struct gguf_file {
  int fd;
  size_t size;
  void *map;
  const uint8_t *data;
  gguf_tensor_t *tensors;
  int n_tensors;
  size_t data_offset; /* start of tensor data section (aligned) */
};

static uint32_t rd_u32(const uint8_t **p, const uint8_t *end) {
  if ((size_t)(end - *p) < 4)
    return 0;
  uint32_t v;
  memcpy(&v, *p, 4);
  *p += 4;
  return v;
}

static uint64_t rd_u64(const uint8_t **p, const uint8_t *end) {
  if ((size_t)(end - *p) < 8)
    return 0;
  uint64_t v;
  memcpy(&v, *p, 8);
  *p += 8;
  return v;
}

static int skip_value(const uint8_t **p, const uint8_t *end, uint32_t type) {
  switch (type) {
  case GGUF_VAL_U8:
  case GGUF_VAL_I8:
  case GGUF_VAL_BOOL:
    if (*p + 1 > end)
      return -1;
    *p += 1;
    return 0;
  case GGUF_VAL_U16:
  case GGUF_VAL_I16:
    if (*p + 2 > end)
      return -1;
    *p += 2;
    return 0;
  case GGUF_VAL_U32:
  case GGUF_VAL_I32:
  case GGUF_VAL_F32:
    if (*p + 4 > end)
      return -1;
    *p += 4;
    return 0;
  case GGUF_VAL_U64:
  case GGUF_VAL_I64:
  case GGUF_VAL_F64:
    if (*p + 8 > end)
      return -1;
    *p += 8;
    return 0;
  case GGUF_VAL_STRING: {
    uint64_t n = rd_u64(p, end);
    if (*p + n > end)
      return -1;
    *p += (size_t)n;
    return 0;
  }
  case GGUF_VAL_ARRAY: {
    uint32_t at = rd_u32(p, end);
    uint64_t n = rd_u64(p, end);
    for (uint64_t i = 0; i < n; i++) {
      if (skip_value(p, end, at) != 0)
        return -1;
    }
    return 0;
  }
  default:
    return -1;
  }
}

static int read_string(const uint8_t **p, const uint8_t *end, char *out,
                       size_t out_n) {
  uint64_t n = rd_u64(p, end);
  if (*p + n > end || n + 1 > out_n)
    return -1;
  memcpy(out, *p, (size_t)n);
  out[n] = 0;
  *p += (size_t)n;
  return 0;
}

static gguf_dtype_t map_dtype(uint32_t t) {
  if (t == GGML_TYPE_F32)
    return GGUF_DTYPE_F32;
  if (t == GGML_TYPE_F16)
    return GGUF_DTYPE_F16;
  return GGUF_DTYPE_UNKNOWN;
}

static size_t dtype_nbytes(gguf_dtype_t dt, size_t nelem) {
  if (dt == GGUF_DTYPE_F32)
    return nelem * 4;
  if (dt == GGUF_DTYPE_F16)
    return nelem * 2;
  return 0;
}

static size_t align_up(size_t v, size_t a) {
  return (v + a - 1) & ~(a - 1);
}

gguf_file *gguf_open(const char *path) {
  int fd = open(path, O_RDONLY);
  if (fd < 0)
    return NULL;

  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size < 24) {
    close(fd);
    return NULL;
  }

  void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) {
    close(fd);
    return NULL;
  }

  gguf_file *gf = calloc(1, sizeof(*gf));
  if (!gf) {
    munmap(map, (size_t)st.st_size);
    close(fd);
    return NULL;
  }

  gf->fd = fd;
  gf->size = (size_t)st.st_size;
  gf->map = map;
  gf->data = (const uint8_t *)map;

  const uint8_t *p = gf->data;
  const uint8_t *end = gf->data + gf->size;

  uint32_t magic = rd_u32(&p, end);
  uint32_t version = rd_u32(&p, end);
  uint64_t n_tensors = rd_u64(&p, end);
  uint64_t n_kv = rd_u64(&p, end);

  if (magic != GGUF_MAGIC || version != GGUF_VERSION || n_tensors > 100000) {
    gguf_close(gf);
    return NULL;
  }

  for (uint64_t i = 0; i < n_kv; i++) {
    char key[256];
    if (read_string(&p, end, key, sizeof(key)) != 0) {
      gguf_close(gf);
      return NULL;
    }
    uint32_t vtype = rd_u32(&p, end);
    if (skip_value(&p, end, vtype) != 0) {
      gguf_close(gf);
      return NULL;
    }
  }

  gf->n_tensors = (int)n_tensors;
  gf->tensors = calloc((size_t)n_tensors, sizeof(gguf_tensor_t));
  if (!gf->tensors) {
    gguf_close(gf);
    return NULL;
  }

  for (int i = 0; i < gf->n_tensors; i++) {
    gguf_tensor_t *t = &gf->tensors[i];
    if (read_string(&p, end, t->name, sizeof(t->name)) != 0) {
      gguf_close(gf);
      return NULL;
    }
    t->ndim = (int)rd_u32(&p, end);
    if (t->ndim < 0 || t->ndim > 8) {
      gguf_close(gf);
      return NULL;
    }
    size_t nelem = 1;
    for (int d = 0; d < t->ndim; d++) {
      t->shape[d] = (int64_t)rd_u64(&p, end);
      if (t->shape[d] < 1) {
        gguf_close(gf);
        return NULL;
      }
      nelem *= (size_t)t->shape[d];
    }
    t->dtype = map_dtype(rd_u32(&p, end));
    /* GGUF: offset is relative to start of aligned data section. */
    t->offset = (size_t)rd_u64(&p, end);
    t->nbytes = dtype_nbytes(t->dtype, nelem);
  }

  /* Data section starts after tensor info, padded to GGUF_ALIGN. */
  gf->data_offset = align_up((size_t)(p - gf->data), GGUF_ALIGN);
  return gf;
}

void gguf_close(gguf_file *gf) {
  if (!gf)
    return;
  free(gf->tensors);
  if (gf->map && gf->map != MAP_FAILED)
    munmap(gf->map, gf->size);
  if (gf->fd >= 0)
    close(gf->fd);
  free(gf);
}

int gguf_tensor_count(const gguf_file *gf) {
  return gf ? gf->n_tensors : 0;
}

const gguf_tensor_t *gguf_tensor_at(const gguf_file *gf, int index) {
  if (!gf || index < 0 || index >= gf->n_tensors)
    return NULL;
  return &gf->tensors[index];
}

const gguf_tensor_t *gguf_find_tensor(const gguf_file *gf, const char *name) {
  if (!gf || !name)
    return NULL;
  for (int i = 0; i < gf->n_tensors; i++) {
    if (!strcmp(gf->tensors[i].name, name))
      return &gf->tensors[i];
  }
  return NULL;
}

const void *gguf_tensor_data(const gguf_file *gf, const gguf_tensor_t *t) {
  if (!gf || !t)
    return NULL;

  size_t abs_off;
  /* Relative (spec) or absolute (legacy converter) offsets. */
  if (t->offset >= gf->data_offset && t->offset + t->nbytes <= gf->size)
    abs_off = t->offset;
  else
    abs_off = gf->data_offset + t->offset;

  if (abs_off + t->nbytes > gf->size)
    return NULL;
  return gf->data + abs_off;
}

size_t gguf_tensor_nelems(const gguf_tensor_t *t) {
  if (!t || t->ndim < 1)
    return 0;
  size_t n = 1;
  for (int i = 0; i < t->ndim; i++)
    n *= (size_t)t->shape[i];
  return n;
}

/* IEEE754 binary16 → f32 (matches ggml soft-float path closely enough). */
static float f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3ffu;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  float out;
  memcpy(&out, &bits, sizeof(out));
  return out;
}

int gguf_tensor_to_f32(const gguf_file *gf, const gguf_tensor_t *t, float *dst,
                       size_t dst_n) {
  const void *raw = gguf_tensor_data(gf, t);
  if (!raw || !dst)
    return -1;
  size_t n = gguf_tensor_nelems(t);
  if (n > dst_n)
    return -1;
  if (t->dtype == GGUF_DTYPE_F32) {
    memcpy(dst, raw, n * sizeof(float));
    return 0;
  }
  if (t->dtype == GGUF_DTYPE_F16) {
    const uint16_t *h = (const uint16_t *)raw;
    for (size_t i = 0; i < n; i++)
      dst[i] = f16_to_f32(h[i]);
    return 0;
  }
  return -1;
}
