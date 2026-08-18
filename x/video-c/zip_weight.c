#include "zip_weight.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct zw_file {
  int fd;
  size_t size;
  uint8_t *map;
  zw_tensor_t *tensors;
  int n_tensors;
};

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

static float bf16_to_f32(uint16_t h) {
  uint32_t bits = (uint32_t)h << 16;
  float out;
  memcpy(&out, &bits, sizeof(out));
  return out;
}

static zw_dtype_t parse_dtype(const char *s) {
  if (strcmp(s, "float32") == 0 || strcmp(s, "F32") == 0)
    return ZW_DTYPE_F32;
  if (strcmp(s, "float16") == 0 || strcmp(s, "F16") == 0)
    return ZW_DTYPE_F16;
  if (strcmp(s, "bfloat16") == 0 || strcmp(s, "BF16") == 0)
    return ZW_DTYPE_BF16;
  return ZW_DTYPE_UNKNOWN;
}

/* Minimal index parser for our generated JSON objects. */
static int parse_index(zw_file *zf, const char *json, size_t n) {
  /* Count "zip_offset" occurrences ≈ tensor count. */
  int cap = 0;
  for (size_t i = 0; i + 10 < n; i++)
    if (strncmp(json + i, "zip_offset", 10) == 0)
      cap++;
  if (cap < 1)
    return -1;
  zf->tensors = calloc((size_t)cap, sizeof(zw_tensor_t));
  if (!zf->tensors)
    return -1;

  const char *p = json;
  const char *end = json + n;
  while (p < end && zf->n_tensors < cap) {
    /* Find next key at object start: "name": { */
    while (p < end && *p != '"')
      p++;
    if (p >= end)
      break;
    const char *ns = ++p;
    while (p < end && *p != '"')
      p++;
    size_t nlen = (size_t)(p - ns);
    if (p < end)
      p++;
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r'))
      p++;
    if (p >= end || *p != ':')
      continue;
    p++;
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r'))
      p++;
    if (p >= end || *p != '{')
      continue;
    if (nlen == 0 || nlen >= sizeof(zf->tensors[0].name))
      continue;
    /* Skip meta keys if any */
    if (ns[0] == '_')
      continue;

    zw_tensor_t *t = &zf->tensors[zf->n_tensors];
    memcpy(t->name, ns, nlen);
    t->name[nlen] = 0;
    t->dtype = ZW_DTYPE_UNKNOWN;
    t->ndim = 0;
    t->zip_offset = 0;
    t->nbytes = 0;

    const char *obj = p;
    int depth = 0;
    const char *obj_end = p;
    for (const char *q = p; q < end; q++) {
      if (*q == '{')
        depth++;
      else if (*q == '}') {
        depth--;
        if (depth == 0) {
          obj_end = q;
          break;
        }
      }
    }
    size_t olen = (size_t)(obj_end - obj);
    char *slice = malloc(olen + 1);
    if (!slice)
      return -1;
    memcpy(slice, obj, olen);
    slice[olen] = 0;

    char *dp = strstr(slice, "\"dtype\"");
    if (dp) {
      char *q = strchr(dp + 7, '"');
      if (q) {
        q++;
        char *q2 = strchr(q, '"');
        if (q2) {
          *q2 = 0;
          t->dtype = parse_dtype(q);
          *q2 = '"';
        }
      }
    }
    char *sp = strstr(slice, "\"shape\"");
    if (sp) {
      char *br = strchr(sp, '[');
      if (br) {
        br++;
        while (*br && *br != ']' && t->ndim < 8) {
          while (*br == ' ' || *br == ',')
            br++;
          char *e = NULL;
          long long v = strtoll(br, &e, 10);
          if (e == br)
            break;
          t->shape[t->ndim++] = (int64_t)v;
          br = e;
        }
      }
    }
    char *zp = strstr(slice, "\"zip_offset\"");
    if (zp) {
      char *c = strchr(zp, ':');
      if (c)
        t->zip_offset = (size_t)strtoull(c + 1, NULL, 10);
    }
    char *np = strstr(slice, "\"nbytes\"");
    if (np) {
      char *c = strchr(np, ':');
      if (c)
        t->nbytes = (size_t)strtoull(c + 1, NULL, 10);
    }
    free(slice);

    if (t->dtype != ZW_DTYPE_UNKNOWN && t->nbytes > 0 && t->ndim > 0)
      zf->n_tensors++;
    p = obj_end + 1;
  }
  return zf->n_tensors > 0 ? 0 : -1;
}

zw_file *zw_open(const char *pth_path, const char *index_json) {
  if (!pth_path || !index_json)
    return NULL;
  FILE *jf = fopen(index_json, "rb");
  if (!jf)
    return NULL;
  fseek(jf, 0, SEEK_END);
  long jsz = ftell(jf);
  fseek(jf, 0, SEEK_SET);
  if (jsz < 2) {
    fclose(jf);
    return NULL;
  }
  char *json = malloc((size_t)jsz + 1);
  if (!json) {
    fclose(jf);
    return NULL;
  }
  if (fread(json, 1, (size_t)jsz, jf) != (size_t)jsz) {
    free(json);
    fclose(jf);
    return NULL;
  }
  fclose(jf);
  json[jsz] = 0;

  int fd = open(pth_path, O_RDONLY);
  if (fd < 0) {
    free(json);
    return NULL;
  }
  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    free(json);
    return NULL;
  }
  zw_file *zf = calloc(1, sizeof(*zf));
  if (!zf) {
    close(fd);
    free(json);
    return NULL;
  }
  zf->fd = fd;
  zf->size = (size_t)st.st_size;
  zf->map = mmap(NULL, zf->size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (zf->map == MAP_FAILED) {
    close(fd);
    free(json);
    free(zf);
    return NULL;
  }
  if (parse_index(zf, json, (size_t)jsz) != 0) {
    free(json);
    zw_close(zf);
    return NULL;
  }
  free(json);
  return zf;
}

void zw_close(zw_file *zf) {
  if (!zf)
    return;
  if (zf->map && zf->map != MAP_FAILED)
    munmap(zf->map, zf->size);
  if (zf->fd >= 0)
    close(zf->fd);
  free(zf->tensors);
  free(zf);
}

const zw_tensor_t *zw_find_tensor(const zw_file *zf, const char *name) {
  if (!zf || !name)
    return NULL;
  for (int i = 0; i < zf->n_tensors; i++)
    if (strcmp(zf->tensors[i].name, name) == 0)
      return &zf->tensors[i];
  return NULL;
}

size_t zw_tensor_nelems(const zw_tensor_t *t) {
  if (!t || t->ndim < 1)
    return 0;
  size_t n = 1;
  for (int i = 0; i < t->ndim; i++)
    n *= (size_t)t->shape[i];
  return n;
}

int zw_tensor_to_f32(const zw_file *zf, const zw_tensor_t *t, float *dst,
                     size_t dst_n) {
  if (!zf || !t || !dst)
    return -1;
  size_t n = zw_tensor_nelems(t);
  if (n > dst_n || t->nbytes == 0)
    return -1;
  if (t->zip_offset + t->nbytes > zf->size)
    return -1;
  const uint8_t *raw = zf->map + t->zip_offset;
  if (t->dtype == ZW_DTYPE_F32) {
    memcpy(dst, raw, n * sizeof(float));
    return 0;
  }
  if (t->dtype == ZW_DTYPE_F16) {
    const uint16_t *h = (const uint16_t *)raw;
    for (size_t i = 0; i < n; i++)
      dst[i] = f16_to_f32(h[i]);
    return 0;
  }
  if (t->dtype == ZW_DTYPE_BF16) {
    const uint16_t *h = (const uint16_t *)raw;
    for (size_t i = 0; i < n; i++)
      dst[i] = bf16_to_f32(h[i]);
    return 0;
  }
  return -1;
}
