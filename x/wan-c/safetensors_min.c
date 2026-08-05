#include "safetensors_min.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct st_file {
  int fd;
  size_t size;
  uint8_t *map;
  size_t header_len;
  size_t data_off; /* 8 + header_len */
  st_tensor_t *tensors;
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

static st_dtype_t parse_dtype(const char *s, size_t n) {
  if (n == 3 && strncmp(s, "F32", 3) == 0)
    return ST_DTYPE_F32;
  if (n == 3 && strncmp(s, "F16", 3) == 0)
    return ST_DTYPE_F16;
  if (n == 4 && strncmp(s, "BF16", 4) == 0)
    return ST_DTYPE_BF16;
  return ST_DTYPE_UNKNOWN;
}

static const char *skip_ws(const char *p, const char *end) {
  while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t'))
    p++;
  return p;
}

/* Count top-level tensor keys (exclude __metadata__). */
static int count_tensors(const char *json, size_t n) {
  int count = 0;
  const char *p = json;
  const char *end = json + n;
  while (p < end) {
    if (*p == '"') {
      const char *s = ++p;
      while (p < end && *p != '"') {
        if (*p == '\\' && p + 1 < end)
          p += 2;
        else
          p++;
      }
      size_t len = (size_t)(p - s);
      if (p < end)
        p++;
      p = skip_ws(p, end);
      if (p < end && *p == ':' && !(len == 12 && strncmp(s, "__metadata__", 12) == 0))
        count++;
    } else {
      p++;
    }
  }
  return count;
}

static int parse_header(st_file *sf) {
  const char *json = (const char *)(sf->map + 8);
  size_t n = sf->header_len;
  const char *end = json + n;
  int cap = count_tensors(json, n);
  if (cap < 1)
    return -1;
  sf->tensors = calloc((size_t)cap, sizeof(st_tensor_t));
  if (!sf->tensors)
    return -1;

  const char *p = json;
  while (p < end && sf->n_tensors < cap) {
    p = skip_ws(p, end);
    if (p >= end)
      break;
    if (*p != '"') {
      p++;
      continue;
    }
    const char *ns = ++p;
    while (p < end && *p != '"') {
      if (*p == '\\' && p + 1 < end)
        p += 2;
      else
        p++;
    }
    size_t nlen = (size_t)(p - ns);
    if (p < end)
      p++;
    p = skip_ws(p, end);
    if (p >= end || *p != ':')
      continue;
    p++;
    if (nlen == 12 && strncmp(ns, "__metadata__", 12) == 0) {
      /* skip object */
      int depth = 0;
      while (p < end) {
        if (*p == '{')
          depth++;
        else if (*p == '}') {
          depth--;
          p++;
          if (depth == 0)
            break;
          continue;
        }
        p++;
      }
      continue;
    }
    if (nlen >= sizeof(sf->tensors[0].name))
      nlen = sizeof(sf->tensors[0].name) - 1;
    st_tensor_t *t = &sf->tensors[sf->n_tensors];
    memcpy(t->name, ns, nlen);
    t->name[nlen] = 0;
    t->dtype = ST_DTYPE_UNKNOWN;
    t->ndim = 0;
    t->data_begin = 0;
    t->data_end = 0;

    p = skip_ws(p, end);
    if (p >= end || *p != '{')
      continue;
    p++;
    while (p < end && *p != '}') {
      p = skip_ws(p, end);
      if (p >= end || *p == '}')
        break;
      if (*p != '"') {
        p++;
        continue;
      }
      const char *ks = ++p;
      while (p < end && *p != '"')
        p++;
      size_t klen = (size_t)(p - ks);
      if (p < end)
        p++;
      p = skip_ws(p, end);
      if (p >= end || *p != ':')
        break;
      p++;
      p = skip_ws(p, end);
      if (klen == 5 && strncmp(ks, "dtype", 5) == 0 && p < end && *p == '"') {
        const char *ds = ++p;
        while (p < end && *p != '"')
          p++;
        t->dtype = parse_dtype(ds, (size_t)(p - ds));
        if (p < end)
          p++;
      } else if (klen == 5 && strncmp(ks, "shape", 5) == 0 && p < end &&
                 *p == '[') {
        p++;
        while (p < end && *p != ']' && t->ndim < 8) {
          p = skip_ws(p, end);
          char *endp = NULL;
          long long v = strtoll(p, &endp, 10);
          if (endp == p)
            break;
          t->shape[t->ndim++] = (int64_t)v;
          p = endp;
          p = skip_ws(p, end);
          if (p < end && *p == ',')
            p++;
        }
        if (p < end && *p == ']')
          p++;
      } else if (klen == 12 && strncmp(ks, "data_offsets", 12) == 0 &&
                 p < end && *p == '[') {
        p++;
        char *e1 = NULL;
        unsigned long long b = strtoull(p, &e1, 10);
        p = e1;
        p = skip_ws(p, end);
        if (p < end && *p == ',')
          p++;
        char *e2 = NULL;
        unsigned long long e = strtoull(p, &e2, 10);
        p = e2;
        t->data_begin = (size_t)b;
        t->data_end = (size_t)e;
        while (p < end && *p != ']')
          p++;
        if (p < end)
          p++;
      } else {
        /* skip value */
        if (p < end && *p == '"') {
          p++;
          while (p < end && *p != '"')
            p++;
          if (p < end)
            p++;
        } else if (p < end && (*p == '[' || *p == '{')) {
          int depth = 0;
          char open = *p;
          char close = (open == '[') ? ']' : '}';
          while (p < end) {
            if (*p == open)
              depth++;
            else if (*p == close) {
              depth--;
              p++;
              if (depth == 0)
                break;
              continue;
            }
            p++;
          }
        } else {
          while (p < end && *p != ',' && *p != '}')
            p++;
        }
      }
      p = skip_ws(p, end);
      if (p < end && *p == ',')
        p++;
    }
    if (p < end && *p == '}')
      p++;
    t->nbytes = (t->data_end > t->data_begin) ? (t->data_end - t->data_begin)
                                              : 0;
    if (t->dtype != ST_DTYPE_UNKNOWN && t->nbytes > 0)
      sf->n_tensors++;
    p = skip_ws(p, end);
    if (p < end && *p == ',')
      p++;
  }
  return sf->n_tensors > 0 ? 0 : -1;
}

st_file *st_open(const char *path) {
  if (!path)
    return NULL;
  int fd = open(path, O_RDONLY);
  if (fd < 0)
    return NULL;
  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size < 16) {
    close(fd);
    return NULL;
  }
  st_file *sf = calloc(1, sizeof(*sf));
  if (!sf) {
    close(fd);
    return NULL;
  }
  sf->fd = fd;
  sf->size = (size_t)st.st_size;
  sf->map = mmap(NULL, sf->size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (sf->map == MAP_FAILED) {
    close(fd);
    free(sf);
    return NULL;
  }
  uint64_t hlen = 0;
  memcpy(&hlen, sf->map, 8);
  if (hlen == 0 || hlen + 8 > sf->size || hlen > (64u * 1024u * 1024u)) {
    st_close(sf);
    return NULL;
  }
  sf->header_len = (size_t)hlen;
  sf->data_off = 8 + sf->header_len;
  if (parse_header(sf) != 0) {
    st_close(sf);
    return NULL;
  }
  return sf;
}

void st_close(st_file *sf) {
  if (!sf)
    return;
  if (sf->map && sf->map != MAP_FAILED)
    munmap(sf->map, sf->size);
  if (sf->fd >= 0)
    close(sf->fd);
  free(sf->tensors);
  free(sf);
}

int st_tensor_count(const st_file *sf) { return sf ? sf->n_tensors : 0; }

const st_tensor_t *st_find_tensor(const st_file *sf, const char *name) {
  if (!sf || !name)
    return NULL;
  for (int i = 0; i < sf->n_tensors; i++)
    if (strcmp(sf->tensors[i].name, name) == 0)
      return &sf->tensors[i];
  return NULL;
}

size_t st_tensor_nelems(const st_tensor_t *t) {
  if (!t || t->ndim < 1)
    return 0;
  size_t n = 1;
  for (int i = 0; i < t->ndim; i++)
    n *= (size_t)t->shape[i];
  return n;
}

int st_tensor_to_f32(const st_file *sf, const st_tensor_t *t, float *dst,
                     size_t dst_n) {
  if (!sf || !t || !dst)
    return -1;
  size_t n = st_tensor_nelems(t);
  if (n > dst_n || t->nbytes == 0)
    return -1;
  if (sf->data_off + t->data_end > sf->size)
    return -1;
  const uint8_t *raw = sf->map + sf->data_off + t->data_begin;
  if (t->dtype == ST_DTYPE_F32) {
    if (t->nbytes < n * 4)
      return -1;
    memcpy(dst, raw, n * sizeof(float));
    return 0;
  }
  if (t->dtype == ST_DTYPE_F16) {
    if (t->nbytes < n * 2)
      return -1;
    const uint16_t *h = (const uint16_t *)raw;
    for (size_t i = 0; i < n; i++)
      dst[i] = f16_to_f32(h[i]);
    return 0;
  }
  if (t->dtype == ST_DTYPE_BF16) {
    if (t->nbytes < n * 2)
      return -1;
    const uint16_t *h = (const uint16_t *)raw;
    for (size_t i = 0; i < n; i++)
      dst[i] = bf16_to_f32(h[i]);
    return 0;
  }
  return -1;
}
