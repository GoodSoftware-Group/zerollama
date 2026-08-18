#define _DARWIN_C_SOURCE 1
#include "h3_convrot.h"

#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int h3_fwht_normalized(float *x, int n) {
  if (!x || n < 1 || (n & (n - 1)) != 0)
    return -1;
  for (int len = 1; len < n; len *= 2) {
    int step = len * 2;
    for (int i = 0; i < n; i += step) {
      for (int j = 0; j < len; j++) {
        float a = x[i + j];
        float b = x[i + j + len];
        x[i + j] = a + b;
        x[i + j + len] = a - b;
      }
    }
  }
  float s = 1.0f / sqrtf((float)n);
  for (int i = 0; i < n; i++)
    x[i] *= s;
  return 0;
}

/* ConvRot regular Hadamard: kronecker of
 *   [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]] / sqrt(n)
 * (comfy_kitchen _build_hadamard). Not Sylvester FWHT. */
static int is_pow4(int n) {
  if (n < 4 || (n & (n - 1)) != 0)
    return 0;
  int p = 0;
  for (int t = n; t > 1; t >>= 1)
    p++;
  return (p & 1) == 0;
}

static int fill_regular_hadamard(float *H, int n) {
  if (!H || !is_pow4(n))
    return -1;
  static float *cache_h;
  static int cache_n;
  if (cache_n == n && cache_h) {
    memcpy(H, cache_h, (size_t)n * (size_t)n * sizeof(float));
    return 0;
  }
  const float h4[16] = {1.f,  1.f, 1.f, -1.f, 1.f, 1.f, -1.f, 1.f,
                        1.f, -1.f, 1.f,  1.f, -1.f, 1.f, 1.f, 1.f};
  float *cur = (float *)malloc(16 * sizeof(float));
  if (!cur)
    return -1;
  memcpy(cur, h4, 16 * sizeof(float));
  int cur_n = 4;
  while (cur_n < n) {
    int next_n = cur_n * 4;
    float *nxt = (float *)malloc((size_t)next_n * (size_t)next_n * sizeof(float));
    if (!nxt) {
      free(cur);
      return -1;
    }
    for (int i = 0; i < cur_n; i++) {
      for (int j = 0; j < cur_n; j++) {
        float a = cur[i * cur_n + j];
        for (int u = 0; u < 4; u++) {
          for (int v = 0; v < 4; v++) {
            nxt[(i * 4 + u) * next_n + (j * 4 + v)] = a * h4[u * 4 + v];
          }
        }
      }
    }
    free(cur);
    cur = nxt;
    cur_n = next_n;
  }
  float s = 1.0f / sqrtf((float)n);
  size_t nn = (size_t)n * (size_t)n;
  for (size_t i = 0; i < nn; i++)
    H[i] = cur[i] * s;
  free(cur);
  free(cache_h);
  cache_h = (float *)malloc(nn * sizeof(float));
  if (cache_h) {
    memcpy(cache_h, H, nn * sizeof(float));
    cache_n = n;
  } else {
    cache_n = 0;
  }
  return 0;
}

int h3_convrot_unrotate(float *w, int rows, int cols, int gs) {
  if (!w || rows < 1 || cols < 1)
    return -1;
  if (gs <= 1)
    return 0;
  if (!is_pow4(gs) || cols % gs != 0)
    return -1;
  int ng = cols / gs;
  int m = rows * ng;
  float *H = (float *)malloc((size_t)gs * (size_t)gs * sizeof(float));
  if (!H)
    return -1;
  static float *tmp_hold;
  static size_t tmp_hold_n;
  size_t need = (size_t)m * (size_t)gs;
  if (tmp_hold_n < need) {
    float *nb = (float *)realloc(tmp_hold, need * sizeof(float));
    if (!nb) {
      free(H);
      return -1;
    }
    tmp_hold = nb;
    tmp_hold_n = need;
  }
  int rc = fill_regular_hadamard(H, gs);
  if (rc == 0) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, gs, gs, 1.0f, w, gs,
                H, gs, 0.0f, tmp_hold, gs);
    memcpy(w, tmp_hold, need * sizeof(float));
  }
  free(H);
  return rc;
}

static void dequant_rows_serial(const int8_t *q, int rows, int cols,
                                const float *scale, float *dst) {
  for (int r = 0; r < rows; r++) {
    float s = scale[r];
    const int8_t *qr = q + (size_t)r * (size_t)cols;
    float *dr = dst + (size_t)r * (size_t)cols;
    int c = 0;
    for (; c + 7 < cols; c += 8) {
      dr[c] = (float)qr[c] * s;
      dr[c + 1] = (float)qr[c + 1] * s;
      dr[c + 2] = (float)qr[c + 2] * s;
      dr[c + 3] = (float)qr[c + 3] * s;
      dr[c + 4] = (float)qr[c + 4] * s;
      dr[c + 5] = (float)qr[c + 5] * s;
      dr[c + 6] = (float)qr[c + 6] * s;
      dr[c + 7] = (float)qr[c + 7] * s;
    }
    for (; c < cols; c++)
      dr[c] = (float)qr[c] * s;
  }
}

static void dequant_rows(const int8_t *q, int rows, int cols, const float *scale,
                         float *dst) {
  long ncore = sysconf(_SC_NPROCESSORS_ONLN);
  if (rows < (int)ncore * 4 || ncore < 2) {
    dequant_rows_serial(q, rows, cols, scale, dst);
    return;
  }
  int per = (rows + (int)ncore - 1) / (int)ncore;
  dispatch_apply((size_t)ncore,
                 dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                 ^(size_t c) {
                   int r0 = (int)c * per;
                   if (r0 >= rows)
                     return;
                   int r1 = r0 + per;
                   if (r1 > rows)
                     r1 = rows;
                   dequant_rows_serial(q + (size_t)r0 * (size_t)cols, r1 - r0,
                                       cols, scale + r0,
                                       dst + (size_t)r0 * (size_t)cols);
                 });
}

int h3_convrot_dequant_i8(const int8_t *q, int rows, int cols,
                          const float *scale, int gs, float *dst) {
  if (!q || !scale || !dst || rows < 1 || cols < 1)
    return -1;
  dequant_rows(q, rows, cols, scale, dst);
  return h3_convrot_unrotate(dst, rows, cols, gs);
}

int h3_convrot_fakequant_act(float *x, int rows, int cols, int gs) {
  if (!x || rows < 1 || cols < 1)
    return -1;
  if (gs > 1 && h3_convrot_unrotate(x, rows, cols, gs) != 0)
    return -1;
  for (int r = 0; r < rows; r++) {
    float *row = x + (size_t)r * (size_t)cols;
    float amax = 0.f;
    for (int c = 0; c < cols; c++) {
      float a = fabsf(row[c]);
      if (a > amax)
        amax = a;
    }
    float scale = amax / 127.f;
    if (scale < 1e-30f)
      scale = 1e-30f;
    for (int c = 0; c < cols; c++) {
      float v = row[c] / scale;
      if (v > 127.f)
        v = 127.f;
      else if (v < -128.f)
        v = -128.f;
      row[c] = (float)lrintf(v) * scale;
    }
  }
  if (gs > 1)
    return h3_convrot_unrotate(x, rows, cols, gs);
  return 0;
}

int h3_comfy_quant_parse(const uint8_t *bytes, size_t n, int *out_gs) {
  if (!out_gs)
    return -1;
  *out_gs = 0;
  if (!bytes || n == 0)
    return 0;
  char buf[256];
  size_t m = n < sizeof(buf) - 1 ? n : sizeof(buf) - 1;
  memcpy(buf, bytes, m);
  buf[m] = 0;
  int convrot = 0;
  if (strstr(buf, "\"convrot\": true") || strstr(buf, "\"convrot\":true"))
    convrot = 1;
  const char *p = strstr(buf, "convrot_groupsize");
  int gs = 0;
  if (p) {
    p = strchr(p, ':');
    if (p)
      gs = (int)strtol(p + 1, NULL, 10);
  }
  if (convrot) {
    if (gs <= 0)
      gs = 256;
    *out_gs = gs;
  }
  return 0;
}
