#include "backend_ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void wan_op_layernorm_f32(float *y, const float *x, const float *w, int N, int D,
                          float eps) {
  for (int n = 0; n < N; n++) {
    const float *xr = x + (size_t)n * D;
    float *yr = y + (size_t)n * D;
    float mean = 0.f;
    for (int d = 0; d < D; d++) mean += xr[d];
    mean /= (float)D;
    float var = 0.f;
    for (int d = 0; d < D; d++) {
      float t = xr[d] - mean;
      var += t * t;
    }
    var /= (float)D;
    float inv = 1.f / sqrtf(var + eps);
    for (int d = 0; d < D; d++) {
      float v = (xr[d] - mean) * inv;
      yr[d] = w ? v * w[d] : v;
    }
  }
}

void wan_op_affine_mul_add_f32(float *y, const float *x, const float *scale,
                               const float *shift, int N, int D) {
  for (int n = 0; n < N; n++) {
    const float *xr = x + (size_t)n * D;
    float *yr = y + (size_t)n * D;
    for (int d = 0; d < D; d++)
      yr[d] = xr[d] * (1.f + scale[d]) + shift[d];
  }
}

void wan_op_gelu_tanh_f32(float *y, const float *x, size_t n) {
  for (size_t i = 0; i < n; i++) {
    float v = x[i];
    float c = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    y[i] = 0.5f * v * (1.f + tanhf(c));
  }
}

void wan_op_gated_residual_f32(float *y, const float *x, const float *delta,
                               const float *gate, int N, int D) {
  for (int n = 0; n < N; n++) {
    const float *xr = x + (size_t)n * D;
    const float *dr = delta + (size_t)n * D;
    float *yr = y + (size_t)n * D;
    for (int d = 0; d < D; d++) yr[d] = xr[d] + dr[d] * gate[d];
  }
}

void wan_op_rmsnorm_f32(float *y, const float *x, const float *w, int N, int D,
                        float eps) {
  for (int n = 0; n < N; n++) {
    const float *xr = x + (size_t)n * D;
    float *yr = y + (size_t)n * D;
    float ss = 0.f;
    for (int d = 0; d < D; d++) ss += xr[d] * xr[d];
    float inv = 1.f / sqrtf(ss / (float)D + eps);
    for (int d = 0; d < D; d++) {
      float v = xr[d] * inv;
      yr[d] = w ? v * w[d] : v;
    }
  }
}

void wan_op_bias_add_f32(float *y, const float *x, const float *bias, int N,
                         int D) {
  for (int n = 0; n < N; n++) {
    const float *xr = x + (size_t)n * D;
    float *yr = y + (size_t)n * D;
    for (int d = 0; d < D; d++) yr[d] = xr[d] + bias[d];
  }
}

void wan_op_scale_bias_f32(float *y, const float *x, const float *scale,
                           const float *bias, int N, int D) {
  for (int n = 0; n < N; n++) {
    const float *xr = x + (size_t)n * D;
    float *yr = y + (size_t)n * D;
    for (int d = 0; d < D; d++) yr[d] = xr[d] * scale[d] + bias[d];
  }
}

void wan_op_transpose_oi_f32(float *dst, const float *src, int out, int in) {
  for (int o = 0; o < out; o++)
    for (int i = 0; i < in; i++)
      dst[(size_t)i * (size_t)out + (size_t)o] =
          src[(size_t)o * (size_t)in + (size_t)i];
}

void wan_op_head_rmsnorm_f32(float *y, const float *x, const float *w, int T,
                             int H, int HD, float eps) {
  int rows = T * H;
  for (int r = 0; r < rows; r++) {
    const float *xr = x + (size_t)r * HD;
    float *yr = y + (size_t)r * HD;
    float ss = 0.f;
    for (int d = 0; d < HD; d++) ss += xr[d] * xr[d];
    float inv = 1.f / sqrtf(ss / (float)HD + eps);
    for (int d = 0; d < HD; d++) {
      float v = xr[d] * inv;
      yr[d] = w ? v * w[d] : v;
    }
  }
}

static void rope_hd_split(int HD, int *d0, int *d1, int *d2) {
  int c = HD / 2;
  *d0 = 2 * (c - 2 * (c / 3));
  *d1 = 2 * (c / 3);
  *d2 = 2 * (c / 3);
  if (*d0 < 2) {
    *d0 = (HD / 6) * 2;
    *d1 = (HD / 6) * 2;
    *d2 = HD - *d0 - *d1;
  }
}

static void fill_rope_freqs(float *freq, int npos, int dim) {
  int half = dim / 2;
  for (int p = 0; p < npos; p++) {
    for (int i = 0; i < half; i++) {
      float ang = (float)p * powf(10000.f, -2.f * (float)i / (float)dim);
      freq[p * dim + 2 * i] = cosf(ang);
      freq[p * dim + 2 * i + 1] = sinf(ang);
    }
  }
}

static void rope_axis_apply(float *x, int dim, const float *freq, int pos) {
  int half = dim / 2;
  const float *f = freq + (size_t)pos * (size_t)dim;
  for (int i = 0; i < half; i++) {
    float cosv = f[2 * i], sinv = f[2 * i + 1];
    float a = x[2 * i], b = x[2 * i + 1];
    x[2 * i] = a * cosv - b * sinv;
    x[2 * i + 1] = a * sinv + b * cosv;
  }
}

int wan_op_rope3_f32(float *y, const float *x, int T, int H, int HD, int grid_t,
                     int grid_h, int grid_w) {
  if (!y || !x || T < 1 || H < 1 || HD < 2 || (HD % 2) != 0) return -1;
  if (grid_t < 1) grid_t = T;
  if (grid_h < 1) grid_h = 1;
  if (grid_w < 1) grid_w = 1;
  if ((size_t)grid_t * (size_t)grid_h * (size_t)grid_w != (size_t)T) return -1;
  size_t n = (size_t)T * (size_t)H * (size_t)HD;
  if (y != x) memcpy(y, x, n * sizeof(float));

  int d0, d1, d2;
  rope_hd_split(HD, &d0, &d1, &d2);
  float *ft = calloc((size_t)grid_t * (size_t)(d0 > 0 ? d0 : 2), sizeof(float));
  float *fh = calloc((size_t)grid_h * (size_t)(d1 > 0 ? d1 : 2), sizeof(float));
  float *fw = calloc((size_t)grid_w * (size_t)(d2 > 0 ? d2 : 2), sizeof(float));
  if (!ft || !fh || !fw) {
    free(ft);
    free(fh);
    free(fw);
    return -1;
  }
  if (d0 >= 2) fill_rope_freqs(ft, grid_t, d0);
  if (d1 >= 2) fill_rope_freqs(fh, grid_h, d1);
  if (d2 >= 2) fill_rope_freqs(fw, grid_w, d2);

  int spat = grid_h * grid_w;
  for (int idx = 0; idx < T; idx++) {
    int ow = idx % grid_w;
    int oh = (idx / grid_w) % grid_h;
    int od = spat > 0 ? (idx / spat) : 0;
    for (int h = 0; h < H; h++) {
      float *row = y + ((size_t)idx * (size_t)H + (size_t)h) * (size_t)HD;
      if (d0 >= 2) rope_axis_apply(row, d0, ft, od);
      if (d1 >= 2) rope_axis_apply(row + d0, d1, fh, oh);
      if (d2 >= 2) rope_axis_apply(row + d0 + d1, d2, fw, ow);
    }
  }
  free(ft);
  free(fh);
  free(fw);
  return 0;
}

void wan_op_attn_sdpa_f32(float *out, const float *q, const float *k,
                          const float *v, int T, int Tk, int H, int HD) {
  float scale = 1.f / sqrtf((float)HD);
  float *scores = malloc((size_t)Tk * sizeof(float));
  if (!scores) return;
  for (int h = 0; h < H; h++) {
    for (int t = 0; t < T; t++) {
      const float *qr = q + ((size_t)t * H + h) * HD;
      float maxv = -1e30f;
      for (int s = 0; s < Tk; s++) {
        const float *kr = k + ((size_t)s * H + h) * HD;
        float dot = 0.f;
        for (int d = 0; d < HD; d++) dot += qr[d] * kr[d];
        scores[s] = dot * scale;
        if (scores[s] > maxv) maxv = scores[s];
      }
      float sum = 0.f;
      for (int s = 0; s < Tk; s++) {
        scores[s] = expf(scores[s] - maxv);
        sum += scores[s];
      }
      float inv = 1.f / sum;
      float *orow = out + ((size_t)t * H + h) * HD;
      for (int d = 0; d < HD; d++) orow[d] = 0.f;
      for (int s = 0; s < Tk; s++) {
        float a = scores[s] * inv;
        const float *vr = v + ((size_t)s * H + h) * HD;
        for (int d = 0; d < HD; d++) orow[d] += a * vr[d];
      }
    }
  }
  free(scores);
}

int wan_op_patch_embed_f32(float *tok, const float *x, const float *w,
                           const float *bias, int Cin, int Tin, int Hin, int Win,
                           int Cout, int kt, int kh, int kw) {
  if (!tok || !x || !w || Cin < 1 || Tin < 1 || Hin < 1 || Win < 1 || Cout < 1 ||
      kt < 1 || kh < 1 || kw < 1)
    return -1;
  if ((Tin % kt) || (Hin % kh) || (Win % kw))
    return -1;
  int tp = Tin / kt, hp = Hin / kh, wp = Win / kw;
  for (int ot = 0; ot < tp; ot++) {
    for (int oh = 0; oh < hp; oh++) {
      for (int ow = 0; ow < wp; ow++) {
        size_t ti =
            (((size_t)ot * (size_t)hp + (size_t)oh) * (size_t)wp + (size_t)ow);
        for (int oc = 0; oc < Cout; oc++) {
          double s = bias ? (double)bias[oc] : 0.0;
          for (int ic = 0; ic < Cin; ic++) {
            for (int kti = 0; kti < kt; kti++) {
              for (int khi = 0; khi < kh; khi++) {
                for (int kwi = 0; kwi < kw; kwi++) {
                  int it = ot * kt + kti;
                  int ih = oh * kh + khi;
                  int iw = ow * kw + kwi;
                  size_t xi =
                      (((((size_t)ic * (size_t)Tin + (size_t)it) * (size_t)Hin +
                         (size_t)ih) *
                        (size_t)Win) +
                       (size_t)iw);
                  size_t wi =
                      ((((((size_t)oc * (size_t)Cin + (size_t)ic) * (size_t)kt +
                          (size_t)kti) *
                         (size_t)kh +
                         (size_t)khi) *
                        (size_t)kw) +
                       (size_t)kwi);
                  s += (double)x[xi] * (double)w[wi];
                }
              }
            }
          }
          tok[ti * (size_t)Cout + (size_t)oc] = (float)s;
        }
      }
    }
  }
  return 0;
}
