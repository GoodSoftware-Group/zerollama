#define _DARWIN_C_SOURCE 1
#include "h3_dit_host.h"

#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int h3_dit_timestep_embedding(const float *timesteps, int n, int dim,
                              float max_period, int flip_sin_to_cos,
                              float *out) {
  if (!timesteps || !out || n < 1 || dim < 2 || (dim % 2) != 0 ||
      max_period <= 0.0f)
    return -1;
  int half = dim / 2;
  for (int i = 0; i < n; i++) {
    float t = timesteps[i];
    float *row = out + (size_t)i * (size_t)dim;
    for (int j = 0; j < half; j++) {
      double exponent =
          -log((double)max_period) * (double)j / (double)half;
      double freq = exp(exponent);
      double angle = (double)t * freq;
      float s = (float)sin(angle);
      float c = (float)cos(angle);
      if (flip_sin_to_cos) {
        row[j] = c;
        row[half + j] = s;
      } else {
        row[j] = s;
        row[half + j] = c;
      }
    }
  }
  return 0;
}

int h3_dit_timestep_embedding_default(const float *timesteps, int n,
                                      float *out) {
  return h3_dit_timestep_embedding(timesteps, n, H3_DIT_TIMESTEP_INPUT_DIM,
                                   10000.0f, 1, out);
}

int h3_dit_rope_inv_freq(float *inv_freq, int n, float theta) {
  if (!inv_freq || n < 1 || theta <= 0.0f)
    return -1;
  for (int i = 0; i < n; i++) {
    double expn = (double)(2 * i) / (double)(2 * n);
    inv_freq[i] = (float)(1.0 / pow((double)theta, expn));
  }
  return 0;
}

int h3_dit_rope_from_positions(const float *position_ids, int seq,
                               const float *inv_freq, int n_freq, float *cos,
                               float *sin) {
  if (!position_ids || !inv_freq || !cos || !sin || seq < 1 || n_freq < 1)
    return -1;
  int rotary = 2 * 3 * n_freq;
  for (int s = 0; s < seq; s++) {
    const float *pos = position_ids + (size_t)s * 3;
    float freqs[3 * 64];
    if (n_freq > 64)
      return -1;
    for (int axis = 0; axis < 3; axis++) {
      for (int i = 0; i < n_freq; i++)
        freqs[axis * n_freq + i] = pos[axis] * inv_freq[i];
    }
    /* concat axes then duplicate for rotate-half */
    float *c = cos + (size_t)s * (size_t)rotary;
    float *sn = sin + (size_t)s * (size_t)rotary;
    int half = 3 * n_freq;
    for (int i = 0; i < half; i++) {
      float a = freqs[i];
      c[i] = cosf(a);
      sn[i] = sinf(a);
      c[half + i] = c[i];
      sn[half + i] = sn[i];
    }
  }
  return 0;
}

int h3_dit_apply_rotary(const float *x, int seq, int head_dim, const float *cos,
                        const float *sin, int rotary_dim, float *out) {
  if (!x || !cos || !sin || !out || seq < 1 || head_dim < 1 || rotary_dim < 1 ||
      rotary_dim > head_dim || (rotary_dim % 2) != 0)
    return -1;
  int half = rotary_dim / 2;
  for (int s = 0; s < seq; s++) {
    const float *xr = x + (size_t)s * (size_t)head_dim;
    float *o = out + (size_t)s * (size_t)head_dim;
    const float *c = cos + (size_t)s * (size_t)rotary_dim;
    const float *sn = sin + (size_t)s * (size_t)rotary_dim;
    for (int i = 0; i < half; i++) {
      float x1 = xr[i];
      float x2 = xr[half + i];
      /* rotated = [-x2, x1]; out = x_rot * cos + rotated * sin */
      o[i] = x1 * c[i] + (-x2) * sn[i];
      o[half + i] = x2 * c[half + i] + x1 * sn[half + i];
    }
    for (int i = rotary_dim; i < head_dim; i++)
      o[i] = xr[i];
  }
  return 0;
}

int h3_dit_condition_proj(const float *text, int seq, int text_dim, int hidden,
                          const float *weight, const float *bias, float *out) {
  if (!text || !weight || !out || seq < 1 || text_dim < 1 || hidden < 1)
    return -1;
  /* out = text @ W^T + b ; W is [hidden, text_dim] */
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, seq, hidden, text_dim,
              1.0f, text, text_dim, weight, text_dim, 0.0f, out, hidden);
  if (bias) {
    for (int s = 0; s < seq; s++) {
      float *row = out + (size_t)s * (size_t)hidden;
      for (int j = 0; j < hidden; j++)
        row[j] += bias[j];
    }
  }
  return 0;
}

static size_t lat_index(int B, int C, int F, int H, int W, int b, int c, int f,
                        int h, int w) {
  (void)B;
  return (((((size_t)b * (size_t)C + (size_t)c) * (size_t)F + (size_t)f) *
               (size_t)H +
           (size_t)h) *
              (size_t)W +
          (size_t)w);
}

int h3_dit_patchify_video(const float *latents, int B, int C, int F, int H,
                          int W, int pt, int ph, int pw, float *rows) {
  if (!latents || !rows || B < 1 || C < 1 || F < 1 || H < 1 || W < 1 ||
      pt < 1 || ph < 1 || pw < 1)
    return -1;
  if ((F % pt) || (H % ph) || (W % pw))
    return -1;
  int Ft = F / pt, Ht = H / ph, Wt = W / pw;
  int row_dim = C * pt * ph * pw;
  size_t row_i = 0;
  for (int b = 0; b < B; b++) {
    for (int ft = 0; ft < Ft; ft++) {
      for (int ht = 0; ht < Ht; ht++) {
        for (int wt = 0; wt < Wt; wt++) {
          float *dst = rows + row_i * (size_t)row_dim;
          int o = 0;
          for (int c = 0; c < C; c++) {
            for (int t = 0; t < pt; t++) {
              for (int hh = 0; hh < ph; hh++) {
                for (int ww = 0; ww < pw; ww++) {
                  dst[o++] = latents[lat_index(
                      B, C, F, H, W, b, c, ft * pt + t, ht * ph + hh,
                      wt * pw + ww)];
                }
              }
            }
          }
          row_i++;
        }
      }
    }
  }
  return 0;
}

int h3_dit_unpatchify_video(const float *rows, int B, int C, int F, int H,
                            int W, int pt, int ph, int pw, float *latents) {
  if (!rows || !latents || B < 1 || C < 1 || F < 1 || H < 1 || W < 1 || pt < 1 ||
      ph < 1 || pw < 1)
    return -1;
  if ((F % pt) || (H % ph) || (W % pw))
    return -1;
  int Ft = F / pt, Ht = H / ph, Wt = W / pw;
  int row_dim = C * pt * ph * pw;
  size_t row_i = 0;
  for (int b = 0; b < B; b++) {
    for (int ft = 0; ft < Ft; ft++) {
      for (int ht = 0; ht < Ht; ht++) {
        for (int wt = 0; wt < Wt; wt++) {
          const float *src = rows + row_i * (size_t)row_dim;
          int o = 0;
          for (int c = 0; c < C; c++) {
            for (int t = 0; t < pt; t++) {
              for (int hh = 0; hh < ph; hh++) {
                for (int ww = 0; ww < pw; ww++) {
                  latents[lat_index(B, C, F, H, W, b, c, ft * pt + t,
                                    ht * ph + hh, wt * pw + ww)] = src[o++];
                }
              }
            }
          }
          row_i++;
        }
      }
    }
  }
  return 0;
}

int h3_dit_pack_audio(const float *latents_2ct, int C, int T, float *rows) {
  /* latents (2,C,T) → rows (2*T, C) channel-major */
  if (!latents_2ct || !rows || C < 1 || T < 1)
    return -1;
  for (int ch = 0; ch < 2; ch++) {
    for (int t = 0; t < T; t++) {
      float *dst = rows + ((size_t)ch * (size_t)T + (size_t)t) * (size_t)C;
      for (int c = 0; c < C; c++)
        dst[c] = latents_2ct[((size_t)ch * (size_t)C + (size_t)c) * (size_t)T +
                             (size_t)t];
    }
  }
  return 0;
}

int h3_dit_unpack_audio(const float *rows, int C, int T, float *latents_2ct) {
  if (!rows || !latents_2ct || C < 1 || T < 1)
    return -1;
  for (int ch = 0; ch < 2; ch++) {
    for (int t = 0; t < T; t++) {
      const float *src =
          rows + ((size_t)ch * (size_t)T + (size_t)t) * (size_t)C;
      for (int c = 0; c < C; c++)
        latents_2ct[((size_t)ch * (size_t)C + (size_t)c) * (size_t)T +
                    (size_t)t] = src[c];
    }
  }
  return 0;
}

void h3_dit_silu(const float *x, float *y, size_t n) {
  for (size_t i = 0; i < n; i++)
    y[i] = x[i] / (1.0f + expf(-x[i]));
}

void h3_dit_silu_mul(const float *gate, const float *up, float *out, size_t n) {
  for (size_t i = 0; i < n; i++)
    out[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
}

int h3_dit_rmsnorm(const float *x, int rows, int dim, float eps,
                   const float *weight, float *out) {
  if (!x || !out || rows < 1 || dim < 1)
    return -1;
  for (int r = 0; r < rows; r++) {
    const float *xr = x + (size_t)r * (size_t)dim;
    float *orow = out + (size_t)r * (size_t)dim;
    double acc = 0.0;
    for (int i = 0; i < dim; i++)
      acc += (double)xr[i] * (double)xr[i];
    float inv = 1.0f / sqrtf((float)(acc / (double)dim) + eps);
    for (int i = 0; i < dim; i++) {
      float v = xr[i] * inv;
      orow[i] = weight ? v * weight[i] : v;
    }
  }
  return 0;
}

int h3_dit_modulate(const float *x, const float *shift, const float *scale,
                    int rows, int dim, float *out) {
  if (!x || !shift || !scale || !out || rows < 1 || dim < 1)
    return -1;
  for (int r = 0; r < rows; r++) {
    const float *xr = x + (size_t)r * (size_t)dim;
    float *orow = out + (size_t)r * (size_t)dim;
    for (int i = 0; i < dim; i++)
      orow[i] = xr[i] * (1.0f + scale[i]) + shift[i];
  }
  return 0;
}

int h3_dit_gated_residual(const float *x, const float *y, const float *gate,
                          int rows, int dim, float *out) {
  if (!x || !y || !gate || !out || rows < 1 || dim < 1)
    return -1;
  for (int r = 0; r < rows; r++) {
    const float *xr = x + (size_t)r * (size_t)dim;
    const float *yr = y + (size_t)r * (size_t)dim;
    float *orow = out + (size_t)r * (size_t)dim;
    for (int i = 0; i < dim; i++)
      orow[i] = xr[i] + gate[i] * yr[i];
  }
  return 0;
}

const float h3_dit_pixel_mean[3] = {0.485f, 0.456f, 0.406f};
const float h3_dit_pixel_std[3] = {0.229f, 0.224f, 0.225f};

void h3_dit_pixel_normalize(const float *rgb01, float *out, size_t n_pixels) {
  for (size_t p = 0; p < n_pixels; p++) {
    const float *in = rgb01 + p * 3;
    float *o = out + p * 3;
    o[0] = (in[0] - h3_dit_pixel_mean[0]) / h3_dit_pixel_std[0];
    o[1] = (in[1] - h3_dit_pixel_mean[1]) / h3_dit_pixel_std[1];
    o[2] = (in[2] - h3_dit_pixel_mean[2]) / h3_dit_pixel_std[2];
  }
}

void h3_dit_pixel_denormalize(const float *norm, float *rgb01, size_t n_pixels) {
  for (size_t p = 0; p < n_pixels; p++) {
    const float *in = norm + p * 3;
    float *o = rgb01 + p * 3;
    o[0] = in[0] * h3_dit_pixel_std[0] + h3_dit_pixel_mean[0];
    o[1] = in[1] * h3_dit_pixel_std[1] + h3_dit_pixel_mean[1];
    o[2] = in[2] * h3_dit_pixel_std[2] + h3_dit_pixel_mean[2];
  }
}

int h3_dit_linear(const float *x, int rows, int in_dim, int out_dim,
                  const float *weight, const float *bias, float *out) {
  return h3_dit_condition_proj(x, rows, in_dim, out_dim, weight, bias, out);
}

int h3_dit_timestep_mlp(const float *emb, int n, int in_dim, int hidden,
                        int out_dim, const float *w_in, const float *b_in,
                        const float *w_out, const float *b_out, float *out) {
  if (!emb || !w_in || !w_out || !out || n < 1 || in_dim < 1 || hidden < 1 ||
      out_dim < 1)
    return -1;
  float *mid = (float *)malloc((size_t)n * (size_t)hidden * sizeof(float));
  float *act = (float *)malloc((size_t)n * (size_t)hidden * sizeof(float));
  if (!mid || !act) {
    free(mid);
    free(act);
    return -1;
  }
  int rc = h3_dit_linear(emb, n, in_dim, hidden, w_in, b_in, mid);
  if (rc == 0)
    h3_dit_silu(mid, act, (size_t)n * (size_t)hidden);
  if (rc == 0)
    rc = h3_dit_linear(act, n, hidden, out_dim, w_out, b_out, out);
  free(mid);
  free(act);
  return rc;
}

int h3_dit_swiglu_ffn(const float *x, int rows, int dim, int ffn_hidden,
                      const float *fc1_w, const float *fc1_b, const float *fc2_w,
                      const float *fc2_b, float *out) {
  if (!x || !fc1_w || !fc2_w || !out || rows < 1 || dim < 1 || ffn_hidden < 1)
    return -1;
  size_t fused_n = (size_t)rows * (size_t)ffn_hidden * 2;
  float *fused = (float *)malloc(fused_n * sizeof(float));
  float *hidden = (float *)malloc((size_t)rows * (size_t)ffn_hidden * sizeof(float));
  if (!fused || !hidden) {
    free(fused);
    free(hidden);
    return -1;
  }
  int rc = h3_dit_linear(x, rows, dim, ffn_hidden * 2, fc1_w, fc1_b, fused);
  if (rc == 0) {
    for (int r = 0; r < rows; r++) {
      const float *gate = fused + (size_t)r * (size_t)ffn_hidden * 2;
      const float *value = gate + ffn_hidden;
      float *h = hidden + (size_t)r * (size_t)ffn_hidden;
      h3_dit_silu_mul(gate, value, h, (size_t)ffn_hidden);
    }
    rc = h3_dit_linear(hidden, rows, ffn_hidden, dim, fc2_w, fc2_b, out);
  }
  free(fused);
  free(hidden);
  return rc;
}

int h3_dit_qkv_split_interleaved(const float *qkv, int rows, int heads,
                                 int head_dim, float *query, float *key,
                                 float *value) {
  if (!qkv || !query || !key || !value || rows < 1 || heads < 1 || head_dim < 1)
    return -1;
  size_t inner = (size_t)heads * 3 * (size_t)head_dim;
  for (int r = 0; r < rows; r++) {
    const float *src = qkv + (size_t)r * inner;
    for (int h = 0; h < heads; h++) {
      const float *blk = src + (size_t)h * 3 * (size_t)head_dim;
      float *q = query + ((size_t)r * heads + h) * (size_t)head_dim;
      float *k = key + ((size_t)r * heads + h) * (size_t)head_dim;
      float *v = value + ((size_t)r * heads + h) * (size_t)head_dim;
      memcpy(q, blk, (size_t)head_dim * sizeof(float));
      memcpy(k, blk + head_dim, (size_t)head_dim * sizeof(float));
      memcpy(v, blk + 2 * head_dim, (size_t)head_dim * sizeof(float));
    }
  }
  return 0;
}

int h3_dit_qkv_split_concat(const float *qkv, int rows, int heads, int head_dim,
                            float *query, float *key, float *value) {
  if (!qkv || !query || !key || !value || rows < 1 || heads < 1 || head_dim < 1)
    return -1;
  size_t inner = (size_t)heads * (size_t)head_dim;
  size_t row_stride = inner * 3;
  for (int r = 0; r < rows; r++) {
    const float *src = qkv + (size_t)r * row_stride;
    const float *qs = src;
    const float *ks = src + inner;
    const float *vs = src + 2 * inner;
    for (int h = 0; h < heads; h++) {
      size_t off = (size_t)h * (size_t)head_dim;
      float *q = query + ((size_t)r * heads + h) * (size_t)head_dim;
      float *k = key + ((size_t)r * heads + h) * (size_t)head_dim;
      float *v = value + ((size_t)r * heads + h) * (size_t)head_dim;
      memcpy(q, qs + off, (size_t)head_dim * sizeof(float));
      memcpy(k, ks + off, (size_t)head_dim * sizeof(float));
      memcpy(v, vs + off, (size_t)head_dim * sizeof(float));
    }
  }
  return 0;
}

int h3_dit_qkv_split(const float *qkv, int rows, int heads, int head_dim,
                     float *query, float *key, float *value) {
  const char *e = getenv("H3_QKV_INTERLEAVED");
  if (e && e[0] && e[0] != '0')
    return h3_dit_qkv_split_interleaved(qkv, rows, heads, head_dim, query, key,
                                        value);
  return h3_dit_qkv_split_concat(qkv, rows, heads, head_dim, query, key, value);
}

int h3_dit_modulate_indexed(const float *x, const float *shift,
                            const float *scale, const int *indices, int rows,
                            int dim, float *out) {
  if (!x || !shift || !scale || !indices || !out || rows < 1 || dim < 1)
    return -1;
  for (int r = 0; r < rows; r++) {
    int idx = indices[r];
    if (idx < 0)
      return -1;
    const float *xr = x + (size_t)r * (size_t)dim;
    const float *sh = shift + (size_t)idx * (size_t)dim;
    const float *sc = scale + (size_t)idx * (size_t)dim;
    float *orow = out + (size_t)r * (size_t)dim;
    for (int i = 0; i < dim; i++)
      orow[i] = xr[i] * (1.0f + sc[i]) + sh[i];
  }
  return 0;
}

int h3_dit_gated_residual_indexed(const float *x, const float *y,
                                  const float *gate, const int *indices,
                                  int rows, int dim, float *out) {
  if (!x || !y || !gate || !indices || !out || rows < 1 || dim < 1)
    return -1;
  for (int r = 0; r < rows; r++) {
    int idx = indices[r];
    if (idx < 0)
      return -1;
    const float *xr = x + (size_t)r * (size_t)dim;
    const float *yr = y + (size_t)r * (size_t)dim;
    const float *g = gate + (size_t)idx * (size_t)dim;
    float *orow = out + (size_t)r * (size_t)dim;
    for (int i = 0; i < dim; i++)
      orow[i] = xr[i] + g[i] * yr[i];
  }
  return 0;
}

int h3_dit_apply_rotary_heads(const float *x, int seq, int heads, int head_dim,
                              const float *cos, const float *sin,
                              int rotary_dim, float *out) {
  if (!x || !cos || !sin || !out || seq < 1 || heads < 1)
    return -1;
  float *tmp = (float *)malloc((size_t)seq * (size_t)head_dim * sizeof(float));
  float *rot = (float *)malloc((size_t)seq * (size_t)head_dim * sizeof(float));
  if (!tmp || !rot) {
    free(tmp);
    free(rot);
    return -1;
  }
  int rc = 0;
  for (int h = 0; h < heads && rc == 0; h++) {
    for (int s = 0; s < seq; s++)
      memcpy(tmp + (size_t)s * (size_t)head_dim,
             x + ((size_t)s * heads + h) * (size_t)head_dim,
             (size_t)head_dim * sizeof(float));
    rc = h3_dit_apply_rotary(tmp, seq, head_dim, cos, sin, rotary_dim, rot);
    for (int s = 0; s < seq && rc == 0; s++)
      memcpy(out + ((size_t)s * heads + h) * (size_t)head_dim,
             rot + (size_t)s * (size_t)head_dim,
             (size_t)head_dim * sizeof(float));
  }
  free(tmp);
  free(rot);
  return rc;
}

int h3_dit_sdpa_f32(float *out, const float *q, const float *k, const float *v,
                    int seq, int heads, int head_dim, float scale) {
  if (!out || !q || !k || !v || seq < 1 || heads < 1 || head_dim < 1)
    return -1;
  long ncore = sysconf(_SC_NPROCESSORS_ONLN);
  if (ncore < 2 || seq < 64 || heads < 4) {
    float *scores = (float *)malloc((size_t)seq * sizeof(float));
    if (!scores)
      return -1;
    for (int h = 0; h < heads; h++) {
      for (int row = 0; row < seq; row++) {
        const float *qr =
            q + ((size_t)row * (size_t)heads + (size_t)h) * (size_t)head_dim;
        float m = -1e30f;
        for (int col = 0; col < seq; col++) {
          const float *kr =
              k + ((size_t)col * (size_t)heads + (size_t)h) * (size_t)head_dim;
          double dot = 0.0;
          for (int d = 0; d < head_dim; d++)
            dot += (double)qr[d] * (double)kr[d];
          float s = (float)dot * scale;
          scores[col] = s;
          if (s > m)
            m = s;
        }
        double l = 0.0;
        for (int col = 0; col < seq; col++) {
          scores[col] = expf(scores[col] - m);
          l += scores[col];
        }
        float inv = (float)(1.0 / l);
        float *orow =
            out + ((size_t)row * (size_t)heads + (size_t)h) * (size_t)head_dim;
        for (int d = 0; d < head_dim; d++)
          orow[d] = 0.f;
        for (int col = 0; col < seq; col++) {
          const float *vr =
              v + ((size_t)col * (size_t)heads + (size_t)h) * (size_t)head_dim;
          float w = scores[col] * inv;
          for (int d = 0; d < head_dim; d++)
            orow[d] += w * vr[d];
        }
      }
    }
    free(scores);
    return 0;
  }
  /* Large seq: parallelize over heads. Each (head,row) output is computed with
   * exactly the same loop order as the serial path, so results are bit-identical. */
  float *scores = (float *)malloc((size_t)heads * (size_t)seq * sizeof(float));
  if (!scores)
    return -1;
  dispatch_apply((size_t)heads,
                 dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                 ^(size_t h) {
                   float *sh = scores + h * (size_t)seq;
                   for (int row = 0; row < seq; row++) {
                     const float *qr =
                         q + ((size_t)row * (size_t)heads + h) * (size_t)head_dim;
                     float m = -1e30f;
                     for (int col = 0; col < seq; col++) {
                       const float *kr = k +
                                         ((size_t)col * (size_t)heads + h) *
                                             (size_t)head_dim;
                       double dot = 0.0;
                       for (int d = 0; d < head_dim; d++)
                         dot += (double)qr[d] * (double)kr[d];
                       float s = (float)dot * scale;
                       sh[col] = s;
                       if (s > m)
                         m = s;
                     }
                     double l = 0.0;
                     for (int col = 0; col < seq; col++) {
                       sh[col] = expf(sh[col] - m);
                       l += sh[col];
                     }
                     float inv = (float)(1.0 / l);
                     float *orow =
                         out + ((size_t)row * (size_t)heads + h) * (size_t)head_dim;
                     for (int d = 0; d < head_dim; d++)
                       orow[d] = 0.f;
                     for (int col = 0; col < seq; col++) {
                       const float *vr = v +
                                         ((size_t)col * (size_t)heads + h) *
                                             (size_t)head_dim;
                       float w = sh[col] * inv;
                       for (int d = 0; d < head_dim; d++)
                         orow[d] += w * vr[d];
                     }
                   }
                 });
  free(scores);
  return 0;
}
