#include "h3_qwen_te_host.h"

#include "h3_audio_vae_host.h"
#include "h3_dit_host.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int h3_qwen_te_rope_tables(size_t tokens, int head_dim, float theta,
                           const uint32_t *position_ids, float *cos,
                           float *sin) {
  if (!cos || !sin || tokens < 1 || head_dim < 2 || (head_dim % 2) != 0 ||
      theta <= 0.f)
    return -1;
  int half = head_dim / 2;
  float *inv = (float *)malloc((size_t)half * sizeof(float));
  if (!inv)
    return -1;
  for (int i = 0; i < half; i++)
    inv[i] = 1.f / powf(theta, (float)(i * 2) / (float)head_dim);
  for (size_t t = 0; t < tokens; t++) {
    for (int i = 0; i < half; i++) {
      size_t axis = 0;
      if (position_ids && i < H3_QWEN_TE_MROPE_FIRST) {
        if (i % 3 == 1)
          axis = 1;
        else if (i % 3 == 2)
          axis = 2;
      }
      float coord = position_ids ? (float)position_ids[axis * tokens + t]
                                 : (float)t;
      float angle = coord * inv[i];
      cos[t * (size_t)half + (size_t)i] = cosf(angle);
      sin[t * (size_t)half + (size_t)i] = sinf(angle);
    }
  }
  free(inv);
  return 0;
}

int h3_qwen_te_apply_rope(float *x, size_t tokens, int heads, int head_dim,
                          const float *cos, const float *sin) {
  if (!x || !cos || !sin || tokens < 1 || heads < 1 || head_dim < 2 ||
      (head_dim % 2) != 0)
    return -1;
  int half = head_dim / 2;
  for (size_t t = 0; t < tokens; t++) {
    const float *c = cos + t * (size_t)half;
    const float *s = sin + t * (size_t)half;
    for (int h = 0; h < heads; h++) {
      float *row = x + ((t * (size_t)heads + (size_t)h) * (size_t)head_dim);
      for (int i = 0; i < half; i++) {
        float x0 = row[i];
        float x1 = row[half + i];
        row[i] = x0 * c[i] - x1 * s[i];
        row[half + i] = x1 * c[i] + x0 * s[i];
      }
    }
  }
  return 0;
}

int h3_qwen_te_head_rmsnorm(float *x, size_t tokens, int heads, int dim,
                            float eps, const float *weight) {
  if (!x || tokens < 1 || heads < 1 || dim < 1 || eps <= 0.f)
    return -1;
  for (size_t t = 0; t < tokens; t++) {
    for (int h = 0; h < heads; h++) {
      float *row = x + ((t * (size_t)heads + (size_t)h) * (size_t)dim);
      double acc = 0.0;
      for (int i = 0; i < dim; i++)
        acc += (double)row[i] * (double)row[i];
      float inv = 1.f / sqrtf((float)(acc / (double)dim) + eps);
      for (int i = 0; i < dim; i++) {
        float y = row[i] * inv;
        row[i] = weight ? y * weight[i] : y;
      }
    }
  }
  return 0;
}

int h3_qwen_te_repeat_kv(const float *src, size_t tokens, int n_kv, int n_q,
                         int head_dim, float *dst) {
  if (!src || !dst || tokens < 1 || n_kv < 1 || n_q < n_kv || head_dim < 1)
    return -1;
  if (n_q % n_kv)
    return -1;
  int rep = n_q / n_kv;
  for (size_t t = 0; t < tokens; t++) {
    for (int hq = 0; hq < n_q; hq++) {
      int hk = hq / rep;
      memcpy(dst + ((t * (size_t)n_q + (size_t)hq) * (size_t)head_dim),
             src + ((t * (size_t)n_kv + (size_t)hk) * (size_t)head_dim),
             (size_t)head_dim * sizeof(float));
    }
  }
  return 0;
}

int h3_qwen_te_layer(float *hidden, size_t tokens, int hidden_dim, int n_q,
                     int n_kv, int head_dim, int ffn, const float *input_norm,
                     const float *Wq, const float *Wk, const float *Wv,
                     const float *Wo, const float *q_norm, const float *k_norm,
                     const float *post_norm, const float *Wgate,
                     const float *Wup, const float *Wdown,
                     const float *rope_cos, const float *rope_sin) {
  if (!hidden || !input_norm || !Wq || !Wk || !Wv || !Wo || !q_norm ||
      !k_norm || !post_norm || !Wgate || !Wup || !Wdown || !rope_cos ||
      !rope_sin || tokens < 1)
    return -1;
  int q_dim = n_q * head_dim;
  int kv_dim = n_kv * head_dim;
  size_t T = tokens;
  float *norm = (float *)malloc(T * (size_t)hidden_dim * sizeof(float));
  float *q = (float *)malloc(T * (size_t)q_dim * sizeof(float));
  float *k = (float *)malloc(T * (size_t)kv_dim * sizeof(float));
  float *v = (float *)malloc(T * (size_t)kv_dim * sizeof(float));
  float *krep = (float *)malloc(T * (size_t)q_dim * sizeof(float));
  float *vrep = (float *)malloc(T * (size_t)q_dim * sizeof(float));
  float *attn = (float *)malloc(T * (size_t)q_dim * sizeof(float));
  float *proj = (float *)malloc(T * (size_t)hidden_dim * sizeof(float));
  float *gate = (float *)malloc(T * (size_t)ffn * sizeof(float));
  float *up = (float *)malloc(T * (size_t)ffn * sizeof(float));
  if (!norm || !q || !k || !v || !krep || !vrep || !attn || !proj || !gate ||
      !up) {
    free(norm);
    free(q);
    free(k);
    free(v);
    free(krep);
    free(vrep);
    free(attn);
    free(proj);
    free(gate);
    free(up);
    return -1;
  }
  if (h3_dit_rmsnorm(hidden, (int)T, hidden_dim, H3_QWEN_TE_RMS_EPS, input_norm,
                     norm) != 0)
    goto fail;
  if (h3_dit_linear(norm, (int)T, hidden_dim, q_dim, Wq, NULL, q) != 0)
    goto fail;
  if (h3_dit_linear(norm, (int)T, hidden_dim, kv_dim, Wk, NULL, k) != 0)
    goto fail;
  if (h3_dit_linear(norm, (int)T, hidden_dim, kv_dim, Wv, NULL, v) != 0)
    goto fail;
  /* linear out is [T, heads*dim] = [T, heads, dim] packed */
  if (h3_qwen_te_head_rmsnorm(q, T, n_q, head_dim, H3_QWEN_TE_RMS_EPS, q_norm) !=
      0)
    goto fail;
  if (h3_qwen_te_head_rmsnorm(k, T, n_kv, head_dim, H3_QWEN_TE_RMS_EPS, k_norm) !=
      0)
    goto fail;
  if (h3_qwen_te_apply_rope(q, T, n_q, head_dim, rope_cos, rope_sin) != 0)
    goto fail;
  if (h3_qwen_te_apply_rope(k, T, n_kv, head_dim, rope_cos, rope_sin) != 0)
    goto fail;
  if (h3_qwen_te_repeat_kv(k, T, n_kv, n_q, head_dim, krep) != 0)
    goto fail;
  if (h3_qwen_te_repeat_kv(v, T, n_kv, n_q, head_dim, vrep) != 0)
    goto fail;
  float scale = 1.f / sqrtf((float)head_dim);
  if (h3_audio_vae_sdpa_causal_f32(attn, q, krep, vrep, 1, (int)T, n_q,
                                   head_dim, scale) != 0)
    goto fail;
  if (h3_dit_linear(attn, (int)T, q_dim, hidden_dim, Wo, NULL, proj) != 0)
    goto fail;
  for (size_t i = 0; i < T * (size_t)hidden_dim; i++)
    hidden[i] += proj[i];
  if (h3_dit_rmsnorm(hidden, (int)T, hidden_dim, H3_QWEN_TE_RMS_EPS, post_norm,
                     norm) != 0)
    goto fail;
  if (h3_dit_linear(norm, (int)T, hidden_dim, ffn, Wgate, NULL, gate) != 0)
    goto fail;
  if (h3_dit_linear(norm, (int)T, hidden_dim, ffn, Wup, NULL, up) != 0)
    goto fail;
  h3_dit_silu_mul(gate, up, gate, T * (size_t)ffn);
  if (h3_dit_linear(gate, (int)T, ffn, hidden_dim, Wdown, NULL, proj) != 0)
    goto fail;
  for (size_t i = 0; i < T * (size_t)hidden_dim; i++)
    hidden[i] += proj[i];
  free(norm);
  free(q);
  free(k);
  free(v);
  free(krep);
  free(vrep);
  free(attn);
  free(proj);
  free(gate);
  free(up);
  return 0;
fail:
  free(norm);
  free(q);
  free(k);
  free(v);
  free(krep);
  free(vrep);
  free(attn);
  free(proj);
  free(gate);
  free(up);
  return -1;
}

void h3_qwen_te_hash_embed(const uint32_t *ids, size_t n, int dim,
                           float *hidden) {
  if (!ids || !hidden || n < 1 || dim < 1)
    return;
  for (size_t t = 0; t < n; t++) {
    uint32_t id = ids[t];
    for (int d = 0; d < dim; d++) {
      float a = (float)((id + 1u) * (uint32_t)(d + 1)) * 1e-4f;
      hidden[t * (size_t)dim + (size_t)d] = sinf(a);
    }
  }
}
