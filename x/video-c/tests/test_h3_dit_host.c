/* Rematch DiT host: timestep embed, RoPE inv_freq, condition_proj, apply_rotary. */
#include "h3_dit_host.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int close_f(float a, float b, float eps) {
  return fabsf(a - b) <= eps;
}

static int test_timestep(void) {
  /* Spot-check fixtures/h3_mlx_timestep.json (MLX flip_sin_to_cos). */
  float ts[] = {0.0f, 0.25f, 0.5f, 1.0f};
  float out[4 * H3_DIT_TIMESTEP_INPUT_DIM];
  CHECK(h3_dit_timestep_embedding_default(ts, 4, out) == 0);
  /* t=0 → cos=1, sin=0 across freqs */
  for (int j = 0; j < 128; j++) {
    CHECK(close_f(out[j], 1.0f, 1e-5f));
    CHECK(close_f(out[128 + j], 0.0f, 1e-5f));
  }
  /* t=0.5 first cos freqs from dump */
  float *r2 = out + 2 * 256;
  CHECK(close_f(r2[0], 0.8775825619f, 1e-5f));
  CHECK(close_f(r2[1], 0.8936932685f, 1e-5f));
  CHECK(close_f(r2[2], 0.9077185313f, 1e-5f));
  CHECK(close_f(r2[3], 0.9199195445f, 1e-5f));
  return 0;
}

static int test_rope_inv(void) {
  float inv[H3_DIT_ROPE_INV_FREQ_LEN];
  CHECK(h3_dit_rope_inv_freq(inv, H3_DIT_ROPE_INV_FREQ_LEN, H3_DIT_ROPE_THETA) ==
        0);
  CHECK(close_f(inv[0], 1.0f, 1e-6f));
  CHECK(close_f(inv[1], 0.5623413252f, 1e-6f));
  CHECK(close_f(inv[4], 0.1f, 1e-6f));
  CHECK(close_f(inv[15], 0.00017782794f, 1e-8f));

  float pos[] = {0.1f, 0.2f, 0.3f, 1.0f, 0.0f, 0.5f};
  float cos[2 * H3_DIT_ROPE_DIM], sin[2 * H3_DIT_ROPE_DIM];
  CHECK(h3_dit_rope_from_positions(pos, 2, inv, H3_DIT_ROPE_INV_FREQ_LEN, cos,
                                   sin) == 0);
  /* First angle = 0.1 * 1.0 */
  CHECK(close_f(cos[0], cosf(0.1f), 1e-5f));
  CHECK(close_f(sin[0], sinf(0.1f), 1e-5f));
  /* Duplicate half */
  CHECK(close_f(cos[48], cos[0], 1e-6f));
  return 0;
}

static int test_apply_rotary(void) {
  float inv[H3_DIT_ROPE_INV_FREQ_LEN];
  CHECK(h3_dit_rope_inv_freq(inv, H3_DIT_ROPE_INV_FREQ_LEN, H3_DIT_ROPE_THETA) ==
        0);
  float pos[] = {0.5f, 0.0f, 0.0f};
  float cos[H3_DIT_ROPE_DIM], sin[H3_DIT_ROPE_DIM];
  CHECK(h3_dit_rope_from_positions(pos, 1, inv, H3_DIT_ROPE_INV_FREQ_LEN, cos,
                                   sin) == 0);
  float x[H3_DIT_HEAD_DIM];
  float out[H3_DIT_HEAD_DIM];
  for (int i = 0; i < H3_DIT_HEAD_DIM; i++)
    x[i] = 0.01f * (float)(i + 1);
  CHECK(h3_dit_apply_rotary(x, 1, H3_DIT_HEAD_DIM, cos, sin, H3_DIT_ROPE_DIM,
                            out) == 0);
  /* Pass-through beyond rotary_dim */
  for (int i = H3_DIT_ROPE_DIM; i < H3_DIT_HEAD_DIM; i++)
    CHECK(close_f(out[i], x[i], 1e-6f));
  /* Manual check first pair */
  float x1 = x[0], x2 = x[48];
  CHECK(close_f(out[0], x1 * cos[0] + (-x2) * sin[0], 1e-5f));
  CHECK(close_f(out[48], x2 * cos[48] + x1 * sin[48], 1e-5f));
  return 0;
}

static int test_condition_proj(void) {
  const int seq = 2, td = 4, hid = 3;
  float text[] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f};
  /* W [hid, td] */
  float W[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f};
  float bias[] = {0.1f, 0.2f, 0.3f};
  float out[6];
  CHECK(h3_dit_condition_proj(text, seq, td, hid, W, bias, out) == 0);
  CHECK(close_f(out[0], 1.f + 0.1f, 1e-5f));
  CHECK(close_f(out[1], 5.f + 0.2f, 1e-5f));
  CHECK(close_f(out[2], 9.f + 0.3f, 1e-5f));
  CHECK(close_f(out[3], 2.f + 0.1f, 1e-5f));
  CHECK(close_f(out[4], 6.f + 0.2f, 1e-5f));
  CHECK(close_f(out[5], 10.f + 0.3f, 1e-5f));
  CHECK(H3_DIT_TEXT_DIM == 5120);
  CHECK(H3_DIT_HIDDEN_SIZE == 5376);
  CHECK(H3_DIT_INNER_DIM == 7168);
  CHECK(H3_DIT_VIDEO_PATCH_DIM == 96);
  return 0;
}

static int test_patchify(void) {
  const int B = 1, C = 24, F = 4, H = 8, W = 8;
  const int pt = 1, ph = 2, pw = 2;
  const int n = B * C * F * H * W;
  const int nrows = B * (F / pt) * (H / ph) * (W / pw);
  const int row_dim = C * pt * ph * pw;
  float *lat = malloc((size_t)n * sizeof(float));
  float *rows = malloc((size_t)nrows * (size_t)row_dim * sizeof(float));
  float *back = malloc((size_t)n * sizeof(float));
  CHECK(lat && rows && back);
  for (int i = 0; i < n; i++)
    lat[i] = 0.01f * (float)i;
  CHECK(h3_dit_patchify_video(lat, B, C, F, H, W, pt, ph, pw, rows) == 0);
  CHECK(close_f(rows[0], 0.0f, 1e-6f));
  CHECK(close_f(rows[1], 0.01f, 1e-6f));
  CHECK(close_f(rows[2], 0.08f, 1e-6f));
  CHECK(close_f(rows[3], 0.09f, 1e-6f));
  CHECK(close_f(rows[4], 2.56f, 1e-5f));
  CHECK(close_f(rows[row_dim], 0.02f, 1e-6f)); /* row1 */
  float sum0 = 0.f;
  for (int i = 0; i < row_dim; i++)
    sum0 += rows[i];
  CHECK(close_f(sum0, 2830.5600586f, 1e-2f));
  CHECK(h3_dit_unpatchify_video(rows, B, C, F, H, W, pt, ph, pw, back) == 0);
  for (int i = 0; i < n; i++)
    CHECK(close_f(back[i], lat[i], 1e-5f));
  free(lat);
  free(rows);
  free(back);
  return 0;
}

static int test_silu_rms(void) {
  float x[] = {-2.f, -1.f, 0.f, 1.f, 2.f};
  float y[5];
  h3_dit_silu(x, y, 5);
  CHECK(close_f(y[0], -0.2384058386f, 1e-5f));
  CHECK(close_f(y[3], 0.7310585976f, 1e-5f));
  float gate[] = {0.5f, -0.5f, 1.0f};
  float up[] = {2.f, 3.f, 4.f};
  float sm[3];
  h3_dit_silu_mul(gate, up, sm, 3);
  CHECK(close_f(sm[0], 0.6224593520f, 1e-5f));
  CHECK(close_f(sm[2], 2.9242343903f, 1e-5f));
  float v[] = {1.f, 2.f, 3.f, 4.f};
  float rn[4];
  CHECK(h3_dit_rmsnorm(v, 1, 4, 1e-5f, NULL, rn) == 0);
  CHECK(close_f(rn[0], 0.3651481271f, 1e-5f));
  CHECK(close_f(rn[3], 1.4605925083f, 1e-5f));
  return 0;
}

static int test_audio_pack(void) {
  const int C = 32, T = 5;
  float lat[2 * 32 * 5];
  float rows[2 * 5 * 32];
  float back[2 * 32 * 5];
  for (int i = 0; i < 2 * C * T; i++)
    lat[i] = (float)i;
  CHECK(h3_dit_pack_audio(lat, C, T, rows) == 0);
  CHECK(close_f(rows[0], 0.f, 1e-6f));
  CHECK(close_f(rows[1], (float)T, 1e-6f));
  for (int i = 0; i < 2 * T * C; i++)
    rows[i] = (float)i;
  CHECK(h3_dit_unpack_audio(rows, C, T, back) == 0);
  CHECK(close_f(back[0], 0.f, 1e-6f));
  CHECK(close_f(back[1], 32.f, 1e-6f));
  CHECK(close_f(back[4], 128.f, 1e-6f));
  CHECK(close_f(back[C * T], 160.f, 1e-6f));
  CHECK(h3_dit_pack_audio(back, C, T, rows) == 0);
  for (int i = 0; i < 2 * T * C; i++)
    CHECK(close_f(rows[i], (float)i, 1e-6f));
  return 0;
}

static int test_modulate_pixel(void) {
  float x[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  float shift[] = {0.1f, 0.2f, 0.3f};
  float scale[] = {1.f, 0.5f, 0.f};
  float out[6];
  CHECK(h3_dit_modulate(x, shift, scale, 2, 3, out) == 0);
  CHECK(close_f(out[0], 1.f * 2.f + 0.1f, 1e-5f));
  CHECK(close_f(out[1], 2.f * 1.5f + 0.2f, 1e-5f));
  CHECK(close_f(out[2], 3.f * 1.f + 0.3f, 1e-5f));
  CHECK(close_f(out[3], 4.f * 2.f + 0.1f, 1e-5f));
  float y[] = {0.5f, 0.5f, 0.5f, 1.f, 1.f, 1.f};
  float gate[] = {2.f, 0.f, 1.f};
  CHECK(h3_dit_gated_residual(x, y, gate, 2, 3, out) == 0);
  CHECK(close_f(out[0], 1.f + 2.f * 0.5f, 1e-5f));
  CHECK(close_f(out[1], 2.f + 0.f, 1e-5f));
  CHECK(close_f(out[4], 5.f + 0.f, 1e-5f));

  float rgb[] = {0.5f, 0.5f, 0.5f};
  float norm[3], back[3];
  h3_dit_pixel_normalize(rgb, norm, 1);
  CHECK(close_f(norm[0], (0.5f - 0.485f) / 0.229f, 1e-5f));
  h3_dit_pixel_denormalize(norm, back, 1);
  CHECK(close_f(back[0], 0.5f, 1e-5f));
  CHECK(close_f(back[1], 0.5f, 1e-5f));
  CHECK(close_f(back[2], 0.5f, 1e-5f));
  return 0;
}

int main(void) {
  if (test_timestep())
    return 1;
  if (test_rope_inv())
    return 1;
  if (test_apply_rotary())
    return 1;
  if (test_condition_proj())
    return 1;
  if (test_patchify())
    return 1;
  if (test_silu_rms())
    return 1;
  if (test_audio_pack())
    return 1;
  if (test_modulate_pixel())
    return 1;

  {
    const int seq = 2, td = 4, hid = 3;
    float text[] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f};
    float W[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f};
    float bias[] = {0.1f, 0.2f, 0.3f};
    float out[6], out2[6];
    CHECK(h3_dit_linear(text, seq, td, hid, W, bias, out) == 0);
    CHECK(h3_dit_condition_proj(text, seq, td, hid, W, bias, out2) == 0);
    for (int i = 0; i < 6; i++)
      CHECK(close_f(out[i], out2[i], 1e-6f));

    float emb[] = {1.f, 0.f};
    float win[] = {1.f, 0.f, 0.f, 1.f};
    float bin[] = {0.f, 0.f};
    float wout[] = {1.f, 1.f};
    float bout[] = {0.5f};
    float mlp[1];
    CHECK(h3_dit_timestep_mlp(emb, 1, 2, 2, 1, win, bin, wout, bout, mlp) == 0);
    /* mid = [1, 0], silu(1)+silu(0)+0.5 */
    float s1 = 1.f / (1.f + expf(-1.f));
    float s0 = 0.f / (1.f + expf(0.f));
    CHECK(close_f(mlp[0], s1 + s0 + 0.5f, 1e-5f));

    float x[] = {1.f, -1.f};
    float fc1[] = {
        /* out=4 (2 gate + 2 value), in=2 */
        1.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f,
    };
    float fc2[] = {1.f, 1.f, 0.f, 0.f};
    float sw[2];
    CHECK(h3_dit_swiglu_ffn(x, 1, 2, 2, fc1, NULL, fc2, NULL, sw) == 0);

    float qkv[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float q[2], k[2], v[2];
    CHECK(h3_dit_qkv_split_interleaved(qkv, 1, 1, 2, q, k, v) == 0);
    CHECK(close_f(q[0], 1.f, 1e-6f));
    CHECK(close_f(k[0], 3.f, 1e-6f));
    CHECK(close_f(v[0], 5.f, 1e-6f));
    /* Concat [Q|K|V]: two heads, hd=1 → q=[1,2] k=[3,4] v=[5,6] */
    float q2[2], k2[2], v2[2];
    CHECK(h3_dit_qkv_split_concat(qkv, 1, 2, 1, q2, k2, v2) == 0);
    CHECK(close_f(q2[0], 1.f, 1e-6f) && close_f(q2[1], 2.f, 1e-6f));
    CHECK(close_f(k2[0], 3.f, 1e-6f) && close_f(k2[1], 4.f, 1e-6f));
    CHECK(close_f(v2[0], 5.f, 1e-6f) && close_f(v2[1], 6.f, 1e-6f));
  }

  {
    /* h3_dit_sdpa_f32 scalar paths (serial + parallel-over-heads, seq >= 64)
     * must be bit-identical to the serial reference. H3_SDPA_BLAS=0 pins the
     * legacy path; the BLAS default is checked against it with a tolerance
     * below (different accumulation order, same math). */
    setenv("H3_SDPA_BLAS", "0", 1);
    const int seq = 128, heads = 8, hd = 32;
    float scale = 1.0f / sqrtf((float)hd);
    size_t total = (size_t)seq * (size_t)heads * (size_t)hd;
    float *q = (float *)malloc(total * sizeof(float));
    float *k = (float *)malloc(total * sizeof(float));
    float *v = (float *)malloc(total * sizeof(float));
    float *ref = (float *)malloc(total * sizeof(float));
    float *out = (float *)malloc(total * sizeof(float));
    CHECK(q && k && v && ref && out);
    for (size_t i = 0; i < total; i++) {
      q[i] = (float)((i * 2654435761u) % 1000) / 1000.0f - 0.5f;
      k[i] = (float)((i * 40503u + 17u) % 1000) / 1000.0f - 0.5f;
      v[i] = (float)((i * 7741u + 9u) % 1000) / 1000.0f - 0.5f;
    }
    /* Serial reference (identical loop order). */
    float *scores = (float *)malloc((size_t)seq * sizeof(float));
    CHECK(scores);
    for (int h = 0; h < heads; h++) {
      for (int row = 0; row < seq; row++) {
        const float *qr = q + ((size_t)row * heads + h) * (size_t)hd;
        float m = -1e30f;
        for (int col = 0; col < seq; col++) {
          const float *kr = k + ((size_t)col * heads + h) * (size_t)hd;
          double dot = 0.0;
          for (int d = 0; d < hd; d++)
            dot += (double)qr[d] * kr[d];
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
        float *orow = ref + ((size_t)row * heads + h) * (size_t)hd;
        for (int d = 0; d < hd; d++)
          orow[d] = 0.f;
        for (int col = 0; col < seq; col++) {
          const float *vr = v + ((size_t)col * heads + h) * (size_t)hd;
          float w = scores[col] * inv;
          for (int d = 0; d < hd; d++)
            orow[d] += w * vr[d];
        }
      }
    }
    free(scores);
    CHECK(h3_dit_sdpa_f32(out, q, k, v, seq, heads, hd, scale) == 0);
    for (size_t i = 0; i < total; i++) {
      if (out[i] != ref[i]) {
        fprintf(stderr, "FAIL sdpa parallel path bit-diff at %zu: %a vs %a\n",
                i, out[i], ref[i]);
        return 1;
      }
    }
    /* BLAS default path: same math, different accumulation order — tolerance. */
    setenv("H3_SDPA_BLAS", "1", 1);
    CHECK(h3_dit_sdpa_f32(out, q, k, v, seq, heads, hd, scale) == 0);
    unsetenv("H3_SDPA_BLAS");
    double blas_maxdiff = 0.0;
    for (size_t i = 0; i < total; i++) {
      double d = fabs((double)out[i] - (double)ref[i]);
      if (d > blas_maxdiff)
        blas_maxdiff = d;
    }
    if (blas_maxdiff > 1e-4) {
      fprintf(stderr, "FAIL sdpa blas path diff %.6g > 1e-4\n", blas_maxdiff);
      return 1;
    }
    printf("test_h3_dit_host sdpa blas maxdiff=%.3g\n", blas_maxdiff);
    free(q);
    free(k);
    free(v);
    free(ref);
    free(out);
  }

  printf("test_h3_dit_host OK\n");
  return 0;
}
