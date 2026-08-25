/* Rematch audio VAE host kernels vs antirez test_audio_gpu host references. */
#include "h3_audio_vae_host.h"

#include <math.h>
#include <stdio.h>

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

static int check_vec(const float *got, const float *want, int n, float eps,
                     const char *label) {
  float maxe = 0.f;
  for (int i = 0; i < n; i++) {
    float e = fabsf(got[i] - want[i]);
    if (e > maxe)
      maxe = e;
  }
  if (maxe > eps) {
    fprintf(stderr, "FAIL %s max abs %.7g (tol %.7g)\n", label, maxe, eps);
    return 1;
  }
  return 0;
}

static int test_geometry(void) {
  CHECK(h3_audio_vae_hop_from_rates() == H3_AUDIO_VAE_HOP_LENGTH);
  CHECK(h3_audio_vae_pcm_samples(37) == 37 * 800);
  CHECK(h3_audio_vae_pad_samples(801) == 1600);
  float filt[H3_AUDIO_VAE_FILTER_SIZE];
  CHECK(h3_audio_vae_activation_filter(filt, H3_AUDIO_VAE_FILTER_SIZE) == 0);
  return 0;
}

static int test_ops(void) {
  /* Same fixture as antirez/h3.c tests/test_audio_gpu.c */
  const float input_values[] = {0.2f, -0.3f, 0.5f, 0.1f,
                                -0.4f, 0.7f, 0.8f, -0.2f};
  const float vector_values[] = {
      0.2f, -0.1f, 0.3f, 0.4f, -0.5f, 0.6f, -0.2f, 0.7f, 0.1f, -0.4f, 0.3f, 0.5f,
  };
  const float magnitudes[] = {1.5f, 0.75f};
  const float bias_values[] = {0.1f, -0.2f};
  const float want_norm[] = {
      0.3144854510f, -0.1572427255f, 0.4717281765f, 0.6289709020f,
      -0.7862136275f, 0.9434563530f, -0.1470871014f, 0.5148048547f,
      0.0735435507f, -0.2941742027f, 0.2206306520f, 0.3677177534f};
  const float want_conv[] = {0.6346252667f, -0.0896846740f, 0.2886912706f,
                             0.3662853402f, 0.0213786372f,  -0.3691501666f,
                             0.4459339961f, 0.0206306520f};
  const float transpose_weight[] = {0.2f, 0.3f, -0.1f, 0.4f,
                                    -0.5f, 0.1f, 0.25f, 0.2f};
  const float transpose_bias[] = {0.05f};
  const float want_trans[] = {0.08f, 0.005f, 0.23f, -0.405f, 0.22f, 0.265f};
  float filter[12] = {0};
  filter[5] = filter[6] = 0.5f;
  const float alpha_log[] = {0.0f, 0.1f};
  const float beta_log[] = {0.0f, -0.2f};
  const float want_act[] = {0.2394695030f, -0.1705839540f, 0.7298488468f,
                            0.1148576085f, -0.2483533548f, 1.2963163912f,
                            1.3145997606f, -0.1412925005f};
  const float snake_alpha[] = {0.5f, 1.2f};
  const float want_snake[] = {0.2199334221f, -0.1965857206f, 0.6224174379f,
                              0.1119425105f, -0.3210609942f, 1.1620778130f,
                              1.1032932900f, -0.1529145512f};

  float normalized[12];
  CHECK(h3_audio_vae_weight_norm_f32(normalized, vector_values, magnitudes, 2,
                                     6) == 0);
  if (check_vec(normalized, want_norm, 12, 2e-6f, "weight_norm"))
    return 1;

  float conv[8];
  CHECK(h3_audio_vae_conv1d_f32(conv, input_values, normalized, bias_values, 4,
                                2, 2, 3, 1, 1) == 0);
  if (check_vec(conv, want_conv, 8, 2e-5f, "conv1d"))
    return 1;

  float trans[6];
  CHECK(h3_audio_vae_conv_transpose1d_f32(trans, input_values, transpose_weight,
                                          transpose_bias, 3, 2, 1, 4, 2, 1) ==
        0);
  if (check_vec(trans, want_trans, 6, 2e-5f, "conv_transpose1d"))
    return 1;

  float act[8];
  CHECK(h3_audio_vae_alias_free_snake_f32(act, input_values, alpha_log,
                                          beta_log, filter, filter, 4, 2) == 0);
  if (check_vec(act, want_act, 8, 2e-5f, "alias_free_snake"))
    return 1;

  float snake[8];
  CHECK(h3_audio_vae_snake1d_f32(snake, input_values, snake_alpha, 1, 4, 2) ==
        0);
  if (check_vec(snake, want_snake, 8, 2e-5f, "snake1d"))
    return 1;

  (void)close_f;
  return 0;
}

static int test_encoder_kernels(void) {
  float x[] = {1.f, 2.f, 3.f, 4.f};
  float w[] = {1.f, 1.f, 1.f, 1.f};
  float b[] = {0.f, 0.f, 0.f, 0.f};
  float ln[4];
  CHECK(h3_audio_vae_layer_norm_f32(ln, x, w, b, 1, 4, 1e-5f) == 0);
  float mean = 2.5f;
  float var = ((1 - mean) * (1 - mean) + (2 - mean) * (2 - mean) +
               (3 - mean) * (3 - mean) + (4 - mean) * (4 - mean)) /
              4.f;
  float inv = 1.f / sqrtf(var + 1e-5f);
  CHECK(close_f(ln[0], (1.f - mean) * inv, 1e-5f));
  CHECK(close_f(ln[3], (4.f - mean) * inv, 1e-5f));

  float W[] = {1.f, 0.f, 0.f, 1.f}; /* 2x2 identity */
  float bias[] = {0.1f, 0.2f};
  float y[2];
  CHECK(h3_audio_vae_linear_f32(y, x, W, bias, 1, 2, 2) == 0);
  CHECK(close_f(y[0], 1.1f, 1e-5f));
  CHECK(close_f(y[1], 2.2f, 1e-5f));

  float gate[] = {0.5f};
  float lin[] = {2.f};
  float g[1];
  CHECK(h3_audio_vae_geglu_f32(g, gate, lin, 1) == 0);
  float cube = 0.5f * 0.5f * 0.5f;
  float gelu = 0.5f * 0.5f *
               (1.f + tanhf(0.7978845608028654f * (0.5f + 0.044715f * cube)));
  CHECK(close_f(g[0], gelu * 2.f, 1e-5f));

  float qkv[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  float qb[] = {0.1f, 0.2f};
  float kb[] = {0.f, 0.f};
  float vb[] = {-0.1f, 0.f};
  float q[2], k[2], v[2];
  CHECK(h3_audio_vae_qkv_split_f32(q, k, v, qkv, qb, kb, vb, 1, 2) == 0);
  CHECK(close_f(q[0], 1.1f, 1e-6f));
  CHECK(close_f(k[0], 3.f, 1e-6f));
  CHECK(close_f(v[0], 4.9f, 1e-6f));

  /* 1 batch, seq=2, 1 head, dim=1: Q=K=V = [1, 1] */
  float qq[] = {1.f, 1.f};
  float kk[] = {1.f, 1.f};
  float vv[] = {2.f, 4.f};
  float att[2];
  CHECK(h3_audio_vae_sdpa_causal_f32(att, qq, kk, vv, 1, 2, 1, 1, 1.f) == 0);
  CHECK(close_f(att[0], 2.f, 1e-5f)); /* only pos 0 */
  /* row1 softmax over both equal scores → mean of 2 and 4 */
  CHECK(close_f(att[1], 3.f, 1e-5f));

  float attended[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
  float p2[2];
  CHECK(h3_audio_vae_attention_pool_f32(p2, attended, 1, 2, 4, 2) == 0);
  CHECK(close_f(p2[0], (1.f + 2.f + 5.f + 6.f) / 4.f, 1e-5f));
  CHECK(close_f(p2[1], (3.f + 4.f + 7.f + 8.f) / 4.f, 1e-5f));
  CHECK(h3_audio_vae_conv1d_out_length(800, 4, 2, 1, 1) > 0);
  return 0;
}

int main(void) {
  if (test_geometry())
    return 1;
  if (test_ops())
    return 1;
  if (test_encoder_kernels())
    return 1;
  printf("test_h3_audio_vae_host OK (geometry + antirez ops rematch)\n");
  return 0;
}
