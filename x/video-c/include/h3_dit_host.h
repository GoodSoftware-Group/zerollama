/* MiniMax-H3 DiT host math (timestep embed, RoPE, condition_proj dims) — no weights. */
#ifndef H3_DIT_HOST_H
#define H3_DIT_HOST_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  H3_DIT_HIDDEN_SIZE = 5376,
  H3_DIT_TEXT_DIM = 5120,
  H3_DIT_NUM_LAYERS = 50,
  /* Full model: the pruned int8 export has 50 blocks and needs all of them.
   * Truncating (e.g. the old 24-layer default) cuts the residual stack so the
   * final AdaLN/RMSNorm sees a wrong hidden state -> audio velocity ~70x too
   * large and a ~93%-clipped waveform. Verified against ComfyUI _forward. */
  H3_DIT_DEFAULT_GENERATE_LAYERS = 50,
  H3_DIT_NUM_HEADS = 56,
  H3_DIT_HEAD_DIM = 128,
  H3_DIT_FFN_HIDDEN = 14336,
  H3_DIT_TIMESTEP_INPUT_DIM = 256,
  H3_DIT_TIME_EMBED_HIDDEN = 5376,
  H3_DIT_TIME_EMBED_DIM = 2688,
  H3_DIT_ROPE_INV_FREQ_LEN = 16,
  H3_DIT_TOKEN_REFINER_LAYERS = 2,
  H3_DIT_LATENTS_DIM = 24,
  H3_DIT_PATCH_T = 1,
  H3_DIT_PATCH_H = 2,
  H3_DIT_PATCH_W = 2
};

#define H3_DIT_ROPE_THETA 10000.0
#define H3_DIT_INNER_DIM (H3_DIT_NUM_HEADS * H3_DIT_HEAD_DIM) /* 7168 */
#define H3_DIT_ROPE_DIM (2 * 3 * H3_DIT_ROPE_INV_FREQ_LEN)     /* 96 */
#define H3_DIT_VIDEO_PATCH_DIM                                                     \
  (H3_DIT_LATENTS_DIM * H3_DIT_PATCH_T * H3_DIT_PATCH_H * H3_DIT_PATCH_W) /* 96 */

/*
 * Sinusoidal timestep embedding (diffusers Timesteps / MLX timestep_embedding).
 * timesteps[n], out[n * dim]. Default flip_sin_to_cos=1 (cos || sin halves).
 */
int h3_dit_timestep_embedding(const float *timesteps, int n, int dim,
                              float max_period, int flip_sin_to_cos,
                              float *out);

/* Release defaults: dim=256, max_period=10000, flip_sin_to_cos. */
int h3_dit_timestep_embedding_default(const float *timesteps, int n,
                                      float *out);

/* inv_freq[n] for n = rope_inv_freq_len (theta ** (-2i/(2n))). */
int h3_dit_rope_inv_freq(float *inv_freq, int n, float theta);

/*
 * position_ids [seq, 3] → cos/sin [seq, 2*3*n] (rotate-half width).
 * Matches MLX RotaryPosEmbed3D.
 */
int h3_dit_rope_from_positions(const float *position_ids, int seq,
                               const float *inv_freq, int n_freq, float *cos,
                               float *sin);

/*
 * Apply rotary to leading rotary_dim channels (Comfy rms_rope_split_half /
 * rotate-half: pair i with i+half). x/out [seq, head_dim], cos/sin [seq, rotary_dim].
 */
int h3_dit_apply_rotary(const float *x, int seq, int head_dim, const float *cos,
                        const float *sin, int rotary_dim, float *out);

/* condition_proj: Linear(text_dim → hidden) y = x @ W^T + b; W [hidden, text]. */
int h3_dit_condition_proj(const float *text, int seq, int text_dim, int hidden,
                          const float *weight, const float *bias, float *out);

/*
 * Pack (B,C,F,H,W) → (B*num_patches, C*pt*ph*pw); frame-major then row-major.
 * Latents and rows are contiguous row-major with strides
 *   latent: ((((b*C+c)*F+f)*H+h)*W+w)
 */
int h3_dit_patchify_video(const float *latents, int B, int C, int F, int H,
                          int W, int pt, int ph, int pw, float *rows);
int h3_dit_unpatchify_video(const float *rows, int B, int C, int F, int H,
                            int W, int pt, int ph, int pw, float *latents);

/* Audio rows: channel-major (2*T, C) ↔ VAE (2, C, T). */
int h3_dit_pack_audio(const float *latents_2ct, int C, int T, float *rows);
int h3_dit_unpack_audio(const float *rows, int C, int T, float *latents_2ct);

/* Activations used by timestep MLP / SwiGLU FFN / AdaLN. */
void h3_dit_silu(const float *x, float *y, size_t n);
void h3_dit_silu_mul(const float *gate, const float *up, float *out, size_t n);
/* RMSNorm over last axis of width `dim` for `rows` tokens; weight may be NULL. */
int h3_dit_rmsnorm(const float *x, int rows, int dim, float eps,
                   const float *weight, float *out);

/* AdaLN modulate: out = x * (1 + scale) + shift (broadcast last axis). */
int h3_dit_modulate(const float *x, const float *shift, const float *scale,
                    int rows, int dim, float *out);
/* Fused residual: out = x + gate * y (same shape). */
int h3_dit_gated_residual(const float *x, const float *y, const float *gate,
                          int rows, int dim, float *out);

/* ImageNet pixel stats for video VAE encode/decode (MLX packing.PIXEL_*). */
extern const float h3_dit_pixel_mean[3];
extern const float h3_dit_pixel_std[3];
void h3_dit_pixel_normalize(const float *rgb01, float *out, size_t n_pixels);
void h3_dit_pixel_denormalize(const float *norm, float *rgb01, size_t n_pixels);

/* Same GEMM as condition_proj: y = x @ W^T + b. */
int h3_dit_linear(const float *x, int rows, int in_dim, int out_dim,
                  const float *weight, const float *bias, float *out);

/* Timestep MLP: Linear(in→hidden) → SiLU → Linear(hidden→out). */
int h3_dit_timestep_mlp(const float *emb, int n, int in_dim, int hidden,
                        int out_dim, const float *w_in, const float *b_in,
                        const float *w_out, const float *b_out, float *out);

/* SwiGLU FFN: fc1 fused [gate; value] along out, SiLU(gate)*value, fc2. */
int h3_dit_swiglu_ffn(const float *x, int rows, int dim, int ffn_hidden,
                      const float *fc1_w, const float *fc1_b, const float *fc2_w,
                      const float *fc2_b, float *out);

/* Interleaved QKV: last dim (heads, 3, head_dim) → Q/K/V [rows, heads, head_dim]. */
int h3_dit_qkv_split_interleaved(const float *qkv, int rows, int heads,
                                 int head_dim, float *query, float *key,
                                 float *value);

/* Concat QKV (Comfy Attention.split): last dim [Q | K | V] each heads*head_dim. */
int h3_dit_qkv_split_concat(const float *qkv, int rows, int heads, int head_dim,
                            float *query, float *key, float *value);

/* Default concat (Comfy int8 pack). H3_QKV_INTERLEAVED=1 restores MLX layout. */
int h3_dit_qkv_split(const float *qkv, int rows, int heads, int head_dim,
                     float *query, float *key, float *value);

#define H3_DIT_NORM_EPS 1e-5f

/* shift/scale tables [table_rows, dim]; indices[rows] into the table. */
int h3_dit_modulate_indexed(const float *x, const float *shift,
                            const float *scale, const int *indices, int rows,
                            int dim, float *out);
int h3_dit_gated_residual_indexed(const float *x, const float *y,
                                  const float *gate, const int *indices,
                                  int rows, int dim, float *out);

/* x [seq, heads, head_dim]; cos/sin [seq, rotary_dim]. */
int h3_dit_apply_rotary_heads(const float *x, int seq, int heads, int head_dim,
                              const float *cos, const float *sin,
                              int rotary_dim, float *out);

/* Full (non-causal) SDPA. q/k/v/out [seq, heads, head_dim]. */
int h3_dit_sdpa_f32(float *out, const float *q, const float *k, const float *v,
                    int seq, int heads, int head_dim, float scale);

#ifdef __cplusplus
}
#endif

#endif /* H3_DIT_HOST_H */
