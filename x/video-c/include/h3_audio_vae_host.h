/* Portable MiniMax-H3 audio VAE host geometry + Kaiser-sinc (no GPU/weights). */
#ifndef H3_AUDIO_VAE_HOST_H
#define H3_AUDIO_VAE_HOST_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  H3_AUDIO_VAE_LATENT_CHANNELS = 32,
  H3_AUDIO_VAE_LATENT_DIM = 2048,
  H3_AUDIO_VAE_DECODER_DIM = 1024,
  H3_AUDIO_VAE_STEREO = 2,
  H3_AUDIO_VAE_STAGES = 7,
  H3_AUDIO_VAE_RESBLOCKS = 3,
  H3_AUDIO_VAE_RESIDUAL_PAIRS = 3,
  H3_AUDIO_VAE_FILTER_SIZE = 12,
  H3_AUDIO_VAE_SAMPLE_RATE = 32000,
  H3_AUDIO_VAE_HOP_LENGTH = 800,
  H3_AUDIO_VAE_ENCODER_STAGES = 5,
  H3_AUDIO_VAE_ENCODER_RESIDUALS = 3,
  H3_AUDIO_VAE_ENCODER_HEADS = 8,
  H3_AUDIO_VAE_ENCODER_DIM = 64
};

/* Release BigVGAN upsample schedule (product = hop 800). */
extern const int h3_audio_vae_upsample_rates[H3_AUDIO_VAE_STAGES];
extern const int h3_audio_vae_upsample_kernels[H3_AUDIO_VAE_STAGES];
extern const int h3_audio_vae_residual_kernels[H3_AUDIO_VAE_RESBLOCKS];
extern const int h3_audio_vae_residual_dilations[H3_AUDIO_VAE_RESIDUAL_PAIRS];
extern const int h3_audio_vae_encoder_strides[H3_AUDIO_VAE_ENCODER_STAGES];
extern const int h3_audio_vae_encoder_dilations[H3_AUDIO_VAE_ENCODER_RESIDUALS];

/* Product of upsample rates; should equal H3_AUDIO_VAE_HOP_LENGTH. */
int h3_audio_vae_hop_from_rates(void);

/* Decode length: latent_t * hop. Encode pad: ceil(samples/hop)*hop. */
int h3_audio_vae_pcm_samples(int latent_t);
int h3_audio_vae_pad_samples(int samples);

/*
 * Kaiser-windowed sinc low-pass, shape [kernel_size], sum-normalized.
 * BigVGAN Activation1d defaults for ratio=2, kernel=12:
 *   cutoff = 0.5/ratio, half_width = 0.6/ratio.
 * Returns 0 on success.
 */
int h3_audio_vae_kaiser_sinc(double cutoff, double half_width, int kernel_size,
                             float *dst, size_t dst_n);

/* Convenience: Activation1d default filter (ratio=2, kernel=12). */
int h3_audio_vae_activation_filter(float *dst, size_t dst_n);

/*
 * Host reference kernels (channels-last layout matching antirez Metal tests).
 * All return 0 on success. magnitude is [outer] scalars (folded weight_g).
 */
int h3_audio_vae_weight_norm_f32(float *dst, const float *vector,
                                 const float *magnitude, int outer, int inner);
int h3_audio_vae_conv1d_f32(float *dst, const float *input, const float *weight,
                            const float *bias, int length, int in_ch, int out_ch,
                            int kernel, int padding, int dilation);
int h3_audio_vae_conv1d_out_length(int length, int kernel, int stride,
                                   int padding, int dilation);
int h3_audio_vae_conv1d_stride_f32(float *dst, const float *input,
                                   const float *weight, const float *bias,
                                   int length, int in_ch, int out_ch, int kernel,
                                   int stride, int padding, int dilation);
int h3_audio_vae_conv_transpose1d_f32(float *dst, const float *input,
                                      const float *weight, const float *bias,
                                      int length, int in_ch, int out_ch,
                                      int kernel, int stride, int padding);
/* Snake1d: x + sin(a x)^2 / (a + 1e-9); alpha[channels]. */
int h3_audio_vae_snake1d_f32(float *dst, const float *input, const float *alpha,
                             int batch, int length, int channels);
/* Fused 2× upsample → SnakeBeta → 2× downsample (BigVGAN Activation1d). */
int h3_audio_vae_alias_free_snake_f32(float *dst, const float *input,
                                      const float *alpha_log,
                                      const float *beta_log,
                                      const float *up_filter,
                                      const float *down_filter, int length,
                                      int channels);

int h3_audio_vae_layer_norm_f32(float *dst, const float *x, const float *weight,
                                const float *bias, int rows, int dim, float eps);
/* y = x @ W^T + b; W [out, in], x/y [rows, in/out]. bias may be NULL. */
int h3_audio_vae_linear_f32(float *dst, const float *x, const float *weight,
                            const float *bias, int rows, int in_dim,
                            int out_dim);
int h3_audio_vae_geglu_f32(float *dst, const float *gate, const float *linear,
                           size_t n);
/* qkv [rows, 3*width] concat Q|K|V → query/key/value [rows, width]. */
int h3_audio_vae_qkv_split_f32(float *query, float *key, float *value,
                               const float *qkv, const float *q_bias,
                               const float *k_bias, const float *v_bias,
                               int rows, int width);
/* Q/K/V/out layout [batch, seq, heads, dim], causal. */
int h3_audio_vae_sdpa_causal_f32(float *out, const float *q, const float *k,
                                 const float *v, int batch, int seq, int heads,
                                 int head_dim, float scale);
int h3_audio_vae_attention_pool_f32(float *out, const float *attended, int rows,
                                    int heads, int head_dim, int out_dim);

#ifdef __cplusplus
}
#endif

#endif /* H3_AUDIO_VAE_HOST_H */
