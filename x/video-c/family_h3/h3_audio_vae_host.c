#define _DARWIN_C_SOURCE 1
#include "h3_audio_vae_host.h"

#include <dispatch/dispatch.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>

#include <Accelerate/Accelerate.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const int h3_audio_vae_upsample_rates[H3_AUDIO_VAE_STAGES] = {5, 5, 2, 2, 2, 2,
                                                              2};
const int h3_audio_vae_upsample_kernels[H3_AUDIO_VAE_STAGES] = {9, 9, 4, 4, 4,
                                                                 4, 4};
const int h3_audio_vae_residual_kernels[H3_AUDIO_VAE_RESBLOCKS] = {3, 7, 11};
const int h3_audio_vae_residual_dilations[H3_AUDIO_VAE_RESIDUAL_PAIRS] = {1, 3,
                                                                          5};
const int h3_audio_vae_encoder_strides[H3_AUDIO_VAE_ENCODER_STAGES] = {2, 4, 4, 5,
                                                                      5};
const int h3_audio_vae_encoder_dilations[H3_AUDIO_VAE_ENCODER_RESIDUALS] = {1, 3,
                                                                           9};

int h3_audio_vae_hop_from_rates(void) {
  int hop = 1;
  for (int i = 0; i < H3_AUDIO_VAE_STAGES; i++)
    hop *= h3_audio_vae_upsample_rates[i];
  return hop;
}

int h3_audio_vae_pcm_samples(int latent_t) {
  if (latent_t < 1)
    return 0;
  return latent_t * H3_AUDIO_VAE_HOP_LENGTH;
}

int h3_audio_vae_pad_samples(int samples) {
  if (samples < 0)
    return 0;
  const int hop = H3_AUDIO_VAE_HOP_LENGTH;
  return ((samples + hop - 1) / hop) * hop;
}

/* Modified Bessel I0 (Abramowitz & Stegun / Numerical Recipes). */
static double bessel_i0(double x) {
  double ax = fabs(x);
  if (ax < 3.75) {
    double y = x / 3.75;
    y *= y;
    return 1.0 +
           y * (3.5156229 +
                y * (3.0899424 +
                     y * (1.2067492 +
                          y * (0.2659732 +
                               y * (0.0360768 + y * 0.0045813)))));
  }
  double y = 3.75 / ax;
  return (exp(ax) / sqrt(ax)) *
         (0.39894228 +
          y * (0.01328592 +
               y * (0.00225319 +
                    y * (-0.00157565 +
                         y * (0.00916281 +
                              y * (-0.02057706 +
                                   y * (0.02635537 +
                                        y * (-0.01647633 + y * 0.00392377))))))));
}

/* numpy.kaiser(M, beta) with periodic=False. */
static double kaiser_window_at(int M, double beta, int n) {
  if (M <= 1)
    return 1.0;
  double alpha = 0.5 * (double)(M - 1);
  double t = ((double)n - alpha) / alpha;
  double arg = 1.0 - t * t;
  if (arg < 0.0)
    arg = 0.0;
  return bessel_i0(beta * sqrt(arg)) / bessel_i0(beta);
}

/* np.sinc(x) = sin(pi x)/(pi x) */
static double np_sinc(double x) {
  if (x == 0.0)
    return 1.0;
  double y = M_PI * x;
  return sin(y) / y;
}

int h3_audio_vae_kaiser_sinc(double cutoff, double half_width, int kernel_size,
                             float *dst, size_t dst_n) {
  if (!dst || kernel_size < 1 || (size_t)kernel_size > dst_n)
    return -1;
  int half_size = kernel_size / 2;
  double attenuation =
      2.285 * (double)(half_size - 1) * M_PI * (4.0 * half_width) + 7.95;
  double beta;
  if (attenuation > 50.0)
    beta = 0.1102 * (attenuation - 8.7);
  else if (attenuation >= 21.0)
    beta = 0.5842 * pow(attenuation - 21.0, 0.4) +
           0.07886 * (attenuation - 21.0);
  else
    beta = 0.0;

  double sum = 0.0;
  double scratch[64];
  if (kernel_size > (int)(sizeof(scratch) / sizeof(scratch[0])))
    return -1;
  for (int i = 0; i < kernel_size; i++) {
    double time;
    if ((kernel_size % 2) == 0)
      time = (double)(i - half_size) + 0.5;
    else
      time = (double)i - (double)half_size;
    double w = kaiser_window_at(kernel_size, beta, i);
    scratch[i] = 2.0 * cutoff * w * np_sinc(2.0 * cutoff * time);
    sum += scratch[i];
  }
  if (sum == 0.0)
    return -1;
  for (int i = 0; i < kernel_size; i++)
    dst[i] = (float)(scratch[i] / sum);
  return 0;
}

int h3_audio_vae_activation_filter(float *dst, size_t dst_n) {
  /* ratio=2 → cutoff=0.5/2, half_width=0.6/2 */
  return h3_audio_vae_kaiser_sinc(0.25, 0.3, H3_AUDIO_VAE_FILTER_SIZE, dst,
                                  dst_n);
}

int h3_audio_vae_weight_norm_f32(float *dst, const float *vector,
                                 const float *magnitude, int outer, int inner) {
  if (!dst || !vector || !magnitude || outer < 1 || inner < 1)
    return -1;
  for (int row = 0; row < outer; row++) {
    const float *v = vector + (size_t)row * (size_t)inner;
    float *o = dst + (size_t)row * (size_t)inner;
    double square = 0.0;
    for (int i = 0; i < inner; i++)
      square += (double)v[i] * (double)v[i];
    if (square <= 0.0)
      return -1;
    float scale = (float)((double)magnitude[row] / sqrt(square));
    for (int i = 0; i < inner; i++)
      o[i] = v[i] * scale;
  }
  return 0;
}

int h3_audio_vae_conv1d_f32(float *dst, const float *input, const float *weight,
                            const float *bias, int length, int in_ch, int out_ch,
                            int kernel, int padding, int dilation) {
  if (!dst || !input || !weight || !bias || length < 1 || in_ch < 1 ||
      out_ch < 1 || kernel < 1 || dilation < 1)
    return -1;
  for (int time = 0; time < length; time++) {
    for (int out = 0; out < out_ch; out++) {
      float sum = bias[out];
      for (int k = 0; k < kernel; k++) {
        int source = time + k * dilation - padding;
        if (source < 0 || source >= length)
          continue;
        for (int in = 0; in < in_ch; in++) {
          sum += input[source * in_ch + in] *
                 weight[(out * in_ch + in) * kernel + k];
        }
      }
      dst[time * out_ch + out] = sum;
    }
  }
  return 0;
}

int h3_audio_vae_conv1d_out_length(int length, int kernel, int stride,
                                   int padding, int dilation) {
  if (length < 1 || kernel < 1 || stride < 1 || dilation < 1)
    return -1;
  long long effective = (long long)dilation * (kernel - 1) + 1;
  long long padded = (long long)length + 2LL * padding;
  if (padded < effective)
    return -1;
  long long out = (padded - effective) / stride + 1;
  if (out < 1 || out > 100000000)
    return -1;
  return (int)out;
}

/* Conv1d via im2col + Accelerate sgemm. A = X[time, in*kernel+k],
 * B = W'[(in*kernel+k), out] (same values as weight[out,in,k]). */
static int conv1d_gemm(float *dst, const float *input, const float *weight,
                       const float *bias, int length, int in_ch, int out_ch,
                       int kernel, int stride, int padding, int dilation,
                       int out_len) {
  int K = in_ch * kernel;
  float *X = (float *)malloc((size_t)out_len * (size_t)K * sizeof(float));
  if (!X)
    return -1;
  for (int time = 0; time < out_len; time++) {
    float *row = X + (size_t)time * (size_t)K;
    for (int k = 0; k < kernel; k++) {
      int source = time * stride + k * dilation - padding;
      const float *src =
          (source >= 0 && source < length) ? input + (size_t)source * in_ch : NULL;
      for (int in = 0; in < in_ch; in++)
        row[in * kernel + k] = src ? src[in] : 0.0f;
    }
  }
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, out_len, out_ch, K,
              1.0f, X, K, weight, K, 0.0f, dst, out_ch);
  free(X);
  if (bias) {
    for (int time = 0; time < out_len; time++)
      for (int out = 0; out < out_ch; out++)
        dst[(size_t)time * out_ch + out] += bias[out];
  }
  return 0;
}

static void conv1d_rows(float *dst, const float *input, const float *weight,
                        const float *bias, int length, int in_ch, int out_ch,
                        int kernel, int stride, int padding, int dilation,
                        int t0, int t1) {
  for (int time = t0; time < t1; time++) {
    for (int out = 0; out < out_ch; out++) {
      float sum = bias[out];
      for (int k = 0; k < kernel; k++) {
        int source = time * stride + k * dilation - padding;
        if (source < 0 || source >= length)
          continue;
        for (int in = 0; in < in_ch; in++)
          sum += input[source * in_ch + in] *
                 weight[(out * in_ch + in) * kernel + k];
      }
      dst[time * out_ch + out] = sum;
    }
  }
}

int h3_audio_vae_conv1d_stride_f32(float *dst, const float *input,
                                   const float *weight, const float *bias,
                                   int length, int in_ch, int out_ch, int kernel,
                                   int stride, int padding, int dilation) {
  int out_len =
      h3_audio_vae_conv1d_out_length(length, kernel, stride, padding, dilation);
  if (!dst || !input || !weight || !bias || out_len < 1)
    return -1;
  /* Accelerate sgemm path: worth it once the im2col matrix is sizable. */
  if ((long long)out_len * (long long)(in_ch * kernel) >= 1 << 14)
    return conv1d_gemm(dst, input, weight, bias, length, in_ch, out_ch, kernel,
                       stride, padding, dilation, out_len);
  long ncore = sysconf(_SC_NPROCESSORS_ONLN);
  if (ncore < 2 || out_len < 32) {
    conv1d_rows(dst, input, weight, bias, length, in_ch, out_ch, kernel,
                stride, padding, dilation, 0, out_len);
    return 0;
  }
  int per = (out_len + (int)ncore - 1) / (int)ncore;
  dispatch_apply((size_t)ncore,
                 dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                 ^(size_t c) {
                   int r0 = (int)c * per;
                   if (r0 >= out_len)
                     return;
                   int r1 = r0 + per;
                   if (r1 > out_len)
                     r1 = out_len;
                   conv1d_rows(dst, input, weight, bias, length, in_ch, out_ch,
                               kernel, stride, padding, dilation, r0, r1);
                 });
  return 0;
}

int h3_audio_vae_conv_transpose1d_f32(float *dst, const float *input,
                                      const float *weight, const float *bias,
                                      int length, int in_ch, int out_ch,
                                      int kernel, int stride, int padding) {
  if (!dst || !input || !weight || !bias || length < 1 || in_ch < 1 ||
      out_ch < 1 || kernel < 1 || stride < 1)
    return -1;
  int out_len = (length - 1) * stride + kernel - 2 * padding;
  if (out_len < 1)
    return -1;
  long ncore = sysconf(_SC_NPROCESSORS_ONLN);
  if (ncore < 2 || out_len < 32) {
    for (int time = 0; time < out_len; time++) {
      for (int out = 0; out < out_ch; out++) {
        float sum = bias[out];
        for (int k = 0; k < kernel; k++) {
          int numerator = time + padding - k;
          if (numerator < 0 || numerator % stride)
            continue;
          int source = numerator / stride;
          if (source >= length)
            continue;
          for (int in = 0; in < in_ch; in++) {
            sum += input[source * in_ch + in] *
                   weight[(in * out_ch + out) * kernel + k];
          }
        }
        dst[time * out_ch + out] = sum;
      }
    }
    return 0;
  }
  int per = (out_len + (int)ncore - 1) / (int)ncore;
  dispatch_apply((size_t)ncore,
                 dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                 ^(size_t c) {
                   int t0 = (int)c * per;
                   if (t0 >= out_len)
                     return;
                   int t1 = t0 + per;
                   if (t1 > out_len)
                     t1 = out_len;
                   for (int time = t0; time < t1; time++) {
                     for (int out = 0; out < out_ch; out++) {
                       float sum = bias[out];
                       for (int k = 0; k < kernel; k++) {
                         int numerator = time + padding - k;
                         if (numerator < 0 || numerator % stride)
                           continue;
                         int source = numerator / stride;
                         if (source >= length)
                           continue;
                         for (int in = 0; in < in_ch; in++) {
                           sum += input[source * in_ch + in] *
                                  weight[(in * out_ch + out) * kernel + k];
                         }
                       }
                       dst[time * out_ch + out] = sum;
                     }
                   }
                 });
  return 0;
}

int h3_audio_vae_snake1d_f32(float *dst, const float *input, const float *alpha,
                             int batch, int length, int channels) {
  if (!dst || !input || !alpha || batch < 1 || length < 1 || channels < 1)
    return -1;
  size_t n = (size_t)batch * (size_t)length * (size_t)channels;
  for (size_t i = 0; i < n; i++) {
    float a = alpha[i % (size_t)channels];
    float x = input[i];
    float wave = sinf(a * x);
    dst[i] = x + wave * wave / (a + 1e-9f);
  }
  return 0;
}

static float upsample_at(const float *input, const float *filter2, int length,
                         int channels, int channel, int up_time) {
  int raw = up_time + 15;
  float result = 0.0f;
  for (int k = 0; k < H3_AUDIO_VAE_FILTER_SIZE; k++) {
    int numerator = raw - k;
    if (numerator < 0 || (numerator % 2))
      continue;
    int source = numerator / 2 - 5;
    if (source < 0)
      source = 0;
    if (source >= length)
      source = length - 1;
    result += input[source * channels + channel] * filter2[k];
  }
  return result;
}

int h3_audio_vae_alias_free_snake_f32(float *dst, const float *input,
                                      const float *alpha_log,
                                      const float *beta_log,
                                      const float *up_filter,
                                      const float *down_filter, int length,
                                      int channels) {
  if (!dst || !input || !alpha_log || !beta_log || !up_filter || !down_filter ||
      length < 1 || channels < 1)
    return -1;
  /* Precompute per-channel exp(alpha)/exp(beta) and 2*up_filter once; the
   * original code recomputed expf() per (time, channel) and 2*filter per tap. */
  float *exp_alpha = (float *)malloc((size_t)channels * sizeof(float));
  float *exp_beta = (float *)malloc((size_t)channels * sizeof(float));
  float *up2 = (float *)malloc(H3_AUDIO_VAE_FILTER_SIZE * sizeof(float));
  if (!exp_alpha || !exp_beta || !up2) {
    free(exp_alpha);
    free(exp_beta);
    free(up2);
    return -1;
  }
  for (int channel = 0; channel < channels; channel++) {
    exp_alpha[channel] = expf(alpha_log[channel]);
    exp_beta[channel] = expf(beta_log[channel]);
  }
  for (int k = 0; k < H3_AUDIO_VAE_FILTER_SIZE; k++)
    up2[k] = 2.0f * up_filter[k];

  long ncore = sysconf(_SC_NPROCESSORS_ONLN);
  if (ncore < 2 || length < 32) {
    for (int time = 0; time < length; time++) {
      for (int channel = 0; channel < channels; channel++) {
        float alpha = exp_alpha[channel];
        float beta = exp_beta[channel];
        float result = 0.0f;
        for (int k = 0; k < H3_AUDIO_VAE_FILTER_SIZE; k++) {
          int up_time = time * 2 + k - 5;
          if (up_time < 0)
            up_time = 0;
          if (up_time >= length * 2)
            up_time = length * 2 - 1;
          float value =
              upsample_at(input, up2, length, channels, channel, up_time);
          float sine = sinf(alpha * value);
          value += sine * sine / (beta + 1e-9f);
          result += value * down_filter[k];
        }
        dst[time * channels + channel] = result;
      }
    }
  } else {
    int per = (length + (int)ncore - 1) / (int)ncore;
    dispatch_apply((size_t)ncore,
                   dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                   ^(size_t c) {
                     int t0 = (int)c * per;
                     if (t0 >= length)
                       return;
                     int t1 = t0 + per;
                     if (t1 > length)
                       t1 = length;
                     for (int time = t0; time < t1; time++) {
                       for (int channel = 0; channel < channels; channel++) {
                         float alpha = exp_alpha[channel];
                         float beta = exp_beta[channel];
                         float result = 0.0f;
                         for (int k = 0; k < H3_AUDIO_VAE_FILTER_SIZE; k++) {
                           int up_time = time * 2 + k - 5;
                           if (up_time < 0)
                             up_time = 0;
                           if (up_time >= length * 2)
                             up_time = length * 2 - 1;
                           float value = upsample_at(input, up2, length, channels,
                                                     channel, up_time);
                           float sine = sinf(alpha * value);
                           value += sine * sine / (beta + 1e-9f);
                           result += value * down_filter[k];
                         }
                         dst[time * channels + channel] = result;
                       }
                     }
                   });
  }
  free(exp_alpha);
  free(exp_beta);
  free(up2);
  return 0;
}

int h3_audio_vae_layer_norm_f32(float *dst, const float *x, const float *weight,
                                const float *bias, int rows, int dim,
                                float eps) {
  if (!dst || !x || !weight || !bias || rows < 1 || dim < 1)
    return -1;
  for (int r = 0; r < rows; r++) {
    const float *xr = x + (size_t)r * (size_t)dim;
    float *orow = dst + (size_t)r * (size_t)dim;
    double mean = 0.0;
    for (int i = 0; i < dim; i++)
      mean += xr[i];
    mean /= (double)dim;
    double var = 0.0;
    for (int i = 0; i < dim; i++) {
      double d = xr[i] - mean;
      var += d * d;
    }
    var /= (double)dim;
    float inv = 1.0f / sqrtf((float)var + eps);
    for (int i = 0; i < dim; i++)
      orow[i] = ((xr[i] - (float)mean) * inv) * weight[i] + bias[i];
  }
  return 0;
}

int h3_audio_vae_linear_f32(float *dst, const float *x, const float *weight,
                            const float *bias, int rows, int in_dim,
                            int out_dim) {
  if (!dst || !x || !weight || rows < 1 || in_dim < 1 || out_dim < 1)
    return -1;
  for (int r = 0; r < rows; r++) {
    const float *xr = x + (size_t)r * (size_t)in_dim;
    float *orow = dst + (size_t)r * (size_t)out_dim;
    for (int o = 0; o < out_dim; o++) {
      const float *w = weight + (size_t)o * (size_t)in_dim;
      double sum = bias ? bias[o] : 0.0;
      for (int i = 0; i < in_dim; i++)
        sum += (double)xr[i] * (double)w[i];
      orow[o] = (float)sum;
    }
  }
  return 0;
}

int h3_audio_vae_geglu_f32(float *dst, const float *gate, const float *linear,
                           size_t n) {
  if (!dst || !gate || !linear)
    return -1;
  const float k = 0.7978845608028654f;
  for (size_t i = 0; i < n; i++) {
    float x = gate[i];
    float cube = x * x * x;
    float gelu = 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * cube)));
    dst[i] = gelu * linear[i];
  }
  return 0;
}

int h3_audio_vae_qkv_split_f32(float *query, float *key, float *value,
                               const float *qkv, const float *q_bias,
                               const float *k_bias, const float *v_bias,
                               int rows, int width) {
  if (!query || !key || !value || !qkv || !q_bias || !k_bias || !v_bias ||
      rows < 1 || width < 1)
    return -1;
  for (int r = 0; r < rows; r++) {
    const float *src = qkv + (size_t)r * (size_t)width * 3;
    float *q = query + (size_t)r * (size_t)width;
    float *k = key + (size_t)r * (size_t)width;
    float *v = value + (size_t)r * (size_t)width;
    for (int c = 0; c < width; c++) {
      q[c] = src[c] + q_bias[c];
      k[c] = src[width + c] + k_bias[c];
      v[c] = src[2 * width + c] + v_bias[c];
    }
  }
  return 0;
}

int h3_audio_vae_sdpa_causal_f32(float *out, const float *q, const float *k,
                                 const float *v, int batch, int seq, int heads,
                                 int head_dim, float scale) {
  if (!out || !q || !k || !v || batch < 1 || seq < 1 || heads < 1 ||
      head_dim < 1)
    return -1;
  float *scores = (float *)malloc((size_t)seq * sizeof(float));
  if (!scores)
    return -1;
  for (int b = 0; b < batch; b++) {
    for (int h = 0; h < heads; h++) {
      for (int row = 0; row < seq; row++) {
        const float *qr =
            q + ((((size_t)b * seq + row) * heads + h) * (size_t)head_dim);
        float m = -1e30f;
        for (int col = 0; col <= row; col++) {
          const float *kr =
              k + ((((size_t)b * seq + col) * heads + h) * (size_t)head_dim);
          double dot = 0.0;
          for (int d = 0; d < head_dim; d++)
            dot += (double)qr[d] * (double)kr[d];
          float s = (float)dot * scale;
          scores[col] = s;
          if (s > m)
            m = s;
        }
        double l = 0.0;
        for (int col = 0; col <= row; col++) {
          scores[col] = expf(scores[col] - m);
          l += scores[col];
        }
        float inv = (float)(1.0 / l);
        float *orow =
            out + ((((size_t)b * seq + row) * heads + h) * (size_t)head_dim);
        for (int d = 0; d < head_dim; d++)
          orow[d] = 0.f;
        for (int col = 0; col <= row; col++) {
          const float *vr =
              v + ((((size_t)b * seq + col) * heads + h) * (size_t)head_dim);
          float w = scores[col] * inv;
          for (int d = 0; d < head_dim; d++)
            orow[d] += w * vr[d];
        }
      }
    }
  }
  free(scores);
  return 0;
}

int h3_audio_vae_attention_pool_f32(float *out, const float *attended, int rows,
                                    int heads, int head_dim, int out_dim) {
  if (!out || !attended || rows < 1 || heads < 1 || head_dim < 1 || out_dim < 1)
    return -1;
  if (head_dim % out_dim)
    return -1;
  int pool = head_dim / out_dim;
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < out_dim; c++) {
      float sum = 0.f;
      for (int h = 0; h < heads; h++) {
        const float *base =
            attended + ((size_t)r * heads + h) * (size_t)head_dim +
            (size_t)c * (size_t)pool;
        for (int i = 0; i < pool; i++)
          sum += base[i];
      }
      out[(size_t)r * (size_t)out_dim + (size_t)c] =
          sum / (float)(heads * pool);
    }
  }
  return 0;
}
