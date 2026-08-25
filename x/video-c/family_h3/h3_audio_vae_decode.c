#include "h3_audio_vae_decode.h"
#include "h3_audio_vae_host.h"
#include "h3_prof.h"
#include "h3_st_store.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
  STAGES = H3_AUDIO_VAE_STAGES,
  RESBLOCKS = H3_AUDIO_VAE_RESBLOCKS,
  RESIDUAL_PAIRS = H3_AUDIO_VAE_RESIDUAL_PAIRS,
  LATENT_CHANNELS = H3_AUDIO_VAE_LATENT_CHANNELS,
  LATENT_DIM = H3_AUDIO_VAE_LATENT_DIM,
  DECODER_DIM = H3_AUDIO_VAE_DECODER_DIM,
  STEREO = H3_AUDIO_VAE_STEREO,
  FILTER_SIZE = H3_AUDIO_VAE_FILTER_SIZE,
  HOP_LENGTH = H3_AUDIO_VAE_HOP_LENGTH,
  SAMPLE_RATE = H3_AUDIO_VAE_SAMPLE_RATE
};

typedef struct {
  int in_ch, out_ch, kernel, padding, dilation, stride, transpose, has_bias;
  float *weight; /* folded */
  float *bias;   /* may be NULL → treat as zeros via zero_bias */
  float *zero_bias;
} host_conv;

typedef struct {
  float *alpha;
  float *beta;
} host_act;

typedef struct {
  host_act activations[6]; /* 3 pairs × 2 */
  host_conv convs1[3];
  host_conv convs2[3];
} host_resblock;

typedef struct {
  host_conv upsample;
  host_resblock blocks[3];
} host_stage;

static void fail1(char *error, size_t n, const char *msg) {
  if (error && n)
    snprintf(error, n, "%s", msg);
}

static void fail2(char *error, size_t n, const char *a, const char *b) {
  if (error && n)
    snprintf(error, n, "%s: %s", a, b);
}

static float *xmalloc_f(size_t n) {
  if (!n)
    return NULL;
  float *p = (float *)malloc(n * sizeof(float));
  return p;
}

static void free_conv(host_conv *c) {
  if (!c)
    return;
  free(c->weight);
  free(c->bias);
  free(c->zero_bias);
  memset(c, 0, sizeof(*c));
}

static void free_act(host_act *a) {
  if (!a)
    return;
  free(a->alpha);
  free(a->beta);
  memset(a, 0, sizeof(*a));
}

static void free_stage(host_stage *s) {
  if (!s)
    return;
  free_conv(&s->upsample);
  for (int b = 0; b < RESBLOCKS; b++) {
    for (int i = 0; i < 6; i++)
      free_act(&s->blocks[b].activations[i]);
    for (int p = 0; p < RESIDUAL_PAIRS; p++) {
      free_conv(&s->blocks[b].convs1[p]);
      free_conv(&s->blocks[b].convs2[p]);
    }
  }
  memset(s, 0, sizeof(*s));
}

static int load_vec(const h3_st_store *st, const char *name, float **out,
                    size_t n, char *error, size_t error_size) {
  float *buf = xmalloc_f(n);
  if (!buf) {
    fail1(error, error_size, "oom loading tensor");
    return 0;
  }
  char err[256];
  if (h3_st_store_load_f32(st, name, buf, n, err, sizeof(err)) != 0) {
    fail2(error, error_size, name, err);
    free(buf);
    return 0;
  }
  *out = buf;
  return 1;
}

static int ensure_bias(host_conv *c) {
  if (c->bias)
    return 1;
  c->zero_bias = xmalloc_f((size_t)c->out_ch);
  if (!c->zero_bias)
    return 0;
  memset(c->zero_bias, 0, (size_t)c->out_ch * sizeof(float));
  return 1;
}

static const float *bias_ptr(const host_conv *c) {
  return c->bias ? c->bias : c->zero_bias;
}

static int fold_weight_norm(host_conv *c, const float *vector,
                            const float *magnitude, char *error,
                            size_t error_size) {
  int outer = c->transpose ? c->in_ch : c->out_ch;
  int inner_ch = c->transpose ? c->out_ch : c->in_ch;
  size_t inner = (size_t)inner_ch * (size_t)c->kernel;
  size_t elems = (size_t)outer * inner;
  c->weight = xmalloc_f(elems);
  if (!c->weight) {
    fail1(error, error_size, "oom weight fold");
    return 0;
  }
  if (h3_audio_vae_weight_norm_f32(c->weight, vector, magnitude, outer,
                                   (int)inner) != 0) {
    fail1(error, error_size, "weight_norm failed");
    return 0;
  }
  return 1;
}

static int load_plain_conv(const h3_st_store *st, host_conv *c,
                           const char *prefix, int in_ch, int out_ch, int kernel,
                           int padding, int dilation, int has_bias, char *error,
                           size_t error_size) {
  memset(c, 0, sizeof(*c));
  c->in_ch = in_ch;
  c->out_ch = out_ch;
  c->kernel = kernel;
  c->padding = padding;
  c->dilation = dilation;
  c->stride = 1;
  c->transpose = 0;
  c->has_bias = has_bias;
  char name[192];
  snprintf(name, sizeof(name), "%s.weight", prefix);
  size_t w_n = (size_t)out_ch * (size_t)in_ch * (size_t)kernel;
  if (!load_vec(st, name, &c->weight, w_n, error, error_size))
    return 0;
  if (has_bias) {
    snprintf(name, sizeof(name), "%s.bias", prefix);
    if (!load_vec(st, name, &c->bias, (size_t)out_ch, error, error_size))
      return 0;
  } else if (!ensure_bias(c)) {
    fail1(error, error_size, "oom bias");
    return 0;
  }
  return 1;
}

static int load_ln(const h3_st_store *st, float **weight, float **bias,
                   const char *prefix, int dim, char *error, size_t error_size) {
  char name[192];
  snprintf(name, sizeof(name), "%s.weight", prefix);
  if (!load_vec(st, name, weight, (size_t)dim, error, error_size))
    return 0;
  snprintf(name, sizeof(name), "%s.bias", prefix);
  return load_vec(st, name, bias, (size_t)dim, error, error_size);
}

static int load_linear(const h3_st_store *st, float **weight, float **bias,
                       const char *prefix, int in_dim, int out_dim, int has_bias,
                       char *error, size_t error_size) {
  char name[192];
  snprintf(name, sizeof(name), "%s.weight", prefix);
  size_t n = (size_t)out_dim * (size_t)in_dim;
  if (!load_vec(st, name, weight, n, error, error_size))
    return 0;
  if (bias)
    *bias = NULL;
  if (!has_bias)
    return 1;
  if (!bias)
    return 0;
  snprintf(name, sizeof(name), "%s.bias", prefix);
  return load_vec(st, name, bias, (size_t)out_dim, error, error_size);
}

static int load_norm_conv(const h3_st_store *st, host_conv *c,
                          const char *prefix, int in_ch, int out_ch, int kernel,
                          int padding, int dilation, int stride, int transpose,
                          int has_bias, char *error, size_t error_size) {
  memset(c, 0, sizeof(*c));
  c->in_ch = in_ch;
  c->out_ch = out_ch;
  c->kernel = kernel;
  c->padding = padding;
  c->dilation = dilation;
  c->stride = stride;
  c->transpose = transpose;
  c->has_bias = has_bias;
  int outer = transpose ? in_ch : out_ch;
  int inner_ch = transpose ? out_ch : in_ch;
  size_t v_n = (size_t)outer * (size_t)inner_ch * (size_t)kernel;
  char name[192];
  float *vector = NULL, *magnitude = NULL;
  snprintf(name, sizeof(name), "%s.weight_v", prefix);
  if (!load_vec(st, name, &vector, v_n, error, error_size))
    return 0;
  snprintf(name, sizeof(name), "%s.weight_g", prefix);
  /* Stored as [outer,1,1]. */
  if (!load_vec(st, name, &magnitude, (size_t)outer, error, error_size)) {
    free(vector);
    return 0;
  }
  int ok = fold_weight_norm(c, vector, magnitude, error, error_size);
  free(vector);
  free(magnitude);
  if (!ok)
    return 0;
  if (has_bias) {
    snprintf(name, sizeof(name), "%s.bias", prefix);
    if (!load_vec(st, name, &c->bias, (size_t)out_ch, error, error_size))
      return 0;
  } else if (!ensure_bias(c)) {
    fail1(error, error_size, "oom bias");
    return 0;
  }
  return 1;
}

static int load_activation(const h3_st_store *st, host_act *a,
                           const char *prefix, int channels, char *error,
                           size_t error_size) {
  memset(a, 0, sizeof(*a));
  char name[224];
  snprintf(name, sizeof(name), "%s.act.alpha", prefix);
  if (!load_vec(st, name, &a->alpha, (size_t)channels, error, error_size))
    return 0;
  snprintf(name, sizeof(name), "%s.act.beta", prefix);
  if (!load_vec(st, name, &a->beta, (size_t)channels, error, error_size))
    return 0;
  return 1;
}

static int parse_float_array(const char *json, const char *key, float *values,
                             size_t count, char *error, size_t error_size) {
  char pattern[64];
  snprintf(pattern, sizeof(pattern), "\"%s\"", key);
  const char *cursor = strstr(json, pattern);
  if (!cursor || !(cursor = strchr(cursor + strlen(pattern), ':')) ||
      !(cursor = strchr(cursor, '['))) {
    fail2(error, error_size, "config missing", key);
    return 0;
  }
  cursor++;
  for (size_t i = 0; i < count; i++) {
    while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
           *cursor == '\t')
      cursor++;
    errno = 0;
    char *end = NULL;
    float value = strtof(cursor, &end);
    if (errno || end == cursor || !isfinite(value)) {
      fail2(error, error_size, "config malformed", key);
      return 0;
    }
    values[i] = value;
    cursor = end;
    while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
           *cursor == '\t')
      cursor++;
    if (i + 1 < count) {
      if (*cursor++ != ',') {
        fail2(error, error_size, "config short", key);
        return 0;
      }
    } else if (*cursor != ']') {
      fail2(error, error_size, "config long", key);
      return 0;
    }
  }
  return 1;
}

static int load_latent_norm(const char *weight_directory, float *mean,
                            float *deviation, char *error, size_t error_size) {
  char path[1100];
  snprintf(path, sizeof(path), "%s/config.json", weight_directory);
  FILE *file = fopen(path, "rb");
  if (!file) {
    fail2(error, error_size, "cannot open", path);
    return 0;
  }
  if (fseek(file, 0, SEEK_END)) {
    fclose(file);
    fail1(error, error_size, "seek config");
    return 0;
  }
  long end = ftell(file);
  if (end < 1 || end > 1024 * 1024 || fseek(file, 0, SEEK_SET)) {
    fclose(file);
    fail1(error, error_size, "invalid config size");
    return 0;
  }
  char *json = (char *)malloc((size_t)end + 1);
  if (!json || fread(json, 1, (size_t)end, file) != (size_t)end) {
    free(json);
    fclose(file);
    fail1(error, error_size, "cannot read config");
    return 0;
  }
  json[end] = '\0';
  fclose(file);
  int ok = parse_float_array(json, "latents_mean", mean, LATENT_CHANNELS,
                             error, error_size) &&
           parse_float_array(json, "latents_std", deviation, LATENT_CHANNELS,
                             error, error_size);
  free(json);
  if (!ok)
    return 0;
  for (int c = 0; c < LATENT_CHANNELS; c++) {
    if (deviation[c] <= 0.0f) {
      fail1(error, error_size, "invalid latents_std");
      return 0;
    }
  }
  return 1;
}

/* Run conv once per stereo batch plane. in/out layout [B, L, C]. */
static int conv_batch(float *dst, const float *src, const host_conv *c,
                      int length) {
  size_t in_plane = (size_t)length * (size_t)c->in_ch;
  int out_len;
  if (c->transpose)
    out_len = (length - 1) * c->stride + c->kernel - 2 * c->padding;
  else
    out_len = h3_audio_vae_conv1d_out_length(length, c->kernel, c->stride,
                                             c->padding, c->dilation);
  if (out_len < 1)
    return 0;
  size_t out_plane = (size_t)out_len * (size_t)c->out_ch;
  for (int b = 0; b < STEREO; b++) {
    const float *in = src + (size_t)b * in_plane;
    float *out = dst + (size_t)b * out_plane;
    int rc;
    if (c->transpose)
      rc = h3_audio_vae_conv_transpose1d_f32(out, in, c->weight, bias_ptr(c),
                                             length, c->in_ch, c->out_ch,
                                             c->kernel, c->stride, c->padding);
    else
      rc = h3_audio_vae_conv1d_stride_f32(out, in, c->weight, bias_ptr(c),
                                          length, c->in_ch, c->out_ch, c->kernel,
                                          c->stride, c->padding, c->dilation);
    if (rc != 0)
      return 0;
  }
  return 1;
}

static int act_batch(float *dst, const float *src, const host_act *a,
                     const float *up_f, const float *down_f, int length,
                     int channels) {
  size_t plane = (size_t)length * (size_t)channels;
  for (int b = 0; b < STEREO; b++) {
    if (h3_audio_vae_alias_free_snake_f32(
            dst + (size_t)b * plane, src + (size_t)b * plane, a->alpha, a->beta,
            up_f, down_f, length, channels) != 0)
      return 0;
  }
  return 1;
}

static void add_scaled(float *dst, const float *a, const float *b, float sa,
                       float sb, size_t n) {
  for (size_t i = 0; i < n; i++)
    dst[i] = sa * a[i] + sb * b[i];
}

static int load_stage(const h3_st_store *st, host_stage *stage, int index,
                      char *error, size_t error_size) {
  memset(stage, 0, sizeof(*stage));
  int in_ch = DECODER_DIM >> index;
  int out_ch = DECODER_DIM >> (index + 1);
  int rate = h3_audio_vae_upsample_rates[index];
  int kernel = h3_audio_vae_upsample_kernels[index];
  int padding = (kernel - rate) / 2;
  char prefix[192];
  snprintf(prefix, sizeof(prefix), "decoder.ups.%d.0", index);
  if (!load_norm_conv(st, &stage->upsample, prefix, in_ch, out_ch, kernel,
                      padding, 1, rate, 1, 1, error, error_size))
    return 0;
  for (int block_index = 0; block_index < RESBLOCKS; block_index++) {
    int global = index * RESBLOCKS + block_index;
    host_resblock *block = &stage->blocks[block_index];
    int residual_kernel = h3_audio_vae_residual_kernels[block_index];
    for (int pair = 0; pair < RESIDUAL_PAIRS; pair++) {
      for (int which = 0; which < 2; which++) {
        int activation = pair * 2 + which;
        snprintf(prefix, sizeof(prefix),
                 "decoder.resblocks.%d.activations.%d", global, activation);
        if (!load_activation(st, &block->activations[activation], prefix,
                             out_ch, error, error_size))
          return 0;
      }
      int dilation = h3_audio_vae_residual_dilations[pair];
      int dilated_padding = dilation * (residual_kernel - 1) / 2;
      snprintf(prefix, sizeof(prefix), "decoder.resblocks.%d.convs1.%d",
               global, pair);
      if (!load_norm_conv(st, &block->convs1[pair], prefix, out_ch, out_ch,
                          residual_kernel, dilated_padding, dilation, 1, 0, 1,
                          error, error_size))
        return 0;
      snprintf(prefix, sizeof(prefix), "decoder.resblocks.%d.convs2.%d",
               global, pair);
      if (!load_norm_conv(st, &block->convs2[pair], prefix, out_ch, out_ch,
                          residual_kernel, (residual_kernel - 1) / 2, 1, 1, 0,
                          1, error, error_size))
        return 0;
    }
  }
  return 1;
}

static int run_stage(float **hidden_io, int *length_io, host_stage *stage,
                     int index, const float *up_f, const float *down_f,
                     char *error, size_t error_size) {
  int input_length = *length_io;
  int channels = DECODER_DIM >> (index + 1);
  int kernel = h3_audio_vae_upsample_kernels[index];
  int stride = h3_audio_vae_upsample_rates[index];
  int padding = (kernel - stride) / 2;
  long long expanded =
      (long long)(input_length - 1) * stride + kernel - 2 * padding;
  if (expanded < 1 || expanded > 100000000) {
    fail1(error, error_size, "AudioVAE stage length overflow");
    return 0;
  }
  int output_length = (int)expanded;
  size_t elements = (size_t)STEREO * (size_t)output_length * (size_t)channels;
  float *upsampled = xmalloc_f(elements);
  float *sum = xmalloc_f(elements);
  float *work = xmalloc_f(elements);
  float *activated = xmalloc_f(elements);
  float *branch = xmalloc_f(elements);
  if (!upsampled || !sum || !work || !activated || !branch) {
    free(upsampled);
    free(sum);
    free(work);
    free(activated);
    free(branch);
    fail1(error, error_size, "oom AudioVAE stage");
    return 0;
  }
  int ok = conv_batch(upsampled, *hidden_io, &stage->upsample, input_length);
  for (int block_index = 0; ok && block_index < RESBLOCKS; block_index++) {
    host_resblock *block = &stage->blocks[block_index];
    float *target = block_index == 0 ? sum : work;
    memcpy(target, upsampled, elements * sizeof(float));
    for (int pair = 0; ok && pair < RESIDUAL_PAIRS; pair++) {
      ok = act_batch(activated, target, &block->activations[pair * 2], up_f,
                     down_f, output_length, channels) &&
           conv_batch(branch, activated, &block->convs1[pair],
                      output_length) &&
           act_batch(activated, branch, &block->activations[pair * 2 + 1], up_f,
                     down_f, output_length, channels) &&
           conv_batch(branch, activated, &block->convs2[pair],
                      output_length);
      if (ok)
        add_scaled(target, target, branch, 1.0f, 1.0f, elements);
    }
    if (ok && block_index == 1)
      add_scaled(sum, sum, work, 1.0f, 1.0f, elements);
    if (ok && block_index == 2)
      add_scaled(sum, sum, work, 1.0f / 3.0f, 1.0f / 3.0f, elements);
  }
  free(upsampled);
  free(work);
  free(activated);
  free(branch);
  if (!ok) {
    free(sum);
    fail1(error, error_size, "AudioVAE stage failed");
    return 0;
  }
  free(*hidden_io);
  *hidden_io = sum;
  *length_io = output_length;
  return 1;
}

void h3_audio_waveform_host_free(h3_audio_waveform_host *w) {
  if (!w)
    return;
  free(w->pcm);
  memset(w, 0, sizeof(*w));
}

int h3_audio_vae_decode_host(const char *weight_directory,
                             const float *normalized_latent, int latent_length,
                             h3_audio_waveform_host *output, char *error,
                             size_t error_size) {
  double t_dec = h3_prof_now_ms ? h3_prof_now_ms() : 0;
  if (error && error_size)
    error[0] = '\0';
  if (output)
    memset(output, 0, sizeof(*output));
  if (!weight_directory || !*weight_directory || !normalized_latent || !output ||
      latent_length < 1 || latent_length > 1000000 / HOP_LENGTH) {
    fail1(error, error_size, "invalid AudioVAE decode arguments");
    return 0;
  }

  float mean[LATENT_CHANNELS], deviation[LATENT_CHANNELS];
  if (!load_latent_norm(weight_directory, mean, deviation, error, error_size))
    return 0;

  char err[256];
  h3_st_store *st = h3_st_store_open(weight_directory, err, sizeof(err));
  if (!st) {
    fail2(error, error_size, "open weights", err);
    return 0;
  }
  h3_st_store_set_prof_tag(st, "h3_avae_wload");

  float *up_f = NULL, *down_f = NULL;
  if (!load_vec(st, "decoder.activation_post.upsample.filter", &up_f,
                FILTER_SIZE, error, error_size) ||
      !load_vec(st, "decoder.activation_post.downsample.lowpass.filter",
                &down_f, FILTER_SIZE, error, error_size)) {
    free(up_f);
    free(down_f);
    h3_st_store_free(st);
    return 0;
  }

  /* Denormalize + transpose to [B, T, C]. */
  size_t latent_elems =
      (size_t)STEREO * (size_t)latent_length * (size_t)LATENT_CHANNELS;
  float *rows = xmalloc_f(latent_elems);
  if (!rows) {
    fail1(error, error_size, "oom latent");
    free(up_f);
    free(down_f);
    h3_st_store_free(st);
    return 0;
  }
  size_t destination = 0;
  for (int stereo = 0; stereo < STEREO; stereo++)
    for (int time = 0; time < latent_length; time++)
      for (int channel = 0; channel < LATENT_CHANNELS; channel++) {
        size_t source = ((size_t)channel * STEREO + (size_t)stereo) *
                            (size_t)latent_length +
                        (size_t)time;
        rows[destination++] =
            normalized_latent[source] * deviation[channel] + mean[channel];
      }

  host_conv projection = {0}, pre = {0};
  float *projected = xmalloc_f((size_t)STEREO * (size_t)latent_length *
                               (size_t)LATENT_DIM);
  float *hidden = xmalloc_f((size_t)STEREO * (size_t)latent_length *
                            (size_t)DECODER_DIM);
  int ok = projected && hidden &&
           load_plain_conv(st, &projection, "dec_in_proj", LATENT_CHANNELS,
                           LATENT_DIM, 1, 0, 1, 1, error, error_size) &&
           load_norm_conv(st, &pre, "decoder.conv_pre", LATENT_DIM, DECODER_DIM,
                          7, 3, 1, 1, 0, 1, error, error_size);
  if (ok)
    ok = conv_batch(projected, rows, &projection, latent_length) &&
         conv_batch(hidden, projected, &pre, latent_length);
  free(rows);
  free(projected);
  free_conv(&projection);
  free_conv(&pre);
  if (!ok) {
    free(hidden);
    free(up_f);
    free(down_f);
    h3_st_store_free(st);
    if (error && error_size && !error[0])
      fail1(error, error_size, "AudioVAE input failed");
    return 0;
  }

  int length = latent_length;
  for (int index = 0; ok && index < STAGES; index++) {
    double t_s = h3_prof_now_ms ? h3_prof_now_ms() : 0;
    host_stage stage;
    ok = load_stage(st, &stage, index, error, error_size) &&
         run_stage(&hidden, &length, &stage, index, up_f, down_f, error,
                   error_size);
    free_stage(&stage);
    if (getenv("H3_AVAE_DBG")) {
      fprintf(stderr,
              "video-c: avae stage %d len=%d->%d ch=%d took %.1f ms\n", index,
              0, length, DECODER_DIM >> (index + 1),
              (h3_prof_now_ms ? h3_prof_now_ms() : 0) - t_s);
      fflush(stderr);
    }
  }

  if (ok && length != latent_length * HOP_LENGTH) {
    fail1(error, error_size, "AudioVAE produced wrong sample count");
    ok = 0;
  }

  if (ok) {
    host_act post_act = {0};
    host_conv post = {0};
    size_t hidden_elems = (size_t)STEREO * (size_t)length * 8;
    size_t wave_elems = (size_t)STEREO * (size_t)length;
    float *activated = xmalloc_f(hidden_elems);
    float *waveform = xmalloc_f(wave_elems);
    ok = activated && waveform &&
         load_activation(st, &post_act, "decoder.activation_post", 8, error,
                         error_size) &&
         load_norm_conv(st, &post, "decoder.conv_post", 8, 1, 7, 3, 1, 1, 0, 0,
                        error, error_size);
    if (ok)
      ok = act_batch(activated, hidden, &post_act, up_f, down_f, length, 8) &&
           conv_batch(waveform, activated, &post, length);
    free(activated);
    free_act(&post_act);
    free_conv(&post);
    if (ok) {
      for (size_t i = 0; i < wave_elems; i++) {
        float v = waveform[i];
        if (v > 1.0f)
          v = 1.0f;
        if (v < -1.0f)
          v = -1.0f;
        waveform[i] = v;
      }
      output->pcm = waveform;
      output->channels = STEREO;
      output->samples = length;
      output->sample_rate = SAMPLE_RATE;
      waveform = NULL;
    }
    free(waveform);
  }

  free(hidden);
  free(up_f);
  free(down_f);
  h3_st_store_free(st);
  if (getenv("H3_AVAE_DBG")) {
    fprintf(stderr, "video-c: avae total %.1f ms (lt=%d)\n",
            (h3_prof_now_ms ? h3_prof_now_ms() : 0) - t_dec, latent_length);
    fflush(stderr);
  }
  if (!ok)
    h3_audio_waveform_host_free(output);
  return ok;
}

void h3_audio_vae_fill_unit_latent(float *latent, size_t n) {
  if (!latent)
    return;
  for (size_t i = 0; i < n; i++)
    latent[i] =
        ((float)((i * 1103515245u + 12345u) % 1000) / 1000.0f) - 0.5f;
}

int h3_audio_waveform_write_wav(const h3_audio_waveform_host *w, const char *path,
                                char *error, size_t error_size) {
  if (!w || !w->pcm || !path || w->channels != 2 || w->samples < 1 ||
      w->sample_rate < 1) {
    fail1(error, error_size, "invalid wav arguments");
    return 0;
  }
  FILE *f = fopen(path, "wb");
  if (!f) {
    fail2(error, error_size, "cannot create wav", path);
    return 0;
  }
  uint32_t data_bytes =
      (uint32_t)w->channels * (uint32_t)w->samples * 2u;
  uint32_t riff_size = 36u + data_bytes;
  uint32_t byte_rate = (uint32_t)w->sample_rate * (uint32_t)w->channels * 2u;
  uint16_t block_align = (uint16_t)(w->channels * 2);
  uint16_t audio_fmt = 1, ch = (uint16_t)w->channels, bits = 16;
  uint32_t rate = (uint32_t)w->sample_rate;
  uint32_t fmt_chunk = 16;
  fwrite("RIFF", 1, 4, f);
  fwrite(&riff_size, 4, 1, f);
  fwrite("WAVEfmt ", 1, 8, f);
  fwrite(&fmt_chunk, 4, 1, f);
  fwrite(&audio_fmt, 2, 1, f);
  fwrite(&ch, 2, 1, f);
  fwrite(&rate, 4, 1, f);
  fwrite(&byte_rate, 4, 1, f);
  fwrite(&block_align, 2, 1, f);
  fwrite(&bits, 2, 1, f);
  fwrite("data", 1, 4, f);
  fwrite(&data_bytes, 4, 1, f);
  for (int t = 0; t < w->samples; t++) {
    for (int c = 0; c < w->channels; c++) {
      float v = w->pcm[(size_t)c * (size_t)w->samples + (size_t)t];
      if (v > 1.0f)
        v = 1.0f;
      if (v < -1.0f)
        v = -1.0f;
      int s = (int)lrintf(v * 32767.0f);
      if (s > 32767)
        s = 32767;
      if (s < -32768)
        s = -32768;
      int16_t s16 = (int16_t)s;
      fwrite(&s16, 2, 1, f);
    }
  }
  int ok = ferror(f) == 0;
  fclose(f);
  if (!ok)
    fail1(error, error_size, "wav write failed");
  return ok;
}

void h3_audio_latent_host_free(h3_audio_latent_host *z) {
  if (!z)
    return;
  free(z->values);
  memset(z, 0, sizeof(*z));
}

static int encoder_linear_rows(float *dst, const float *src, const float *w,
                               const float *b, int rows, int in_dim,
                               int out_dim) {
  return h3_audio_vae_linear_f32(dst, src, w, b, rows, in_dim, out_dim) == 0;
}

int h3_audio_vae_encode_host(const char *weight_directory, const float *pcm,
                             int samples, h3_audio_latent_host *output,
                             char *error, size_t error_size) {
  if (error && error_size)
    error[0] = '\0';
  if (output)
    memset(output, 0, sizeof(*output));
  if (!weight_directory || !*weight_directory || !pcm || !output || samples < 1 ||
      samples > SAMPLE_RATE * 15) {
    fail1(error, error_size, "invalid AudioVAE encode arguments");
    return 0;
  }

  float mean[LATENT_CHANNELS], deviation[LATENT_CHANNELS];
  if (!load_latent_norm(weight_directory, mean, deviation, error, error_size))
    return 0;

  char err[256];
  h3_st_store *st = h3_st_store_open(weight_directory, err, sizeof(err));
  if (!st) {
    fail2(error, error_size, "open weights", err);
    return 0;
  }
  h3_st_store_set_prof_tag(st, "h3_avae_wload");

  int padded = h3_audio_vae_pad_samples(samples);
  int length = padded;
  float *hidden = xmalloc_f((size_t)STEREO * (size_t)padded * H3_AUDIO_VAE_ENCODER_DIM);
  float *input = xmalloc_f((size_t)STEREO * (size_t)padded);
  if (!hidden || !input) {
    fail1(error, error_size, "oom AudioVAE encoder input");
    free(hidden);
    free(input);
    h3_st_store_free(st);
    return 0;
  }
  memset(input, 0, (size_t)STEREO * (size_t)padded * sizeof(float));
  for (int ch = 0; ch < STEREO; ch++)
    memcpy(input + (size_t)ch * (size_t)padded, pcm + (size_t)ch * (size_t)samples,
           (size_t)samples * sizeof(float));

  host_conv conv0 = {0};
  int ok = load_norm_conv(st, &conv0, "encoder.block.0", 1, H3_AUDIO_VAE_ENCODER_DIM,
                          7, 3, 1, 1, 0, 1, error, error_size) &&
           conv_batch(hidden, input, &conv0, padded);
  free(input);
  free_conv(&conv0);
  if (!ok) {
    free(hidden);
    h3_st_store_free(st);
    return 0;
  }

  int channels = H3_AUDIO_VAE_ENCODER_DIM;
  for (int stage = 1; ok && stage <= H3_AUDIO_VAE_ENCODER_STAGES; stage++) {
    for (int residual = 0; ok && residual < H3_AUDIO_VAE_ENCODER_RESIDUALS;
         residual++) {
      char name[224];
      float *alpha1 = NULL, *alpha2 = NULL;
      host_conv conv1 = {0}, conv2 = {0};
      int dilation = h3_audio_vae_encoder_dilations[residual];
      snprintf(name, sizeof(name), "encoder.block.%d.block.%d.block.0.alpha",
               stage, residual);
      ok = load_vec(st, name, &alpha1, (size_t)channels, error, error_size);
      snprintf(name, sizeof(name), "encoder.block.%d.block.%d.block.1", stage,
               residual);
      if (ok)
        ok = load_norm_conv(st, &conv1, name, channels, channels, 7,
                            3 * dilation, dilation, 1, 0, 1, error, error_size);
      snprintf(name, sizeof(name), "encoder.block.%d.block.%d.block.2.alpha",
               stage, residual);
      if (ok)
        ok = load_vec(st, name, &alpha2, (size_t)channels, error, error_size);
      snprintf(name, sizeof(name), "encoder.block.%d.block.%d.block.3", stage,
               residual);
      if (ok)
        ok = load_norm_conv(st, &conv2, name, channels, channels, 1, 0, 1, 1, 0,
                            1, error, error_size);
      size_t elems = (size_t)STEREO * (size_t)length * (size_t)channels;
      float *activated = ok ? xmalloc_f(elems) : NULL;
      float *branch = ok ? xmalloc_f(elems) : NULL;
      if (ok && (!activated || !branch)) {
        fail1(error, error_size, "oom AudioVAE encoder residual");
        ok = 0;
      }
      if (ok)
        ok = h3_audio_vae_snake1d_f32(activated, hidden, alpha1, STEREO, length,
                                      channels) == 0 &&
             conv_batch(branch, activated, &conv1, length) &&
             h3_audio_vae_snake1d_f32(activated, branch, alpha2, STEREO, length,
                                      channels) == 0 &&
             conv_batch(branch, activated, &conv2, length);
      if (ok)
        add_scaled(hidden, hidden, branch, 1.0f, 1.0f, elems);
      free(alpha1);
      free(alpha2);
      free(activated);
      free(branch);
      free_conv(&conv1);
      free_conv(&conv2);
    }

    int stride = h3_audio_vae_encoder_strides[stage - 1];
    int kernel = stride * 2;
    int padding = (stride + 1) / 2;
    int out_len = h3_audio_vae_conv1d_out_length(length, kernel, stride, padding,
                                                 1);
    char name[192];
    float *alpha = NULL;
    host_conv down = {0};
    snprintf(name, sizeof(name), "encoder.block.%d.block.3.alpha", stage);
    if (ok)
      ok = load_vec(st, name, &alpha, (size_t)channels, error, error_size);
    snprintf(name, sizeof(name), "encoder.block.%d.block.4", stage);
    if (ok)
      ok = load_norm_conv(st, &down, name, channels, channels * 2, kernel,
                          padding, 1, stride, 0, 1, error, error_size) &&
           out_len > 0;
    size_t in_elems = (size_t)STEREO * (size_t)length * (size_t)channels;
    size_t out_elems = (size_t)STEREO * (size_t)out_len * (size_t)channels * 2;
    float *activated = ok ? xmalloc_f(in_elems) : NULL;
    float *down_h = ok ? xmalloc_f(out_elems) : NULL;
    if (ok && (!activated || !down_h)) {
      fail1(error, error_size, "oom AudioVAE encoder downsample");
      ok = 0;
    }
    if (ok)
      ok = h3_audio_vae_snake1d_f32(activated, hidden, alpha, STEREO, length,
                                    channels) == 0 &&
           conv_batch(down_h, activated, &down, length);
    free(alpha);
    free(activated);
    free_conv(&down);
    if (ok) {
      free(hidden);
      hidden = down_h;
      length = out_len;
      channels *= 2;
    } else {
      free(down_h);
    }
  }

  if (ok && (channels != LATENT_DIM || length != padded / HOP_LENGTH)) {
    fail1(error, error_size, "AudioVAE encoder produced invalid geometry");
    ok = 0;
  }

  if (ok) {
    float *alpha = NULL;
    host_conv convf = {0};
    size_t elems = (size_t)STEREO * (size_t)length * (size_t)LATENT_DIM;
    ok = load_vec(st, "encoder.block.6.alpha", &alpha, (size_t)LATENT_DIM, error,
                  error_size) &&
         load_norm_conv(st, &convf, "encoder.block.7", LATENT_DIM, LATENT_DIM, 3,
                        1, 1, 1, 0, 1, error, error_size);
    float *activated = ok ? xmalloc_f(elems) : NULL;
    float *next = ok ? xmalloc_f(elems) : NULL;
    if (ok && (!activated || !next)) {
      fail1(error, error_size, "oom AudioVAE encoder final");
      ok = 0;
    }
    if (ok)
      ok = h3_audio_vae_snake1d_f32(activated, hidden, alpha, STEREO, length,
                                    LATENT_DIM) == 0 &&
           conv_batch(next, activated, &convf, length);
    free(alpha);
    free(activated);
    free_conv(&convf);
    if (ok) {
      free(hidden);
      hidden = next;
    } else {
      free(next);
    }
  }

  int rows = STEREO * length;
  float *base = NULL;
  if (ok) {
    float *n3w = NULL, *n3b = NULL, *pw = NULL, *pb = NULL;
    float *normalized = xmalloc_f((size_t)rows * LATENT_DIM);
    base = xmalloc_f((size_t)rows * LATENT_CHANNELS);
    ok = normalized && base &&
         load_ln(st, &n3w, &n3b, "pre_block.norm3", LATENT_DIM, error,
                 error_size) &&
         load_linear(st, &pw, &pb, "pre_block.proj", LATENT_DIM, LATENT_CHANNELS,
                     1, error, error_size) &&
         h3_audio_vae_layer_norm_f32(normalized, hidden, n3w, n3b, rows,
                                     LATENT_DIM, 1e-5f) == 0 &&
         encoder_linear_rows(base, normalized, pw, pb, rows, LATENT_DIM,
                             LATENT_CHANNELS);
    free(n3w);
    free(n3b);
    free(pw);
    free(pb);
    free(normalized);
  }

  if (ok) {
    float *n1w = NULL, *n1b = NULL, *qkv_w = NULL, *q_b = NULL, *k_b = NULL,
          *v_b = NULL, *proj_w = NULL, *proj_b = NULL;
    int hd = LATENT_DIM / H3_AUDIO_VAE_ENCODER_HEADS;
    float *normalized = xmalloc_f((size_t)rows * LATENT_DIM);
    float *qkv = xmalloc_f((size_t)rows * LATENT_DIM * 3);
    float *query = xmalloc_f((size_t)rows * LATENT_DIM);
    float *key = xmalloc_f((size_t)rows * LATENT_DIM);
    float *value = xmalloc_f((size_t)rows * LATENT_DIM);
    float *attended = xmalloc_f((size_t)rows * LATENT_DIM);
    float *pooled = xmalloc_f((size_t)rows * LATENT_CHANNELS);
    float *projected = xmalloc_f((size_t)rows * LATENT_CHANNELS);
    ok = normalized && qkv && query && key && value && attended && pooled &&
         projected &&
         load_ln(st, &n1w, &n1b, "pre_block.norm1", LATENT_DIM, error,
                 error_size) &&
         load_linear(st, &qkv_w, NULL, "pre_block.attn.qkv", LATENT_DIM,
                     LATENT_DIM * 3, 0, error, error_size) &&
         load_vec(st, "pre_block.attn.q_bias", &q_b, (size_t)LATENT_DIM, error,
                  error_size) &&
         load_vec(st, "pre_block.attn.zero_k_bias", &k_b, (size_t)LATENT_DIM,
                  error, error_size) &&
         load_vec(st, "pre_block.attn.v_bias", &v_b, (size_t)LATENT_DIM, error,
                  error_size) &&
         load_linear(st, &proj_w, &proj_b, "pre_block.attn.proj", LATENT_CHANNELS,
                     LATENT_CHANNELS, 1, error, error_size) &&
         h3_audio_vae_layer_norm_f32(normalized, hidden, n1w, n1b, rows,
                                     LATENT_DIM, 1e-5f) == 0 &&
         encoder_linear_rows(qkv, normalized, qkv_w, NULL, rows, LATENT_DIM,
                             LATENT_DIM * 3) &&
         h3_audio_vae_qkv_split_f32(query, key, value, qkv, q_b, k_b, v_b, rows,
                                    LATENT_DIM) == 0 &&
         h3_audio_vae_sdpa_causal_f32(attended, query, key, value, STEREO,
                                      length, H3_AUDIO_VAE_ENCODER_HEADS, hd,
                                      1.0f / sqrtf((float)hd)) == 0 &&
         h3_audio_vae_attention_pool_f32(pooled, attended, rows,
                                         H3_AUDIO_VAE_ENCODER_HEADS, hd,
                                         LATENT_CHANNELS) == 0 &&
         encoder_linear_rows(projected, pooled, proj_w, proj_b, rows,
                             LATENT_CHANNELS, LATENT_CHANNELS);
    if (ok)
      add_scaled(base, base, projected, 1.0f, 1.0f,
                 (size_t)rows * LATENT_CHANNELS);
    free(n1w);
    free(n1b);
    free(qkv_w);
    free(q_b);
    free(k_b);
    free(v_b);
    free(proj_w);
    free(proj_b);
    free(normalized);
    free(qkv);
    free(query);
    free(key);
    free(value);
    free(attended);
    free(pooled);
    free(projected);
  }

  if (ok) {
    float *n2w = NULL, *n2b = NULL, *nmw = NULL, *nmb = NULL;
    float *w0 = NULL, *b0 = NULL, *w1 = NULL, *b1 = NULL, *w2 = NULL, *b2 = NULL;
    float *norm2 = xmalloc_f((size_t)rows * LATENT_CHANNELS);
    float *normm = xmalloc_f((size_t)rows * LATENT_CHANNELS);
    float *gate = xmalloc_f((size_t)rows * LATENT_CHANNELS * 2);
    float *lin = xmalloc_f((size_t)rows * LATENT_CHANNELS * 2);
    float *geglu = xmalloc_f((size_t)rows * LATENT_CHANNELS * 2);
    float *branch = xmalloc_f((size_t)rows * LATENT_CHANNELS);
    ok = norm2 && normm && gate && lin && geglu && branch &&
         load_ln(st, &n2w, &n2b, "pre_block.norm2", LATENT_CHANNELS, error,
                 error_size) &&
         load_ln(st, &nmw, &nmb, "pre_block.mlp.norm", LATENT_CHANNELS, error,
                 error_size) &&
         load_linear(st, &w0, &b0, "pre_block.mlp.w0", LATENT_CHANNELS,
                     LATENT_CHANNELS * 2, 1, error, error_size) &&
         load_linear(st, &w1, &b1, "pre_block.mlp.w1", LATENT_CHANNELS,
                     LATENT_CHANNELS * 2, 1, error, error_size) &&
         load_linear(st, &w2, &b2, "pre_block.mlp.w2", LATENT_CHANNELS * 2,
                     LATENT_CHANNELS, 1, error, error_size) &&
         h3_audio_vae_layer_norm_f32(norm2, base, n2w, n2b, rows, LATENT_CHANNELS,
                                     1e-5f) == 0 &&
         h3_audio_vae_layer_norm_f32(normm, norm2, nmw, nmb, rows,
                                     LATENT_CHANNELS, 1e-5f) == 0 &&
         encoder_linear_rows(gate, normm, w0, b0, rows, LATENT_CHANNELS,
                             LATENT_CHANNELS * 2) &&
         encoder_linear_rows(lin, normm, w1, b1, rows, LATENT_CHANNELS,
                             LATENT_CHANNELS * 2) &&
         h3_audio_vae_geglu_f32(geglu, gate, lin,
                                (size_t)rows * LATENT_CHANNELS * 2) == 0 &&
         encoder_linear_rows(branch, geglu, w2, b2, rows, LATENT_CHANNELS * 2,
                             LATENT_CHANNELS);
    if (ok)
      add_scaled(base, base, branch, 1.0f, 1.0f,
                 (size_t)rows * LATENT_CHANNELS);
    free(n2w);
    free(n2b);
    free(nmw);
    free(nmb);
    free(w0);
    free(b0);
    free(w1);
    free(b1);
    free(w2);
    free(b2);
    free(norm2);
    free(normm);
    free(gate);
    free(lin);
    free(geglu);
    free(branch);
  }

  if (ok) {
    host_conv mean_proj = {0};
    size_t elems = (size_t)STEREO * (size_t)length * (size_t)LATENT_CHANNELS;
    float *rows_bt = xmalloc_f(elems);
    output->values = xmalloc_f(elems);
    ok = rows_bt && output->values &&
         load_plain_conv(st, &mean_proj, "mean_proj", LATENT_CHANNELS,
                         LATENT_CHANNELS, 1, 0, 1, 1, error, error_size) &&
         conv_batch(rows_bt, base, &mean_proj, length);
    if (ok) {
      for (int channel = 0; channel < LATENT_CHANNELS; channel++)
        for (int stereo = 0; stereo < STEREO; stereo++)
          for (int time = 0; time < length; time++) {
            size_t source = ((size_t)stereo * (size_t)length + (size_t)time) *
                                LATENT_CHANNELS +
                            (size_t)channel;
            size_t dest = ((size_t)channel * STEREO + (size_t)stereo) *
                              (size_t)length +
                          (size_t)time;
            output->values[dest] =
                (rows_bt[source] - mean[channel]) / deviation[channel];
          }
      output->channels = LATENT_CHANNELS;
      output->stereo = STEREO;
      output->length = length;
    }
    free(rows_bt);
    free_conv(&mean_proj);
  }

  free(base);
  free(hidden);
  h3_st_store_free(st);
  if (!ok)
    h3_audio_latent_host_free(output);
  return ok;
}
