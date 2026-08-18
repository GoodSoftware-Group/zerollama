#include "h3_video_vae_encode.h"
#include "h3_dit_host.h"
#include "h3_st_store.h"
#include "h3_video_vae_host.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
  RGB = H3_VIDEO_VAE_IN_CHANNELS,
  LATENT_CH = H3_VIDEO_VAE_LATENT_CHANNELS,
  MOMENT_CH = H3_VIDEO_VAE_MOMENT_CHANNELS,
  LEVELS = H3_VIDEO_VAE_LEVELS,
  BLOCKS = H3_VIDEO_VAE_LAYERS_PER_BLOCK,
  GROUPS = H3_VIDEO_VAE_NORM_GROUPS,
  SPATIAL = 16
};

typedef struct {
  int in_ch, out_ch, kernel, st, sh, sw;
  int depth_front, hb, ha, wb, wa;
  float *weight;
  float *bias;
} vconv;

typedef struct {
  float *weight;
  float *bias;
} vnorm;

typedef struct {
  vnorm norm1, norm2;
  vconv conv1, conv2, shortcut;
  int has_shortcut;
} vblock;

typedef struct {
  vblock blocks[BLOCKS];
  vconv downsample;
  int has_downsample;
} vlevel;

static void fail1(char *error, size_t n, const char *msg) {
  if (error && n)
    snprintf(error, n, "%s", msg);
}

static void fail2(char *error, size_t n, const char *a, const char *b) {
  if (error && n)
    snprintf(error, n, "%s: %s", a, b);
}

static float *xmalloc_f(size_t n) {
  return n ? (float *)malloc(n * sizeof(float)) : NULL;
}

static void free_conv(vconv *c) {
  if (!c)
    return;
  free(c->weight);
  free(c->bias);
  memset(c, 0, sizeof(*c));
}

static void free_norm(vnorm *n) {
  if (!n)
    return;
  free(n->weight);
  free(n->bias);
  memset(n, 0, sizeof(*n));
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

static int load_conv(const h3_st_store *st, vconv *c, const char *prefix,
                     int in_ch, int out_ch, int kernel, int stride_t, int stride_h,
                     int stride_w, int depth_front, int hb, int ha, int wb, int wa,
                     char *error, size_t error_size) {
  memset(c, 0, sizeof(*c));
  c->in_ch = in_ch;
  c->out_ch = out_ch;
  c->kernel = kernel;
  c->st = stride_t;
  c->sh = stride_h;
  c->sw = stride_w;
  c->depth_front = depth_front;
  c->hb = hb;
  c->ha = ha;
  c->wb = wb;
  c->wa = wa;
  char name[192];
  snprintf(name, sizeof(name), "%s.weight", prefix);
  size_t wn = (size_t)out_ch * (size_t)in_ch * (size_t)kernel * (size_t)kernel *
              (size_t)kernel;
  if (!load_vec(st, name, &c->weight, wn, error, error_size))
    return 0;
  snprintf(name, sizeof(name), "%s.bias", prefix);
  return load_vec(st, name, &c->bias, (size_t)out_ch, error, error_size);
}

static int load_norm(const h3_st_store *st, vnorm *n, const char *prefix,
                     int channels, char *error, size_t error_size) {
  memset(n, 0, sizeof(*n));
  char name[192];
  snprintf(name, sizeof(name), "%s.weight", prefix);
  if (!load_vec(st, name, &n->weight, (size_t)channels, error, error_size))
    return 0;
  snprintf(name, sizeof(name), "%s.bias", prefix);
  return load_vec(st, name, &n->bias, (size_t)channels, error, error_size);
}

static int parse_float_array(const char *json, const char *key, float *values,
                             int count, char *error, size_t error_size) {
  char pattern[64];
  snprintf(pattern, sizeof(pattern), "\"%s\"", key);
  const char *cursor = strstr(json, pattern);
  if (!cursor || !(cursor = strchr(cursor + strlen(pattern), ':')) ||
      !(cursor = strchr(cursor, '['))) {
    fail2(error, error_size, "video VAE config missing", key);
    return 0;
  }
  cursor++;
  for (int i = 0; i < count; i++) {
    while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
           *cursor == '\t')
      cursor++;
    errno = 0;
    char *end = NULL;
    float value = strtof(cursor, &end);
    if (errno || end == cursor || !isfinite(value)) {
      fail2(error, error_size, "malformed", key);
      return 0;
    }
    values[i] = value;
    cursor = end;
    while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
           *cursor == '\t')
      cursor++;
    if (i + 1 < count) {
      if (*cursor++ != ',') {
        fail2(error, error_size, "short", key);
        return 0;
      }
    } else if (*cursor != ']') {
      fail2(error, error_size, "long", key);
      return 0;
    }
  }
  return 1;
}

static int load_latent_norm(const char *weight_directory, float *mean,
                            float *deviation, char *error, size_t error_size) {
  char path[1100];
  snprintf(path, sizeof(path), "%s/../config.json", weight_directory);
  FILE *file = fopen(path, "rb");
  if (!file) {
    fail2(error, error_size, "open video VAE config", path);
    return 0;
  }
  if (fseek(file, 0, SEEK_END)) {
    fclose(file);
    fail1(error, error_size, "seek video VAE config");
    return 0;
  }
  long end = ftell(file);
  if (end < 1 || end > 1024 * 1024 || fseek(file, 0, SEEK_SET)) {
    fclose(file);
    fail1(error, error_size, "invalid video VAE config");
    return 0;
  }
  char *json = (char *)malloc((size_t)end + 1);
  if (!json || fread(json, 1, (size_t)end, file) != (size_t)end) {
    free(json);
    fclose(file);
    fail1(error, error_size, "read video VAE config");
    return 0;
  }
  json[end] = '\0';
  fclose(file);
  int ok = parse_float_array(json, "latents_mean", mean, LATENT_CH, error,
                             error_size) &&
           parse_float_array(json, "latents_std", deviation, LATENT_CH, error,
                             error_size);
  free(json);
  if (!ok)
    return 0;
  for (int c = 0; c < LATENT_CH; c++) {
    if (deviation[c] <= 0.f) {
      fail1(error, error_size, "invalid latents_std");
      return 0;
    }
  }
  return 1;
}

static size_t ndhwc(int d, int h, int w, int c) {
  return (size_t)d * (size_t)h * (size_t)w * (size_t)c;
}

static int run_conv(float **hidden_io, int *d, int *h, int *w, const vconv *c,
                    char *error, size_t error_size) {
  int needs_pad = c->depth_front || c->hb || c->ha || c->wb || c->wa;
  int pd = *d + c->depth_front;
  int ph = *h + c->hb + c->ha;
  int pw = *w + c->wb + c->wa;
  int od = (pd - c->kernel) / c->st + 1;
  int oh = (ph - c->kernel) / c->sh + 1;
  int ow = (pw - c->kernel) / c->sw + 1;
  if (od < 1 || oh < 1 || ow < 1) {
    fail1(error, error_size, "video encoder conv geometry");
    return 0;
  }
  float *padded = NULL;
  const float *src = *hidden_io;
  int sd = *d, sh = *h, sw = *w;
  if (needs_pad) {
    padded = xmalloc_f(ndhwc(pd, ph, pw, c->in_ch));
    if (!padded) {
      fail1(error, error_size, "oom video encoder pad");
      return 0;
    }
    if (h3_video_vae_pad_ndhwc_f32(padded, src, 1, *d, *h, *w, c->in_ch,
                                   c->depth_front, c->hb, c->ha, c->wb,
                                   c->wa) != 0) {
      free(padded);
      fail1(error, error_size, "video encoder pad failed");
      return 0;
    }
    src = padded;
    sd = pd;
    sh = ph;
    sw = pw;
  }
  float *out = xmalloc_f(ndhwc(od, oh, ow, c->out_ch));
  if (!out) {
    free(padded);
    fail1(error, error_size, "oom video encoder conv");
    return 0;
  }
  int rc = h3_video_vae_conv3d_f32(out, src, c->weight, c->bias, 1, sd, sh, sw,
                                   c->in_ch, c->out_ch, c->kernel, c->kernel,
                                   c->kernel, c->st, c->sh, c->sw);
  free(padded);
  if (rc != 0) {
    free(out);
    fail1(error, error_size, "video encoder conv3d failed");
    return 0;
  }
  free(*hidden_io);
  *hidden_io = out;
  *d = od;
  *h = oh;
  *w = ow;
  return 1;
}

static int run_block(float **hidden_io, int d, int h, int w, int in_ch,
                     int out_ch, const vblock *block, char *error,
                     size_t error_size) {
  size_t in_n = ndhwc(d, h, w, in_ch);
  size_t out_n = ndhwc(d, h, w, out_ch);
  float *norm1 = xmalloc_f(in_n);
  float *work = NULL;
  float *norm2 = xmalloc_f(out_n);
  float *out = NULL;
  float *shortcut = NULL;
  if (!norm1 || !norm2) {
    free(norm1);
    free(norm2);
    fail1(error, error_size, "oom video encoder block");
    return 0;
  }
  int ok = h3_video_vae_group_norm_silu_f32(norm1, *hidden_io, block->norm1.weight,
                                            block->norm1.bias, 1, d, h, w, in_ch,
                                            GROUPS, 1e-6f) == 0;
  work = *hidden_io;
  *hidden_io = norm1;
  int td = d, th = h, tw = w;
  if (ok)
    ok = run_conv(hidden_io, &td, &th, &tw, &block->conv1, error, error_size);
  if (ok && (td != d || th != h || tw != w)) {
    fail1(error, error_size, "video encoder residual spatial mismatch");
    ok = 0;
  }
  if (ok)
    ok = h3_video_vae_group_norm_silu_f32(norm2, *hidden_io, block->norm2.weight,
                                          block->norm2.bias, 1, d, h, w, out_ch,
                                          GROUPS, 1e-6f) == 0;
  if (ok) {
    free(*hidden_io);
    *hidden_io = norm2;
    norm2 = NULL;
    ok = run_conv(hidden_io, &td, &th, &tw, &block->conv2, error, error_size);
  }
  const float *resid = work;
  if (ok && block->has_shortcut) {
    shortcut = xmalloc_f(out_n);
    if (!shortcut) {
      fail1(error, error_size, "oom shortcut");
      ok = 0;
    } else {
      memcpy(shortcut, work, in_n * sizeof(float));
      float *sc = shortcut;
      int sd = d, sh = h, sw = w;
      ok = run_conv(&sc, &sd, &sh, &sw, &block->shortcut, error, error_size);
      shortcut = sc;
      resid = shortcut;
    }
  }
  if (ok) {
    out = *hidden_io;
    for (size_t i = 0; i < out_n; i++)
      out[i] = resid[i] + out[i];
  }
  free(work);
  free(norm2);
  free(shortcut);
  if (!ok) {
    free(*hidden_io);
    *hidden_io = NULL;
  }
  return ok;
}

void h3_video_latent_host_free(h3_video_latent_host *z) {
  if (!z)
    return;
  free(z->values);
  memset(z, 0, sizeof(*z));
}

typedef struct {
  vconv conv_in, conv_out, quant;
  vnorm norm_out;
  vlevel levels[LEVELS];
  float mean[LATENT_CH];
  float deviation[LATENT_CH];
} venc;

static void free_encoder(venc *e) {
  if (!e)
    return;
  free_conv(&e->conv_in);
  free_conv(&e->conv_out);
  free_conv(&e->quant);
  free_norm(&e->norm_out);
  for (int level = 0; level < LEVELS; level++) {
    for (int block = 0; block < BLOCKS; block++) {
      free_norm(&e->levels[level].blocks[block].norm1);
      free_norm(&e->levels[level].blocks[block].norm2);
      free_conv(&e->levels[level].blocks[block].conv1);
      free_conv(&e->levels[level].blocks[block].conv2);
      free_conv(&e->levels[level].blocks[block].shortcut);
    }
    free_conv(&e->levels[level].downsample);
  }
  memset(e, 0, sizeof(*e));
}

static int load_encoder(const char *weight_directory, venc *e, char *error,
                        size_t error_size) {
  memset(e, 0, sizeof(*e));
  if (!load_latent_norm(weight_directory, e->mean, e->deviation, error,
                        error_size))
    return 0;
  char err[256];
  h3_st_store *st = h3_st_store_open(weight_directory, err, sizeof(err));
  if (!st) {
    fail2(error, error_size, "open video VAE weights", err);
    return 0;
  }
  int ok = load_conv(st, &e->conv_in, "encoder.conv_in", 3, 128, 3, 1, 1, 1, 2, 1,
                     1, 1, 1, error, error_size);
  int prev = 128;
  for (int level = 0; ok && level < LEVELS; level++) {
    int channels = h3_video_vae_block_out_channels[level];
    int ss = h3_video_vae_spatial_downsample[level];
    int ts = h3_video_vae_temporal_downsample[level];
    for (int block = 0; ok && block < BLOCKS; block++) {
      int in_ch = block ? channels : prev;
      char prefix[192], name[224];
      snprintf(prefix, sizeof(prefix), "encoder.down.%d.block.%d", level, block);
      snprintf(name, sizeof(name), "%s.norm1", prefix);
      ok = load_norm(st, &e->levels[level].blocks[block].norm1, name, in_ch,
                     error, error_size);
      snprintf(name, sizeof(name), "%s.conv1", prefix);
      if (ok)
        ok = load_conv(st, &e->levels[level].blocks[block].conv1, name, in_ch,
                       channels, 3, 1, 1, 1, 2, 1, 1, 1, 1, error, error_size);
      snprintf(name, sizeof(name), "%s.norm2", prefix);
      if (ok)
        ok = load_norm(st, &e->levels[level].blocks[block].norm2, name, channels,
                       error, error_size);
      snprintf(name, sizeof(name), "%s.conv2", prefix);
      if (ok)
        ok = load_conv(st, &e->levels[level].blocks[block].conv2, name, channels,
                       channels, 3, 1, 1, 1, 2, 1, 1, 1, 1, error, error_size);
      if (ok && in_ch != channels) {
        snprintf(name, sizeof(name), "%s.nin_shortcut", prefix);
        ok = load_conv(st, &e->levels[level].blocks[block].shortcut, name, in_ch,
                       channels, 1, 1, 1, 1, 0, 0, 0, 0, 0, error, error_size);
        e->levels[level].blocks[block].has_shortcut = 1;
      }
    }
    if (ok && ss * ts > 1) {
      char name[192];
      snprintf(name, sizeof(name), "encoder.down.%d.downsample.conv", level);
      int spatial_tail = ss == 2 ? 1 : 0;
      ok = load_conv(st, &e->levels[level].downsample, name, channels, channels, 3,
                     ts, ss, ss, 2, 0, spatial_tail, 0, spatial_tail, error,
                     error_size);
      e->levels[level].has_downsample = 1;
    }
    prev = channels;
  }
  if (ok)
    ok = load_norm(st, &e->norm_out, "encoder.norm_out", 1024, error,
                   error_size) &&
         load_conv(st, &e->conv_out, "encoder.conv_out", 1024, MOMENT_CH, 3, 1, 1,
                   1, 2, 1, 1, 1, 1, error, error_size) &&
         load_conv(st, &e->quant, "quant_conv", MOMENT_CH, MOMENT_CH, 1, 1, 1, 1,
                   0, 0, 0, 0, 0, error, error_size);
  h3_st_store_free(st);
  if (!ok)
    free_encoder(e);
  return ok;
}

static int run_encoder(const venc *e, const float *pixels, int frames,
                       int height, int width, h3_video_latent_host *output,
                       char *error, size_t error_size) {
  if (output)
    memset(output, 0, sizeof(*output));
  int max_hw = getenv("H3_VAE_ALLOW_LARGE") ? 512 : H3_VIDEO_VAE_TILE_PIXELS;
  if (!e || !pixels || !output || frames < 1 ||
      frames > H3_VIDEO_VAE_CLIP_LENGTH || height < 32 || width < 32 ||
      height > max_hw || width > max_hw || height % SPATIAL || width % SPATIAL) {
    fail1(error, error_size, "invalid video VAE encode arguments");
    return 0;
  }
  size_t pix_n = ndhwc(frames, height, width, RGB);
  float *hidden = xmalloc_f(pix_n);
  int ok = hidden != NULL;
  if (!ok)
    fail1(error, error_size, "oom pixels");
  if (ok) {
    size_t dest = 0;
    for (int t = 0; t < frames; t++)
      for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
          for (int c = 0; c < RGB; c++) {
            size_t src = (((size_t)c * frames + t) * height + y) * width + x;
            hidden[dest++] =
                (pixels[src] - h3_dit_pixel_mean[c]) / h3_dit_pixel_std[c];
          }
  }
  int d = frames, h = height, w = width;
  if (ok)
    ok = run_conv(&hidden, &d, &h, &w, &e->conv_in, error, error_size);
  int prev = 128;
  for (int level = 0; ok && level < LEVELS; level++) {
    int channels = h3_video_vae_block_out_channels[level];
    for (int block = 0; ok && block < BLOCKS; block++) {
      int in_ch = block ? channels : prev;
      ok = run_block(&hidden, d, h, w, in_ch, channels,
                     &e->levels[level].blocks[block], error, error_size);
    }
    if (ok && e->levels[level].has_downsample)
      ok = run_conv(&hidden, &d, &h, &w, &e->levels[level].downsample, error,
                    error_size);
    prev = channels;
  }
  if (ok) {
    size_t hn = ndhwc(d, h, w, 1024);
    float *normed = xmalloc_f(hn);
    if (!normed) {
      fail1(error, error_size, "oom output norm");
      ok = 0;
    } else {
      ok = h3_video_vae_group_norm_silu_f32(normed, hidden, e->norm_out.weight,
                                            e->norm_out.bias, 1, d, h, w, 1024,
                                            GROUPS, 1e-6f) == 0;
      free(hidden);
      hidden = normed;
    }
  }
  if (ok)
    ok = run_conv(&hidden, &d, &h, &w, &e->conv_out, error, error_size);
  if (ok)
    ok = run_conv(&hidden, &d, &h, &w, &e->quant, error, error_size);
  if (ok) {
    size_t latent_n = (size_t)LATENT_CH * d * h * w;
    output->values = xmalloc_f(latent_n);
    if (!output->values) {
      fail1(error, error_size, "oom latent");
      ok = 0;
    } else {
      for (int c = 0; c < LATENT_CH; c++)
        for (int t = 0; t < d; t++)
          for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
              size_t source =
                  (((size_t)t * h + y) * w + x) * MOMENT_CH + (size_t)c;
              size_t dest = (((size_t)c * d + t) * h + y) * w + (size_t)x;
              output->values[dest] =
                  (hidden[source] - e->mean[c]) / e->deviation[c];
            }
      output->channels = LATENT_CH;
      output->time = d;
      output->height = h;
      output->width = w;
    }
  }
  free(hidden);
  if (!ok)
    h3_video_latent_host_free(output);
  return ok;
}

int h3_video_vae_encode_host(const char *weight_directory, const float *pixels,
                             int frames, int height, int width,
                             h3_video_latent_host *output, char *error,
                             size_t error_size) {
  if (error && error_size)
    error[0] = '\0';
  if (output)
    memset(output, 0, sizeof(*output));
  if (!weight_directory || !*weight_directory || !pixels || !output ||
      frames < 1 || frames > H3_VIDEO_VAE_CLIP_LENGTH || height < 32 ||
      width < 32 || height > 512 || width > 512 || height % SPATIAL ||
      width % SPATIAL) {
    fail1(error, error_size, "invalid video VAE encode arguments");
    return 0;
  }
  venc enc;
  if (!load_encoder(weight_directory, &enc, error, error_size))
    return 0;
  int tile_px = h3_video_vae_configured_tile_pixels(height, width);
  int ok;
  if (height <= tile_px && width <= tile_px) {
    ok = run_encoder(&enc, pixels, frames, height, width, output, error,
                     error_size);
    free_encoder(&enc);
    return ok;
  }

  h3_video_vae_tile_axis y_axis, x_axis;
  if (h3_video_vae_tile_axis_build(height, tile_px, &y_axis) != 0 ||
      h3_video_vae_tile_axis_build(width, tile_px, &x_axis) != 0) {
    h3_video_vae_tile_axis_free(&y_axis);
    h3_video_vae_tile_axis_free(&x_axis);
    free_encoder(&enc);
    fail1(error, error_size, "cannot build video VAE encode tile plan");
    return 0;
  }
  int tile_count = y_axis.count * x_axis.count;
  float **latents = (float **)calloc((size_t)tile_count, sizeof(float *));
  float *rgb_tile =
      (float *)calloc((size_t)RGB * (size_t)frames * (size_t)y_axis.length *
                          (size_t)x_axis.length,
                      sizeof(float));
  ok = latents && rgb_tile;
  int time = 0, tile_lh = 0, tile_lw = 0;
  if (!ok)
    fail1(error, error_size, "oom video VAE encode tiles");
  for (int ty = 0; ok && ty < y_axis.count; ty++)
    for (int tx = 0; ok && tx < x_axis.count; tx++) {
      if (h3_video_vae_extract_rgb_f32(pixels, frames, height, width,
                                       y_axis.starts[ty], x_axis.starts[tx],
                                       y_axis.length, x_axis.length,
                                       rgb_tile) != 0) {
        fail1(error, error_size, "cannot extract video VAE RGB tile");
        ok = 0;
        break;
      }
      h3_video_latent_host z;
      memset(&z, 0, sizeof(z));
      if (!run_encoder(&enc, rgb_tile, frames, y_axis.length, x_axis.length, &z,
                       error, error_size)) {
        ok = 0;
        break;
      }
      if (!time) {
        time = z.time;
        tile_lh = z.height;
        tile_lw = z.width;
      } else if (z.time != time || z.height != tile_lh || z.width != tile_lw ||
                 z.channels != LATENT_CH) {
        fail1(error, error_size, "video VAE encode tile shape mismatch");
        h3_video_latent_host_free(&z);
        ok = 0;
        break;
      }
      latents[ty * x_axis.count + tx] = z.values;
      z.values = NULL;
      h3_video_latent_host_free(&z);
    }
  if (ok) {
    int full_lh = height / SPATIAL;
    int full_lw = width / SPATIAL;
    output->values = (float *)calloc(
        (size_t)LATENT_CH * (size_t)time * (size_t)full_lh * (size_t)full_lw,
        sizeof(float));
    if (!output->values) {
      fail1(error, error_size, "oom tiled latent");
      ok = 0;
    } else if (h3_video_vae_stitch_latent_tiles_f32(latents, &y_axis, &x_axis,
                                                    LATENT_CH, time,
                                                    output->values) != 0) {
      fail1(error, error_size, "cannot stitch video VAE latent tiles");
      ok = 0;
    } else {
      output->channels = LATENT_CH;
      output->time = time;
      output->height = full_lh;
      output->width = full_lw;
    }
  }
  for (int i = 0; i < tile_count; i++)
    free(latents ? latents[i] : NULL);
  free(latents);
  free(rgb_tile);
  h3_video_vae_tile_axis_free(&y_axis);
  h3_video_vae_tile_axis_free(&x_axis);
  free_encoder(&enc);
  if (!ok)
    h3_video_latent_host_free(output);
  return ok;
}
