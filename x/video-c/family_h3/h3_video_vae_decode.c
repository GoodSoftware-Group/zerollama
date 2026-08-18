#include "h3_video_vae_decode.h"

#include "h3_audio_vae_host.h"
#include "h3_dit_host.h"
#include "h3_st_store.h"
#include "h3_video_vae_host.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
  C = H3_VIDEO_VAE_LATENT_CHANNELS,
  HIDDEN = H3_VIDEO_VAE_DECODER_DIM,
  HEADS = H3_VIDEO_VAE_DECODER_HEADS,
  HEAD_DIM = H3_VIDEO_VAE_DECODER_HEAD_DIM,
  LAYERS = H3_VIDEO_VAE_DECODER_LAYERS,
  REG = H3_VIDEO_VAE_DECODER_REGISTERS,
  SUFFIX = H3_VIDEO_VAE_DECODER_SUFFIX,
  CHUNK_T = H3_VIDEO_VAE_CHUNK_LATENT_T,
  ROPE_HALF = H3_VIDEO_VAE_ROPE_HALF,
  FFN = H3_VIDEO_VAE_FFN,
  PATCH = H3_VIDEO_VAE_OUTPUT_PATCH,
  SPATIAL = 16,
  FRAME_OFF = H3_VIDEO_VAE_FRAME_OFFSET
};

static void fail1(char *error, size_t n, const char *msg) {
  if (error && n)
    snprintf(error, n, "%s", msg);
}

static struct timespec vae_t0;
static int vae_tile_i, vae_tile_n, vae_chunk_i, vae_chunk_n;

static int vae_quiet(void) {
  const char *q = getenv("H3_VAE_QUIET");
  return q && q[0] && q[0] != '0';
}

static double vae_elapsed(void) {
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
    return 0.0;
  return (double)(now.tv_sec - vae_t0.tv_sec) +
         (double)(now.tv_nsec - vae_t0.tv_nsec) * 1e-9;
}

static void vae_log(const char *fmt, ...) {
  if (vae_quiet())
    return;
  fprintf(stderr, "video-c: vae [%.1fs] ", vae_elapsed());
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  fflush(stderr);
}

static float *xmalloc_f(size_t n) {
  return n ? (float *)malloc(n * sizeof(float)) : NULL;
}

void h3_video_frames_host_free(h3_video_frames_host *frames) {
  if (!frames)
    return;
  free(frames->rgb);
  memset(frames, 0, sizeof(*frames));
}

int h3_video_vae_repeat_last_time(const float *src, int channels, int src_t,
                                  int height, int width, int dst_t, float *dst) {
  if (!src || !dst || channels < 1 || src_t < 1 || dst_t < src_t || height < 1 ||
      width < 1)
    return -1;
  size_t hw = (size_t)height * (size_t)width;
  for (int c = 0; c < channels; c++) {
    for (int t = 0; t < dst_t; t++) {
      int st = t < src_t ? t : src_t - 1;
      memcpy(dst + ((size_t)c * (size_t)dst_t + (size_t)t) * hw,
             src + ((size_t)c * (size_t)src_t + (size_t)st) * hw,
             hw * sizeof(float));
    }
  }
  return 0;
}

int h3_video_frames_write_ppm(const h3_video_frames_host *frames, int index,
                              const char *path, char *error, size_t error_size) {
  if (!frames || !frames->rgb || !path || index < 0 || index >= frames->frames) {
    fail1(error, error_size, "invalid PPM frame");
    return 0;
  }
  FILE *fp = fopen(path, "wb");
  if (!fp) {
    fail1(error, error_size, "cannot open PPM path");
    return 0;
  }
  int h = frames->height, w = frames->width;
  if (fprintf(fp, "P6\n%d %d\n255\n", w, h) < 0) {
    fclose(fp);
    fail1(error, error_size, "cannot write PPM header");
    return 0;
  }
  const float *src =
      frames->rgb + (size_t)index * (size_t)h * (size_t)w * 3;
  for (size_t i = 0; i < (size_t)h * (size_t)w * 3; i++) {
    float v = src[i];
    if (v < 0.f)
      v = 0.f;
    if (v > 1.f)
      v = 1.f;
    unsigned char b = (unsigned char)(v * 255.f + 0.5f);
    if (fwrite(&b, 1, 1, fp) != 1) {
      fclose(fp);
      fail1(error, error_size, "cannot write PPM pixels");
      return 0;
    }
  }
  fclose(fp);
  return 1;
}

static int ppm_skip(FILE *fp) {
  int c;
  do {
    c = fgetc(fp);
    if (c == '#') {
      while (c != EOF && c != '\n')
        c = fgetc(fp);
    }
  } while (c == ' ' || c == '\t' || c == '\n' || c == '\r');
  if (c == EOF)
    return EOF;
  ungetc(c, fp);
  return 0;
}

int h3_video_frames_read_ppm(const char *path, h3_video_frames_host *frames,
                             char *error, size_t error_size) {
  if (error && error_size)
    error[0] = '\0';
  if (frames)
    memset(frames, 0, sizeof(*frames));
  if (!path || !frames) {
    fail1(error, error_size, "invalid PPM path");
    return 0;
  }
  FILE *fp = fopen(path, "rb");
  if (!fp) {
    fail1(error, error_size, "cannot open PPM path");
    return 0;
  }
  char magic[3];
  if (fread(magic, 1, 2, fp) != 2 || magic[0] != 'P' || magic[1] != '6') {
    fclose(fp);
    fail1(error, error_size, "PPM must be binary P6");
    return 0;
  }
  int width = 0, height = 0, maxv = 0;
  if (ppm_skip(fp) != 0 || fscanf(fp, "%d", &width) != 1 || ppm_skip(fp) != 0 ||
      fscanf(fp, "%d", &height) != 1 || ppm_skip(fp) != 0 ||
      fscanf(fp, "%d", &maxv) != 1 || maxv != 255 || width < 1 || height < 1) {
    fclose(fp);
    fail1(error, error_size, "malformed PPM header");
    return 0;
  }
  int c = fgetc(fp);
  if (c != '\n' && c != ' ' && c != '\t' && c != '\r') {
    fclose(fp);
    fail1(error, error_size, "malformed PPM header whitespace");
    return 0;
  }
  size_t n = (size_t)width * (size_t)height * 3;
  unsigned char *raw = (unsigned char *)malloc(n);
  float *rgb = (float *)malloc(n * sizeof(float));
  if (!raw || !rgb || fread(raw, 1, n, fp) != n) {
    free(raw);
    free(rgb);
    fclose(fp);
    fail1(error, error_size, "cannot read PPM pixels");
    return 0;
  }
  fclose(fp);
  for (size_t i = 0; i < n; i++)
    rgb[i] = (float)raw[i] / 255.f;
  free(raw);
  frames->frames = 1;
  frames->width = width;
  frames->height = height;
  frames->rgb = rgb;
  return 1;
}

int h3_video_frames_to_cthw(const h3_video_frames_host *frames, int time,
                            float *cthw) {
  if (!frames || !frames->rgb || !cthw || time < 1 || frames->frames < 1)
    return -1;
  int h = frames->height, w = frames->width;
  for (int t = 0; t < time; t++) {
    int src_f = t < frames->frames ? t : 0;
    const float *src =
        frames->rgb + (size_t)src_f * (size_t)h * (size_t)w * 3;
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
        for (int c = 0; c < 3; c++) {
          size_t dest =
              (((size_t)c * (size_t)time + (size_t)t) * (size_t)h + (size_t)y) *
                  (size_t)w +
              (size_t)x;
          cthw[dest] = src[((size_t)y * (size_t)w + (size_t)x) * 3 + (size_t)c];
        }
  }
  return 0;
}

static int parse_float_array(const char *json, const char *key, float *values,
                             size_t count, char *error, size_t error_size) {
  char pattern[64];
  snprintf(pattern, sizeof(pattern), "\"%s\"", key);
  const char *cursor = strstr(json, pattern);
  if (!cursor || !(cursor = strchr(cursor + strlen(pattern), ':')) ||
      !(cursor = strchr(cursor, '['))) {
    fail1(error, error_size, "video VAE config is missing a key");
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
      fail1(error, error_size, "video VAE config has malformed array");
      return 0;
    }
    values[i] = value;
    cursor = end;
    while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
           *cursor == '\t')
      cursor++;
    if (i + 1 < count) {
      if (*cursor++ != ',') {
        fail1(error, error_size, "video VAE config has short array");
        return 0;
      }
    } else if (*cursor != ']') {
      fail1(error, error_size, "video VAE config has long array");
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
    fail1(error, error_size, "cannot open video VAE config.json");
    return 0;
  }
  if (fseek(file, 0, SEEK_END)) {
    fclose(file);
    fail1(error, error_size, "cannot seek video VAE config");
    return 0;
  }
  long end = ftell(file);
  if (end < 1 || end > 1024 * 1024 || fseek(file, 0, SEEK_SET)) {
    fclose(file);
    fail1(error, error_size, "invalid video VAE config size");
    return 0;
  }
  char *json = (char *)malloc((size_t)end + 1);
  if (!json || fread(json, 1, (size_t)end, file) != (size_t)end) {
    free(json);
    fclose(file);
    fail1(error, error_size, "cannot read video VAE config");
    return 0;
  }
  json[end] = '\0';
  fclose(file);
  int ok = parse_float_array(json, "latents_mean", mean, C, error, error_size) &&
           parse_float_array(json, "latents_std", deviation, C, error,
                             error_size);
  free(json);
  if (ok) {
    for (int i = 0; i < C; i++) {
      if (deviation[i] <= 0.f) {
        fail1(error, error_size, "video VAE latents_std is invalid");
        return 0;
      }
    }
  }
  return ok;
}

static int load_vec(const h3_st_store *st, const char *name, float **out,
                    size_t n, char *error, size_t error_size) {
  const float *cached = h3_st_store_get_f32(st, name, NULL, error, error_size);
  if (cached) {
    *out = (float *)cached;
    return 1;
  }
  float *buf = xmalloc_f(n);
  if (!buf) {
    fail1(error, error_size, "oom loading tensor");
    return 0;
  }
  if (h3_st_store_load_f32(st, name, buf, n, error, error_size) != 0) {
    free(buf);
    return 0;
  }
  *out = buf;
  return 1;
}

/* Free only buffers load_vec allocated; store-owned cache pointers are freed by
 * h3_st_store_free. */
static void wfree(const h3_st_store *st, void *p) {
  if (p && !h3_st_store_owns(st, p))
    free(p);
}

static int prepare_input(float *rows, const float *input, const float *mean,
                         const float *deviation, int latent_t, int lh, int lw) {
  size_t row = 0;
  for (int t = 0; t < CHUNK_T; t++) {
    int source_t = t < latent_t ? t : latent_t - 1;
    for (int h = 0; h < lh; h++)
      for (int w = 0; w < lw; w++)
        for (int c = 0; c < C; c++) {
          size_t source = (((size_t)c * (size_t)latent_t + (size_t)source_t) *
                               (size_t)lh +
                           (size_t)h) *
                              (size_t)lw +
                          (size_t)w;
          rows[row++] = input[source] * deviation[c] + mean[c];
        }
  }
  return 1;
}

static int prepare_rope(float *cosines, float *sines, int lh, int lw, int seq) {
  int row = 0;
  for (int t = 0; t < CHUNK_T; t++)
    for (int h = 0; h < lh; h++)
      for (int w = 0; w < lw; w++, row++) {
        float axes[] = {
            2.0f * (((float)t + 0.5f) / (float)CHUNK_T) - 1.0f,
            2.0f * (((float)h + 0.5f) / (float)lh) - 1.0f,
            2.0f * (((float)w + 0.5f) / (float)lw) - 1.0f};
        for (int axis = 0; axis < 3; axis++)
          for (int frequency = 0; frequency < 8; frequency++) {
            float inverse =
                1.0f / powf(H3_VIDEO_VAE_DECODER_ROPE_THETA,
                            (float)frequency * 0.125f);
            float angle = 2.0f * 3.14159265358979323846f * axes[axis] * inverse;
            size_t offset = (size_t)row * ROPE_HALF + (size_t)axis * 8 +
                            (size_t)frequency;
            cosines[offset] = cosf(angle);
            sines[offset] = sinf(angle);
          }
      }
  while (row < seq) {
    for (int i = 0; i < ROPE_HALF; i++) {
      cosines[(size_t)row * ROPE_HALF + i] = 1.0f;
      sines[(size_t)row * ROPE_HALF + i] = 0.0f;
    }
    row++;
  }
  return 1;
}

static int unpack_frames(h3_video_frames_host *output, const float *projected,
                         int lh, int lw, int output_frames) {
  int pixel_h = lh * SPATIAL;
  int pixel_w = lw * SPATIAL;
  size_t rgb_n = (size_t)output_frames * (size_t)pixel_h * (size_t)pixel_w * 3;
  float *rgb = xmalloc_f(rgb_n);
  if (!rgb)
    return 0;
  for (int frame = 0; frame < output_frames; frame++) {
    int decoded_t = frame + FRAME_OFF;
    if (output_frames == H3_VIDEO_VAE_FIRST_CHUNK_FRAMES && frame >= 17)
      decoded_t += 3;
    int patch_t = decoded_t / 4;
    int within_t = decoded_t % 4;
    for (int y = 0; y < pixel_h; y++) {
      int patch_y = y / SPATIAL, within_y = y % SPATIAL;
      for (int x = 0; x < pixel_w; x++) {
        int patch_x = x / SPATIAL, within_x = x % SPATIAL;
        size_t patch = ((size_t)patch_t * (size_t)lh + (size_t)patch_y) *
                           (size_t)lw +
                       (size_t)patch_x;
        for (int channel = 0; channel < 3; channel++) {
          size_t component =
              ((((size_t)channel * 4 + (size_t)within_t) * SPATIAL +
                (size_t)within_y) *
                   SPATIAL +
               (size_t)within_x);
          float value = projected[patch * PATCH + component] *
                            h3_dit_pixel_std[channel] +
                        h3_dit_pixel_mean[channel];
          float v = value;
          if (v < 0.f)
            v = 0.f;
          if (v > 1.f)
            v = 1.f;
          size_t dest = (((size_t)frame * (size_t)pixel_h + (size_t)y) *
                             (size_t)pixel_w +
                         (size_t)x) *
                            3 +
                        (size_t)channel;
          rgb[dest] = v;
        }
      }
    }
  }
  output->frames = output_frames;
  output->height = pixel_h;
  output->width = pixel_w;
  output->rgb = rgb;
  return 1;
}

static int run_block(const h3_st_store *st, int index, float *hidden, float *norm,
                     float *qkv, float *query, float *key, float *value,
                     float *heads, float *branch, float *ff1, float *activated,
                     const float *rope_cos, const float *rope_sin, int seq,
                     char *error, size_t error_size) {
  char name[160];
  float *norm1 = NULL, *qkv_w = NULL, *qkv_b = NULL, *out_w = NULL, *out_b = NULL,
        *scale1 = NULL, *norm2 = NULL, *w1 = NULL, *w1_b = NULL, *w2 = NULL,
        *w2_b = NULL, *scale2 = NULL;
  snprintf(name, sizeof(name), "decoder.transformer_blocks.%d.norm1.weight",
           index);
  int ok = load_vec(st, name, &norm1, HIDDEN, error, error_size);
  snprintf(name, sizeof(name), "decoder.transformer_blocks.%d.attn.to_qkv.weight",
           index);
  ok = ok && load_vec(st, name, &qkv_w, (size_t)HIDDEN * 3 * HIDDEN, error,
                      error_size);
  snprintf(name, sizeof(name), "decoder.transformer_blocks.%d.attn.to_qkv.bias",
           index);
  ok = ok && load_vec(st, name, &qkv_b, (size_t)HIDDEN * 3, error, error_size);
  snprintf(name, sizeof(name), "decoder.transformer_blocks.%d.attn.to_out.weight",
           index);
  ok = ok && load_vec(st, name, &out_w, (size_t)HIDDEN * HIDDEN, error,
                      error_size);
  snprintf(name, sizeof(name), "decoder.transformer_blocks.%d.attn.to_out.bias",
           index);
  ok = ok && load_vec(st, name, &out_b, HIDDEN, error, error_size);
  snprintf(name, sizeof(name), "decoder.transformer_blocks.%d.scale1", index);
  ok = ok && load_vec(st, name, &scale1, HIDDEN, error, error_size);
  snprintf(name, sizeof(name), "decoder.transformer_blocks.%d.norm2.weight",
           index);
  ok = ok && load_vec(st, name, &norm2, HIDDEN, error, error_size);
  snprintf(name, sizeof(name), "decoder.transformer_blocks.%d.ff.w1.weight",
           index);
  ok = ok && load_vec(st, name, &w1, (size_t)FFN * 2 * HIDDEN, error, error_size);
  snprintf(name, sizeof(name), "decoder.transformer_blocks.%d.ff.w1.bias", index);
  ok = ok && load_vec(st, name, &w1_b, (size_t)FFN * 2, error, error_size);
  snprintf(name, sizeof(name), "decoder.transformer_blocks.%d.ff.w2.weight",
           index);
  ok = ok && load_vec(st, name, &w2, (size_t)HIDDEN * FFN, error, error_size);
  snprintf(name, sizeof(name), "decoder.transformer_blocks.%d.ff.w2.bias", index);
  ok = ok && load_vec(st, name, &w2_b, HIDDEN, error, error_size);
  snprintf(name, sizeof(name), "decoder.transformer_blocks.%d.scale2", index);
  ok = ok && load_vec(st, name, &scale2, HIDDEN, error, error_size);
  if (!ok)
    goto done;
  if (h3_dit_rmsnorm(hidden, seq, HIDDEN, 1e-5f, norm1, norm) != 0 ||
      h3_dit_linear(norm, seq, HIDDEN, HIDDEN * 3, qkv_w, qkv_b, qkv) != 0 ||
      h3_video_vae_qkv_rope_f32(query, key, value, qkv, rope_cos, rope_sin, seq,
                               HEADS, HEAD_DIM, ROPE_HALF, 1e-5f) != 0 ||
      h3_video_vae_sdpa_f32(heads, query, key, value, seq, HEADS, HEAD_DIM,
                            1.0f / sqrtf((float)HEAD_DIM)) != 0 ||
      h3_dit_linear(heads, seq, HIDDEN, HIDDEN, out_w, out_b, branch) != 0 ||
      h3_video_vae_scale_add_f32(hidden, hidden, branch, scale1, seq, HIDDEN) !=
          0 ||
      h3_dit_rmsnorm(hidden, seq, HIDDEN, 1e-5f, norm2, norm) != 0 ||
      h3_dit_linear(norm, seq, HIDDEN, FFN * 2, w1, w1_b, ff1) != 0 ||
      h3_video_vae_swiglu_f32(activated, ff1, seq, FFN) != 0 ||
      h3_dit_linear(activated, seq, FFN, HIDDEN, w2, w2_b, branch) != 0 ||
      h3_video_vae_scale_add_f32(hidden, hidden, branch, scale2, seq, HIDDEN) !=
          0) {
    fail1(error, error_size, "video VAE transformer block failed");
    ok = 0;
  }
done:
  wfree(st, norm1);
  wfree(st, qkv_w);
  wfree(st, qkv_b);
  wfree(st, out_w);
  wfree(st, out_b);
  wfree(st, scale1);
  wfree(st, norm2);
  wfree(st, w1);
  wfree(st, w1_b);
  wfree(st, w2);
  wfree(st, w2_b);
  wfree(st, scale2);
  return ok;
}

static int decode_first_chunk(const char *weight_directory,
                              const float *normalized_latent, int latent_time,
                              int latent_height, int latent_width,
                              h3_video_frames_host *output, char *error,
                              size_t error_size);

static int decode_spatial_chunk(const char *weight_directory,
                                const float *normalized_latent, int full_t,
                                int full_h, int full_w, int start_t, int tile_t,
                                const h3_video_vae_tile_axis *y_axis,
                                const h3_video_vae_tile_axis *x_axis,
                                h3_video_frames_host *output, char *error,
                                size_t error_size) {
  if (output)
    memset(output, 0, sizeof(*output));
  int tile_lh = y_axis->length / SPATIAL;
  int tile_lw = x_axis->length / SPATIAL;
  if (tile_lh < 1 || tile_lw < 1 ||
      ((size_t)tile_lh * (size_t)tile_lw > 16 * 16 &&
       !getenv("H3_VAE_ALLOW_LARGE"))) {
    fail1(error, error_size,
          "host ViT tile > 16x16 latent needs H3_VAE_ALLOW_LARGE=1");
    return 0;
  }
  int tile_count = y_axis->count * x_axis->count;
  float **tiles = (float **)calloc((size_t)tile_count, sizeof(float *));
  float *latent_tile =
      xmalloc_f((size_t)C * (size_t)tile_t * (size_t)tile_lh * (size_t)tile_lw);
  if (!tiles || !latent_tile) {
    free(tiles);
    free(latent_tile);
    fail1(error, error_size, "oom allocating spatial VAE tiles");
    return 0;
  }
  int want_frames = h3_video_vae_first_chunk_frames(tile_t);
  int ok = 1;
  vae_tile_n = tile_count;
  vae_log("spatial %dx%d tiles (%dpx) latent %dx%dx%d → %d frames", y_axis->count,
          x_axis->count, y_axis->length, tile_t, tile_lh, tile_lw, want_frames);
  for (int ty = 0; ty < y_axis->count && ok; ty++) {
    for (int tx = 0; tx < x_axis->count && ok; tx++) {
      vae_tile_i = ty * x_axis->count + tx + 1;
      vae_log("tile %d/%d y=%d x=%d (vit %dL)", vae_tile_i, vae_tile_n, ty, tx,
              LAYERS);
      if (h3_video_vae_extract_latent_f32(
              normalized_latent, full_t, full_h, full_w, start_t,
              y_axis->starts[ty] / SPATIAL, x_axis->starts[tx] / SPATIAL, tile_t,
              tile_lh, tile_lw, latent_tile) != 0) {
        fail1(error, error_size, "cannot extract spatial video VAE tile");
        ok = 0;
        break;
      }
      h3_video_frames_host decoded;
      memset(&decoded, 0, sizeof(decoded));
      ok = decode_first_chunk(weight_directory, latent_tile, tile_t, tile_lh,
                              tile_lw, &decoded, error, error_size);
      if (!ok) {
        h3_video_frames_host_free(&decoded);
        break;
      }
      if (decoded.frames != want_frames || decoded.height != y_axis->length ||
          decoded.width != x_axis->length) {
        fail1(error, error_size, "spatial tile decode returned the wrong shape");
        h3_video_frames_host_free(&decoded);
        ok = 0;
        break;
      }
      tiles[ty * x_axis->count + tx] = decoded.rgb;
      decoded.rgb = NULL;
      vae_log("tile %d/%d done", vae_tile_i, vae_tile_n);
    }
  }
  int full_h_px = y_axis->starts[y_axis->count - 1] + y_axis->length;
  int full_w_px = x_axis->starts[x_axis->count - 1] + x_axis->length;
  float *rgb = NULL;
  if (ok) {
    vae_log("stitch %d tiles → %dx%d", tile_count, full_h_px, full_w_px);
    rgb = xmalloc_f((size_t)want_frames * (size_t)full_h_px * (size_t)full_w_px *
                    3);
    if (!rgb || h3_video_vae_stitch_tiles_f32(tiles, y_axis, x_axis, want_frames,
                                              rgb) != 0) {
      fail1(error, error_size, "cannot stitch video VAE tiles");
      ok = 0;
    }
  }
  for (int i = 0; i < tile_count; i++)
    free(tiles[i]);
  free(tiles);
  free(latent_tile);
  if (!ok) {
    free(rgb);
    return 0;
  }
  output->frames = want_frames;
  output->height = full_h_px;
  output->width = full_w_px;
  output->rgb = rgb;
  return 1;
}

static int decode_first_chunk(const char *weight_directory,
                              const float *normalized_latent, int latent_time,
                              int latent_height, int latent_width,
                              h3_video_frames_host *output, char *error,
                              size_t error_size) {
  if (output)
    memset(output, 0, sizeof(*output));
  if (error && error_size)
    error[0] = '\0';
  if (!weight_directory || !normalized_latent || !output ||
      (latent_time != 2 && latent_time != CHUNK_T) || latent_height < 1 ||
      latent_width < 1 || latent_height > H3_VIDEO_VAE_TILE_PIXELS / SPATIAL ||
      latent_width > H3_VIDEO_VAE_TILE_PIXELS / SPATIAL) {
    fail1(error, error_size,
          "video VAE decode wants T=2 or T=7 and latent H/W in [1,16] (no tiling)");
    return 0;
  }

  int patches = CHUNK_T * latent_height * latent_width;
  int seq = patches + SUFFIX;
  int output_frames = h3_video_vae_first_chunk_frames(latent_time);
  if (output_frames < 1) {
    fail1(error, error_size, "video VAE first-chunk frame count is invalid");
    return 0;
  }
  float mean[C], deviation[C];
  if (!load_latent_norm(weight_directory, mean, deviation, error, error_size))
    return 0;

  h3_st_store *st = h3_st_store_open(weight_directory, error, error_size);
  if (!st)
    return 0;
  h3_st_store_set_prof_tag(st, "h3_vvae_wload");

  float *post_w = NULL, *post_b = NULL, *embed_w = NULL, *embed_b = NULL,
        *registers = NULL, *norm_out_w = NULL, *norm_out_b = NULL, *proj_w = NULL,
        *proj_b = NULL;
  float *latent_rows = xmalloc_f((size_t)patches * C);
  float *post = xmalloc_f((size_t)patches * C);
  float *patch_hidden = xmalloc_f((size_t)patches * HIDDEN);
  float *hidden = xmalloc_f((size_t)seq * HIDDEN);
  float *norm = xmalloc_f((size_t)seq * HIDDEN);
  float *qkv = xmalloc_f((size_t)seq * HIDDEN * 3);
  float *query = xmalloc_f((size_t)seq * HIDDEN);
  float *key = xmalloc_f((size_t)seq * HIDDEN);
  float *value = xmalloc_f((size_t)seq * HIDDEN);
  float *heads = xmalloc_f((size_t)seq * HIDDEN);
  float *branch = xmalloc_f((size_t)seq * HIDDEN);
  float *ff1 = xmalloc_f((size_t)seq * FFN * 2);
  float *activated = xmalloc_f((size_t)seq * FFN);
  float *projected = xmalloc_f((size_t)seq * PATCH);
  float *rope_cos = xmalloc_f((size_t)seq * ROPE_HALF);
  float *rope_sin = xmalloc_f((size_t)seq * ROPE_HALF);
  int ok = latent_rows && post && patch_hidden && hidden && norm && qkv &&
           query && key && value && heads && branch && ff1 && activated &&
           projected && rope_cos && rope_sin;
  if (!ok)
    fail1(error, error_size, "oom allocating video VAE activations");

  ok = ok && load_vec(st, "post_quant_conv.weight", &post_w, (size_t)C * C,
                      error, error_size) &&
       load_vec(st, "post_quant_conv.bias", &post_b, C, error, error_size) &&
       load_vec(st, "decoder.x_embedder.weight", &embed_w, (size_t)HIDDEN * C,
                error, error_size) &&
       load_vec(st, "decoder.x_embedder.bias", &embed_b, HIDDEN, error,
                error_size) &&
       load_vec(st, "decoder.register_tokens", &registers, (size_t)REG * HIDDEN,
                error, error_size);

  if (ok) {
    prepare_input(latent_rows, normalized_latent, mean, deviation, latent_time,
                  latent_height, latent_width);
    prepare_rope(rope_cos, rope_sin, latent_height, latent_width, seq);
    if (h3_dit_linear(latent_rows, patches, C, C, post_w, post_b, post) != 0 ||
        h3_dit_linear(post, patches, C, HIDDEN, embed_w, embed_b, patch_hidden) !=
            0) {
      fail1(error, error_size, "video VAE input projection failed");
      ok = 0;
    }
  }
  if (ok) {
    memcpy(hidden, patch_hidden, (size_t)patches * HIDDEN * sizeof(float));
    memcpy(hidden + (size_t)patches * HIDDEN, registers,
           (size_t)REG * HIDDEN * sizeof(float));
    memset(hidden + (size_t)(patches + REG) * HIDDEN, 0, HIDDEN * sizeof(float));
  }
  wfree(st, post_w);
  post_w = NULL;
  wfree(st, post_b);
  post_b = NULL;
  wfree(st, embed_w);
  embed_w = NULL;
  wfree(st, embed_b);
  embed_b = NULL;
  wfree(st, registers);
  registers = NULL;

  vae_log("vit seq=%d patches=%d start", seq, patches);
  for (int layer = 0; layer < LAYERS && ok; layer++) {
    if (vae_tile_n > 0)
      vae_log("tile %d/%d vit %d/%d seq=%d", vae_tile_i, vae_tile_n, layer + 1,
              LAYERS, seq);
    else
      vae_log("vit %d/%d seq=%d", layer + 1, LAYERS, seq);
    ok = run_block(st, layer, hidden, norm, qkv, query, key, value, heads,
                   branch, ff1, activated, rope_cos, rope_sin, seq, error,
                   error_size);
  }

  ok = ok &&
       load_vec(st, "decoder.norm_out.weight", &norm_out_w, HIDDEN, error,
                error_size) &&
       load_vec(st, "decoder.norm_out.bias", &norm_out_b, HIDDEN, error,
                error_size) &&
       load_vec(st, "decoder.proj_out.weight", &proj_w, (size_t)PATCH * HIDDEN,
                error, error_size) &&
       load_vec(st, "decoder.proj_out.bias", &proj_b, PATCH, error, error_size);
  if (ok) {
    if (h3_audio_vae_layer_norm_f32(norm, hidden, norm_out_w, norm_out_b, seq,
                                    HIDDEN, 1e-5f) != 0 ||
        h3_dit_linear(norm, seq, HIDDEN, PATCH, proj_w, proj_b, projected) !=
            0) {
      fail1(error, error_size, "video VAE output projection failed");
      ok = 0;
    }
  }
  if (ok &&
      !unpack_frames(output, projected, latent_height, latent_width,
                     output_frames)) {
    fail1(error, error_size, "oom unpacking video VAE frames");
    ok = 0;
  }

  free(latent_rows);
  free(post);
  free(patch_hidden);
  free(hidden);
  free(norm);
  free(qkv);
  free(query);
  free(key);
  free(value);
  free(heads);
  free(branch);
  free(ff1);
  free(activated);
  free(projected);
  free(rope_cos);
  free(rope_sin);
  wfree(st, norm_out_w);
  wfree(st, norm_out_b);
  wfree(st, proj_w);
  wfree(st, proj_b);
  h3_st_store_free(st);
  if (!ok)
    h3_video_frames_host_free(output);
  return ok;
}

int h3_video_vae_decode_host(const char *weight_directory,
                             const float *normalized_latent, int latent_time,
                             int latent_height, int latent_width,
                             h3_video_frames_host *output, char *error,
                             size_t error_size) {
  if (output)
    memset(output, 0, sizeof(*output));
  if (error && error_size)
    error[0] = '\0';
  int out_frames = h3_video_vae_output_frames(latent_time);
  if (!weight_directory || !normalized_latent || !output || out_frames < 1 ||
      latent_height < 1 || latent_width < 1) {
    fail1(error, error_size,
          "video VAE decode wants T=2 or T>=7 with (T-2)%5==0");
    return 0;
  }
  int pixel_h = latent_height * SPATIAL;
  int pixel_w = latent_width * SPATIAL;
  int tile_pixels = h3_video_vae_configured_tile_pixels(pixel_h, pixel_w);
  if (!getenv("H3_VAE_ALLOW_LARGE") && tile_pixels > H3_VIDEO_VAE_TILE_PIXELS)
    tile_pixels = H3_VIDEO_VAE_TILE_PIXELS;
  h3_video_vae_tile_axis y_axis, x_axis;
  memset(&y_axis, 0, sizeof(y_axis));
  memset(&x_axis, 0, sizeof(x_axis));
  if (h3_video_vae_tile_axis_build(pixel_h, tile_pixels, &y_axis) != 0 ||
      h3_video_vae_tile_axis_build(pixel_w, tile_pixels, &x_axis) != 0) {
    h3_video_vae_tile_axis_free(&y_axis);
    h3_video_vae_tile_axis_free(&x_axis);
    fail1(error, error_size, "cannot build video VAE tile plan");
    return 0;
  }
  vae_t0 = (struct timespec){0, 0};
  clock_gettime(CLOCK_MONOTONIC, &vae_t0);
  vae_tile_i = vae_tile_n = vae_chunk_i = vae_chunk_n = 0;
  vae_log("decode latent %dx%dx%d → %dx%dx%d tile=%dpx grid %dx%d vit %dL",
          latent_time, latent_height, latent_width, out_frames, pixel_h, pixel_w,
          tile_pixels, y_axis.count, x_axis.count, LAYERS);
  if (latent_time == 2 && (y_axis.count > 1 || x_axis.count > 1)) {
    int ok = decode_spatial_chunk(weight_directory, normalized_latent, latent_time,
                                  latent_height, latent_width, 0, 2, &y_axis,
                                  &x_axis, output, error, error_size);
    vae_log(ok ? "decode done" : "decode failed");
    h3_video_vae_tile_axis_free(&y_axis);
    h3_video_vae_tile_axis_free(&x_axis);
    return ok;
  }
  if (latent_time == 2 ||
      (latent_time == CHUNK_T && y_axis.count == 1 && x_axis.count == 1)) {
    int ok = decode_first_chunk(weight_directory, normalized_latent, latent_time,
                                latent_height, latent_width, output, error,
                                error_size);
    vae_log(ok ? "decode done" : "decode failed");
    h3_video_vae_tile_axis_free(&y_axis);
    h3_video_vae_tile_axis_free(&x_axis);
    return ok;
  }

  int chunks = (latent_time - 2) / 5;
  vae_chunk_n = chunks;
  vae_log("temporal %d chunks (T=%d)", chunks, latent_time);
  size_t frame_n = (size_t)pixel_h * (size_t)pixel_w * 3;
  float *final_rgb = xmalloc_f((size_t)out_frames * frame_n);
  float *overlap = xmalloc_f(5 * frame_n);
  if (!final_rgb || !overlap) {
    free(final_rgb);
    free(overlap);
    h3_video_vae_tile_axis_free(&y_axis);
    h3_video_vae_tile_axis_free(&x_axis);
    fail1(error, error_size, "oom allocating chunked video VAE output");
    return 0;
  }
  int ok = 1;
  for (int chunk = 0; chunk < chunks && ok; chunk++) {
    vae_chunk_i = chunk + 1;
    vae_log("chunk %d/%d", vae_chunk_i, vae_chunk_n);
    h3_video_frames_host decoded;
    memset(&decoded, 0, sizeof(decoded));
    ok = decode_spatial_chunk(weight_directory, normalized_latent, latent_time,
                              latent_height, latent_width, chunk * 5, CHUNK_T,
                              &y_axis, &x_axis, &decoded, error, error_size);
    if (!ok) {
      h3_video_frames_host_free(&decoded);
      break;
    }
    if (decoded.frames != H3_VIDEO_VAE_FIRST_CHUNK_FRAMES ||
        decoded.height != pixel_h || decoded.width != pixel_w) {
      fail1(error, error_size, "chunk decode returned the wrong shape");
      h3_video_frames_host_free(&decoded);
      ok = 0;
      break;
    }
    if (chunk) {
      if (h3_video_vae_temporal_overlap_blend_f32(decoded.rgb, overlap, 5,
                                                  frame_n) != 0) {
        fail1(error, error_size, "temporal overlap blend failed");
        h3_video_frames_host_free(&decoded);
        ok = 0;
        break;
      }
    }
    memcpy(final_rgb + (size_t)chunk * 17 * frame_n, decoded.rgb,
           17 * frame_n * sizeof(float));
    memcpy(overlap, decoded.rgb + 17 * frame_n, 5 * frame_n * sizeof(float));
    h3_video_frames_host_free(&decoded);
  }
  if (ok) {
    memcpy(final_rgb + (size_t)chunks * 17 * frame_n, overlap,
           5 * frame_n * sizeof(float));
    output->frames = out_frames;
    output->height = pixel_h;
    output->width = pixel_w;
    output->rgb = final_rgb;
    final_rgb = NULL;
  }
  free(final_rgb);
  free(overlap);
  h3_video_vae_tile_axis_free(&y_axis);
  h3_video_vae_tile_axis_free(&x_axis);
  vae_log(ok ? "decode done" : "decode failed");
  if (!ok)
    h3_video_frames_host_free(output);
  return ok;
}
