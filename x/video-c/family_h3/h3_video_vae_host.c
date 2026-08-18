#define _DARWIN_C_SOURCE 1
#include "h3_video_vae_host.h"

#include <dispatch/dispatch.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

const int h3_video_vae_block_out_channels[H3_VIDEO_VAE_LEVELS] = {
    128, 256, 256, 512, 512, 1024};
const int h3_video_vae_spatial_downsample[H3_VIDEO_VAE_LEVELS] = {2, 2, 2, 2,
                                                                 1, 1};
const int h3_video_vae_temporal_downsample[H3_VIDEO_VAE_LEVELS] = {1, 2, 2, 1,
                                                                  1, 1};

int h3_video_vae_spatial_ratio(void) {
  int r = 1;
  for (int i = 0; i < H3_VIDEO_VAE_LEVELS; i++)
    r *= h3_video_vae_spatial_downsample[i];
  return r;
}

int h3_video_vae_temporal_ratio(void) {
  int r = 1;
  for (int i = 0; i < H3_VIDEO_VAE_LEVELS; i++)
    r *= h3_video_vae_temporal_downsample[i];
  return r;
}

int h3_video_vae_decoder_dim(void) { return H3_VIDEO_VAE_DECODER_DIM; }

int h3_video_vae_tokens_chunk_size(void) {
  int t = h3_video_vae_temporal_ratio();
  return (H3_VIDEO_VAE_CLIP_LENGTH + t - 1) / t;
}

int h3_video_vae_latent_hw(int pixels) {
  int r = h3_video_vae_spatial_ratio();
  if (pixels < 1 || pixels % r)
    return -1;
  return pixels / r;
}

int h3_video_vae_first_chunk_frames(int latent_time) {
  if (latent_time == 2)
    return 5;
  if (latent_time == H3_VIDEO_VAE_CHUNK_LATENT_T)
    return H3_VIDEO_VAE_FIRST_CHUNK_FRAMES;
  return -1;
}

int h3_video_vae_output_frames(int latent_time) {
  if (latent_time == 2)
    return 5;
  if (latent_time >= H3_VIDEO_VAE_CHUNK_LATENT_T &&
      (latent_time - 2) % 5 == 0)
    return ((latent_time - 2) / 5) * 17 + 5;
  return -1;
}

int h3_video_vae_temporal_overlap_blend_f32(float *current, const float *overlap,
                                           int overlap_frames,
                                           size_t frame_elements) {
  if (!current || !overlap || overlap_frames < 1 || frame_elements < 1)
    return -1;
  for (int frame = 0; frame < overlap_frames; frame++) {
    float alpha = (float)frame / (float)overlap_frames;
    size_t base = (size_t)frame * frame_elements;
    for (size_t i = 0; i < frame_elements; i++)
      current[base + i] =
          overlap[base + i] * (1.0f - alpha) + current[base + i] * alpha;
  }
  return 0;
}

int h3_video_vae_reflect_coord(int coordinate, int length) {
  if (length < 1)
    return -1;
  if (coordinate < 0)
    return -coordinate;
  if (coordinate >= length)
    return 2 * length - coordinate - 2;
  return coordinate;
}

int h3_video_vae_pad_ndhwc_f32(float *dst, const float *src, int batch, int depth,
                               int height, int width, int channels,
                               int depth_front, int height_before,
                               int height_after, int width_before,
                               int width_after) {
  if (!dst || !src || batch < 1 || depth < 1 || height < 2 || width < 2 ||
      channels < 1)
    return -1;
  if (height_before >= height || height_after >= height ||
      width_before >= width || width_after >= width)
    return -1;
  int out_d = depth + depth_front;
  int out_h = height + height_before + height_after;
  int out_w = width + width_before + width_after;
  for (int b = 0; b < batch; b++) {
    for (int t = 0; t < out_d; t++) {
      for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
          float *o = dst + (((((size_t)b * out_d + t) * out_h + y) * out_w + x) *
                            (size_t)channels);
          if (t < depth_front) {
            for (int c = 0; c < channels; c++)
              o[c] = 0.f;
            continue;
          }
          int sy = h3_video_vae_reflect_coord(y - height_before, height);
          int sx = h3_video_vae_reflect_coord(x - width_before, width);
          int st = t - depth_front;
          const float *in =
              src + (((((size_t)b * depth + st) * height + sy) * width + sx) *
                     (size_t)channels);
          for (int c = 0; c < channels; c++)
            o[c] = in[c];
        }
      }
    }
  }
  return 0;
}

int h3_video_vae_conv3d_f32(float *dst, const float *src, const float *weight,
                            const float *bias, int batch, int depth, int height,
                            int width, int in_ch, int out_ch, int kd, int kh,
                            int kw, int stride_t, int stride_h, int stride_w) {
  if (!dst || !src || !weight || batch < 1 || depth < kd || height < kh ||
      width < kw || in_ch < 1 || out_ch < 1 || kd < 1 || kh < 1 || kw < 1 ||
      stride_t < 1 || stride_h < 1 || stride_w < 1)
    return -1;
  int od = (depth - kd) / stride_t + 1;
  int oh = (height - kh) / stride_h + 1;
  int ow = (width - kw) / stride_w + 1;
  for (int b = 0; b < batch; b++) {
    for (int t = 0; t < od; t++) {
      for (int y = 0; y < oh; y++) {
        for (int x = 0; x < ow; x++) {
          for (int oc = 0; oc < out_ch; oc++) {
            double sum = bias ? bias[oc] : 0.0;
            for (int kd_i = 0; kd_i < kd; kd_i++) {
              int it = t * stride_t + kd_i;
              for (int kh_i = 0; kh_i < kh; kh_i++) {
                int iy = y * stride_h + kh_i;
                for (int kw_i = 0; kw_i < kw; kw_i++) {
                  int ix = x * stride_w + kw_i;
                  const float *in =
                      src + (((((size_t)b * depth + it) * height + iy) * width +
                              ix) *
                             (size_t)in_ch);
                  for (int ic = 0; ic < in_ch; ic++)
                    sum += (double)in[ic] *
                           (double)weight[((((size_t)oc * in_ch + ic) * kd +
                                            kd_i) *
                                               kh +
                                           kh_i) *
                                              kw +
                                          kw_i];
                }
              }
            }
            dst[(((((size_t)b * od + t) * oh + y) * ow + x) * (size_t)out_ch) +
                (size_t)oc] = (float)sum;
          }
        }
      }
    }
  }
  return 0;
}

int h3_video_vae_group_norm_silu_f32(float *dst, const float *src,
                                     const float *weight, const float *bias,
                                     int batch, int depth, int height, int width,
                                     int channels, int groups, float eps) {
  if (!dst || !src || !weight || !bias || batch < 1 || depth < 1 || height < 1 ||
      width < 1 || channels < 1 || groups < 1 || channels % groups ||
      !(eps > 0.f))
    return -1;
  int cpg = channels / groups;
  int spatial = height * width;
  int elements = spatial * cpg;
  for (int b = 0; b < batch; b++) {
    for (int t = 0; t < depth; t++) {
      int plane = b * depth + t;
      for (int g = 0; g < groups; g++) {
        double mean = 0.0;
        for (int s = 0; s < spatial; s++) {
          const float *px =
              src + ((size_t)plane * spatial + s) * (size_t)channels +
              (size_t)g * cpg;
          for (int c = 0; c < cpg; c++)
            mean += px[c];
        }
        mean /= (double)elements;
        double var = 0.0;
        for (int s = 0; s < spatial; s++) {
          const float *px =
              src + ((size_t)plane * spatial + s) * (size_t)channels +
              (size_t)g * cpg;
          for (int c = 0; c < cpg; c++) {
            double d = px[c] - mean;
            var += d * d;
          }
        }
        float inv = 1.0f / sqrtf((float)(var / (double)elements) + eps);
        for (int s = 0; s < spatial; s++) {
          const float *px =
              src + ((size_t)plane * spatial + s) * (size_t)channels +
              (size_t)g * cpg;
          float *ox =
              dst + ((size_t)plane * spatial + s) * (size_t)channels +
              (size_t)g * cpg;
          for (int c = 0; c < cpg; c++) {
            int ch = g * cpg + c;
            float v = (px[c] - (float)mean) * inv * weight[ch] + bias[ch];
            ox[c] = v / (1.0f + expf(-v));
          }
        }
      }
    }
  }
  return 0;
}

int h3_video_vae_qkv_rope_f32(float *query, float *key, float *value,
                              const float *qkv, const float *rope_cos,
                              const float *rope_sin, int seq, int heads,
                              int head_dim, int rope_half, float eps) {
  if (!query || !key || !value || !qkv || !rope_cos || !rope_sin || seq < 1 ||
      heads < 1 || head_dim < 1 || rope_half < 1 || rope_half * 2 > head_dim)
    return -1;
  for (int row = 0; row < seq; row++) {
    for (int head = 0; head < heads; head++) {
      const float *base =
          qkv + ((size_t)row * heads + head) * (size_t)head_dim * 3;
      double qss = 0.0, kss = 0.0;
      for (int d = 0; d < head_dim; d++) {
        qss += (double)base[d] * base[d];
        kss += (double)base[head_dim + d] * base[head_dim + d];
      }
      float qi = 1.0f / sqrtf((float)(qss / (double)head_dim) + eps);
      float ki = 1.0f / sqrtf((float)(kss / (double)head_dim) + eps);
      float *q = query + ((size_t)row * heads + head) * (size_t)head_dim;
      float *k = key + ((size_t)row * heads + head) * (size_t)head_dim;
      float *v = value + ((size_t)row * heads + head) * (size_t)head_dim;
      for (int d = 0; d < head_dim; d++) {
        float q0 = base[d] * qi;
        float k0 = base[head_dim + d] * ki;
        if (d < rope_half) {
          float q1 = base[d + rope_half] * qi;
          float k1 = base[head_dim + d + rope_half] * ki;
          float c = rope_cos[(size_t)row * rope_half + d];
          float s = rope_sin[(size_t)row * rope_half + d];
          q0 = q0 * c - q1 * s;
          k0 = k0 * c - k1 * s;
        } else if (d < rope_half * 2) {
          int pair = d - rope_half;
          float q1 = base[pair] * qi;
          float k1 = base[head_dim + pair] * ki;
          float c = rope_cos[(size_t)row * rope_half + pair];
          float s = rope_sin[(size_t)row * rope_half + pair];
          q0 = q0 * c + q1 * s;
          k0 = k0 * c + k1 * s;
        }
        q[d] = q0;
        k[d] = k0;
        v[d] = base[head_dim * 2 + d];
      }
    }
  }
  return 0;
}

int h3_video_vae_sdpa_f32(float *out, const float *q, const float *k,
                          const float *v, int seq, int heads, int head_dim,
                          float scale) {
  if (!out || !q || !k || !v || seq < 1 || heads < 1 || head_dim < 1)
    return -1;
  long ncore = sysconf(_SC_NPROCESSORS_ONLN);
  if (ncore < 2 || seq < 64 || heads < 4) {
    float *scores = (float *)malloc((size_t)seq * sizeof(float));
    if (!scores)
      return -1;
    for (int h = 0; h < heads; h++) {
      for (int row = 0; row < seq; row++) {
        const float *qr = q + ((size_t)row * heads + h) * (size_t)head_dim;
        float m = -1e30f;
        for (int col = 0; col < seq; col++) {
          const float *kr = k + ((size_t)col * heads + h) * (size_t)head_dim;
          double dot = 0.0;
          for (int d = 0; d < head_dim; d++)
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
        float *orow = out + ((size_t)row * heads + h) * (size_t)head_dim;
        for (int d = 0; d < head_dim; d++)
          orow[d] = 0.f;
        for (int col = 0; col < seq; col++) {
          const float *vr = v + ((size_t)col * heads + h) * (size_t)head_dim;
          float w = scores[col] * inv;
          for (int d = 0; d < head_dim; d++)
            orow[d] += w * vr[d];
        }
      }
    }
    free(scores);
    return 0;
  }
  /* Large seq: parallelize over heads; same loop order per (head,row) → bit-exact. */
  float *scores = (float *)malloc((size_t)heads * (size_t)seq * sizeof(float));
  if (!scores)
    return -1;
  dispatch_apply((size_t)heads,
                 dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                 ^(size_t h) {
                   float *sh = scores + h * (size_t)seq;
                   for (int row = 0; row < seq; row++) {
                     const float *qr = q + ((size_t)row * heads + h) * (size_t)head_dim;
                     float m = -1e30f;
                     for (int col = 0; col < seq; col++) {
                       const float *kr = k + ((size_t)col * heads + h) * (size_t)head_dim;
                       double dot = 0.0;
                       for (int d = 0; d < head_dim; d++)
                         dot += (double)qr[d] * kr[d];
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
                     float *orow = out + ((size_t)row * heads + h) * (size_t)head_dim;
                     for (int d = 0; d < head_dim; d++)
                       orow[d] = 0.f;
                     for (int col = 0; col < seq; col++) {
                       const float *vr = v + ((size_t)col * heads + h) * (size_t)head_dim;
                       float w = sh[col] * inv;
                       for (int d = 0; d < head_dim; d++)
                         orow[d] += w * vr[d];
                     }
                   }
                 });
  free(scores);
  return 0;
}

int h3_video_vae_swiglu_f32(float *dst, const float *fused, int rows,
                            int width) {
  if (!dst || !fused || rows < 1 || width < 1)
    return -1;
  for (int r = 0; r < rows; r++) {
    const float *gate = fused + (size_t)r * (size_t)width * 2;
    const float *up = gate + width;
    float *o = dst + (size_t)r * (size_t)width;
    for (int i = 0; i < width; i++)
      o[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
  }
  return 0;
}

int h3_video_vae_scale_add_f32(float *dst, const float *residual,
                               const float *branch, const float *scale, int rows,
                               int dim) {
  if (!dst || !residual || !branch || !scale || rows < 1 || dim < 1)
    return -1;
  for (int r = 0; r < rows; r++) {
    const float *x = residual + (size_t)r * (size_t)dim;
    const float *y = branch + (size_t)r * (size_t)dim;
    float *o = dst + (size_t)r * (size_t)dim;
    for (int i = 0; i < dim; i++)
      o[i] = x[i] + y[i] * scale[i];
  }
  return 0;
}

void h3_video_vae_tile_axis_free(h3_video_vae_tile_axis *axis) {
  if (!axis)
    return;
  free(axis->starts);
  free(axis->overlaps);
  memset(axis, 0, sizeof(*axis));
}

int h3_video_vae_tile_overlap(int tile_pixels) {
  int overlap = tile_pixels / 4;
  if (overlap < 16)
    overlap = 16;
  return overlap - (overlap % 16);
}

int h3_video_vae_tile_count_for_extent(int extent, int tile_pixels) {
  if (extent <= tile_pixels)
    return 1;
  int overlap = h3_video_vae_tile_overlap(tile_pixels);
  int count = (extent + tile_pixels - 1) / tile_pixels;
  while (tile_pixels * count - overlap * (count - 1) < extent)
    count++;
  return count;
}

int h3_video_vae_configured_tile_pixels(int pixel_height, int pixel_width) {
  const char *value = getenv("H3_VAE_TILE_PIXELS");
  if (value && *value) {
    char *end = NULL;
    long pixels = strtol(value, &end, 10);
    if (end && !*end && pixels >= 32 && pixels <= 512 && pixels % 16 == 0)
      return (int)pixels;
  }
  int best = H3_VIDEO_VAE_TILE_PIXELS;
  uint64_t best_score = UINT64_MAX;
  /* Decode ViT refuses latent tiles > 16×16 (256 px) unless H3_VAE_ALLOW_LARGE.
   * Do not pick 320 px for 768² — that is 20×20 and trips the guard. */
  int max_px = H3_VIDEO_VAE_TILE_PIXELS;
  if (getenv("H3_VAE_ALLOW_LARGE"))
    max_px = 320;
  for (int pixels = H3_VIDEO_VAE_TILE_PIXELS; pixels <= max_px; pixels += 16) {
    uint64_t tiles =
        (uint64_t)h3_video_vae_tile_count_for_extent(pixel_height, pixels) *
        (uint64_t)h3_video_vae_tile_count_for_extent(pixel_width, pixels);
    uint64_t score = tiles * (uint64_t)pixels * (uint64_t)pixels;
    if (score < best_score) {
      best = pixels;
      best_score = score;
    }
  }
  return best;
}

int h3_video_vae_tile_axis_build(int extent, int tile_pixels,
                                 h3_video_vae_tile_axis *axis) {
  if (!axis)
    return -1;
  memset(axis, 0, sizeof(*axis));
  if (extent < 1 || extent % 16 || tile_pixels < 16 || tile_pixels % 16)
    return -1;
  if (extent <= tile_pixels) {
    axis->count = 1;
    axis->length = extent;
    axis->starts = (int *)calloc(1, sizeof(int));
    return axis->starts ? 0 : -1;
  }
  int overlap = h3_video_vae_tile_overlap(tile_pixels);
  int count = h3_video_vae_tile_count_for_extent(extent, tile_pixels);
  axis->starts = (int *)calloc((size_t)count, sizeof(int));
  axis->overlaps = (int *)malloc((size_t)(count - 1) * sizeof(int));
  if (!axis->starts || !axis->overlaps) {
    h3_video_vae_tile_axis_free(axis);
    return -1;
  }
  for (int i = 0; i < count - 1; i++)
    axis->overlaps[i] = overlap;
  int remaining = tile_pixels * count - overlap * (count - 1) - extent;
  for (int unit = 0; unit < remaining / 16; unit++)
    axis->overlaps[unit % (count - 1)] += 16;
  for (int i = 1; i < count; i++)
    axis->starts[i] =
        axis->starts[i - 1] + tile_pixels - axis->overlaps[i - 1];
  axis->count = count;
  axis->length = tile_pixels;
  return 0;
}

int h3_video_vae_extract_latent_f32(const float *latent, int full_t, int full_h,
                                    int full_w, int start_t, int start_y,
                                    int start_x, int tile_t, int tile_h,
                                    int tile_w, float *tile) {
  if (!latent || !tile || tile_t < 1 || tile_h < 1 || tile_w < 1 || start_t < 0 ||
      start_y < 0 || start_x < 0 || start_t + tile_t > full_t ||
      start_y + tile_h > full_h || start_x + tile_w > full_w)
    return -1;
  for (int c = 0; c < H3_VIDEO_VAE_LATENT_CHANNELS; c++)
    for (int t = 0; t < tile_t; t++)
      for (int y = 0; y < tile_h; y++) {
        size_t source = (((size_t)c * (size_t)full_t + (size_t)(start_t + t)) *
                             (size_t)full_h +
                         (size_t)(start_y + y)) *
                            (size_t)full_w +
                        (size_t)start_x;
        size_t dest = (((size_t)c * (size_t)tile_t + (size_t)t) * (size_t)tile_h +
                       (size_t)y) *
                      (size_t)tile_w;
        memcpy(tile + dest, latent + source, (size_t)tile_w * sizeof(float));
      }
  return 0;
}

int h3_video_vae_extract_rgb_f32(const float *pixels, int frames, int full_h,
                                 int full_w, int start_y, int start_x,
                                 int tile_h, int tile_w, float *tile) {
  if (!pixels || !tile || frames < 1 || tile_h < 1 || tile_w < 1 || start_y < 0 ||
      start_x < 0 || start_y + tile_h > full_h || start_x + tile_w > full_w)
    return -1;
  for (int c = 0; c < H3_VIDEO_VAE_IN_CHANNELS; c++)
    for (int t = 0; t < frames; t++)
      for (int y = 0; y < tile_h; y++) {
        size_t source =
            (((size_t)c * (size_t)frames + (size_t)t) * (size_t)full_h +
             (size_t)(start_y + y)) *
                (size_t)full_w +
            (size_t)start_x;
        size_t dest = (((size_t)c * (size_t)frames + (size_t)t) * (size_t)tile_h +
                       (size_t)y) *
                      (size_t)tile_w;
        memcpy(tile + dest, pixels + source, (size_t)tile_w * sizeof(float));
      }
  return 0;
}

int h3_video_vae_stitch_tiles_f32(float **tiles, const h3_video_vae_tile_axis *y,
                                  const h3_video_vae_tile_axis *x, int frames,
                                  float *rgb) {
  if (!tiles || !y || !x || !rgb || frames < 1 || y->count < 1 || x->count < 1 ||
      !y->starts || !x->starts)
    return -1;
  int full_h = y->starts[y->count - 1] + y->length;
  int full_w = x->starts[x->count - 1] + x->length;
  for (int ty = 0; ty < y->count; ty++)
    for (int tx = 0; tx < x->count; tx++) {
      int index = ty * x->count + tx;
      const float *current = tiles[index];
      if (!current)
        return -1;
      const float *above = ty ? tiles[index - x->count] : NULL;
      const float *left = tx ? tiles[index - 1] : NULL;
      int overlap_y = ty ? y->overlaps[ty - 1] : 0;
      int overlap_x = tx ? x->overlaps[tx - 1] : 0;
      int keep_h = y->length - (ty + 1 < y->count ? y->overlaps[ty] : 0);
      int keep_w = x->length - (tx + 1 < x->count ? x->overlaps[tx] : 0);
      for (int frame = 0; frame < frames; frame++)
        for (int row = 0; row < keep_h; row++)
          for (int col = 0; col < keep_w; col++)
            for (int ch = 0; ch < 3; ch++) {
              size_t local =
                  (((size_t)frame * (size_t)y->length + (size_t)row) *
                       (size_t)x->length +
                   (size_t)col) *
                      3 +
                  (size_t)ch;
              float value = current[local];
              if (above && row < overlap_y) {
                size_t top =
                    (((size_t)frame * (size_t)y->length +
                      (size_t)(y->length - overlap_y + row)) *
                         (size_t)x->length +
                     (size_t)col) *
                        3 +
                    (size_t)ch;
                float alpha = (float)row / (float)overlap_y;
                value = above[top] * (1.0f - alpha) + value * alpha;
              }
              if (left && col < overlap_x) {
                size_t prior =
                    (((size_t)frame * (size_t)y->length + (size_t)row) *
                         (size_t)x->length +
                     (size_t)(x->length - overlap_x + col)) *
                        3 +
                    (size_t)ch;
                float alpha = (float)col / (float)overlap_x;
                value = left[prior] * (1.0f - alpha) + value * alpha;
              }
              size_t dest = (((size_t)frame * (size_t)full_h +
                              (size_t)(y->starts[ty] + row)) *
                                 (size_t)full_w +
                             (size_t)(x->starts[tx] + col)) *
                                3 +
                            (size_t)ch;
              rgb[dest] = value;
            }
    }
  return 0;
}

int h3_video_vae_stitch_latent_tiles_f32(float **tiles,
                                         const h3_video_vae_tile_axis *y,
                                         const h3_video_vae_tile_axis *x,
                                         int channels, int time,
                                         float *latent) {
  if (!tiles || !y || !x || !latent || channels < 1 || time < 1 ||
      y->count < 1 || x->count < 1 || !y->starts || !x->starts)
    return -1;
  int ratio = h3_video_vae_spatial_ratio();
  if (y->length % ratio || x->length % ratio)
    return -1;
  int tile_h = y->length / ratio;
  int tile_w = x->length / ratio;
  int full_h = (y->starts[y->count - 1] + y->length) / ratio;
  int full_w = (x->starts[x->count - 1] + x->length) / ratio;
  for (int ty = 0; ty < y->count; ty++)
    for (int tx = 0; tx < x->count; tx++) {
      int index = ty * x->count + tx;
      const float *current = tiles[index];
      if (!current)
        return -1;
      const float *above = ty ? tiles[index - x->count] : NULL;
      const float *left = tx ? tiles[index - 1] : NULL;
      int overlap_y = ty ? y->overlaps[ty - 1] / ratio : 0;
      int overlap_x = tx ? x->overlaps[tx - 1] / ratio : 0;
      int keep_h = tile_h - (ty + 1 < y->count ? y->overlaps[ty] / ratio : 0);
      int keep_w = tile_w - (tx + 1 < x->count ? x->overlaps[tx] / ratio : 0);
      int start_y = y->starts[ty] / ratio;
      int start_x = x->starts[tx] / ratio;
      for (int c = 0; c < channels; c++)
        for (int t = 0; t < time; t++)
          for (int row = 0; row < keep_h; row++)
            for (int col = 0; col < keep_w; col++) {
              size_t local =
                  (((size_t)c * (size_t)time + (size_t)t) * (size_t)tile_h +
                   (size_t)row) *
                      (size_t)tile_w +
                  (size_t)col;
              float value = current[local];
              if (above && row < overlap_y) {
                size_t top =
                    (((size_t)c * (size_t)time + (size_t)t) * (size_t)tile_h +
                     (size_t)(tile_h - overlap_y + row)) *
                        (size_t)tile_w +
                    (size_t)col;
                float alpha = (float)row / (float)overlap_y;
                value = above[top] * (1.0f - alpha) + value * alpha;
              }
              if (left && col < overlap_x) {
                size_t prior =
                    (((size_t)c * (size_t)time + (size_t)t) * (size_t)tile_h +
                     (size_t)row) *
                        (size_t)tile_w +
                    (size_t)(tile_w - overlap_x + col);
                float alpha = (float)col / (float)overlap_x;
                value = left[prior] * (1.0f - alpha) + value * alpha;
              }
              size_t dest =
                  (((size_t)c * (size_t)time + (size_t)t) * (size_t)full_h +
                   (size_t)(start_y + row)) *
                      (size_t)full_w +
                  (size_t)(start_x + col);
              latent[dest] = value;
            }
    }
  return 0;
}
