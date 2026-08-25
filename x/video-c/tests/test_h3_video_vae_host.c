#include "h3_host.h"
#include "h3_video_vae_host.h"

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

static int test_kernels(void) {
  CHECK(h3_video_vae_reflect_coord(-1, 4) == 1);
  CHECK(h3_video_vae_reflect_coord(4, 4) == 2);
  CHECK(h3_video_vae_reflect_coord(0, 4) == 0);

  float src[] = {1.f, 2.f, 3.f, 4.f};
  float dst[3 * 4 * 4];
  CHECK(h3_video_vae_pad_ndhwc_f32(dst, src, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1) ==
        0);
  CHECK(close_f(dst[0], 0.f, 1e-6f));
  int t1 = 1 * 4 * 4;
  CHECK(close_f(dst[t1 + 0 * 4 + 0], 4.f, 1e-6f));

  float in[] = {1.f, 2.f};
  float w[] = {3.f, 4.f};
  float b[] = {0.5f};
  float o[1];
  CHECK(h3_video_vae_conv3d_f32(o, in, w, b, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1,
                                1) == 0);
  CHECK(close_f(o[0], 1.f * 3.f + 2.f * 4.f + 0.5f, 1e-5f));

  float x[] = {1.f, 3.f, 5.f, 7.f};
  float gw[] = {1.f, 1.f};
  float gb[] = {0.f, 0.f};
  float gn[4];
  CHECK(h3_video_vae_group_norm_silu_f32(gn, x, gw, gb, 1, 1, 2, 1, 2, 2,
                                         1e-5f) == 0);
  float m0 = 3.f;
  float v0 = ((1 - 3) * (1 - 3) + (5 - 3) * (5 - 3)) / 2.f;
  float inv0 = 1.f / sqrtf(v0 + 1e-5f);
  float n00 = (1.f - m0) * inv0;
  float s00 = n00 / (1.f + expf(-n00));
  CHECK(close_f(gn[0], s00, 1e-5f));

  float residual[] = {1.f, 2.f};
  float branch[] = {3.f, 4.f};
  float scale[] = {0.5f, 0.25f};
  float sa[2];
  CHECK(h3_video_vae_scale_add_f32(sa, residual, branch, scale, 1, 2) == 0);
  CHECK(close_f(sa[0], 1.f + 3.f * 0.5f, 1e-6f));
  CHECK(close_f(sa[1], 2.f + 4.f * 0.25f, 1e-6f));

  float fused[] = {0.f, 2.f, 3.f, 5.f};
  float sw[2];
  CHECK(h3_video_vae_swiglu_f32(sw, fused, 1, 2) == 0);
  CHECK(close_f(sw[0], (0.f / (1.f + expf(0.f))) * 3.f, 1e-6f));
  float silu2 = 2.f / (1.f + expf(-2.f));
  CHECK(close_f(sw[1], silu2 * 5.f, 1e-5f));
  return 0;
}

int main(void) {
  CHECK(h3_video_vae_spatial_ratio() == H3_VAE_SPATIAL_RATIO);
  CHECK(h3_video_vae_spatial_ratio() == 16);
  CHECK(h3_video_vae_temporal_ratio() == 4);
  CHECK(h3_video_vae_decoder_dim() == 2048);
  CHECK(h3_video_vae_tokens_chunk_size() == 5);
  CHECK(h3_video_vae_first_chunk_frames(2) == 5);
  CHECK(h3_video_vae_first_chunk_frames(7) == 22);
  CHECK(h3_video_vae_first_chunk_frames(1) == -1);
  CHECK(h3_video_vae_output_frames(2) == 5);
  CHECK(h3_video_vae_output_frames(7) == 22);
  CHECK(h3_video_vae_output_frames(12) == 39);
  CHECK(h3_video_vae_output_frames(8) == -1);
  {
    float cur[10];
    float prev[10];
    for (int i = 0; i < 10; i++) {
      cur[i] = 0.f;
      prev[i] = 1.f;
    }
    CHECK(h3_video_vae_temporal_overlap_blend_f32(cur, prev, 5, 2) == 0);
    CHECK(close_f(cur[0], 1.f, 1e-6f));
  CHECK(close_f(cur[8], 0.2f, 1e-5f));
  }
  {
    CHECK(h3_video_vae_tile_count_for_extent(32, 256) == 1);
    CHECK(h3_video_vae_tile_count_for_extent(256, 256) == 1);
    CHECK(h3_video_vae_tile_count_for_extent(512, 256) == 3);
    h3_video_vae_tile_axis axis;
    CHECK(h3_video_vae_tile_axis_build(32, 256, &axis) == 0);
    CHECK(axis.count == 1 && axis.length == 32 && axis.starts[0] == 0);
    h3_video_vae_tile_axis_free(&axis);
    CHECK(h3_video_vae_tile_axis_build(512, 256, &axis) == 0);
    CHECK(axis.count == 3 && axis.length == 256);
    CHECK(axis.starts[0] == 0);
    CHECK(axis.starts[1] == 256 - axis.overlaps[0]);
    CHECK(axis.starts[2] == axis.starts[1] + 256 - axis.overlaps[1]);
    CHECK(axis.starts[2] + 256 >= 512);
    h3_video_vae_tile_axis_free(&axis);
    CHECK(h3_video_vae_configured_tile_pixels(768, 768) == 256);
    CHECK(h3_video_vae_tile_axis_build(768, 256, &axis) == 0);
    CHECK(axis.length == 256 && axis.count >= 3);
    CHECK(axis.length / 16 <= 16);
    CHECK(axis.starts[axis.count - 1] + axis.length >= 768);
    h3_video_vae_tile_axis_free(&axis);
  }
  {
    float src[24 * 2 * 4 * 4];
    float tile[24 * 2 * 2 * 2];
    for (int i = 0; i < 24 * 2 * 4 * 4; i++)
      src[i] = (float)i;
    CHECK(h3_video_vae_extract_latent_f32(src, 2, 4, 4, 0, 1, 1, 2, 2, 2,
                                          tile) == 0);
    CHECK(close_f(tile[0], src[((0 * 2 + 0) * 4 + 1) * 4 + 1], 1e-6f));
  }
  {
    h3_video_vae_tile_axis y, x;
    memset(&y, 0, sizeof(y));
    memset(&x, 0, sizeof(x));
    y.count = 1;
    y.length = 2;
    y.starts = (int *)calloc(1, sizeof(int));
    x.count = 1;
    x.length = 2;
    x.starts = (int *)calloc(1, sizeof(int));
    CHECK(y.starts && x.starts);
    float tile[2 * 2 * 3];
    for (int i = 0; i < 12; i++)
      tile[i] = 0.25f;
    float *ts[1] = {tile};
    float rgb[12];
    memset(rgb, 0, sizeof(rgb));
    CHECK(h3_video_vae_stitch_tiles_f32(ts, &y, &x, 1, rgb) == 0);
    CHECK(close_f(rgb[0], 0.25f, 1e-6f));
    CHECK(close_f(rgb[11], 0.25f, 1e-6f));
    h3_video_vae_tile_axis_free(&y);
    h3_video_vae_tile_axis_free(&x);
  }
  {
    CHECK(h3_video_vae_tile_overlap(256) == 64);
    CHECK(h3_video_vae_tile_overlap(32) == 16);
    CHECK(h3_video_vae_tile_count_for_extent(48, 32) == 2);
    h3_video_vae_tile_axis y, x;
    CHECK(h3_video_vae_tile_axis_build(48, 32, &y) == 0);
    CHECK(h3_video_vae_tile_axis_build(32, 32, &x) == 0);
    CHECK(y.count == 2 && y.length == 32 && y.overlaps[0] == 16);
    CHECK(x.count == 1 && x.length == 32);
    float t0[1 * 1 * 2 * 2];
    float t1[1 * 1 * 2 * 2];
    for (int i = 0; i < 4; i++) {
      t0[i] = 1.f;
      t1[i] = 3.f;
    }
    float *ts[2] = {t0, t1};
    float z[1 * 1 * 3 * 2];
    memset(z, 0, sizeof(z));
    CHECK(h3_video_vae_stitch_latent_tiles_f32(ts, &y, &x, 1, 1, z) == 0);
    CHECK(close_f(z[0], 1.f, 1e-6f));
    CHECK(close_f(z[2], 1.f, 1e-5f));
    CHECK(close_f(z[4], 3.f, 1e-6f));
    h3_video_vae_tile_axis_free(&y);
    h3_video_vae_tile_axis_free(&x);
  }
  CHECK(h3_video_vae_latent_hw(513) == -1);
  CHECK(H3_VIDEO_VAE_CLIP_LENGTH == 17);
  CHECK(H3_VIDEO_VAE_TOKEN_DROP == 3);
  CHECK(H3_VIDEO_VAE_DECODER_LAYERS == 36);
  CHECK(H3_VIDEO_VAE_LATENT_CHANNELS == 24);
  CHECK(h3_video_encoder_latent_t(17) == 5);
  if (test_kernels())
    return 1;
  printf("test_h3_video_vae_host OK\n");
  return 0;
}
