/* Rematch antirez/h3.c host layout + schedule (tests/test_h3.c subset). */
#include "h3_host.h"
#include "h3_reuse.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int close_enough(double a, double b, double eps) {
  return fabs(a - b) <= eps;
}

static int test_temporal(void) {
  CHECK(h3_align_frame_count(22) == 22);
  CHECK(h3_align_frame_count(23) == 39);
  h3_temporal_shape t = h3_temporal(22);
  CHECK(t.frame_count == 22);
  CHECK(t.video_t == h3_video_latent_t(22));
  return 0;
}

static int test_schedule(void) {
  h3_sigma_schedule schedule;
  CHECK(h3_schedule_build(20, &schedule));
  CHECK(schedule.steps == 20);
  CHECK(schedule.video[0] == 1.0f && schedule.audio[0] == 1.0f);
  CHECK(schedule.video[20] == 0.0f && schedule.audio[20] == 0.0f);
  CHECK(close_enough(schedule.video[1], 0.995633185, 1e-7));
  CHECK(close_enough(schedule.audio[1], 0.982758582, 1e-7));
  CHECK(close_enough(schedule.video[19], 0.387096792, 1e-7));
  CHECK(close_enough(schedule.audio[19], 0.136363640, 1e-7));

  CHECK(h3_serving_schedule_build(4, &schedule));
  CHECK(schedule.steps == 4);
  CHECK(close_enough(schedule.video[1], 36.0 / 37.0, 1e-7));
  CHECK(close_enough(schedule.audio[1], 9.0 / 10.0, 1e-7));
  CHECK(!h3_serving_schedule_build(1, &schedule));

  /* Comfy BasicScheduler scheduler=simple steps=8 on ModelSamplingAV shift=12.
   * DiscreteFlow table subsample equals host linspace base 1−i/8. */
  CHECK(h3_serving_schedule_build(8, &schedule));
  CHECK(schedule.steps == 8);
  CHECK(schedule.video[0] == 1.0f && schedule.video[8] == 0.0f);
  CHECK(close_enough(schedule.video[1], 0.988235294, 1e-7));
  CHECK(close_enough(schedule.video[6], 0.8, 1e-7));
  CHECK(close_enough(schedule.video[7], 0.631578947, 1e-7));
  CHECK(close_enough(schedule.audio[7], 0.3, 1e-7));
  CHECK(close_enough(h3_time_shift_sigma(schedule.video[7], 12.0, 3.0),
                     schedule.audio[7], 1e-6));
  return 0;
}

static int test_dit_reuse_schedule(void) {
  uint8_t selected[50];
  const int aggressive[] = {0, 3, 6, 9, 12, 15, 18, 19};
  CHECK(h3_dit_reuse_schedule(20, 3, selected, sizeof(selected)) == 8);
  for (int step = 0; step < 20; step++) {
    int expected = 0;
    for (size_t i = 0; i < sizeof(aggressive) / sizeof(*aggressive); i++)
      expected |= step == aggressive[i];
    CHECK(selected[step] == expected);
  }
  CHECK(h3_dit_reuse_schedule(20, 2, selected, sizeof(selected)) == 11);
  for (int step = 0; step < 20; step++)
    CHECK(selected[step] == (step % 2 == 0 || step == 19));
  CHECK(h3_dit_reuse_schedule(50, 3, selected, sizeof(selected)) == 18);
  CHECK(h3_dit_reuse_schedule(20, 1, selected, sizeof(selected)) == 20);
  CHECK(h3_dit_reuse_schedule(20, 3, selected, 19) == -1);
  float last[] = {1.f, 2.f};
  float prev[] = {0.f, 1.f};
  float out[2];
  CHECK(h3_dit_extrapolation_ratio(0.5f, 1.f, 2.f, 1) == 0.5f);
  h3_dit_extrapolate_velocity(out, last, prev, 2, 0.5f, 1.f, 2.f, 1);
  CHECK(close_enough(out[0], 1.5, 1e-6));
  CHECK(close_enough(out[1], 2.5, 1e-6));
  char pbuf[768];
  CHECK(h3_resolve_dit_pack_path(pbuf, sizeof(pbuf)));
  CHECK(strstr(pbuf, "pruned_int8_convrot.safetensors") != NULL);
  return 0;
}

static int test_rope_grids(void) {
  double axis[16];
  CHECK(h3_rope_spatial_axis(32, 2, 32.0, axis, 16) == 16);
  CHECK(close_enough(axis[0], 0.0, 1e-12));
  CHECK(close_enough(axis[1], 2.0, 1e-12));
  CHECK(close_enough(axis[15], 30.0, 1e-12));
  /* Non-square 16x64 → sqrt_area=32 */
  double h_ax[8], w_ax[32];
  CHECK(h3_rope_spatial_axis(16, 2, 32.0, h_ax, 8) == 8);
  CHECK(h3_rope_spatial_axis(64, 2, 32.0, w_ax, 32) == 32);
  CHECK(close_enough(h_ax[0], (1.0 - 0.5) / 2.0 * 32.0, 1e-9));

  double tg[7];
  CHECK(h3_rope_temporal_grid(7, 10.0, tg, 7) == 7);
  CHECK(close_enough(tg[0], 10.0, 1e-12));
  CHECK(close_enough(tg[1], 10.0 + 5.0 / 3.0, 1e-12));
  CHECK(close_enough(tg[2], 10.0 + 5.0 / 3.0 * (1 + 4), 1e-12));
  CHECK(close_enough(h3_rope_temporal_span(7), 36.66666666666667, 1e-9));
  CHECK(close_enough(h3_rope_temporal_span(16), 86.66666666666667, 1e-9));
  CHECK(close_enough(H3_ROPE_FRAME_RESCALE, 5.0 / 3.0, 1e-15));
  return 0;
}

static int check_segments(const h3_layout *layout, const size_t (*bounds)[2],
                          const h3_segment_kind *kinds, size_t count) {
  CHECK(layout->segment_count == count);
  for (size_t index = 0; index < count; index++) {
    CHECK(layout->segments[index].start == bounds[index][0]);
    CHECK(layout->segments[index].stop == bounds[index][1]);
    CHECK(layout->segments[index].kind == kinds[index]);
  }
  return 0;
}

static void position_checksums(const h3_layout *layout, double sums[3],
                               double weighted[3]) {
  memset(sums, 0, 3 * sizeof(*sums));
  memset(weighted, 0, 3 * sizeof(*weighted));
  for (size_t index = 0; index < layout->seq_len; index++) {
    const double values[3] = {layout->positions[index].t,
                              layout->positions[index].h,
                              layout->positions[index].w};
    for (int axis = 0; axis < 3; axis++) {
      sums[axis] += values[axis];
      weighted[axis] += (double)(index + 1) * values[axis];
    }
  }
}

static int test_layout_tiny(void) {
  h3_layout_spec spec = {12, 2, 2, 2, 8, 5, NULL, 0, NULL, 0};
  h3_layout layout;
  char error[256];
  CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
  CHECK(layout.seq_len == 30);
  const size_t bounds[][2] = {{0, 12}, {12, 28}, {28, 30}};
  const h3_segment_kind kinds[] = {H3_SEG_TEXT, H3_SEG_AUDIO, H3_SEG_VIDEO};
  if (check_segments(&layout, bounds, kinds, 3))
    return 1;
  double sums[3], weighted[3];
  position_checksums(&layout, sums, weighted);
  CHECK(close_enough(sums[0], 339.6666666666667, 1e-10));
  CHECK(close_enough(weighted[0], 6498.0, 1e-10));
  /* 2×2 latent, patch 2×2 → one spatial site; Comfy h/w axes are 0. */
  CHECK(close_enough(sums[1], 0.0, 1e-12));
  CHECK(close_enough(sums[2], 0.0, 1e-12));
  h3_layout_free(&layout);
  return 0;
}

static int test_layout_128_t2va(void) {
  /* Comfy PackedLayout(12, 2, 8, 8, 8) T2VA — spatial RoPE is live. */
  h3_layout_spec spec = {12, 2, 8, 8, 8, 5, NULL, 0, NULL, 0};
  h3_layout layout;
  char error[256];
  CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
  CHECK(layout.seq_len == 60);
  const size_t bounds[][2] = {{0, 12}, {12, 28}, {28, 60}};
  const h3_segment_kind kinds[] = {H3_SEG_TEXT, H3_SEG_AUDIO, H3_SEG_VIDEO};
  if (check_segments(&layout, bounds, kinds, 3))
    return 1;
  double sums[3], weighted[3];
  position_checksums(&layout, sums, weighted);
  CHECK(close_enough(sums[0], 724.6666666666665, 1e-9));
  CHECK(close_enough(sums[1], 384.0, 1e-9));
  CHECK(close_enough(sums[2], 576.0, 1e-9));
  h3_layout_free(&layout);
  return 0;
}

static int test_layout_fl2va(void) {
  int keyframes[] = {0, 55};
  h3_layout_spec spec = {128, 17, 30, 54, 93, 56, keyframes, 2, NULL, 0};
  h3_layout layout;
  char error[256];
  CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
  CHECK(layout.seq_len == 8009);
  const size_t bounds[][2] = {
      {0, 128}, {128, 533}, {533, 938}, {938, 1124}, {1124, 8009}};
  const h3_segment_kind kinds[] = {H3_SEG_TEXT, H3_SEG_COND, H3_SEG_COND,
                                   H3_SEG_AUDIO, H3_SEG_VIDEO};
  if (check_segments(&layout, bounds, kinds, 5))
    return 1;
  CHECK(layout.img_cond_rows == 810);
  CHECK(layout.img_target_rows == 6885);
  double sums[3], weighted[3];
  position_checksums(&layout, sums, weighted);
  CHECK(close_enough(sums[0], 1360927.0, 1e-6));
  CHECK(close_enough(weighted[0], 5883121758.0, 1e-4));
  CHECK(close_enough(sums[1], 117002.11801356057, 1e-7));
  CHECK(close_enough(sums[2], 119830.2393846486, 1e-7));
  h3_layout_free(&layout);
  return 0;
}

static int test_layout_ref2va(void) {
  h3_layout_ref references[] = {{H3_LAYOUT_REF_IMAGE, 0, 16, 24, 0},
                                {H3_LAYOUT_REF_VIDEO, 7, 16, 24, 48},
                                {H3_LAYOUT_REF_AUDIO, 0, 0, 0, 80}};
  h3_layout_spec spec = {192, 17, 30, 54, 93, 56, NULL, 0, references, 3};
  h3_layout layout;
  char error[256];
  CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
  CHECK(layout.seq_len == 8287);
  double sums[3], weighted[3];
  position_checksums(&layout, sums, weighted);
  CHECK(close_enough(sums[0], 2818905.0, 1e-5));
  CHECK(close_enough(weighted[0], 12788836714.0, 1e-3));
  h3_layout_free(&layout);
  return 0;
}

static int test_res_multistep(void) {
  /* First RES step is CONST Euler: denoised = x + σ v, same as host Euler. */
  float x[2] = {1.f, 0.f};
  float v[2] = {1.f, -2.f};
  float den[2], old[2], out[2];
  h3_const_denoised_from_host_velocity(den, x, v, 2, 1.f);
  CHECK(fabsf(den[0] - 2.f) < 1e-6f && fabsf(den[1] + 2.f) < 1e-6f);
  float sig[4] = {1.f, 0.75f, 0.5f, 0.f};
  CHECK(h3_res_step(out, x, den, NULL, 2, sig, 0, 3));
  CHECK(fabsf(out[0] - 1.25f) < 1e-6f);
  CHECK(fabsf(out[1] + 0.5f) < 1e-6f);
  float xe[2] = {1.f, 0.f};
  CHECK(h3_euler_velocity_step(xe, v, 2, 1.f, 0.75f));
  CHECK(fabsf(xe[0] - out[0]) < 1e-6f && fabsf(xe[1] - out[1]) < 1e-6f);

  /* Second order (Comfy sample_res_multistep, η=0). Scalar oracle. */
  old[0] = den[0];
  old[1] = den[1];
  x[0] = out[0];
  x[1] = out[1];
  v[0] = 0.5f;
  v[1] = 0.25f;
  h3_const_denoised_from_host_velocity(den, x, v, 2, 0.75f);
  CHECK(h3_res_step(out, x, den, old, 2, sig, 1, 3));
  CHECK(fabsf(out[0] - 1.2809746f) < 1e-5f);

  /* σ′=0 is Euler even with old_denoised (Comfy). */
  old[0] = den[0];
  x[0] = out[0];
  v[0] = 2.f;
  h3_const_denoised_from_host_velocity(den, x, v, 1, 0.5f);
  CHECK(h3_res_step(out, x, den, old, 1, sig, 2, 3));
  CHECK(fabsf(out[0] - 2.2809746f) < 1e-5f);
  return 0;
}

int main(void) {
  if (test_temporal())
    return 1;
  if (test_schedule())
    return 1;
  if (test_dit_reuse_schedule())
    return 1;
  if (test_rope_grids())
    return 1;
  if (test_layout_tiny())
    return 1;
  if (test_layout_128_t2va())
    return 1;
  if (test_layout_fl2va())
    return 1;
  if (test_layout_ref2va())
    return 1;
  if (test_res_multistep())
    return 1;
  printf("test_h3_host OK (antirez rematch)\n");
  return 0;
}
