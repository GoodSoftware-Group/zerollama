#include "h3_adaln_host.h"
#include "h3_dit_pack.h"

#include <stdio.h>
#include <stdlib.h>

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int test_tiny_plan(void) {
  h3_layout_spec spec = {12, 2, 2, 2, 8, 5, NULL, 0, NULL, 0};
  h3_layout layout;
  char err[256];
  CHECK(h3_layout_build(&spec, &layout, err, sizeof(err)));
  CHECK(layout.seq_len == 30);
  h3_dit_seq_plan plan;
  CHECK(h3_dit_seq_plan_from_layout(&layout, &plan) == 0);
  CHECK(plan.seq == 30 && plan.nt == 12 && plan.na == 16 && plan.nv == 2);
  CHECK(plan.text_index[0] == 0 && plan.text_index[11] == 11);
  CHECK(plan.audio_index[0] == 12 && plan.audio_index[15] == 27);
  CHECK(plan.video_index[0] == 28 && plan.video_index[1] == 29);
  for (int i = 0; i < 12; i++)
    CHECK(plan.tags[i] == H3_ADALN_TAG_TEXT);
  for (int i = 12; i < 28; i++)
    CHECK(plan.tags[i] == H3_ADALN_TAG_AUDIO);
  CHECK(plan.tags[28] == H3_ADALN_TAG_VIDEO);
  CHECK(plan.tags[29] == H3_ADALN_TAG_VIDEO);
  int mixed[12];
  for (int i = 0; i < 12; i++)
    mixed[i] = (i < 3) ? 0 : 1;
  CHECK(h3_dit_seq_plan_apply_text_tags(&plan, mixed, 12) == 0);
  CHECK(plan.tags[0] == 0 && plan.tags[2] == 0 && plan.tags[3] == 1);
  CHECK(plan.tags[28] == H3_ADALN_TAG_VIDEO);
  CHECK(h3_dit_seq_plan_apply_text_tags(&plan, NULL, 12) == 0);
  CHECK(plan.position_ids[0] == 0.f);
  h3_dit_seq_plan_free(&plan);
  h3_layout_free(&layout);
  return 0;
}

static int test_canvas_768(void) {
  h3_dit_t2va_geom geom;
  CHECK(h3_dit_t2va_geom_build(768, 768, 5, &geom) == 0);
  CHECK(geom.pixel_w == 768 && geom.pixel_h == 768);
  CHECK(geom.frames == 5 && geom.latent_t == 2);
  CHECK(geom.latent_w == 48 && geom.latent_h == 48);
  CHECK(geom.nv == 1152 && geom.na == 16);
  CHECK(geom.video_n == 24ull * 2 * 48 * 48);
  CHECK(geom.audio_n == 2ull * 32 * 8);
  h3_layout_spec spec = {12, geom.latent_t, geom.latent_h, geom.latent_w,
                         geom.audio_t, geom.frames, NULL, 0, NULL, 0};
  h3_layout layout;
  char err[256];
  CHECK(h3_layout_build(&spec, &layout, err, sizeof(err)));
  CHECK(layout.seq_len == 12 + 16 + 1152);
  h3_dit_seq_plan plan;
  CHECK(h3_dit_seq_plan_from_layout(&layout, &plan) == 0);
  CHECK(plan.nv == 1152 && plan.na == 16 && plan.nt == 12);
  h3_dit_seq_plan_free(&plan);
  h3_layout_free(&layout);
  return 0;
}

static int test_canvas_256_lab(void) {
  h3_dit_t2va_geom geom;
  CHECK(h3_dit_t2va_geom_build(256, 256, 5, &geom) == 0);
  CHECK(geom.pixel_w == 256 && geom.pixel_h == 256);
  CHECK(geom.latent_w == 16 && geom.latent_h == 16);
  CHECK(geom.nv == 128 && geom.na == 16);
  CHECK(h3_dit_t2va_geom_build(224, 224, 5, &geom) == 0);
  CHECK(geom.pixel_w == 224 && geom.latent_w == 14 && geom.nv == 98);
  CHECK(h3_dit_t2va_geom_build(100, 100, 5, &geom) != 0);
  return 0;
}

static int test_canvas_1344_oracle(void) {
  /* Comfy MiniMaxH3ImageToVideo 1344×768 length=5. */
  h3_dit_t2va_geom geom;
  CHECK(h3_dit_t2va_geom_build(1344, 768, 5, &geom) == 0);
  CHECK(geom.pixel_w == 1344 && geom.pixel_h == 768);
  CHECK(geom.frames == 5 && geom.latent_t == 2);
  CHECK(geom.latent_w == 84 && geom.latent_h == 48);
  CHECK(geom.nv == 2016 && geom.na == 16);
  h3_layout_spec spec = {39, geom.latent_t, geom.latent_h, geom.latent_w,
                         geom.audio_t, geom.frames, NULL, 0, NULL, 0};
  h3_layout layout;
  char err[256];
  CHECK(h3_layout_build(&spec, &layout, err, sizeof(err)));
  CHECK(layout.seq_len == 39 + 16 + 2016);
  h3_dit_seq_plan plan;
  CHECK(h3_dit_seq_plan_from_layout(&layout, &plan) == 0);
  CHECK(plan.nt == 39 && plan.nv == 2016 && plan.seq == 2071);
  h3_dit_seq_plan_free(&plan);
  h3_layout_free(&layout);
  return 0;
}

int main(void) {
  if (test_tiny_plan())
    return 1;
  if (test_canvas_768())
    return 1;
  if (test_canvas_256_lab())
    return 1;
  if (test_canvas_1344_oracle())
    return 1;
  printf("test_h3_dit_pack OK\n");
  return 0;
}
