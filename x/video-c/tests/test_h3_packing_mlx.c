/* Rematch minimax-h3-mlx packing geometry against vendored h3_host. */
#include "h3_host.h"

#include <stdio.h>

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int test_frames(void) {
  /* fixtures/h3_mlx_packing.json */
  struct {
    int req, aligned, video_t, audio_t;
  } cases[] = {
      {1, 5, 2, 8},
      {5, 5, 2, 8},
      {6, 22, 7, 37},
      {22, 22, 7, 37},
      {23, 39, 12, 65},
      {100, 107, 32, 178},
      {360, 362, 107, 603},
      {361, 362, 107, 603},
  };
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
    int a = h3_align_frame_count(cases[i].req);
    CHECK(a == cases[i].aligned);
    CHECK(h3_video_latent_t(a) == cases[i].video_t);
    h3_temporal_shape t = h3_temporal(cases[i].req);
    CHECK(t.frame_count == cases[i].aligned);
    CHECK(t.video_t == cases[i].video_t);
    CHECK(t.audio_t == cases[i].audio_t);
  }
  return 0;
}

static int test_canvas(void) {
  /* MLX resolve_canvas_size → (height, width); h3_adapt_canvas → (w,h). */
  struct {
    int aw, ah, width, height;
  } cases[] = {
      {16, 9, 1344, 768},
      {9, 16, 768, 1344},
      {1, 1, 768, 768},
      {4, 1, 2016, 512},
      {1, 4, 512, 2016},
      {21, 9, 1536, 672},
      {3, 2, 1152, 768},
  };
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
    int w = 0, h = 0;
    CHECK(h3_adapt_canvas(cases[i].aw * 100, cases[i].ah * 100, &w, &h));
    CHECK(w == cases[i].width);
    CHECK(h == cases[i].height);
  }
  return 0;
}

int main(void) {
  if (test_frames())
    return 1;
  if (test_canvas())
    return 1;
  printf("test_h3_packing_mlx OK\n");
  return 0;
}
